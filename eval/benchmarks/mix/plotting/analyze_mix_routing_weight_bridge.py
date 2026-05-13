from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROUTING_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "routing_light" / "a4"
DEFAULT_WEIGHT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "weight_analysis" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_routing_weight_bridge"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relate routing coactivation patterns to router-row and expert-weight similarity. "
            "This helps test whether the router tends to co-select or confuse experts that are "
            "geometrically similar."
        )
    )
    parser.add_argument("--routing-root", type=Path, default=DEFAULT_ROUTING_ROOT)
    parser.add_argument("--weight-root", type=Path, default=DEFAULT_WEIGHT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-names",
        default="FlexOlmo-8x7B-1T-a4-55B-v2,FlexOlmo-8x7B-1T-a4-55B-v2-rt",
        help="Comma-separated model names to include.",
    )
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include.")
    parser.add_argument("--run-label", default="native_full")
    return parser.parse_args()


def parse_model_names(raw_value: str) -> list[str]:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not names:
        raise ValueError("Provide at least one model name.")
    return names


def parse_datasets(raw_value: str | None) -> set[str] | None:
    if not raw_value:
        return None
    return {part.strip() for part in raw_value.split(",") if part.strip()}


def model_display_name(model_name: str) -> str:
    return model_name.replace("FlexOlmo-8x7B-1T-", "")


def dataset_display_name(dataset_name: str) -> str:
    return {
        "mkqa_en_da": "MGQA (EN/DA)",
        "gsm8k_subset": "GSM8K",
        "mbpp_subset": "MBPP",
        "pubmedqa_subset": "PubMedQA",
        "ag_news_subset": "AG News",
        "common_gen_subset": "CommonGen",
    }.get(dataset_name, dataset_name)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_routing_aggregate(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    for record in records:
        if record.get("record_type") == "routing_aggregate":
            return record
    raise ValueError(f"Could not find `routing_aggregate` in {path}.")


def load_weight_matrices(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing weight-analysis matrices: {path}")
    return np.load(path)


def discover_datasets(routing_root: Path, model_names: list[str], run_label: str) -> list[str]:
    discovered: set[str] = set()
    for model_name in model_names:
        model_root = routing_root / model_name
        if not model_root.exists():
            continue
        for dataset_dir in model_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            analysis_path = dataset_dir / run_label / "routing_analysis.jsonl"
            if analysis_path.exists():
                discovered.add(dataset_dir.name)
    return sorted(discovered)


def matrix_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {tuple(matrix.shape)}.")
    row_idx, col_idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row_idx, col_idx]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if x.shape != y.shape:
        raise ValueError(f"Correlation inputs must match in shape, got {x.shape} and {y.shape}.")
    if x.size < 3:
        return None
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    x_valid = x[valid].astype(float, copy=False)
    y_valid = y[valid].astype(float, copy=False)
    if np.allclose(x_valid, x_valid[0]) or np.allclose(y_valid, y_valid[0]):
        return None
    if method == "pearson":
        return float(np.corrcoef(x_valid, y_valid)[0, 1])
    if method == "spearman":
        return float(np.corrcoef(rankdata(x_valid), rankdata(y_valid))[0, 1])
    raise ValueError(f"Unsupported method `{method}`.")


def top_pair(matrix: np.ndarray) -> tuple[int, int, float]:
    row_idx, col_idx = np.triu_indices(matrix.shape[0], k=1)
    if len(row_idx) == 0:
        return (0, 0, float("nan"))
    scores = matrix[row_idx, col_idx]
    best_idx = int(np.nanargmax(scores))
    return int(row_idx[best_idx]), int(col_idx[best_idx]), float(scores[best_idx])


def build_pair_rows(
    model_name: str,
    dataset_name: str,
    layer_idx: int,
    coactivation: np.ndarray,
    router_similarity: np.ndarray | None,
    gate_up_similarity: np.ndarray | None,
    down_proj_similarity: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_idx, col_idx = np.triu_indices(coactivation.shape[0], k=1)
    for left, right in zip(row_idx.tolist(), col_idx.tolist()):
        rows.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "layer": int(layer_idx),
                "expert_left": int(left),
                "expert_right": int(right),
                "coactivation": float(coactivation[left, right]),
                "router_similarity": (
                    float(router_similarity[left, right]) if router_similarity is not None else np.nan
                ),
                "gate_up_similarity": (
                    float(gate_up_similarity[left, right]) if gate_up_similarity is not None else np.nan
                ),
                "down_proj_similarity": (
                    float(down_proj_similarity[left, right]) if down_proj_similarity is not None else np.nan
                ),
            }
        )
    return rows


def build_summary_rows(
    routing_root: Path,
    weight_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    run_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        matrices = load_weight_matrices(weight_root / model_name / "weight_analysis_matrices.npz")
        for dataset_name in dataset_names:
            analysis_path = routing_root / model_name / dataset_name / run_label / "routing_analysis.jsonl"
            if not analysis_path.exists():
                continue
            aggregate = load_routing_aggregate(analysis_path)
            layer_matrices = aggregate.get("layer_coactivation_matrices") or []

            for layer_idx, layer_coactivation in enumerate(layer_matrices):
                if layer_coactivation is None:
                    continue
                coactivation = np.asarray(layer_coactivation, dtype=np.float32)
                if coactivation.ndim != 2 or coactivation.shape[0] != coactivation.shape[1]:
                    continue

                router_key = f"layer_{layer_idx}_router_similarity"
                gate_key = f"layer_{layer_idx}_gate_up_similarity"
                down_key = f"layer_{layer_idx}_down_proj_similarity"

                router_similarity = (
                    np.asarray(matrices[router_key], dtype=np.float32) if router_key in matrices.files else None
                )
                gate_up_similarity = (
                    np.asarray(matrices[gate_key], dtype=np.float32) if gate_key in matrices.files else None
                )
                down_proj_similarity = (
                    np.asarray(matrices[down_key], dtype=np.float32) if down_key in matrices.files else None
                )

                pair_rows.extend(
                    build_pair_rows(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        layer_idx=layer_idx,
                        coactivation=coactivation,
                        router_similarity=router_similarity,
                        gate_up_similarity=gate_up_similarity,
                        down_proj_similarity=down_proj_similarity,
                    )
                )

                coactivation_vec = matrix_upper_triangle(coactivation)
                router_vec = matrix_upper_triangle(router_similarity) if router_similarity is not None else None
                gate_up_vec = matrix_upper_triangle(gate_up_similarity) if gate_up_similarity is not None else None
                down_proj_vec = matrix_upper_triangle(down_proj_similarity) if down_proj_similarity is not None else None

                top_left, top_right, top_value = top_pair(coactivation)
                summary_rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "layer": int(layer_idx),
                        "num_experts": int(coactivation.shape[0]),
                        "mean_offdiag_coactivation": float(coactivation_vec.mean()) if coactivation_vec.size else np.nan,
                        "router_corr_pearson": (
                            safe_corr(coactivation_vec, router_vec, "pearson") if router_vec is not None else None
                        ),
                        "router_corr_spearman": (
                            safe_corr(coactivation_vec, router_vec, "spearman") if router_vec is not None else None
                        ),
                        "gate_up_corr_pearson": (
                            safe_corr(coactivation_vec, gate_up_vec, "pearson") if gate_up_vec is not None else None
                        ),
                        "gate_up_corr_spearman": (
                            safe_corr(coactivation_vec, gate_up_vec, "spearman") if gate_up_vec is not None else None
                        ),
                        "down_proj_corr_pearson": (
                            safe_corr(coactivation_vec, down_proj_vec, "pearson") if down_proj_vec is not None else None
                        ),
                        "down_proj_corr_spearman": (
                            safe_corr(coactivation_vec, down_proj_vec, "spearman") if down_proj_vec is not None else None
                        ),
                        "top_coactivation_pair": f"{top_left},{top_right}",
                        "top_coactivation_value": top_value,
                        "top_pair_router_similarity": (
                            float(router_similarity[top_left, top_right])
                            if router_similarity is not None
                            else np.nan
                        ),
                        "top_pair_gate_up_similarity": (
                            float(gate_up_similarity[top_left, top_right])
                            if gate_up_similarity is not None
                            else np.nan
                        ),
                        "top_pair_down_proj_similarity": (
                            float(down_proj_similarity[top_left, top_right])
                            if down_proj_similarity is not None
                            else np.nan
                        ),
                    }
                )

    return summary_rows, pair_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_layerwise_correlations(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("router_corr_spearman", "Coactivation vs Router Similarity"),
        ("gate_up_corr_spearman", "Coactivation vs Gate-Up Similarity"),
        ("down_proj_corr_spearman", "Coactivation vs Down-Proj Similarity"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.8), squeeze=False)
    axes = axes[0]

    for ax, (metric_name, title) in zip(axes, metrics):
        for model_name, frame in summary.groupby("model_name", sort=False):
            grouped = (
                frame.groupby("layer", dropna=False)[metric_name]
                .mean()
                .reset_index()
                .sort_values("layer")
            )
            if grouped.empty:
                continue
            ax.plot(
                grouped["layer"],
                grouped[metric_name],
                marker="o",
                linewidth=2,
                label=model_display_name(str(model_name)),
            )
        ax.axhline(0.0, color="#555555", linewidth=1, alpha=0.6)
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=4)
        ax.set_xlabel("Layer", fontsize=11, fontweight="semibold")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean Spearman correlation", fontsize=11, fontweight="semibold")
    axes[-1].legend(frameon=False, fontsize=10, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.88, wspace=0.25)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_dataset_heatmap(summary: pd.DataFrame, metric_name: str, title: str, output_path: Path) -> None:
    if summary.empty:
        return
    heatmap_frame = (
        summary.groupby(["dataset_name", "model_name"], dropna=False)[metric_name]
        .mean()
        .reset_index()
    )
    if heatmap_frame.empty:
        return
    pivot = heatmap_frame.pivot(index="dataset_name", columns="model_name", values=metric_name)
    row_labels = [dataset_display_name(name) for name in pivot.index]
    col_labels = [model_display_name(name) for name in pivot.columns]
    matrix = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(2.8 + 1.8 * len(col_labels), 1.8 + 0.75 * len(row_labels)))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=5)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "--" if np.isnan(value) else f"{value:.2f}"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if np.isfinite(value) and abs(value) > 0.55 else "black",
                fontweight="semibold",
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Mean Spearman correlation", fontsize=11, fontweight="semibold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.22, right=0.93, bottom=0.22, top=0.88)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_readme(
    path: Path,
    routing_root: Path,
    weight_root: Path,
    output_root: Path,
    run_label: str,
    model_names: list[str],
    dataset_names: list[str],
) -> None:
    lines = [
        "# Routing / Weight Bridge Analysis",
        "",
        "## What this analysis measures",
        "This bridge asks whether expert pairs that co-activate together are also similar in parameter space.",
        "It joins two existing artifact families:",
        "- routing aggregate coactivation matrices from `routing_analysis.jsonl`",
        "- weight-side similarity matrices from `weight_analysis_matrices.npz`",
        "",
        "The main question is:",
        "- when the router repeatedly co-selects an expert pair, is that pair geometrically similar?",
        "",
        "## Why this is useful",
        "This is the most direct compact summary for the hypothesis that the router struggles because expert identities are not distinct enough.",
        "If coactivation tracks router-row similarity, that points to router-side redundancy.",
        "If coactivation tracks expert MLP similarity, that points to expert-side redundancy.",
        "If neither tracks strongly, the issue is more likely input-side overlap or a routing policy issue rather than a simple weight-geometry story.",
        "",
        "## Key artifacts",
        "- `routing_weight_bridge_summary.csv`: per-model, per-dataset, per-layer correlations",
        "- `routing_weight_bridge_pairs.csv`: per-expert-pair table for deeper inspection",
        "- `routing_weight_bridge_layerwise.png`: layer-wise correlation profiles",
        "- `routing_weight_bridge_router_heatmap.png`: dataset/model summary for router-row similarity",
        "- `routing_weight_bridge_gate_up_heatmap.png`: dataset/model summary for expert `gate_up_proj` similarity",
        "- `routing_weight_bridge_down_proj_heatmap.png`: dataset/model summary for expert `down_proj` similarity",
        "",
        "## How to run",
        "```bash",
        "python3 eval/benchmarks/mix/plotting/analyze_mix_routing_weight_bridge.py \\",
        f"  --routing-root {routing_root} \\",
        f"  --weight-root {weight_root} \\",
        f"  --output-root {output_root} \\",
        f"  --model-names {','.join(model_names)} \\",
        f"  --run-label {run_label}",
        "```",
        "",
        "Datasets discovered:",
        *[f"- `{dataset_name}`" for dataset_name in dataset_names],
        "",
        "## How to interpret",
        "- Positive correlation means highly coactivated expert pairs also tend to be more similar.",
        "- Strong router-row correlation supports a router-identity / row-separability explanation.",
        "- Strong expert-weight correlation supports an expert-redundancy explanation.",
        "- Weak correlation means the router may be responding to overlapping inputs even if the stored expert identities are distinct.",
        "",
        "Notes:",
        f"- Run label analyzed here: `{run_label}`",
        "- This analysis uses aggregate coactivation, so it is compact enough for remote transfer and reruns.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_names = parse_model_names(args.model_names)
    selected_datasets = parse_datasets(args.datasets)
    dataset_names = discover_datasets(args.routing_root, model_names, args.run_label)
    if selected_datasets is not None:
        dataset_names = [name for name in dataset_names if name in selected_datasets]
    if not dataset_names:
        raise ValueError(
            f"No datasets with run label `{args.run_label}` were found under {args.routing_root}."
        )

    summary_rows, pair_rows = build_summary_rows(
        routing_root=args.routing_root,
        weight_root=args.weight_root,
        model_names=model_names,
        dataset_names=dataset_names,
        run_label=args.run_label,
    )
    if not summary_rows:
        raise ValueError("No bridge rows were generated. Check that routing and weight artifacts overlap.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "routing_weight_bridge_summary.csv", summary_rows)
    write_csv(args.output_root / "routing_weight_bridge_pairs.csv", pair_rows)

    summary_frame = pd.DataFrame(summary_rows)
    plot_layerwise_correlations(
        summary_frame,
        args.output_root / "routing_weight_bridge_layerwise.png",
    )
    plot_dataset_heatmap(
        summary_frame,
        metric_name="router_corr_spearman",
        title="Coactivation vs Router Similarity",
        output_path=args.output_root / "routing_weight_bridge_router_heatmap.png",
    )
    plot_dataset_heatmap(
        summary_frame,
        metric_name="gate_up_corr_spearman",
        title="Coactivation vs Gate-Up Similarity",
        output_path=args.output_root / "routing_weight_bridge_gate_up_heatmap.png",
    )
    plot_dataset_heatmap(
        summary_frame,
        metric_name="down_proj_corr_spearman",
        title="Coactivation vs Down-Proj Similarity",
        output_path=args.output_root / "routing_weight_bridge_down_proj_heatmap.png",
    )
    write_readme(
        args.output_root / "README.md",
        routing_root=args.routing_root,
        weight_root=args.weight_root,
        output_root=args.output_root,
        run_label=args.run_label,
        model_names=model_names,
        dataset_names=dataset_names,
    )
    print(f"Wrote routing/weight bridge analysis to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
