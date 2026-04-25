from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.plotting.routing import (
    extract_routing_aggregate,
    load_jsonl_records,
    plot_routing_outputs,
)


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "routing_light" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_coactivation"
DEFAULT_MODELS = [
    "FlexOlmo-8x7B-1T-a4-55B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2-rt",
]
DEFAULT_PUBLIC_EXPERT_IDX = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate layer-wise co-activation heatmaps and expert-combination upset plots "
            "for the mix routing outputs."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-name", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--run-label", default="native_full")
    parser.add_argument("--public-expert-idx", type=int, default=DEFAULT_PUBLIC_EXPERT_IDX)
    return parser.parse_args()


def discover_models(results_root: Path, requested: list[str]) -> list[str]:
    if requested:
        return requested
    return [model_name for model_name in DEFAULT_MODELS if (results_root / model_name).exists()]


def discover_datasets(results_root: Path, model_names: list[str], requested: list[str], run_label: str) -> list[str]:
    if requested:
        return requested

    discovered: set[str] = set()
    for model_name in model_names:
        model_root = results_root / model_name
        if not model_root.exists():
            continue
        for dataset_dir in model_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            if (dataset_dir / run_label / "routing_analysis.jsonl").exists():
                discovered.add(dataset_dir.name)
    return sorted(discovered)


def load_aggregate(results_root: Path, model_name: str, dataset_name: str, run_label: str) -> dict | None:
    analysis_path = results_root / model_name / dataset_name / run_label / "routing_analysis.jsonl"
    if not analysis_path.exists():
        return None
    return extract_routing_aggregate(load_jsonl_records(analysis_path))


def load_record_count(results_root: Path, model_name: str, dataset_name: str, run_label: str) -> int:
    records_path = results_root / model_name / dataset_name / run_label / "routing_records.jsonl"
    if not records_path.exists():
        return 0
    return len(load_jsonl_records(records_path))


def summarize_matrix(matrix: np.ndarray, public_expert_idx: int) -> dict[str, float | int | str]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square co-activation matrix, received shape {tuple(matrix.shape)}.")

    num_experts = int(matrix.shape[0])
    diag_values = np.diag(matrix).astype(float)
    offdiag_mask = ~np.eye(num_experts, dtype=bool)
    offdiag_values = matrix[offdiag_mask].astype(float)

    offdiag_mean = float(np.mean(offdiag_values)) if offdiag_values.size else 0.0
    offdiag_max = float(np.max(offdiag_values)) if offdiag_values.size else 0.0
    diag_mean = float(np.mean(diag_values)) if diag_values.size else 0.0

    public_row = np.delete(matrix[public_expert_idx].astype(float), public_expert_idx)
    public_col = np.delete(matrix[:, public_expert_idx].astype(float), public_expert_idx)
    public_mean = float(np.mean(np.concatenate([public_row, public_col]))) if public_row.size else 0.0

    upper_values: list[float] = []
    upper_pairs: list[tuple[int, int]] = []
    for row_idx in range(num_experts):
        for col_idx in range(row_idx + 1, num_experts):
            upper_pairs.append((row_idx, col_idx))
            upper_values.append(float(matrix[row_idx, col_idx]))

    if upper_values:
        best_idx = int(np.argmax(upper_values))
        dominant_pair = upper_pairs[best_idx]
        dominant_pair_value = upper_values[best_idx]
        dominant_pair_label = f"{dominant_pair[0]}-{dominant_pair[1]}"
    else:
        dominant_pair_label = ""
        dominant_pair_value = 0.0

    return {
        "num_experts": num_experts,
        "diag_mean": diag_mean,
        "offdiag_mean": offdiag_mean,
        "offdiag_max": offdiag_max,
        "public_offdiag_mean": public_mean,
        "dominant_pair": dominant_pair_label,
        "dominant_pair_value": dominant_pair_value,
    }


def build_summary_tables(
    results_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    run_label: str,
    public_expert_idx: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict] = []
    layer_rows: list[dict] = []

    for model_name in model_names:
        for dataset_name in dataset_names:
            aggregate = load_aggregate(results_root, model_name, dataset_name, run_label)
            if not aggregate:
                continue

            aggregate_matrix = np.asarray(aggregate["coactivation_matrix"], dtype=float)
            aggregate_rows.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "run_label": run_label,
                    "record_count": load_record_count(results_root, model_name, dataset_name, run_label),
                    **summarize_matrix(aggregate_matrix, public_expert_idx=public_expert_idx),
                }
            )

            for layer_idx, matrix in enumerate(aggregate.get("layer_coactivation_matrices", [])):
                if matrix is None:
                    continue
                layer_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "run_label": run_label,
                        "layer": layer_idx,
                        **summarize_matrix(np.asarray(matrix, dtype=float), public_expert_idx=public_expert_idx),
                    }
                )

    aggregate_frame = pd.DataFrame(aggregate_rows)
    layer_frame = pd.DataFrame(layer_rows)

    comparison_rows: list[dict] = []
    if not aggregate_frame.empty and len(model_names) == 2:
        left_model, right_model = model_names[0], model_names[1]
        for dataset_name in sorted(aggregate_frame["dataset_name"].drop_duplicates()):
            left = aggregate_frame[
                (aggregate_frame["dataset_name"] == dataset_name) & (aggregate_frame["model_name"] == left_model)
            ]
            right = aggregate_frame[
                (aggregate_frame["dataset_name"] == dataset_name) & (aggregate_frame["model_name"] == right_model)
            ]
            if left.empty or right.empty:
                continue
            left_row = left.iloc[0]
            right_row = right.iloc[0]
            comparison_rows.append(
                {
                    "dataset_name": dataset_name,
                    "left_model": left_model,
                    "right_model": right_model,
                    "offdiag_mean_delta": float(right_row["offdiag_mean"] - left_row["offdiag_mean"]),
                    "public_offdiag_mean_delta": float(
                        right_row["public_offdiag_mean"] - left_row["public_offdiag_mean"]
                    ),
                    "offdiag_max_delta": float(right_row["offdiag_max"] - left_row["offdiag_max"]),
                    "dominant_pair_left": str(left_row["dominant_pair"]),
                    "dominant_pair_right": str(right_row["dominant_pair"]),
                    "dominant_pair_value_delta": float(
                        right_row["dominant_pair_value"] - left_row["dominant_pair_value"]
                    ),
                }
            )

    comparison_frame = pd.DataFrame(comparison_rows)
    return aggregate_frame, layer_frame, comparison_frame


def plot_summary_metrics(
    aggregate_frame: pd.DataFrame,
    output_root: Path,
) -> Path | None:
    if aggregate_frame.empty:
        return None

    metric_specs = [
        ("offdiag_mean", "Mean Off-Diagonal"),
        ("public_offdiag_mean", "Public Off-Diagonal"),
        ("offdiag_max", "Max Off-Diagonal"),
        ("dominant_pair_value", "Dominant Pair Strength"),
    ]
    model_names = list(aggregate_frame["model_name"].drop_duplicates())
    dataset_names = list(aggregate_frame["dataset_name"].drop_duplicates())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    colors = ["#5B6C8F", "#C96B3B", "#5A9367", "#8C5A9E"]
    width = 0.38 if len(model_names) == 2 else max(0.18, 0.9 / max(len(model_names), 1))
    x = np.arange(len(dataset_names))

    for ax, (metric_key, title) in zip(axes, metric_specs):
        for idx, model_name in enumerate(model_names):
            subset = aggregate_frame[aggregate_frame["model_name"] == model_name]
            values = []
            for dataset_name in dataset_names:
                row = subset[subset["dataset_name"] == dataset_name]
                values.append(float(row.iloc[0][metric_key]) if not row.empty else np.nan)
            offset = (idx - (len(model_names) - 1) / 2.0) * width
            ax.bar(x + offset, values, width=width, label=model_name, color=colors[idx % len(colors)], alpha=0.92)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper right")
    output_path = output_root / "coactivation_summary_metrics.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_per_run_plots(
    results_root: Path,
    output_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    run_label: str,
) -> list[dict]:
    rows: list[dict] = []

    for model_name in model_names:
        for dataset_name in dataset_names:
            analysis_path = results_root / model_name / dataset_name / run_label / "routing_analysis.jsonl"
            records_path = results_root / model_name / dataset_name / run_label / "routing_records.jsonl"
            if not analysis_path.exists():
                continue

            run_output_dir = output_root / dataset_name / model_name / run_label
            result = plot_routing_outputs(
                routing_analysis_path=analysis_path,
                output_dir=run_output_dir,
                eval_records_path=records_path if records_path.exists() else None,
            )
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "run_label": run_label,
                    "analysis_path": str(analysis_path),
                    "records_path": str(records_path) if records_path.exists() else "",
                    "usage_plot_path": result.get("usage_plot_path", ""),
                    "coactivation_plot_path": result.get("coactivation_plot_path", ""),
                    "num_layerwise_coactivation_plots": len(result.get("layerwise_coactivation_plot_paths", [])),
                    "num_layerwise_upset_plots": len(result.get("layerwise_upset_plot_paths", [])),
                }
            )
    return rows


def plot_aggregate_coactivation_grid(
    results_root: Path,
    output_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    run_label: str,
) -> Path | None:
    available = [
        (dataset_name, model_name, load_aggregate(results_root, model_name, dataset_name, run_label))
        for dataset_name in dataset_names
        for model_name in model_names
    ]
    available = [(dataset_name, model_name, aggregate) for dataset_name, model_name, aggregate in available if aggregate]
    if not available:
        return None

    fig, axes = plt.subplots(
        len(dataset_names),
        len(model_names),
        figsize=(5 * len(model_names), 4 * len(dataset_names)),
        constrained_layout=True,
        squeeze=False,
    )

    vmax = 0.0
    for _dataset_name, _model_name, aggregate in available:
        matrix = np.asarray(aggregate["coactivation_matrix"], dtype=float)
        if matrix.size:
            vmax = max(vmax, float(np.nanmax(matrix)))
    vmax = vmax if vmax > 0 else 1.0

    for row_idx, dataset_name in enumerate(dataset_names):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx][col_idx]
            aggregate = load_aggregate(results_root, model_name, dataset_name, run_label)
            if not aggregate:
                ax.axis("off")
                ax.set_title(f"{model_name}\n{dataset_name}\nmissing")
                continue
            matrix = np.asarray(aggregate["coactivation_matrix"], dtype=float)
            sns.heatmap(
                matrix,
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                ax=ax,
                cbar=(row_idx == 0 and col_idx == len(model_names) - 1),
            )
            ax.set_title(f"{dataset_name}\n{model_name}")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Expert")

    output_path = output_root / "aggregate_coactivation_heatmaps.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_layerwise_coactivation_overview(
    results_root: Path,
    output_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    run_label: str,
) -> list[Path]:
    output_paths: list[Path] = []

    for dataset_name in dataset_names:
        aggregates = {
            model_name: load_aggregate(results_root, model_name, dataset_name, run_label)
            for model_name in model_names
        }
        aggregates = {model_name: aggregate for model_name, aggregate in aggregates.items() if aggregate}
        if not aggregates:
            continue

        num_layers = max(len(aggregate.get("layer_coactivation_matrices", [])) for aggregate in aggregates.values())
        if num_layers == 0:
            continue

        fig, axes = plt.subplots(
            num_layers,
            len(model_names),
            figsize=(5 * len(model_names), 3.6 * num_layers),
            constrained_layout=True,
            squeeze=False,
        )

        vmax = 0.0
        for aggregate in aggregates.values():
            for matrix in aggregate.get("layer_coactivation_matrices", []):
                if matrix is None:
                    continue
                vmax = max(vmax, float(np.nanmax(np.asarray(matrix, dtype=float))))
        vmax = vmax if vmax > 0 else 1.0

        for layer_idx in range(num_layers):
            for col_idx, model_name in enumerate(model_names):
                ax = axes[layer_idx][col_idx]
                aggregate = aggregates.get(model_name)
                if aggregate is None or layer_idx >= len(aggregate.get("layer_coactivation_matrices", [])):
                    ax.axis("off")
                    continue
                matrix = aggregate["layer_coactivation_matrices"][layer_idx]
                if matrix is None:
                    ax.axis("off")
                    continue
                sns.heatmap(
                    np.asarray(matrix, dtype=float),
                    cmap="viridis",
                    vmin=0.0,
                    vmax=vmax,
                    ax=ax,
                    cbar=(layer_idx == 0 and col_idx == len(model_names) - 1),
                )
                ax.set_title(f"Layer {layer_idx} | {model_name}")
                ax.set_xlabel("Expert")
                ax.set_ylabel("Expert")

        output_path = output_root / f"{dataset_name}_layerwise_coactivation_grid.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def write_readme(
    output_root: Path,
    results_frame: pd.DataFrame,
    grid_path: Path | None,
    layerwise_grid_paths: list[Path],
    aggregate_summary_path: Path | None,
    layer_summary_path: Path | None,
    comparison_summary_path: Path | None,
    summary_plot_path: Path | None,
) -> None:
    lines = [
        "# Mix Co-Activation Outputs",
        "",
        "This directory contains co-activation heatmaps and expert-combination upset plots",
        "generated from saved mix routing outputs.",
        "",
        "Main files:",
    ]
    if grid_path is not None:
        lines.append(f"- `aggregate_coactivation_heatmaps.png`: one aggregate co-activation heatmap per dataset/model.")
    if layerwise_grid_paths:
        lines.append("- `*_layerwise_coactivation_grid.png`: layer-wise co-activation heatmap grids per dataset.")
    if summary_plot_path is not None:
        lines.append("- `coactivation_summary_metrics.png`: compact bar plots for co-activation summary metrics.")
    lines.extend(
        [
            "- `index.csv`: index of generated per-run plots and counts.",
            "- `coactivation_aggregate_summary.csv`: one row per dataset/model with compact co-activation metrics.",
            "- `coactivation_layer_summary.csv`: one row per dataset/model/layer with the same metrics.",
            "- `coactivation_model_comparison.csv`: direct model deltas per dataset.",
            "",
            "Per-run directories also contain:",
            "- `routing_usage_bar.png`",
            "- `routing_coactivation_heatmap.png`",
            "- `routing_coactivation_heatmap_layer_<n>.png`",
            "- `expert_combination_upset_layer_<n>.png`",
            "",
            "Generated runs:",
        ]
    )
    for row in results_frame.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_name}` / `{row.model_name}` / `{row.run_label}`: "
            f"{row.num_layerwise_coactivation_plots} layer heatmaps, "
            f"{row.num_layerwise_upset_plots} upset plots."
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = args.results_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = discover_models(results_root, args.model_name)
    if not model_names:
        raise ValueError(f"No models were found under {results_root}.")

    dataset_names = discover_datasets(results_root, model_names, args.dataset, args.run_label)
    if not dataset_names:
        raise ValueError(f"No datasets with run label `{args.run_label}` were found under {results_root}.")

    index_rows = generate_per_run_plots(
        results_root=results_root,
        output_root=output_root,
        model_names=model_names,
        dataset_names=dataset_names,
        run_label=args.run_label,
    )
    if not index_rows:
        raise ValueError("No co-activation plots were generated. Check that routing outputs exist.")

    index_frame = pd.DataFrame(index_rows)
    index_frame.to_csv(output_root / "index.csv", index=False)

    aggregate_summary, layer_summary, comparison_summary = build_summary_tables(
        results_root=results_root,
        model_names=model_names,
        dataset_names=dataset_names,
        run_label=args.run_label,
        public_expert_idx=args.public_expert_idx,
    )
    aggregate_summary_path = output_root / "coactivation_aggregate_summary.csv"
    layer_summary_path = output_root / "coactivation_layer_summary.csv"
    comparison_summary_path = output_root / "coactivation_model_comparison.csv"
    aggregate_summary.to_csv(aggregate_summary_path, index=False)
    layer_summary.to_csv(layer_summary_path, index=False)
    comparison_summary.to_csv(comparison_summary_path, index=False)

    aggregate_grid_path = plot_aggregate_coactivation_grid(
        results_root=results_root,
        output_root=output_root,
        model_names=model_names,
        dataset_names=dataset_names,
        run_label=args.run_label,
    )
    layerwise_grid_paths = plot_layerwise_coactivation_overview(
        results_root=results_root,
        output_root=output_root,
        model_names=model_names,
        dataset_names=dataset_names,
        run_label=args.run_label,
    )
    summary_plot_path = plot_summary_metrics(aggregate_summary, output_root=output_root)
    write_readme(
        output_root=output_root,
        results_frame=index_frame,
        grid_path=aggregate_grid_path,
        layerwise_grid_paths=layerwise_grid_paths,
        aggregate_summary_path=aggregate_summary_path,
        layer_summary_path=layer_summary_path,
        comparison_summary_path=comparison_summary_path,
        summary_plot_path=summary_plot_path,
    )

    print(f"Wrote mix co-activation analysis to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
