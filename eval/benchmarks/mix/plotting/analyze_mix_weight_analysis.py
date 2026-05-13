from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "weight_analysis" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_weight_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze compact FlexOlmo weight-analysis outputs for the 55B FlexOlmo pair."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--model-names",
        default="FlexOlmo-8x7B-1T-a4-55B-v2,FlexOlmo-8x7B-1T-a4-55B-v2-rt",
        help="Comma-separated model names to compare.",
    )
    parser.add_argument("--selected-layers", help="Optional comma-separated layer indices to visualize.")
    return parser.parse_args()


def parse_model_names(raw_value: str) -> list[str]:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(names) < 2:
        raise ValueError("Provide at least two model names.")
    return names


def model_display_name(model_name: str) -> str:
    return model_name.replace("FlexOlmo-8x7B-1T-", "")


def load_summary_frame(results_root: Path, model_name: str) -> pd.DataFrame:
    path = results_root / model_name / "weight_analysis_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing weight-analysis summary for `{model_name}` at {path}.")
    frame = pd.read_csv(path)
    frame["model_name"] = model_name
    return frame


def load_run_manifest(results_root: Path, model_name: str) -> dict:
    path = results_root / model_name / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run manifest for `{model_name}` at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_matrices(results_root: Path, model_name: str):
    path = results_root / model_name / "weight_analysis_matrices.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing weight-analysis matrices for `{model_name}` at {path}.")
    return np.load(path)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_comparison_rows(frame: pd.DataFrame, model_names: list[str]) -> list[dict]:
    left_model, right_model = model_names[:2]
    rows: list[dict] = []
    for layer in sorted(frame["layer"].drop_duplicates()):
        left = frame[(frame["model_name"] == left_model) & (frame["layer"] == layer)]
        right = frame[(frame["model_name"] == right_model) & (frame["layer"] == layer)]
        if left.empty or right.empty:
            continue
        left_row = left.iloc[0]
        right_row = right.iloc[0]
        rows.append(
            {
                "layer": int(layer),
                "router_mean_offdiag_similarity_left": float(left_row["router_mean_offdiag_similarity"]),
                "router_mean_offdiag_similarity_right": float(right_row["router_mean_offdiag_similarity"]),
                "router_offdiag_delta": float(
                    right_row["router_mean_offdiag_similarity"] - left_row["router_mean_offdiag_similarity"]
                ),
                "gate_up_mean_offdiag_similarity_left": float(left_row["gate_up_mean_offdiag_similarity"]),
                "gate_up_mean_offdiag_similarity_right": float(right_row["gate_up_mean_offdiag_similarity"]),
                "gate_up_offdiag_delta": float(
                    right_row["gate_up_mean_offdiag_similarity"] - left_row["gate_up_mean_offdiag_similarity"]
                ),
                "down_proj_mean_offdiag_similarity_left": float(left_row["down_proj_mean_offdiag_similarity"]),
                "down_proj_mean_offdiag_similarity_right": float(right_row["down_proj_mean_offdiag_similarity"]),
                "down_proj_offdiag_delta": float(
                    right_row["down_proj_mean_offdiag_similarity"] - left_row["down_proj_mean_offdiag_similarity"]
                ),
                "gate_up_public_distance_left": float(left_row["gate_up_mean_public_distance"]),
                "gate_up_public_distance_right": float(right_row["gate_up_mean_public_distance"]),
            }
        )
    return rows


def plot_similarity_profiles(frame: pd.DataFrame, model_names: list[str], output_path: Path) -> None:
    metrics = [
        ("router_mean_offdiag_similarity", "Router Row Offdiag Similarity"),
        ("gate_up_mean_offdiag_similarity", "Expert Gate-Up Offdiag Similarity"),
        ("down_proj_mean_offdiag_similarity", "Expert Down-Proj Offdiag Similarity"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.8), squeeze=False)
    axes = axes[0]
    colors = {model_names[0]: "#5B6C8F", model_names[1]: "#C96B3B"}

    for ax, (metric_name, title) in zip(axes, metrics):
        for model_name in model_names[:2]:
            subset = frame[frame["model_name"] == model_name].sort_values("layer")
            if subset.empty:
                continue
            ax.plot(
                subset["layer"],
                subset[metric_name],
                marker="o",
                linewidth=2,
                color=colors[model_name],
                label=model_display_name(model_name),
            )
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=4)
        ax.set_xlabel("Layer", fontsize=11.5, fontweight="semibold")
        ax.tick_params(labelsize=10.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean Offdiag Cosine", fontsize=11.5, fontweight="semibold")
    legend = axes[-1].legend(frameon=False, fontsize=10, loc="best")
    for text in legend.get_texts():
        text.set_fontweight("semibold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.86, wspace=0.22)
    fig.suptitle("Weight Similarity Profiles", fontsize=14.5, fontweight="bold", y=0.96)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_public_distance_profiles(frame: pd.DataFrame, model_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = {model_names[0]: "#5B6C8F", model_names[1]: "#C96B3B"}
    for model_name in model_names[:2]:
        subset = frame[frame["model_name"] == model_name].sort_values("layer")
        if subset.empty:
            continue
        ax.plot(
            subset["layer"],
            subset["gate_up_mean_public_distance"],
            marker="o",
            linewidth=2,
            color=colors[model_name],
            label=model_display_name(model_name),
        )
    ax.set_title("Mean Expert Distance to Public", fontsize=12.5, fontweight="semibold", pad=4)
    ax.set_xlabel("Layer", fontsize=11.5, fontweight="semibold")
    ax.set_ylabel("Mean L2 Distance", fontsize=11.5, fontweight="semibold")
    ax.tick_params(labelsize=10.5)
    ax.grid(alpha=0.25)
    legend = ax.legend(frameon=False, fontsize=10, loc="best")
    for text in legend.get_texts():
        text.set_fontweight("semibold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.86)
    fig.suptitle("Public Distance Profile", fontsize=14.5, fontweight="bold", y=0.96)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_heatmaps(
    matrices_by_model: dict[str, object],
    model_names: list[str],
    layer_ids: list[int],
    matrix_suffix: str,
    title: str,
    cmap: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(layer_ids), len(model_names[:2]), figsize=(5.2 * len(model_names[:2]), 4.0 * len(layer_ids)), squeeze=False)
    image = None
    for row_idx, layer_idx in enumerate(layer_ids):
        for col_idx, model_name in enumerate(model_names[:2]):
            ax = axes[row_idx][col_idx]
            key = f"layer_{layer_idx}_{matrix_suffix}"
            if key not in matrices_by_model[model_name]:
                ax.axis("off")
                continue
            matrix = np.asarray(matrices_by_model[model_name][key], dtype=np.float32)
            vmin, vmax = (-1.0, 1.0) if "similarity" in matrix_suffix else (float(np.min(matrix)), float(np.max(matrix)))
            image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{model_display_name(model_name)} | layer {layer_idx}", fontsize=11.5, fontweight="semibold", pad=4)
            ax.set_xlabel("Expert", fontsize=10.5, fontweight="semibold")
            ax.set_ylabel("Expert", fontsize=10.5, fontweight="semibold")
            ax.tick_params(labelsize=9.5)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.88)
        colorbar.set_label("Cosine similarity" if "similarity" in matrix_suffix else "Distance", fontsize=11.0, fontweight="semibold")
        colorbar.ax.tick_params(labelsize=10.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.90, wspace=0.18, hspace=0.28)
    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=0.96)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_readme(path: Path, results_root: Path, output_root: Path, model_names: list[str]) -> None:
    lines = [
        "# 55B Weight Analysis",
        "",
        "## What this analysis measures",
        "This track analyzes parameter-side geometry rather than activations or routing behavior.",
        "It compares:",
        "- router row similarity (`layer.mlp.gate.weight` rows)",
        "- expert MLP weight similarity (`gate_up_proj` and `down_proj`)",
        "- how far experts lie from the public expert in weight space",
        "",
        "## Key artifacts",
        "- `weight_analysis_summary.csv`: per-layer summaries for each model",
        "- `weight_analysis_comparison.csv`: compact left/right comparison table",
        "- `router_row_similarity_heatmaps.png`: per-layer router-row cosine similarity",
        "- `expert_gate_up_similarity_heatmaps.png`: per-layer expert `gate_up_proj` cosine similarity",
        "- `expert_down_proj_similarity_heatmaps.png`: per-layer expert `down_proj` cosine similarity",
        "- `weight_similarity_profiles.png`: layer-wise mean off-diagonal similarity profiles",
        "- `public_distance_profiles.png`: layer-wise mean distance to the public expert",
        "",
        "## How to run",
        "Suite:",
        "```bash",
        "python3 eval/benchmarks/mix/runners/run_mix_weight_analysis_suite.py \\",
        "  --config eval/benchmarks/mix/configs/mix_suite_config.55b_pair.weight_analysis.json",
        "```",
        "",
        "Plotting:",
        "```bash",
        f"python3 eval/benchmarks/mix/plotting/analyze_mix_weight_analysis.py \\",
        f"  --results-root {results_root} \\",
        f"  --output-root {output_root} \\",
        f"  --model-names {','.join(model_names[:2])}",
        "```",
        "",
        "## How to interpret",
        "- High off-diagonal router-row cosine means the router's expert identifiers are less distinct.",
        "- High expert MLP off-diagonal cosine means the experts themselves may be internally redundant.",
        "- Small public distance means an expert is close to the public expert in parameter space.",
        "- Differences between router-row similarity and expert-weight similarity can help separate routing redundancy from expert redundancy.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    model_names = parse_model_names(args.model_names)

    frames = [load_summary_frame(results_root, model_name) for model_name in model_names]
    combined_frame = pd.concat(frames, ignore_index=True)
    matrices_by_model = {model_name: load_matrices(results_root, model_name) for model_name in model_names}
    manifests_by_model = {model_name: load_run_manifest(results_root, model_name) for model_name in model_names}

    if args.selected_layers:
        layer_ids = [int(part.strip()) for part in args.selected_layers.split(",") if part.strip()]
    else:
        selected_sets = [set(manifests_by_model[model_name].get("selected_layers", [])) for model_name in model_names]
        common = set.intersection(*selected_sets) if selected_sets else set()
        layer_ids = sorted(common) if common else sorted(combined_frame["layer"].drop_duplicates())

    output_root.mkdir(parents=True, exist_ok=True)
    combined_frame.to_csv(output_root / "weight_analysis_summary.csv", index=False)
    comparison_rows = build_comparison_rows(combined_frame, model_names)
    write_csv(comparison_rows, output_root / "weight_analysis_comparison.csv")

    plot_similarity_profiles(combined_frame, model_names, output_root / "weight_similarity_profiles.png")
    plot_public_distance_profiles(combined_frame, model_names, output_root / "public_distance_profiles.png")
    plot_heatmaps(
        matrices_by_model,
        model_names,
        layer_ids,
        matrix_suffix="router_similarity",
        title="Router Row Similarity",
        cmap="coolwarm",
        output_path=output_root / "router_row_similarity_heatmaps.png",
    )
    plot_heatmaps(
        matrices_by_model,
        model_names,
        layer_ids,
        matrix_suffix="gate_up_similarity",
        title="Expert Gate-Up Similarity",
        cmap="coolwarm",
        output_path=output_root / "expert_gate_up_similarity_heatmaps.png",
    )
    plot_heatmaps(
        matrices_by_model,
        model_names,
        layer_ids,
        matrix_suffix="down_proj_similarity",
        title="Expert Down-Proj Similarity",
        cmap="coolwarm",
        output_path=output_root / "expert_down_proj_similarity_heatmaps.png",
    )
    write_readme(output_root / "README.md", results_root=results_root, output_root=output_root, model_names=model_names)
    print(f"Wrote weight-analysis comparison to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
