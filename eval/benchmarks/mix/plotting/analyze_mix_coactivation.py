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
    lines.extend(
        [
            "- `index.csv`: index of generated per-run plots and counts.",
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
    write_readme(
        output_root=output_root,
        results_frame=index_frame,
        grid_path=aggregate_grid_path,
        layerwise_grid_paths=layerwise_grid_paths,
    )

    print(f"Wrote mix co-activation analysis to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
