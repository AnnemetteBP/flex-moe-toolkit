from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from flex_moe_toolkit.prev_analysis.plots import plot_expert_combination_upset


def load_jsonl_records(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_routing_aggregate(records: list[dict]) -> dict:
    for record in records:
        if record.get("record_type") == "routing_aggregate":
            return record
    raise ValueError("No `routing_aggregate` record was found in the provided JSONL.")


def save_usage_bar_plot(usage, path: str | Path, title: str = "Aggregate Expert Usage"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    expert_indices = list(range(len(usage)))
    ax.bar(expert_indices, usage, color="#2f6db2")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Normalized usage")
    ax.set_title(title)
    ax.set_xticks(expert_indices)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_coactivation_heatmap(matrix, path: str | Path, title: str = "Aggregate Coactivation Matrix"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, cmap="viridis", ax=ax)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Expert")
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_layerwise_coactivation_heatmaps(
    matrices,
    output_dir: str | Path,
    stem: str = "routing_coactivation_heatmap_layer",
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for layer_idx, matrix in enumerate(matrices):
        if matrix is None:
            continue
        path = output_dir / f"{stem}_{layer_idx}.png"
        save_coactivation_heatmap(
            matrix,
            path,
            title=f"Layer {layer_idx} Coactivation Matrix Aggregated Across Examples",
        )
        paths.append(path)
    return paths


def save_layerwise_upset_plots(
    eval_records: list[dict],
    output_dir: str | Path,
    stem: str = "expert_combination_upset_layer",
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assume all records have the same number of layers
    if not eval_records:
        return []
    num_layers = len(eval_records[0].get("layer_token_topk_combination_counts", {}))
    paths = []
    for layer_idx in range(num_layers):
        layer_combinations = Counter()
        for record in eval_records:
            layer_counts = record.get("layer_token_topk_combination_counts", {}).get(str(layer_idx), {})
            layer_combinations.update(
                {
                    tuple(int(expert) for expert in combo_key.split(",") if expert != ""): count
                    for combo_key, count in layer_counts.items()
                }
            )
        if layer_combinations:
            path = output_dir / f"{stem}_{layer_idx}.png"
            plot_expert_combination_upset(
                combination_counts=layer_combinations,
                path=path,
                title=f"Token-Level Expert Combinations (Layer {layer_idx})",
            )
            paths.append(path)
    return paths


def build_token_combination_counter(eval_records: list[dict]) -> Counter:
    aggregate_counter = Counter()
    for record in eval_records:
        token_counts = record.get("token_topk_combination_counts", {})
        aggregate_counter.update(
            {
                tuple(int(expert) for expert in combo_key.split(",") if expert != ""): count
                for combo_key, count in token_counts.items()
            }
        )
    return aggregate_counter


def build_layer_combination_counter(eval_records: list[dict]) -> Counter:
    aggregate_counter = Counter()
    for record in eval_records:
        layer_combos = record.get("layer_activated_experts", [])
        for layer_combo in layer_combos:
            if layer_combo:  # skip empty
                combo = tuple(sorted(int(expert) for expert in layer_combo))
                aggregate_counter[combo] += 1
    return aggregate_counter


def plot_routing_outputs(
    routing_analysis_path: str | Path,
    output_dir: str | Path,
    eval_records_path: str | Path | None = None,
) -> dict[str, str | list[str]]:
    routing_records = load_jsonl_records(routing_analysis_path)
    routing_aggregate = extract_routing_aggregate(routing_records)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usage_plot_path = output_dir / "routing_usage_bar.png"
    coactivation_plot_path = output_dir / "routing_coactivation_heatmap.png"

    save_usage_bar_plot(routing_aggregate["usage"], usage_plot_path)
    save_coactivation_heatmap(routing_aggregate["coactivation_matrix"], coactivation_plot_path)
    layerwise_paths = save_layerwise_coactivation_heatmaps(
        routing_aggregate.get("layer_coactivation_matrices", []),
        output_dir,
    )

    result = {
        "usage_plot_path": str(usage_plot_path),
        "coactivation_plot_path": str(coactivation_plot_path),
        "layerwise_coactivation_plot_paths": [str(path) for path in layerwise_paths],
    }

    if eval_records_path is not None:
        eval_records = load_jsonl_records(eval_records_path)
        layerwise_upset_paths = save_layerwise_upset_plots(
            eval_records,
            output_dir,
        )
        result["layerwise_upset_plot_paths"] = [str(path) for path in layerwise_upset_paths]

    return result
