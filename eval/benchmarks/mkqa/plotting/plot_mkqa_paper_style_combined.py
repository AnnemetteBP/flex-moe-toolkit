from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.prev_analysis.plots import plot_expert_combination_upset


MODEL_ORDER = ["FlexOlmo-8x7B-1T-a2-55B-v2", "FlexOlmo-8x7B-1T-a4-55B-v2", "FlexOlmo-8x7B-1T-a7-55B-v2"]
SOURCE_SPECS = {
    "prompt": ("input_token_ids", "prompt_router_token_summaries_by_layer", "Input token ID"),
    "predicted": ("predicted_output_token_ids", "predicted_router_token_summaries_by_layer", "Predicted output token ID"),
    "ground_truth": ("ground_truth_output_token_ids", "ground_truth_router_token_summaries_by_layer", "Ground-truth output token ID"),
}
LANGUAGES = ["en", "da"]
SELECTED_COACTIVATION_LAYERS = [7, 15]
TARGET_LAYER_ID = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-style combined-model figures from MKQA combined-native routing outputs."
    )
    parser.add_argument(
        "--results-root",
        default="eval_results/mkqa/smoke/flexolmo/multi_family",
        help="Combined-native MKQA results root.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where paper-style plots will be written. Defaults to <results-root>/paper_style_plots.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional model names to include. Defaults to all combined-native models in the results root.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k used for paper-style vocab/domain specialization recomputation. Defaults to 1.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_models(results_root: Path, requested: list[str]) -> list[str]:
    if requested:
        return requested
    available = sorted(p.name for p in (results_root / "routing").iterdir() if p.is_dir())
    return [name for name in MODEL_ORDER if name in available] + [name for name in available if name not in MODEL_ORDER]


def load_bundle(results_root: Path, model_name: str) -> dict:
    routing_dir = results_root / "routing" / model_name / "mkqa_en_da" / "native_full"
    records = load_jsonl(routing_dir / "routing_records.jsonl")
    analysis = load_jsonl(routing_dir / "routing_analysis.jsonl")
    manifest = json.loads((results_root / "routing" / model_name / "mkqa_en_da" / "run_manifest.json").read_text())
    return {
        "model_name": model_name,
        "routing_records": records,
        "routing_analysis": analysis,
        "run_manifest": manifest,
    }


def _flatten_topk_ids(layer_summary: dict, top_k: int) -> np.ndarray:
    top1 = np.array(layer_summary["top1_expert_ids"]).reshape(-1, 1)
    if top_k <= 1:
        return top1

    top2_ids = np.array(layer_summary["top2_expert_ids"]).reshape(-1, 1)
    if top_k == 2:
        return np.concatenate([top1, top2_ids], axis=1)

    raise ValueError("Paper-style recomputation currently supports top_k up to 2 from saved routing summaries.")


def compute_vocab_specialization_from_records(records: list[dict], source: str, top_k: int) -> dict:
    token_field, summary_field, _label = SOURCE_SPECS[source]
    num_layers = len(records[0][summary_field])
    num_experts = int(records[0]["num_available_experts"])

    specialization_per_layer = []
    specialization_per_expert_per_layer = []
    specialization_matrix_per_layer = []

    for layer_idx in range(num_layers):
        token_expert_counts: dict[int, np.ndarray] = {}
        token_totals: Counter[int] = Counter()

        for record in records:
            token_ids = record.get(token_field) or []
            layer_summary = record.get(summary_field) or []
            if not token_ids or not layer_summary:
                continue
            topk_ids = _flatten_topk_ids(layer_summary[layer_idx], top_k=top_k)
            for token_id, expert_ids in zip(token_ids, topk_ids.tolist()):
                if int(token_id) not in token_expert_counts:
                    token_expert_counts[int(token_id)] = np.zeros(num_experts, dtype=float)
                token_totals[int(token_id)] += len(expert_ids)
                for expert_idx in expert_ids:
                    if int(expert_idx) < 0:
                        continue
                    token_expert_counts[int(token_id)][int(expert_idx)] += 1.0

        valid_tokens = sorted(token_totals)
        if not valid_tokens:
            specialization_matrix = np.zeros((0, num_experts), dtype=float)
            specialization_by_expert = np.zeros(num_experts, dtype=float)
        else:
            specialization_matrix = np.vstack(
                [
                    token_expert_counts[token_id] / max(1.0, float(token_totals[token_id]))
                    for token_id in valid_tokens
                ]
            )
            specialization_by_expert = specialization_matrix.mean(axis=0)

        specialization_per_layer.append(float(specialization_by_expert.mean() if specialization_by_expert.size else 0.0))
        specialization_per_expert_per_layer.append(specialization_by_expert.tolist())
        specialization_matrix_per_layer.append(specialization_matrix.tolist())

    return {
        "specialization_per_layer": specialization_per_layer,
        "specialization_per_expert_per_layer": specialization_per_expert_per_layer,
        "specialization_matrix_per_layer": specialization_matrix_per_layer,
    }


def compute_domain_specialization_from_records(records: list[dict], source: str, top_k: int) -> dict:
    _token_field, summary_field, _label = SOURCE_SPECS[source]
    num_layers = len(records[0][summary_field])
    num_experts = int(records[0]["num_available_experts"])
    language_to_idx = {language: idx for idx, language in enumerate(LANGUAGES)}

    specialization_by_layer = []
    for layer_idx in range(num_layers):
        counts_d_e = np.zeros((len(LANGUAGES), num_experts), dtype=float)
        counts_d = np.zeros(len(LANGUAGES), dtype=float)

        for record in records:
            layer_summary = record.get(summary_field) or []
            if not layer_summary:
                continue
            topk_ids = _flatten_topk_ids(layer_summary[layer_idx], top_k=top_k)
            language_idx = language_to_idx[record["language"]]
            for expert_ids in topk_ids.tolist():
                counts_d[language_idx] += 1.0
                for expert_idx in expert_ids:
                    if int(expert_idx) < 0:
                        continue
                    counts_d_e[language_idx, int(expert_idx)] += 1.0

        spec = np.zeros_like(counts_d_e)
        for domain_idx in range(len(LANGUAGES)):
            if counts_d[domain_idx] > 0:
                spec[domain_idx] = counts_d_e[domain_idx] / counts_d[domain_idx]
        specialization_by_layer.append(spec)

    return {
        "specialization_by_layer": specialization_by_layer,
        "uniform_reference": float(top_k) / float(num_experts),
    }


def aggregate_token_pair_coactivation_for_layer(records: list[dict], layer_idx: int) -> np.ndarray:
    num_experts = int(records[0]["num_available_experts"])
    matrix = np.zeros((num_experts, num_experts), dtype=float)

    for record in records:
        layer = record["prompt_topk_experts_by_layer"][layer_idx]
        token_assignments = layer[0] if layer and isinstance(layer[0], list) and layer[0] and isinstance(layer[0][0], list) else layer
        for token_experts in token_assignments:
            for left in token_experts:
                for right in token_experts:
                    matrix[int(left), int(right)] += 1.0
    return matrix


def aggregate_combination_counter_for_layer(records: list[dict], layer_idx: int) -> Counter:
    counter = Counter()
    for record in records:
        layer = record["prompt_topk_experts_by_layer"][layer_idx]
        token_assignments = layer[0] if layer and isinstance(layer[0], list) and layer[0] and isinstance(layer[0][0], list) else layer
        for token_experts in token_assignments:
            counter[tuple(sorted(int(expert_idx) for expert_idx in token_experts))] += 1
    return counter


def plot_vocab_figure(bundle: dict, output_dir: Path, top_k: int) -> Path:
    records = bundle["routing_records"]
    results = {source: compute_vocab_specialization_from_records(records, source=source, top_k=top_k) for source in SOURCE_SPECS}
    target_layer = min(TARGET_LAYER_ID, len(next(iter(results.values()))["specialization_per_expert_per_layer"]) - 1)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"prompt": "#2f6db2", "predicted": "#b24c2f", "ground_truth": "#2a9d8f"}

    for source, (_token_field, _summary_field, label) in SOURCE_SPECS.items():
        layer_values = results[source]["specialization_per_layer"]
        expert_values = results[source]["specialization_per_expert_per_layer"][target_layer]
        ax_left.plot(range(len(layer_values)), layer_values, marker="o", linewidth=1.7, color=colors[source], label=label)
        ax_right.plot(range(len(expert_values)), expert_values, marker="o", linewidth=1.5, color=colors[source], label=label)

    ax_left.set_xlabel("Layer")
    ax_left.set_ylabel("Average specialization")
    ax_left.set_title("Vocabulary Specialization by Layer")
    ax_left.legend(frameon=False, fontsize=8)
    ax_left.grid(axis="y", alpha=0.3)

    ax_right.set_xlabel("Expert")
    ax_right.set_ylabel("Specialization")
    ax_right.set_title(f"Vocabulary Specialization by Expert - Layer {target_layer}")
    ax_right.legend(frameon=False, fontsize=8)
    ax_right.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{bundle['model_name']} Vocabulary Specialization (k={top_k})", y=1.02)
    fig.tight_layout()
    path = output_dir / f"vocab_specialization_k{top_k}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_domain_figures(bundle: dict, output_dir: Path, top_k: int) -> list[Path]:
    records = bundle["routing_records"]
    paths = []
    for source, (_token_field, _summary_field, label) in SOURCE_SPECS.items():
        results = compute_domain_specialization_from_records(records, source=source, top_k=top_k)
        specialization_by_layer = results["specialization_by_layer"]
        num_layers = len(specialization_by_layer)
        target_layer = min(TARGET_LAYER_ID, num_layers - 1)
        uniform = results["uniform_reference"]

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5))
        expert_axis = np.arange(specialization_by_layer[target_layer].shape[-1])
        for language_idx, language in enumerate(LANGUAGES):
            mean_specialization = [float(layer_spec[language_idx].mean()) for layer_spec in specialization_by_layer]
            ax_left.plot(range(num_layers), mean_specialization, marker="o", linewidth=1.6, label=language)
            ax_right.plot(expert_axis, specialization_by_layer[target_layer][language_idx], marker="o", linewidth=1.6, label=language)

        ax_left.axhline(uniform, color="#b24c2f", linestyle="--", linewidth=1.4, label="uniform")
        ax_left.set_xlabel("Layer")
        ax_left.set_ylabel("Mean domain specialization")
        ax_left.set_title(f"{label} Domain Specialization by Layer")
        ax_left.legend(frameon=False, fontsize=8)
        ax_left.grid(axis="y", alpha=0.3)

        ax_right.axhline(uniform, color="#b24c2f", linestyle="--", linewidth=1.4, label="uniform")
        ax_right.set_xlabel("Expert")
        ax_right.set_ylabel("Domain specialization")
        ax_right.set_title(f"{label} Domain Specialization - Layer {target_layer}")
        ax_right.legend(frameon=False, fontsize=8)
        ax_right.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{bundle['model_name']} Domain Specialization (k={top_k})", y=1.02)
        fig.tight_layout()
        path = output_dir / f"domain_specialization_{source}_k{top_k}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_coactivation_figures(bundle: dict, output_dir: Path) -> tuple[Path, list[Path], list[Path]]:
    records = bundle["routing_records"]
    aggregate_matrix = np.zeros((int(records[0]["num_available_experts"]), int(records[0]["num_available_experts"])), dtype=float)
    for layer_idx in range(len(records[0]["prompt_topk_experts_by_layer"])):
        aggregate_matrix += aggregate_token_pair_coactivation_for_layer(records, layer_idx)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(aggregate_matrix, cmap="viridis", ax=ax)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Expert")
    ax.set_title("Token-Level Coactivation Across All Layers")
    aggregate_path = output_dir / "coactivation_heatmap_fixed.png"
    fig.savefig(aggregate_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    layer_paths = []
    upset_paths = []
    for layer_idx in SELECTED_COACTIVATION_LAYERS:
        if layer_idx >= len(records[0]["prompt_topk_experts_by_layer"]):
            continue
        layer_matrix = aggregate_token_pair_coactivation_for_layer(records, layer_idx)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(layer_matrix, cmap="viridis", ax=ax)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Expert")
        ax.set_title(f"Token-Level Coactivation Layer {layer_idx}")
        layer_path = output_dir / f"coactivation_heatmap_fixed_layer_{layer_idx}.png"
        fig.savefig(layer_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        layer_paths.append(layer_path)

        counter = aggregate_combination_counter_for_layer(records, layer_idx)
        upset_path = output_dir / f"expert_combination_upset_layer_{layer_idx}.png"
        plot_expert_combination_upset(
            combination_counts=counter,
            path=upset_path,
            title=f"Expert Combinations Layer {layer_idx}",
            max_combinations=12,
        )
        upset_paths.append(upset_path)

    return aggregate_path, layer_paths, upset_paths


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root) if args.output_root else results_root / "paper_style_plots"
    output_root.mkdir(parents=True, exist_ok=True)

    bundles = [load_bundle(results_root, model_name) for model_name in selected_models(results_root, args.model_name)]
    manifest: dict[str, object] = {"results_root": str(results_root), "output_root": str(output_root), "models": {}}

    for bundle in bundles:
        model_output_dir = output_root / bundle["model_name"]
        model_output_dir.mkdir(parents=True, exist_ok=True)

        vocab_path = plot_vocab_figure(bundle, model_output_dir, top_k=args.top_k)
        domain_paths = plot_domain_figures(bundle, model_output_dir, top_k=args.top_k)
        aggregate_path, layer_paths, upset_paths = plot_coactivation_figures(bundle, model_output_dir)

        manifest["models"][bundle["model_name"]] = {
            "vocab_figure": str(vocab_path),
            "domain_figures": [str(path) for path in domain_paths],
            "coactivation_aggregate": str(aggregate_path),
            "coactivation_layers": [str(path) for path in layer_paths],
            "coactivation_upset_plots": [str(path) for path in upset_paths],
        }

    manifest_path = output_root / "paper_style_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote paper-style plot manifest to {manifest_path}")


if __name__ == "__main__":
    main()
