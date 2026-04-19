from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.prev_analysis.plots import plot_expert_combination_upset


MODEL_ORDER = ["FlexOlmo-8x7B-1T-a2-55B-v2", "FlexOlmo-8x7B-1T-a4-55B-v2", "FlexOlmo-8x7B-1T-a7-55B-v2"]
SOURCE_ORDER = ["prompt", "predicted", "ground_truth"]
LANGUAGE_ORDER = ["en", "da"]
SELECTED_LAYERS = [0, 7, 15, 31]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate combined-model collapse-diagnostic figures from MKQA combined-native outputs."
    )
    parser.add_argument(
        "--results-root",
        default="eval_results/mkqa_results_smoke_combined",
        help="Combined-native MKQA results root.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where figures will be written. Defaults to <results-root>/collapse_diagnostics.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional model names to include. Defaults to all combined-native models in the results root.",
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


def load_model_bundle(results_root: Path, model_name: str) -> dict:
    routing_root = results_root / "routing" / model_name / "mkqa_en_da" / "native_full"
    vocab_path = results_root / "vocab_specialization" / model_name / "native_full" / "vocab_specialization_summary.jsonl"
    domain_path = results_root / "domain_specialization" / model_name / "native_full" / "domain_specialization_summary.jsonl"

    routing_records = load_jsonl(routing_root / "routing_records.jsonl")
    routing_analysis = load_jsonl(routing_root / "routing_analysis.jsonl")
    run_manifest = json.loads((results_root / "routing" / model_name / "mkqa_en_da" / "run_manifest.json").read_text())
    vocab_records = load_jsonl(vocab_path)
    domain_records = load_jsonl(domain_path)

    return {
        "model_name": model_name,
        "routing_records": routing_records,
        "routing_analysis": routing_analysis,
        "run_manifest": run_manifest,
        "vocab_records": vocab_records,
        "domain_records": domain_records,
    }


def aggregate_top1_usage_heatmap(routing_records: list[dict]) -> np.ndarray:
    num_layers = len(routing_records[0]["prompt_router_token_summaries_by_layer"])
    num_experts = routing_records[0]["run_manifest_num_experts"] if "run_manifest_num_experts" in routing_records[0] else 8
    heatmap = np.zeros((num_experts, num_layers), dtype=float)

    for record in routing_records:
        for layer_summary in record["prompt_router_token_summaries_by_layer"]:
            layer_idx = int(layer_summary["layer_idx"])
            top1_ids = np.array(layer_summary["top1_expert_ids"]).reshape(-1)
            for expert_idx in top1_ids.tolist():
                heatmap[int(expert_idx), layer_idx] += 1

    column_sums = heatmap.sum(axis=0, keepdims=True)
    column_sums[column_sums == 0] = 1.0
    return heatmap / column_sums


def collect_layer_distribution(routing_records: list[dict], field: str, layer_idx: int) -> list[float]:
    values = []
    for record in routing_records:
        layer_summary = record["prompt_router_token_summaries_by_layer"][layer_idx]
        tensor = np.array(layer_summary[field]).reshape(-1)
        values.extend(tensor.astype(float).tolist())
    return values


def plot_top1_usage_heatmap(bundle: dict, output_dir: Path) -> Path:
    routing_records = bundle["routing_records"]
    num_experts = int(bundle["run_manifest"]["model_num_experts"])
    for record in routing_records:
        record["run_manifest_num_experts"] = num_experts
    heatmap = aggregate_top1_usage_heatmap(routing_records)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap, cmap="mako", ax=ax, vmin=0.0, vmax=max(0.25, float(heatmap.max())))
    ax.set_title(f"{bundle['model_name']} Top-1 Expert Usage by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert")
    path = output_dir / "top1_usage_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ridge_like_distributions(bundle: dict, field: str, title: str, xlabel: str, stem: str, output_dir: Path) -> Path:
    routing_records = bundle["routing_records"]
    num_layers = len(routing_records[0]["prompt_router_token_summaries_by_layer"])

    fig, axes = plt.subplots(num_layers, 1, figsize=(8, max(10, num_layers * 0.35)), sharex=True)
    if num_layers == 1:
        axes = [axes]

    color = "#2f6db2" if "entropy" in field else "#b24c2f"
    for layer_idx, ax in enumerate(axes):
        values = collect_layer_distribution(routing_records, field=field, layer_idx=layer_idx)
        if values:
            ax.hist(values, bins=25, density=True, color=color, alpha=0.75)
        ax.set_ylabel(str(layer_idx), rotation=0, labelpad=12, va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=True)
    axes[0].set_title(f"{bundle['model_name']} {title}")
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    path = output_dir / stem
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def aggregate_token_combination_counter_for_layer(routing_records: list[dict], layer_idx: int) -> Counter:
    counter = Counter()
    for record in routing_records:
        layer = record["prompt_topk_experts_by_layer"][layer_idx]
        token_assignments = layer[0] if layer and isinstance(layer[0], list) and layer[0] and isinstance(layer[0][0], list) else layer
        for token_experts in token_assignments:
            combo = tuple(sorted(int(expert_idx) for expert_idx in token_experts))
            counter[combo] += 1
    return counter


def baseline_corrected_coactivation_matrix(coactivation_matrix: np.ndarray, activation_counts: np.ndarray, k: int, num_experts: int) -> np.ndarray:
    baseline = np.zeros_like(coactivation_matrix, dtype=float)
    offdiag = 0.0 if num_experts <= 1 else max(0.0, (k - 1) / (num_experts - 1))
    for row_idx in range(num_experts):
        if activation_counts[row_idx] <= 0:
            continue
        baseline[row_idx, :] = offdiag
        baseline[row_idx, row_idx] = 1.0
    return coactivation_matrix - baseline


def plot_residual_coactivation(bundle: dict, output_dir: Path) -> tuple[Path, list[Path]]:
    aggregate = bundle["routing_analysis"][-1]
    num_experts = int(bundle["run_manifest"]["model_num_experts"])
    effective_top_k = int(bundle["run_manifest"]["model_native_top_k"])

    matrix = np.array(aggregate["coactivation_matrix"], dtype=float)
    activation_counts = np.array(aggregate["activation_counts"], dtype=float)
    residual = baseline_corrected_coactivation_matrix(matrix, activation_counts, effective_top_k, num_experts)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(residual, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title(f"{bundle['model_name']} Residual Co-Activation")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Expert")
    aggregate_path = output_dir / "coactivation_residual_heatmap.png"
    fig.savefig(aggregate_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    layer_paths = []
    for layer_idx in SELECTED_LAYERS:
        if layer_idx >= len(aggregate["layer_coactivation_matrices"]):
            continue
        layer_matrix = np.array(aggregate["layer_coactivation_matrices"][layer_idx], dtype=float)
        layer_residual = baseline_corrected_coactivation_matrix(layer_matrix, activation_counts, effective_top_k, num_experts)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(layer_residual, cmap="coolwarm", center=0.0, ax=ax)
        ax.set_title(f"{bundle['model_name']} Residual Co-Activation Layer {layer_idx}")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Expert")
        path = output_dir / f"coactivation_residual_heatmap_layer_{layer_idx}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        layer_paths.append(path)

    return aggregate_path, layer_paths


def plot_upset_plots(bundle: dict, output_dir: Path) -> list[Path]:
    paths = []
    for layer_idx in SELECTED_LAYERS:
        counter = aggregate_token_combination_counter_for_layer(bundle["routing_records"], layer_idx)
        if not counter:
            continue
        path = output_dir / f"expert_combination_upset_layer_{layer_idx}.png"
        plot_expert_combination_upset(
            combination_counts=counter,
            path=path,
            title=f"{bundle['model_name']} Layer {layer_idx} Expert Combinations",
            max_combinations=12,
        )
        paths.append(path)
    return paths


def plot_vocab_paper_style(models: list[dict], output_dir: Path) -> list[Path]:
    paths = []
    for source in SOURCE_ORDER:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 4.5))
        for bundle in models:
            records = [record for record in bundle["vocab_records"] if record["source"] == source]
            records = sorted(records, key=lambda item: int(item["layer_idx"]))
            layers = [int(record["layer_idx"]) for record in records]
            means = [float(record["mean_specialization"]) for record in records]
            ax_left.plot(layers, means, marker="o", linewidth=1.8, label=bundle["model_name"])

            layer7 = next((record for record in records if int(record["layer_idx"]) == 7), records[min(7, len(records) - 1)])
            expert_values = layer7["mean_specialization_by_expert"]
            expert_indices = sorted(int(key.split("_")[-1]) for key in expert_values.keys())
            values = [float(expert_values[f"expert_{expert_idx}"]) for expert_idx in expert_indices]
            ax_right.plot(expert_indices, values, marker="o", linewidth=1.5, label=bundle["model_name"])

        ax_left.set_title(f"{source.title()} Vocabulary Specialization by Layer")
        ax_left.set_xlabel("Layer")
        ax_left.set_ylabel("Mean specialization")
        ax_left.grid(axis="y", alpha=0.3)

        ax_right.set_title(f"{source.title()} Vocabulary Specialization in Layer 7")
        ax_right.set_xlabel("Expert")
        ax_right.set_ylabel("Specialization")
        ax_right.grid(axis="y", alpha=0.3)
        ax_right.legend(fontsize=8)

        fig.tight_layout()
        path = output_dir / f"{source}_vocab_specialization_paper_style.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_domain_ridge(models: list[dict], output_dir: Path) -> list[Path]:
    paths = []
    for source in SOURCE_ORDER:
        for language in LANGUAGE_ORDER:
            fig, axes = plt.subplots(len(models), 1, figsize=(8, 2.4 * len(models)), sharex=True)
            if len(models) == 1:
                axes = [axes]
            for ax, bundle in zip(axes, models):
                records = [
                    record
                    for record in bundle["domain_records"]
                    if record["source"] == source and language in record["language_specialization"]
                ]
                values = [float(record["language_specialization"][language]) for record in records]
                ax.hist(values, bins=25, density=True, color="#2a9d8f", alpha=0.8)
                ax.set_ylabel(bundle["model_name"].replace("FlexOlmo-8x7B-1T-", ""), rotation=0, labelpad=34, va="center")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            axes[0].set_title(f"{source.title()} {language.upper()} Domain Specialization Distribution")
            axes[-1].set_xlabel("Domain specialization")
            fig.tight_layout()
            path = output_dir / f"{source}_{language}_domain_specialization_ridge.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def plot_cross_model_metric(models: list[dict], metric_key: str, ylabel: str, title: str, stem: str, output_dir: Path) -> Path:
    labels = []
    values = []
    for bundle in models:
        aggregate = bundle["routing_analysis"][-1]
        labels.append(bundle["model_name"])
        values.append(float(aggregate[metric_key]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=["#3a86ff", "#ff9f1c", "#e63946"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    path = output_dir / stem
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root) if args.output_root else results_root / "collapse_diagnostics"
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = selected_models(results_root, args.model_name)
    bundles = [load_model_bundle(results_root, model_name) for model_name in model_names]

    manifest: dict[str, object] = {
        "results_root": str(results_root),
        "output_root": str(output_root),
        "models": {},
        "cross_model": {},
    }

    for bundle in bundles:
        model_output = output_root / bundle["model_name"]
        routing_dir = model_output / "routing"
        routing_dir.mkdir(parents=True, exist_ok=True)
        vocab_dir = model_output / "vocab"
        vocab_dir.mkdir(parents=True, exist_ok=True)

        per_model_paths = {
            "top1_usage_heatmap": str(plot_top1_usage_heatmap(bundle, routing_dir)),
            "entropy_ridge": str(
                plot_ridge_like_distributions(
                    bundle,
                    field="token_entropy",
                    title="Prompt Router Entropy by Layer",
                    xlabel="Token entropy",
                    stem="prompt_entropy_ridge.png",
                    output_dir=routing_dir,
                )
            ),
            "margin_ridge": str(
                plot_ridge_like_distributions(
                    bundle,
                    field="top1_top2_margin",
                    title="Prompt Top1-Top2 Margin by Layer",
                    xlabel="Top1-Top2 probability margin",
                    stem="prompt_margin_ridge.png",
                    output_dir=routing_dir,
                )
            ),
            "selected_mass_ridge": str(
                plot_ridge_like_distributions(
                    bundle,
                    field="selected_expert_prob_mass",
                    title="Selected Expert Probability Mass by Layer",
                    xlabel="Selected expert probability mass",
                    stem="selected_mass_ridge.png",
                    output_dir=routing_dir,
                )
            ),
        }
        aggregate_path, layer_paths = plot_residual_coactivation(bundle, routing_dir)
        per_model_paths["coactivation_residual_heatmap"] = str(aggregate_path)
        per_model_paths["coactivation_residual_layer_heatmaps"] = [str(path) for path in layer_paths]
        per_model_paths["upset_plots"] = [str(path) for path in plot_upset_plots(bundle, routing_dir)]
        manifest["models"][bundle["model_name"]] = per_model_paths

    cross_dir = output_root / "cross_model"
    cross_dir.mkdir(parents=True, exist_ok=True)
    manifest["cross_model"]["vocab_paper_style"] = [
        str(path) for path in plot_vocab_paper_style(bundles, cross_dir)
    ]
    manifest["cross_model"]["domain_ridge"] = [
        str(path) for path in plot_domain_ridge(bundles, cross_dir)
    ]
    manifest["cross_model"]["selected_prob_mass_bar"] = str(
        plot_cross_model_metric(
            bundles,
            metric_key="mean_selected_expert_prob_mass",
            ylabel="Mean selected expert probability mass",
            title="Selected Expert Probability Mass Across a2/a4/a7",
            stem="selected_expert_probability_mass_across_models.png",
            output_dir=cross_dir,
        )
    )
    manifest["cross_model"]["entropy_bar"] = str(
        plot_cross_model_metric(
            bundles,
            metric_key="mean_token_entropy",
            ylabel="Mean token entropy",
            title="Prompt Router Entropy Across a2/a4/a7",
            stem="prompt_entropy_across_models.png",
            output_dir=cross_dir,
        )
    )

    manifest_path = output_root / "collapse_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote collapse diagnostics manifest to {manifest_path}")


if __name__ == "__main__":
    main()
