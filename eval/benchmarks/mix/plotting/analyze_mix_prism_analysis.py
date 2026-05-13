from __future__ import annotations

import argparse
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


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "prism_analysis" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_prism_analysis"

MODEL_LABELS = {
    "FlexOlmo-8x7B-1T-a4-55B-v2": "a4-55B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2-rt": "a4-55B-v2-rt",
}

DATASET_LABELS = {
    "mkqa_en_da": "MGQA (EN/DA)",
    "gsm8k_subset": "GSM8K",
    "mbpp_subset": "MBPP",
    "pubmedqa_subset": "PubMedQA",
}

COMPONENT_LABELS = {
    "embedding": "Embedding",
    "attention_out": "Attention",
    "moe_out": "MoE",
    "public_moe_out": "Public MoE",
    "nonpublic_moe_out": "Non-public MoE",
    "block_update": "Block update",
    "final_hidden_state": "Final hidden",
    "model_logits": "Model logits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize compact prism-analysis outputs for the 55B FlexOlmo pair.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def load_records(results_root: Path) -> pd.DataFrame:
    frames = []
    for model_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        model_name = model_dir.name
        for dataset_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            records_path = dataset_dir / "prism_component_records.csv"
            if not records_path.exists():
                continue
            frame = pd.read_csv(records_path)
            if frame.empty:
                continue
            frame["model_name"] = model_name
            frame["dataset_name"] = dataset_dir.name
            frames.append(frame)
    if not frames:
        raise ValueError(f"No prism records were found under {results_root}.")
    combined = pd.concat(frames, ignore_index=True)
    combined["model_label"] = combined["model_name"].map(lambda name: MODEL_LABELS.get(name, name))
    combined["dataset_label"] = combined["dataset_name"].map(lambda name: DATASET_LABELS.get(name, name))
    combined["component_label"] = combined["component"].map(lambda name: COMPONENT_LABELS.get(name, name))
    return combined


def aggregate_records(records: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_name", "model_label", "dataset_name", "dataset_label", "component", "component_label", "layer_idx"]
    summary = (
        records.groupby(group_cols, dropna=False)
        .agg(
            num_examples=("example_id", "nunique"),
            mean_target_logit=("mean_target_logit", "mean"),
            mean_target_rank=("mean_target_rank", "mean"),
            top1_match_rate=("top1_match_rate", "mean"),
            mean_target_margin=("mean_target_margin", "mean"),
            mean_vector_norm=("mean_vector_norm", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["dataset_name", "model_name", "layer_idx", "component"])


def plot_layerwise_components(summary: pd.DataFrame, output_path: Path) -> None:
    layerwise = summary[summary["component"].isin({"attention_out", "moe_out", "block_update"})].copy()
    if layerwise.empty:
        return
    datasets = list(layerwise["dataset_label"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 3.3 * len(datasets)), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    palette = {"Attention": "#2a9d8f", "MoE": "#e76f51", "Block update": "#264653"}
    style_map = {MODEL_LABELS.get("FlexOlmo-8x7B-1T-a4-55B-v2", "a4-55B-v2"): "-", MODEL_LABELS.get("FlexOlmo-8x7B-1T-a4-55B-v2-rt", "a4-55B-v2-rt"): "--"}

    for ax, dataset_label in zip(axes, datasets):
        subset = layerwise[layerwise["dataset_label"] == dataset_label]
        for (component_label, model_label), frame in subset.groupby(["component_label", "model_label"], sort=False):
            frame = frame.sort_values("layer_idx")
            ax.plot(
                frame["layer_idx"],
                frame["mean_target_logit"],
                label=f"{component_label} / {model_label}",
                color=palette.get(component_label, None),
                linestyle=style_map.get(model_label, "-"),
                linewidth=2,
                marker="o",
                alpha=0.95,
            )
        ax.set_title(dataset_label)
        ax.set_ylabel("Mean target logit")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Layer")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_public_vs_nonpublic(summary: pd.DataFrame, output_path: Path) -> None:
    subset = summary[summary["component"].isin({"public_moe_out", "nonpublic_moe_out"})].copy()
    if subset.empty:
        return
    datasets = list(subset["dataset_label"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 3.3 * len(datasets)), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    palette = {"Public MoE": "#111111", "Non-public MoE": "#6c8ebf"}
    style_map = {MODEL_LABELS.get("FlexOlmo-8x7B-1T-a4-55B-v2", "a4-55B-v2"): "-", MODEL_LABELS.get("FlexOlmo-8x7B-1T-a4-55B-v2-rt", "a4-55B-v2-rt"): "--"}
    for ax, dataset_label in zip(axes, datasets):
        current = subset[subset["dataset_label"] == dataset_label]
        for (component_label, model_label), frame in current.groupby(["component_label", "model_label"], sort=False):
            frame = frame.sort_values("layer_idx")
            ax.plot(
                frame["layer_idx"],
                frame["mean_target_logit"],
                label=f"{component_label} / {model_label}",
                color=palette.get(component_label, None),
                linestyle=style_map.get(model_label, "-"),
                linewidth=2,
                marker="o",
                alpha=0.95,
            )
        ax.set_title(dataset_label)
        ax.set_ylabel("Mean target logit")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Layer")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_terminal_components(summary: pd.DataFrame, output_path: Path) -> None:
    subset = summary[summary["component"].isin({"embedding", "final_hidden_state", "model_logits"})].copy()
    last_layer = summary[summary["component"].isin({"attention_out", "moe_out", "block_update"})].copy()
    if last_layer.empty and subset.empty:
        return
    if not last_layer.empty:
        last_by_model_dataset = last_layer.groupby(["dataset_name", "model_name"])["layer_idx"].max().reset_index()
        subset_last = last_layer.merge(last_by_model_dataset, on=["dataset_name", "model_name", "layer_idx"], how="inner")
        subset = pd.concat([subset, subset_last], ignore_index=True)
    subset["component_plot_label"] = subset["component_label"]
    ordered_components = [
        "Embedding",
        "Attention",
        "MoE",
        "Block update",
        "Final hidden",
        "Model logits",
    ]
    subset["component_plot_label"] = pd.Categorical(subset["component_plot_label"], categories=ordered_components, ordered=True)
    subset = subset.sort_values("component_plot_label")
    g = sns.catplot(
        data=subset,
        x="component_plot_label",
        y="mean_target_logit",
        hue="model_label",
        col="dataset_label",
        kind="bar",
        height=3.6,
        aspect=0.95,
        sharey=False,
    )
    g.set_axis_labels("", "Mean target logit")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    if g._legend is not None:
        g._legend.remove()
    g.fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    g.fig.tight_layout(rect=(0, 0, 1, 0.95))
    g.fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)


def write_readme(path: Path, results_root: Path, output_root: Path) -> None:
    lines = [
        "# Mix Prism Analysis",
        "",
        "## What this analysis measures",
        "This analysis traces approximate answer-token contributions from compact FlexOlmo components.",
        "It focuses on teacher-forced reference answers and projects selected intermediate representations through the final unembedding head.",
        "",
        "Tracked components:",
        "- `embedding`: prompt/answer token embeddings before any decoder computation.",
        "- `attention_out`: raw self-attention submodule output from a selected decoder layer.",
        "- `moe_out`: raw sparse MoE output from a selected decoder layer.",
        "- `public_moe_out`: approximate MoE contribution from the public expert only.",
        "- `nonpublic_moe_out`: approximate MoE contribution from the non-public experts.",
        "- `block_update`: full decoder-layer residual update (`layer_out - layer_in`).",
        "- `final_hidden_state`: the final normalized hidden state passed into the LM head.",
        "- `model_logits`: the actual model logits on the answer-token prediction positions.",
        "",
        "## Key artifacts",
        "- `prism_component_records.csv`: combined per-example records across models/datasets.",
        "- `prism_component_summary.csv`: aggregated component summary table.",
        "- `prism_layerwise_target_logit.png`: layer-wise attention/MoE/block target-logit trends.",
        "- `prism_public_vs_nonpublic.png`: public vs non-public MoE target-logit trends.",
        "- `prism_terminal_components.png`: compact overview of embedding / last-layer / final components.",
        "",
        "## How to run",
        "Capture compact prism records:",
        "```bash",
        "python3 eval/benchmarks/mix/runners/run_mix_prism_analysis_suite.py \\",
        "  --config eval/benchmarks/mix/configs/mix_suite_config.55b_pair.prism_analysis.json",
        "```",
        "",
        "Aggregate and plot the results:",
        "```bash",
        "python3 eval/benchmarks/mix/plotting/analyze_mix_prism_analysis.py \\",
        f"  --results-root {results_root} \\",
        f"  --output-root {output_root}",
        "```",
        "",
        "## How to interpret",
        "- Larger `mean_target_logit` means the projected component points more strongly toward the reference answer tokens.",
        "- `mean_target_rank` measures how highly the reference tokens rank under the projected component alone; lower is better.",
        "- `top1_match_rate` measures how often the projected component alone places the reference token at rank 1.",
        "- Intermediate component projections are approximate. They are intended for comparative diagnosis, not exact additive attribution.",
        "- `public_moe_out` vs `nonpublic_moe_out` is especially useful for checking whether the public expert dominates answer-directed contribution even when specialist routing is active.",
        "",
        "Notes:",
        "- This track intentionally stores compact answer-token summaries instead of raw activations.",
        "- Intermediate components are projected through the final model norm + LM head as a prism-style approximation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = load_records(results_root)
    summary = aggregate_records(records)
    records.to_csv(output_root / "prism_component_records.csv", index=False)
    summary.to_csv(output_root / "prism_component_summary.csv", index=False)

    plot_layerwise_components(summary, output_root / "prism_layerwise_target_logit.png")
    plot_public_vs_nonpublic(summary, output_root / "prism_public_vs_nonpublic.png")
    plot_terminal_components(summary, output_root / "prism_terminal_components.png")
    write_readme(output_root / "README.md", results_root=results_root, output_root=output_root)
    print(f"Wrote prism analysis outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
