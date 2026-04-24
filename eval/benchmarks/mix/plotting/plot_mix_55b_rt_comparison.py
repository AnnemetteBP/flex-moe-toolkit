from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_V2_ROOT = (
    PROJECT_ROOT
    / "FlexOlmo-8x7B-1T-a4-55B-v2-small"
    / "eval_results"
    / "mix"
    / "full"
    / "flexolmo"
    / "a4"
    / "FlexOlmo-8x7B-1T-a4-55B-v2"
)
DEFAULT_RT_ROOT = (
    PROJECT_ROOT
    / "FlexOlmo-8x7B-1T-a4-55B-v2-rt-small"
    / "eval_results"
    / "mix"
    / "full"
    / "flexolmo"
    / "a4"
    / "FlexOlmo-8x7B-1T-a4-55B-v2-rt"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_v2_vs_rt"

DATASET_ORDER = [
    "mkqa_en_da",
    "gsm8k_subset",
    "pubmedqa_subset",
    "ag_news_subset",
    "common_gen_subset",
    "mbpp_subset",
]
METRIC_ORDER = [
    ("mean_token_entropy", "Mean Token Entropy"),
    ("mean_top1_top2_margin", "Top-1/Top-2 Margin"),
    ("mean_selected_expert_prob_mass", "Selected Expert Prob. Mass"),
    ("offdiag_ratio", "Offdiag. Ratio"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare aggregate routing behavior for FlexOlmo 55B-v2 vs 55B-v2-rt on the mix suite."
    )
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--rt-root", type=Path, default=DEFAULT_RT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_routing_batches(model_root: Path, model_label: str) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset_name in DATASET_ORDER:
        analysis_path = model_root / dataset_name / "native_full" / "routing_analysis.jsonl"
        if not analysis_path.exists():
            continue
        for record in load_jsonl(analysis_path):
            if record.get("record_type") != "routing_batch":
                continue
            usage = record.get("usage", [])
            row = {
                "model_label": model_label,
                "dataset_name": dataset_name,
                "language": record.get("language", "unknown"),
                "example_id": record.get("example_id"),
                "mean_token_entropy": float(record["mean_token_entropy"]),
                "mean_top1_prob": float(record["mean_top1_prob"]),
                "mean_top2_prob": float(record["mean_top2_prob"]),
                "mean_top1_top2_margin": float(record["mean_top1_top2_margin"]),
                "mean_selected_expert_prob_mass": float(record["mean_selected_expert_prob_mass"]),
                "offdiag_ratio": float(record["offdiag_ratio"]),
            }
            for idx, value in enumerate(usage):
                row[f"usage_{idx}"] = float(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"No routing_batch rows found under {model_root}")
    return pd.DataFrame(rows)


def dataset_language_label(dataset_name: str, language: str) -> str:
    if dataset_name == "mkqa_en_da":
        return f"{dataset_name}:{language}"
    return dataset_name


def build_metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.assign(dataset_label=frame.apply(lambda row: dataset_language_label(row["dataset_name"], row["language"]), axis=1))
        .groupby(["model_label", "dataset_name", "dataset_label", "language"], as_index=False)[
            [name for name, _ in METRIC_ORDER]
        ]
        .mean()
    )
    return grouped


def build_usage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    usage_columns = sorted(column for column in frame.columns if column.startswith("usage_"))
    grouped = (
        frame.assign(dataset_label=frame.apply(lambda row: dataset_language_label(row["dataset_name"], row["language"]), axis=1))
        .groupby(["model_label", "dataset_name", "dataset_label", "language"], as_index=False)[usage_columns]
        .mean()
    )
    return grouped


def save_csvs(metric_summary: pd.DataFrame, usage_summary: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    metric_summary.to_csv(output_root / "mix_55b_metric_summary.csv", index=False)
    usage_summary.to_csv(output_root / "mix_55b_expert_usage_summary.csv", index=False)


def plot_metric_bars(metric_summary: pd.DataFrame, output_root: Path) -> None:
    ordered_labels = []
    for dataset_name in DATASET_ORDER:
        subset = metric_summary[metric_summary["dataset_name"] == dataset_name]
        labels = subset["dataset_label"].drop_duplicates().tolist()
        ordered_labels.extend(sorted(labels))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()
    model_order = ["55B-v2", "55B-v2-rt"]
    colors = {"55B-v2": "#5B6C8F", "55B-v2-rt": "#C96B3B"}

    x = np.arange(len(ordered_labels))
    width = 0.38

    for ax, (metric_key, title) in zip(axes, METRIC_ORDER):
        for offset_idx, model_label in enumerate(model_order):
            subset = metric_summary[metric_summary["model_label"] == model_label]
            values = []
            for label in ordered_labels:
                row = subset[subset["dataset_label"] == label]
                values.append(float(row.iloc[0][metric_key]) if not row.empty else np.nan)
            ax.bar(
                x + (offset_idx - 0.5) * width,
                values,
                width=width,
                label=model_label,
                color=colors[model_label],
                alpha=0.92,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper right")
    fig.suptitle("Mix Routing Comparison: 55B-v2 vs 55B-v2-rt", fontsize=16)
    fig.savefig(output_root / "mix_55b_metric_comparison.png", dpi=200)
    plt.close(fig)


def plot_usage_heatmaps(usage_summary: pd.DataFrame, output_root: Path) -> None:
    usage_columns = sorted(column for column in usage_summary.columns if column.startswith("usage_"))
    dataset_labels = []
    for dataset_name in DATASET_ORDER:
        subset = usage_summary[usage_summary["dataset_name"] == dataset_name]
        dataset_labels.extend(sorted(subset["dataset_label"].drop_duplicates().tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    model_order = ["55B-v2", "55B-v2-rt"]

    for ax, model_label in zip(axes, model_order):
        subset = usage_summary[usage_summary["model_label"] == model_label]
        matrix = []
        for label in dataset_labels:
            row = subset[subset["dataset_label"] == label]
            if row.empty:
                matrix.append([np.nan] * len(usage_columns))
            else:
                matrix.append([float(row.iloc[0][column]) for column in usage_columns])
        image = ax.imshow(np.array(matrix), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
        ax.set_title(model_label)
        ax.set_xticks(range(len(usage_columns)))
        ax.set_xticklabels([column.replace("usage_", "E") for column in usage_columns])
        ax.set_yticks(range(len(dataset_labels)))
        ax.set_yticklabels(dataset_labels)
    cbar = fig.colorbar(image, ax=axes, shrink=0.9)
    cbar.set_label("Mean Expert Usage Share")
    fig.suptitle("Expert Usage by Dataset", fontsize=16)
    fig.savefig(output_root / "mix_55b_expert_usage_heatmaps.png", dpi=200)
    plt.close(fig)


def write_readme(metric_summary: pd.DataFrame, output_root: Path) -> None:
    pivot_rows = []
    for dataset_label in metric_summary["dataset_label"].drop_duplicates():
        v2 = metric_summary[
            (metric_summary["dataset_label"] == dataset_label) & (metric_summary["model_label"] == "55B-v2")
        ]
        rt = metric_summary[
            (metric_summary["dataset_label"] == dataset_label) & (metric_summary["model_label"] == "55B-v2-rt")
        ]
        if v2.empty or rt.empty:
            continue
        pivot_rows.append(
            {
                "dataset_label": dataset_label,
                "entropy_delta_rt_minus_v2": float(rt.iloc[0]["mean_token_entropy"] - v2.iloc[0]["mean_token_entropy"]),
                "margin_delta_rt_minus_v2": float(
                    rt.iloc[0]["mean_top1_top2_margin"] - v2.iloc[0]["mean_top1_top2_margin"]
                ),
                "selected_mass_delta_rt_minus_v2": float(
                    rt.iloc[0]["mean_selected_expert_prob_mass"] - v2.iloc[0]["mean_selected_expert_prob_mass"]
                ),
            }
        )
    lines = [
        "# Mix 55B Routing Comparison",
        "",
        "This directory compares aggregate routing behavior for:",
        "- `FlexOlmo-8x7B-1T-a4-55B-v2`",
        "- `FlexOlmo-8x7B-1T-a4-55B-v2-rt`",
        "",
        "Transferred files only include `routing_analysis.jsonl`, `routing_summary.jsonl`, and manifests.",
        "So these outputs support aggregate routing interpretation, not token-level vocab/domain analyses.",
        "",
        "## Per-Dataset Deltas (`rt - v2`)",
        "",
    ]
    for row in pivot_rows:
        lines.append(
            f"- `{row['dataset_label']}`: "
            f"entropy {row['entropy_delta_rt_minus_v2']:+.4f}, "
            f"margin {row['margin_delta_rt_minus_v2']:+.4f}, "
            f"selected-mass {row['selected_mass_delta_rt_minus_v2']:+.4f}"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    v2_frame = load_routing_batches(args.v2_root, "55B-v2")
    rt_frame = load_routing_batches(args.rt_root, "55B-v2-rt")
    combined = pd.concat([v2_frame, rt_frame], ignore_index=True)

    metric_summary = build_metric_summary(combined)
    usage_summary = build_usage_summary(combined)

    save_csvs(metric_summary, usage_summary, args.output_root)
    plot_metric_bars(metric_summary, args.output_root)
    plot_usage_heatmaps(usage_summary, args.output_root)
    write_readme(metric_summary, args.output_root)

    print(f"Wrote comparison outputs to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
