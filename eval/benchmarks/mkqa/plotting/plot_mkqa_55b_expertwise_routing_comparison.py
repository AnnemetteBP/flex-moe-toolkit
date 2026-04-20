from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
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


DEFAULT_MODELS = [
    "FlexOlmo-8x7B-1T-a4-55B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2-rt",
    "FlexOlmo-8x7B-1T-a8-55B-v2",
    "FlexOlmo-8x7B-1T-a8-55B-v2-rt",
]
DEFAULT_ASSUMED_EXPERT_LABELS = {
    0: "public",
    1: "code",
    2: "creative",
    3: "math",
    4: "news",
    5: "pes2o",
    6: "reddit",
    7: "danish",
}
LANGUAGE_COLORS = {
    "en": "#4c78a8",
    "da": "#f58518",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate expert-wise routing comparison plots for focused 55B A4/A8 models."
    )
    parser.add_argument("--a4-root", default="eval_results/mkqa/full/flexolmo/a4")
    parser.add_argument("--a8-root", default="eval_results/mkqa/full/flexolmo/a8")
    parser.add_argument(
        "--output-root",
        default="eval_results/mkqa/full/flexolmo/comparisons/a4_a8_55b_expertwise",
    )
    parser.add_argument("--expert-label", action="append", default=[])
    return parser.parse_args()


def parse_expert_labels(raw_labels: list[str]) -> dict[int, str]:
    labels = dict(DEFAULT_ASSUMED_EXPERT_LABELS)
    for raw_item in raw_labels:
        if "=" not in raw_item:
            raise ValueError(f"Invalid --expert-label value `{raw_item}`. Expected <idx>=<label>.")
        idx_str, label = raw_item.split("=", 1)
        labels[int(idx_str.strip())] = label.strip()
    return labels


def expert_label(expert_idx: int, labels: dict[int, str]) -> str:
    return labels.get(expert_idx, f"expert_{expert_idx}")


def model_root_for_name(model_name: str, a4_root: Path, a8_root: Path) -> Path:
    if "-a4-" in model_name:
        return a4_root
    if "-a8-" in model_name:
        return a8_root
    raise ValueError(f"Could not determine family root for model `{model_name}`.")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_ids(value) -> list[int]:
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(flatten_ids(item))
        return flattened
    return [int(value)]


def load_joined_records(model_name: str, a4_root: Path, a8_root: Path) -> list[dict]:
    family_root = model_root_for_name(model_name, a4_root, a8_root)
    routing_records = load_jsonl(
        family_root / "routing" / model_name / "mkqa_en_da" / "native_full" / "routing_records.jsonl"
    )
    accuracy_records = load_jsonl(
        family_root / "accuracy" / model_name / "native_full" / "accuracy_records.jsonl"
    )
    accuracy_by_example = {row["example_id"]: row for row in accuracy_records}
    joined = []
    for record in routing_records:
        accuracy = accuracy_by_example.get(record["example_id"], {})
        joined.append(
            {
                **record,
                "exact_match": accuracy.get("exact_match"),
                "relaxed_match": accuracy.get("relaxed_match"),
                "token_f1": accuracy.get("token_f1"),
                "is_scorable": accuracy.get("is_scorable"),
            }
        )
    return joined


def mean_prob_by_language(records: list[dict]) -> dict[str, np.ndarray]:
    by_language = {}
    languages = sorted({record["language"] for record in records})
    num_layers = len(records[0]["prompt_router_probs_by_layer"])
    num_experts = len(records[0]["prompt_router_probs_by_layer"][0][0])
    for language in languages:
        language_records = [record for record in records if record["language"] == language]
        matrix = np.zeros((num_layers, num_experts), dtype=float)
        for layer_idx in range(num_layers):
            token_rows = []
            for record in language_records:
                token_rows.extend(record["prompt_router_probs_by_layer"][layer_idx])
            matrix[layer_idx] = np.mean(np.asarray(token_rows, dtype=float), axis=0)
        by_language[language] = matrix
    return by_language


def rank_share_by_language(records: list[dict], rank_field: str) -> dict[str, np.ndarray]:
    by_language = {}
    num_layers = len(records[0]["prompt_router_token_summaries_by_layer"])
    num_experts = len(records[0]["prompt_router_probs_by_layer"][0][0])
    for language in sorted({record["language"] for record in records}):
        language_records = [record for record in records if record["language"] == language]
        matrix = np.zeros((num_layers, num_experts), dtype=float)
        for layer_idx in range(num_layers):
            counts = np.zeros(num_experts, dtype=float)
            total = 0.0
            for record in language_records:
                ids = flatten_ids(record["prompt_router_token_summaries_by_layer"][layer_idx][rank_field])
                for expert_idx in ids:
                    counts[expert_idx] += 1.0
                total += len(ids)
            matrix[layer_idx] = counts / total if total else counts
        by_language[language] = matrix
    return by_language


def top1_top2_confusion(records: list[dict], language: str | None = None) -> np.ndarray:
    filtered = [record for record in records if language is None or record["language"] == language]
    num_experts = len(filtered[0]["prompt_router_probs_by_layer"][0][0])
    matrix = np.zeros((num_experts, num_experts), dtype=float)
    for record in filtered:
        for layer_summary in record["prompt_router_token_summaries_by_layer"]:
            top1 = flatten_ids(layer_summary["top1_expert_ids"])
            top2 = flatten_ids(layer_summary["top2_expert_ids"])
            for left, right in zip(top1, top2):
                matrix[left, right] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def mean_prob_for_subset(records: list[dict], subset_filter) -> np.ndarray | None:
    filtered = [record for record in records if subset_filter(record)]
    if not filtered:
        return None
    num_layers = len(filtered[0]["prompt_router_probs_by_layer"])
    num_experts = len(filtered[0]["prompt_router_probs_by_layer"][0][0])
    matrix = np.zeros((num_layers, num_experts), dtype=float)
    for layer_idx in range(num_layers):
        token_rows = []
        for record in filtered:
            token_rows.extend(record["prompt_router_probs_by_layer"][layer_idx])
        matrix[layer_idx] = np.mean(np.asarray(token_rows, dtype=float), axis=0)
    return matrix


def plot_mean_probability_profiles(model_names: list[str], model_rows: dict[str, dict], expert_labels: dict[int, str], output_path: Path) -> None:
    fig, axes = plt.subplots(len(model_names), 2, figsize=(13.5, max(4.0 * len(model_names), 5.0)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for row_idx, model_name in enumerate(model_names):
        for col_idx, language in enumerate(("en", "da")):
            ax = axes[row_idx, col_idx]
            matrix = model_rows[model_name]["mean_prob_by_language"][language]
            sns.heatmap(
                matrix.T,
                cmap="mako",
                vmin=0.0,
                vmax=max(0.3, float(matrix.max())),
                ax=ax,
                cbar=(row_idx == 0 and col_idx == 1),
                cbar_kws={"label": "Mean Router Probability"},
            )
            ax.set_title(f"{model_name.replace('FlexOlmo-8x7B-1T-', '')} | {language.upper()}")
            ax.set_xlabel("Layer")
            if col_idx == 0:
                ax.set_ylabel("Expert")
                ax.set_yticks(np.arange(matrix.shape[1]) + 0.5)
                ax.set_yticklabels([expert_label(idx, expert_labels) for idx in range(matrix.shape[1])], rotation=0)
            else:
                ax.set_ylabel("")
    fig.suptitle("Expert-Wise Mean Router Probability by Language", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rank_share_profiles(model_names: list[str], model_rows: dict[str, dict], expert_labels: dict[int, str], rank_key: str, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(len(model_names), 2, figsize=(13.5, max(4.0 * len(model_names), 5.0)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for row_idx, model_name in enumerate(model_names):
        for col_idx, language in enumerate(("en", "da")):
            ax = axes[row_idx, col_idx]
            matrix = model_rows[model_name][rank_key][language]
            sns.heatmap(
                matrix.T,
                cmap="crest",
                vmin=0.0,
                vmax=max(0.3, float(matrix.max())),
                ax=ax,
                cbar=(row_idx == 0 and col_idx == 1),
                cbar_kws={"label": "Share"},
            )
            ax.set_title(f"{model_name.replace('FlexOlmo-8x7B-1T-', '')} | {language.upper()}")
            ax.set_xlabel("Layer")
            if col_idx == 0:
                ax.set_ylabel("Expert")
                ax.set_yticks(np.arange(matrix.shape[1]) + 0.5)
                ax.set_yticklabels([expert_label(idx, expert_labels) for idx in range(matrix.shape[1])], rotation=0)
            else:
                ax.set_ylabel("")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(model_names: list[str], model_rows: dict[str, dict], expert_labels: dict[int, str], output_path: Path) -> None:
    fig, axes = plt.subplots(2, len(model_names), figsize=(5.0 * len(model_names), 9.0), sharex=True, sharey=True)
    languages = ("en", "da")
    for col_idx, model_name in enumerate(model_names):
        for row_idx, language in enumerate(languages):
            ax = axes[row_idx, col_idx]
            matrix = model_rows[model_name]["top1_top2_confusion"][language]
            sns.heatmap(
                matrix,
                cmap="rocket_r",
                vmin=0.0,
                vmax=max(0.5, float(matrix.max())),
                ax=ax,
                cbar=(row_idx == 0 and col_idx == len(model_names) - 1),
                cbar_kws={"label": "P(top2 | top1)"},
            )
            ax.set_title(f"{model_name.replace('FlexOlmo-8x7B-1T-', '')} | {language.upper()}")
            ax.set_xlabel("Top-2 Expert")
            ax.set_ylabel("Top-1 Expert")
            tick_labels = [expert_label(idx, expert_labels) for idx in range(matrix.shape[0])]
            ax.set_xticks(np.arange(matrix.shape[0]) + 0.5)
            ax.set_xticklabels(tick_labels, rotation=35, ha="right")
            ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
            ax.set_yticklabels(tick_labels, rotation=0)
    fig.suptitle("Expert-Pair Competition: Top-1 vs Top-2", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_danish_correctness(model_names: list[str], model_rows: dict[str, dict], expert_labels: dict[int, str], output_path: Path) -> None:
    fig, axes = plt.subplots(2, len(model_names), figsize=(5.0 * len(model_names), 8.5), sharex=True, sharey=True)
    subset_specs = [
        ("da_correct_prob", "Danish Correct"),
        ("da_incorrect_prob", "Danish Incorrect"),
    ]
    for row_idx, (metric_key, title) in enumerate(subset_specs):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            matrix = model_rows[model_name][metric_key]
            if matrix is None:
                ax.axis("off")
                continue
            sns.heatmap(
                matrix.T,
                cmap="mako",
                vmin=0.0,
                vmax=max(0.3, float(matrix.max())),
                ax=ax,
                cbar=(row_idx == 0 and col_idx == len(model_names) - 1),
                cbar_kws={"label": "Mean Router Probability"},
            )
            ax.set_title(f"{model_name.replace('FlexOlmo-8x7B-1T-', '')} | {title}")
            ax.set_xlabel("Layer")
            if col_idx == 0:
                ax.set_ylabel("Expert")
                ax.set_yticks(np.arange(matrix.shape[1]) + 0.5)
                ax.set_yticklabels([expert_label(idx, expert_labels) for idx in range(matrix.shape[1])], rotation=0)
            else:
                ax.set_ylabel("")
    fig.suptitle("Danish Examples: Correct vs Incorrect Expert Probabilities", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(model_names: list[str], output_path: Path) -> None:
    lines = [
        "# Expert-Wise Routing Comparison",
        "",
        "- These plots use saved routing records to ask whether the router treats experts as weakly separable or effectively interchangeable.",
        "- The emphasis is on expert-wise probability mass, top-1/top-2 competition, and Danish correct-vs-incorrect splits.",
        "- Expert labels are provisional for the `8x7B` checkpoints: `0=public`, `1=code`, `2=creative`, `3=math`, `4=news`, `5=pes2o`, `6=reddit`, `7=danish`.",
        "- The strongest support is for `0=public`; the final `7=danish` assignment remains an informed working assumption until confirmed from the original merge or training command.",
        "",
        "## Models",
        "",
    ]
    for model_name in model_names:
        lines.append(f"- `{model_name}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    a4_root = Path(args.a4_root)
    a8_root = Path(args.a8_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    expert_labels = parse_expert_labels(args.expert_label)

    model_rows = {}
    for model_name in DEFAULT_MODELS:
        records = load_joined_records(model_name, a4_root, a8_root)
        model_rows[model_name] = {
            "mean_prob_by_language": mean_prob_by_language(records),
            "top1_share": rank_share_by_language(records, "top1_expert_ids"),
            "top2_share": rank_share_by_language(records, "top2_expert_ids"),
            "top1_top2_confusion": {
                "en": top1_top2_confusion(records, language="en"),
                "da": top1_top2_confusion(records, language="da"),
            },
            "da_correct_prob": mean_prob_for_subset(
                records,
                lambda record: record["language"] == "da" and bool(record.get("relaxed_match")),
            ),
            "da_incorrect_prob": mean_prob_for_subset(
                records,
                lambda record: record["language"] == "da" and record.get("is_scorable") and not bool(record.get("relaxed_match")),
            ),
        }

    plot_mean_probability_profiles(DEFAULT_MODELS, model_rows, expert_labels, output_root / "expertwise_mean_probability.png")
    plot_rank_share_profiles(
        DEFAULT_MODELS,
        model_rows,
        expert_labels,
        rank_key="top1_share",
        title="Expert-Wise Top-1 Share by Language",
        output_path=output_root / "expertwise_top1_share.png",
    )
    plot_rank_share_profiles(
        DEFAULT_MODELS,
        model_rows,
        expert_labels,
        rank_key="top2_share",
        title="Expert-Wise Top-2 Share by Language",
        output_path=output_root / "expertwise_top2_share.png",
    )
    plot_confusion_matrices(DEFAULT_MODELS, model_rows, expert_labels, output_root / "expertwise_top1_top2_confusion.png")
    plot_danish_correctness(DEFAULT_MODELS, model_rows, expert_labels, output_root / "danish_correct_vs_incorrect_probability.png")
    write_readme(DEFAULT_MODELS, output_root / "README.md")
    print(f"Wrote expert-wise routing comparison plots to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
