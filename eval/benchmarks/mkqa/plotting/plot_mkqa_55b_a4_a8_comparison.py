from __future__ import annotations

import argparse
import csv
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
METRIC_COLORS = {
    "overall": "#4c78a8",
    "en": "#54a24b",
    "da": "#f58518",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate focused comparison plots for A4 vs A8 55B v2/rt MKQA results."
    )
    parser.add_argument(
        "--a4-root",
        default="eval_results/mkqa/full/flexolmo/a4",
        help="Results root for A4 family.",
    )
    parser.add_argument(
        "--a8-root",
        default="eval_results/mkqa/full/flexolmo/a8",
        help="Results root for A8 family.",
    )
    parser.add_argument(
        "--output-root",
        default="eval_results/mkqa/full/flexolmo/comparisons/a4_a8_55b",
        help="Directory where the comparison plots will be written.",
    )
    parser.add_argument(
        "--expert-label",
        action="append",
        default=[],
        help="Optional expert label mapping in the form <idx>=<label>.",
    )
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


def flatten_token_expert_ids(value) -> list[int]:
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(flatten_token_expert_ids(item))
        return flattened
    return [int(value)]


def top_token_combinations_by_language(records: list[dict], min_top_n: int = 4, max_top_n: int = 8) -> list[dict]:
    total_counter: Counter[tuple[int, ...]] = Counter()
    language_counters: dict[str, Counter[tuple[int, ...]]] = {}
    for record in records:
        language = record["language"]
        language_counter = language_counters.setdefault(language, Counter())
        for raw_key, count in (record.get("token_topk_combination_counts") or {}).items():
            combo = tuple(int(part) for part in raw_key.split(",") if part)
            total_counter[combo] += int(count)
            language_counter[combo] += int(count)

    observed_experts = set()
    for combo in total_counter:
        observed_experts.update(combo)

    rows = []
    covered_experts: set[int] = set()
    for combo, total_count in total_counter.most_common():
        row = {"combo": combo, "total": int(total_count)}
        for language, counter in language_counters.items():
            row[language] = int(counter.get(combo, 0))
        rows.append(row)
        covered_experts.update(combo)
        if len(rows) >= min_top_n and covered_experts >= observed_experts:
            break
        if len(rows) >= max_top_n:
            break
    return rows


def mean_layer_metric_by_language(records: list[dict], metric_key: str) -> dict[str, np.ndarray]:
    by_language = {}
    for language in sorted({record["language"] for record in records}):
        language_records = [record for record in records if record["language"] == language]
        rows = []
        for record in language_records:
            rows.append([float(layer_metrics[metric_key]) for layer_metrics in record["layer_batch_routing_metrics"]])
        by_language[language] = np.mean(np.array(rows, dtype=float), axis=0)
    return by_language


def aggregate_rank_share_by_language(records: list[dict], rank_field: str, expert_idx: int) -> dict[str, np.ndarray]:
    by_language = {}
    num_layers = len(records[0]["prompt_router_token_summaries_by_layer"])
    for language in sorted({record["language"] for record in records}):
        language_records = [record for record in records if record["language"] == language]
        shares = np.zeros(num_layers, dtype=float)
        for layer_idx in range(num_layers):
            counts = 0.0
            total = 0.0
            for record in language_records:
                ids = record["prompt_router_token_summaries_by_layer"][layer_idx][rank_field]
                flattened = flatten_token_expert_ids(ids)
                counts += sum(1 for value in flattened if value == expert_idx)
                total += len(flattened)
            shares[layer_idx] = counts / total if total else 0.0
        by_language[language] = shares
    return by_language


def load_accuracy_row(model_name: str, a4_root: Path, a8_root: Path) -> dict:
    family_root = model_root_for_name(model_name, a4_root, a8_root)
    overview_path = family_root / "accuracy" / "mkqa_accuracy_overview.csv"
    with overview_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["model_name"] == model_name:
            for key in (
                "exact_match_accuracy",
                "relaxed_match_accuracy",
                "mean_token_f1",
                "en_exact_match_accuracy",
                "en_relaxed_match_accuracy",
                "da_exact_match_accuracy",
                "da_relaxed_match_accuracy",
                "en_mean_token_f1",
                "da_mean_token_f1",
            ):
                row[key] = float(row[key])
            return row

    summary_path = family_root / "accuracy" / model_name / "native_full" / "accuracy_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "model_name": model_name,
        "exact_match_accuracy": float(summary["exact_match_accuracy"]),
        "relaxed_match_accuracy": float(summary["relaxed_match_accuracy"]),
        "mean_token_f1": float(summary["mean_token_f1"]),
        "en_exact_match_accuracy": float(summary["by_language"].get("en", {}).get("exact_match_accuracy", 0.0)),
        "en_relaxed_match_accuracy": float(summary["by_language"].get("en", {}).get("relaxed_match_accuracy", 0.0)),
        "da_exact_match_accuracy": float(summary["by_language"].get("da", {}).get("exact_match_accuracy", 0.0)),
        "da_relaxed_match_accuracy": float(summary["by_language"].get("da", {}).get("relaxed_match_accuracy", 0.0)),
        "en_mean_token_f1": float(summary["by_language"].get("en", {}).get("mean_token_f1", 0.0)),
        "da_mean_token_f1": float(summary["by_language"].get("da", {}).get("mean_token_f1", 0.0)),
    }


def format_combo(combo: tuple[int, ...], expert_labels: dict[int, str]) -> str:
    return " + ".join(expert_label(idx, expert_labels) for idx in combo)


def plot_top_combinations(
    model_names: list[str],
    model_rows: dict[str, dict],
    expert_labels: dict[int, str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(model_names), figsize=(5.6 * len(model_names), 5.0), sharey=False)
    if len(model_names) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, model_names):
        combos = model_rows[model_name]["top_combinations_by_language"]
        labels = [format_combo(row["combo"], expert_labels) for row in combos][::-1]
        y_positions = np.arange(len(labels))
        left = np.zeros(len(labels), dtype=float)
        for language in ("en", "da"):
            values = np.array([row.get(language, 0) for row in combos][::-1], dtype=float)
            ax.barh(y_positions, values, left=left, color=LANGUAGE_COLORS[language], label=language.upper())
            left += values
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
        ax.set_xlabel("Token Count")
        ax.grid(True, axis="x", alpha=0.25)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LANGUAGE_COLORS["en"]),
        plt.Rectangle((0, 0), 1, 1, color=LANGUAGE_COLORS["da"]),
    ]
    fig.suptitle("Most Frequent Prompt Expert Combinations by Language", y=0.995)
    fig.legend(handles, ["English", "Danish"], loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_router_dynamics(model_names: list[str], model_rows: dict[str, dict], output_path: Path) -> None:
    metric_panels = [
        ("layer_entropy", "Mean Token Entropy"),
        ("layer_margin", "Mean Top1-Top2 Margin"),
        ("layer_selected_mass", "Mean Selected Expert Probability Mass"),
    ]
    fig, axes = plt.subplots(len(metric_panels), len(model_names), figsize=(5.2 * len(model_names), 10.4), sharex=True)
    axes = np.atleast_2d(axes)
    for row_idx, (metric_key, metric_title) in enumerate(metric_panels):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            for language in ("en", "da"):
                values = model_rows[model_name][metric_key].get(language)
                if values is None:
                    continue
                ax.plot(values, label=language.upper(), color=LANGUAGE_COLORS[language], linewidth=2.0)
            ax.set_title(f"{model_name.replace('FlexOlmo-8x7B-1T-', '')} | {metric_title}")
            ax.set_xlabel("Layer")
            if col_idx == 0:
                ax.set_ylabel(metric_title)
            ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Router Dynamics by Layer", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_danish_rank_share(model_names: list[str], model_rows: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, len(model_names), figsize=(5.2 * len(model_names), 7.0), sharex=True, sharey="row")
    rank_specs = [("danish_top1_share", "Danish Top-1 Share"), ("danish_top2_share", "Danish Top-2 Share")]
    for row_idx, (metric_key, title) in enumerate(rank_specs):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            for language in ("en", "da"):
                values = model_rows[model_name][metric_key].get(language)
                if values is None:
                    continue
                ax.plot(values, label=language.upper(), color=LANGUAGE_COLORS[language], linewidth=2.0)
            ax.set_title(f"{model_name.replace('FlexOlmo-8x7B-1T-', '')} | {title}")
            ax.set_xlabel("Layer")
            if col_idx == 0:
                ax.set_ylabel(title)
            ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Assumed Danish Expert Rank Share by Layer", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(model_names: list[str], accuracy_rows: dict[str, dict], output_path: Path) -> None:
    x = np.arange(len(model_names))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))

    relaxed = [accuracy_rows[name]["relaxed_match_accuracy"] for name in model_names]
    en_relaxed = [accuracy_rows[name]["en_relaxed_match_accuracy"] for name in model_names]
    da_relaxed = [accuracy_rows[name]["da_relaxed_match_accuracy"] for name in model_names]
    axes[0].bar(x - width, relaxed, width=width, color=METRIC_COLORS["overall"], label="Overall")
    axes[0].bar(x, en_relaxed, width=width, color=METRIC_COLORS["en"], label="English")
    axes[0].bar(x + width, da_relaxed, width=width, color=METRIC_COLORS["da"], label="Danish")
    axes[0].set_title("Relaxed Match Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name.replace("FlexOlmo-8x7B-1T-", "") for name in model_names], rotation=25, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    f1 = [accuracy_rows[name]["mean_token_f1"] for name in model_names]
    en_f1 = [accuracy_rows[name]["en_mean_token_f1"] for name in model_names]
    da_f1 = [accuracy_rows[name]["da_mean_token_f1"] for name in model_names]
    axes[1].bar(x - width, f1, width=width, color=METRIC_COLORS["overall"], label="Overall")
    axes[1].bar(x, en_f1, width=width, color=METRIC_COLORS["en"], label="English")
    axes[1].bar(x + width, da_f1, width=width, color=METRIC_COLORS["da"], label="Danish")
    axes[1].set_title("Mean Token F1")
    axes[1].set_ylabel("F1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([name.replace("FlexOlmo-8x7B-1T-", "") for name in model_names], rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("A4 vs A8 Accuracy Comparison", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(model_names: list[str], accuracy_rows: dict[str, dict], output_path: Path) -> None:
    lines = [
        "# A4 vs A8 55B Comparison",
        "",
        "- This folder compares `a4/a8` and `v2/rt` for the 55B checkpoints.",
        "- The combination panel is especially informative for A4; for A8 it mostly confirms that all experts are active.",
        "- The router-dynamics and accuracy panels are the more meaningful A4-vs-A8 comparisons.",
        "",
        "## Accuracy Snapshot",
        "",
    ]
    for model_name in model_names:
        row = accuracy_rows[model_name]
        lines.append(
            f"- `{model_name}`: relaxed={row['relaxed_match_accuracy']:.3%}, "
            f"en={row['en_relaxed_match_accuracy']:.3%}, da={row['da_relaxed_match_accuracy']:.3%}, "
            f"f1={row['mean_token_f1']:.3%}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    a4_root = Path(args.a4_root)
    a8_root = Path(args.a8_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    expert_labels = parse_expert_labels(args.expert_label)

    model_rows = {}
    accuracy_rows = {}
    for model_name in DEFAULT_MODELS:
        family_root = model_root_for_name(model_name, a4_root, a8_root)
        records_path = family_root / "routing" / model_name / "mkqa_en_da" / "native_full" / "routing_records.jsonl"
        records = load_jsonl(records_path)
        model_rows[model_name] = {
            "top_combinations_by_language": top_token_combinations_by_language(records),
            "layer_entropy": mean_layer_metric_by_language(records, "mean_token_entropy"),
            "layer_margin": mean_layer_metric_by_language(records, "mean_top1_top2_margin"),
            "layer_selected_mass": mean_layer_metric_by_language(records, "mean_selected_expert_prob_mass"),
            "danish_top1_share": aggregate_rank_share_by_language(records, "top1_expert_ids", expert_idx=7),
            "danish_top2_share": aggregate_rank_share_by_language(records, "top2_expert_ids", expert_idx=7),
        }
        accuracy_rows[model_name] = load_accuracy_row(model_name, a4_root, a8_root)

    plot_top_combinations(DEFAULT_MODELS, model_rows, expert_labels, output_root / "a4_a8_top_combinations_by_language.png")
    plot_router_dynamics(DEFAULT_MODELS, model_rows, output_root / "a4_a8_router_dynamics.png")
    plot_danish_rank_share(DEFAULT_MODELS, model_rows, output_root / "a4_a8_danish_rank_share.png")
    plot_accuracy(DEFAULT_MODELS, accuracy_rows, output_root / "a4_a8_accuracy.png")
    write_readme(DEFAULT_MODELS, accuracy_rows, output_root / "README.md")
    print(f"Wrote A4 vs A8 comparison plots to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
