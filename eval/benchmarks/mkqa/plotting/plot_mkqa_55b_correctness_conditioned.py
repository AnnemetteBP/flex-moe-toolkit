from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
OUTCOME_COLORS = {
    "correct": "#1b9e77",
    "incorrect": "#d95f02",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate correctness-conditioned routing summaries for focused 55B MKQA runs."
    )
    parser.add_argument("--a4-root", default="eval_results/mkqa/full/flexolmo/a4")
    parser.add_argument("--a8-root", default="eval_results/mkqa/full/flexolmo/a8")
    parser.add_argument(
        "--output-root",
        default="eval_results/mkqa/full/flexolmo/comparisons/a4_a8_55b_correctness",
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


def flatten_numeric(value) -> list[float]:
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(flatten_numeric(item))
        return flattened
    return [float(value)]


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


def per_example_metrics(record: dict, public_idx: int = 0, danish_idx: int = 7) -> dict[str, float | str | bool]:
    num_layers = len(record["prompt_router_token_summaries_by_layer"])
    num_tokens = 0
    top1_public = 0.0
    top2_danish = 0.0
    entropy_values = []
    margin_values = []
    selected_mass_values = []
    public_prob_values = []
    danish_prob_values = []

    for layer_idx in range(num_layers):
        layer_summary = record["prompt_router_token_summaries_by_layer"][layer_idx]
        top1_ids = flatten_ids(layer_summary["top1_expert_ids"])
        top2_ids = flatten_ids(layer_summary["top2_expert_ids"])
        entropies = flatten_numeric(layer_summary["token_entropy"])
        margins = flatten_numeric(layer_summary["top1_top2_margin"])
        selected_mass = flatten_numeric(layer_summary["selected_expert_prob_mass"])
        probs = np.asarray(record["prompt_router_probs_by_layer"][layer_idx], dtype=float)

        num_tokens += len(top1_ids)
        top1_public += sum(1.0 for idx in top1_ids if idx == public_idx)
        top2_danish += sum(1.0 for idx in top2_ids if idx == danish_idx)
        entropy_values.extend(entropies)
        margin_values.extend(margins)
        selected_mass_values.extend(selected_mass)
        public_prob_values.extend(probs[:, public_idx].tolist())
        if probs.shape[1] > danish_idx:
            danish_prob_values.extend(probs[:, danish_idx].tolist())
        else:
            danish_prob_values.extend([math.nan] * probs.shape[0])

    mean_entropy = float(np.mean(entropy_values)) if entropy_values else math.nan
    mean_margin = float(np.mean(margin_values)) if margin_values else math.nan
    mean_selected_mass = float(np.mean(selected_mass_values)) if selected_mass_values else math.nan
    mean_public_prob = float(np.nanmean(public_prob_values)) if public_prob_values else math.nan
    mean_danish_prob = float(np.nanmean(danish_prob_values)) if danish_prob_values else math.nan
    public_minus_danish = mean_public_prob - mean_danish_prob if not math.isnan(mean_danish_prob) else math.nan

    return {
        "example_id": record["example_id"],
        "language": record["language"],
        "relaxed_match": bool(record.get("relaxed_match")),
        "is_scorable": bool(record.get("is_scorable")),
        "token_f1": float(record.get("token_f1") or 0.0),
        "mean_entropy": mean_entropy,
        "mean_margin": mean_margin,
        "mean_selected_mass": mean_selected_mass,
        "mean_public_prob": mean_public_prob,
        "mean_danish_prob": mean_danish_prob,
        "public_minus_danish": public_minus_danish,
        "top1_public_share": (top1_public / num_tokens) if num_tokens else math.nan,
        "top2_danish_share": (top2_danish / num_tokens) if num_tokens else math.nan,
    }


def summarize_rows(rows: list[dict], group_key: str) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["language"], row[group_key])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    metric_keys = [
        "mean_entropy",
        "mean_margin",
        "mean_selected_mass",
        "mean_public_prob",
        "mean_danish_prob",
        "public_minus_danish",
        "top1_public_share",
        "top2_danish_share",
        "token_f1",
    ]
    for (language, outcome), group_rows in sorted(grouped.items()):
        summary = {
            "language": language,
            "outcome": outcome,
            "num_examples": len(group_rows),
        }
        for key in metric_keys:
            values = [float(row[key]) for row in group_rows if not math.isnan(float(row[key]))]
            summary[key] = float(np.mean(values)) if values else math.nan
        summary_rows.append(summary)
    return summary_rows


def outcome_label(row: dict) -> str:
    return "correct" if bool(row.get("relaxed_match")) else "incorrect"


def plot_correctness_bars(model_names: list[str], summary_by_model: dict[str, list[dict]], output_path: Path) -> None:
    metrics = [
        ("mean_entropy", "Mean entropy"),
        ("mean_margin", "Mean top1-top2 margin"),
        ("mean_selected_mass", "Mean selected mass"),
        ("public_minus_danish", "Public - Danish prob"),
    ]
    fig, axes = plt.subplots(len(metrics), len(model_names), figsize=(4.1 * len(model_names), 3.0 * len(metrics)), sharey="row")
    axes = np.atleast_2d(axes)

    x_positions = np.arange(2)
    width = 0.35
    for col_idx, model_name in enumerate(model_names):
        summary_rows = summary_by_model[model_name]
        by_key = {(row["language"], row["outcome"]): row for row in summary_rows}
        for row_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for lang_idx, language in enumerate(("en", "da")):
                values = [
                    by_key.get((language, "correct"), {}).get(metric_key, math.nan),
                    by_key.get((language, "incorrect"), {}).get(metric_key, math.nan),
                ]
                offset = (-0.5 + lang_idx) * width
                ax.bar(
                    x_positions + offset,
                    values,
                    width=width,
                    color=LANGUAGE_COLORS[language],
                    alpha=0.85,
                    label=language.upper() if row_idx == 0 and col_idx == 0 else None,
                )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(["correct", "incorrect"], rotation=0)
            ax.grid(True, axis="y", alpha=0.25)
            if row_idx == 0:
                ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
            if col_idx == 0:
                ax.set_ylabel(metric_label)
    handles = [
        Patch(facecolor=LANGUAGE_COLORS["en"], label="English"),
        Patch(facecolor=LANGUAGE_COLORS["da"], label="Danish"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Correct vs Incorrect Routing Metrics", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_f1_relationships(model_names: list[str], example_rows_by_model: dict[str, list[dict]], output_path: Path) -> None:
    metrics = [
        ("mean_entropy", "Mean entropy"),
        ("mean_margin", "Mean top1-top2 margin"),
        ("public_minus_danish", "Public - Danish prob"),
        ("top2_danish_share", "Top-2 Danish share"),
    ]
    fig, axes = plt.subplots(len(metrics), len(model_names), figsize=(4.1 * len(model_names), 3.0 * len(metrics)), sharex=False, sharey="row")
    axes = np.atleast_2d(axes)

    for col_idx, model_name in enumerate(model_names):
        rows = [row for row in example_rows_by_model[model_name] if row["is_scorable"]]
        for row_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for language in ("en", "da"):
                lang_rows = [row for row in rows if row["language"] == language and not math.isnan(float(row[metric_key]))]
                if not lang_rows:
                    continue
                x = [float(row[metric_key]) for row in lang_rows]
                y = [float(row["token_f1"]) for row in lang_rows]
                ax.scatter(
                    x,
                    y,
                    s=18,
                    alpha=0.55,
                    color=LANGUAGE_COLORS[language],
                    label=language.upper() if row_idx == 0 and col_idx == 0 else None,
                )
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
            if col_idx == 0:
                ax.set_ylabel("Token F1")
            if row_idx == len(metrics) - 1:
                ax.set_xlabel(metric_label)
    handles = [
        Patch(facecolor=LANGUAGE_COLORS["en"], label="English"),
        Patch(facecolor=LANGUAGE_COLORS["da"], label="Danish"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Per-Example Routing Metrics vs Token F1", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correctness_deltas(model_names: list[str], summary_by_model: dict[str, list[dict]], output_path: Path) -> None:
    metrics = [
        ("mean_entropy", "Entropy"),
        ("mean_margin", "Margin"),
        ("mean_selected_mass", "Selected mass"),
        ("public_minus_danish", "Public-Danish"),
        ("top1_public_share", "Top1 public"),
        ("top2_danish_share", "Top2 Danish"),
        ("token_f1", "Token F1"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    x_positions = np.arange(len(model_names))
    width = 0.34

    for metric_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[metric_idx]
        en_values = []
        da_values = []
        for model_name in model_names:
            rows = summary_by_model[model_name]
            by_key = {(row["language"], row["outcome"]): row for row in rows}
            for language, bucket in (("en", en_values), ("da", da_values)):
                correct = float(by_key.get((language, "correct"), {}).get(metric_key, math.nan))
                incorrect = float(by_key.get((language, "incorrect"), {}).get(metric_key, math.nan))
                bucket.append(correct - incorrect if not (math.isnan(correct) or math.isnan(incorrect)) else math.nan)

        ax.bar(x_positions - width / 2, en_values, width=width, color=LANGUAGE_COLORS["en"], alpha=0.85)
        ax.bar(x_positions + width / 2, da_values, width=width, color=LANGUAGE_COLORS["da"], alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
        ax.set_title(metric_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([name.replace("FlexOlmo-8x7B-1T-", "") for name in model_names], rotation=40, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

    handles = [
        Patch(facecolor=LANGUAGE_COLORS["en"], label="English"),
        Patch(facecolor=LANGUAGE_COLORS["da"], label="Danish"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Correct - Incorrect Metric Differences", y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correctness_summary_heatmap(model_names: list[str], summary_by_model: dict[str, list[dict]], output_path: Path) -> None:
    metric_specs = [
        ("mean_entropy", "Entropy"),
        ("mean_margin", "Margin"),
        ("mean_selected_mass", "Selected mass"),
        ("public_minus_danish", "Public-Danish"),
        ("top1_public_share", "Top1 public"),
        ("top2_danish_share", "Top2 Danish"),
        ("token_f1", "Token F1"),
    ]
    row_labels = []
    matrix = []
    for model_name in model_names:
        rows = summary_by_model[model_name]
        by_key = {(row["language"], row["outcome"]): row for row in rows}
        for language in ("en", "da"):
            row_labels.append(f"{model_name.replace('FlexOlmo-8x7B-1T-', '')}\n{language.upper()}")
            row_values = []
            for metric_key, _ in metric_specs:
                correct = float(by_key.get((language, "correct"), {}).get(metric_key, math.nan))
                incorrect = float(by_key.get((language, "incorrect"), {}).get(metric_key, math.nan))
                row_values.append(correct - incorrect if not (math.isnan(correct) or math.isnan(incorrect)) else math.nan)
            matrix.append(row_values)

    matrix_array = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.7 * len(metric_specs) + 1.5, 0.7 * len(row_labels) + 2.0))
    sns.heatmap(
        matrix_array,
        cmap="coolwarm",
        center=0.0,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Correct - Incorrect"},
        ax=ax,
    )
    ax.set_xticks(np.arange(len(metric_specs)) + 0.5)
    ax.set_xticklabels([label for _, label in metric_specs], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    ax.set_yticklabels(row_labels, rotation=0)
    ax.set_title("Correct - Incorrect Routing Summary Heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(model_names: list[str], output_path: Path, expert_labels: dict[int, str]) -> None:
    danish_label = expert_labels.get(7, "expert_7")
    lines = [
        "# Correctness-Conditioned Routing",
        "",
        "- These summaries use only saved routing and accuracy records, so they can be generated immediately without new model runs.",
        "- The aim is to connect router confidence and expert competition to actual task performance.",
        "",
        "## What This Can Tell Us Now",
        "",
        "- Whether correct answers coincide with lower entropy, larger top1-top2 margin, or higher selected probability mass.",
        f"- Whether success correlates with a smaller or larger `public - {danish_label}` probability gap.",
        f"- Whether the assumed `{danish_label}` expert appears more often as a top-2 competitor on successful examples.",
        "- The difference-style views are often more readable than raw correct-vs-incorrect bars because many metrics are numerically very close.",
        "",
        "## What This Cannot Prove",
        "",
        "- These record-based analyses do not directly test the expert-weight `column addition` or static indistinguishability hypothesis.",
        "- They do provide functional proxies: if experts were effectively indistinguishable to the router, we would expect weak domain-conditional differences in top-1/top-2 competition and weak correctness-conditioned routing structure.",
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

    example_rows_by_model: dict[str, list[dict]] = {}
    summary_by_model: dict[str, list[dict]] = {}
    combined_example_rows = []
    combined_summary_rows = []

    for model_name in DEFAULT_MODELS:
        records = load_joined_records(model_name, a4_root, a8_root)
        example_rows = []
        for record in records:
            row = per_example_metrics(record)
            row["model_name"] = model_name
            row["outcome"] = outcome_label(row)
            example_rows.append(row)
        summary_rows = summarize_rows(example_rows, "outcome")
        for row in summary_rows:
            row["model_name"] = model_name
        example_rows_by_model[model_name] = example_rows
        summary_by_model[model_name] = summary_rows
        combined_example_rows.extend(example_rows)
        combined_summary_rows.extend(summary_rows)

    example_fieldnames = [
        "model_name",
        "example_id",
        "language",
        "outcome",
        "relaxed_match",
        "is_scorable",
        "token_f1",
        "mean_entropy",
        "mean_margin",
        "mean_selected_mass",
        "mean_public_prob",
        "mean_danish_prob",
        "public_minus_danish",
        "top1_public_share",
        "top2_danish_share",
    ]
    summary_fieldnames = [
        "model_name",
        "language",
        "outcome",
        "num_examples",
        "mean_entropy",
        "mean_margin",
        "mean_selected_mass",
        "mean_public_prob",
        "mean_danish_prob",
        "public_minus_danish",
        "top1_public_share",
        "top2_danish_share",
        "token_f1",
    ]
    write_csv(output_root / "correctness_conditioned_examples.csv", combined_example_rows, example_fieldnames)
    write_csv(output_root / "correctness_conditioned_summary.csv", combined_summary_rows, summary_fieldnames)

    plot_correctness_bars(DEFAULT_MODELS, summary_by_model, output_root / "correct_vs_incorrect_routing_metrics.png")
    plot_f1_relationships(DEFAULT_MODELS, example_rows_by_model, output_root / "routing_metrics_vs_token_f1.png")
    plot_correctness_deltas(DEFAULT_MODELS, summary_by_model, output_root / "correct_minus_incorrect_deltas.png")
    plot_correctness_summary_heatmap(DEFAULT_MODELS, summary_by_model, output_root / "correct_minus_incorrect_heatmap.png")
    write_readme(DEFAULT_MODELS, output_root / "README.md", expert_labels)
    print(f"Wrote correctness-conditioned routing outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
