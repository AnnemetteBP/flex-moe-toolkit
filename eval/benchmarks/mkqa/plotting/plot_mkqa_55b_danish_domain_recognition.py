from __future__ import annotations

import argparse
import csv
import json
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
LANGUAGE_COLORS = {
    "en": "#4c78a8",
    "da": "#f58518",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Danish-vs-English domain-recognition summaries from saved MKQA routing records."
    )
    parser.add_argument("--a4-root", default="eval_results/mkqa/full/flexolmo/a4")
    parser.add_argument("--a8-root", default="eval_results/mkqa/full/flexolmo/a8")
    parser.add_argument(
        "--output-root",
        default="eval_results/mkqa/full/flexolmo/comparisons/a4_a8_55b_domain_recognition",
    )
    parser.add_argument("--public-expert-idx", type=int, default=0)
    parser.add_argument("--danish-expert-idx", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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


def load_records(model_name: str, a4_root: Path, a8_root: Path) -> list[dict]:
    family_root = model_root_for_name(model_name, a4_root, a8_root)
    path = family_root / "routing" / model_name / "mkqa_en_da" / "native_full" / "routing_records.jsonl"
    return load_jsonl(path)


def per_example_summary(record: dict, public_idx: int, danish_idx: int) -> dict[str, float | str]:
    num_layers = len(record["prompt_router_token_summaries_by_layer"])
    public_probs = []
    danish_probs = []
    public_top1 = 0.0
    danish_top1 = 0.0
    danish_top2 = 0.0
    total_tokens = 0.0
    danish_rank_sum = 0.0
    danish_rank_count = 0.0
    for layer_idx in range(num_layers):
        probs = np.asarray(record["prompt_router_probs_by_layer"][layer_idx], dtype=float)
        top1_ids = flatten_ids(record["prompt_router_token_summaries_by_layer"][layer_idx]["top1_expert_ids"])
        top2_ids = flatten_ids(record["prompt_router_token_summaries_by_layer"][layer_idx]["top2_expert_ids"])
        total_tokens += len(top1_ids)
        public_top1 += sum(1.0 for idx in top1_ids if idx == public_idx)
        danish_top1 += sum(1.0 for idx in top1_ids if idx == danish_idx)
        danish_top2 += sum(1.0 for idx in top2_ids if idx == danish_idx)
        public_probs.extend(probs[:, public_idx].tolist())
        danish_probs.extend(probs[:, danish_idx].tolist())
        descending = np.argsort(-probs, axis=1)
        for row in descending:
            positions = np.where(row == danish_idx)[0]
            if positions.size:
                danish_rank_sum += float(positions[0] + 1)
                danish_rank_count += 1.0
    return {
        "language": record["language"],
        "mean_public_prob": float(np.mean(public_probs)),
        "mean_danish_prob": float(np.mean(danish_probs)),
        "public_top1_share": float(public_top1 / total_tokens) if total_tokens else 0.0,
        "danish_top1_share": float(danish_top1 / total_tokens) if total_tokens else 0.0,
        "danish_top2_share": float(danish_top2 / total_tokens) if total_tokens else 0.0,
        "mean_danish_rank": float(danish_rank_sum / danish_rank_count) if danish_rank_count else float("nan"),
    }


def language_metrics_from_summaries(example_summaries: list[dict]) -> dict[str, dict[str, float]]:
    metrics = {}
    for language in ("en", "da"):
        subset = [row for row in example_summaries if row["language"] == language]
        mean_public_prob = np.mean([row["mean_public_prob"] for row in subset])
        mean_danish_prob = np.mean([row["mean_danish_prob"] for row in subset])
        metrics[language] = {
            "mean_public_prob": float(mean_public_prob),
            "mean_danish_prob": float(mean_danish_prob),
            "public_minus_danish": float(mean_public_prob - mean_danish_prob),
            "public_top1_share": float(np.mean([row["public_top1_share"] for row in subset])),
            "danish_top1_share": float(np.mean([row["danish_top1_share"] for row in subset])),
            "danish_top2_share": float(np.mean([row["danish_top2_share"] for row in subset])),
            "mean_danish_rank": float(np.mean([row["mean_danish_rank"] for row in subset])),
        }
    return metrics


def language_metrics(records: list[dict], public_idx: int, danish_idx: int) -> dict[str, dict[str, float]]:
    example_summaries = [
        per_example_summary(record, public_idx=public_idx, danish_idx=danish_idx)
        for record in records
    ]
    return language_metrics_from_summaries(example_summaries)


def bootstrap_language_metrics(
    records: list[dict],
    public_idx: int,
    danish_idx: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    example_summaries = [
        per_example_summary(record, public_idx=public_idx, danish_idx=danish_idx)
        for record in records
    ]
    by_language = {language: [row for row in example_summaries if row["language"] == language] for language in ("en", "da")}
    point = language_metrics_from_summaries(example_summaries)
    metric_names = list(next(iter(point.values())).keys())

    sampled = {
        "en": {metric: [] for metric in metric_names},
        "da": {metric: [] for metric in metric_names},
        "delta": {metric: [] for metric in metric_names},
    }
    for _ in range(bootstrap_samples):
        resampled_records = []
        for language in ("en", "da"):
            group = by_language[language]
            indices = rng.integers(0, len(group), size=len(group))
            resampled_records.extend(group[idx] for idx in indices)
        sample_metrics = language_metrics_from_summaries(resampled_records)
        for language in ("en", "da"):
            for metric in metric_names:
                sampled[language][metric].append(sample_metrics[language][metric])
        for metric in metric_names:
            sampled["delta"][metric].append(sample_metrics["da"][metric] - sample_metrics["en"][metric])

    ci = {"en": {}, "da": {}, "delta": {}}
    for bucket in ("en", "da", "delta"):
        for metric in metric_names:
            values = np.asarray(sampled[bucket][metric], dtype=float)
            ci[bucket][metric] = {
                "mean": float(np.mean(values)),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
            }
    return ci


def plot_language_bars(model_names: list[str], metrics_by_model: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    metric_specs = [
        ("mean_danish_prob", "Mean Danish prob"),
        ("danish_top2_share", "Danish top-2 share"),
        ("public_minus_danish", "Public - Danish prob"),
        ("mean_danish_rank", "Mean Danish rank"),
    ]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(4.1 * len(metric_specs), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    x_positions = np.arange(len(model_names))
    width = 0.34
    for metric_idx, (metric_key, metric_label) in enumerate(metric_specs):
        ax = axes[metric_idx]
        en_values = [metrics_by_model[model_name]["en"][metric_key] for model_name in model_names]
        da_values = [metrics_by_model[model_name]["da"][metric_key] for model_name in model_names]
        ax.bar(x_positions - width / 2, en_values, width=width, color=LANGUAGE_COLORS["en"], alpha=0.85)
        ax.bar(x_positions + width / 2, da_values, width=width, color=LANGUAGE_COLORS["da"], alpha=0.85)
        ax.set_title(metric_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([name.replace("FlexOlmo-8x7B-1T-", "") for name in model_names], rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        if metric_key == "mean_danish_rank":
            ax.invert_yaxis()
    handles = [
        Patch(facecolor=LANGUAGE_COLORS["en"], label="English"),
        Patch(facecolor=LANGUAGE_COLORS["da"], label="Danish"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Danish Expert Domain Recognition Summary", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_language_deltas(model_names: list[str], metrics_by_model: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    metric_specs = [
        ("mean_danish_prob", "Danish prob"),
        ("danish_top2_share", "Danish top-2"),
        ("public_minus_danish", "Public-Danish"),
        ("mean_danish_rank", "Danish rank"),
    ]
    fig, ax = plt.subplots(figsize=(1.8 * len(metric_specs) + 2.0, 4.5))
    width = 0.18
    x_positions = np.arange(len(metric_specs))
    for model_idx, model_name in enumerate(model_names):
        deltas = []
        for metric_key, _ in metric_specs:
            da_value = metrics_by_model[model_name]["da"][metric_key]
            en_value = metrics_by_model[model_name]["en"][metric_key]
            deltas.append(da_value - en_value)
        ax.bar(
            x_positions + (model_idx - (len(model_names) - 1) / 2) * width,
            deltas,
            width=width,
            alpha=0.85,
            label=model_name.replace("FlexOlmo-8x7B-1T-", ""),
        )
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in metric_specs], rotation=25, ha="right")
    ax.set_ylabel("Danish - English")
    ax.set_title("How Much More Danish-Like Is the Routing Signal?")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_language_bars_with_ci(
    model_names: list[str],
    metrics_by_model: dict[str, dict[str, dict[str, float]]],
    ci_by_model: dict[str, dict[str, dict[str, dict[str, float]]]],
    output_path: Path,
) -> None:
    metric_specs = [
        ("mean_danish_prob", "Mean Danish prob"),
        ("danish_top2_share", "Danish top-2 share"),
        ("public_minus_danish", "Public - Danish prob"),
        ("mean_danish_rank", "Mean Danish rank"),
    ]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(4.1 * len(metric_specs), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    x_positions = np.arange(len(model_names))
    width = 0.34
    for metric_idx, (metric_key, metric_label) in enumerate(metric_specs):
        ax = axes[metric_idx]
        en_values = [metrics_by_model[model_name]["en"][metric_key] for model_name in model_names]
        da_values = [metrics_by_model[model_name]["da"][metric_key] for model_name in model_names]
        en_err = np.array(
            [
                [
                    metrics_by_model[model_name]["en"][metric_key] - ci_by_model[model_name]["en"][metric_key]["ci_low"],
                    ci_by_model[model_name]["en"][metric_key]["ci_high"] - metrics_by_model[model_name]["en"][metric_key],
                ]
                for model_name in model_names
            ]
        ).T
        da_err = np.array(
            [
                [
                    metrics_by_model[model_name]["da"][metric_key] - ci_by_model[model_name]["da"][metric_key]["ci_low"],
                    ci_by_model[model_name]["da"][metric_key]["ci_high"] - metrics_by_model[model_name]["da"][metric_key],
                ]
                for model_name in model_names
            ]
        ).T
        ax.bar(
            x_positions - width / 2,
            en_values,
            yerr=en_err,
            capsize=3,
            width=width,
            color=LANGUAGE_COLORS["en"],
            alpha=0.85,
        )
        ax.bar(
            x_positions + width / 2,
            da_values,
            yerr=da_err,
            capsize=3,
            width=width,
            color=LANGUAGE_COLORS["da"],
            alpha=0.85,
        )
        ax.set_title(metric_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([name.replace("FlexOlmo-8x7B-1T-", "") for name in model_names], rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        if metric_key == "mean_danish_rank":
            ax.invert_yaxis()
    handles = [
        Patch(facecolor=LANGUAGE_COLORS["en"], label="English"),
        Patch(facecolor=LANGUAGE_COLORS["da"], label="Danish"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Danish Expert Domain Recognition with 95% Bootstrap CIs", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_language_deltas_with_ci(
    model_names: list[str],
    metrics_by_model: dict[str, dict[str, dict[str, float]]],
    ci_by_model: dict[str, dict[str, dict[str, dict[str, float]]]],
    output_path: Path,
) -> None:
    metric_specs = [
        ("mean_danish_prob", "Danish prob"),
        ("danish_top2_share", "Danish top-2"),
        ("public_minus_danish", "Public-Danish"),
        ("mean_danish_rank", "Danish rank"),
    ]
    fig, ax = plt.subplots(figsize=(1.9 * len(metric_specs) + 2.5, 4.8))
    width = 0.18
    x_positions = np.arange(len(metric_specs))
    for model_idx, model_name in enumerate(model_names):
        deltas = []
        lower = []
        upper = []
        for metric_key, _ in metric_specs:
            point = metrics_by_model[model_name]["da"][metric_key] - metrics_by_model[model_name]["en"][metric_key]
            delta_ci = ci_by_model[model_name]["delta"][metric_key]
            deltas.append(point)
            lower.append(point - delta_ci["ci_low"])
            upper.append(delta_ci["ci_high"] - point)
        yerr = np.vstack([lower, upper])
        ax.bar(
            x_positions + (model_idx - (len(model_names) - 1) / 2) * width,
            deltas,
            yerr=yerr,
            capsize=3,
            width=width,
            alpha=0.85,
            label=model_name.replace("FlexOlmo-8x7B-1T-", ""),
        )
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in metric_specs], rotation=25, ha="right")
    ax.set_ylabel("Danish - English")
    ax.set_title("How Much More Danish-Like Is the Routing Signal? (95% Bootstrap CIs)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    path: Path,
    model_names: list[str],
    metrics_by_model: dict[str, dict[str, dict[str, float]]],
    ci_by_model: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> None:

    fieldnames = [
        "model_name",
        "language",
        "mean_public_prob",
        "mean_public_prob_ci_low",
        "mean_public_prob_ci_high",
        "mean_danish_prob",
        "mean_danish_prob_ci_low",
        "mean_danish_prob_ci_high",
        "public_minus_danish",
        "public_minus_danish_ci_low",
        "public_minus_danish_ci_high",
        "public_top1_share",
        "public_top1_share_ci_low",
        "public_top1_share_ci_high",
        "danish_top1_share",
        "danish_top1_share_ci_low",
        "danish_top1_share_ci_high",
        "danish_top2_share",
        "danish_top2_share_ci_low",
        "danish_top2_share_ci_high",
        "mean_danish_rank",
        "mean_danish_rank_ci_low",
        "mean_danish_rank_ci_high",
        "delta_mean_public_prob",
        "delta_mean_public_prob_ci_low",
        "delta_mean_public_prob_ci_high",
        "delta_mean_danish_prob",
        "delta_mean_danish_prob_ci_low",
        "delta_mean_danish_prob_ci_high",
        "delta_public_minus_danish",
        "delta_public_minus_danish_ci_low",
        "delta_public_minus_danish_ci_high",
        "delta_public_top1_share",
        "delta_public_top1_share_ci_low",
        "delta_public_top1_share_ci_high",
        "delta_danish_top1_share",
        "delta_danish_top1_share_ci_low",
        "delta_danish_top1_share_ci_high",
        "delta_danish_top2_share",
        "delta_danish_top2_share_ci_low",
        "delta_danish_top2_share_ci_high",
        "delta_mean_danish_rank",
        "delta_mean_danish_rank_ci_low",
        "delta_mean_danish_rank_ci_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in model_names:
            for language in ("en", "da"):
                row = {"model_name": model_name, "language": language}
                for metric_key, point in metrics_by_model[model_name][language].items():
                    row[metric_key] = point
                    row[f"{metric_key}_ci_low"] = ci_by_model[model_name][language][metric_key]["ci_low"]
                    row[f"{metric_key}_ci_high"] = ci_by_model[model_name][language][metric_key]["ci_high"]
                for metric_key in metrics_by_model[model_name]["en"].keys():
                    delta_point = metrics_by_model[model_name]["da"][metric_key] - metrics_by_model[model_name]["en"][metric_key]
                    row[f"delta_{metric_key}"] = delta_point
                    row[f"delta_{metric_key}_ci_low"] = ci_by_model[model_name]["delta"][metric_key]["ci_low"]
                    row[f"delta_{metric_key}_ci_high"] = ci_by_model[model_name]["delta"][metric_key]["ci_high"]
                writer.writerow(row)


def write_readme(output_path: Path) -> None:
    lines = [
        "# Danish Domain Recognition",
        "",
        "- This analysis asks a narrow and useful question: does the router give the assumed Danish expert a stronger signal on Danish than on English?",
        "- It does not prove expert distinctness by itself, but it does test whether the router assigns domain-informative preference to the expected specialist.",
        "",
        "## Metrics",
        "",
        "- `mean_danish_prob`: mean routing probability assigned to the assumed Danish expert.",
        "- `danish_top2_share`: fraction of token-layer decisions where the assumed Danish expert is in rank 2.",
        "- `public_minus_danish`: how much larger the public-expert probability is than the assumed Danish-expert probability.",
        "- `mean_danish_rank`: average rank position of the assumed Danish expert among all experts. Lower is stronger.",
        "- The CSV includes 95% bootstrap confidence intervals for both per-language estimates and Danish-minus-English deltas.",
        "",
        "## Interpretation",
        "",
        "- If Danish is recognized as a distinct domain by the router, we would expect higher `mean_danish_prob`, higher `danish_top2_share`, lower `public_minus_danish`, and lower `mean_danish_rank` on Danish than on English.",
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    a4_root = Path(args.a4_root)
    a8_root = Path(args.a8_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_by_model = {}
    ci_by_model = {}
    for model_name in DEFAULT_MODELS:
        records = load_records(model_name, a4_root, a8_root)
        metrics_by_model[model_name] = language_metrics(
            records,
            public_idx=args.public_expert_idx,
            danish_idx=args.danish_expert_idx,
        )
        ci_by_model[model_name] = bootstrap_language_metrics(
            records,
            public_idx=args.public_expert_idx,
            danish_idx=args.danish_expert_idx,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )

    write_summary_csv(
        output_root / "danish_domain_recognition_summary.csv",
        DEFAULT_MODELS,
        metrics_by_model,
        ci_by_model,
    )
    plot_language_bars(DEFAULT_MODELS, metrics_by_model, output_root / "danish_domain_recognition_bars.png")
    plot_language_deltas(DEFAULT_MODELS, metrics_by_model, output_root / "danish_domain_recognition_deltas.png")
    plot_language_bars_with_ci(
        DEFAULT_MODELS,
        metrics_by_model,
        ci_by_model,
        output_root / "danish_domain_recognition_bars_with_ci.png",
    )
    plot_language_deltas_with_ci(
        DEFAULT_MODELS,
        metrics_by_model,
        ci_by_model,
        output_root / "danish_domain_recognition_deltas_with_ci.png",
    )
    write_readme(output_root / "README.md")
    print(f"Wrote Danish domain-recognition outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
