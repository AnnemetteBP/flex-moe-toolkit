from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import re
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


MODEL_PATTERN = re.compile(r"^FlexOlmo-8x7B-1T-a4-(?P<size>\d+)B(?:-(?P<variant>.+))?$")
VARIANT_ORDER = {
    "": 0,
    "v1": 1,
    "v2": 2,
    "float32": 3,
    "rt": 4,
    "rt4": 5,
}
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
DEFAULT_FOCUS_MODELS = [
    "FlexOlmo-8x7B-1T-a4-5B",
    "FlexOlmo-8x7B-1T-a4-25B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2-rt",
]
LANGUAGE_COLORS = {
    "en": "#4c78a8",
    "da": "#f58518",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate interpretation-focused routing diagnostics from saved A4 MKQA routing records."
    )
    parser.add_argument(
        "--results-root",
        default="eval_results/mkqa/full/flexolmo/a4",
        help="MKQA combined-native results root containing routing outputs.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where diagnostics outputs will be written. Defaults to <results-root>/routing_diagnostics.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional explicit model names to include. Defaults to all complete native-full A4 routing runs.",
    )
    parser.add_argument(
        "--focus-model",
        action="append",
        default=[],
        help="Optional explicit model names for the bar/combination panels. Defaults to representative A4 checkpoints.",
    )
    parser.add_argument(
        "--public-expert-idx",
        type=int,
        default=0,
        help="Expert index treated as public. Defaults to 0.",
    )
    parser.add_argument(
        "--expert-label",
        action="append",
        default=[],
        help="Optional explicit expert label mapping in the form `<idx>=<label>`.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_expert_labels(raw_labels: list[str], public_expert_idx: int) -> dict[int, str]:
    labels = dict(DEFAULT_ASSUMED_EXPERT_LABELS)
    labels[public_expert_idx] = "public"
    for raw_item in raw_labels:
        if "=" not in raw_item:
            raise ValueError(f"Invalid `--expert-label` value `{raw_item}`. Expected `<idx>=<label>`.")
        idx_str, label = raw_item.split("=", 1)
        labels[int(idx_str.strip())] = label.strip()
    return labels


def expert_label(expert_idx: int, labels: dict[int, str]) -> str:
    return labels.get(expert_idx, f"expert_{expert_idx}")


def model_sort_key(model_name: str) -> tuple[int, tuple[int, str]]:
    match = MODEL_PATTERN.match(model_name)
    if not match:
        return (999, (999, model_name))
    size_b = int(match.group("size"))
    variant = match.group("variant") or ""
    return (size_b, (VARIANT_ORDER.get(variant, 99), variant))


def iter_complete_models(results_root: Path) -> list[str]:
    routing_root = results_root / "routing"
    if not routing_root.exists():
        return []
    model_names = []
    for model_dir in routing_root.iterdir():
        native_root = model_dir / "mkqa_en_da" / "native_full"
        if not model_dir.is_dir():
            continue
        if all(
            (native_root / filename).exists()
            for filename in ("routing_records.jsonl", "routing_summary.jsonl", "routing_analysis.jsonl")
        ):
            model_names.append(model_dir.name)
    return sorted(model_names, key=model_sort_key)


def native_root(results_root: Path, model_name: str) -> Path:
    return results_root / "routing" / model_name / "mkqa_en_da" / "native_full"


def flatten_token_expert_ids(value) -> list[int]:
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(flatten_token_expert_ids(item))
        return flattened
    return [int(value)]


def per_layer_top1_share(records: list[dict], num_experts: int, target_expert_idx: int) -> list[float]:
    num_layers = len(records[0]["prompt_router_token_summaries_by_layer"])
    shares = []
    for layer_idx in range(num_layers):
        counts = np.zeros(num_experts, dtype=float)
        for record in records:
            top1_ids = record["prompt_router_token_summaries_by_layer"][layer_idx]["top1_expert_ids"]
            for expert_idx in flatten_token_expert_ids(top1_ids):
                counts[expert_idx] += 1.0
        total = float(counts.sum()) or 1.0
        shares.append(float(counts[target_expert_idx] / total))
    return shares


def per_layer_dominant_share(records: list[dict], num_experts: int) -> list[float]:
    num_layers = len(records[0]["prompt_router_token_summaries_by_layer"])
    shares = []
    for layer_idx in range(num_layers):
        counts = np.zeros(num_experts, dtype=float)
        for record in records:
            top1_ids = record["prompt_router_token_summaries_by_layer"][layer_idx]["top1_expert_ids"]
            for expert_idx in flatten_token_expert_ids(top1_ids):
                counts[expert_idx] += 1.0
        total = float(counts.sum()) or 1.0
        shares.append(float(counts.max() / total))
    return shares


def aggregate_top1_usage(records: list[dict], num_experts: int) -> np.ndarray:
    counts = np.zeros(num_experts, dtype=float)
    for record in records:
        for layer_summary in record["prompt_router_token_summaries_by_layer"]:
            for expert_idx in flatten_token_expert_ids(layer_summary["top1_expert_ids"]):
                counts[expert_idx] += 1.0
    total = float(counts.sum()) or 1.0
    return counts / total


def language_specific_public_share(records: list[dict], num_experts: int, public_expert_idx: int) -> dict[str, list[float]]:
    by_language = {}
    languages = sorted({record["language"] for record in records})
    for language in languages:
        language_records = [record for record in records if record["language"] == language]
        by_language[language] = per_layer_top1_share(
            language_records,
            num_experts=num_experts,
            target_expert_idx=public_expert_idx,
        )
    return by_language


def top_token_combinations(records: list[dict], top_n: int = 10) -> list[tuple[tuple[int, ...], int]]:
    counter: Counter[tuple[int, ...]] = Counter()
    for record in records:
        for raw_key, count in (record.get("token_topk_combination_counts") or {}).items():
            combo = tuple(int(part) for part in raw_key.split(",") if part)
            counter[combo] += int(count)
    return counter.most_common(top_n)


def top_token_combinations_by_language(records: list[dict], min_top_n: int = 5, max_top_n: int = 12) -> list[dict]:
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
    ranked = total_counter.most_common()
    for combo, total_count in ranked:
        row = {
            "combo": combo,
            "total": int(total_count),
        }
        for language, counter in language_counters.items():
            row[language] = int(counter.get(combo, 0))
        rows.append(row)
        covered_experts.update(combo)
        if len(rows) >= min_top_n and covered_experts >= observed_experts:
            break
        if len(rows) >= max_top_n:
            break
    return rows


def choose_focus_models(all_models: list[str], explicit_focus: list[str]) -> list[str]:
    if explicit_focus:
        return [name for name in explicit_focus if name in all_models]
    return [name for name in DEFAULT_FOCUS_MODELS if name in all_models]


def plot_usage_bars(focus_models: list[str], model_rows: dict[str, dict], expert_labels: dict[int, str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(focus_models), figsize=(5.2 * len(focus_models), 4.8), sharey=True)
    if len(focus_models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, focus_models):
        usage = model_rows[model_name]["top1_usage"]
        expert_indices = np.arange(len(usage))
        bar_colors = ["#c94f3d" if idx == 0 else "#4c78a8" for idx in expert_indices]
        ax.bar(expert_indices, usage, color=bar_colors)
        ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
        ax.set_xticks(expert_indices)
        ax.set_xticklabels([expert_label(idx, expert_labels) for idx in expert_indices], rotation=45, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Share of Prompt Top-1 Selections")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Aggregate Prompt Top-1 Expert Usage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dominant_share_heatmap(model_names: list[str], model_rows: dict[str, dict], output_path: Path) -> None:
    matrix = np.array([model_rows[name]["layer_dominant_share"] for name in model_names], dtype=float)
    labels = [name.replace("FlexOlmo-8x7B-1T-", "") for name in model_names]
    fig, ax = plt.subplots(figsize=(16, max(6, 0.38 * len(model_names) + 2)))
    sns.heatmap(matrix, cmap="mako", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Dominant Expert Share by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")
    ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_language_public_diff(model_names: list[str], model_rows: dict[str, dict], output_path: Path) -> None:
    diff_rows = []
    for name in model_names:
        per_language = model_rows[name]["public_share_by_language"]
        if "da" in per_language and "en" in per_language:
            diff_rows.append(np.array(per_language["da"]) - np.array(per_language["en"]))
        else:
            diff_rows.append(np.zeros_like(np.array(model_rows[name]["layer_dominant_share"])))
    matrix = np.vstack(diff_rows)
    labels = [name.replace("FlexOlmo-8x7B-1T-", "") for name in model_names]
    fig, ax = plt.subplots(figsize=(16, max(6, 0.38 * len(model_names) + 2)))
    sns.heatmap(matrix, cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0, ax=ax)
    ax.set_title("Public Top-1 Share Difference by Layer (Danish - English)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")
    ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_combo(combo: tuple[int, ...], expert_labels: dict[int, str]) -> str:
    return " + ".join(expert_label(idx, expert_labels) for idx in combo)


def plot_top_combinations_all_models(
    model_names: list[str],
    model_rows: dict[str, dict],
    expert_labels: dict[int, str],
    output_path: Path,
) -> None:
    num_models = len(model_names)
    ncols = 4
    nrows = int(np.ceil(num_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, model_name in zip(axes, model_names):
        combos = model_rows[model_name]["top_combinations_by_language"]
        labels = [format_combo(row["combo"], expert_labels) for row in combos][::-1]
        y_positions = np.arange(len(labels))
        left = np.zeros(len(labels), dtype=float)

        for language in ("en", "da"):
            values = np.array([row.get(language, 0) for row in combos][::-1], dtype=float)
            ax.barh(
                y_positions,
                values,
                left=left,
                color=LANGUAGE_COLORS[language],
                label=language.upper(),
            )
            left += values

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
        ax.set_xlabel("Token Count")
        ax.grid(True, axis="x", alpha=0.25)

    for ax in axes[num_models:]:
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LANGUAGE_COLORS["en"]),
        plt.Rectangle((0, 0), 1, 1, color=LANGUAGE_COLORS["da"]),
    ]
    fig.suptitle("Most Frequent Prompt Expert Combinations by Language", y=0.992)
    fig.legend(
        handles,
        ["English", "Danish"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(model_names: list[str], focus_models: list[str], model_rows: dict[str, dict], expert_labels: dict[int, str], output_path: Path) -> None:
    lines = [
        "# A4 Routing Diagnostics",
        "",
        "- This folder contains interpretation-focused plots derived from saved `routing_records.jsonl` files.",
        (
            "- Expert labels use the current working hypothesis for native A4 checkpoints: "
            "`0=public, 1=code, 2=creative, 3=math, 4=news, 5=pes2o, 6=reddit, 7=danish`."
        ),
        "- Only `0=public` is verified directly in this repo; the remaining labels are provisional.",
        "",
        "## Included Plots",
        "",
        "- `a4_focus_top1_usage_bars.png`: aggregate prompt top-1 expert usage for representative checkpoints.",
        "- `a4_layer_dominance_heatmap.png`: per-layer share captured by the dominant expert.",
        "- `a4_public_share_da_minus_en_heatmap.png`: Danish minus English public top-1 share by layer.",
        "- `a4_all_top_combinations_by_language.png`: most common prompt expert tuples for every analyzed A4 checkpoint, stacked by English vs Danish token counts.",
        "",
        "## Quick Read",
        "",
    ]

    median_public_share = float(np.median([row["top1_usage"][0] for row in model_rows.values()]))
    lines.append(f"- Median aggregate prompt top-1 share routed to `public`: {median_public_share:.3%}.")
    for model_name in focus_models:
        row = model_rows[model_name]
        top_combo_row = row["top_combinations_by_language"][0]
        top_combo = top_combo_row["combo"]
        top_combo_count = top_combo_row["total"]
        en_count = top_combo_row.get("en", 0)
        da_count = top_combo_row.get("da", 0)
        lines.append(
            f"- `{model_name}`: public top-1 share = {row['top1_usage'][0]:.3%}; "
            f"most common prompt combination = `{format_combo(top_combo, expert_labels)}` "
            f"({top_combo_count} tokens: en={en_count}, da={da_count})."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root) if args.output_root else results_root / "routing_diagnostics"
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = args.model_name or iter_complete_models(results_root)
    if not model_names:
        raise SystemExit(f"No complete A4 native-full routing runs found under {results_root}.")
    expert_labels = parse_expert_labels(args.expert_label, public_expert_idx=args.public_expert_idx)
    focus_models = choose_focus_models(model_names, args.focus_model)
    if not focus_models:
        raise SystemExit("No focus models were selected for the diagnostic panels.")

    model_rows = {}
    for model_name in model_names:
        records = load_jsonl(native_root(results_root, model_name) / "routing_records.jsonl")
        num_experts = int(records[0]["num_available_experts"])
        model_rows[model_name] = {
            "top1_usage": aggregate_top1_usage(records, num_experts=num_experts),
            "layer_dominant_share": per_layer_dominant_share(records, num_experts=num_experts),
            "public_share_by_language": language_specific_public_share(
                records,
                num_experts=num_experts,
                public_expert_idx=args.public_expert_idx,
            ),
            "top_combinations": top_token_combinations(records),
            "top_combinations_by_language": top_token_combinations_by_language(records),
        }

    plot_usage_bars(
        focus_models=focus_models,
        model_rows=model_rows,
        expert_labels=expert_labels,
        output_path=output_root / "a4_focus_top1_usage_bars.png",
    )
    plot_dominant_share_heatmap(
        model_names=model_names,
        model_rows=model_rows,
        output_path=output_root / "a4_layer_dominance_heatmap.png",
    )
    plot_language_public_diff(
        model_names=model_names,
        model_rows=model_rows,
        output_path=output_root / "a4_public_share_da_minus_en_heatmap.png",
    )
    plot_top_combinations_all_models(
        model_names=model_names,
        model_rows=model_rows,
        expert_labels=expert_labels,
        output_path=output_root / "a4_all_top_combinations_by_language.png",
    )
    write_readme(
        model_names=model_names,
        focus_models=focus_models,
        model_rows=model_rows,
        expert_labels=expert_labels,
        output_path=output_root / "README.md",
    )

    print(f"Wrote A4 routing diagnostics for {len(model_names)} models to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
