from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
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
DEFAULT_MODELS = [
    "FlexOlmo-8x7B-1T-a4-5B",
    "FlexOlmo-8x7B-1T-a4-25B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2-rt",
]
DEFAULT_LAYER_LABELS = [
    ("early", 3),
    ("mid", 15),
    ("late", 23),
    ("last", 31),
]
LANGUAGE_COLORS = {
    "en": "#4c78a8",
    "da": "#f58518",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dominant top-1/top-2 A4 routing bars across selected layers."
    )
    parser.add_argument("--results-root", default="eval_results/mkqa/full/flexolmo/a4")
    parser.add_argument(
        "--output-root",
        help="Directory where the figure will be written. Defaults to <results-root>/routing/plots.",
    )
    parser.add_argument("--model-name", action="append", default=[])
    parser.add_argument("--layer", action="append", type=int, default=[])
    parser.add_argument("--public-expert-idx", type=int, default=0)
    parser.add_argument("--expert-label", action="append", default=[])
    return parser.parse_args()


def parse_expert_labels(raw_labels: list[str], public_expert_idx: int) -> dict[int, str]:
    labels = dict(DEFAULT_ASSUMED_EXPERT_LABELS)
    labels[public_expert_idx] = "public"
    for raw_item in raw_labels:
        if "=" not in raw_item:
            raise ValueError(f"Invalid --expert-label value `{raw_item}`. Expected <idx>=<label>.")
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


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_available_models(results_root: Path) -> list[str]:
    routing_root = results_root / "routing"
    if not routing_root.exists():
        return []
    model_names = []
    for model_dir in routing_root.iterdir():
        if not model_dir.is_dir():
            continue
        records_path = model_dir / "mkqa_en_da" / "native_full" / "routing_records.jsonl"
        if records_path.exists():
            model_names.append(model_dir.name)
    return sorted(model_names, key=model_sort_key)


def selected_models(results_root: Path, requested: list[str]) -> list[str]:
    available = discover_available_models(results_root)
    if requested:
        return [name for name in requested if name in available]
    return [name for name in DEFAULT_MODELS if name in available]


def flatten_ids(value) -> list[int]:
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(flatten_ids(item))
        return flattened
    return [int(value)]


def load_records(results_root: Path, model_name: str) -> list[dict]:
    records_path = (
        results_root / "routing" / model_name / "mkqa_en_da" / "native_full" / "routing_records.jsonl"
    )
    records = load_jsonl(records_path)
    if not records:
        raise ValueError(f"No routing records found for `{model_name}`.")
    return records


def resolve_layer_specs(num_layers: int, requested_layers: list[int]) -> list[tuple[str, int]]:
    if requested_layers:
        return [(f"layer {idx}", idx) for idx in requested_layers if 0 <= idx < num_layers]
    resolved = []
    for label, candidate in DEFAULT_LAYER_LABELS:
        resolved.append((label, min(candidate, num_layers - 1)))
    deduped = []
    seen = set()
    for label, idx in resolved:
        key = (label, idx)
        if key not in seen:
            deduped.append((label, idx))
            seen.add(key)
    return deduped


def dominant_rank_bars(records: list[dict], layer_idx: int, rank_field: str, num_experts: int) -> dict[str, np.ndarray]:
    result = {}
    for language in ("en", "da"):
        counts = np.zeros(num_experts, dtype=float)
        total = 0.0
        for record in records:
            if record["language"] != language:
                continue
            ids = flatten_ids(record["prompt_router_token_summaries_by_layer"][layer_idx][rank_field])
            for expert_idx in ids:
                counts[expert_idx] += 1.0
            total += len(ids)
        shares = counts / total if total else counts
        dominant = np.zeros_like(shares)
        if shares.size and shares.max() > 0:
            dominant[int(np.argmax(shares))] = shares.max()
        result[language] = dominant
    return result


def plot_dominant_bars(
    model_names: list[str],
    model_records: dict[str, list[dict]],
    layer_specs: list[tuple[str, int]],
    expert_labels: dict[int, str],
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    num_rows = len(layer_specs)
    num_cols = len(model_names)
    num_experts = len(model_records[model_names[0]][0]["prompt_router_probs_by_layer"][0][0])
    expert_axis = np.arange(num_experts)
    width = 0.18

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4.7 * num_cols, 2.8 * num_rows + 0.8),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    ymax = 0.0
    precomputed = {}
    for model_name in model_names:
        precomputed[model_name] = {}
        for layer_label, layer_idx in layer_specs:
            top1 = dominant_rank_bars(model_records[model_name], layer_idx, "top1_expert_ids", num_experts)
            top2 = dominant_rank_bars(model_records[model_name], layer_idx, "top2_expert_ids", num_experts)
            precomputed[model_name][layer_idx] = {"top1": top1, "top2": top2}
            ymax = max(
                ymax,
                float(top1["en"].max()),
                float(top1["da"].max()),
                float(top2["en"].max()),
                float(top2["da"].max()),
            )

    for row_idx, (layer_label, layer_idx) in enumerate(layer_specs):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            top1 = precomputed[model_name][layer_idx]["top1"]
            top2 = precomputed[model_name][layer_idx]["top2"]

            ax.bar(
                expert_axis - 1.5 * width,
                top1["en"],
                width=width,
                color=LANGUAGE_COLORS["en"],
                alpha=0.95,
                label="English top-1" if row_idx == 0 and col_idx == 0 else None,
            )
            ax.bar(
                expert_axis - 0.5 * width,
                top2["en"],
                width=width,
                color=LANGUAGE_COLORS["en"],
                alpha=0.55,
                hatch="///",
                edgecolor=LANGUAGE_COLORS["en"],
                label="English top-2" if row_idx == 0 and col_idx == 0 else None,
            )
            ax.bar(
                expert_axis + 0.5 * width,
                top1["da"],
                width=width,
                color=LANGUAGE_COLORS["da"],
                alpha=0.95,
                label="Danish top-1" if row_idx == 0 and col_idx == 0 else None,
            )
            ax.bar(
                expert_axis + 1.5 * width,
                top2["da"],
                width=width,
                color=LANGUAGE_COLORS["da"],
                alpha=0.55,
                hatch="///",
                edgecolor=LANGUAGE_COLORS["da"],
                label="Danish top-2" if row_idx == 0 and col_idx == 0 else None,
            )

            ax.set_ylim(0.0, max(0.35, ymax * 1.12))
            ax.set_xticks(expert_axis)
            ax.set_xticklabels([expert_label(idx, expert_labels) for idx in expert_axis], rotation=40, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
            if col_idx == 0:
                ax.set_ylabel(f"{layer_label}\nshare")
            else:
                ax.set_ylabel("")

    legend_handles = [
        Patch(facecolor=LANGUAGE_COLORS["en"], edgecolor="none", label="English top-1"),
        Patch(facecolor=LANGUAGE_COLORS["en"], edgecolor=LANGUAGE_COLORS["en"], hatch="///", alpha=0.55, label="English top-2"),
        Patch(facecolor=LANGUAGE_COLORS["da"], edgecolor="none", label="Danish top-1"),
        Patch(facecolor=LANGUAGE_COLORS["da"], edgecolor=LANGUAGE_COLORS["da"], hatch="///", alpha=0.55, label="Danish top-2"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("A4 Dominant Top-1 and Top-2 Experts Across Selected Layers", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(model_names: list[str], output_path: Path) -> None:
    lines = [
        "# A4 Dominant Top-1/Top-2 Bars",
        "",
        "- Each panel uses standard vertical bars over expert IDs.",
        "- For each language and rank (`top-1`, `top-2`), only the dominant expert for that layer is shown with a non-zero bar.",
        "- Solid bars indicate `top-1`; hatched bars indicate `top-2`.",
        "- The current working expert mapping is `0=public`, `1=code`, `2=creative`, `3=math`, `4=news`, `5=pes2o`, `6=reddit`, `7=danish`.",
        "- The `7=danish` assignment remains provisional pending confirmation from the original merge or training command.",
        "",
        "## Models",
        "",
    ]
    for model_name in model_names:
        lines.append(f"- `{model_name}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root) if args.output_root else results_root / "routing" / "plots"
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = selected_models(results_root, args.model_name)
    if not model_names:
        raise ValueError("No A4 models were found for plotting.")

    model_records = {model_name: load_records(results_root, model_name) for model_name in model_names}
    num_layers = len(model_records[model_names[0]][0]["prompt_router_token_summaries_by_layer"])
    layer_specs = resolve_layer_specs(num_layers, args.layer)
    expert_labels = parse_expert_labels(args.expert_label, args.public_expert_idx)

    output_path = output_root / "a4_dominant_top1_top2_selected_layers.png"
    plot_dominant_bars(model_names, model_records, layer_specs, expert_labels, output_path)
    write_readme(model_names, output_root / "README_a4_dominant_top1_top2.md")
    print(f"Wrote dominant top-1/top-2 A4 plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
