from __future__ import annotations

import argparse
import json
import os
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
        description="Generate paper-style A4 top-2 expert comparison plots from saved MKQA routing records."
    )
    parser.add_argument(
        "--results-root",
        default="eval_results/mkqa/full/flexolmo/a4",
        help="MKQA A4 results root containing routing outputs.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where the figure will be written. Defaults to <results-root>/routing/plots.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional explicit A4 model names to include.",
    )
    parser.add_argument(
        "--layer",
        action="append",
        type=int,
        default=[],
        help="Optional explicit layer indices to plot. If omitted, uses early/mid/late/last defaults.",
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
        help="Optional expert label mapping in the form <idx>=<label>.",
    )
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


def load_rank_matrix(results_root: Path, model_name: str, rank_field: str) -> np.ndarray:
    records_path = (
        results_root / "routing" / model_name / "mkqa_en_da" / "native_full" / "routing_records.jsonl"
    )
    records = load_jsonl(records_path)
    if not records:
        raise ValueError(f"No routing records found for `{model_name}`.")

    num_layers = len(records[0]["prompt_router_token_summaries_by_layer"])
    num_experts = len(records[0]["prompt_router_probs_by_layer"][0][0])
    matrix = np.zeros((num_layers, num_experts, 2), dtype=float)

    for language_idx, language in enumerate(("en", "da")):
        language_records = [record for record in records if record["language"] == language]
        for layer_idx in range(num_layers):
            counts = np.zeros(num_experts, dtype=float)
            total = 0.0
            for record in language_records:
                ids = flatten_ids(
                    record["prompt_router_token_summaries_by_layer"][layer_idx][rank_field]
                )
                for expert_idx in ids:
                    counts[expert_idx] += 1.0
                total += len(ids)
            if total:
                matrix[layer_idx, :, language_idx] = counts / total

    return matrix


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


def plot_rank_grid(
    model_names: list[str],
    top1_matrices: dict[str, np.ndarray],
    top2_matrices: dict[str, np.ndarray],
    layer_specs: list[tuple[str, int]],
    expert_labels: dict[int, str],
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    num_rows = len(layer_specs)
    num_model_cols = len(model_names)
    num_cols = num_model_cols * 2
    num_experts = next(iter(top1_matrices.values())).shape[1]
    expert_axis = np.arange(num_experts)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3.9 * num_cols, 2.9 * num_rows + 0.8),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row_idx, (layer_label, layer_idx) in enumerate(layer_specs):
        for model_idx, model_name in enumerate(model_names):
            for rank_offset, (rank_name, matrices) in enumerate(
                (("top-1", top1_matrices), ("top-2", top2_matrices))
            ):
                col_idx = model_idx * 2 + rank_offset
                ax = axes[row_idx, col_idx]
                matrix = matrices[model_name]
                layer_values = matrix[layer_idx]

                ax.plot(
                    expert_axis,
                    layer_values[:, 0],
                    color=LANGUAGE_COLORS["en"],
                    marker="o",
                    linewidth=1.9,
                    markersize=4.2,
                    label="English",
                )
                ax.plot(
                    expert_axis,
                    layer_values[:, 1],
                    color=LANGUAGE_COLORS["da"],
                    marker="o",
                    linewidth=1.9,
                    markersize=4.2,
                    label="Danish",
                )
                ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.3)
                ax.set_ylim(
                    0.0,
                    max(
                        0.65,
                        float(top1_matrices[model_name].max()) * 1.08,
                        float(top2_matrices[model_name].max()) * 1.08,
                    ),
                )
                ax.set_xticks(expert_axis)
                ax.set_xticklabels(
                    [expert_label(idx, expert_labels) for idx in expert_axis],
                    rotation=40,
                    ha="right",
                )
                ax.grid(True, axis="y", alpha=0.25)

                if row_idx == 0:
                    model_title = model_name.replace("FlexOlmo-8x7B-1T-", "")
                    ax.set_title(f"{model_title}\n{rank_name}")
                if col_idx == 0:
                    ax.set_ylabel(f"{layer_label}\nShare")
                else:
                    ax.set_ylabel("")

    handles = [
        plt.Line2D([0], [0], color=LANGUAGE_COLORS["en"], marker="o", linewidth=1.9, label="English"),
        plt.Line2D([0], [0], color=LANGUAGE_COLORS["da"], marker="o", linewidth=1.9, label="Danish"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("A4 Top-1 vs Top-2 Expert Selection Across Selected Layers", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(model_names: list[str], output_path: Path) -> None:
    lines = [
        "# A4 Top-1 vs Top-2 Paper-Style Comparison",
        "",
        "- This figure shows which experts are selected as the token-level top-1 and top-2 experts across selected layers.",
        "- Each panel plots expert-wise share, with English and Danish shown as separate curves.",
        "- The paired top-1/top-2 layout is intended to clarify cases where an expert is often the runner-up without dominating the full routing combinations.",
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

    expert_labels = parse_expert_labels(args.expert_label, args.public_expert_idx)
    top1_matrices = {
        model_name: load_rank_matrix(results_root, model_name, "top1_expert_ids")
        for model_name in model_names
    }
    top2_matrices = {
        model_name: load_rank_matrix(results_root, model_name, "top2_expert_ids")
        for model_name in model_names
    }
    num_layers = next(iter(top1_matrices.values())).shape[0]
    layer_specs = resolve_layer_specs(num_layers, args.layer)

    output_path = output_root / "a4_top1_top2_selected_layers.png"
    plot_rank_grid(model_names, top1_matrices, top2_matrices, layer_specs, expert_labels, output_path)
    write_readme(model_names, output_root / "README_a4_top1_top2_paper_style.md")
    print(f"Wrote A4 top-1/top-2 paper-style plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
