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
        description="Generate paper-style A4 domain-specialization comparison plots from saved MKQA summaries."
    )
    parser.add_argument(
        "--results-root",
        default="eval_results/mkqa/full/flexolmo/a4",
        help="MKQA A4 results root containing domain_specialization outputs.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where the figure will be written. Defaults to <results-root>/domain_specialization/plots.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional explicit A4 model names to include.",
    )
    parser.add_argument(
        "--source",
        default="prompt",
        choices=["prompt", "predicted", "ground_truth"],
        help="Domain-specialization source to plot. Defaults to prompt.",
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
    domain_root = results_root / "domain_specialization"
    if not domain_root.exists():
        return []
    model_names = []
    for model_dir in domain_root.iterdir():
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "native_full" / "domain_specialization_summary.jsonl"
        if summary_path.exists():
            model_names.append(model_dir.name)
    return sorted(model_names, key=model_sort_key)


def selected_models(results_root: Path, requested: list[str]) -> list[str]:
    available = discover_available_models(results_root)
    if requested:
        return [name for name in requested if name in available]
    return [name for name in DEFAULT_MODELS if name in available]


def load_domain_summary(results_root: Path, model_name: str) -> list[dict]:
    summary_path = (
        results_root
        / "domain_specialization"
        / model_name
        / "native_full"
        / "domain_specialization_summary.jsonl"
    )
    return load_jsonl(summary_path)


def build_language_matrix(rows: list[dict], source: str) -> np.ndarray:
    filtered = [row for row in rows if row.get("source") == source]
    if not filtered:
        raise ValueError(f"No rows found for source `{source}`.")
    num_layers = max(int(row["layer_idx"]) for row in filtered) + 1
    num_experts = max(int(row["expert_idx"]) for row in filtered) + 1
    matrix = np.zeros((num_layers, num_experts, 2), dtype=float)
    for row in filtered:
        layer_idx = int(row["layer_idx"])
        expert_idx = int(row["expert_idx"])
        matrix[layer_idx, expert_idx, 0] = float(row["language_specialization"].get("en", 0.0))
        matrix[layer_idx, expert_idx, 1] = float(row["language_specialization"].get("da", 0.0))
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


def plot_domain_specialization_grid(
    model_names: list[str],
    matrices: dict[str, np.ndarray],
    layer_specs: list[tuple[str, int]],
    expert_labels: dict[int, str],
    source: str,
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    num_rows = len(layer_specs)
    num_cols = len(model_names)
    num_experts = next(iter(matrices.values())).shape[1]
    expert_axis = np.arange(num_experts)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4.8 * num_cols, 2.9 * num_rows + 0.8),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row_idx, (layer_label, layer_idx) in enumerate(layer_specs):
        for col_idx, model_name in enumerate(model_names):
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
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(expert_axis)
            ax.set_xticklabels(
                [expert_label(idx, expert_labels) for idx in expert_axis],
                rotation=40,
                ha="right",
            )
            ax.grid(True, axis="y", alpha=0.25)

            if row_idx == 0:
                ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
            if col_idx == 0:
                ax.set_ylabel(f"{layer_label}\nlayer {layer_idx}\nSpecialization")
            else:
                ax.set_ylabel("")
            if row_idx == num_rows - 1:
                ax.set_xlabel("Expert")
            else:
                ax.set_xlabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        f"A4 {source.capitalize()} Domain Specialization Across Selected Layers",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    model_names: list[str],
    layer_specs: list[tuple[str, int]],
    source: str,
    output_path: Path,
) -> None:
    lines = [
        "# A4 Domain Specialization Paper-Style Plots",
        "",
        f"- Source: `{source}`",
        f"- Models: {', '.join(f'`{name}`' for name in model_names)}",
        "- Layers shown:",
    ]
    for label, idx in layer_specs:
        lines.append(f"  - `{label}` -> layer {idx}")
    lines.extend(
        [
            "",
            "## Figure",
            "",
            f"- `a4_{source}_domain_specialization_selected_layers.png`: selected-layer expert curves for English vs Danish across the chosen A4 checkpoints.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root) if args.output_root else results_root / "domain_specialization" / "plots"
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = selected_models(results_root, args.model_name)
    if not model_names:
        raise SystemExit(f"No matching A4 domain-specialization runs found under {results_root}.")

    expert_labels = parse_expert_labels(args.expert_label, public_expert_idx=args.public_expert_idx)
    matrices = {}
    for model_name in model_names:
        rows = load_domain_summary(results_root, model_name)
        matrices[model_name] = build_language_matrix(rows, source=args.source)

    num_layers = next(iter(matrices.values())).shape[0]
    layer_specs = resolve_layer_specs(num_layers=num_layers, requested_layers=args.layer)
    if not layer_specs:
        raise SystemExit("No valid layers were selected.")

    figure_path = output_root / f"a4_{args.source}_domain_specialization_selected_layers.png"
    plot_domain_specialization_grid(
        model_names=model_names,
        matrices=matrices,
        layer_specs=layer_specs,
        expert_labels=expert_labels,
        source=args.source,
        output_path=figure_path,
    )
    write_readme(
        model_names=model_names,
        layer_specs=layer_specs,
        source=args.source,
        output_path=output_root / "README_a4_domain_specialization_paper_style.md",
    )
    print(f"Wrote paper-style A4 domain specialization plot to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
