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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-style A4 co-activation comparison plots from saved MKQA routing analysis."
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
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Plot baseline-corrected residual co-activation instead of raw normalized co-activation.",
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
        analysis_path = model_dir / "mkqa_en_da" / "native_full" / "routing_analysis.jsonl"
        if analysis_path.exists():
            model_names.append(model_dir.name)
    return sorted(model_names, key=model_sort_key)


def selected_models(results_root: Path, requested: list[str]) -> list[str]:
    available = discover_available_models(results_root)
    if requested:
        return [name for name in requested if name in available]
    return [name for name in DEFAULT_MODELS if name in available]


def load_aggregate_record(results_root: Path, model_name: str) -> dict:
    analysis_path = (
        results_root
        / "routing"
        / model_name
        / "mkqa_en_da"
        / "native_full"
        / "routing_analysis.jsonl"
    )
    rows = load_jsonl(analysis_path)
    if not rows:
        raise ValueError(f"No routing analysis rows found for {model_name}.")
    aggregate = rows[-1]
    if aggregate.get("record_type") != "routing_aggregate":
        raise ValueError(f"Final routing analysis row for {model_name} is not a routing_aggregate.")
    return aggregate


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


def baseline_corrected_coactivation_matrix(
    coactivation_matrix: np.ndarray,
    activation_counts: np.ndarray,
    k: int,
    num_experts: int,
) -> np.ndarray:
    baseline = np.zeros_like(coactivation_matrix, dtype=float)
    offdiag = 0.0 if num_experts <= 1 else max(0.0, (k - 1) / (num_experts - 1))
    for row_idx in range(num_experts):
        if activation_counts[row_idx] <= 0:
            continue
        baseline[row_idx, :] = offdiag
        baseline[row_idx, row_idx] = 1.0
    return coactivation_matrix - baseline


def plot_coactivation_grid(
    model_names: list[str],
    matrices: dict[str, list[np.ndarray]],
    activation_counts: dict[str, np.ndarray],
    layer_specs: list[tuple[str, int]],
    expert_labels: dict[int, str],
    use_residual: bool,
    output_path: Path,
) -> None:
    sns.set_theme(style="white")
    num_rows = len(layer_specs)
    num_cols = len(model_names)
    num_experts = next(iter(matrices.values()))[0].shape[0]

    if use_residual:
        vmax = 0.0
        for model_name in model_names:
            for layer_idx in range(len(matrices[model_name])):
                residual = baseline_corrected_coactivation_matrix(
                    matrices[model_name][layer_idx],
                    activation_counts[model_name],
                    k=2,
                    num_experts=num_experts,
                )
                vmax = max(vmax, float(np.abs(residual).max()))
        heatmap_kwargs = {
            "cmap": "coolwarm",
            "center": 0.0,
            "vmin": -vmax,
            "vmax": vmax,
        }
        colorbar_label = "Residual Co-Activation"
    else:
        vmax = max(float(matrix.max()) for layers in matrices.values() for matrix in layers)
        heatmap_kwargs = {
            "cmap": "mako",
            "vmin": 0.0,
            "vmax": vmax,
        }
        colorbar_label = "Normalized Co-Activation"

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4.8 * num_cols, 3.1 * num_rows + 0.8),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row_idx, (layer_label, layer_idx) in enumerate(layer_specs):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            matrix = matrices[model_name][layer_idx]
            if use_residual:
                matrix = baseline_corrected_coactivation_matrix(
                    matrix,
                    activation_counts[model_name],
                    k=2,
                    num_experts=num_experts,
                )
            sns.heatmap(
                matrix,
                ax=ax,
                cbar=(row_idx == 0 and col_idx == num_cols - 1),
                cbar_kws={"label": colorbar_label},
                **heatmap_kwargs,
            )
            if row_idx == 0:
                ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
            if col_idx == 0:
                ax.set_ylabel(f"{layer_label}\nlayer {layer_idx}\nExpert")
            else:
                ax.set_ylabel("")
            if row_idx == num_rows - 1:
                ax.set_xlabel("Expert")
            else:
                ax.set_xlabel("")
            ax.set_xticks(np.arange(num_experts) + 0.5)
            ax.set_xticklabels(
                [expert_label(idx, expert_labels) for idx in range(num_experts)],
                rotation=40,
                ha="right",
            )
            ax.set_yticks(np.arange(num_experts) + 0.5)
            ax.set_yticklabels([expert_label(idx, expert_labels) for idx in range(num_experts)], rotation=0)

    title = "A4 Residual Co-Activation Across Selected Layers" if use_residual else "A4 Co-Activation Across Selected Layers"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    model_names: list[str],
    layer_specs: list[tuple[str, int]],
    output_path: Path,
) -> None:
    lines = [
        "# A4 Co-Activation Paper-Style Plots",
        "",
        f"- Models: {', '.join(f'`{name}`' for name in model_names)}",
        "- Layers shown:",
    ]
    for label, idx in layer_specs:
        lines.append(f"  - `{label}` -> layer {idx}")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `a4_coactivation_selected_layers.png`: raw normalized co-activation matrices across selected layers.",
            "- `a4_residual_coactivation_selected_layers.png`: baseline-corrected residual co-activation matrices across selected layers.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root) if args.output_root else results_root / "routing" / "plots"
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = selected_models(results_root, args.model_name)
    if not model_names:
        raise SystemExit(f"No matching A4 routing runs found under {results_root}.")

    expert_labels = parse_expert_labels(args.expert_label, public_expert_idx=args.public_expert_idx)
    matrices = {}
    activation_count_map = {}
    for model_name in model_names:
        aggregate = load_aggregate_record(results_root, model_name)
        matrices[model_name] = [np.array(matrix, dtype=float) for matrix in aggregate["layer_coactivation_matrices"]]
        activation_count_map[model_name] = np.array(aggregate["activation_counts"], dtype=float)

    num_layers = len(next(iter(matrices.values())))
    layer_specs = resolve_layer_specs(num_layers=num_layers, requested_layers=args.layer)
    if not layer_specs:
        raise SystemExit("No valid layers were selected.")

    plot_coactivation_grid(
        model_names=model_names,
        matrices=matrices,
        activation_counts=activation_count_map,
        layer_specs=layer_specs,
        expert_labels=expert_labels,
        use_residual=args.residual,
        output_path=output_root / (
            "a4_residual_coactivation_selected_layers.png"
            if args.residual
            else "a4_coactivation_selected_layers.png"
        ),
    )
    write_readme(
        model_names=model_names,
        layer_specs=layer_specs,
        output_path=output_root / "README_a4_coactivation_paper_style.md",
    )
    figure_name = (
        "a4_residual_coactivation_selected_layers.png"
        if args.residual
        else "a4_coactivation_selected_layers.png"
    )
    print(f"Wrote A4 co-activation plot to {output_root / figure_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
