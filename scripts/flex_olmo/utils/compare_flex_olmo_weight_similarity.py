from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys

import torch
from transformers import FlexOlmoForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from flex_moe_toolkit.pipelines.flex_olmo_weights import analyze_flex_olmo_weights
from flex_moe_toolkit.utils.jsonl import to_jsonable


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
SELECTED_LAYERS = [3, 15, 23, 31]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare expert weight similarity for selected FlexOlmo checkpoints."
    )
    parser.add_argument(
        "--model-root",
        required=True,
        help="Directory containing model folders, e.g. /work/training/FlexMoRE/models",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional explicit model names to compare. Defaults to the focused 55B A4/A8 set.",
    )
    parser.add_argument(
        "--output-root",
        default="eval_results/weight_similarity/flexolmo_55b_comparison",
        help="Directory where summaries and figures will be written.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device: cpu, cuda, or an explicit device like cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Torch dtype used when loading checkpoints.",
    )
    parser.add_argument(
        "--public-expert-idx",
        type=int,
        default=0,
        help="Expert index treated as the public expert.",
    )
    parser.add_argument(
        "--expert-label",
        action="append",
        default=[],
        help="Optional expert label mapping in the form <idx>=<label>.",
    )
    return parser.parse_args()


def parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


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


def load_model_analysis(model_path: Path, device: torch.device, dtype_name: str, public_expert_idx: int) -> dict:
    model = FlexOlmoForCausalLM.from_pretrained(str(model_path), torch_dtype=parse_dtype(dtype_name))
    model.to(device)
    model.eval()
    analysis = analyze_flex_olmo_weights(model, public_expert_idx=public_expert_idx)
    analysis["model_name"] = model_path.name
    analysis["model_path"] = str(model_path)
    analysis["model_impl_path"] = str(inspect.getsourcefile(FlexOlmoForCausalLM))
    del model
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
    return analysis


def plot_offdiag_similarity(analyses: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharex=True)
    for analysis in analyses:
        model_name = analysis["model_name"].replace("FlexOlmo-8x7B-1T-", "")
        layers = [record["layer_idx"] for record in analysis["layer_weight_analysis"]]
        gate = [record["gate_up_proj_similarity_summary"]["mean_offdiag_similarity"] for record in analysis["layer_weight_analysis"]]
        axes[0].plot(layers, gate, linewidth=2.0, marker="o", label=model_name)
        if analysis["layer_weight_analysis"][0].get("down_proj_similarity_summary") is not None:
            down = [record["down_proj_similarity_summary"]["mean_offdiag_similarity"] for record in analysis["layer_weight_analysis"]]
            axes[1].plot(layers, down, linewidth=2.0, marker="o", label=model_name)

    axes[0].set_title("Gate/Up Projection Off-Diagonal Similarity")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean Cosine Similarity")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_title("Down Projection Off-Diagonal Similarity")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean Cosine Similarity")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Expert Weight Similarity Across Layers", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_public_distance(analyses: list[dict], expert_labels: dict[int, str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(analyses), figsize=(5.2 * len(analyses), 5.0), sharey=True)
    if len(analyses) == 1:
        axes = [axes]
    for ax, analysis in zip(axes, analyses):
        model_name = analysis["model_name"].replace("FlexOlmo-8x7B-1T-", "")
        layer_matrix = np.array(
            [record["gate_up_proj_public_distance"] for record in analysis["layer_weight_analysis"]],
            dtype=float,
        )
        sns.heatmap(
            layer_matrix.T,
            cmap="mako",
            ax=ax,
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "Distance from Public Expert"},
        )
        ax.set_title(model_name)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Expert")
        ax.set_yticks(np.arange(layer_matrix.shape[1]) + 0.5)
        ax.set_yticklabels([expert_label(idx, expert_labels) for idx in range(layer_matrix.shape[1])], rotation=0)
    fig.suptitle("Gate/Up Projection Distance from Public Expert", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selected_layer_heatmaps(analyses: list[dict], expert_labels: dict[int, str], matrix_key: str, title: str, stem: str, output_root: Path) -> None:
    for layer_idx in SELECTED_LAYERS:
        fig, axes = plt.subplots(1, len(analyses), figsize=(5.2 * len(analyses), 4.8), sharex=True, sharey=True)
        if len(analyses) == 1:
            axes = [axes]
        for ax, analysis in zip(axes, analyses):
            layer_records = analysis["layer_weight_analysis"]
            if layer_idx >= len(layer_records):
                ax.axis("off")
                continue
            matrix = np.array(layer_records[layer_idx][matrix_key], dtype=float)
            sns.heatmap(
                matrix,
                cmap="coolwarm",
                center=0.0 if matrix.min() < 0 else None,
                vmin=min(-1.0, float(matrix.min())) if matrix.min() < 0 else 0.0,
                vmax=1.0,
                ax=ax,
                cbar=(ax is axes[-1]),
                cbar_kws={"label": "Cosine Similarity"},
            )
            ax.set_title(analysis["model_name"].replace("FlexOlmo-8x7B-1T-", ""))
            ax.set_xlabel("Expert")
            ax.set_ylabel("Expert")
            tick_labels = [expert_label(idx, expert_labels) for idx in range(matrix.shape[0])]
            ax.set_xticks(np.arange(matrix.shape[0]) + 0.5)
            ax.set_xticklabels(tick_labels, rotation=35, ha="right")
            ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
            ax.set_yticklabels(tick_labels, rotation=0)
        fig.suptitle(f"{title} | Layer {layer_idx}", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_root / f"{stem}_layer_{layer_idx}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_summary(analyses: list[dict], output_path: Path) -> None:
    summary_rows = []
    for analysis in analyses:
        gate_mean = analysis["summary"]["mean_gate_up_proj_offdiag_similarity"]
        down_mean = analysis["summary"]["mean_down_proj_offdiag_similarity"]
        danish_distance = float(np.mean([record["gate_up_proj_public_distance"][7] for record in analysis["layer_weight_analysis"]]))
        summary_rows.append(
            {
                "model_name": analysis["model_name"],
                "mean_gate_up_proj_offdiag_similarity": gate_mean,
                "mean_down_proj_offdiag_similarity": down_mean,
                "mean_danish_distance_from_public": danish_distance,
            }
        )
    output_path.write_text(json.dumps(to_jsonable(summary_rows), indent=2, sort_keys=True), encoding="utf-8")


def write_readme(analyses: list[dict], output_path: Path) -> None:
    lines = [
        "# FlexOlmo Weight Similarity Comparison",
        "",
        "- These plots test whether expert FFN weight matrices are becoming too similar to distinguish reliably.",
        "- `gate_up_proj_similarity` is the most direct proxy for expert matrix similarity in the current pipeline.",
        "- `public_distance` summarizes how far each expert's gate/up weights move away from the assumed public expert.",
        "",
        "## Models",
        "",
    ]
    for analysis in analyses:
        lines.append(
            f"- `{analysis['model_name']}`: "
            f"mean gate/up off-diagonal similarity = {analysis['summary']['mean_gate_up_proj_offdiag_similarity']:.4f}, "
            f"mean down-proj off-diagonal similarity = {analysis['summary']['mean_down_proj_offdiag_similarity']:.4f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_names = args.model_name or DEFAULT_MODELS
    expert_labels = parse_expert_labels(args.expert_label, public_expert_idx=args.public_expert_idx)

    analyses = []
    for model_name in model_names:
        model_path = Path(args.model_root) / model_name
        analyses.append(
            load_model_analysis(
                model_path=model_path,
                device=device,
                dtype_name=args.dtype,
                public_expert_idx=args.public_expert_idx,
            )
        )

    details_dir = output_root / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for analysis in analyses:
        detail_path = details_dir / f"{analysis['model_name']}.weights.json"
        detail_path.write_text(json.dumps(to_jsonable(analysis), indent=2, sort_keys=True), encoding="utf-8")

    plot_offdiag_similarity(analyses, output_root / "offdiag_similarity_by_layer.png")
    plot_public_distance(analyses, expert_labels, output_root / "public_distance_heatmap.png")
    plot_selected_layer_heatmaps(
        analyses,
        expert_labels,
        matrix_key="gate_up_proj_similarity",
        title="Gate/Up Projection Expert Similarity",
        stem="gate_up_proj_similarity",
        output_root=output_root,
    )
    plot_selected_layer_heatmaps(
        analyses,
        expert_labels,
        matrix_key="down_proj_similarity",
        title="Down Projection Expert Similarity",
        stem="down_proj_similarity",
        output_root=output_root,
    )
    write_summary(analyses, output_root / "summary.json")
    write_readme(analyses, output_root / "README.md")
    print(f"Wrote weight-similarity comparison to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
