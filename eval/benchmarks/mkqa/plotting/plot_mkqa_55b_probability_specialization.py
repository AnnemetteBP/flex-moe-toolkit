from __future__ import annotations

import argparse
import json
import os
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
        description="Plot probability-weighted language specialization for A4/A8 55B comparison models."
    )
    parser.add_argument("--a4-root", default="eval_results/mkqa/full/flexolmo/a4")
    parser.add_argument("--a8-root", default="eval_results/mkqa/full/flexolmo/a8")
    parser.add_argument(
        "--output-root",
        default="eval_results/mkqa/full/flexolmo/comparisons/a4_a8_55b_probability",
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


def mean_prob_by_language(records: list[dict]) -> dict[str, np.ndarray]:
    by_language = {}
    languages = sorted({record["language"] for record in records})
    num_layers = len(records[0]["prompt_router_probs_by_layer"])
    num_experts = len(records[0]["prompt_router_probs_by_layer"][0][0])
    for language in languages:
        language_records = [record for record in records if record["language"] == language]
        matrix = np.zeros((num_layers, num_experts), dtype=float)
        for layer_idx in range(num_layers):
            token_prob_rows = []
            for record in language_records:
                token_prob_rows.extend(record["prompt_router_probs_by_layer"][layer_idx])
            matrix[layer_idx] = np.mean(np.array(token_prob_rows, dtype=float), axis=0)
        by_language[language] = matrix
    return by_language


def resolve_layer_specs(num_layers: int) -> list[tuple[str, int]]:
    return [(label, min(idx, num_layers - 1)) for label, idx in DEFAULT_LAYER_LABELS]


def plot_language_probability_heatmaps(model_names: list[str], model_rows: dict[str, dict], expert_labels: dict[int, str], output_path: Path) -> None:
    fig, axes = plt.subplots(len(model_names), 2, figsize=(13.5, max(4.0 * len(model_names), 5.0)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    languages = ("en", "da")
    for row_idx, model_name in enumerate(model_names):
        for col_idx, language in enumerate(languages):
            ax = axes[row_idx, col_idx]
            matrix = model_rows[model_name]["prob_by_language"][language]
            sns.heatmap(
                matrix.T,
                cmap="mako",
                vmin=0.0,
                vmax=max(0.3, float(matrix.max())),
                ax=ax,
                cbar=(row_idx == 0 and col_idx == len(languages) - 1),
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
    fig.suptitle("Probability-Weighted Expert Usage by Language", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_language_probability_diff(model_names: list[str], model_rows: dict[str, dict], expert_labels: dict[int, str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(model_names), figsize=(5.2 * len(model_names), 5.0), sharey=True)
    if len(model_names) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, model_names):
        da = model_rows[model_name]["prob_by_language"]["da"]
        en = model_rows[model_name]["prob_by_language"]["en"]
        diff = da - en
        sns.heatmap(
            diff.T,
            cmap="coolwarm",
            center=0.0,
            vmin=-max(0.2, float(np.abs(diff).max())),
            vmax=max(0.2, float(np.abs(diff).max())),
            ax=ax,
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "DA - EN Probability"},
        )
        ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
        ax.set_xlabel("Layer")
        ax.set_yticks(np.arange(diff.shape[1]) + 0.5)
        ax.set_yticklabels([expert_label(idx, expert_labels) for idx in range(diff.shape[1])], rotation=0)
        ax.set_ylabel("Expert")
    fig.suptitle("Language Preference from Router Probabilities", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selected_layer_curves(model_names: list[str], model_rows: dict[str, dict], expert_labels: dict[int, str], output_path: Path) -> None:
    layer_specs = resolve_layer_specs(next(iter(model_rows.values()))["prob_by_language"]["en"].shape[0])
    fig, axes = plt.subplots(len(layer_specs), len(model_names), figsize=(5.2 * len(model_names), 3.0 * len(layer_specs) + 0.8), sharey=True)
    axes = np.atleast_2d(axes)
    for row_idx, (layer_label, layer_idx) in enumerate(layer_specs):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            x = np.arange(model_rows[model_name]["prob_by_language"]["en"].shape[1])
            for language in ("en", "da"):
                y = model_rows[model_name]["prob_by_language"][language][layer_idx]
                ax.plot(x, y, marker="o", linewidth=1.8, color=LANGUAGE_COLORS[language], label=language.upper())
            ax.set_xticks(x)
            ax.set_xticklabels([expert_label(idx, expert_labels) for idx in x], rotation=35, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
            if row_idx == 0:
                ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
            if col_idx == 0:
                ax.set_ylabel(f"{layer_label}\nlayer {layer_idx}\nMean Probability")
            if row_idx == len(layer_specs) - 1:
                ax.set_xlabel("Expert")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Selected-Layer Probability-Weighted Specialization", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_public_vs_danish(model_names: list[str], model_rows: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(model_names), figsize=(5.2 * len(model_names), 4.8), sharey=True)
    if len(model_names) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, model_names):
        for language in ("en", "da"):
            matrix = model_rows[model_name]["prob_by_language"][language]
            public_minus_danish = matrix[:, 0] - matrix[:, 7]
            danish_prob = matrix[:, 7]
            ax.plot(public_minus_danish, color=LANGUAGE_COLORS[language], linewidth=2.0, label=f"{language.upper()}: public - danish")
            ax.plot(danish_prob, color=LANGUAGE_COLORS[language], linewidth=1.6, linestyle="--", alpha=0.9, label=f"{language.upper()}: danish")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(model_name.replace("FlexOlmo-8x7B-1T-", ""))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probability / Gap")
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Public-vs-Danish Probability Competition", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(model_names: list[str], output_path: Path) -> None:
    lines = [
        "# A4 vs A8 Probability-Weighted Specialization",
        "",
        "- These plots summarize mean router probability mass rather than binary expert activation.",
        "- This is especially important for A8, where activation-based views become trivial when all experts are active.",
        "- The key question is whether the assumed Danish expert receives more probability mass on Danish prompts, and whether router tuning changes that pattern.",
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
        family_root = model_root_for_name(model_name, a4_root, a8_root)
        records = load_jsonl(family_root / "routing" / model_name / "mkqa_en_da" / "native_full" / "routing_records.jsonl")
        model_rows[model_name] = {
            "prob_by_language": mean_prob_by_language(records),
        }

    plot_language_probability_heatmaps(DEFAULT_MODELS, model_rows, expert_labels, output_root / "a4_a8_probability_by_language.png")
    plot_language_probability_diff(DEFAULT_MODELS, model_rows, expert_labels, output_root / "a4_a8_probability_da_minus_en.png")
    plot_selected_layer_curves(DEFAULT_MODELS, model_rows, expert_labels, output_root / "a4_a8_probability_selected_layers.png")
    plot_public_vs_danish(DEFAULT_MODELS, model_rows, output_root / "a4_a8_public_vs_danish.png")
    write_readme(DEFAULT_MODELS, output_root / "README.md")
    print(f"Wrote A4 vs A8 probability-specialization plots to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
