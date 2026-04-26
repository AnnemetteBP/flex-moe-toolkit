from __future__ import annotations

import argparse
import csv
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


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "routing_light" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_top1_top2_confusion"
DEFAULT_MODELS = [
    "FlexOlmo-8x7B-1T-a4-55B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2-rt",
]
DEFAULT_EXPERT_LABELS = {
    0: "Public",
    1: "Code",
    2: "Creative\nWriting",
    3: "Math",
    4: "News",
    5: "Academic",
    6: "Reddit",
    7: "Danish",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mix top-1 vs top-2 expert competition heatmaps from routing-light records."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-name", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--run-label", default="native_full")
    parser.add_argument("--expert-label", action="append", default=[])
    return parser.parse_args()


def parse_expert_labels(raw_labels: list[str]) -> dict[int, str]:
    labels = dict(DEFAULT_EXPERT_LABELS)
    for raw_item in raw_labels:
        if "=" not in raw_item:
            raise ValueError(f"Invalid --expert-label value `{raw_item}`. Expected <idx>=<label>.")
        idx_str, label = raw_item.split("=", 1)
        labels[int(idx_str.strip())] = label.strip()
    return labels


def expert_label(expert_idx: int, labels: dict[int, str]) -> str:
    return labels.get(expert_idx, f"expert_{expert_idx}")


def model_display_name(model_name: str) -> str:
    return model_name.replace("FlexOlmo-8x7B-1T-", "")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_ids(value) -> list[int]:
    if isinstance(value, list):
        flattened: list[int] = []
        for item in value:
            flattened.extend(flatten_ids(item))
        return flattened
    return [int(value)]


def discover_models(results_root: Path, requested: list[str]) -> list[str]:
    if requested:
        return requested
    return [model_name for model_name in DEFAULT_MODELS if (results_root / model_name).exists()]


def discover_datasets(results_root: Path, model_names: list[str], requested: list[str], run_label: str) -> list[str]:
    if requested:
        return requested

    discovered: set[str] = set()
    for model_name in model_names:
        model_root = results_root / model_name
        if not model_root.exists():
            continue
        for dataset_dir in model_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            if (dataset_dir / run_label / "routing_records.jsonl").exists():
                discovered.add(dataset_dir.name)
    return sorted(discovered)


def load_records(results_root: Path, model_name: str, dataset_name: str, run_label: str) -> list[dict]:
    path = results_root / model_name / dataset_name / run_label / "routing_records.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def top1_top2_confusion(records: list[dict], language: str | None = None) -> np.ndarray | None:
    filtered = [record for record in records if language is None or record.get("language") == language]
    if not filtered:
        return None

    num_experts = max(DEFAULT_EXPERT_LABELS.keys()) + 1
    matrix = np.zeros((num_experts, num_experts), dtype=float)
    for record in filtered:
        for layer_summary in record.get("prompt_router_token_summaries_by_layer") or []:
            top1 = flatten_ids(layer_summary["top1_expert_ids"])
            top2 = flatten_ids(layer_summary["top2_expert_ids"])
            for left, right in zip(top1, top2):
                matrix[left, right] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def dataset_language_groups(dataset_name: str, records: list[dict]) -> list[tuple[str, str | None]]:
    languages = sorted({str(record.get("language", "unknown")) for record in records})
    if dataset_name == "mkqa_en_da" and languages:
        return [(language.upper(), language) for language in languages]
    return [("ALL", None)]


def dataset_label(dataset_name: str) -> str:
    return {
        "mkqa_en_da": "MGQA (EN/DA)",
        "gsm8k_subset": "GSM8K",
        "mbpp_subset": "MBPP",
        "pubmedqa_subset": "PubMedQA",
    }.get(dataset_name, dataset_name)


def build_summary_rows(
    results_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    run_label: str,
    expert_labels: dict[int, str],
) -> list[dict]:
    rows: list[dict] = []
    for dataset_name in dataset_names:
        reference_records: list[dict] = []
        for model_name in model_names:
            reference_records = load_records(results_root, model_name, dataset_name, run_label)
            if reference_records:
                break
        if not reference_records:
            continue
        for display_language, raw_language in dataset_language_groups(dataset_name, reference_records):
            for model_name in model_names:
                records = load_records(results_root, model_name, dataset_name, run_label)
                matrix = top1_top2_confusion(records, language=raw_language)
                if matrix is None:
                    continue
                for row_idx in range(matrix.shape[0]):
                    runner_up_idx = int(np.argmax(matrix[row_idx]))
                    rows.append(
                        {
                            "dataset_name": dataset_name,
                            "dataset_label": dataset_label(dataset_name),
                            "language": display_language,
                            "model_name": model_name,
                            "top1_expert": expert_label(row_idx, expert_labels),
                            "dominant_top2_expert": expert_label(runner_up_idx, expert_labels),
                            "dominant_top2_probability": float(matrix[row_idx, runner_up_idx]),
                            "self_runner_up_probability": float(matrix[row_idx, row_idx]),
                        }
                    )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrices(
    results_root: Path,
    output_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    run_label: str,
    expert_labels: dict[int, str],
) -> Path | None:
    row_specs: list[tuple[str, str, str | None]] = []
    for dataset_name in dataset_names:
        reference_records: list[dict] = []
        for model_name in model_names:
            reference_records = load_records(results_root, model_name, dataset_name, run_label)
            if reference_records:
                break
        if not reference_records:
            continue
        for display_language, raw_language in dataset_language_groups(dataset_name, reference_records):
            row_specs.append((dataset_name, display_language, raw_language))

    if not row_specs:
        return None

    fig, axes = plt.subplots(
        len(row_specs),
        len(model_names),
        figsize=(5.2 * len(model_names), 4.15 * len(row_specs)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    vmax = 0.0
    cached: dict[tuple[str, str, str | None], np.ndarray] = {}
    for dataset_name, display_language, raw_language in row_specs:
        for model_name in model_names:
            records = load_records(results_root, model_name, dataset_name, run_label)
            matrix = top1_top2_confusion(records, language=raw_language)
            if matrix is None:
                continue
            cached[(dataset_name, model_name, raw_language)] = matrix
            vmax = max(vmax, float(matrix.max()))
    vmax = max(vmax, 0.5)

    for row_idx, (dataset_name, display_language, raw_language) in enumerate(row_specs):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx][col_idx]
            matrix = cached.get((dataset_name, model_name, raw_language))
            if matrix is None:
                ax.axis("off")
                continue
            show_cbar = row_idx == 0 and col_idx == len(model_names) - 1
            sns.heatmap(
                matrix,
                cmap="rocket_r",
                vmin=0.0,
                vmax=vmax,
                ax=ax,
                cbar=show_cbar,
                cbar_kws={"label": "P(top2 | top1)"},
            )
            if show_cbar and fig.axes:
                colorbar_ax = fig.axes[-1]
                colorbar_ax.set_ylabel("P(top2 | top1)", fontweight="semibold")
            ax.set_title(
                f"{model_display_name(model_name)} | {dataset_label(dataset_name)} | {display_language}",
                fontweight="bold",
                fontsize=12.0,
                pad=3,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            tick_labels = [expert_label(idx, expert_labels) for idx in range(matrix.shape[0])]
            ax.set_xticks(np.arange(matrix.shape[0]) + 0.5)
            ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=10)
            ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
            ax.set_yticklabels(tick_labels, rotation=0, fontsize=10)

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.11, top=0.93, wspace=0.10, hspace=0.32)
    fig.suptitle("Mix Expert-Pair Competition: Top-1 vs Top-2", y=0.945, fontweight="bold", fontsize=16)
    fig.supxlabel("Top-2 Expert", y=0.045, fontweight="semibold", fontsize=12)
    fig.supylabel("Top-1 Expert", x=0.045, fontweight="semibold", fontsize=12)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "mix_top1_top2_confusion.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_readme(
    output_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    run_label: str,
) -> None:
    lines = [
        "# Mix Top-1 vs Top-2 Competition",
        "",
        "This directory contains top-1 vs top-2 expert competition heatmaps derived from mix routing-light records.",
        "",
        "Interpretation:",
        "- Each heatmap row is normalized as $P(\\mathrm{top2}=j \\mid \\mathrm{top1}=i)$.",
        "- These matrices show local competition structure in the router ranking, not general co-activation.",
        "- They are useful for identifying which experts act as common runner-up alternatives to a given winning expert.",
        "",
        "Models:",
    ]
    for model_name in model_names:
        lines.append(f"- `{model_name}`")
    lines.extend(
        [
            "",
            "Datasets:",
        ]
    )
    for dataset_name in dataset_names:
        lines.append(f"- `{dataset_name}`")
    lines.append("")
    lines.append(f"Run label: `{run_label}`")
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    expert_labels = parse_expert_labels(args.expert_label)
    model_names = discover_models(args.results_root, args.model_name)
    if not model_names:
        raise ValueError(f"No models were found under {args.results_root}.")
    dataset_names = discover_datasets(args.results_root, model_names, args.dataset, args.run_label)
    if not dataset_names:
        raise ValueError(f"No datasets with run label `{args.run_label}` were found under {args.results_root}.")

    output_path = plot_confusion_matrices(
        results_root=args.results_root,
        output_root=args.output_root,
        model_names=model_names,
        dataset_names=dataset_names,
        run_label=args.run_label,
        expert_labels=expert_labels,
    )
    if output_path is None:
        raise ValueError("No top1/top2 confusion matrices could be generated from the available records.")

    summary_rows = build_summary_rows(
        results_root=args.results_root,
        model_names=model_names,
        dataset_names=dataset_names,
        run_label=args.run_label,
        expert_labels=expert_labels,
    )
    write_csv(summary_rows, args.output_root / "mix_top1_top2_confusion_summary.csv")
    write_readme(args.output_root, model_names, dataset_names, args.run_label)
    print(f"Wrote mix top1/top2 confusion analysis to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
