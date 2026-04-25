from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "router_direction" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_router_direction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze router-direction / expert-alignment outputs for the 55B FlexOlmo pair."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--model-names",
        default="FlexOlmo-8x7B-1T-a4-55B-v2,FlexOlmo-8x7B-1T-a4-55B-v2-rt",
        help="Comma-separated model names to compare.",
    )
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include.")
    return parser.parse_args()


def parse_model_names(raw_value: str) -> list[str]:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(names) < 2:
        raise ValueError("Provide at least two model names.")
    return names


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cosine_matrix(weight_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weight_matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-9, None)
    return normalized @ normalized.T


def load_suite_manifest(results_root: Path, model_name: str) -> dict:
    manifest_path = results_root / model_name / "router_direction_suite_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing router-direction suite manifest for `{model_name}` at {manifest_path}. "
            "Run the router-direction suite first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_dataset_records(results_root: Path, model_name: str, dataset_name: str) -> list[dict]:
    path = results_root / model_name / dataset_name / "router_direction_records.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def load_router_weights(results_root: Path, model_name: str) -> dict[int, np.ndarray]:
    npz_path = results_root / model_name / "router_weights.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing router weights for `{model_name}` at {npz_path}.")
    data = np.load(npz_path)
    weights: dict[int, np.ndarray] = {}
    for key in data.files:
        if not key.startswith("layer_") or not key.endswith("_weights"):
            continue
        layer_idx = int(key[len("layer_") : -len("_weights")])
        weights[layer_idx] = np.asarray(data[key], dtype=np.float32)
    return weights


def summarize_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for record in records:
        key = (str(record["dataset_name"]), str(record.get("language", "unknown")), int(record["layer"]))
        grouped.setdefault(key, []).append(record)

    rows: list[dict] = []
    for (dataset_name, language, layer), items in sorted(grouped.items()):
        rows.append(
            {
                "dataset_name": dataset_name,
                "language": language,
                "layer": layer,
                "num_examples": len(items),
                "mean_top1_alignment": float(np.mean([item["top1_alignment"] for item in items])),
                "mean_alignment_margin": float(np.mean([item["alignment_margin"] for item in items])),
                "mean_alignment_entropy": float(np.mean([item["alignment_entropy"] for item in items])),
                "top1_agreement_rate": float(np.mean([1.0 if item.get("agreement_top1") else 0.0 for item in items])),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_weight_heatmaps(
    weights_by_model: dict[str, dict[int, np.ndarray]],
    model_names: list[str],
    output_path: Path,
) -> None:
    layer_ids = sorted(set(weights_by_model[model_names[0]]) & set(weights_by_model[model_names[1]]))
    if not layer_ids:
        return
    fig, axes = plt.subplots(
        len(layer_ids),
        len(model_names),
        figsize=(5 * len(model_names), 4 * len(layer_ids)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, layer_idx in enumerate(layer_ids):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx][col_idx]
            matrix = cosine_matrix(weights_by_model[model_name][layer_idx])
            image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.set_title(f"{model_name} | layer {layer_idx}")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Expert")
    fig.colorbar(image, ax=axes, shrink=0.85, label="Cosine similarity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_alignment_metrics(
    rows_by_model: dict[str, list[dict]],
    model_names: list[str],
    output_path: Path,
) -> None:
    all_rows = [row for model_rows in rows_by_model.values() for row in model_rows]
    datasets = sorted({row["dataset_name"] for row in all_rows})
    metrics = [
        ("mean_top1_alignment", "Mean Top-1 Alignment"),
        ("mean_alignment_margin", "Alignment Margin"),
        ("top1_agreement_rate", "Top-1 Agreement Rate"),
    ]
    fig, axes = plt.subplots(
        len(datasets),
        len(metrics),
        figsize=(6 * len(metrics), 4 * len(datasets)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {model_names[0]: "#5B6C8F", model_names[1]: "#C96B3B"}
    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, (metric_key, title) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            for model_name in model_names:
                dataset_rows = [
                    row for row in rows_by_model[model_name]
                    if row["dataset_name"] == dataset_name and row["language"] == "unknown"
                ]
                if not dataset_rows:
                    dataset_rows = [row for row in rows_by_model[model_name] if row["dataset_name"] == dataset_name]
                dataset_rows = sorted(dataset_rows, key=lambda item: int(item["layer"]))
                if not dataset_rows:
                    continue
                ax.plot(
                    [int(row["layer"]) for row in dataset_rows],
                    [float(row[metric_key]) for row in dataset_rows],
                    marker="o",
                    label=model_name,
                    color=colors[model_name],
                )
            ax.set_title(f"{dataset_name} | {title}")
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.25)
            if metric_key == "top1_agreement_rate":
                ax.set_ylim(-0.05, 1.05)
    axes[0][0].legend(frameon=False, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_top1_expert_heatmaps(
    records_by_model: dict[str, list[dict]],
    model_names: list[str],
    output_path: Path,
) -> None:
    layer_ids = sorted({int(record["layer"]) for records in records_by_model.values() for record in records})
    datasets = sorted({str(record["dataset_name"]) for records in records_by_model.values() for record in records})
    num_experts = 0
    for records in records_by_model.values():
        for record in records:
            num_experts = max(num_experts, int(record["top1_aligned_expert"]) + 1)
    if num_experts == 0:
        return

    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 8), squeeze=False, constrained_layout=True)
    axes = axes[0]
    for ax, model_name in zip(axes, model_names):
        matrix = np.zeros((len(datasets) * len(layer_ids), num_experts), dtype=np.float32)
        ylabels = []
        for dataset_idx, dataset_name in enumerate(datasets):
            for layer_pos, layer_idx in enumerate(layer_ids):
                row_idx = dataset_idx * len(layer_ids) + layer_pos
                subset = [
                    record for record in records_by_model[model_name]
                    if str(record["dataset_name"]) == dataset_name and int(record["layer"]) == layer_idx
                ]
                if subset:
                    counts = np.zeros(num_experts, dtype=np.float32)
                    for record in subset:
                        counts[int(record["top1_aligned_expert"])] += 1.0
                    matrix[row_idx] = counts / counts.sum()
                ylabels.append(f"{dataset_name}|L{layer_idx}")
        image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
        ax.set_title(model_name)
        ax.set_xlabel("Top-1 aligned expert")
        ax.set_xticks(range(num_experts))
        ax.set_xticklabels([f"E{i}" for i in range(num_experts)])
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.85, label="Frequency")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_readme(
    output_path: Path,
    model_names: list[str],
) -> None:
    lines = [
        "# Router Direction Analysis",
        "",
        "Compared models:",
        f"- `{model_names[0]}`",
        f"- `{model_names[1]}`",
        "",
        "Artifacts:",
        "- `router_direction_summary.csv`",
        "- `router_weight_cosine_heatmaps.png`",
        "- `router_direction_alignment_metrics.png`",
        "- `router_direction_top1_expert_heatmaps.png`",
        "",
        "Interpretation guide:",
        "- Weight cosine heatmaps show how similar expert router vectors are within a layer.",
        "- Alignment metrics show how strongly pre-router activations align with expert directions by layer.",
        "- Top-1 agreement rate measures whether geometric top-1 alignment matches the actual routed top-1 expert.",
        "- Top-1 aligned expert heatmaps show which expert directions dominate per dataset/layer.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    model_names = parse_model_names(args.model_names)
    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}

    records_by_model: dict[str, list[dict]] = {}
    summary_rows_by_model: dict[str, list[dict]] = {}
    weights_by_model: dict[str, dict[int, np.ndarray]] = {}

    for model_name in model_names:
        suite_manifest = load_suite_manifest(results_root, model_name)
        dataset_names = sorted(suite_manifest["datasets"].keys())
        if selected_datasets is not None:
            dataset_names = [dataset for dataset in dataset_names if dataset in selected_datasets]
        model_records: list[dict] = []
        for dataset_name in dataset_names:
            model_records.extend(load_dataset_records(results_root, model_name, dataset_name))
        if not model_records:
            raise ValueError(f"No router-direction records found for `{model_name}`.")
        records_by_model[model_name] = model_records
        summary_rows_by_model[model_name] = summarize_records(model_records)
        weights_by_model[model_name] = load_router_weights(results_root, model_name)

    combined_rows: list[dict] = []
    for model_name in model_names:
        for row in summary_rows_by_model[model_name]:
            combined_rows.append({"model_name": model_name, **row})

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(combined_rows, output_root / "router_direction_summary.csv")
    plot_weight_heatmaps(weights_by_model, model_names, output_root / "router_weight_cosine_heatmaps.png")
    plot_alignment_metrics(summary_rows_by_model, model_names, output_root / "router_direction_alignment_metrics.png")
    plot_top1_expert_heatmaps(records_by_model, model_names, output_root / "router_direction_top1_expert_heatmaps.png")
    write_readme(output_root / "README.md", model_names)
    print(f"Wrote router-direction analysis to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
