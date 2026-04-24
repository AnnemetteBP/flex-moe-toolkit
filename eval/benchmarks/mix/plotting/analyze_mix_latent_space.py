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


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "focused" / "55b_pair" / "latent_space"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_latent_space"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze compact latent-space captures for the 55B FlexOlmo pair."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--model-names",
        default="FlexOlmo-8x7B-1T-a4-55B-v2,FlexOlmo-8x7B-1T-a4-55B-v2-rt",
        help="Comma-separated model names to compare.",
    )
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include.")
    parser.add_argument("--pca-dataset", default="mkqa_en_da")
    parser.add_argument("--representation-source", default="pre_router", choices=("hidden_state", "pre_router"))
    parser.add_argument("--pca-representation", default="last", choices=("mean", "last"))
    parser.add_argument("--pca-layer", type=int, default=-1)
    return parser.parse_args()


def parse_model_names(raw_value: str) -> list[str]:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(names) < 2:
        raise ValueError("Provide at least two model names.")
    return names


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def parse_layer_keys(npz_data) -> dict[str, dict[str, list[int]]]:
    layers_by_source: dict[str, dict[str, list[int]]] = {}
    for key in npz_data.files:
        if "_layer_" not in key:
            continue
        source_name, remainder = key.split("_layer_", 1)
        if "_" not in remainder:
            continue
        layer_str, repr_name = remainder.split("_", 1)
        layers_by_source.setdefault(source_name, {}).setdefault(repr_name, []).append(int(layer_str))
    for source_name in list(layers_by_source):
        for repr_name in list(layers_by_source[source_name]):
            layers_by_source[source_name][repr_name] = sorted(set(layers_by_source[source_name][repr_name]))
    return layers_by_source


def load_dataset_bundle(results_root: Path, model_name: str, dataset_name: str) -> dict:
    dataset_dir = results_root / model_name / dataset_name
    npz_path = dataset_dir / "prompt_latents.npz"
    metadata_path = dataset_dir / "metadata.jsonl"
    run_manifest_path = dataset_dir / "run_manifest.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing latent file: {npz_path}")
    metadata = load_jsonl(metadata_path)
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "npz": np.load(npz_path),
        "metadata": metadata,
        "run_manifest": json.loads(run_manifest_path.read_text(encoding="utf-8")),
    }


def build_language_groups(metadata: list[dict]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(metadata):
        groups.setdefault(str(row.get("language", "unknown")), []).append(idx)
    return {language: np.asarray(indices, dtype=int) for language, indices in groups.items()}


def summarize_similarity_rows(
    bundles_by_model: dict[str, dict],
    model_names: list[str],
    dataset_name: str,
) -> list[dict]:
    rows: list[dict] = []
    base_bundle = bundles_by_model[model_names[0]]
    comparison_bundle = bundles_by_model[model_names[1]]
    base_npz = base_bundle["npz"]
    comparison_npz = comparison_bundle["npz"]
    layer_keys = parse_layer_keys(base_npz)
    base_groups = build_language_groups(base_bundle["metadata"])
    comparison_groups = build_language_groups(comparison_bundle["metadata"])

    for source_name, by_repr in sorted(layer_keys.items()):
        for repr_name, layer_ids in sorted(by_repr.items()):
            for layer_idx in layer_ids:
                key = f"{source_name}_layer_{layer_idx}_{repr_name}"
                base_vectors = np.asarray(base_npz[key], dtype=np.float32)
                comparison_vectors = np.asarray(comparison_npz[key], dtype=np.float32)
                base_mean = base_vectors.mean(axis=0)
                comparison_mean = comparison_vectors.mean(axis=0)
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "representation_source": source_name,
                        "representation": repr_name,
                        "layer": layer_idx,
                        "group": "all",
                        "metric": "cross_model_cosine",
                        "value": cosine_similarity(base_mean, comparison_mean),
                        "model_a": model_names[0],
                        "model_b": model_names[1],
                    }
                )

                for language in sorted(set(base_groups) & set(comparison_groups)):
                    base_lang = base_vectors[base_groups[language]]
                    comparison_lang = comparison_vectors[comparison_groups[language]]
                    rows.append(
                        {
                            "dataset_name": dataset_name,
                            "representation_source": source_name,
                            "representation": repr_name,
                            "layer": layer_idx,
                            "group": language,
                            "metric": "cross_model_cosine",
                            "value": cosine_similarity(base_lang.mean(axis=0), comparison_lang.mean(axis=0)),
                            "model_a": model_names[0],
                            "model_b": model_names[1],
                        }
                    )

                if len(base_groups) >= 2:
                    languages = sorted(base_groups)
                    if len(languages) >= 2:
                        lang_a, lang_b = languages[:2]
                        rows.append(
                            {
                                "dataset_name": dataset_name,
                                "representation_source": source_name,
                                "representation": repr_name,
                                "layer": layer_idx,
                                "group": f"{model_names[0]}:{lang_a}_vs_{lang_b}",
                                "metric": "within_model_language_cosine",
                                "value": cosine_similarity(
                                    base_vectors[base_groups[lang_a]].mean(axis=0),
                                    base_vectors[base_groups[lang_b]].mean(axis=0),
                                ),
                                "model_a": model_names[0],
                                "model_b": model_names[0],
                            }
                        )
                if len(comparison_groups) >= 2:
                    languages = sorted(comparison_groups)
                    if len(languages) >= 2:
                        lang_a, lang_b = languages[:2]
                        rows.append(
                            {
                                "dataset_name": dataset_name,
                                "representation_source": source_name,
                                "representation": repr_name,
                                "layer": layer_idx,
                                "group": f"{model_names[1]}:{lang_a}_vs_{lang_b}",
                                "metric": "within_model_language_cosine",
                                "value": cosine_similarity(
                                    comparison_vectors[comparison_groups[lang_a]].mean(axis=0),
                                    comparison_vectors[comparison_groups[lang_b]].mean(axis=0),
                                ),
                                "model_a": model_names[1],
                                "model_b": model_names[1],
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


def plot_similarity_rows(rows: list[dict], output_path: Path, model_names: list[str]) -> None:
    datasets = sorted({row["dataset_name"] for row in rows})
    representations = sorted({row["representation"] for row in rows})
    sources = sorted({row["representation_source"] for row in rows})
    fig, axes = plt.subplots(
        len(datasets),
        len(representations) * len(sources),
        figsize=(6 * len(representations) * len(sources), 4 * len(datasets)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, dataset_name in enumerate(datasets):
        for source_idx, source_name in enumerate(sources):
            for repr_idx, repr_name in enumerate(representations):
                col_idx = source_idx * len(representations) + repr_idx
                ax = axes[row_idx][col_idx]
                subset = [
                    row for row in rows
                    if row["dataset_name"] == dataset_name
                    and row["representation_source"] == source_name
                    and row["representation"] == repr_name
                    and row["metric"] == "cross_model_cosine"
                ]
                groups = sorted({row["group"] for row in subset})
                for group in groups:
                    group_rows = sorted(
                        (row for row in subset if row["group"] == group),
                        key=lambda item: int(item["layer"]),
                    )
                    ax.plot(
                        [int(row["layer"]) for row in group_rows],
                        [float(row["value"]) for row in group_rows],
                        marker="o",
                        label=group,
                    )
                ax.set_title(f"{dataset_name} | {source_name} | {repr_name}")
                ax.set_xlabel("Layer")
                ax.set_ylabel("Cosine")
                ax.set_ylim(-0.05, 1.05)
                if groups:
                    ax.legend(frameon=False, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Latent Similarity: {model_names[0]} vs {model_names[1]}", fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return centered @ basis


def plot_pca(
    bundles_by_model: dict[str, dict],
    model_names: list[str],
    dataset_name: str,
    representation_source: str,
    representation: str,
    layer_idx: int,
    output_path: Path,
) -> None:
    points = []
    metadata_rows = []
    for model_name in model_names:
        bundle = bundles_by_model[model_name]
        npz = bundle["npz"]
        key = f"{representation_source}_layer_{layer_idx}_{representation}"
        if key not in npz.files:
            raise ValueError(f"Layer key `{key}` was not found for dataset `{dataset_name}`.")
        vectors = np.asarray(npz[key], dtype=np.float32)
        points.append(vectors)
        for row in bundle["metadata"]:
            metadata_rows.append(
                {
                    "model_name": model_name,
                    "language": row.get("language", "unknown"),
                }
            )
    matrix = np.concatenate(points, axis=0)
    projection = pca_2d(matrix)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    colors = {"en": "#1b6ca8", "da": "#d95f02", "unknown": "#555555"}
    markers = {model_names[0]: "o", model_names[1]: "^"}

    for point, meta in zip(projection, metadata_rows):
        ax.scatter(
            float(point[0]),
            float(point[1]),
            color=colors.get(meta["language"], colors["unknown"]),
            marker=markers.get(meta["model_name"], "o"),
            alpha=0.75,
            s=26,
        )

    legend_labels = []
    for model_name in model_names:
        for language in sorted({row["language"] for row in metadata_rows}):
            legend_labels.append((model_name, language))

    for model_name, language in legend_labels:
        ax.scatter(
            [],
            [],
            color=colors.get(language, colors["unknown"]),
            marker=markers.get(model_name, "o"),
            label=f"{model_name} | {language}",
        )

    ax.set_title(f"PCA: {dataset_name} | {representation_source} | layer {layer_idx} | {representation}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pca_grid(
    bundles_by_dataset: dict[str, dict[str, dict]],
    model_names: list[str],
    datasets: list[str],
    representation_sources: list[str],
    representation: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(datasets),
        len(representation_sources),
        figsize=(7 * len(representation_sources), 5 * len(datasets)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {"en": "#1b6ca8", "da": "#d95f02", "unknown": "#555555"}
    markers = {model_names[0]: "o", model_names[1]: "^"}

    for row_idx, dataset_name in enumerate(datasets):
        bundles_by_model = bundles_by_dataset[dataset_name]
        first_bundle = bundles_by_model[model_names[0]]
        layer_keys = parse_layer_keys(first_bundle["npz"])
        for col_idx, representation_source in enumerate(representation_sources):
            ax = axes[row_idx][col_idx]
            available_layers = sorted(layer_keys[representation_source][representation])
            layer_idx = available_layers[-1]

            points = []
            metadata_rows = []
            for model_name in model_names:
                bundle = bundles_by_model[model_name]
                npz = bundle["npz"]
                key = f"{representation_source}_layer_{layer_idx}_{representation}"
                vectors = np.asarray(npz[key], dtype=np.float32)
                points.append(vectors)
                for row in bundle["metadata"]:
                    metadata_rows.append(
                        {
                            "model_name": model_name,
                            "language": row.get("language", "unknown"),
                        }
                    )

            matrix = np.concatenate(points, axis=0)
            projection = pca_2d(matrix)
            for point, meta in zip(projection, metadata_rows):
                ax.scatter(
                    float(point[0]),
                    float(point[1]),
                    color=colors.get(meta["language"], colors["unknown"]),
                    marker=markers.get(meta["model_name"], "o"),
                    alpha=0.75,
                    s=22,
                )
            ax.set_title(f"{dataset_name} | {representation_source} | layer {layer_idx}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

    legend_labels = []
    all_languages = sorted(
        {
            row.get("language", "unknown")
            for dataset_name in datasets
            for model_name in model_names
            for row in bundles_by_dataset[dataset_name][model_name]["metadata"]
        }
    )
    for model_name in model_names:
        for language in all_languages:
            legend_labels.append((model_name, language))

    for model_name, language in legend_labels:
        axes[0][0].scatter(
            [],
            [],
            color=colors.get(language, colors["unknown"]),
            marker=markers.get(model_name, "o"),
            label=f"{model_name} | {language}",
        )
    axes[0][0].legend(frameon=False, fontsize=8, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_readme(
    path: Path,
    rows: list[dict],
    model_names: list[str],
    pca_dataset: str,
    representation_source: str,
    pca_representation: str,
    pca_layer: int,
) -> None:
    lines = [
        "# 55B Latent Space Comparison",
        "",
        f"Compared models:",
        f"- `{model_names[0]}`",
        f"- `{model_names[1]}`",
        f"- PCA source: `{representation_source}`",
        "",
        "Artifacts:",
        "- `latent_space_similarity_summary.csv`",
        "- `latent_space_similarity_plot.png`",
        f"- `latent_space_pca_grid_{pca_representation}.png`",
        f"- `latent_space_pca_{pca_dataset}_{representation_source}_layer_{pca_layer}_{pca_representation}.png`",
        "",
        "Interpretation guide:",
        "- `cross_model_cosine` close to 1 means the two models occupy very similar centroid directions.",
        "- lower cosine on a dataset or language suggests stronger representational divergence there.",
        "- `within_model_language_cosine` helps show whether English and Danish remain entangled or separate.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    model_names = parse_model_names(args.model_names)
    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}

    available_datasets = None
    for model_name in model_names:
        suite_manifest_path = results_root / model_name / "latent_space_suite_manifest.json"
        if not suite_manifest_path.exists():
            raise FileNotFoundError(
                f"Missing latent-space suite manifest for `{model_name}` at {suite_manifest_path}. "
                "Run the latent-space capture suite first."
            )
        suite_manifest = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
        model_datasets = set(suite_manifest["datasets"].keys())
        available_datasets = model_datasets if available_datasets is None else (available_datasets & model_datasets)
    datasets = sorted(available_datasets or set())
    if selected_datasets is not None:
        datasets = [dataset for dataset in datasets if dataset in selected_datasets]
    if not datasets:
        raise ValueError("No common latent-space datasets were found for the selected models.")

    all_rows: list[dict] = []
    bundles_for_pca = None
    bundles_by_dataset: dict[str, dict[str, dict]] = {}
    for dataset_name in datasets:
        bundles_by_model = {
            model_name: load_dataset_bundle(results_root=results_root, model_name=model_name, dataset_name=dataset_name)
            for model_name in model_names
        }
        bundles_by_dataset[dataset_name] = bundles_by_model
        all_rows.extend(summarize_similarity_rows(bundles_by_model, model_names=model_names, dataset_name=dataset_name))
        if dataset_name == args.pca_dataset:
            bundles_for_pca = bundles_by_model

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "latent_space_similarity_summary.csv"
    write_csv(all_rows, csv_path)
    plot_similarity_rows(all_rows, output_root / "latent_space_similarity_plot.png", model_names=model_names)
    grid_sources = [source for source in ("pre_router", "hidden_state") if any(row["representation_source"] == source for row in all_rows)]
    if grid_sources and datasets:
        plot_pca_grid(
            bundles_by_dataset=bundles_by_dataset,
            model_names=model_names,
            datasets=datasets,
            representation_sources=grid_sources,
            representation=args.pca_representation,
            output_path=output_root / f"latent_space_pca_grid_{args.pca_representation}.png",
        )

    if bundles_for_pca is not None:
        pca_layer = args.pca_layer
        first_bundle = bundles_for_pca[model_names[0]]
        layer_keys = parse_layer_keys(first_bundle["npz"])
        if args.representation_source not in layer_keys:
            available_sources = ", ".join(sorted(layer_keys))
            raise ValueError(
                f"Representation source `{args.representation_source}` not available. Choices: {available_sources}"
            )
        if args.pca_representation not in layer_keys[args.representation_source]:
            available_reprs = ", ".join(sorted(layer_keys[args.representation_source]))
            raise ValueError(
                f"Representation `{args.pca_representation}` not available for `{args.representation_source}`. "
                f"Choices: {available_reprs}"
            )
        if pca_layer < 0:
            available_layers = sorted(layer_keys[args.representation_source][args.pca_representation])
            pca_layer = available_layers[pca_layer]
        plot_pca(
            bundles_by_model=bundles_for_pca,
            model_names=model_names,
            dataset_name=args.pca_dataset,
            representation_source=args.representation_source,
            representation=args.pca_representation,
            layer_idx=pca_layer,
            output_path=output_root / (
                f"latent_space_pca_{args.pca_dataset}_{args.representation_source}_"
                f"layer_{pca_layer}_{args.pca_representation}.png"
            ),
        )

    write_readme(
        output_root / "README.md",
        rows=all_rows,
        model_names=model_names,
        pca_dataset=args.pca_dataset,
        representation_source=args.representation_source,
        pca_representation=args.pca_representation,
        pca_layer=pca_layer if bundles_for_pca is not None else args.pca_layer,
    )
    print(f"Wrote latent-space analysis to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
