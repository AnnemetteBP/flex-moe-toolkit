from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ACCURACY_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "expert_sweep" / "a4" / "accuracy"
DEFAULT_LATENT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "expert_sweep" / "a4" / "latent_space"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "expert_sweep"

DATASET_DISPLAY_NAMES = {
    "mkqa_en_da": "MGQA (EN/DA)",
    "gsm8k_subset": "GSM8K",
    "mbpp_subset": "MBPP",
    "pubmedqa_subset": "PubMedQA",
}

SPECIALIST_DISPLAY_NAMES = {
    "Flex-public-2x7B-1T": "Public 2x7B",
    "Flex-public-2x7B-1T-v1": "Public 2x7B v1",
    "Flex-math-2x7B-1T": "Math",
    "Flex-news-2x7B-1T": "News",
    "Flex-pes2o-2x7B-1T": "Academic",
}

VERSION_ORDER = {"base": 0, "v1": 1, "v2": 2}
VERSION_COLORS = {"base": "#5B6C8F", "v1": "#C96B3B", "v2": "#2A9D8F"}
LANGUAGE_MARKERS = {"en": "o", "da": "^", "unknown": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze standalone expert-sweep performance and latent-space outputs."
    )
    parser.add_argument("--accuracy-root", type=Path, default=DEFAULT_ACCURACY_ROOT)
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include.")
    parser.add_argument("--public-model", default="Flex-public-2x7B-1T")
    parser.add_argument("--pca-dataset", default="mkqa_en_da")
    parser.add_argument("--pca-representation", default="last", choices=("mean", "last"))
    parser.add_argument("--pca-layer", type=int, default=-1)
    parser.add_argument(
        "--pca-model-names",
        default="Flex-public-2x7B-1T,Flex-danish-2x7B-5B,Flex-danish-2x7B-10B-v2,Flex-danish-2x7B-25B-v2,Flex-danish-2x7B-55B-v2",
        help="Comma-separated model names to include in the PCA figure.",
    )
    parser.add_argument(
        "--layerwise-model-names",
        default="Flex-danish-2x7B-5B,Flex-danish-2x7B-10B-v2,Flex-danish-2x7B-25B-v2,Flex-danish-2x7B-55B-v2",
        help="Comma-separated model names to compare against the public model in layer-wise latent geometry.",
    )
    return parser.parse_args()


def dataset_display_name(dataset_name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)


def parse_model_names(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def model_display_name(model_name: str) -> str:
    if model_name in SPECIALIST_DISPLAY_NAMES:
        return SPECIALIST_DISPLAY_NAMES[model_name]
    match = re.match(r"Flex-danish-2x7B-(\d+B)(?:-(v\d))?$", model_name)
    if match:
        tokens = match.group(1)
        version = match.group(2)
        return f"{tokens}-{version}" if version else tokens
    return model_name.replace("Flex-", "").replace("-2x7B-1T", "")


def parse_danish_variant(model_name: str) -> dict[str, object] | None:
    match = re.match(r"Flex-danish-2x7B-(\d+)B(?:-(v\d))?$", model_name)
    if not match:
        return None
    tokens_b = int(match.group(1))
    version = match.group(2) or "base"
    return {
        "tokens_b": tokens_b,
        "tokens_label": f"{tokens_b}B",
        "version": version,
        "version_order": VERSION_ORDER[version],
    }


def expert_sort_key(model_name: str) -> tuple:
    if model_name == "Flex-public-2x7B-1T":
        return (0, 0, 0, model_name)
    if model_name == "Flex-public-2x7B-1T-v1":
        return (0, 1, 0, model_name)
    if model_name == "Flex-math-2x7B-1T":
        return (1, 0, 0, model_name)
    if model_name == "Flex-news-2x7B-1T":
        return (1, 1, 0, model_name)
    if model_name == "Flex-pes2o-2x7B-1T":
        return (1, 2, 0, model_name)
    parsed = parse_danish_variant(model_name)
    if parsed is not None:
        return (2, int(parsed["tokens_b"]), int(parsed["version_order"]), model_name)
    return (3, 0, 0, model_name)


def build_expert_palette(model_names: list[str], public_model: str) -> dict[str, tuple]:
    palette: dict[str, tuple] = {}
    if public_model in model_names:
        palette[public_model] = (0.1, 0.1, 0.1, 1.0)
    non_public = [name for name in model_names if name != public_model]
    if non_public:
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(non_public), 2)))
        for idx, model_name in enumerate(non_public):
            palette[model_name] = colors[idx]
    return palette


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    for source_name, by_repr in layers_by_source.items():
        for repr_name, values in by_repr.items():
            by_repr[repr_name] = sorted(set(values))
    return layers_by_source


def load_dataset_bundle(results_root: Path, model_name: str, dataset_name: str) -> dict:
    dataset_dir = results_root / model_name / dataset_name
    npz_path = dataset_dir / "prompt_latents.npz"
    metadata_path = dataset_dir / "metadata.jsonl"
    manifest_path = dataset_dir / "run_manifest.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing latent file: {npz_path}")
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "npz": np.load(npz_path),
        "metadata": load_jsonl(metadata_path),
        "run_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
    }


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return centered @ basis


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def mean_squared_radius(vectors: np.ndarray) -> float:
    if vectors.shape[0] == 0:
        return 0.0
    centroid = vectors.mean(axis=0, keepdims=True)
    distances = np.sum((vectors - centroid) ** 2, axis=1)
    return float(distances.mean())


def centroid_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.linalg.norm(vec_a.mean(axis=0) - vec_b.mean(axis=0)))


def build_language_groups(metadata: list[dict]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(metadata):
        groups.setdefault(str(row.get("language", "unknown")), []).append(idx)
    return {language: np.asarray(indices, dtype=int) for language, indices in groups.items()}


def load_accuracy_frame(accuracy_root: Path, selected_datasets: set[str] | None) -> pd.DataFrame:
    rows: list[dict] = []
    for overview_path in sorted(accuracy_root.glob("*/expert_eval_overview.csv")):
        frame = pd.read_csv(overview_path)
        rows.extend(frame.to_dict(orient="records"))
    if not rows:
        raise ValueError(f"No expert eval overview files were found under {accuracy_root}.")
    result = pd.DataFrame(rows)
    if selected_datasets is not None:
        result = result[result["dataset_name"].isin(selected_datasets)].copy()
    result["dataset_display"] = result["dataset_name"].map(dataset_display_name)
    result["model_display"] = result["model_name"].map(model_display_name)
    return result


def build_danish_scaling_frame(accuracy_frame: pd.DataFrame, public_model: str) -> pd.DataFrame:
    rows: list[dict] = []
    public_mkqa = accuracy_frame[
        (accuracy_frame["model_name"] == public_model) & (accuracy_frame["dataset_name"] == "mkqa_en_da")
    ]
    public_da_accuracy = float(public_mkqa.iloc[0]["da_accuracy"]) if not public_mkqa.empty else np.nan
    public_non_da = accuracy_frame[
        (accuracy_frame["model_name"] == public_model) & (accuracy_frame["dataset_name"] != "mkqa_en_da")
    ]
    public_non_da_mean_score = (
        float(public_non_da["mean_score"].mean()) if not public_non_da.empty else np.nan
    )

    for model_name in sorted(accuracy_frame["model_name"].drop_duplicates()):
        parsed = parse_danish_variant(model_name)
        if parsed is None:
            continue
        mkqa_row = accuracy_frame[
            (accuracy_frame["model_name"] == model_name) & (accuracy_frame["dataset_name"] == "mkqa_en_da")
        ]
        if mkqa_row.empty:
            continue
        non_da_rows = accuracy_frame[
            (accuracy_frame["model_name"] == model_name) & (accuracy_frame["dataset_name"] != "mkqa_en_da")
        ]
        rows.append(
            {
                "model_name": model_name,
                "model_display": model_display_name(model_name),
                "tokens_b": parsed["tokens_b"],
                "tokens_label": parsed["tokens_label"],
                "version": parsed["version"],
                "version_order": parsed["version_order"],
                "da_accuracy": float(mkqa_row.iloc[0]["da_accuracy"]),
                "da_mean_score": float(mkqa_row.iloc[0]["da_mean_score"]),
                "en_accuracy": float(mkqa_row.iloc[0]["en_accuracy"]),
                "en_mean_score": float(mkqa_row.iloc[0]["en_mean_score"]),
                "non_da_mean_score": float(non_da_rows["mean_score"].mean()) if not non_da_rows.empty else np.nan,
                "non_da_mean_f1": float(non_da_rows["mean_token_f1"].mean()) if not non_da_rows.empty else np.nan,
                "delta_da_accuracy_vs_public": float(mkqa_row.iloc[0]["da_accuracy"]) - public_da_accuracy,
                "delta_non_da_mean_score_vs_public": (
                    float(non_da_rows["mean_score"].mean()) - public_non_da_mean_score if not non_da_rows.empty else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["tokens_b", "version_order", "model_name"]).reset_index(drop=True)
    return result


def plot_performance_scaling(
    scaling_frame: pd.DataFrame,
    accuracy_frame: pd.DataFrame,
    public_model: str,
    output_path: Path,
) -> None:
    if scaling_frame.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    x_values = sorted(scaling_frame["tokens_b"].drop_duplicates())
    x_labels = [f"{value}B" for value in x_values]

    public_mkqa = accuracy_frame[
        (accuracy_frame["model_name"] == public_model) & (accuracy_frame["dataset_name"] == "mkqa_en_da")
    ]
    public_da_accuracy = float(public_mkqa.iloc[0]["da_accuracy"]) if not public_mkqa.empty else np.nan
    public_non_da = accuracy_frame[
        (accuracy_frame["model_name"] == public_model) & (accuracy_frame["dataset_name"] != "mkqa_en_da")
    ]
    public_non_da_mean_score = float(public_non_da["mean_score"].mean()) if not public_non_da.empty else np.nan

    for version in ("base", "v1", "v2"):
        subset = scaling_frame[scaling_frame["version"] == version].sort_values("tokens_b")
        if subset.empty:
            continue
        axes[0].plot(
            subset["tokens_b"],
            subset["da_accuracy"],
            marker="o",
            linewidth=2,
            color=VERSION_COLORS[version],
            label=version,
        )
        axes[1].plot(
            subset["tokens_b"],
            subset["non_da_mean_score"],
            marker="o",
            linewidth=2,
            color=VERSION_COLORS[version],
            label=version,
        )

    if not np.isnan(public_da_accuracy):
        axes[0].axhline(public_da_accuracy, color="#444444", linestyle="--", linewidth=1.6, label="Public")
    if not np.isnan(public_non_da_mean_score):
        axes[1].axhline(public_non_da_mean_score, color="#444444", linestyle="--", linewidth=1.6, label="Public")

    axes[0].set_title("Danish Accuracy on MGQA (DA)", fontsize=12.5, fontweight="semibold", pad=4)
    axes[1].set_title("Mean Non-Danish Score", fontsize=12.5, fontweight="semibold", pad=4)
    for ax in axes:
        ax.set_xlabel("Danish Training Scale", fontsize=11.5, fontweight="semibold")
        ax.set_xticks(x_values, x_labels)
        ax.tick_params(labelsize=10.5)
        ax.grid(alpha=0.22)
        legend = ax.legend(frameon=False, fontsize=10, loc="best")
        for text in legend.get_texts():
            text.set_fontweight("semibold")
    axes[0].set_ylabel("Accuracy", fontsize=11.5, fontweight="semibold")
    axes[1].set_ylabel("Mean Score", fontsize=11.5, fontweight="semibold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.88, wspace=0.22)
    fig.suptitle("Standalone Expert Sweep: Danish Scaling", fontsize=14.5, fontweight="bold", y=0.96)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_accuracy_overview(accuracy_frame: pd.DataFrame, output_path: Path) -> None:
    if accuracy_frame.empty:
        return
    column_specs: list[tuple[str, str, str]] = []
    available_datasets = set(accuracy_frame["dataset_name"])
    if "mkqa_en_da" in available_datasets:
        column_specs.extend(
            [
                ("mkqa_en_da", "en_accuracy", "MGQA EN"),
                ("mkqa_en_da", "da_accuracy", "MGQA DA"),
            ]
        )
    for dataset_name in ("gsm8k_subset", "mbpp_subset", "pubmedqa_subset"):
        if dataset_name in available_datasets:
            column_specs.append((dataset_name, "accuracy", dataset_display_name(dataset_name)))
    model_names = sorted(accuracy_frame["model_name"].drop_duplicates(), key=expert_sort_key)
    model_labels = [model_display_name(name) for name in model_names]

    matrix = np.full((len(model_names), len(column_specs)), np.nan, dtype=float)
    for row_idx, model_name in enumerate(model_names):
        for col_idx, (dataset_name, metric_name, _display_name) in enumerate(column_specs):
            subset = accuracy_frame[
                (accuracy_frame["model_name"] == model_name) & (accuracy_frame["dataset_name"] == dataset_name)
            ]
            if not subset.empty:
                value = subset.iloc[0].get(metric_name)
                matrix[row_idx, col_idx] = float(value) if pd.notna(value) else np.nan

    fig, ax = plt.subplots(figsize=(8.2, max(6.0, 0.38 * len(model_names) + 1.2)))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=max(1.0, float(np.nanmax(matrix))))
    ax.set_xticks(range(len(column_specs)), [display_name for _dataset, _metric, display_name in column_specs])
    ax.set_yticks(range(len(model_names)), model_labels, fontweight="semibold")
    ax.set_xlabel("Dataset", fontsize=11.5, fontweight="semibold")
    ax.set_ylabel("Model", fontsize=11.5, fontweight="semibold")
    ax.set_title("Accuracy Overview", fontsize=13, fontweight="bold", pad=5)
    ax.tick_params(axis="x", labelsize=10.0, rotation=18)
    ax.tick_params(axis="y", labelsize=10.0)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.6,
                color="white" if value < 0.55 else "black",
                fontweight="semibold",
            )

    for col_idx in range(matrix.shape[1]):
        column = matrix[:, col_idx]
        valid_indices = np.where(~np.isnan(column))[0]
        if valid_indices.size == 0:
            continue
        sorted_indices = valid_indices[np.argsort(column[valid_indices])[::-1]]
        top_idx = int(sorted_indices[0])
        ax.add_patch(
            Rectangle(
                (col_idx - 0.5, top_idx - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
            )
        )
        if sorted_indices.size >= 2:
            second_idx = int(sorted_indices[1])
            ax.add_patch(
                Rectangle(
                    (col_idx - 0.5, second_idx - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.8,
                    linestyle="--",
                )
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.92)
    colorbar.set_label("Accuracy", fontsize=11.0, fontweight="semibold")
    colorbar.ax.tick_params(labelsize=10.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.16, right=0.93, bottom=0.10, top=0.91)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def build_latent_geometry_rows(
    latent_root: Path,
    datasets: list[str],
    public_model: str,
) -> list[dict]:
    available_models = sorted(path.name for path in latent_root.iterdir() if path.is_dir())
    if public_model not in available_models:
        raise ValueError(f"Public model `{public_model}` was not found under {latent_root}.")

    rows: list[dict] = []
    public_bundles = {
        dataset_name: load_dataset_bundle(latent_root, public_model, dataset_name)
        for dataset_name in datasets
        if (latent_root / public_model / dataset_name / "prompt_latents.npz").exists()
    }
    for model_name in available_models:
        for dataset_name in datasets:
            dataset_dir = latent_root / model_name / dataset_name
            if not (dataset_dir / "prompt_latents.npz").exists():
                continue
            bundle = load_dataset_bundle(latent_root, model_name, dataset_name)
            layer_keys = parse_layer_keys(bundle["npz"])
            if "hidden_state" not in layer_keys:
                continue
            public_bundle = public_bundles.get(dataset_name)
            language_groups = build_language_groups(bundle["metadata"])

            for representation in sorted(layer_keys["hidden_state"]):
                for layer_idx in layer_keys["hidden_state"][representation]:
                    key = f"hidden_state_layer_{layer_idx}_{representation}"
                    vectors = np.asarray(bundle["npz"][key], dtype=np.float32)
                    row = {
                        "model_name": model_name,
                        "model_display": model_display_name(model_name),
                        "dataset_name": dataset_name,
                        "dataset_display": dataset_display_name(dataset_name),
                        "representation_source": "hidden_state",
                        "representation": representation,
                        "layer": layer_idx,
                        "within_variance": mean_squared_radius(vectors),
                        "num_examples": int(vectors.shape[0]),
                    }
                    if public_bundle is not None:
                        public_vectors = np.asarray(public_bundle["npz"][key], dtype=np.float32)
                        public_var = mean_squared_radius(public_vectors)
                        dist = centroid_distance(vectors, public_vectors)
                        row["distance_to_public"] = dist
                        row["cosine_to_public"] = cosine_similarity(vectors.mean(axis=0), public_vectors.mean(axis=0))
                        row["separation_ratio_to_public"] = dist / np.sqrt(max(0.5 * (public_var + row["within_variance"]), 1e-9))
                    else:
                        row["distance_to_public"] = np.nan
                        row["cosine_to_public"] = np.nan
                        row["separation_ratio_to_public"] = np.nan

                    if "en" in language_groups and "da" in language_groups:
                        en_vectors = vectors[language_groups["en"]]
                        da_vectors = vectors[language_groups["da"]]
                        lang_dist = centroid_distance(en_vectors, da_vectors)
                        row["en_da_centroid_distance"] = lang_dist
                        row["en_da_cosine"] = cosine_similarity(en_vectors.mean(axis=0), da_vectors.mean(axis=0))
                        row["en_da_separation_ratio"] = lang_dist / np.sqrt(
                            max(0.5 * (mean_squared_radius(en_vectors) + mean_squared_radius(da_vectors)), 1e-9)
                        )
                    else:
                        row["en_da_centroid_distance"] = np.nan
                        row["en_da_cosine"] = np.nan
                        row["en_da_separation_ratio"] = np.nan
                    rows.append(row)
    return rows


def plot_latent_pca(
    latent_root: Path,
    model_names: list[str],
    dataset_name: str,
    representation: str,
    layer_idx: int,
    output_path: Path,
) -> None:
    points = []
    metadata_rows = []
    model_colors = build_expert_palette(model_names, public_model=model_names[0])

    for model_name in model_names:
        bundle = load_dataset_bundle(latent_root, model_name, dataset_name)
        key = f"hidden_state_layer_{layer_idx}_{representation}"
        vectors = np.asarray(bundle["npz"][key], dtype=np.float32)
        points.append(vectors)
        for row in bundle["metadata"]:
            metadata_rows.append({"model_name": model_name, "language": row.get("language", "unknown")})

    projection = pca_2d(np.concatenate(points, axis=0))
    fig, ax = plt.subplots(figsize=(8.4, 6.6))

    for point, meta in zip(projection, metadata_rows):
        ax.scatter(
            float(point[0]),
            float(point[1]),
            color=model_colors[meta["model_name"]],
            marker=LANGUAGE_MARKERS.get(meta["language"], LANGUAGE_MARKERS["unknown"]),
            alpha=0.75,
            s=28,
        )

    ax.set_title(
        f"{dataset_display_name(dataset_name)} | Hidden state | layer {layer_idx}",
        fontsize=13,
        pad=5,
        fontweight="bold",
    )
    ax.set_xlabel("PC1", fontsize=12, fontweight="semibold")
    ax.set_ylabel("PC2", fontsize=12, fontweight="semibold")
    ax.tick_params(labelsize=10.5)

    legend_handles = []
    for model_name in model_names:
        legend_handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor=model_colors[model_name],
                markeredgecolor=model_colors[model_name],
                markersize=7,
                label=model_display_name(model_name),
            )
        )
    for language, marker in (("en", "o"), ("da", "^")):
        legend_handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=marker,
                markerfacecolor="#555555",
                markeredgecolor="#555555",
                markersize=7,
                label=language.upper(),
            )
        )
    legend = fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=10.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(4, max(1, len(legend_handles))),
    )
    for text in legend.get_texts():
        text.set_fontweight("semibold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.86)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_distance_to_public_by_layer(
    geometry_frame: pd.DataFrame,
    dataset_name: str,
    model_names: list[str],
    representation: str,
    public_model: str,
    output_path: Path,
) -> None:
    subset = geometry_frame[
        (geometry_frame["dataset_name"] == dataset_name) & (geometry_frame["representation"] == representation)
    ].copy()
    if subset.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    metrics = [
        ("distance_to_public", "Centroid Distance to Public"),
        ("separation_ratio_to_public", "Separation Ratio to Public"),
    ]
    model_colors = build_expert_palette(model_names, public_model=public_model)

    for ax, (metric_key, title) in zip(axes, metrics):
        for model_name in model_names:
            model_subset = subset[subset["model_name"] == model_name].sort_values("layer")
            if model_subset.empty:
                continue
            ax.plot(
                model_subset["layer"],
                model_subset[metric_key],
                marker="o",
                linewidth=2,
                color=model_colors[model_name],
                label=model_display_name(model_name),
            )
        ax.set_title(title, fontsize=12.5, fontweight="semibold", pad=4)
        ax.set_xlabel("Layer", fontsize=11.5, fontweight="semibold")
        ax.tick_params(labelsize=10.5)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Value", fontsize=11.5, fontweight="semibold")
    legend = axes[1].legend(frameon=False, fontsize=14, loc="best")
    for text in legend.get_texts():
        text.set_fontweight("semibold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.88, wspace=0.22)
    fig.suptitle(
        f"Layer-wise Distance to Public | {dataset_display_name(dataset_name)}",
        fontsize=14.5,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def write_readme(path: Path) -> None:
    lines = [
        "# Standalone Expert Sweep Analysis",
        "",
        "Artifacts:",
        "- `expert_eval_summary.csv`",
        "- `expert_danish_scaling_summary.csv`",
        "- `expert_latent_geometry_summary.csv`",
        "- `expert_danish_scaling.png`",
        "- `expert_latent_pca_*.png`",
        "- `expert_distance_to_public_by_layer_*.png`",
        "",
        "Interpretation guide:",
        "- The performance summaries show whether Danish variants improve on Danish while preserving broader capability.",
        "- `distance_to_public` and `cosine_to_public` quantify how far each expert drifts from the public baseline.",
        "- `en_da_*` metrics on MGQA show whether a model internally separates English and Danish more strongly.",
        "- The PCA figure is a visual guide; the geometry CSV is the more stable quantitative summary.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}

    accuracy_frame = load_accuracy_frame(args.accuracy_root, selected_datasets)
    scaling_frame = build_danish_scaling_frame(accuracy_frame, args.public_model)

    suite_manifests = sorted(args.latent_root.glob("*/latent_space_suite_manifest.json"))
    if not suite_manifests:
        raise ValueError(f"No latent-space suite manifests were found under {args.latent_root}.")

    available_datasets = None
    for manifest_path in suite_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = set(manifest.get("datasets", {}).keys())
        available_datasets = datasets if available_datasets is None else (available_datasets | datasets)
    latent_datasets = sorted(available_datasets or set())
    if selected_datasets is not None:
        latent_datasets = [name for name in latent_datasets if name in selected_datasets]

    geometry_rows = build_latent_geometry_rows(args.latent_root, latent_datasets, args.public_model)
    geometry_frame = pd.DataFrame(geometry_rows)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    accuracy_frame.to_csv(output_root / "expert_eval_summary.csv", index=False)
    scaling_frame.to_csv(output_root / "expert_danish_scaling_summary.csv", index=False)
    geometry_frame.to_csv(output_root / "expert_latent_geometry_summary.csv", index=False)
    plot_accuracy_overview(accuracy_frame, output_root / "expert_accuracy_overview.png")

    plot_performance_scaling(
        scaling_frame=scaling_frame,
        accuracy_frame=accuracy_frame,
        public_model=args.public_model,
        output_path=output_root / "expert_danish_scaling.png",
    )

    pca_model_names = parse_model_names(args.pca_model_names)
    if args.pca_dataset in latent_datasets:
        bundle = load_dataset_bundle(args.latent_root, pca_model_names[0], args.pca_dataset)
        layer_keys = parse_layer_keys(bundle["npz"])
        available_layers = sorted(layer_keys["hidden_state"][args.pca_representation])
        pca_layer = available_layers[args.pca_layer] if args.pca_layer < 0 else args.pca_layer
        plot_latent_pca(
            latent_root=args.latent_root,
            model_names=pca_model_names,
            dataset_name=args.pca_dataset,
            representation=args.pca_representation,
            layer_idx=pca_layer,
            output_path=output_root / (
                f"expert_latent_pca_{args.pca_dataset}_layer_{pca_layer}_{args.pca_representation}.png"
            ),
        )

    layerwise_model_names = parse_model_names(args.layerwise_model_names)
    if not geometry_frame.empty:
        plot_distance_to_public_by_layer(
            geometry_frame=geometry_frame,
            dataset_name=args.pca_dataset,
            model_names=layerwise_model_names,
            representation=args.pca_representation,
            public_model=args.public_model,
            output_path=output_root / (
                f"expert_distance_to_public_by_layer_{args.pca_dataset}_{args.pca_representation}.png"
            ),
        )

    write_readme(output_root / "README.md")
    print(f"Wrote expert-sweep analysis to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
