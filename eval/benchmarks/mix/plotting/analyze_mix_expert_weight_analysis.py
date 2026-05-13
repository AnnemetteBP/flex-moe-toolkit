from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "expert_weight_analysis" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "expert_weight_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze standalone-expert weight-analysis outputs across public and Danish checkpoints."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--public-model", default="Flex-public-2x7B-1T")
    parser.add_argument(
        "--focus-component",
        default="layer_31_gate_up_proj",
        help="Component name to use for the main model-vs-model heatmap.",
    )
    return parser.parse_args()


SPECIALIST_DISPLAY_NAMES = {
    "Flex-public-7B-1T": "Public 7B",
    "Flex-public-2x7B-1T": "Public 2x7B",
    "Flex-public-2x7B-1T-v1": "Public 2x7B v1",
    "Flex-math-2x7B-1T": "Math",
    "Flex-news-2x7B-1T": "News",
    "Flex-pes2o-2x7B-1T": "Academic",
}

VERSION_COLORS = {"base": "#5B6C8F", "v1": "#C96B3B", "v2": "#2A9D8F"}
VERSION_ORDER = {"base": 0, "v1": 1, "v2": 2}


def model_display_name(model_name: str) -> str:
    if model_name in SPECIALIST_DISPLAY_NAMES:
        return SPECIALIST_DISPLAY_NAMES[model_name]
    match = re.match(r"Flex-danish-2x7B-(\d+B)(?:-(v\d))?$", model_name)
    if match:
        tokens = match.group(1)
        version = match.group(2)
        return f"{tokens}-{version}" if version else tokens
    return model_name


def parse_danish_variant(model_name: str) -> dict[str, object] | None:
    match = re.match(r"Flex-danish-2x7B-(\d+)B(?:-(v\d))?$", model_name)
    if not match:
        return None
    tokens_b = int(match.group(1))
    version = match.group(2) or "base"
    return {"tokens_b": tokens_b, "version": version, "version_order": VERSION_ORDER[version]}


def expert_sort_key(model_name: str) -> tuple:
    if model_name == "Flex-public-7B-1T":
        return (0, 0, 0, model_name)
    if model_name == "Flex-public-2x7B-1T":
        return (0, 1, 0, model_name)
    if model_name == "Flex-public-2x7B-1T-v1":
        return (0, 2, 0, model_name)
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


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.shape != vec_b.shape or vec_a.size == 0:
        return np.nan
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return np.nan
    return float(np.dot(vec_a, vec_b) / denom)


def load_summary(results_root: Path, model_name: str) -> pd.DataFrame:
    path = results_root / model_name / "expert_weight_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing expert weight summary for `{model_name}` at {path}.")
    frame = pd.read_csv(path)
    frame["model_name"] = model_name
    frame["model_display"] = model_display_name(model_name)
    return frame


def load_fingerprints(results_root: Path, model_name: str):
    path = results_root / model_name / "expert_weight_fingerprints.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing expert weight fingerprints for `{model_name}` at {path}.")
    return np.load(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_similarity_heatmap(model_names: list[str], model_labels: list[str], matrix: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(model_labels)), model_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(model_labels)), model_labels)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=5)
    ax.tick_params(labelsize=10.0)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                ax.text(col_idx, row_idx, "--", ha="center", va="center", fontsize=8.5, color="black")
            else:
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if abs(value) > 0.55 else "black",
                    fontweight="semibold",
                )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.90)
    colorbar.set_label("Approx. cosine similarity", fontsize=11.0, fontweight="semibold")
    colorbar.ax.tick_params(labelsize=10.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.20, right=0.93, bottom=0.20, top=0.90)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_danish_similarity_to_public(frame: pd.DataFrame, component_name: str, public_model: str, output_path: Path) -> None:
    subset = frame[(frame["component_name"] == component_name)].copy()
    subset = subset[subset["model_name"].apply(lambda value: parse_danish_variant(str(value)) is not None)]
    if subset.empty:
        return
    subset["tokens_b"] = subset["model_name"].map(lambda value: parse_danish_variant(str(value))["tokens_b"])
    subset["version"] = subset["model_name"].map(lambda value: parse_danish_variant(str(value))["version"])
    subset["version_order"] = subset["model_name"].map(lambda value: parse_danish_variant(str(value))["version_order"])
    subset = subset.sort_values(["tokens_b", "version_order", "model_name"])

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for version in ("base", "v1", "v2"):
        version_subset = subset[subset["version"] == version]
        if version_subset.empty:
            continue
        ax.plot(
            version_subset["tokens_b"],
            version_subset["approx_cosine_to_public"],
            marker="o",
            linewidth=2,
            color=VERSION_COLORS[version],
            label=version,
        )
    ax.set_title(f"Similarity to {model_display_name(public_model)} | {component_name}", fontsize=12.5, fontweight="semibold", pad=4)
    ax.set_xlabel("Danish Training Scale", fontsize=11.5, fontweight="semibold")
    ax.set_ylabel("Approx. cosine to public", fontsize=11.5, fontweight="semibold")
    ax.set_xticks(sorted(subset["tokens_b"].drop_duplicates()))
    ax.tick_params(labelsize=10.5)
    ax.grid(alpha=0.25)
    legend = ax.legend(frameon=False, fontsize=10, loc="best")
    for text in legend.get_texts():
        text.set_fontweight("semibold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.88)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_readme(path: Path, results_root: Path, output_root: Path, public_model: str, focus_component: str) -> None:
    lines = [
        "# Standalone Expert Weight Analysis",
        "",
        "## What this analysis measures",
        "This track analyzes standalone expert/public checkpoints at the parameter level.",
        "It focuses on compact component-wise fingerprints for:",
        "- token embeddings",
        "- selected MLP blocks such as `gate_up_proj`, `up_proj`, `gate_proj`, and `down_proj`",
        "",
        "The saved fingerprints are compact sampled weight sketches, so cross-model cosine values should be interpreted as approximate similarity summaries rather than exact full-tensor cosine values.",
        "",
        "## Key artifacts",
        "- `expert_weight_summary_all_models.csv`: per-model per-component parameter summaries",
        "- `expert_weight_similarity_<component>.csv`: approximate model-vs-model similarity matrix for a focus component",
        "- `expert_weight_similarity_<component>.png`: heatmap view of that matrix",
        "- `danish_similarity_to_public_<component>.png`: scaling view of Danish checkpoints relative to the chosen public baseline",
        "",
        "## How to run",
        "Suite:",
        "```bash",
        "python3 eval/benchmarks/mix/runners/run_mix_expert_weight_analysis_suite.py \\",
        "  --config eval/benchmarks/mix/configs/mix_suite_config.expert_weight_analysis.da_public.json",
        "```",
        "",
        "Plotting:",
        "```bash",
        f"python3 eval/benchmarks/mix/plotting/analyze_mix_expert_weight_analysis.py \\",
        f"  --results-root {results_root} \\",
        f"  --output-root {output_root} \\",
        f"  --public-model {public_model} \\",
        f"  --focus-component {focus_component}",
        "```",
        "",
        "## How to interpret",
        "- High similarity means two checkpoints have similar parameter structure for that component.",
        "- Similarity to the public baseline helps test whether Danish training makes the expert more distinct from public.",
        "- Shape mismatches are reported as undefined (`--`) rather than forced into misleading comparisons.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)

    model_names = sorted([path.name for path in results_root.iterdir() if path.is_dir()], key=expert_sort_key)
    if not model_names:
        raise ValueError(f"No model outputs were found under {results_root}.")

    summary_frames = [load_summary(results_root, model_name) for model_name in model_names]
    summary_frame = pd.concat(summary_frames, ignore_index=True)
    fingerprints_by_model = {model_name: load_fingerprints(results_root, model_name) for model_name in model_names}

    output_root.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output_root / "expert_weight_summary_all_models.csv", index=False)

    component_names = sorted(summary_frame["component_name"].drop_duplicates())
    comparison_rows: list[dict] = []
    for component_name in component_names:
        matrix = np.full((len(model_names), len(model_names)), np.nan, dtype=np.float32)
        for row_idx, model_name_a in enumerate(model_names):
            for col_idx, model_name_b in enumerate(model_names):
                npz_a = fingerprints_by_model[model_name_a]
                npz_b = fingerprints_by_model[model_name_b]
                if component_name not in npz_a.files or component_name not in npz_b.files:
                    continue
                fp_a = npz_a[component_name]
                fp_b = npz_b[component_name]
                matrix[row_idx, col_idx] = cosine_similarity(np.asarray(fp_a, dtype=np.float32), np.asarray(fp_b, dtype=np.float32))
                comparison_rows.append(
                    {
                        "component_name": component_name,
                        "model_left": model_name_a,
                        "model_right": model_name_b,
                        "approx_cosine": float(matrix[row_idx, col_idx]) if not np.isnan(matrix[row_idx, col_idx]) else np.nan,
                    }
                )
        if component_name == args.focus_component:
            model_labels = [model_display_name(name) for name in model_names]
            pd.DataFrame(matrix, index=model_labels, columns=model_labels).to_csv(
                output_root / f"expert_weight_similarity_{component_name}.csv"
            )
            plot_similarity_heatmap(
                model_names=model_names,
                model_labels=model_labels,
                matrix=matrix,
                title=f"Approx. Weight Similarity | {component_name}",
                output_path=output_root / f"expert_weight_similarity_{component_name}.png",
            )

            public_index = model_names.index(args.public_model) if args.public_model in model_names else None
            if public_index is not None:
                public_sim_rows = []
                for row_idx, model_name in enumerate(model_names):
                    public_sim_rows.append(
                        {
                            "model_name": model_name,
                            "model_display": model_display_name(model_name),
                            "component_name": component_name,
                            "approx_cosine_to_public": float(matrix[row_idx, public_index]) if not np.isnan(matrix[row_idx, public_index]) else np.nan,
                        }
                    )
                public_sim_frame = pd.DataFrame(public_sim_rows)
                public_sim_frame.to_csv(output_root / f"expert_weight_public_similarity_{component_name}.csv", index=False)
                plot_danish_similarity_to_public(
                    frame=public_sim_frame,
                    component_name=component_name,
                    public_model=args.public_model,
                    output_path=output_root / f"danish_similarity_to_public_{component_name}.png",
                )

    write_csv(output_root / "expert_weight_pairwise_similarity.csv", comparison_rows)
    write_readme(
        output_root / "README.md",
        results_root=results_root,
        output_root=output_root,
        public_model=args.public_model,
        focus_component=args.focus_component,
    )
    print(f"Wrote standalone expert weight analysis comparison to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
