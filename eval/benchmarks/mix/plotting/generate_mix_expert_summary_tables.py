from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "expert_sweep"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "expert_sweep_tables"

DATASET_DISPLAY_NAMES = {
    "mkqa_en_da": "MGQA (EN/DA)",
    "gsm8k_subset": "GSM8K",
    "mbpp_subset": "MBPP",
    "pubmedqa_subset": "PubMedQA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CSV and LaTeX summary tables for standalone expert-sweep analyses."
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--public-model", default="Flex-public-2x7B-1T")
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, output_root: Path, stem: str, caption: str, label: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / f"{stem}.csv", index=False)
    latex = df.to_latex(
        index=False,
        float_format=lambda value: f"{value:.3f}" if isinstance(value, float) else str(value),
        escape=False,
        caption=caption,
        label=label,
    )
    (output_root / f"{stem}.tex").write_text(latex, encoding="utf-8")


def build_performance_overview(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[
        [
            "model_display",
            "dataset_name",
            "accuracy",
            "mean_score",
            "mean_token_f1",
            "en_accuracy",
            "da_accuracy",
        ]
    ].copy()
    subset["Dataset"] = subset["dataset_name"].map(lambda value: DATASET_DISPLAY_NAMES.get(value, value))
    subset = subset.rename(
        columns={
            "model_display": "Model",
            "accuracy": "Accuracy",
            "mean_score": "Mean Score",
            "mean_token_f1": "Mean F1",
            "en_accuracy": "EN Acc.",
            "da_accuracy": "DA Acc.",
        }
    )
    subset = subset.drop(columns=["dataset_name"])
    return subset.sort_values(["Dataset", "Model"]).reset_index(drop=True)


def build_danish_scaling_table(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[
        [
            "model_display",
            "tokens_label",
            "version",
            "da_accuracy",
            "da_mean_score",
            "non_da_mean_score",
            "delta_da_accuracy_vs_public",
            "delta_non_da_mean_score_vs_public",
        ]
    ].copy()
    subset = subset.rename(
        columns={
            "model_display": "Model",
            "tokens_label": "B Tokens",
            "version": "Version",
            "da_accuracy": "DA Acc.",
            "da_mean_score": "DA Score",
            "non_da_mean_score": "Non-DA Score",
            "delta_da_accuracy_vs_public": "$\\Delta$ DA Acc. vs Public",
            "delta_non_da_mean_score_vs_public": "$\\Delta$ Non-DA Score vs Public",
        }
    )
    return subset.reset_index(drop=True)


def build_latent_geometry_last_layer(frame: pd.DataFrame, public_model: str) -> pd.DataFrame:
    subset = frame[
        (frame["representation"] == "last")
        & (frame["model_name"] != public_model)
    ].copy()
    subset = subset.sort_values("layer").groupby(["model_name", "dataset_name"], as_index=False).tail(1)
    subset["Dataset"] = subset["dataset_name"].map(lambda value: DATASET_DISPLAY_NAMES.get(value, value))
    subset = subset[
        [
            "model_display",
            "Dataset",
            "layer",
            "cosine_to_public",
            "distance_to_public",
            "separation_ratio_to_public",
            "within_variance",
        ]
    ].rename(
        columns={
            "model_display": "Model",
            "layer": "Layer",
            "cosine_to_public": "Cosine to Public",
            "distance_to_public": "Dist. to Public",
            "separation_ratio_to_public": "Sep. Ratio to Public",
            "within_variance": "Within Var.",
        }
    )
    return subset.sort_values(["Dataset", "Model"]).reset_index(drop=True)


def build_mkqa_language_table(frame: pd.DataFrame, public_model: str) -> pd.DataFrame:
    subset = frame[
        (frame["dataset_name"] == "mkqa_en_da")
        & (frame["representation"] == "last")
        & (frame["model_name"] != public_model)
    ].copy()
    subset = subset.sort_values("layer").groupby(["model_name", "dataset_name"], as_index=False).tail(1)
    subset = subset[
        [
            "model_display",
            "layer",
            "en_da_cosine",
            "en_da_centroid_distance",
            "en_da_separation_ratio",
        ]
    ].rename(
        columns={
            "model_display": "Model",
            "layer": "Layer",
            "en_da_cosine": "EN/DA Cosine",
            "en_da_centroid_distance": "EN/DA Dist.",
            "en_da_separation_ratio": "EN/DA Sep. Ratio",
        }
    )
    return subset.sort_values("Model").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    analysis_root = args.analysis_root

    performance = load_csv(analysis_root / "expert_eval_summary.csv")
    scaling = load_csv(analysis_root / "expert_danish_scaling_summary.csv")
    geometry = load_csv(analysis_root / "expert_latent_geometry_summary.csv")

    performance_table = build_performance_overview(performance)
    scaling_table = build_danish_scaling_table(scaling)
    geometry_table = build_latent_geometry_last_layer(geometry, args.public_model)
    mkqa_table = build_mkqa_language_table(geometry, args.public_model)

    write_table(
        performance_table,
        args.output_root,
        "expert_performance_overview",
        "Standalone expert-sweep performance summary across the selected mix datasets.",
        "tab:expert_performance_overview",
    )
    write_table(
        scaling_table,
        args.output_root,
        "expert_danish_scaling",
        "Danish-expert scaling summary relative to the public baseline.",
        "tab:expert_danish_scaling",
    )
    write_table(
        geometry_table,
        args.output_root,
        "expert_latent_geometry_last_layer",
        "Last-layer hidden-state geometry for standalone experts relative to the public baseline.",
        "tab:expert_latent_geometry_last_layer",
    )
    write_table(
        mkqa_table,
        args.output_root,
        "expert_mkqa_language_geometry",
        "Last-layer English/Danish hidden-state geometry for standalone experts on MGQA (EN/DA).",
        "tab:expert_mkqa_language_geometry",
    )
    print(f"Wrote expert-sweep summary tables to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
