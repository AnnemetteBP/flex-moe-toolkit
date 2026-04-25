from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_COACTIVATION_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_coactivation"
DEFAULT_LATENT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_latent_space"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_summary_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact CSV and LaTeX summary tables for mix co-activation and latent analyses."
    )
    parser.add_argument("--coactivation-root", type=Path, default=DEFAULT_COACTIVATION_ROOT)
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-names",
        default="FlexOlmo-8x7B-1T-a4-55B-v2,FlexOlmo-8x7B-1T-a4-55B-v2-rt",
        help="Comma-separated model names in left,right comparison order.",
    )
    return parser.parse_args()


def parse_model_names(raw_value: str) -> list[str]:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(names) != 2:
        raise ValueError("Expected exactly two model names for summary-table comparisons.")
    return names


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, output_root: Path, stem: str, caption: str, label: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{stem}.csv"
    tex_path = output_root / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(
        index=False,
        float_format=lambda value: f"{value:.3f}" if isinstance(value, float) else str(value),
        escape=False,
        caption=caption,
        label=label,
    )
    tex_path.write_text(latex, encoding="utf-8")


def build_coactivation_table(coactivation_root: Path, model_names: list[str]) -> pd.DataFrame:
    path = coactivation_root / "coactivation_aggregate_summary.csv"
    frame = load_csv(path)
    left_model, right_model = model_names

    rows: list[dict] = []
    for dataset_name in sorted(frame["dataset_name"].drop_duplicates()):
        left = frame[(frame["dataset_name"] == dataset_name) & (frame["model_name"] == left_model)]
        right = frame[(frame["dataset_name"] == dataset_name) & (frame["model_name"] == right_model)]
        if left.empty or right.empty:
            continue
        left_row = left.iloc[0]
        right_row = right.iloc[0]
        rows.append(
            {
                "Dataset": dataset_name,
                "Public Offdiag (v2)": float(left_row["public_offdiag_mean"]),
                "Public Offdiag (rt)": float(right_row["public_offdiag_mean"]),
                "$\\Delta$ Public": float(right_row["public_offdiag_mean"] - left_row["public_offdiag_mean"]),
                "Offdiag Mean (v2)": float(left_row["offdiag_mean"]),
                "Offdiag Mean (rt)": float(right_row["offdiag_mean"]),
                "$\\Delta$ Offdiag": float(right_row["offdiag_mean"] - left_row["offdiag_mean"]),
                "Top Pair (v2)": str(left_row["dominant_pair"]),
                "Top Pair (rt)": str(right_row["dominant_pair"]),
                "$\\Delta$ Top Pair Strength": float(
                    right_row["dominant_pair_value"] - left_row["dominant_pair_value"]
                ),
            }
        )

    return pd.DataFrame(rows)


def select_last_layer(frame: pd.DataFrame, dataset_name: str, source_name: str, representation: str) -> int | None:
    subset = frame[
        (frame["dataset_name"] == dataset_name)
        & (frame["representation_source"] == source_name)
        & (frame["representation"] == representation)
    ]
    if subset.empty:
        return None
    return int(subset["layer"].max())


def metric_value(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    source_name: str,
    representation: str,
    layer: int,
    group: str,
    metric: str,
) -> float | None:
    subset = frame[
        (frame["dataset_name"] == dataset_name)
        & (frame["representation_source"] == source_name)
        & (frame["representation"] == representation)
        & (frame["layer"] == layer)
        & (frame["group"] == group)
        & (frame["metric"] == metric)
    ]
    if subset.empty:
        return None
    return float(subset.iloc[0]["value"])


def build_latent_geometry_table(latent_root: Path, model_names: list[str]) -> pd.DataFrame:
    path = latent_root / "latent_space_similarity_summary.csv"
    frame = load_csv(path)
    left_model, right_model = model_names
    representation = "last"
    rows: list[dict] = []

    for dataset_name in sorted(frame["dataset_name"].drop_duplicates()):
        for source_name in ("pre_router", "hidden_state"):
            layer = select_last_layer(frame, dataset_name, source_name, representation)
            if layer is None:
                continue
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Source": source_name,
                    "Layer": layer,
                    "Cross-Model Cosine": metric_value(
                        frame,
                        dataset_name=dataset_name,
                        source_name=source_name,
                        representation=representation,
                        layer=layer,
                        group="all",
                        metric="cross_model_cosine",
                    ),
                    "Centroid Dist.": metric_value(
                        frame,
                        dataset_name=dataset_name,
                        source_name=source_name,
                        representation=representation,
                        layer=layer,
                        group="all",
                        metric="cross_model_centroid_distance",
                    ),
                    "Sep. Ratio": metric_value(
                        frame,
                        dataset_name=dataset_name,
                        source_name=source_name,
                        representation=representation,
                        layer=layer,
                        group="all",
                        metric="cross_model_separation_ratio",
                    ),
                    f"Within Var ({left_model})": metric_value(
                        frame,
                        dataset_name=dataset_name,
                        source_name=source_name,
                        representation=representation,
                        layer=layer,
                        group=left_model,
                        metric="within_group_variance",
                    ),
                    f"Within Var ({right_model})": metric_value(
                        frame,
                        dataset_name=dataset_name,
                        source_name=source_name,
                        representation=representation,
                        layer=layer,
                        group=right_model,
                        metric="within_group_variance",
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_mkqa_language_table(latent_root: Path) -> pd.DataFrame:
    path = latent_root / "latent_space_similarity_summary.csv"
    frame = load_csv(path)
    dataset_name = "mkqa_en_da"
    representation = "last"
    rows: list[dict] = []

    for source_name in ("pre_router", "hidden_state"):
        layer = select_last_layer(frame, dataset_name, source_name, representation)
        if layer is None:
            continue
        rows.append(
            {
                "Source": source_name,
                "Layer": layer,
                "EN Cross-Model Cosine": metric_value(
                    frame,
                    dataset_name=dataset_name,
                    source_name=source_name,
                    representation=representation,
                    layer=layer,
                    group="en",
                    metric="cross_model_cosine",
                ),
                "DA Cross-Model Cosine": metric_value(
                    frame,
                    dataset_name=dataset_name,
                    source_name=source_name,
                    representation=representation,
                    layer=layer,
                    group="da",
                    metric="cross_model_cosine",
                ),
                "EN Sep. Ratio": metric_value(
                    frame,
                    dataset_name=dataset_name,
                    source_name=source_name,
                    representation=representation,
                    layer=layer,
                    group="en",
                    metric="cross_model_separation_ratio",
                ),
                "DA Sep. Ratio": metric_value(
                    frame,
                    dataset_name=dataset_name,
                    source_name=source_name,
                    representation=representation,
                    layer=layer,
                    group="da",
                    metric="cross_model_separation_ratio",
                ),
            }
        )
    return pd.DataFrame(rows)


def write_readme(output_root: Path) -> None:
    text = """# 55B Summary Tables

This directory contains compact CSV and LaTeX tables derived from the mix comparison outputs.

Files:
- `coactivation_dataset_comparison.csv/.tex`
- `latent_geometry_last_layer.csv/.tex`
- `mkqa_language_geometry.csv/.tex`

Intended use:
- quick paper/slides tables
- side-by-side support for the co-activation and latent-space figures
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_names = parse_model_names(args.model_names)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    coactivation_table = build_coactivation_table(args.coactivation_root, model_names)
    latent_table = build_latent_geometry_table(args.latent_root, model_names)
    mkqa_table = build_mkqa_language_table(args.latent_root)

    write_table(
        coactivation_table,
        output_root,
        "coactivation_dataset_comparison",
        "Co-activation summary comparison for the 55B FlexOlmo pair.",
        "tab:mix_coactivation_summary",
    )
    write_table(
        latent_table,
        output_root,
        "latent_geometry_last_layer",
        "Last-layer latent geometry summary for pre-router and hidden-state representations.",
        "tab:mix_latent_geometry",
    )
    write_table(
        mkqa_table,
        output_root,
        "mkqa_language_geometry",
        "MKQA English/Danish geometry comparison for the 55B FlexOlmo pair.",
        "tab:mix_mkqa_language_geometry",
    )
    write_readme(output_root)

    print(f"Wrote mix summary tables to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
