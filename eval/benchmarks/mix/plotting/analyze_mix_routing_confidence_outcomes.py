from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "routing_light" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_routing_confidence"

MODEL_COLORS = {
    "FlexOlmo-8x7B-1T-a4-55B-v2": "#701f57",
    "FlexOlmo-8x7B-1T-a4-55B-v2-rt": "#e13343",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the relationship between routing confidence and downstream outcome quality."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-names",
        default="FlexOlmo-8x7B-1T-a4-55B-v2,FlexOlmo-8x7B-1T-a4-55B-v2-rt",
        help="Comma-separated model names to include.",
    )
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include.")
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=5,
        help="Number of confidence buckets for bucketed accuracy/score plots.",
    )
    return parser.parse_args()


def parse_model_names(raw_value: str) -> list[str]:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not names:
        raise ValueError("Provide at least one model name.")
    return names


def sanitize_name(name: str) -> str:
    allowed = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "item"


def model_display_name(model_name: str) -> str:
    return model_name.replace("FlexOlmo-8x7B-1T-", "")


def dataset_display_name(dataset_name: str) -> str:
    return {
        "mkqa_en_da": "MGQA (EN/DA)",
        "gsm8k_subset": "GSM8K",
        "mbpp_subset": "MBPP",
        "pubmedqa_subset": "PubMedQA",
        "ag_news_subset": "AG News",
        "common_gen_subset": "CommonGen",
    }.get(dataset_name, dataset_name)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_generation_artifacts(text: str) -> str:
    cleaned = text.strip()
    for prefix in ("answer:", "response:", "final answer:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def relaxed_match(prediction: str, reference: str) -> bool:
    pred = normalize_text(strip_generation_artifacts(prediction))
    ref = normalize_text(reference)
    return pred == ref or pred in ref or ref in pred


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(strip_generation_artifacts(prediction)).split()
    ref_tokens = normalize_text(reference).split()
    if pred_tokens == ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    common = sum(min(pred_counts[token], ref_counts.get(token, 0)) for token in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def classify_pubmedqa(prediction: str) -> str:
    normalized = normalize_text(strip_generation_artifacts(prediction))
    for label in ("yes", "no", "maybe"):
        if normalized.startswith(label):
            return label
    return normalized.split(" ", 1)[0] if normalized else ""


def score_prediction(record: dict[str, Any]) -> dict[str, Any]:
    scoring_mode = str(record.get("scoring_mode", "qa"))
    prediction_text = str(record.get("predicted_output_text", ""))
    reference = str(record.get("reference_answer", ""))
    cleaned_prediction = strip_generation_artifacts(prediction_text)

    if scoring_mode == "classification":
        predicted_label = classify_pubmedqa(cleaned_prediction)
        reference_label = normalize_text(reference)
        is_correct = predicted_label == reference_label
        return {
            "score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "exact_match": is_correct,
            "relaxed_match": is_correct,
            "token_f1": 1.0 if is_correct else 0.0,
            "normalized_prediction": predicted_label,
            "normalized_reference": reference_label,
        }

    exact = normalize_text(cleaned_prediction) == normalize_text(reference)
    relaxed = relaxed_match(cleaned_prediction, reference)
    f1 = token_f1(cleaned_prediction, reference)
    return {
        "score": f1,
        "is_correct": relaxed,
        "exact_match": exact,
        "relaxed_match": relaxed,
        "token_f1": f1,
        "normalized_prediction": normalize_text(cleaned_prediction),
        "normalized_reference": normalize_text(reference),
    }


def infer_dataset_name(model_name: str, path: Path) -> str:
    parts = path.parts
    if model_name in parts:
        model_idx = parts.index(model_name)
        if model_idx + 1 < len(parts):
            return parts[model_idx + 1]
    return "unknown"


def build_example_metric_rows(
    results_root: Path,
    model_names: list[str],
    selected_datasets: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        model_root = results_root / model_name
        if not model_root.exists():
            raise FileNotFoundError(f"Missing model results directory: {model_root}")
        for records_path in sorted(model_root.glob("*/native_full/routing_records.jsonl")):
            dataset_name = infer_dataset_name(model_name, records_path)
            if selected_datasets is not None and dataset_name not in selected_datasets:
                continue
            for record in load_jsonl(records_path):
                scored = score_prediction(record)
                base = {
                    "model_name": model_name,
                    "dataset_name": str(record.get("dataset_name", dataset_name)),
                    "language": str(record.get("language", "unknown")),
                    "example_id": str(record.get("example_id", "")),
                    "scoring_mode": str(record.get("scoring_mode", "qa")),
                    "prediction_text": str(record.get("predicted_output_text", "")),
                    "reference_answer": str(record.get("reference_answer", "")),
                    "is_correct": bool(scored["is_correct"]),
                    "is_correct_float": 1.0 if bool(scored["is_correct"]) else 0.0,
                    "score": float(scored["score"]),
                    "exact_match": 1.0 if bool(scored["exact_match"]) else 0.0,
                    "relaxed_match": 1.0 if bool(scored["relaxed_match"]) else 0.0,
                    "token_f1": float(scored["token_f1"]),
                }
                for phase_name, layer_summaries in (
                    ("prompt", record.get("prompt_router_summary_by_layer") or []),
                    ("predicted", record.get("predicted_router_summary_by_layer") or []),
                ):
                    for layer_summary in layer_summaries:
                        rows.append(
                            {
                                **base,
                                "phase": phase_name,
                                "layer": int(layer_summary["layer_idx"]),
                                "mean_top1_prob": float(layer_summary["mean_top1_prob"]),
                                "mean_top2_prob": float(layer_summary["mean_top2_prob"]),
                                "mean_top1_top2_margin": float(layer_summary["mean_top1_top2_margin"]),
                                "mean_token_entropy": float(layer_summary["mean_token_entropy"]),
                                "mean_selected_expert_prob_mass": float(
                                    layer_summary["mean_selected_expert_prob_mass"]
                                ),
                            }
                        )
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float | None:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return None
    x_values = valid["x"].to_numpy(dtype=float)
    y_values = valid["y"].to_numpy(dtype=float)
    if np.allclose(x_values, x_values[0]) or np.allclose(y_values, y_values[0]):
        return None
    if method == "pearson":
        return float(np.corrcoef(x_values, y_values)[0, 1])
    if method == "spearman":
        return float(np.corrcoef(rankdata(x_values), rankdata(y_values))[0, 1])
    raise ValueError(method)


def build_correlation_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    confidence_metrics = [
        "mean_top1_prob",
        "mean_top1_top2_margin",
        "mean_token_entropy",
        "mean_selected_expert_prob_mass",
    ]
    outcome_metrics = ["is_correct_float", "score", "token_f1"]
    rows: list[dict[str, Any]] = []
    for (model_name, dataset_name, phase, layer), subset in frame.groupby(
        ["model_name", "dataset_name", "phase", "layer"], sort=True
    ):
        for confidence_metric in confidence_metrics:
            for outcome_metric in outcome_metrics:
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "phase": phase,
                        "layer": int(layer),
                        "confidence_metric": confidence_metric,
                        "outcome_metric": outcome_metric,
                        "pearson_r": safe_corr(subset[confidence_metric], subset[outcome_metric], "pearson"),
                        "spearman_r": safe_corr(subset[confidence_metric], subset[outcome_metric], "spearman"),
                        "num_examples": int(len(subset)),
                    }
                )
    return rows


def bucketize(values: pd.Series, num_buckets: int) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=min(num_buckets, len(values)), labels=False, duplicates="drop")


def build_bucket_rows(frame: pd.DataFrame, num_buckets: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model_name, dataset_name, phase), subset in frame.groupby(
        ["model_name", "dataset_name", "phase"], sort=True
    ):
        if subset.empty:
            continue
        last_layer = int(subset["layer"].max())
        layer_subset = subset[subset["layer"] == last_layer].copy()
        if len(layer_subset) < 3:
            continue
        for confidence_metric in ("mean_top1_top2_margin", "mean_top1_prob", "mean_selected_expert_prob_mass"):
            bucket_ids = bucketize(layer_subset[confidence_metric], num_buckets)
            layer_subset["bucket_id"] = bucket_ids
            for bucket_id, bucket_frame in layer_subset.groupby("bucket_id", sort=True):
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "phase": phase,
                        "layer": last_layer,
                        "confidence_metric": confidence_metric,
                        "bucket_id": int(bucket_id),
                        "num_examples": int(len(bucket_frame)),
                        "mean_confidence": float(bucket_frame[confidence_metric].mean()),
                        "mean_accuracy": float(bucket_frame["is_correct_float"].mean()),
                        "mean_score": float(bucket_frame["score"].mean()),
                        "mean_token_f1": float(bucket_frame["token_f1"].mean()),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_correlation_by_dataset(correlations: pd.DataFrame, output_root: Path, model_names: list[str]) -> list[Path]:
    output_paths: list[Path] = []
    metric_specs = [
        ("mean_top1_top2_margin", "score", "Margin vs Score"),
        ("mean_top1_top2_margin", "is_correct_float", "Margin vs Accuracy"),
        ("mean_token_entropy", "score", "Entropy vs Score"),
        ("mean_token_entropy", "is_correct_float", "Entropy vs Accuracy"),
    ]
    for (dataset_name, phase), subset in correlations.groupby(["dataset_name", "phase"], sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for ax, (confidence_metric, outcome_metric, title) in zip(axes.flat, metric_specs):
            for model_name in model_names:
                model_subset = subset[
                    (subset["model_name"] == model_name)
                    & (subset["confidence_metric"] == confidence_metric)
                    & (subset["outcome_metric"] == outcome_metric)
                ].sort_values("layer")
                if model_subset.empty:
                    continue
                ax.plot(
                    model_subset["layer"],
                    model_subset["spearman_r"],
                    marker="o",
                    label=model_display_name(model_name),
                    color=MODEL_COLORS.get(model_name),
                )
            ax.set_title(title, fontsize=11.5, pad=4, fontweight="semibold")
            ax.set_xlabel("Layer", fontsize=11, fontweight="semibold")
            ax.set_ylabel("Spearman r", fontsize=11, fontweight="semibold")
            ax.tick_params(labelsize=10)
            ax.axhline(0.0, color="#999999", linewidth=0.8, alpha=0.7)
            ax.grid(alpha=0.25)
        legend = axes[0][0].legend(frameon=False, fontsize=10, loc="best")
        for text in legend.get_texts():
            text.set_fontweight("semibold")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88, wspace=0.20, hspace=0.28)
        fig.suptitle(
            f"{dataset_display_name(dataset_name)} | {phase.capitalize()} routing confidence vs outcome",
            fontsize=14,
            y=0.96,
            fontweight="bold",
        )
        output_path = output_root / f"routing_confidence_correlation_{sanitize_name(dataset_name)}_{phase}.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def plot_bucket_summary(buckets: pd.DataFrame, output_root: Path, model_names: list[str]) -> list[Path]:
    output_paths: list[Path] = []
    for (dataset_name, phase), subset in buckets.groupby(["dataset_name", "phase"], sort=True):
        margin_subset = subset[subset["confidence_metric"] == "mean_top1_top2_margin"]
        if margin_subset.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), squeeze=False)
        plot_specs = [
            ("mean_accuracy", "Accuracy by margin bucket"),
            ("mean_score", "Score by margin bucket"),
        ]
        for ax, (metric_name, title) in zip(axes[0], plot_specs):
            for model_name in model_names:
                model_subset = margin_subset[margin_subset["model_name"] == model_name].sort_values("bucket_id")
                if model_subset.empty:
                    continue
                ax.plot(
                    model_subset["bucket_id"] + 1,
                    model_subset[metric_name],
                    marker="o",
                    label=model_display_name(model_name),
                    color=MODEL_COLORS.get(model_name),
                )
            ax.set_title(title, fontsize=11.5, pad=4, fontweight="semibold")
            ax.set_xlabel("Confidence bucket (low → high)", fontsize=11, fontweight="semibold")
            ax.set_ylabel("Mean value", fontsize=11, fontweight="semibold")
            ax.tick_params(labelsize=10)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25)
        legend = axes[0][0].legend(frameon=False, fontsize=10, loc="best")
        for text in legend.get_texts():
            text.set_fontweight("semibold")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.84, wspace=0.22)
        fig.suptitle(
            f"{dataset_display_name(dataset_name)} | {phase.capitalize()} margin-confidence buckets",
            fontsize=14,
            y=0.94,
            fontweight="bold",
        )
        output_path = output_root / f"routing_confidence_buckets_{sanitize_name(dataset_name)}_{phase}.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def write_readme(output_path: Path, model_names: list[str], datasets: list[str]) -> None:
    lines = [
        "# Routing Confidence vs Outcome",
        "",
        "Compared models:",
        *[f"- `{model_name}`" for model_name in model_names],
        "",
        "Datasets:",
        *[f"- `{dataset_name}`" for dataset_name in datasets],
        "",
        "Artifacts:",
        "- `routing_confidence_outcome_records.csv`: per-example, per-layer routing confidence joined with outcome scores.",
        "- `routing_confidence_correlations.csv`: Pearson/Spearman correlations between routing confidence metrics and outcomes.",
        "- `routing_confidence_bucket_summary.csv`: last-layer confidence-bucket summaries for accuracy and score.",
        "- `routing_confidence_correlation_<dataset>_<phase>.png`: layer-wise confidence/outcome correlations.",
        "- `routing_confidence_buckets_<dataset>_<phase>.png`: last-layer bucketed accuracy/score curves.",
        "",
        "Interpretation guide:",
        "- Positive correlation between `mean_top1_top2_margin` and `score` means larger router margin tends to align with better outputs.",
        "- Negative correlation between `mean_token_entropy` and `score` means lower-entropy routing tends to align with better outputs.",
        "- If RT shows stronger confidence/outcome correlation than non-RT, that suggests router confidence is more informative after tuning.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_names = parse_model_names(args.model_names)
    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}

    rows = build_example_metric_rows(
        results_root=args.results_root,
        model_names=model_names,
        selected_datasets=selected_datasets,
    )
    if not rows:
        raise ValueError(f"No routing records were found under {args.results_root}.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "routing_confidence_outcome_records.csv", rows)

    frame = pd.DataFrame(rows)
    correlation_rows = build_correlation_rows(frame)
    bucket_rows = build_bucket_rows(frame, num_buckets=args.num_buckets)
    write_csv(output_root / "routing_confidence_correlations.csv", correlation_rows)
    write_csv(output_root / "routing_confidence_bucket_summary.csv", bucket_rows)

    correlation_frame = pd.DataFrame(correlation_rows)
    bucket_frame = pd.DataFrame(bucket_rows)
    plot_correlation_by_dataset(correlation_frame, output_root=output_root, model_names=model_names)
    plot_bucket_summary(bucket_frame, output_root=output_root, model_names=model_names)

    datasets = sorted(frame["dataset_name"].drop_duplicates().tolist())
    write_readme(output_root / "README.md", model_names=model_names, datasets=datasets)
    print(f"Wrote routing-confidence analysis to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())