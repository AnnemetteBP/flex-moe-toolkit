from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import string
import sys
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_STOP_PATTERNS = (
    "\nQuestion",
    "\nSpørgsmål",
    "\nSvar:",
    "\nAnswer:",
    "\n```",
    "```",
    "\n###",
    "###",
)
ARTICLE_PATTERN = re.compile(r"\b(a|an|the|en|et|den|det|de)\b", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
TRAILING_ZERO_NUMBER_PATTERN = re.compile(r"\b(-?\d+)\.0+\b")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute post-hoc MKQA answer accuracy from saved routing_records.jsonl files."
    )
    parser.add_argument(
        "--routing-root",
        required=True,
        help="Root containing per-model MKQA routing outputs, e.g. eval_results/mkqa/full/flexolmo/a4/routing",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where accuracy summaries will be written. Defaults to <routing-root>/../accuracy",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional explicit model names to score. Defaults to all models with native_full routing records.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = TRAILING_ZERO_NUMBER_PATTERN.sub(r"\1", text)
    text = "".join(char for char in text if char not in string.punctuation)
    text = ARTICLE_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def strip_generation_artifacts(text: str | None) -> str:
    if text is None:
        return ""
    cleaned = text.strip()
    for pattern in DEFAULT_STOP_PATTERNS:
        if pattern in cleaned:
            cleaned = cleaned.split(pattern, 1)[0]
    cleaned = cleaned.strip()
    cleaned = cleaned.lstrip(":,- ")
    return cleaned.strip()


def relaxed_match(prediction: str, reference: str) -> bool:
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)
    if not normalized_prediction or not normalized_reference:
        return normalized_prediction == normalized_reference
    if normalized_prediction == normalized_reference:
        return True
    return (
        normalized_prediction in normalized_reference
        or normalized_reference in normalized_prediction
    )


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = sum((pred_counts & ref_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_text(prediction) == normalize_text(reference)


def find_model_names(routing_root: Path) -> list[str]:
    names = []
    for model_dir in routing_root.iterdir():
        records_path = model_dir / "mkqa_en_da" / "native_full" / "routing_records.jsonl"
        if model_dir.is_dir() and records_path.exists():
            names.append(model_dir.name)
    return sorted(names)


def resolve_records_path(routing_root: Path, model_name: str) -> Path:
    candidate_paths = [
        routing_root / model_name / "mkqa_en_da" / "native_full" / "routing_records.jsonl",
        routing_root / "mkqa_en_da" / "native_full" / "routing_records.jsonl",
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find routing_records.jsonl for model `{model_name}` under `{routing_root}`. "
        f"Tried: {', '.join(str(path) for path in candidate_paths)}"
    )


def score_records(records: list[dict]) -> tuple[list[dict], dict]:
    scored_rows = []
    aggregate = {
        "num_examples": 0,
        "num_scorable": 0,
        "num_exact_match": 0,
        "num_relaxed_match": 0,
        "sum_token_f1": 0.0,
        "by_language": defaultdict(
            lambda: {
                "num_examples": 0,
                "num_scorable": 0,
                "num_exact_match": 0,
                "num_relaxed_match": 0,
                "sum_token_f1": 0.0,
            }
        ),
    }

    for record in records:
        language = record.get("language", "unknown")
        reference = record.get("reference_answer")
        prediction = strip_generation_artifacts(record.get("predicted_output_text"))
        is_scorable = reference is not None
        em = False
        rm = False
        f1 = None
        aggregate["num_examples"] += 1
        aggregate["by_language"][language]["num_examples"] += 1
        if is_scorable:
            em = exact_match(prediction, reference)
            rm = relaxed_match(prediction, reference)
            f1 = token_f1(prediction, reference)
            aggregate["num_scorable"] += 1
            aggregate["num_exact_match"] += int(em)
            aggregate["num_relaxed_match"] += int(rm)
            aggregate["sum_token_f1"] += float(f1)
            aggregate["by_language"][language]["num_scorable"] += 1
            aggregate["by_language"][language]["num_exact_match"] += int(em)
            aggregate["by_language"][language]["num_relaxed_match"] += int(rm)
            aggregate["by_language"][language]["sum_token_f1"] += float(f1)
        scored_rows.append(
            {
                "example_id": record.get("example_id"),
                "language": language,
                "question": record.get("question"),
                "reference_answer": reference,
                "predicted_output_text_raw": record.get("predicted_output_text"),
                "predicted_output_text_clean": prediction,
                "reference_answer_normalized": normalize_text(reference) if is_scorable else None,
                "predicted_output_text_normalized": normalize_text(prediction) if is_scorable else None,
                "is_scorable": is_scorable,
                "exact_match": em if is_scorable else None,
                "relaxed_match": rm if is_scorable else None,
                "token_f1": f1,
            }
        )

    summary = {
        "num_examples": aggregate["num_examples"],
        "num_scorable": aggregate["num_scorable"],
        "exact_match_accuracy": (
            aggregate["num_exact_match"] / aggregate["num_scorable"] if aggregate["num_scorable"] else 0.0
        ),
        "relaxed_match_accuracy": (
            aggregate["num_relaxed_match"] / aggregate["num_scorable"] if aggregate["num_scorable"] else 0.0
        ),
        "mean_token_f1": (
            aggregate["sum_token_f1"] / aggregate["num_scorable"] if aggregate["num_scorable"] else 0.0
        ),
        "by_language": {},
    }
    for language, stats in sorted(aggregate["by_language"].items()):
        summary["by_language"][language] = {
            "num_examples": stats["num_examples"],
            "num_scorable": stats["num_scorable"],
            "exact_match_accuracy": (
                stats["num_exact_match"] / stats["num_scorable"] if stats["num_scorable"] else 0.0
            ),
            "relaxed_match_accuracy": (
                stats["num_relaxed_match"] / stats["num_scorable"] if stats["num_scorable"] else 0.0
            ),
            "mean_token_f1": (
                stats["sum_token_f1"] / stats["num_scorable"] if stats["num_scorable"] else 0.0
            ),
        }
    return scored_rows, summary


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=False) + "\n")


def main() -> int:
    args = parse_args()
    routing_root = Path(args.routing_root)
    output_root = Path(args.output_root) if args.output_root else routing_root.parent / "accuracy"
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = args.model_name or find_model_names(routing_root)
    if not model_names:
        raise SystemExit(f"No models with native_full routing records found under {routing_root}")

    summary_rows = []
    for model_name in model_names:
        records_path = resolve_records_path(routing_root, model_name)
        records = load_jsonl(records_path)
        scored_rows, summary = score_records(records)
        model_output_dir = output_root / model_name / "native_full"
        write_jsonl(scored_rows, model_output_dir / "accuracy_records.jsonl")
        (model_output_dir / "accuracy_summary.json").write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "routing_records_path": str(records_path),
                    **summary,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        summary_rows.append(
            {
                "model_name": model_name,
                "num_examples": summary["num_examples"],
                "num_scorable": summary["num_scorable"],
                "exact_match_accuracy": summary["exact_match_accuracy"],
                "relaxed_match_accuracy": summary["relaxed_match_accuracy"],
                "mean_token_f1": summary["mean_token_f1"],
                "en_exact_match_accuracy": summary["by_language"].get("en", {}).get("exact_match_accuracy", 0.0),
                "en_relaxed_match_accuracy": summary["by_language"].get("en", {}).get("relaxed_match_accuracy", 0.0),
                "da_exact_match_accuracy": summary["by_language"].get("da", {}).get("exact_match_accuracy", 0.0),
                "da_relaxed_match_accuracy": summary["by_language"].get("da", {}).get("relaxed_match_accuracy", 0.0),
                "en_mean_token_f1": summary["by_language"].get("en", {}).get("mean_token_f1", 0.0),
                "da_mean_token_f1": summary["by_language"].get("da", {}).get("mean_token_f1", 0.0),
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: row["relaxed_match_accuracy"], reverse=True)
    csv_path = output_root / "mkqa_accuracy_overview.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    (output_root / "mkqa_accuracy_overview.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote MKQA post-hoc accuracy summaries for {len(summary_rows)} models to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
