from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.utils import load_tokenizer_with_known_fixes
from eval.benchmarks.mix.runners.run_mix_analysis import (
    apply_chat_template_if_requested,
    load_allowed_model_names,
    load_jsonl_records,
    load_manifest_entries,
)


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_tokenization_stats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tokenization fragmentation across mix datasets and languages."
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include from the manifest.")
    parser.add_argument("--max-examples-per-dataset", type=int, default=200)
    parser.add_argument("--model-path", help="Explicit tokenizer/model path.")
    parser.add_argument("--model-name", help="Model name from model_paths/all_models.txt.")
    parser.add_argument("--model-root", help="Directory containing model folders.")
    parser.add_argument("--tokenizer-path", help="Optional tokenizer path. Defaults to resolved model path.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    if not args.model_name or not args.model_root:
        raise ValueError("Provide either --model-path or both --model-name and --model-root.")
    allowed_names = load_allowed_model_names(args.model_registry)
    if args.model_name not in allowed_names:
        raise ValueError(f"Model name `{args.model_name}` was not found in {args.model_registry}.")
    return str(Path(args.model_root) / args.model_name)


def dataset_display_name(dataset_name: str) -> str:
    return {
        "mkqa_en_da": "MGQA (EN/DA)",
        "gsm8k_subset": "GSM8K",
        "mbpp_subset": "MBPP",
        "pubmedqa_subset": "PubMedQA",
        "ag_news_subset": "AG News",
        "common_gen_subset": "CommonGen",
    }.get(dataset_name, dataset_name)


def sanitize_name(name: str) -> str:
    allowed = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "item"


WORD_RE = re.compile(r"\S+")


def count_subtokens(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def analyze_example(tokenizer, text: str) -> dict[str, float | int]:
    words = WORD_RE.findall(text)
    prompt_token_count = count_subtokens(tokenizer, text)
    prompt_char_count = len(text)

    if not words:
        return {
            "prompt_token_count": prompt_token_count,
            "prompt_word_count": 0,
            "prompt_char_count": prompt_char_count,
            "tokens_per_word": 0.0,
            "chars_per_token": 0.0,
            "mean_subtokens_per_word": 0.0,
            "median_subtokens_per_word": 0.0,
            "multi_subtoken_word_share": 0.0,
            "high_fragment_word_share": 0.0,
            "max_subtokens_for_word": 0,
        }

    word_piece_counts = np.asarray([count_subtokens(tokenizer, word) for word in words], dtype=np.int32)
    return {
        "prompt_token_count": int(prompt_token_count),
        "prompt_word_count": int(len(words)),
        "prompt_char_count": int(prompt_char_count),
        "tokens_per_word": float(prompt_token_count / max(len(words), 1)),
        "chars_per_token": float(prompt_char_count / max(prompt_token_count, 1)),
        "mean_subtokens_per_word": float(word_piece_counts.mean()),
        "median_subtokens_per_word": float(np.median(word_piece_counts)),
        "multi_subtoken_word_share": float((word_piece_counts >= 2).mean()),
        "high_fragment_word_share": float((word_piece_counts >= 3).mean()),
        "max_subtokens_for_word": int(word_piece_counts.max()),
    }


def build_rows(
    tokenizer,
    manifest_entries: list[dict[str, Any]],
    max_examples_per_dataset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_entry in manifest_entries:
        dataset_name = dataset_entry["name"]
        records = load_jsonl_records(dataset_entry["path"], max_examples=max_examples_per_dataset)
        prompting_config = dict(dataset_entry.get("prompting", {}))
        for record in records:
            prompt = record.get("prompt")
            if not prompt:
                continue
            normalized_prompt = apply_chat_template_if_requested(tokenizer, prompt, prompting_config)
            stats = analyze_example(tokenizer, normalized_prompt)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_label": dataset_display_name(dataset_name),
                    "domain": str(dataset_entry.get("domain", "unknown")),
                    "language": str(record.get("language", "unknown")),
                    "example_id": str(record.get("example_id", "")),
                    **stats,
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


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(row)

    metrics = [
        "prompt_token_count",
        "prompt_word_count",
        "prompt_char_count",
        "tokens_per_word",
        "chars_per_token",
        "mean_subtokens_per_word",
        "median_subtokens_per_word",
        "multi_subtoken_word_share",
        "high_fragment_word_share",
        "max_subtokens_for_word",
    ]
    result: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        out = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        out["num_examples"] = len(items)
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            out[f"mean_{metric}"] = float(values.mean())
        result.append(out)
    return result


def plot_language_summary(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in summary_rows if row["language"] in {"en", "da"}]
    if not rows:
        return
    datasets = sorted({row["dataset_label"] for row in rows})
    metrics = [
        ("mean_tokens_per_word", "Tokens / word"),
        ("mean_multi_subtoken_word_share", "Share of multi-piece words"),
        ("mean_high_fragment_word_share", "Share of 3+ piece words"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.3 * len(metrics), 4.8), squeeze=False)
    colors = {"en": "#1b6ca8", "da": "#d95f02"}

    for ax, (metric_key, title) in zip(axes[0], metrics):
        x = np.arange(len(datasets))
        width = 0.36
        for offset_idx, language in enumerate(("en", "da")):
            values = []
            for dataset_label in datasets:
                matched = next(
                    (
                        row
                        for row in rows
                        if row["dataset_label"] == dataset_label and row["language"] == language
                    ),
                    None,
                )
                values.append(float(matched[metric_key]) if matched is not None else np.nan)
            ax.bar(x + (offset_idx - 0.5) * width, values, width=width, color=colors[language], label=language.upper())
        ax.set_title(title, fontsize=11.5, pad=4, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("Mean value", fontsize=11.5, fontweight="semibold")
    legend = axes[0][0].legend(frameon=False, fontsize=10, loc="best")
    for text in legend.get_texts():
        text.set_fontweight("semibold")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.84, wspace=0.25)
    fig.suptitle("English vs Danish Tokenization Fragmentation", fontsize=14, y=0.95, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_readme(path: Path, model_path: str) -> None:
    lines = [
        "# Mix Tokenization Statistics",
        "",
        f"Tokenizer source: `{model_path}`",
        "",
        "Artifacts:",
        "- `tokenization_example_stats.csv`: per-example tokenization statistics.",
        "- `tokenization_by_dataset_language.csv`: dataset/language aggregated tokenization statistics.",
        "- `tokenization_by_language.csv`: language-only aggregate statistics.",
        "- `tokenization_en_da_summary.png`: compact English vs Danish comparison figure.",
        "",
        "Interpretation guide:",
        "- Higher `tokens_per_word` means more fragmented tokenization.",
        "- Higher `multi_subtoken_word_share` means more words split into at least two pieces.",
        "- Higher `high_fragment_word_share` means more severe 3+ subtoken splitting.",
        "- Lower `chars_per_token` can also indicate finer-grained fragmentation.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args)
    tokenizer = load_tokenizer_with_known_fixes(args.tokenizer_path or model_path)

    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}

    manifest_entries = load_manifest_entries(args.manifest_path, selected_datasets)
    rows = build_rows(tokenizer, manifest_entries, max_examples_per_dataset=args.max_examples_per_dataset)
    if not rows:
        raise ValueError("No examples were loaded for tokenization analysis.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    write_csv(output_root / "tokenization_example_stats.csv", rows)
    dataset_language_summary = aggregate_rows(rows, ["dataset_name", "dataset_label", "domain", "language"])
    language_summary = aggregate_rows(rows, ["language"])
    write_csv(output_root / "tokenization_by_dataset_language.csv", dataset_language_summary)
    write_csv(output_root / "tokenization_by_language.csv", language_summary)
    plot_language_summary(dataset_language_summary, output_root / "tokenization_en_da_summary.png")
    write_readme(output_root / "README.md", model_path=model_path)

    print(f"Wrote tokenization analysis to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
