from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM

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
    resolve_device,
)
from eval.benchmarks.mix.runners.run_mix_causal_intervention import score_prediction


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "expert_sweep" / "accuracy"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact standalone-expert generation scoring on selected mix datasets."
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include from the manifest.")
    parser.add_argument("--max-examples-per-dataset", type=int, default=150)
    parser.add_argument("--model-path", help="Explicit model path or HF identifier.")
    parser.add_argument("--model-name", help="Model name from model_paths/all_models.txt.")
    parser.add_argument("--model-root", help="Directory containing model folders on UCloud.")
    parser.add_argument("--tokenizer-path", help="Optional tokenizer path. Defaults to the resolved model path.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--default-max-new-tokens", type=int, default=32)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    return getattr(torch, dtype_name)


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    if not args.model_name or not args.model_root:
        raise ValueError("Provide either --model-path or both --model-name and --model-root.")
    allowed_names = load_allowed_model_names(args.model_registry)
    if args.model_name not in allowed_names:
        raise ValueError(f"Model name `{args.model_name}` was not found in {args.model_registry}.")
    return str(Path(args.model_root) / args.model_name)


def resolved_model_name(args: argparse.Namespace, model_path: str) -> str:
    if args.model_name:
        return args.model_name
    return Path(model_path).name


def load_model_and_tokenizer(model_path: str, tokenizer_path: str | None, device: torch.device, dtype_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=parse_dtype(dtype_name),
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer_with_known_fixes(tokenizer_path or model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either `pad_token_id` or `eos_token_id`.")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def encode_prompt(tokenizer, prompt: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def normalize_example(tokenizer, record: dict[str, Any], dataset_name: str, dataset_entry: dict) -> dict[str, Any]:
    prompt = record.get("prompt")
    if not prompt:
        raise ValueError(f"Dataset `{dataset_name}` contains a record without `prompt`.")
    prompting_config = dict(dataset_entry.get("prompting", {}))
    generation_config = dict(dataset_entry.get("generation", {}))
    normalized = dict(record)
    normalized["dataset_name"] = dataset_name
    normalized["prompt"] = apply_chat_template_if_requested(tokenizer, prompt, prompting_config)
    normalized["generation_config"] = generation_config
    normalized["scoring_mode"] = record.get("scoring_mode", dataset_entry.get("scoring_mode", "qa"))
    return normalized


def generate_prediction(
    model,
    tokenizer,
    example: dict[str, Any],
    *,
    max_length: int,
    default_max_new_tokens: int,
    device: torch.device,
) -> str:
    inputs = encode_prompt(tokenizer, example["prompt"], max_length=max_length, device=device)
    generation_cfg = dict(example.get("generation_config", {}))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", default_max_new_tokens))
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :].detach().cpu()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate_dataset(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    *,
    max_length: int,
    default_max_new_tokens: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in examples:
        prediction_text = generate_prediction(
            model,
            tokenizer,
            example,
            max_length=max_length,
            default_max_new_tokens=default_max_new_tokens,
            device=device,
        )
        scored = score_prediction(example, prediction_text)
        records.append(
            {
                "example_id": example["example_id"],
                "dataset_name": example["dataset_name"],
                "language": example.get("language", "unknown"),
                "question": example.get("question"),
                "reference_answer": example.get("reference_answer"),
                "scoring_mode": example.get("scoring_mode"),
                "prediction_text": prediction_text,
                **scored,
            }
        )

    summary = {
        "num_examples": len(records),
        "mean_score": (
            sum(float(item["score"]) for item in records) / len(records)
            if records
            else 0.0
        ),
        "accuracy": (
            sum(1 for item in records if item["is_correct"]) / len(records)
            if records
            else 0.0
        ),
        "mean_token_f1": (
            sum(float(item["token_f1"]) for item in records) / len(records)
            if records
            else 0.0
        ),
        "by_language": {},
    }
    languages = sorted({item["language"] for item in records})
    for language in languages:
        subset = [item for item in records if item["language"] == language]
        summary["by_language"][language] = {
            "num_examples": len(subset),
            "accuracy": sum(1 for item in subset if item["is_correct"]) / len(subset) if subset else 0.0,
            "mean_score": sum(float(item["score"]) for item in subset) / len(subset) if subset else 0.0,
            "mean_token_f1": sum(float(item["token_f1"]) for item in subset) / len(subset) if subset else 0.0,
        }
    return records, summary


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args)
    model_name = resolved_model_name(args, model_path)
    device = resolve_device(args.device)
    tokenizer_path = args.tokenizer_path or model_path
    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, device=device, dtype_name=args.dtype)
    manifest_entries = load_manifest_entries(args.manifest_path, selected_datasets)
    output_root = Path(args.output_root) / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    overview_rows: list[dict[str, Any]] = []
    dataset_manifest: dict[str, Any] = {}
    for entry in manifest_entries:
        dataset_name = str(entry["name"])
        raw_records = load_jsonl_records(entry["path"], max_examples=args.max_examples_per_dataset)
        examples = [normalize_example(tokenizer, record, dataset_name, entry) for record in raw_records]
        eval_records, summary = evaluate_dataset(
            model,
            tokenizer,
            examples,
            max_length=args.max_length,
            default_max_new_tokens=args.default_max_new_tokens,
            device=device,
        )
        dataset_dir = output_root / dataset_name
        write_jsonl(dataset_dir / "eval_records.jsonl", eval_records)
        (dataset_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        overview_rows.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "num_examples": summary["num_examples"],
                "accuracy": summary["accuracy"],
                "mean_score": summary["mean_score"],
                "mean_token_f1": summary["mean_token_f1"],
                "en_accuracy": summary["by_language"].get("en", {}).get("accuracy"),
                "da_accuracy": summary["by_language"].get("da", {}).get("accuracy"),
                "en_mean_score": summary["by_language"].get("en", {}).get("mean_score"),
                "da_mean_score": summary["by_language"].get("da", {}).get("mean_score"),
            }
        )
        dataset_manifest[dataset_name] = {
            "num_examples": len(examples),
            "eval_records_path": str(dataset_dir / "eval_records.jsonl"),
            "eval_summary_path": str(dataset_dir / "eval_summary.json"),
        }

    write_csv(output_root / "expert_eval_overview.csv", overview_rows)
    (output_root / "expert_eval_manifest.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_path": model_path,
                "datasets": dataset_manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote standalone expert eval to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
