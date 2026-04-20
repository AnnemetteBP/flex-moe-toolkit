from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import torch
from transformers import FlexOlmoForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.pipelines.flex_olmo_eval import build_run_specs
from flex_moe_toolkit.pipelines.flex_olmo_routing_dataset import (
    analyze_prompt_dataset_across_runs,
    save_dataset_run_outputs,
)
from flex_moe_toolkit.utils import load_tokenizer_with_known_fixes


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mkqa_results" / "routing"
DEFAULT_DATA_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mkqa" / "data" / "mkqa_da_en_subset.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run prompt-level routing analysis for EN/DA MKQA data on a real FlexOlmo checkpoint."
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-path", help="Explicit path or HF identifier for the FlexOlmo checkpoint.")
    parser.add_argument("--model-name", help="Model name from model_paths/all_models.txt.")
    parser.add_argument("--model-root", help="Directory containing model folders on UCloud.")
    parser.add_argument("--tokenizer-path", help="Optional tokenizer path. Defaults to the resolved model path.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
    parser.add_argument("--languages", default="en,da")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--public-expert-idx", type=int, default=0)
    parser.add_argument("--combined-active-experts", default="2,4,7")
    parser.add_argument(
        "--routing-run-mode",
        default="restricted_sweep",
        choices=("restricted_sweep", "native_only", "native_plus_restricted"),
        help=(
            "How to construct evaluation runs. "
            "`native_only` evaluates the checkpoint as-is, which is the correct mode for real a2/a4/a7 checkpoints. "
            "`restricted_sweep` preserves the older subset-masking behavior. "
            "`native_plus_restricted` runs both."
        ),
    )
    parser.add_argument("--expert-order")
    parser.add_argument("--include-individual-experts", action="store_true")
    parser.add_argument(
        "--capture-router-tensors",
        action="store_true",
        help="Save raw prompt router logits and probabilities per layer for later analysis.",
    )
    parser.add_argument(
        "--capture-hidden-states",
        action="store_true",
        help="Save hidden states and token-wise hidden-state norms for selected layers.",
    )
    parser.add_argument(
        "--hidden-state-layers",
        help="Comma-separated hidden-state layer indices to save. Supports negative indices like -1.",
    )
    parser.add_argument(
        "--skip-output-token-capture",
        action="store_true",
        help="Skip greedy continuation capture. By default the runner stores predicted and ground-truth output token ids.",
    )
    parser.add_argument(
        "--default-max-new-tokens",
        type=int,
        default=16,
        help="Fallback generation length for examples without a reference answer.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def resolved_model_name(args, model_path: str) -> str:
    if args.model_name:
        return args.model_name
    return Path(model_path).name


def with_leading_model_fields(record: dict, model_name: str, model_path: str) -> dict:
    return {
        "model_name": model_name,
        "model_path": model_path,
        **record,
    }


def parse_hidden_state_layers(raw_value: str | None) -> list[int] | None:
    if not raw_value:
        return None
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def load_allowed_model_names(path: str | Path) -> set[str]:
    path = Path(path)
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def resolve_model_path(args) -> str:
    if args.model_path:
        return args.model_path
    if not args.model_name or not args.model_root:
        raise ValueError("Provide either --model-path or both --model-name and --model-root.")

    allowed_names = load_allowed_model_names(args.model_registry)
    if args.model_name not in allowed_names:
        raise ValueError(f"Model name `{args.model_name}` was not found in {args.model_registry}.")

    return str(Path(args.model_root) / args.model_name)


def load_model_and_tokenizer(model_path: str, tokenizer_path: str | None, device: torch.device, dtype_name: str):
    model = FlexOlmoForCausalLM.from_pretrained(
        model_path,
        torch_dtype=parse_dtype(dtype_name),
    )
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer_with_known_fixes(tokenizer_path or model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer must define either `pad_token_id` or `eos_token_id` for prompt tokenization."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_mkqa_examples(path: str | Path, languages: list[str], max_examples: int | None = None):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized = []

    prompt_templates = {
        "en": "Question: {question}\nAnswer:",
        "da": "Spørgsmål: {question}\nSvar:",
    }
    per_language_counts = {language: 0 for language in languages}

    for index, record in enumerate(records):
        for language in languages:
            question_key = f"question_{language}"
            answer_key = f"answer_{language}"
            question = record.get(question_key)
            answer = record.get(answer_key)
            if not question:
                continue
            if max_examples is not None and per_language_counts[language] >= max_examples:
                continue

            normalized.append(
                {
                    "example_id": f"mkqa_{language}_{index:05d}",
                    "language": language,
                    "question": question,
                    "reference_answer": answer,
                    "prompt": prompt_templates.get(language, "Question: {question}\nAnswer:").format(
                        question=question
                    ),
                }
            )
            per_language_counts[language] += 1

    return normalized


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model_path = resolve_model_path(args)
    model_name = resolved_model_name(args, model_path)
    languages = [language.strip() for language in args.languages.split(",") if language.strip()]
    hidden_state_layers = parse_hidden_state_layers(args.hidden_state_layers)
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
        dtype_name=args.dtype,
    )

    expert_order = None
    if args.expert_order:
        expert_order = tuple(int(part.strip()) for part in args.expert_order.split(",") if part.strip())

    run_specs = build_run_specs(
        num_experts=model.config.num_experts,
        public_expert_idx=args.public_expert_idx,
        combined_active_counts=tuple(
            int(part.strip()) for part in args.combined_active_experts.split(",") if part.strip()
        ),
        include_individual_experts=args.include_individual_experts,
        expert_order=expert_order,
        routing_run_mode=args.routing_run_mode,
    )

    examples = load_mkqa_examples(
        path=args.data_path,
        languages=languages,
        max_examples=args.max_examples,
    )
    if not examples:
        raise ValueError("No MKQA examples were loaded after language and subsample filtering.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_name = "mkqa_" + "_".join(languages)

    run_results = analyze_prompt_dataset_across_runs(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        run_specs=run_specs,
        max_length=args.max_length,
        device=device,
        capture_output_token_ids=not args.skip_output_token_capture,
        default_max_new_tokens=args.default_max_new_tokens,
        capture_router_tensors=args.capture_router_tensors,
        capture_hidden_states=args.capture_hidden_states,
        hidden_state_layers=hidden_state_layers,
    )
    for payload in run_results.values():
        payload["records"] = [
            with_leading_model_fields(record, model_name=model_name, model_path=model_path)
            for record in payload["records"]
        ]
        payload["summaries"] = [
            with_leading_model_fields(record, model_name=model_name, model_path=model_path)
            for record in payload["summaries"]
        ]
        payload["routing_analysis_records"] = [
            with_leading_model_fields(record, model_name=model_name, model_path=model_path)
            for record in payload["routing_analysis_records"]
        ]
    manifest = save_dataset_run_outputs(
        output_root=output_root,
        dataset_name=dataset_name,
        run_results=run_results,
        sort_keys=False,
    )

    overall_manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "tokenizer_path": args.tokenizer_path or model_path,
        "model_impl_path": str(inspect.getsourcefile(FlexOlmoForCausalLM)),
        "device": str(device),
        "dtype": args.dtype,
        "languages": languages,
        "num_examples": len(examples),
        "routing_run_mode": args.routing_run_mode,
        "model_native_top_k": int(model.config.num_experts_per_tok),
        "model_num_experts": int(model.config.num_experts),
        "capture_output_token_ids": not args.skip_output_token_capture,
        "default_max_new_tokens": args.default_max_new_tokens,
        "capture_router_tensors": args.capture_router_tensors,
        "capture_hidden_states": args.capture_hidden_states,
        "hidden_state_layers": hidden_state_layers,
        "run_labels": [run_spec.label for run_spec in run_specs],
        "runs": manifest,
    }
    manifest_path = output_root / dataset_name / "run_manifest.json"
    manifest_path.write_text(json.dumps(overall_manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Using FlexOlmo implementation from {overall_manifest['model_impl_path']}")
    print(f"Resolved model: {model_name} -> {model_path}")
    print(f"Loaded {len(examples)} MKQA prompt examples across languages: {', '.join(languages)}")
    print("Configured routing runs: " + ", ".join(overall_manifest["run_labels"]))
    print(f"Wrote MKQA routing analysis manifest to {manifest_path}")


if __name__ == "__main__":
    main()
