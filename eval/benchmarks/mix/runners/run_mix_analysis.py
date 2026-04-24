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


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "full"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the mix benchmark suite for one FlexOlmo checkpoint. "
            "The model is loaded once and evaluated across all datasets listed in the shared manifest."
        )
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include from the manifest.")
    parser.add_argument("--max-examples-per-dataset", type=int)
    parser.add_argument("--model-path", help="Explicit path or HF identifier for the FlexOlmo checkpoint.")
    parser.add_argument("--model-name", help="Model name from model_paths/all_models.txt.")
    parser.add_argument("--model-root", help="Directory containing model folders on UCloud.")
    parser.add_argument("--tokenizer-path", help="Optional tokenizer path. Defaults to the resolved model path.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
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
    )
    parser.add_argument("--expert-order")
    parser.add_argument("--include-individual-experts", action="store_true")
    parser.add_argument("--capture-router-tensors", action="store_true")
    parser.add_argument("--capture-hidden-states", action="store_true")
    parser.add_argument(
        "--hidden-state-layers",
        help=(
            "Comma-separated hidden-state layer indices to save. Supports negative indices like -1 "
            "or presets like `early_mid_late_last`."
        ),
    )
    parser.add_argument("--skip-output-token-capture", action="store_true")
    parser.add_argument("--default-max-new-tokens", type=int, default=16)
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


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _fractional_positions(count: int, fractions: tuple[float, ...]) -> list[int]:
    if count <= 1:
        return [0]
    return _dedupe_preserve_order(
        [min(max(int(round(fraction * (count - 1))), 0), count - 1) for fraction in fractions]
    )


def parse_hidden_state_layers(raw_value: str | None, *, num_hidden_layers: int | None = None) -> list[int] | None:
    if not raw_value:
        return None
    normalized = raw_value.strip().lower()
    if normalized == "early_mid_late_last":
        if num_hidden_layers is None:
            raise ValueError("`num_hidden_layers` is required to resolve layer preset `early_mid_late_last`.")
        return _fractional_positions(num_hidden_layers + 1, (0.0, 0.33, 0.66, 1.0))
    if normalized == "early_mid_last":
        if num_hidden_layers is None:
            raise ValueError("`num_hidden_layers` is required to resolve layer preset `early_mid_last`.")
        return _fractional_positions(num_hidden_layers + 1, (0.0, 0.5, 1.0))
    if normalized == "early_late_last":
        if num_hidden_layers is None:
            raise ValueError("`num_hidden_layers` is required to resolve layer preset `early_late_last`.")
        return _fractional_positions(num_hidden_layers + 1, (0.0, 0.75, 1.0))
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


def load_jsonl_records(path: str | Path, max_examples: int | None = None) -> list[dict]:
    records = []
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_examples is not None and len(records) >= max_examples:
                break
    return records


def apply_chat_template_if_requested(tokenizer, prompt: str, prompting_config: dict) -> str:
    chat_template_config = dict(prompting_config.get("chat_template", {}))
    if not chat_template_config.get("enabled", False):
        return prompt

    user_template = str(chat_template_config.get("user_template", "{prompt}"))
    user_content = user_template.format(prompt=prompt)
    messages = []

    system_prompt = chat_template_config.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": user_content})

    add_generation_prompt = bool(chat_template_config.get("add_generation_prompt", True))
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def normalize_example(tokenizer, record: dict, dataset_name: str, dataset_entry: dict) -> dict:
    prompt = record.get("prompt")
    if not prompt:
        raise ValueError(f"Dataset `{dataset_name}` contains a record without `prompt`.")
    example_id = record.get("example_id")
    if not example_id:
        raise ValueError(f"Dataset `{dataset_name}` contains a record without `example_id`.")

    prompting_config = dict(dataset_entry.get("prompting", {}))
    generation_config = dict(dataset_entry.get("generation", {}))
    tokenization_config = dict(dataset_entry.get("tokenization", {}))
    normalized_prompt = apply_chat_template_if_requested(tokenizer, prompt, prompting_config)

    normalized = dict(record)
    normalized.setdefault("dataset_name", dataset_name)
    normalized.setdefault("dataset", dataset_name)
    normalized.setdefault("benchmark", record.get("source_benchmark", dataset_name))
    normalized["prompt"] = normalized_prompt
    normalized["prompting_config"] = prompting_config
    normalized["generation_config"] = generation_config
    normalized["tokenization_config"] = tokenization_config
    return normalized


def load_manifest_entries(path: str | Path, selected_datasets: set[str] | None) -> list[dict]:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("datasets", [])
    if selected_datasets is None:
        return entries
    return [entry for entry in entries if entry.get("name") in selected_datasets]


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model_path = resolve_model_path(args)
    model_name = resolved_model_name(args, model_path)
    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
        dtype_name=args.dtype,
    )
    hidden_state_layers = parse_hidden_state_layers(
        args.hidden_state_layers,
        num_hidden_layers=int(model.config.num_hidden_layers),
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

    manifest_entries = load_manifest_entries(args.manifest_path, selected_datasets)
    if not manifest_entries:
        raise ValueError("No mix datasets were selected from the manifest.")

    model_output_root = Path(args.output_root) / model_name
    model_output_root.mkdir(parents=True, exist_ok=True)

    suite_manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "tokenizer_path": args.tokenizer_path or model_path,
        "model_impl_path": str(inspect.getsourcefile(FlexOlmoForCausalLM)),
        "device": str(device),
        "dtype": args.dtype,
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "routing_run_mode": args.routing_run_mode,
        "model_native_top_k": int(model.config.num_experts_per_tok),
        "model_num_experts": int(model.config.num_experts),
        "capture_output_token_ids": not args.skip_output_token_capture,
        "default_max_new_tokens": args.default_max_new_tokens,
        "capture_router_tensors": args.capture_router_tensors,
        "capture_hidden_states": args.capture_hidden_states,
        "hidden_state_layers": hidden_state_layers,
        "run_labels": [run_spec.label for run_spec in run_specs],
        "datasets": {},
    }

    for dataset_entry in manifest_entries:
        dataset_name = str(dataset_entry["name"])
        dataset_path = Path(dataset_entry["path"])
        records = load_jsonl_records(dataset_path, max_examples=args.max_examples_per_dataset)
        examples = [
            normalize_example(
                tokenizer=tokenizer,
                record=record,
                dataset_name=dataset_name,
                dataset_entry=dataset_entry,
            )
            for record in records
        ]
        if not examples:
            raise ValueError(f"No examples were loaded for dataset `{dataset_name}` from {dataset_path}.")

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

        dataset_manifest = save_dataset_run_outputs(
            output_root=model_output_root,
            dataset_name=dataset_name,
            run_results=run_results,
            sort_keys=False,
        )

        run_manifest_path = model_output_root / dataset_name / "run_manifest.json"
        run_manifest = {
            "model_name": model_name,
            "model_path": model_path,
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "num_examples": len(examples),
            "domain": dataset_entry.get("domain"),
            "scoring_mode": dataset_entry.get("scoring_mode"),
            "prompting": dataset_entry.get("prompting", {}),
            "generation": dataset_entry.get("generation", {}),
            "tokenization": dataset_entry.get("tokenization", {}),
            "routing_run_mode": args.routing_run_mode,
            "run_labels": [run_spec.label for run_spec in run_specs],
            "runs": dataset_manifest,
        }
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")

        suite_manifest["datasets"][dataset_name] = {
            "path": str(dataset_path),
            "domain": dataset_entry.get("domain"),
            "scoring_mode": dataset_entry.get("scoring_mode"),
            "num_examples": len(examples),
            "prompting": dataset_entry.get("prompting", {}),
            "generation": dataset_entry.get("generation", {}),
            "tokenization": dataset_entry.get("tokenization", {}),
            "run_manifest_path": str(run_manifest_path),
        }

        print(
            f"Finished routing diagnostics for {model_name} on {dataset_name} "
            f"({len(examples)} examples)."
        )

    suite_manifest_path = model_output_root / "mix_suite_manifest.json"
    suite_manifest_path.write_text(json.dumps(suite_manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Using FlexOlmo implementation from {suite_manifest['model_impl_path']}")
    print(f"Resolved model: {model_name} -> {model_path}")
    print("Configured routing runs: " + ", ".join(suite_manifest["run_labels"]))
    print(f"Wrote mix suite manifest to {suite_manifest_path}")


if __name__ == "__main__":
    main()
