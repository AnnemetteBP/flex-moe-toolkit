from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import numpy as np
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
    parse_hidden_state_layers,
    resolve_device,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "expert_sweep" / "latent_space"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture compact hidden-state prompt latents for standalone expert/public checkpoints."
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
    parser.add_argument(
        "--selected-layers",
        default="early_mid_late_last",
        help="Comma-separated hidden-state layer indices to save.",
    )
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


def normalize_example(tokenizer, record: dict[str, Any], dataset_name: str, dataset_entry: dict) -> dict[str, Any]:
    prompt = record.get("prompt")
    if not prompt:
        raise ValueError(f"Dataset `{dataset_name}` contains a record without `prompt`.")
    prompting_config = dict(dataset_entry.get("prompting", {}))
    return {
        "example_id": record["example_id"],
        "dataset_name": dataset_name,
        "language": record.get("language", "unknown"),
        "prompt": apply_chat_template_if_requested(tokenizer, prompt, prompting_config),
        "question": record.get("question"),
    }


def encode_prompt(tokenizer, prompt: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def select_hidden_state_layers(hidden_states, selected_layers: list[int]) -> dict[int, torch.Tensor]:
    num_layers = len(hidden_states)
    result: dict[int, torch.Tensor] = {}
    for layer_idx in selected_layers:
        normalized_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        if normalized_idx < 0 or normalized_idx >= num_layers:
            raise ValueError(f"Hidden-state layer index {layer_idx} is out of range for {num_layers} tensors.")
        if normalized_idx not in result:
            result[normalized_idx] = hidden_states[normalized_idx]
    return result


def capture_dataset_latents(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    selected_layers: list[int],
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    vectors: dict[int, dict[str, list[np.ndarray]]] = {
        layer: {"mean": [], "last": []}
        for layer in selected_layers
    }
    metadata: list[dict[str, Any]] = []

    for example in examples:
        inputs = encode_prompt(tokenizer, example["prompt"], max_length=max_length, device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
                use_cache=False,
            )
        selected = select_hidden_state_layers(outputs.hidden_states, selected_layers)
        attention_mask = inputs.get("attention_mask")
        seq_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(inputs["input_ids"].shape[-1])
        for layer_idx, tensor in selected.items():
            layer_tensor = tensor[0, :seq_len, :].detach().cpu().float()
            vectors[layer_idx]["mean"].append(layer_tensor.mean(dim=0).numpy())
            vectors[layer_idx]["last"].append(layer_tensor[-1].numpy())
        metadata.append(
            {
                "example_id": example["example_id"],
                "dataset_name": example["dataset_name"],
                "language": example["language"],
                "num_input_tokens": seq_len,
            }
        )

    arrays: dict[str, np.ndarray] = {}
    for layer_idx, reductions in vectors.items():
        for reduction_name, items in reductions.items():
            arrays[f"hidden_state_layer_{layer_idx}_{reduction_name}"] = np.stack(items, axis=0).astype(np.float32)
    return arrays, metadata


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise ValueError("Model config must define `num_hidden_layers` for hidden-state capture.")
    selected_layers = parse_hidden_state_layers(args.selected_layers, num_hidden_layers=num_hidden_layers)
    output_root = Path(args.output_root) / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_manifest: dict[str, Any] = {}
    for entry in manifest_entries:
        dataset_name = str(entry["name"])
        raw_records = load_jsonl_records(entry["path"], max_examples=args.max_examples_per_dataset)
        examples = [normalize_example(tokenizer, record, dataset_name, entry) for record in raw_records]
        arrays, metadata = capture_dataset_latents(
            model,
            tokenizer,
            examples,
            selected_layers=selected_layers,
            max_length=args.max_length,
            device=device,
        )
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_dir / "prompt_latents.npz", **arrays)
        write_jsonl(dataset_dir / "metadata.jsonl", metadata)
        manifest = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "num_examples": len(metadata),
            "selected_layers": selected_layers,
            "arrays": sorted(arrays.keys()),
        }
        (dataset_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        dataset_manifest[dataset_name] = {
            "num_examples": len(metadata),
            "latents_path": str(dataset_dir / "prompt_latents.npz"),
            "metadata_path": str(dataset_dir / "metadata.jsonl"),
            "run_manifest_path": str(dataset_dir / "run_manifest.json"),
        }

    (output_root / "latent_space_suite_manifest.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_path": model_path,
                "datasets": dataset_manifest,
                "representation_sources": ["hidden_state"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote standalone expert latents to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
