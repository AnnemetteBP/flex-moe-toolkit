from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from transformers import FlexOlmoForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.utils import load_tokenizer_with_known_fixes
from flex_moe_toolkit.adapters.flex_olmo import iter_flex_olmo_layers
from eval.benchmarks.mix.runners.run_mix_analysis import (
    apply_chat_template_if_requested,
    load_jsonl_records,
    load_manifest_entries,
    load_allowed_model_names,
    parse_dtype,
    parse_hidden_state_layers,
    resolve_device,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "focused" / "55b_pair" / "latent_space"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture compact prompt-level latent-space representations for selected mix datasets."
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include from the manifest.")
    parser.add_argument("--max-examples-per-dataset", type=int, default=75)
    parser.add_argument("--model-path", help="Explicit path or HF identifier for the FlexOlmo checkpoint.")
    parser.add_argument("--model-name", help="Model name from model_paths/all_models.txt.")
    parser.add_argument("--model-root", help="Directory containing model folders on UCloud.")
    parser.add_argument("--tokenizer-path", help="Optional tokenizer path. Defaults to the resolved model path.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--selected-layers",
        default="0,8,16,24,-1",
        help=(
            "Comma-separated hidden-state layer indices to save. Supports negative indices like -1 "
            "or presets like `early_mid_late_last`."
        ),
    )
    parser.add_argument(
        "--representation-sources",
        default="hidden_state,pre_router",
        help="Comma-separated latent sources to save. Supported: hidden_state, pre_router.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


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
    model = FlexOlmoForCausalLM.from_pretrained(model_path, torch_dtype=parse_dtype(dtype_name))
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer_with_known_fixes(tokenizer_path or model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either `pad_token_id` or `eos_token_id`.")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def normalize_example(tokenizer, record: dict, dataset_name: str, dataset_entry: dict) -> dict:
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


def parse_representation_sources(raw_value: str) -> list[str]:
    sources = [part.strip() for part in raw_value.split(",") if part.strip()]
    allowed = {"hidden_state", "pre_router"}
    invalid = [source for source in sources if source not in allowed]
    if invalid:
        raise ValueError(f"Unsupported representation sources: {', '.join(invalid)}")
    if not sources:
        raise ValueError("Provide at least one representation source.")
    return sources


class PreRouterCapture:
    def __init__(self, selected_layers: list[int]):
        self.selected_layers = selected_layers
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, args):
            self.outputs[layer_idx] = args[0].detach().cpu().float()
        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            self.handles.append(layers[layer_idx].mlp.register_forward_pre_hook(self._make_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def encode_prompt(tokenizer, prompt: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def capture_dataset_latents(
    model,
    tokenizer,
    examples: list[dict],
    selected_layers: list[int],
    representation_sources: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    vectors: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {
        source: {
            layer: {"mean": [], "last": []}
            for layer in selected_layers
        }
        for source in representation_sources
    }
    metadata: list[dict] = []
    capture_pre_router = "pre_router" in representation_sources

    for example in examples:
        inputs = encode_prompt(tokenizer, example["prompt"], max_length=max_length, device=device)
        pre_router_capture = PreRouterCapture(selected_layers) if capture_pre_router else None
        if pre_router_capture is not None:
            pre_router_capture.attach(model)
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states="hidden_state" in representation_sources,
                    use_cache=False,
                )
        finally:
            if pre_router_capture is not None:
                pre_router_capture.remove()

        seq_len = int(inputs["input_ids"].shape[-1])
        metadata.append(
            {
                "example_id": example["example_id"],
                "dataset_name": example["dataset_name"],
                "language": example["language"],
                "question": example.get("question"),
                "num_input_tokens": seq_len,
            }
        )
        if "hidden_state" in representation_sources:
            selected_hidden_states = select_hidden_state_layers(outputs.hidden_states, selected_layers)
            for layer_idx, tensor in selected_hidden_states.items():
                token_states = tensor[0].detach().cpu().float()
                vectors["hidden_state"][layer_idx]["mean"].append(token_states.mean(dim=0).numpy())
                vectors["hidden_state"][layer_idx]["last"].append(token_states[-1].numpy())

        if capture_pre_router and pre_router_capture is not None:
            for layer_idx, tensor in pre_router_capture.outputs.items():
                token_states = tensor[0]
                vectors["pre_router"][layer_idx]["mean"].append(token_states.mean(dim=0).numpy())
                vectors["pre_router"][layer_idx]["last"].append(token_states[-1].numpy())

    arrays: dict[str, np.ndarray] = {}
    for source_name, source_layers in sorted(vectors.items()):
        for layer_idx in sorted(source_layers):
            arrays[f"{source_name}_layer_{layer_idx}_mean"] = np.stack(source_layers[layer_idx]["mean"], axis=0)
            arrays[f"{source_name}_layer_{layer_idx}_last"] = np.stack(source_layers[layer_idx]["last"], axis=0)
    return arrays, metadata


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}

    model_path = resolve_model_path(args)
    model_name = args.model_name or Path(model_path).name
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
        dtype_name=args.dtype,
    )
    selected_layers = parse_hidden_state_layers(
        args.selected_layers,
        num_hidden_layers=int(model.config.num_hidden_layers),
    )
    representation_sources = parse_representation_sources(args.representation_sources)
    if not selected_layers:
        raise ValueError("Provide at least one selected layer.")

    manifest_entries = load_manifest_entries(args.manifest_path, selected_datasets)
    if not manifest_entries:
        raise ValueError("No mix datasets were selected from the manifest.")

    model_output_root = Path(args.output_root) / model_name
    model_output_root.mkdir(parents=True, exist_ok=True)

    suite_manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "selected_layers": selected_layers,
        "representation_sources": representation_sources,
        "max_examples_per_dataset": args.max_examples_per_dataset,
        "datasets": {},
    }

    for dataset_entry in manifest_entries:
        dataset_name = str(dataset_entry["name"])
        records = load_jsonl_records(dataset_entry["path"], max_examples=args.max_examples_per_dataset)
        examples = [
            normalize_example(tokenizer, record=record, dataset_name=dataset_name, dataset_entry=dataset_entry)
            for record in records
        ]
        if not examples:
            continue

        arrays, metadata = capture_dataset_latents(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            selected_layers=selected_layers,
            representation_sources=representation_sources,
            max_length=args.max_length,
            device=device,
        )

        dataset_dir = model_output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        npz_path = dataset_dir / "prompt_latents.npz"
        metadata_path = dataset_dir / "metadata.jsonl"
        np.savez_compressed(npz_path, **arrays)
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in metadata:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest = {
            "dataset_name": dataset_name,
            "num_examples": len(metadata),
            "selected_layers": selected_layers,
            "representation_sources": representation_sources,
            "npz_path": str(npz_path),
            "metadata_path": str(metadata_path),
            "arrays": {key: list(value.shape) for key, value in arrays.items()},
        }
        (dataset_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        suite_manifest["datasets"][dataset_name] = manifest
        print(f"Wrote latent-space capture for {model_name} on {dataset_name} ({len(metadata)} examples)")

    (model_output_root / "latent_space_suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote latent-space suite manifest to {model_output_root / 'latent_space_suite_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
