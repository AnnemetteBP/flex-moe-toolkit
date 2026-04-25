from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import FlexOlmoForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.adapters.flex_olmo import iter_flex_olmo_layers
from flex_moe_toolkit.utils import load_tokenizer_with_known_fixes
from eval.benchmarks.mix.runners.run_mix_analysis import (
    apply_chat_template_if_requested,
    load_allowed_model_names,
    load_jsonl_records,
    load_manifest_entries,
    parse_dtype,
    resolve_device,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "router_direction" / "a4"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run router-direction / expert-alignment analysis for a FlexOlmo checkpoint."
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
    parser.add_argument("--selected-layers", default="early_mid_late_last")
    parser.add_argument(
        "--position-policy",
        default="last_prompt_token",
        choices=("last_prompt_token", "mean_prompt"),
    )
    parser.add_argument(
        "--alignment-metric",
        default="cosine",
        choices=("cosine", "dot"),
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


def parse_decoder_layers(raw_value: str, num_hidden_layers: int) -> list[int]:
    normalized = raw_value.strip().lower()
    if normalized == "early_mid_late_last":
        if num_hidden_layers <= 1:
            return [0]
        return sorted(
            {
                0,
                int(round((num_hidden_layers - 1) * 0.33)),
                int(round((num_hidden_layers - 1) * 0.66)),
                num_hidden_layers - 1,
            }
        )
    if normalized == "early_mid_last":
        if num_hidden_layers <= 1:
            return [0]
        return sorted({0, int(round((num_hidden_layers - 1) * 0.5)), num_hidden_layers - 1})
    if normalized == "early_late_last":
        if num_hidden_layers <= 1:
            return [0]
        return sorted({0, int(round((num_hidden_layers - 1) * 0.75)), num_hidden_layers - 1})
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


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


def encode_prompt(tokenizer, prompt: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def reduce_tokens(tensor: torch.Tensor, prompt_length: int, position_policy: str) -> torch.Tensor:
    prompt_tensor = tensor[0, :prompt_length, :]
    if position_policy == "last_prompt_token":
        return prompt_tensor[-1].detach().cpu().float()
    if position_policy == "mean_prompt":
        return prompt_tensor.mean(dim=0).detach().cpu().float()
    raise ValueError(f"Unsupported position policy `{position_policy}`.")


def summarize_router_choice(
    router_probs: torch.Tensor,
    top_k_index: torch.Tensor | None,
    prompt_length: int,
    position_policy: str,
) -> dict[str, object]:
    prompt_probs = router_probs[0, :prompt_length, :].detach().cpu().float()
    if position_policy == "last_prompt_token":
        selected_probs = prompt_probs[-1]
        selected_topk = top_k_index[prompt_length - 1].detach().cpu().tolist() if top_k_index is not None else []
    elif position_policy == "mean_prompt":
        selected_probs = prompt_probs.mean(dim=0)
        selected_topk = []
    else:
        raise ValueError(f"Unsupported position policy `{position_policy}`.")
    top_values, top_indices = torch.topk(selected_probs, k=min(2, selected_probs.shape[-1]), dim=-1)
    top1 = int(top_indices[0].item())
    top2 = int(top_indices[1].item()) if top_indices.shape[0] > 1 else None
    return {
        "actual_top1_expert": top1,
        "actual_top2_expert": top2,
        "actual_topk_experts": selected_topk,
        "actual_top1_prob": float(top_values[0].item()),
        "actual_top2_prob": float(top_values[1].item()) if top_values.shape[0] > 1 else 0.0,
    }


class RouterDirectionCapture:
    def __init__(self, selected_layers: list[int], prompt_length: int, position_policy: str):
        self.selected_layers = selected_layers
        self.prompt_length = prompt_length
        self.position_policy = position_policy
        self.pre_router_states: dict[int, torch.Tensor] = {}
        self.router_outputs: dict[int, dict[str, torch.Tensor | None]] = {}
        self.handles = []

    def _make_pre_hook(self, layer_idx: int):
        def hook(_module, args):
            hidden_states = args[0]
            self.pre_router_states[layer_idx] = reduce_tokens(hidden_states, self.prompt_length, self.position_policy)
        return hook

    def _make_router_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            router_probs = output[0].detach().cpu().float()
            top_k_index = output[2].detach().cpu() if len(output) > 2 else None
            seq_len = router_probs.shape[0]
            reshaped_probs = router_probs.unsqueeze(0) if router_probs.ndim == 2 else router_probs
            reshaped_topk = top_k_index if top_k_index is None else top_k_index.reshape(seq_len, -1)
            self.router_outputs[layer_idx] = {
                "router_probs": reshaped_probs,
                "top_k_index": reshaped_topk,
            }
        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            self.handles.append(layers[layer_idx].mlp.register_forward_pre_hook(self._make_pre_hook(layer_idx)))
            self.handles.append(layers[layer_idx].mlp.gate.register_forward_hook(self._make_router_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def compute_alignment_scores(state: torch.Tensor, weight_matrix: torch.Tensor, metric: str) -> torch.Tensor:
    state = state.float()
    weights = weight_matrix.float()
    if metric == "dot":
        return torch.mv(weights, state)
    if metric == "cosine":
        normalized_state = state / torch.linalg.vector_norm(state).clamp_min(1e-9)
        normalized_weights = weights / torch.linalg.vector_norm(weights, dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.mv(normalized_weights, normalized_state)
    raise ValueError(f"Unsupported alignment metric `{metric}`.")


def alignment_entropy(scores: torch.Tensor) -> float:
    probs = torch.softmax(scores, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum()
    return float(entropy.item())


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for record in records:
        key = (str(record["dataset_name"]), str(record.get("language", "unknown")), int(record["layer"]))
        grouped[key].append(record)

    summaries: list[dict] = []
    for (dataset_name, language, layer), items in sorted(grouped.items()):
        summaries.append(
            {
                "record_type": "router_direction_summary",
                "dataset_name": dataset_name,
                "language": language,
                "layer": layer,
                "num_examples": len(items),
                "mean_top1_alignment": sum(float(item["top1_alignment"]) for item in items) / len(items),
                "mean_top2_alignment": sum(float(item["top2_alignment"]) for item in items) / len(items),
                "mean_alignment_margin": sum(float(item["alignment_margin"]) for item in items) / len(items),
                "mean_alignment_entropy": sum(float(item["alignment_entropy"]) for item in items) / len(items),
                "top1_agreement_rate": (
                    sum(1.0 for item in items if item.get("agreement_top1")) / len(items)
                ),
            }
        )
    return summaries


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
    selected_layers = parse_decoder_layers(args.selected_layers, int(model.config.num_hidden_layers))
    manifest_entries = load_manifest_entries(args.manifest_path, selected_datasets)
    if not manifest_entries:
        raise ValueError("No mix datasets were selected from the manifest.")

    model_output_root = Path(args.output_root) / model_name
    model_output_root.mkdir(parents=True, exist_ok=True)

    router_weights = {}
    for layer_idx in selected_layers:
        layer = list(iter_flex_olmo_layers(model))[layer_idx]
        router_weights[f"layer_{layer_idx}_weights"] = layer.mlp.gate.weight.detach().cpu().float().numpy()
    np.savez_compressed(model_output_root / "router_weights.npz", **router_weights)

    suite_manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "selected_layers": selected_layers,
        "alignment_metric": args.alignment_metric,
        "position_policy": args.position_policy,
        "max_examples_per_dataset": args.max_examples_per_dataset,
        "router_weights_path": str(model_output_root / "router_weights.npz"),
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

        dataset_records: list[dict] = []
        for example in examples:
            inputs = encode_prompt(tokenizer, example["prompt"], max_length=args.max_length, device=device)
            prompt_length = int(inputs["input_ids"].shape[-1])
            capture = RouterDirectionCapture(selected_layers, prompt_length=prompt_length, position_policy=args.position_policy)
            capture.attach(model)
            try:
                with torch.no_grad():
                    model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        use_cache=False,
                    )
            finally:
                capture.remove()

            for layer_idx in selected_layers:
                state = capture.pre_router_states[layer_idx]
                weight_matrix = torch.from_numpy(router_weights[f"layer_{layer_idx}_weights"])
                scores = compute_alignment_scores(state, weight_matrix, metric=args.alignment_metric)
                top_scores, top_indices = torch.topk(scores, k=min(2, scores.shape[0]), dim=-1)
                routing_info = summarize_router_choice(
                    router_probs=capture.router_outputs[layer_idx]["router_probs"],
                    top_k_index=capture.router_outputs[layer_idx]["top_k_index"],
                    prompt_length=prompt_length,
                    position_policy=args.position_policy,
                )
                top1_expert = int(top_indices[0].item())
                top2_expert = int(top_indices[1].item()) if top_indices.shape[0] > 1 else None
                actual_top1 = routing_info["actual_top1_expert"]
                dataset_records.append(
                    {
                        "record_type": "router_direction_record",
                        "example_id": example["example_id"],
                        "dataset_name": dataset_name,
                        "language": example["language"],
                        "model_name": model_name,
                        "model_path": model_path,
                        "layer": layer_idx,
                        "alignment_metric": args.alignment_metric,
                        "position_policy": args.position_policy,
                        "top1_aligned_expert": top1_expert,
                        "top2_aligned_expert": top2_expert,
                        "top1_alignment": float(top_scores[0].item()),
                        "top2_alignment": float(top_scores[1].item()) if top_scores.shape[0] > 1 else 0.0,
                        "alignment_margin": float((top_scores[0] - top_scores[1]).item()) if top_scores.shape[0] > 1 else 0.0,
                        "alignment_entropy": alignment_entropy(scores),
                        "actual_topk_experts": routing_info["actual_topk_experts"],
                        "actual_top1_expert": actual_top1,
                        "actual_top2_expert": routing_info["actual_top2_expert"],
                        "actual_top1_prob": routing_info["actual_top1_prob"],
                        "actual_top2_prob": routing_info["actual_top2_prob"],
                        "agreement_top1": bool(actual_top1 == top1_expert) if actual_top1 is not None else None,
                    }
                )

        dataset_dir = model_output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        records_path = dataset_dir / "router_direction_records.jsonl"
        summary_path = dataset_dir / "router_direction_summary.jsonl"
        write_jsonl(records_path, dataset_records)
        write_jsonl(summary_path, build_summary(dataset_records))
        manifest = {
            "dataset_name": dataset_name,
            "num_examples": len(examples),
            "records_path": str(records_path),
            "summary_path": str(summary_path),
            "selected_layers": selected_layers,
            "alignment_metric": args.alignment_metric,
            "position_policy": args.position_policy,
        }
        (dataset_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        suite_manifest["datasets"][dataset_name] = manifest
        print(f"Wrote router-direction outputs for {model_name} on {dataset_name} ({len(examples)} examples)")

    (model_output_root / "router_direction_suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote router-direction suite manifest to {model_output_root / 'router_direction_suite_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
