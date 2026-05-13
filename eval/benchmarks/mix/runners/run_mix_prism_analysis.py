from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
    parse_hidden_state_layers,
    resolve_device,
)
from eval.benchmarks.mix.runners.run_mix_router_direction import parse_decoder_layers


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "prism_analysis" / "a4"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact prism-style component analysis for the FlexOlmo 55B pair."
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include from the manifest.")
    parser.add_argument("--max-examples-per-dataset", type=int, default=50)
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
        default="early_mid_late_last",
        help="Comma-separated decoder layer indices or presets like `early_mid_late_last`.",
    )
    parser.add_argument("--public-expert-idx", type=int, default=0)
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
    reference_answer = record.get("reference_answer")
    if not prompt or not reference_answer:
        raise ValueError(f"Dataset `{dataset_name}` requires both `prompt` and `reference_answer` for prism analysis.")
    prompting_config = dict(dataset_entry.get("prompting", {}))
    return {
        "example_id": record["example_id"],
        "dataset_name": dataset_name,
        "language": record.get("language", "unknown"),
        "domain": record.get("domain", dataset_entry.get("domain")),
        "question": record.get("question"),
        "prompt": apply_chat_template_if_requested(tokenizer, prompt, prompting_config),
        "reference_answer": reference_answer,
    }


def build_teacher_forced_batch(
    tokenizer,
    prompt: str,
    reference_answer: str,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(reference_answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError("Reference answer tokenization produced zero tokens.")

    available_answer_tokens = max_length - len(prompt_ids)
    if available_answer_tokens <= 0:
        raise ValueError("Prompt exceeded max_length before answer tokens could be appended.")
    answer_ids = answer_ids[:available_answer_tokens]
    if not answer_ids:
        raise ValueError("Reference answer was fully truncated by max_length.")

    input_ids_list = prompt_ids + answer_ids
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    prompt_length = len(prompt_ids)
    sequence_length = len(input_ids_list)
    predictor_positions = torch.arange(prompt_length - 1, sequence_length - 1, device=device, dtype=torch.long)
    target_token_ids = input_ids[0, prompt_length:sequence_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_length": prompt_length,
        "answer_length": len(answer_ids),
        "sequence_length": sequence_length,
        "predictor_positions": predictor_positions,
        "target_token_ids": target_token_ids,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class PrismCapture:
    def __init__(self, selected_layers: list[int]):
        self.selected_layers = selected_layers
        self.block_inputs: dict[int, torch.Tensor] = {}
        self.attention_outputs: dict[int, torch.Tensor] = {}
        self.mlp_inputs: dict[int, torch.Tensor] = {}
        self.router_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.mlp_outputs: dict[int, torch.Tensor] = {}
        self.block_outputs: dict[int, torch.Tensor] = {}
        self.final_hidden_state: torch.Tensor | None = None
        self.handles = []

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))

        def make_block_input_hook(layer_idx: int):
            def hook(_module, args):
                self.block_inputs[layer_idx] = args[0].detach()
            return hook

        def make_attention_hook(layer_idx: int):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                self.attention_outputs[layer_idx] = tensor.detach()
            return hook

        def make_mlp_input_hook(layer_idx: int):
            def hook(_module, args):
                self.mlp_inputs[layer_idx] = args[0].detach()
            return hook

        def make_router_hook(layer_idx: int):
            def hook(_module, _args, output):
                router_logits, top_k_weights, top_k_index = output
                self.router_outputs[layer_idx] = (
                    router_logits.detach(),
                    top_k_weights.detach(),
                    top_k_index.detach(),
                )
            return hook

        def make_mlp_output_hook(layer_idx: int):
            def hook(_module, _args, output):
                self.mlp_outputs[layer_idx] = output.detach()
            return hook

        def make_block_output_hook(layer_idx: int):
            def hook(_module, _args, output):
                self.block_outputs[layer_idx] = output.detach()
            return hook

        def final_norm_hook(_module, _args, output):
            self.final_hidden_state = output.detach()

        for layer_idx in self.selected_layers:
            layer = layers[layer_idx]
            self.handles.append(layer.register_forward_pre_hook(make_block_input_hook(layer_idx)))
            self.handles.append(layer.self_attn.register_forward_hook(make_attention_hook(layer_idx)))
            self.handles.append(layer.mlp.register_forward_pre_hook(make_mlp_input_hook(layer_idx)))
            self.handles.append(layer.mlp.gate.register_forward_hook(make_router_hook(layer_idx)))
            self.handles.append(layer.mlp.register_forward_hook(make_mlp_output_hook(layer_idx)))
            self.handles.append(layer.register_forward_hook(make_block_output_hook(layer_idx)))
        self.handles.append(model.model.norm.register_forward_hook(final_norm_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def project_component_metrics(
    model,
    component_vectors: torch.Tensor,
    target_token_ids: torch.Tensor,
    apply_final_norm: bool,
) -> dict[str, float]:
    if component_vectors.ndim != 2:
        raise ValueError(f"Expected component vectors to have shape [num_positions, hidden], got {tuple(component_vectors.shape)}")
    with torch.no_grad():
        projected = model.model.norm(component_vectors) if apply_final_norm else component_vectors
        logits = model.lm_head(projected)
        row_idx = torch.arange(logits.shape[0], device=logits.device)
        target_logits = logits[row_idx, target_token_ids]
        target_ranks = (logits > target_logits.unsqueeze(-1)).sum(dim=-1).to(torch.float32) + 1.0
        target_is_top1 = (logits.argmax(dim=-1) == target_token_ids).to(torch.float32)
        masked_logits = logits.clone()
        masked_logits[row_idx, target_token_ids] = float("-inf")
        runner_up_logits = masked_logits.max(dim=-1).values
        target_margin = target_logits - runner_up_logits
        vector_norms = component_vectors.norm(dim=-1)
    return {
        "mean_target_logit": float(target_logits.mean().item()),
        "mean_target_rank": float(target_ranks.mean().item()),
        "top1_match_rate": float(target_is_top1.mean().item()),
        "mean_target_margin": float(target_margin.mean().item()),
        "mean_vector_norm": float(vector_norms.mean().item()),
    }


def compute_split_moe_outputs(layer, mlp_input: torch.Tensor, top_k_weights: torch.Tensor, top_k_index: torch.Tensor, public_expert_idx: int) -> dict[str, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = mlp_input.shape
    flat_inputs = mlp_input.reshape(-1, hidden_dim)
    flat_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])
    flat_indices = top_k_index.reshape(-1, top_k_index.shape[-1])

    expert_mask = F.one_hot(flat_indices, num_classes=layer.mlp.experts.num_experts).permute(2, 1, 0)
    combined = torch.zeros_like(flat_inputs)
    public = torch.zeros_like(flat_inputs)
    nonpublic = torch.zeros_like(flat_inputs)

    for expert_idx_tensor in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
        expert_idx = int(expert_idx_tensor[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = flat_inputs[token_idx]
        gate_up_weight = layer.mlp.experts.gate_up_proj[expert_idx]
        down_proj_weight = layer.mlp.experts.down_proj[expert_idx]
        gate, up = F.linear(current_state, gate_up_weight).chunk(2, dim=-1)
        current_hidden_states = layer.mlp.experts.act_fn(gate) * up
        current_hidden_states = F.linear(current_hidden_states, down_proj_weight)
        current_hidden_states = current_hidden_states * flat_weights[token_idx, top_k_pos, None]
        combined.index_add_(0, token_idx, current_hidden_states.to(combined.dtype))
        if expert_idx == public_expert_idx:
            public.index_add_(0, token_idx, current_hidden_states.to(public.dtype))
        else:
            nonpublic.index_add_(0, token_idx, current_hidden_states.to(nonpublic.dtype))

    return {
        "moe_out": combined.reshape(batch_size, sequence_length, hidden_dim),
        "public_moe_out": public.reshape(batch_size, sequence_length, hidden_dim),
        "nonpublic_moe_out": nonpublic.reshape(batch_size, sequence_length, hidden_dim),
    }


def layer_label(layer_idx: int, num_hidden_layers: int) -> str:
    if layer_idx < 0:
        return "embedding"
    if layer_idx == num_hidden_layers:
        return "final"
    return f"L{layer_idx}"


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["dataset_name"],
            record["language"],
            record["model_name"],
            int(record["layer_idx"]),
            record["component"],
        )
        grouped[key].append(record)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        dataset_name, language, model_name, layer_idx, component = key
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "language": language,
                "model_name": model_name,
                "layer_idx": layer_idx,
                "component": component,
                "num_examples": len(rows),
                "mean_target_logit": float(np.mean([row["mean_target_logit"] for row in rows])),
                "mean_target_rank": float(np.mean([row["mean_target_rank"] for row in rows])),
                "top1_match_rate": float(np.mean([row["top1_match_rate"] for row in rows])),
                "mean_target_margin": float(np.mean([row["mean_target_margin"] for row in rows])),
                "mean_vector_norm": float(np.mean([row["mean_vector_norm"] for row in rows])),
            }
        )
    return summary_rows


def analyze_example(
    model,
    tokenizer,
    example: dict[str, Any],
    selected_layers: list[int],
    max_length: int,
    public_expert_idx: int,
    num_hidden_layers: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    batch = build_teacher_forced_batch(
        tokenizer=tokenizer,
        prompt=example["prompt"],
        reference_answer=example["reference_answer"],
        max_length=max_length,
        device=device,
    )
    capture = PrismCapture(selected_layers)
    capture.attach(model)
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
    capture.remove()

    predictor_positions = batch["predictor_positions"]
    target_token_ids = batch["target_token_ids"]
    embedding = model.get_input_embeddings()(batch["input_ids"])[0]

    records: list[dict[str, Any]] = []
    component_specs = [
        {
            "component": "embedding",
            "layer_idx": -1,
            "layer_label": "embedding",
            "vectors": embedding[predictor_positions],
            "apply_final_norm": True,
        }
    ]

    layers = list(iter_flex_olmo_layers(model))
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        split_outputs = compute_split_moe_outputs(
            layer=layer,
            mlp_input=capture.mlp_inputs[layer_idx],
            top_k_weights=capture.router_outputs[layer_idx][1],
            top_k_index=capture.router_outputs[layer_idx][2],
            public_expert_idx=public_expert_idx,
        )
        component_specs.extend(
            [
                {
                    "component": "attention_out",
                    "layer_idx": layer_idx,
                    "layer_label": layer_label(layer_idx, num_hidden_layers),
                    "vectors": capture.attention_outputs[layer_idx][0, predictor_positions, :],
                    "apply_final_norm": True,
                },
                {
                    "component": "moe_out",
                    "layer_idx": layer_idx,
                    "layer_label": layer_label(layer_idx, num_hidden_layers),
                    "vectors": capture.mlp_outputs[layer_idx][0, predictor_positions, :],
                    "apply_final_norm": True,
                },
                {
                    "component": "public_moe_out",
                    "layer_idx": layer_idx,
                    "layer_label": layer_label(layer_idx, num_hidden_layers),
                    "vectors": split_outputs["public_moe_out"][0, predictor_positions, :],
                    "apply_final_norm": True,
                },
                {
                    "component": "nonpublic_moe_out",
                    "layer_idx": layer_idx,
                    "layer_label": layer_label(layer_idx, num_hidden_layers),
                    "vectors": split_outputs["nonpublic_moe_out"][0, predictor_positions, :],
                    "apply_final_norm": True,
                },
                {
                    "component": "block_update",
                    "layer_idx": layer_idx,
                    "layer_label": layer_label(layer_idx, num_hidden_layers),
                    "vectors": (
                        capture.block_outputs[layer_idx][0, predictor_positions, :]
                        - capture.block_inputs[layer_idx][0, predictor_positions, :]
                    ),
                    "apply_final_norm": True,
                },
            ]
        )

    if capture.final_hidden_state is None:
        raise ValueError("Final hidden-state hook did not fire.")
    component_specs.append(
        {
            "component": "final_hidden_state",
            "layer_idx": num_hidden_layers,
            "layer_label": "final",
            "vectors": capture.final_hidden_state[0, predictor_positions, :],
            "apply_final_norm": False,
        }
    )

    for spec in component_specs:
        metrics = project_component_metrics(
            model=model,
            component_vectors=spec["vectors"],
            target_token_ids=target_token_ids,
            apply_final_norm=bool(spec["apply_final_norm"]),
        )
        records.append(
            {
                "example_id": example["example_id"],
                "dataset_name": example["dataset_name"],
                "language": example["language"],
                "domain": example.get("domain"),
                "question": example.get("question"),
                "component": spec["component"],
                "layer_idx": int(spec["layer_idx"]),
                "layer_label": spec["layer_label"],
                "prompt_length": batch["prompt_length"],
                "answer_length": batch["answer_length"],
                "num_targets": int(target_token_ids.shape[0]),
                **metrics,
            }
        )

    final_logits = outputs.logits[0, predictor_positions, :]
    target_ids = target_token_ids
    row_idx = torch.arange(final_logits.shape[0], device=final_logits.device)
    target_logits = final_logits[row_idx, target_ids]
    target_ranks = (final_logits > target_logits.unsqueeze(-1)).sum(dim=-1).to(torch.float32) + 1.0
    target_is_top1 = (final_logits.argmax(dim=-1) == target_ids).to(torch.float32)
    masked_logits = final_logits.clone()
    masked_logits[row_idx, target_ids] = float("-inf")
    runner_up_logits = masked_logits.max(dim=-1).values
    target_margin = target_logits - runner_up_logits

    records.append(
        {
            "example_id": example["example_id"],
            "dataset_name": example["dataset_name"],
            "language": example["language"],
            "domain": example.get("domain"),
            "question": example.get("question"),
            "component": "model_logits",
            "layer_idx": num_hidden_layers,
            "layer_label": "final",
            "prompt_length": batch["prompt_length"],
            "answer_length": batch["answer_length"],
            "num_targets": int(target_ids.shape[0]),
            "mean_target_logit": float(target_logits.mean().item()),
            "mean_target_rank": float(target_ranks.mean().item()),
            "top1_match_rate": float(target_is_top1.mean().item()),
            "mean_target_margin": float(target_margin.mean().item()),
            "mean_vector_norm": np.nan,
        }
    )

    return records


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
    num_hidden_layers = int(model.config.num_hidden_layers)
    selected_layers = parse_decoder_layers(args.selected_layers, num_hidden_layers)

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
        "public_expert_idx": args.public_expert_idx,
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
        dataset_dir = model_output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        all_records: list[dict[str, Any]] = []
        skipped_examples: list[dict[str, Any]] = []
        for example in examples:
            try:
                all_records.extend(
                    analyze_example(
                        model=model,
                        tokenizer=tokenizer,
                        example=example,
                        selected_layers=selected_layers,
                        max_length=args.max_length,
                        public_expert_idx=args.public_expert_idx,
                        num_hidden_layers=num_hidden_layers,
                        device=device,
                    )
                )
            except Exception as exc:  # pragma: no cover - keep analysis running on odd examples
                skipped_examples.append(
                    {
                        "example_id": example["example_id"],
                        "error": str(exc),
                    }
                )

        summary_rows = summarize_records([{**record, "model_name": model_name} for record in all_records])
        write_csv(
            dataset_dir / "prism_component_records.csv",
            [{**record, "model_name": model_name} for record in all_records],
        )
        write_csv(dataset_dir / "prism_component_summary.csv", summary_rows)
        (dataset_dir / "skipped_examples.json").write_text(
            json.dumps(skipped_examples, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        suite_manifest["datasets"][dataset_name] = {
            "dataset_path": str(Path(dataset_entry["path"]).resolve()),
            "num_examples_loaded": len(examples),
            "num_records": len(all_records),
            "num_skipped_examples": len(skipped_examples),
            "records_path": str(dataset_dir / "prism_component_records.csv"),
            "summary_path": str(dataset_dir / "prism_component_summary.csv"),
        }
        print(f"Wrote prism analysis for {model_name} on {dataset_name} to {dataset_dir}")

    (model_output_root / "prism_analysis_suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote prism suite manifest to {model_output_root / 'prism_analysis_suite_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
