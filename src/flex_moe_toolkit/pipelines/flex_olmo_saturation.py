from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import torch

from flex_moe_toolkit.adapters.flex_olmo import FlexOlmoAdapter
from flex_moe_toolkit.core.routing_diagnostics import compute_router_saturation_from_logits
from flex_moe_toolkit.pipelines.flex_olmo_eval import build_prompt_and_targets, normalize_eval_example


def infer_model_metadata(model_path: str) -> dict[str, object]:
    name = Path(model_path).name or model_path
    lower_name = name.lower()

    active_experts = None
    match = re.search(r"(?:^|[_-])a(\d+)(?:$|[_-])", lower_name)
    if match:
        active_experts = int(match.group(1))

    scale_b = None
    match = re.search(r"(\d+)b", lower_name)
    if match:
        scale_b = int(match.group(1))

    if re.search(r"(?:^|[_-])rt(?:$|[_-])", lower_name):
        variant = "rt"
    elif re.search(r"(?:^|[_-])v2(?:$|[_-])", lower_name):
        variant = "v2"
    elif re.search(r"(?:^|[_-])v1(?:$|[_-])", lower_name):
        variant = "v1"
    else:
        variant = "base"

    return {
        "model_name": name,
        "model_path": model_path,
        "active_experts_variant": active_experts,
        "training_variant": variant,
        "scale_b": scale_b,
        "is_router_tuned": variant == "rt",
    }


def build_saturation_prompt(example: dict[str, Any], example_index: int) -> tuple[dict[str, Any], str]:
    normalized = normalize_eval_example(example, example_index=example_index)
    prompt, _continuations, _target_idx = build_prompt_and_targets(normalized)
    return normalized, prompt


def tokenize_prompt(tokenizer, prompt: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def collect_router_logits(model, adapter, inputs) -> tuple[torch.Tensor, ...]:
    router_logits = adapter.get_router_logits(model, inputs)
    normalized_logits = []

    for layer_router_logits in router_logits:
        normalized = layer_router_logits
        if normalized.ndim == 2:
            normalized = normalized.unsqueeze(0)
        normalized_logits.append(normalized.detach().cpu())

    return tuple(normalized_logits)


def _normalize_top_k_values(top_k, default_top_k: int) -> list[int]:
    if top_k is None:
        return [default_top_k]
    if isinstance(top_k, int):
        return [top_k]
    return sorted(set(int(k) for k in top_k))


def compute_example_router_saturation(
    checkpoint_model,
    final_model,
    tokenizer,
    example: dict[str, Any],
    example_index: int,
    device: torch.device,
    max_length: int,
    top_k=None,
) -> dict[str, Any]:
    adapter = FlexOlmoAdapter()
    normalized, prompt = build_saturation_prompt(example, example_index=example_index)
    inputs = tokenize_prompt(tokenizer, prompt=prompt, max_length=max_length, device=device)

    top_k_values = _normalize_top_k_values(top_k, int(final_model.config.num_experts_per_tok))

    checkpoint_router_logits = collect_router_logits(checkpoint_model, adapter, inputs)
    final_router_logits = collect_router_logits(final_model, adapter, inputs)

    if len(checkpoint_router_logits) != len(final_router_logits):
        raise ValueError(
            "Checkpoint model and final model produced a different number of routing layers: "
            f"{len(checkpoint_router_logits)} vs {len(final_router_logits)}."
        )

    layer_saturation = {k: [] for k in top_k_values}
    for layer_t, layer_T in zip(checkpoint_router_logits, final_router_logits):
        layer_result = compute_router_saturation_from_logits(layer_t, layer_T, top_k=top_k_values)
        for k in top_k_values:
            layer_saturation[k].append(layer_result[k])

    stacked_t = torch.cat([layer.reshape(-1, *layer.shape[-2:]) for layer in checkpoint_router_logits], dim=0)
    stacked_T = torch.cat([layer.reshape(-1, *layer.shape[-2:]) for layer in final_router_logits], dim=0)
    overall_saturation = compute_router_saturation_from_logits(stacked_t, stacked_T, top_k=top_k_values)

    return {
        "record_type": "router_saturation_example",
        "example_id": normalized["example_id"],
        "dataset": normalized["dataset"],
        "benchmark": normalized["benchmark"],
        "language": normalized["language"],
        "task_type": normalized["task_type"],
        "prompt": prompt,
        "top_k_values": top_k_values,
        "overall_router_saturation": overall_saturation,
        "layer_router_saturation": layer_saturation,
    }


def summarize_router_saturation(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No router saturation records were provided.")

    top_k_values = records[0]["top_k_values"]
    layer_means = {}
    for k in top_k_values:
        num_layers = len(records[0]["layer_router_saturation"][k])
        layer_means[k] = []
        for layer_idx in range(num_layers):
            layer_means[k].append(
                sum(record["layer_router_saturation"][k][layer_idx] for record in records) / len(records)
            )

    return {
        "record_type": "router_saturation_summary",
        "num_examples": len(records),
        "top_k_values": top_k_values,
        "overall_router_saturation": {
            k: sum(record["overall_router_saturation"][k] for record in records) / len(records)
            for k in top_k_values
        },
        "layer_router_saturation": layer_means,
    }
