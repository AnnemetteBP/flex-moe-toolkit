from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import json

import torch

from flex_moe_toolkit.core.routing_diagnostics import (
    compute_all_metrics,
    compute_offdiagonal_ratio,
    normalize_coactivation_counts,
)
from flex_moe_toolkit.adapters.flex_olmo import FlexOlmoAdapter
from flex_moe_toolkit.pipelines.flex_olmo import analyze_flex_olmo_routing, restricted_expert_mode
from flex_moe_toolkit.pipelines.flex_olmo_eval import FlexOlmoEvalRunSpec
from flex_moe_toolkit.utils.jsonl import write_jsonl
from flex_moe_toolkit.utils.router_activity import (
    activated_expert_combination,
    count_combinations,
    flatten_topk_experts,
    layer_iou_summary,
)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def ensure_router_logits_3d(router_logits: torch.Tensor) -> torch.Tensor:
    if router_logits.ndim == 2:
        return router_logits.unsqueeze(0)
    if router_logits.ndim == 4:
        return router_logits.flatten(0, 1)
    return router_logits


def tokenize_prompt(
    tokenizer,
    prompt: str,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return move_batch_to_device(encoded, device)


def build_inputs_from_token_ids(
    token_ids: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def select_hidden_state_layers(hidden_states, selected_layers: list[int] | None):
    if hidden_states is None:
        return {}

    num_layers = len(hidden_states)
    if selected_layers is None:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = []
        for layer_idx in selected_layers:
            normalized_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
            if normalized_idx < 0 or normalized_idx >= num_layers:
                raise ValueError(
                    f"Hidden-state layer index {layer_idx} is out of range for {num_layers} hidden-state tensors."
                )
            if normalized_idx not in layer_indices:
                layer_indices.append(normalized_idx)

    return {
        str(layer_idx): hidden_states[layer_idx].detach().cpu()
        for layer_idx in layer_indices
    }


def hidden_state_norms_by_layer(hidden_states_by_layer: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    norms = {}
    for layer_idx, tensor in hidden_states_by_layer.items():
        token_norms = torch.linalg.vector_norm(tensor[0], dim=-1)
        norms[layer_idx] = token_norms.detach().cpu().tolist()
    return norms


def slice_hidden_states_by_suffix(
    hidden_states_by_layer: dict[str, torch.Tensor],
    suffix_length: int,
) -> dict[str, torch.Tensor]:
    if suffix_length <= 0:
        return {}
    return {
        layer_idx: tensor[:, -suffix_length:, :].detach().cpu()
        for layer_idx, tensor in hidden_states_by_layer.items()
    }


def tokenize_reference_answer(tokenizer, answer_text: str | None) -> list[int]:
    if not answer_text:
        return []
    return tokenizer.encode(answer_text, add_special_tokens=False)


def generate_output_token_ids(
    model,
    tokenizer,
    inputs: dict[str, torch.Tensor],
    reference_answer: str | None,
    default_max_new_tokens: int,
) -> tuple[list[int], list[int], str]:
    ground_truth_output_token_ids = tokenize_reference_answer(tokenizer, reference_answer)
    max_new_tokens = len(ground_truth_output_token_ids) if ground_truth_output_token_ids else default_max_new_tokens
    max_new_tokens = max(1, max_new_tokens)

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        generated = model.generate(**generation_kwargs)

    prompt_length = int(inputs["input_ids"].shape[-1])
    predicted_output_token_ids = generated[0, prompt_length:].detach().cpu().tolist()
    predicted_output_text = tokenizer.decode(predicted_output_token_ids, skip_special_tokens=True)
    return ground_truth_output_token_ids, predicted_output_token_ids, predicted_output_text


def slice_continuation_topk_experts(topk_experts, continuation_length: int):
    if continuation_length <= 0:
        return []
    sliced = []
    for layer in topk_experts:
        if layer.ndim == 2:
            continuation = layer[-continuation_length:, :].unsqueeze(0)
        elif layer.ndim == 3:
            continuation = layer[:, -continuation_length:, :]
        else:
            raise ValueError(
                f"Unexpected top-k expert tensor shape {tuple(layer.shape)} while slicing continuation tokens."
            )
        sliced.append(continuation.detach().cpu())
    return sliced


def summarize_router_token_distributions(router_logits, topk_experts) -> list[dict[str, torch.Tensor]]:
    summaries = []

    for layer_idx, (layer_logits, layer_topk_experts) in enumerate(zip(router_logits, topk_experts)):
        normalized_logits = ensure_router_logits_3d(layer_logits)
        normalized_topk = ensure_router_logits_3d(layer_topk_experts)
        probabilities = torch.softmax(normalized_logits, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-9))).sum(dim=-1)

        top_summary_k = min(2, probabilities.shape[-1])
        top_probs, top_indices = torch.topk(probabilities, k=top_summary_k, dim=-1)

        top1_probs = top_probs[..., 0]
        top1_indices = top_indices[..., 0]
        if top_summary_k > 1:
            top2_probs = top_probs[..., 1]
            top2_indices = top_indices[..., 1]
        else:
            top2_probs = torch.zeros_like(top1_probs)
            top2_indices = torch.full_like(top1_indices, -1)

        selected_prob_mass = probabilities.gather(dim=-1, index=normalized_topk).sum(dim=-1)

        summaries.append(
            {
                "layer_idx": layer_idx,
                "top1_expert_ids": top1_indices.detach().cpu(),
                "top1_probs": top1_probs.detach().cpu(),
                "top2_expert_ids": top2_indices.detach().cpu(),
                "top2_probs": top2_probs.detach().cpu(),
                "top1_top2_margin": (top1_probs - top2_probs).detach().cpu(),
                "token_entropy": entropy.detach().cpu(),
                "selected_expert_prob_mass": selected_prob_mass.detach().cpu(),
            }
        )

    return summaries


def slice_router_token_summaries(
    router_token_summaries: list[dict[str, torch.Tensor]],
    suffix_length: int,
) -> list[dict[str, torch.Tensor]]:
    if suffix_length <= 0:
        return []

    sliced = []
    for layer_summary in router_token_summaries:
        sliced.append(
            {
                key: (
                    value[:, -suffix_length:].detach().cpu()
                    if isinstance(value, torch.Tensor) and value.ndim >= 2
                    else value
                )
                for key, value in layer_summary.items()
            }
        )
    return sliced


def aggregate_router_token_summaries(router_token_summaries: list[dict[str, torch.Tensor]]) -> list[dict[str, float | int]]:
    aggregate = []
    for layer_summary in router_token_summaries:
        aggregate.append(
            {
                "layer_idx": int(layer_summary["layer_idx"]),
                "mean_top1_prob": float(layer_summary["top1_probs"].float().mean().item()),
                "mean_top2_prob": float(layer_summary["top2_probs"].float().mean().item()),
                "mean_top1_top2_margin": float(layer_summary["top1_top2_margin"].float().mean().item()),
                "mean_token_entropy": float(layer_summary["token_entropy"].float().mean().item()),
                "mean_selected_expert_prob_mass": float(
                    layer_summary["selected_expert_prob_mass"].float().mean().item()
                ),
            }
        )
    return aggregate


def effective_run_top_k(model, run_spec: FlexOlmoEvalRunSpec) -> int:
    native_top_k = int(model.config.num_experts_per_tok)
    if not run_spec.apply_restricted_routing:
        return native_top_k
    return min(native_top_k, len(run_spec.allowed_experts))


def capture_hidden_state_artifacts(
    model,
    inputs: dict[str, torch.Tensor],
    selected_layers: list[int] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
        )
    selected = select_hidden_state_layers(outputs.hidden_states, selected_layers)
    return selected, hidden_state_norms_by_layer(selected)


def count_token_level_combinations(topk_experts) -> Counter:
    token_counts = Counter()

    for layer in topk_experts:
        for batch in layer:
            batch_list = batch.tolist() if hasattr(batch, "tolist") else batch
            if not isinstance(batch_list, list):
                batch_list = [batch_list]
            for token in batch_list:
                experts = token if isinstance(token, list) else [token]
                combo = tuple(sorted(int(expert_idx) for expert_idx in experts))
                token_counts[combo] += 1

    return token_counts


def analyze_prompt_example(
    model,
    tokenizer,
    example: dict[str, Any],
    run_spec: FlexOlmoEvalRunSpec,
    max_length: int,
    device: torch.device,
    capture_output_token_ids: bool = True,
    default_max_new_tokens: int = 16,
    capture_router_tensors: bool = False,
    capture_hidden_states: bool = False,
    hidden_state_layers: list[int] | None = None,
) -> dict[str, Any]:
    inputs = tokenize_prompt(
        tokenizer=tokenizer,
        prompt=example["prompt"],
        max_length=max_length,
        device=device,
    )

    run_top_k = effective_run_top_k(model, run_spec)

    routing_context = (
        restricted_expert_mode(model, allowed_experts=run_spec.allowed_experts)
        if run_spec.apply_restricted_routing
        else nullcontext(model)
    )

    with routing_context:
        routing = analyze_flex_olmo_routing(
            model,
            inputs,
            top_k=run_top_k,
        )
        prompt_topk_experts = [layer.detach().cpu() for layer in routing["topk_experts"]]
        prompt_router_logits_by_layer = []
        prompt_router_probs_by_layer = []
        prompt_router_token_summaries_by_layer = []
        prompt_router_summary_by_layer = []
        prompt_hidden_states_by_layer = {}
        prompt_hidden_state_norms = {}
        if capture_router_tensors:
            adapter = FlexOlmoAdapter()
            prompt_router_logits_by_layer = [
                layer.detach().cpu() for layer in routing["router_logits"]
            ]
            prompt_router_probs_by_layer = [
                layer.detach().cpu() for layer in adapter.router_logits_to_probs(routing["router_logits"])
            ]
        prompt_router_token_summaries_by_layer = summarize_router_token_distributions(
            routing["router_logits"],
            routing["topk_experts"],
        )
        prompt_router_summary_by_layer = aggregate_router_token_summaries(prompt_router_token_summaries_by_layer)
        if capture_hidden_states:
            (
                prompt_hidden_states_by_layer,
                prompt_hidden_state_norms,
            ) = capture_hidden_state_artifacts(
                model=model,
                inputs=inputs,
                selected_layers=hidden_state_layers,
            )
        ground_truth_output_token_ids = []
        predicted_output_token_ids = []
        predicted_output_text = None
        ground_truth_output_topk_experts = []
        predicted_output_topk_experts = []
        ground_truth_output_hidden_states_by_layer = {}
        ground_truth_output_hidden_state_norms_by_layer = {}
        predicted_output_hidden_states_by_layer = {}
        predicted_output_hidden_state_norms_by_layer = {}
        ground_truth_router_token_summaries_by_layer = []
        ground_truth_router_summary_by_layer = []
        predicted_router_token_summaries_by_layer = []
        predicted_router_summary_by_layer = []
        if capture_output_token_ids:
            (
                ground_truth_output_token_ids,
                predicted_output_token_ids,
                predicted_output_text,
            ) = generate_output_token_ids(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                reference_answer=example.get("reference_answer"),
                default_max_new_tokens=default_max_new_tokens,
            )
            prompt_token_ids = inputs["input_ids"][0].detach().cpu().tolist()
            if ground_truth_output_token_ids:
                ground_truth_inputs = build_inputs_from_token_ids(
                    token_ids=prompt_token_ids + ground_truth_output_token_ids,
                    device=device,
                )
                ground_truth_routing = analyze_flex_olmo_routing(
                    model,
                    ground_truth_inputs,
                    top_k=run_top_k,
                )
                ground_truth_output_topk_experts = slice_continuation_topk_experts(
                    ground_truth_routing["topk_experts"],
                    continuation_length=len(ground_truth_output_token_ids),
                )
                ground_truth_router_token_summaries_by_layer = slice_router_token_summaries(
                    summarize_router_token_distributions(
                        ground_truth_routing["router_logits"],
                        ground_truth_routing["topk_experts"],
                    ),
                    suffix_length=len(ground_truth_output_token_ids),
                )
                ground_truth_router_summary_by_layer = aggregate_router_token_summaries(
                    ground_truth_router_token_summaries_by_layer
                )
                if capture_hidden_states:
                    hidden_states, hidden_state_norms = capture_hidden_state_artifacts(
                        model=model,
                        inputs=ground_truth_inputs,
                        selected_layers=hidden_state_layers,
                    )
                    ground_truth_output_hidden_states_by_layer = slice_hidden_states_by_suffix(
                        hidden_states,
                        suffix_length=len(ground_truth_output_token_ids),
                    )
                    ground_truth_output_hidden_state_norms_by_layer = {
                        layer_idx: norms[-len(ground_truth_output_token_ids):]
                        for layer_idx, norms in hidden_state_norms.items()
                    }

            if predicted_output_token_ids:
                predicted_inputs = build_inputs_from_token_ids(
                    token_ids=prompt_token_ids + predicted_output_token_ids,
                    device=device,
                )
                predicted_routing = analyze_flex_olmo_routing(
                    model,
                    predicted_inputs,
                    top_k=run_top_k,
                )
                predicted_output_topk_experts = slice_continuation_topk_experts(
                    predicted_routing["topk_experts"],
                    continuation_length=len(predicted_output_token_ids),
                )
                predicted_router_token_summaries_by_layer = slice_router_token_summaries(
                    summarize_router_token_distributions(
                        predicted_routing["router_logits"],
                        predicted_routing["topk_experts"],
                    ),
                    suffix_length=len(predicted_output_token_ids),
                )
                predicted_router_summary_by_layer = aggregate_router_token_summaries(
                    predicted_router_token_summaries_by_layer
                )
                if capture_hidden_states:
                    hidden_states, hidden_state_norms = capture_hidden_state_artifacts(
                        model=model,
                        inputs=predicted_inputs,
                        selected_layers=hidden_state_layers,
                    )
                    predicted_output_hidden_states_by_layer = slice_hidden_states_by_suffix(
                        hidden_states,
                        suffix_length=len(predicted_output_token_ids),
                    )
                    predicted_output_hidden_state_norms_by_layer = {
                        layer_idx: norms[-len(predicted_output_token_ids):]
                        for layer_idx, norms in hidden_state_norms.items()
                    }

    first_layer_router_logits = ensure_router_logits_3d(routing["router_logits"][0])
    metrics_top_k = min(run_top_k, first_layer_router_logits.shape[-1])
    batch_metrics = compute_all_metrics(first_layer_router_logits, top_k=metrics_top_k)

    layer_batch_routing_metrics = []
    for layer_idx, layer_router_logits in enumerate(routing["router_logits"]):
        normalized_logits = ensure_router_logits_3d(layer_router_logits)
        layer_metrics = compute_all_metrics(
            normalized_logits,
            top_k=min(run_top_k, normalized_logits.shape[-1]),
        )
        layer_batch_routing_metrics.append(
            {
                "layer_idx": layer_idx,
                "usage": layer_metrics["usage"],
                "entropy_mean": layer_metrics["entropy_mean"],
                "coactivation_counts": layer_metrics["coactivation_counts"],
                "activation_counts": layer_metrics["activation_counts"],
                "coactivation_matrix": layer_metrics["coactivation_matrix"],
                "offdiag_ratio": layer_metrics["offdiag_ratio"],
                "mean_top1_prob": prompt_router_summary_by_layer[layer_idx]["mean_top1_prob"],
                "mean_top2_prob": prompt_router_summary_by_layer[layer_idx]["mean_top2_prob"],
                "mean_top1_top2_margin": prompt_router_summary_by_layer[layer_idx]["mean_top1_top2_margin"],
                "mean_token_entropy": prompt_router_summary_by_layer[layer_idx]["mean_token_entropy"],
                "mean_selected_expert_prob_mass": prompt_router_summary_by_layer[layer_idx][
                    "mean_selected_expert_prob_mass"
                ],
            }
        )

    layer_combos = flatten_topk_experts(routing["topk_experts"])
    global_combo = activated_expert_combination(routing["topk_experts"])
    layer_overlap = layer_iou_summary(layer_combos)
    token_combination_counts = count_token_level_combinations(routing["topk_experts"])

    return {
        "record_type": "routing_example",
        "run_label": run_spec.label,
        "run_kind": run_spec.run_kind,
        "routing_restricted": run_spec.apply_restricted_routing,
        "available_experts": run_spec.allowed_experts,
        "num_available_experts": len(run_spec.allowed_experts),
        "model_native_top_k": int(model.config.num_experts_per_tok),
        "effective_top_k": int(run_top_k),
        "example_id": example["example_id"],
        "language": example["language"],
        "question": example["question"],
        "reference_answer": example.get("reference_answer"),
        "prompt": example["prompt"],
        "num_input_tokens": int(inputs["input_ids"].shape[-1]),
        "input_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
        "prompt_topk_experts_by_layer": prompt_topk_experts,
        "prompt_router_logits_by_layer": prompt_router_logits_by_layer if capture_router_tensors else None,
        "prompt_router_probs_by_layer": prompt_router_probs_by_layer if capture_router_tensors else None,
        "prompt_router_token_summaries_by_layer": prompt_router_token_summaries_by_layer,
        "prompt_router_summary_by_layer": prompt_router_summary_by_layer,
        "prompt_hidden_states_by_layer": prompt_hidden_states_by_layer if capture_hidden_states else None,
        "prompt_hidden_state_norms_by_layer": prompt_hidden_state_norms if capture_hidden_states else None,
        "ground_truth_output_token_ids": ground_truth_output_token_ids,
        "ground_truth_output_topk_experts_by_layer": ground_truth_output_topk_experts,
        "ground_truth_router_token_summaries_by_layer": ground_truth_router_token_summaries_by_layer,
        "ground_truth_router_summary_by_layer": ground_truth_router_summary_by_layer,
        "ground_truth_output_hidden_states_by_layer": (
            ground_truth_output_hidden_states_by_layer if capture_hidden_states else None
        ),
        "ground_truth_output_hidden_state_norms_by_layer": (
            ground_truth_output_hidden_state_norms_by_layer if capture_hidden_states else None
        ),
        "predicted_output_token_ids": predicted_output_token_ids,
        "predicted_output_topk_experts_by_layer": predicted_output_topk_experts,
        "predicted_router_token_summaries_by_layer": predicted_router_token_summaries_by_layer,
        "predicted_router_summary_by_layer": predicted_router_summary_by_layer,
        "predicted_output_hidden_states_by_layer": (
            predicted_output_hidden_states_by_layer if capture_hidden_states else None
        ),
        "predicted_output_hidden_state_norms_by_layer": (
            predicted_output_hidden_state_norms_by_layer if capture_hidden_states else None
        ),
        "predicted_output_text": predicted_output_text,
        "entropy": routing["entropy"],
        "load_balance": routing["load_balance"],
        "expert_usage": routing["expert_usage"],
        "layer_expert_matrix": routing["layer_expert_matrix"],
        "batch_routing_metrics": {
            "usage": batch_metrics["usage"],
            "entropy_mean": batch_metrics["entropy_mean"],
            "coactivation_counts": batch_metrics["coactivation_counts"],
            "activation_counts": batch_metrics["activation_counts"],
            "coactivation_matrix": batch_metrics["coactivation_matrix"],
            "offdiag_ratio": batch_metrics["offdiag_ratio"],
            "mean_top1_prob": prompt_router_summary_by_layer[0]["mean_top1_prob"],
            "mean_top2_prob": prompt_router_summary_by_layer[0]["mean_top2_prob"],
            "mean_top1_top2_margin": prompt_router_summary_by_layer[0]["mean_top1_top2_margin"],
            "mean_token_entropy": prompt_router_summary_by_layer[0]["mean_token_entropy"],
            "mean_selected_expert_prob_mass": prompt_router_summary_by_layer[0]["mean_selected_expert_prob_mass"],
        },
        "layer_batch_routing_metrics": layer_batch_routing_metrics,
        "token_topk_combination_counts": {
            ",".join(str(expert) for expert in combo): count
            for combo, count in sorted(token_combination_counts.items())
        },
        "layer_activated_experts": layer_combos,
        "activated_experts": global_combo,
        "layer_intersection_experts": layer_overlap["layer_intersection_experts"],
        "layer_union_experts": layer_overlap["layer_union_experts"],
        "pairwise_layer_iou": layer_overlap["pairwise_layer_iou"],
        "mean_layer_iou": layer_overlap["mean_layer_iou"],
    }


def summarize_routing_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    by_run = defaultdict(list)
    for record in records:
        by_run[record["run_label"]].append(record)

    for run_label, run_records in by_run.items():
        total = len(run_records)
        by_language = defaultdict(int)
        combination_counts = count_combinations(record["activated_experts"] for record in run_records)
        mean_entropy = sum(float(record["entropy"]) for record in run_records) / total if total else 0.0
        mean_load_balance = (
            sum(float(record["load_balance"]) for record in run_records) / total if total else 0.0
        )

        for record in run_records:
            by_language[record["language"]] += 1

        summaries.append(
            {
                "record_type": "routing_summary",
                "run_label": run_label,
                "run_kind": run_records[0]["run_kind"],
                "routing_restricted": run_records[0]["routing_restricted"],
                "num_examples": total,
                "available_experts": run_records[0]["available_experts"],
                "model_native_top_k": run_records[0]["model_native_top_k"],
                "effective_top_k": run_records[0]["effective_top_k"],
                "mean_entropy": mean_entropy,
                "mean_load_balance": mean_load_balance,
                "mean_layer_iou": (
                    sum(record["mean_layer_iou"] for record in run_records) / total if total else 0.0
                ),
                "by_language": dict(sorted(by_language.items())),
                "activated_expert_combinations": {
                    ",".join(str(expert) for expert in combo): count
                    for combo, count in sorted(combination_counts.items())
                },
            }
        )

    return summaries


def aggregate_routing_analysis(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        raise ValueError("No records were provided for routing analysis.")

    usage_sum = None
    entropy_total = 0.0
    top1_prob_total = 0.0
    top2_prob_total = 0.0
    top1_top2_margin_total = 0.0
    token_entropy_total = 0.0
    selected_expert_prob_mass_total = 0.0
    batch_coactivation_count_sum = None
    batch_activation_count_sum = None
    layer_coactivation_count_sums = []
    layer_activation_count_sums = []
    batch_records = []

    for record in records:
        metrics = record["batch_routing_metrics"]
        usage = metrics["usage"].detach().cpu()
        entropy = float(metrics["entropy_mean"])
        batch_coactivation_counts = metrics["coactivation_counts"].detach().cpu()
        batch_activation_counts = metrics["activation_counts"].detach().cpu()

        usage_sum = usage.clone() if usage_sum is None else usage_sum + usage
        entropy_total += entropy
        top1_prob_total += float(metrics["mean_top1_prob"])
        top2_prob_total += float(metrics["mean_top2_prob"])
        top1_top2_margin_total += float(metrics["mean_top1_top2_margin"])
        token_entropy_total += float(metrics["mean_token_entropy"])
        selected_expert_prob_mass_total += float(metrics["mean_selected_expert_prob_mass"])
        batch_coactivation_count_sum = (
            batch_coactivation_counts.clone()
            if batch_coactivation_count_sum is None
            else batch_coactivation_count_sum + batch_coactivation_counts
        )
        batch_activation_count_sum = (
            batch_activation_counts.clone()
            if batch_activation_count_sum is None
            else batch_activation_count_sum + batch_activation_counts
        )

        for layer_metrics in record["layer_batch_routing_metrics"]:
            layer_idx = layer_metrics["layer_idx"]
            layer_coactivation_counts = layer_metrics["coactivation_counts"].detach().cpu()
            layer_activation_counts = layer_metrics["activation_counts"].detach().cpu()
            while len(layer_coactivation_count_sums) <= layer_idx:
                layer_coactivation_count_sums.append(None)
                layer_activation_count_sums.append(None)
            layer_coactivation_count_sums[layer_idx] = (
                layer_coactivation_counts.clone()
                if layer_coactivation_count_sums[layer_idx] is None
                else layer_coactivation_count_sums[layer_idx] + layer_coactivation_counts
            )
            layer_activation_count_sums[layer_idx] = (
                layer_activation_counts.clone()
                if layer_activation_count_sums[layer_idx] is None
                else layer_activation_count_sums[layer_idx] + layer_activation_counts
            )

        batch_records.append(
            {
                "record_type": "routing_batch",
                "run_label": record["run_label"],
                "example_id": record["example_id"],
                "language": record["language"],
                "usage": usage,
                "entropy_mean": entropy,
                "coactivation_counts": batch_coactivation_counts,
                "activation_counts": batch_activation_counts,
                "coactivation_matrix": metrics["coactivation_matrix"].detach().cpu(),
                "offdiag_ratio": float(metrics["offdiag_ratio"]),
                "mean_top1_prob": float(metrics["mean_top1_prob"]),
                "mean_top2_prob": float(metrics["mean_top2_prob"]),
                "mean_top1_top2_margin": float(metrics["mean_top1_top2_margin"]),
                "mean_token_entropy": float(metrics["mean_token_entropy"]),
                "mean_selected_expert_prob_mass": float(metrics["mean_selected_expert_prob_mass"]),
            }
        )

    aggregate_usage = usage_sum / usage_sum.sum()
    aggregate_entropy = entropy_total / len(records)
    non_null_layer_counts = [matrix for matrix in layer_coactivation_count_sums if matrix is not None]
    if not non_null_layer_counts:
        raise ValueError("No layer-wise coactivation matrices were collected.")

    aggregate_coactivation_matrix = normalize_coactivation_counts(
        batch_coactivation_count_sum,
        batch_activation_count_sum,
    )
    layer_coactivation_matrices = [
        (
            normalize_coactivation_counts(layer_counts, layer_activations)
            if layer_counts is not None and layer_activations is not None
            else None
        )
        for layer_counts, layer_activations in zip(layer_coactivation_count_sums, layer_activation_count_sums)
    ]

    aggregate_record = {
        "record_type": "routing_aggregate",
        "num_batches": len(records),
        "usage": aggregate_usage,
        "entropy_mean": aggregate_entropy,
        "mean_top1_prob": top1_prob_total / len(records),
        "mean_top2_prob": top2_prob_total / len(records),
        "mean_top1_top2_margin": top1_top2_margin_total / len(records),
        "mean_token_entropy": token_entropy_total / len(records),
        "mean_selected_expert_prob_mass": selected_expert_prob_mass_total / len(records),
        "coactivation_counts": batch_coactivation_count_sum,
        "activation_counts": batch_activation_count_sum,
        "coactivation_matrix": aggregate_coactivation_matrix,
        "layer_coactivation_matrices": layer_coactivation_matrices,
        "offdiag_ratio": compute_offdiagonal_ratio(aggregate_coactivation_matrix),
    }

    return batch_records + [aggregate_record], aggregate_record


def analyze_prompt_dataset_across_runs(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    run_specs: list[FlexOlmoEvalRunSpec],
    max_length: int,
    device: torch.device,
    capture_output_token_ids: bool = True,
    default_max_new_tokens: int = 16,
    capture_router_tensors: bool = False,
    capture_hidden_states: bool = False,
    hidden_state_layers: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    results = {}
    for run_spec in run_specs:
        records = [
            analyze_prompt_example(
                model=model,
                tokenizer=tokenizer,
                example=example,
                run_spec=run_spec,
                max_length=max_length,
                device=device,
                capture_output_token_ids=capture_output_token_ids,
                default_max_new_tokens=default_max_new_tokens,
                capture_router_tensors=capture_router_tensors,
                capture_hidden_states=capture_hidden_states,
                hidden_state_layers=hidden_state_layers,
            )
            for example in examples
        ]
        summaries = summarize_routing_records(records)
        routing_analysis_records, routing_aggregate = aggregate_routing_analysis(records)
        results[run_spec.label] = {
            "records": records,
            "summaries": summaries,
            "routing_analysis_records": routing_analysis_records,
            "routing_aggregate": routing_aggregate,
            "run_spec": run_spec,
        }
    return results


def sanitize_name(name: str) -> str:
    allowed = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "dataset"


def save_dataset_run_outputs(
    output_root: str | Path,
    dataset_name: str,
    run_results: dict[str, dict[str, Any]],
    sort_keys: bool = True,
) -> dict[str, dict[str, str]]:
    output_root = Path(output_root)
    dataset_dir = output_root / sanitize_name(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for run_label, payload in run_results.items():
        run_dir = dataset_dir / sanitize_name(run_label)
        run_dir.mkdir(parents=True, exist_ok=True)

        records_path = run_dir / "routing_records.jsonl"
        summary_path = run_dir / "routing_summary.jsonl"
        routing_path = run_dir / "routing_analysis.jsonl"

        write_jsonl(payload["records"], records_path, sort_keys=sort_keys)
        write_jsonl(payload["summaries"], summary_path, sort_keys=sort_keys)
        write_jsonl(payload["routing_analysis_records"], routing_path, sort_keys=sort_keys)

        manifest[run_label] = {
            "records_path": str(records_path),
            "summary_path": str(summary_path),
            "routing_analysis_path": str(routing_path),
        }

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
