from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import inspect
import json
from collections import defaultdict
from pathlib import Path
import re
import string
import sys
from typing import Any
import unicodedata

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
    parse_hidden_state_layers,
    resolve_device,
)
from flex_moe_toolkit.pipelines.flex_olmo import backbone_only_mode, restricted_expert_mode


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "focused" / "55b_pair" / "causal_intervention"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"
DEFAULT_STOP_PATTERNS = (
    "\nQuestion",
    "\nSpørgsmål",
    "\nSvar:",
    "\nAnswer:",
    "\n```",
    "```",
    "\n###",
    "###",
)
ARTICLE_PATTERN = re.compile(r"\b(a|an|the|en|et|den|det|de)\b", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
TRAILING_ZERO_NUMBER_PATTERN = re.compile(r"\b(-?\d+)\.0+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact causal intervention experiments for a target/comparison FlexOlmo model pair."
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include from the manifest.")
    parser.add_argument("--max-examples-per-dataset", type=int, default=75)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--selected-layers",
        default="8,16,24,-1",
        help="Comma-separated decoder layer indices or presets like `early_mid_late_last`.",
    )
    parser.add_argument(
        "--position-policy",
        default="last_prompt_token",
        choices=("last_prompt_token", "mean_prompt"),
    )
    parser.add_argument(
        "--intervention-kind",
        default="remove_delta",
        choices=("remove_delta", "replace_delta_component"),
    )
    parser.add_argument(
        "--source-mode",
        default="comparison_model",
        choices=("comparison_model", "public_only", "backbone_only"),
    )
    parser.add_argument(
        "--delta-aggregation",
        default="per_example",
        choices=("per_example", "dataset_mean"),
    )
    parser.add_argument(
        "--delta-basis",
        default="source_mode_diff",
        choices=("source_mode_diff", "expert_minus_backbone"),
    )
    parser.add_argument("--public-expert-idx", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--tokenizer-path", help="Optional tokenizer path. Defaults to the target model path.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))

    parser.add_argument("--target-model-path")
    parser.add_argument("--target-model-name")
    parser.add_argument("--target-model-root")

    parser.add_argument("--comparison-model-path")
    parser.add_argument("--comparison-model-name")
    parser.add_argument("--comparison-model-root")

    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def resolve_named_model_path(model_name: str | None, model_root: str | None, model_registry: str) -> str:
    if not model_name or not model_root:
        raise ValueError("Provide either an explicit model path or both model name and model root.")
    allowed_names = load_allowed_model_names(model_registry)
    if model_name not in allowed_names:
        raise ValueError(f"Model name `{model_name}` was not found in {model_registry}.")
    return str(Path(model_root) / model_name)


def resolve_model_path(explicit_path: str | None, model_name: str | None, model_root: str | None, model_registry: str) -> str:
    if explicit_path:
        return explicit_path
    return resolve_named_model_path(model_name=model_name, model_root=model_root, model_registry=model_registry)


def resolved_model_name(explicit_name: str | None, model_path: str) -> str:
    return explicit_name or Path(model_path).name


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
    reference_answer = record.get("reference_answer")
    if not reference_answer:
        raise ValueError(
            f"Dataset `{dataset_name}` contains a record without `reference_answer`; "
            "teacher-forced intervention runs require reference answers."
        )
    prompting_config = dict(dataset_entry.get("prompting", {}))
    normalized_prompt = apply_chat_template_if_requested(tokenizer, prompt, prompting_config)
    return {
        "example_id": record["example_id"],
        "dataset_name": dataset_name,
        "language": record.get("language", "unknown"),
        "domain": record.get("domain", dataset_entry.get("domain")),
        "scoring_mode": record.get("scoring_mode", dataset_entry.get("scoring_mode")),
        "generation_config": dict(dataset_entry.get("generation", {})),
        "prompt": normalized_prompt,
        "reference_answer": reference_answer,
        "question": record.get("question"),
        "metadata": dict(record.get("metadata", {})),
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
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_length": len(prompt_ids),
        "answer_length": len(answer_ids),
        "sequence_length": len(input_ids_list),
        "prompt_input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device),
        "prompt_attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device),
    }


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = TRAILING_ZERO_NUMBER_PATTERN.sub(r"\1", text)
    text = "".join(char for char in text if char not in string.punctuation)
    text = ARTICLE_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def strip_generation_artifacts(text: str | None) -> str:
    if text is None:
        return ""
    cleaned = text.strip()
    for pattern in DEFAULT_STOP_PATTERNS:
        if pattern in cleaned:
            cleaned = cleaned.split(pattern, 1)[0]
    cleaned = cleaned.strip()
    cleaned = cleaned.lstrip(":,- ")
    return cleaned.strip()


def relaxed_match(prediction: str, reference: str) -> bool:
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)
    if not normalized_prediction or not normalized_reference:
        return normalized_prediction == normalized_reference
    if normalized_prediction == normalized_reference:
        return True
    return (
        normalized_prediction in normalized_reference
        or normalized_reference in normalized_prediction
    )


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = defaultdict(int)
    ref_counts = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    for token in ref_tokens:
        ref_counts[token] += 1
    common = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def classify_pubmedqa(prediction: str) -> str:
    normalized = normalize_text(strip_generation_artifacts(prediction))
    for label in ("yes", "no", "maybe"):
        if normalized.startswith(label):
            return label
    return normalized.split(" ", 1)[0] if normalized else ""


def score_prediction(example: dict, prediction_text: str) -> dict[str, Any]:
    scoring_mode = str(example.get("scoring_mode", "qa"))
    cleaned_prediction = strip_generation_artifacts(prediction_text)
    reference = str(example["reference_answer"])

    if scoring_mode == "classification":
        predicted_label = classify_pubmedqa(cleaned_prediction)
        reference_label = normalize_text(reference)
        is_correct = predicted_label == reference_label
        return {
            "prediction_text_clean": cleaned_prediction,
            "score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "exact_match": is_correct,
            "relaxed_match": is_correct,
            "token_f1": 1.0 if is_correct else 0.0,
            "normalized_prediction": predicted_label,
            "normalized_reference": reference_label,
        }

    em = normalize_text(cleaned_prediction) == normalize_text(reference)
    rm = relaxed_match(cleaned_prediction, reference)
    f1 = token_f1(cleaned_prediction, reference)
    return {
        "prediction_text_clean": cleaned_prediction,
        "score": f1,
        "is_correct": rm,
        "exact_match": em,
        "relaxed_match": rm,
        "token_f1": f1,
        "normalized_prediction": normalize_text(cleaned_prediction),
        "normalized_reference": normalize_text(reference),
    }


def reduce_positions(tensor: torch.Tensor, prompt_length: int, position_policy: str) -> torch.Tensor:
    prompt_tensor = tensor[0, :prompt_length, :]
    if position_policy == "last_prompt_token":
        return prompt_tensor[-1].detach().cpu().float()
    if position_policy == "mean_prompt":
        return prompt_tensor.mean(dim=0).detach().cpu().float()
    raise ValueError(f"Unsupported position policy `{position_policy}`.")


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
    return parse_hidden_state_layers(raw_value)


def summarize_router_metrics(router_probs: torch.Tensor, prompt_length: int, position_policy: str) -> dict[str, float]:
    prompt_probs = router_probs[0, :prompt_length, :].detach().cpu().float()
    if position_policy == "last_prompt_token":
        selected = prompt_probs[-1:]
    elif position_policy == "mean_prompt":
        selected = prompt_probs
    else:
        raise ValueError(f"Unsupported position policy `{position_policy}`.")

    top_probs, _top_indices = torch.topk(selected, k=min(2, selected.shape[-1]), dim=-1)
    top1 = top_probs[..., 0]
    top2 = top_probs[..., 1] if top_probs.shape[-1] > 1 else torch.zeros_like(top1)
    entropy = -(selected * torch.log(selected.clamp_min(1e-9))).sum(dim=-1)
    return {
        "baseline_top1_prob": float(top1.mean().item()),
        "baseline_top2_prob": float(top2.mean().item()),
        "baseline_margin": float((top1 - top2).mean().item()),
        "baseline_entropy": float(entropy.mean().item()),
    }


class LayerOutputCapture:
    def __init__(self, selected_layers: list[int], prompt_length: int, position_policy: str):
        self.selected_layers = selected_layers
        self.prompt_length = prompt_length
        self.position_policy = position_policy
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            self.outputs[layer_idx] = reduce_positions(output, self.prompt_length, self.position_policy)
        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            self.handles.append(layers[layer_idx].mlp.register_forward_hook(self._make_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class LayerComponentCapture:
    def __init__(self, selected_layers: list[int], prompt_length: int, position_policy: str):
        self.selected_layers = selected_layers
        self.prompt_length = prompt_length
        self.position_policy = position_policy
        self.layer_shapes: dict[int, tuple[int, ...]] = {}
        self.outputs: dict[int, dict[str, torch.Tensor]] = {}
        self.handles = []

    def _make_pre_hook(self, layer_idx: int):
        def hook(_module, args):
            hidden_states = args[0]
            self.layer_shapes[layer_idx] = tuple(hidden_states.shape)
            self.outputs.setdefault(layer_idx, {})["backbone_input"] = reduce_positions(
                hidden_states, self.prompt_length, self.position_policy
            )
        return hook

    def _make_experts_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            layer_shape = self.layer_shapes.get(layer_idx)
            expert_output = output
            if layer_shape is not None and output.ndim == 2 and len(layer_shape) == 3:
                expert_output = output.reshape(layer_shape)
            self.outputs.setdefault(layer_idx, {})["expert_output"] = reduce_positions(
                expert_output, self.prompt_length, self.position_policy
            )
        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            self.handles.append(layers[layer_idx].mlp.register_forward_pre_hook(self._make_pre_hook(layer_idx)))
            self.handles.append(layers[layer_idx].mlp.experts.register_forward_hook(self._make_experts_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class RouterCapture:
    def __init__(self):
        self.outputs: dict[int, dict[str, torch.Tensor]] = {}
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            if isinstance(output, tuple):
                probs = output[0]
                top_k_weights = output[1] if len(output) > 1 else None
                top_k_index = output[2] if len(output) > 2 else None
            else:
                probs = output
                top_k_weights = None
                top_k_index = None
            self.outputs[layer_idx] = {
                "router_probs": probs.detach().cpu().float(),
                "top_k_weights": top_k_weights.detach().cpu().float() if top_k_weights is not None else None,
                "top_k_index": top_k_index.detach().cpu() if top_k_index is not None else None,
            }
        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx, layer in enumerate(layers):
            self.handles.append(layer.mlp.gate.register_forward_hook(self._make_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class LayerOutputIntervention:
    def __init__(
        self,
        layer_idx: int,
        prompt_length: int,
        position_policy: str,
        target_vector: torch.Tensor,
        comparison_vector: torch.Tensor,
        intervention_kind: str,
    ):
        self.layer_idx = layer_idx
        self.prompt_length = prompt_length
        self.position_policy = position_policy
        self.target_vector = target_vector.float()
        self.comparison_vector = comparison_vector.float()
        self.intervention_kind = intervention_kind
        self.handle = None

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        self.handle = layers[self.layer_idx].mlp.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module, _args, output):
        delta = (self.target_vector - self.comparison_vector).to(device=output.device, dtype=output.dtype)
        delta_norm = torch.linalg.vector_norm(delta)
        if torch.isclose(delta_norm, torch.tensor(0.0, device=output.device, dtype=output.dtype)):
            return output
        direction = delta / delta_norm.clamp_min(torch.finfo(output.dtype).eps)

        modified = output.clone()
        if self.position_policy == "last_prompt_token":
            position = self.prompt_length - 1
            token_vector = modified[:, position, :]
            coeff = torch.sum(token_vector * direction, dim=-1, keepdim=True)
            if self.intervention_kind == "remove_delta":
                modified[:, position, :] = token_vector - coeff * direction
            elif self.intervention_kind == "replace_delta_component":
                source_vector = self.comparison_vector.to(device=output.device, dtype=output.dtype)
                source_coeff = torch.sum(source_vector * direction, dim=-1, keepdim=True)
                modified[:, position, :] = token_vector - coeff * direction + source_coeff * direction
            else:
                raise ValueError(f"Unsupported intervention kind `{self.intervention_kind}`.")
            return modified

        if self.position_policy == "mean_prompt":
            token_vectors = modified[:, : self.prompt_length, :]
            coeff = torch.sum(token_vectors * direction, dim=-1, keepdim=True)
            if self.intervention_kind == "remove_delta":
                modified[:, : self.prompt_length, :] = token_vectors - coeff * direction
            elif self.intervention_kind == "replace_delta_component":
                source_vector = self.comparison_vector.to(device=output.device, dtype=output.dtype)
                source_coeff = torch.sum(source_vector * direction, dim=-1, keepdim=True)
                modified[:, : self.prompt_length, :] = token_vectors - coeff * direction + source_coeff * direction
            else:
                raise ValueError(f"Unsupported intervention kind `{self.intervention_kind}`.")
            return modified

        raise ValueError(f"Unsupported position policy `{self.position_policy}`.")


class FrozenRoutingIntervention:
    def __init__(self, cached_routing_by_layer: dict[int, dict[str, torch.Tensor]]):
        self.cached_routing_by_layer = cached_routing_by_layer
        self.original_forwards: list[tuple[Any, Any]] = []

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx, layer in enumerate(layers):
            cached = self.cached_routing_by_layer.get(layer_idx)
            if cached is None:
                continue
            gate = layer.mlp.gate
            original_forward = gate.forward
            self.original_forwards.append((gate, original_forward))

            def frozen_forward(hidden_states, _cached=cached, _orig=original_forward, _gate=gate):
                seq_len = hidden_states.reshape(-1, _gate.hidden_dim).shape[0]
                router_probs = _cached["router_probs"].to(device=hidden_states.device)
                top_k_weights = _cached["top_k_weights"]
                top_k_index = _cached["top_k_index"]
                if router_probs.shape[0] != seq_len or top_k_weights is None or top_k_index is None:
                    return _orig(hidden_states)
                return (
                    router_probs.to(dtype=hidden_states.dtype),
                    top_k_weights.to(device=hidden_states.device, dtype=hidden_states.dtype),
                    top_k_index.to(device=hidden_states.device),
                )

            gate.forward = frozen_forward

    def remove(self) -> None:
        for gate, original_forward in self.original_forwards:
            gate.forward = original_forward
        self.original_forwards.clear()


def run_capture_pass(
    model,
    batch: dict[str, Any],
    selected_layers: list[int],
    position_policy: str,
) -> tuple[float, dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]], dict[int, dict[str, torch.Tensor]]]:
    layer_capture = LayerOutputCapture(
        selected_layers=selected_layers,
        prompt_length=batch["prompt_length"],
        position_policy=position_policy,
    )
    component_capture = LayerComponentCapture(
        selected_layers=selected_layers,
        prompt_length=batch["prompt_length"],
        position_policy=position_policy,
    )
    router_capture = RouterCapture()
    layer_capture.attach(model)
    component_capture.attach(model)
    router_capture.attach(model)
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
        loss = float(outputs.loss.detach().cpu().item())
        return loss, dict(layer_capture.outputs), dict(router_capture.outputs), dict(component_capture.outputs)
    finally:
        layer_capture.remove()
        component_capture.remove()
        router_capture.remove()


def run_intervention_pass(
    model,
    batch: dict[str, Any],
    layer_idx: int,
    position_policy: str,
    target_vector: torch.Tensor,
    comparison_vector: torch.Tensor,
    intervention_kind: str,
    cached_routing_by_layer: dict[int, dict[str, torch.Tensor]] | None = None,
) -> float:
    intervention = LayerOutputIntervention(
        layer_idx=layer_idx,
        prompt_length=batch["prompt_length"],
        position_policy=position_policy,
        target_vector=target_vector,
        comparison_vector=comparison_vector,
        intervention_kind=intervention_kind,
    )
    frozen_routing = FrozenRoutingIntervention(cached_routing_by_layer) if cached_routing_by_layer else None
    intervention.attach(model)
    if frozen_routing is not None:
        frozen_routing.attach(model)
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
        return float(outputs.loss.detach().cpu().item())
    finally:
        if frozen_routing is not None:
            frozen_routing.remove()
        intervention.remove()


def run_generation_pass(
    model,
    batch: dict[str, Any],
    tokenizer,
    max_new_tokens: int,
    layer_idx: int | None = None,
    position_policy: str | None = None,
    target_vector: torch.Tensor | None = None,
    comparison_vector: torch.Tensor | None = None,
    intervention_kind: str | None = None,
    cached_routing_by_layer: dict[int, dict[str, torch.Tensor]] | None = None,
) -> str:
    intervention = None
    if layer_idx is not None and target_vector is not None and comparison_vector is not None and intervention_kind is not None:
        intervention = LayerOutputIntervention(
            layer_idx=layer_idx,
            prompt_length=batch["prompt_length"],
            position_policy=position_policy or "last_prompt_token",
            target_vector=target_vector,
            comparison_vector=comparison_vector,
            intervention_kind=intervention_kind,
        )
    frozen_routing = FrozenRoutingIntervention(cached_routing_by_layer) if cached_routing_by_layer else None
    if intervention is not None:
        intervention.attach(model)
    if frozen_routing is not None:
        frozen_routing.attach(model)
    try:
        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = int(batch["prompt_input_ids"].shape[-1])
        predicted_ids = generated[0, prompt_length:].detach().cpu().tolist()
        return tokenizer.decode(predicted_ids, skip_special_tokens=True)
    finally:
        if frozen_routing is not None:
            frozen_routing.remove()
        if intervention is not None:
            intervention.remove()


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for record in records:
        key = (
            str(record["dataset_name"]),
            str(record.get("language", "unknown")),
            int(record["layer"]),
        )
        grouped[key].append(record)

    summary = []
    for (dataset_name, language, layer), items in sorted(grouped.items()):
        baseline_items = [float(item["delta_loss_baseline"]) for item in items if item["delta_loss_baseline"] is not None]
        ratio_items = [float(item["delta_ratio"]) for item in items if item["delta_ratio"] is not None]
        summary.append(
            {
                "record_type": "causal_intervention_summary",
                "dataset_name": dataset_name,
                "language": language,
                "layer": layer,
                "num_examples": len(items),
                "mean_delta_loss": sum(float(item["delta_loss"]) for item in items) / len(items),
                "mean_baseline_loss": sum(float(item["baseline_loss"]) for item in items) / len(items),
                "mean_intervened_loss": sum(float(item["intervened_loss"]) for item in items) / len(items),
                "mean_delta_norm": sum(float(item["delta_norm"]) for item in items) / len(items),
            "mean_delta_loss_baseline": (
                    sum(baseline_items) / len(baseline_items) if baseline_items else None
                ),
                "mean_delta_ratio": sum(ratio_items) / len(ratio_items) if ratio_items else None,
                "mean_baseline_score": (
                    sum(float(item["baseline_score"]) for item in items if item["baseline_score"] is not None)
                    / len([item for item in items if item["baseline_score"] is not None])
                    if any(item["baseline_score"] is not None for item in items)
                    else None
                ),
                "mean_intervened_score": (
                    sum(float(item["intervened_score"]) for item in items if item["intervened_score"] is not None)
                    / len([item for item in items if item["intervened_score"] is not None])
                    if any(item["intervened_score"] is not None for item in items)
                    else None
                ),
                "mean_score_delta": (
                    sum(float(item["score_delta"]) for item in items if item["score_delta"] is not None)
                    / len([item for item in items if item["score_delta"] is not None])
                    if any(item["score_delta"] is not None for item in items)
                    else None
                ),
                "mean_baseline_top1_prob": sum(float(item["baseline_top1_prob"]) for item in items) / len(items),
                "mean_baseline_margin": sum(float(item["baseline_margin"]) for item in items) / len(items),
                "mean_baseline_entropy": sum(float(item["baseline_entropy"]) for item in items) / len(items),
            }
        )
    return summary


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def source_mode_context(model, source_mode: str, public_expert_idx: int):
    if source_mode == "comparison_model":
        return nullcontext(model)
    if source_mode == "public_only":
        return restricted_expert_mode(model, allowed_experts=[public_expert_idx])
    if source_mode == "backbone_only":
        return backbone_only_mode(model)
    raise ValueError(f"Unsupported source mode `{source_mode}`.")


def select_random_baseline_pair(num_examples: int, current_index: int) -> int | None:
    if num_examples <= 1:
        return None
    return (current_index + 1) % num_examples


def routing_probs_for_metrics(router_record: dict[str, torch.Tensor]) -> torch.Tensor:
    router_probs = router_record["router_probs"]
    if router_probs.ndim == 2:
        return router_probs.unsqueeze(0)
    return router_probs


def compute_dataset_mean_deltas(
    target_baselines: list[dict[str, Any]],
    source_examples: list[dict[str, Any]],
    selected_layers: list[int],
    delta_basis: str,
) -> dict[int, torch.Tensor]:
    layer_deltas: dict[int, list[torch.Tensor]] = {layer_idx: [] for layer_idx in selected_layers}
    if delta_basis == "expert_minus_backbone":
        for baseline_payload in target_baselines:
            component_outputs = baseline_payload["component_outputs"]
            for layer_idx in selected_layers:
                expert_output = component_outputs[layer_idx]["expert_output"]
                backbone_input = component_outputs[layer_idx]["backbone_input"]
                layer_deltas[layer_idx].append((expert_output - backbone_input).float())
    elif delta_basis == "source_mode_diff":
        for baseline_payload, source_payload in zip(target_baselines, source_examples):
            target_outputs = baseline_payload["target_outputs"]
            source_outputs = source_payload["layer_outputs"]
            for layer_idx in selected_layers:
                layer_deltas[layer_idx].append((target_outputs[layer_idx] - source_outputs[layer_idx]).float())
    else:
        raise ValueError(f"Unsupported delta basis `{delta_basis}`.")

    dataset_mean_deltas: dict[int, torch.Tensor] = {}
    for layer_idx, deltas in layer_deltas.items():
        if not deltas:
            continue
        stacked = torch.stack(deltas, dim=0)
        dataset_mean_deltas[layer_idx] = stacked.mean(dim=0)
    return dataset_mean_deltas


def baseline_target_vector(payload: dict[str, Any], layer_idx: int, delta_basis: str) -> torch.Tensor:
    if delta_basis == "expert_minus_backbone":
        return payload["component_outputs"][layer_idx]["expert_output"]
    if delta_basis == "source_mode_diff":
        return payload["target_outputs"][layer_idx]
    raise ValueError(f"Unsupported delta basis `{delta_basis}`.")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    target_model_path = resolve_model_path(
        explicit_path=args.target_model_path,
        model_name=args.target_model_name,
        model_root=args.target_model_root,
        model_registry=args.model_registry,
    )
    comparison_model_path = resolve_model_path(
        explicit_path=args.comparison_model_path,
        model_name=args.comparison_model_name,
        model_root=args.comparison_model_root,
        model_registry=args.model_registry,
    )
    target_model_name = resolved_model_name(args.target_model_name, target_model_path)
    comparison_model_name = resolved_model_name(args.comparison_model_name, comparison_model_path)

    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}

    tokenizer_path = args.tokenizer_path or target_model_path

    manifest_entries = load_manifest_entries(args.manifest_path, selected_datasets)
    if not manifest_entries:
        raise ValueError("No mix datasets were selected from the manifest.")

    model_output_root = (
        Path(args.output_root)
        / f"{target_model_name}__vs__{comparison_model_name}"
    )
    model_output_root.mkdir(parents=True, exist_ok=True)

    target_model, target_tokenizer = load_model_and_tokenizer(
        model_path=target_model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype_name=args.dtype,
    )
    selected_layers = parse_decoder_layers(args.selected_layers, int(target_model.config.num_hidden_layers))
    if not selected_layers:
        raise ValueError("Provide at least one selected layer.")

    comparison_model = None
    use_comparison_model = args.source_mode == "comparison_model" and args.delta_basis == "source_mode_diff"
    if use_comparison_model:
        comparison_model, comparison_tokenizer = load_model_and_tokenizer(
            model_path=comparison_model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            dtype_name=args.dtype,
        )
    else:
        comparison_tokenizer = target_tokenizer

    comparison_vectors_by_dataset: dict[str, dict[str, Any]] = {}

    for dataset_entry in manifest_entries:
        dataset_name = str(dataset_entry["name"])
        records = load_jsonl_records(dataset_entry["path"], max_examples=args.max_examples_per_dataset)
        examples = [
            normalize_example(comparison_tokenizer, record=record, dataset_name=dataset_name, dataset_entry=dataset_entry)
            for record in records
        ]
        dataset_vectors: list[dict[str, Any]] = []
        for example in examples:
            payload = {"example": example}
            if args.delta_basis == "source_mode_diff":
                batch = build_teacher_forced_batch(
                    tokenizer=comparison_tokenizer,
                    prompt=example["prompt"],
                    reference_answer=example["reference_answer"],
                    max_length=args.max_length,
                    device=device,
                )
                if use_comparison_model:
                    context = nullcontext(comparison_model)
                    source_model = comparison_model
                else:
                    context = source_mode_context(target_model, args.source_mode, args.public_expert_idx)
                    source_model = target_model
                with context:
                    _loss, layer_outputs, _router_outputs, _component_outputs = run_capture_pass(
                        model=source_model,
                        batch=batch,
                        selected_layers=selected_layers,
                        position_policy=args.position_policy,
                    )
                payload.update(
                    {
                        "prompt_length": batch["prompt_length"],
                        "layer_outputs": layer_outputs,
                    }
                )
            dataset_vectors.append(payload)
        comparison_vectors_by_dataset[dataset_name] = {
            "dataset_entry": dataset_entry,
            "examples": dataset_vectors,
        }
        print(
            f"Prepared causal source payloads for {args.delta_basis} on {dataset_name} "
            f"({len(dataset_vectors)} examples)."
        )

    if comparison_model is not None:
        free_model(comparison_model)

    suite_manifest = {
        "target_model_name": target_model_name,
        "target_model_path": target_model_path,
        "comparison_model_name": comparison_model_name,
        "comparison_model_path": comparison_model_path,
        "source_mode": args.source_mode,
        "public_expert_idx": args.public_expert_idx,
        "tokenizer_path": tokenizer_path,
        "model_impl_path": str(inspect.getsourcefile(FlexOlmoForCausalLM)),
        "device": str(device),
        "dtype": args.dtype,
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "selected_layers": selected_layers,
        "position_policy": args.position_policy,
        "intervention_kind": args.intervention_kind,
        "delta_aggregation": args.delta_aggregation,
        "delta_basis": args.delta_basis,
        "routing_frozen": True,
        "baseline_kind": "random_real_diff",
        "max_examples_per_dataset": args.max_examples_per_dataset,
        "datasets": {},
    }

    for dataset_name, payload in comparison_vectors_by_dataset.items():
        dataset_dir = model_output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        intervention_records: list[dict] = []
        target_baselines: list[dict[str, Any]] = []
        for item in payload["examples"]:
            example = item["example"]
            batch = build_teacher_forced_batch(
                tokenizer=target_tokenizer,
                prompt=example["prompt"],
                reference_answer=example["reference_answer"],
                max_length=args.max_length,
                device=device,
            )
            baseline_loss, target_outputs, router_outputs, component_outputs = run_capture_pass(
                model=target_model,
                batch=batch,
                selected_layers=selected_layers,
                position_policy=args.position_policy,
            )
            target_baselines.append(
                {
                    "example": example,
                    "batch": batch,
                    "baseline_loss": baseline_loss,
                    "target_outputs": target_outputs,
                    "router_outputs": router_outputs,
                    "component_outputs": component_outputs,
                }
            )

        dataset_mean_deltas = compute_dataset_mean_deltas(
            target_baselines=target_baselines,
            source_examples=payload["examples"],
            selected_layers=selected_layers,
            delta_basis=args.delta_basis,
        )

        for example_idx, item in enumerate(payload["examples"]):
            example = item["example"]
            baseline_payload = target_baselines[example_idx]
            batch = baseline_payload["batch"]
            baseline_loss = baseline_payload["baseline_loss"]
            target_outputs = baseline_payload["target_outputs"]
            router_outputs = baseline_payload["router_outputs"]
            random_baseline_idx = select_random_baseline_pair(len(target_baselines), example_idx)
            max_new_tokens = int(example.get("generation_config", {}).get("max_new_tokens", 16))

            for layer_idx in selected_layers:
                if args.delta_basis == "expert_minus_backbone":
                    target_vector = baseline_payload["component_outputs"][layer_idx]["expert_output"]
                    source_vector = baseline_payload["component_outputs"][layer_idx]["backbone_input"]
                    target_component_name = "expert_output"
                    comparison_component_name = "backbone_input"
                    if args.delta_aggregation == "dataset_mean":
                        comparison_vector = target_vector - dataset_mean_deltas[layer_idx]
                        delta_type = "dataset_mean_expert_minus_backbone"
                    else:
                        comparison_vector = source_vector
                        delta_type = "expert_minus_backbone"
                else:
                    source_vector = item["layer_outputs"][layer_idx]
                    target_vector = target_outputs[layer_idx]
                    target_component_name = "mlp_output"
                    comparison_component_name = (
                        "dataset_mean_source_mode_diff" if args.delta_aggregation == "dataset_mean" else args.source_mode
                    )
                    if args.delta_aggregation == "dataset_mean":
                        comparison_vector = target_vector - dataset_mean_deltas[layer_idx]
                        delta_type = f"dataset_mean_{target_model_name}_minus_{comparison_model_name}"
                    else:
                        comparison_vector = source_vector
                        delta_type = f"{target_model_name}_minus_{comparison_model_name}"
                intervened_loss = run_intervention_pass(
                    model=target_model,
                    batch=batch,
                    layer_idx=layer_idx,
                    position_policy=args.position_policy,
                    target_vector=target_vector,
                    comparison_vector=comparison_vector,
                    intervention_kind=args.intervention_kind,
                    cached_routing_by_layer=router_outputs,
                )
                router_metrics = summarize_router_metrics(
                    router_probs=routing_probs_for_metrics(router_outputs[layer_idx]),
                    prompt_length=batch["prompt_length"],
                    position_policy=args.position_policy,
                )
                delta = target_vector - comparison_vector
                baseline_delta_loss = None
                delta_ratio = None
                baseline_pair_example_id = None
                if random_baseline_idx is not None:
                    random_vector = baseline_target_vector(
                        target_baselines[random_baseline_idx], layer_idx, args.delta_basis
                    )
                    baseline_pair_example_id = target_baselines[random_baseline_idx]["example"]["example_id"]
                    baseline_intervened_loss = run_intervention_pass(
                        model=target_model,
                        batch=batch,
                        layer_idx=layer_idx,
                        position_policy=args.position_policy,
                        target_vector=target_vector,
                        comparison_vector=random_vector,
                        intervention_kind=args.intervention_kind,
                        cached_routing_by_layer=router_outputs,
                    )
                    baseline_delta_loss = baseline_intervened_loss - baseline_loss
                    if abs(baseline_delta_loss) > 1e-12:
                        delta_ratio = (intervened_loss - baseline_loss) / baseline_delta_loss

                baseline_prediction = run_generation_pass(
                    model=target_model,
                    batch=batch,
                    tokenizer=target_tokenizer,
                    max_new_tokens=max_new_tokens,
                    cached_routing_by_layer=router_outputs,
                )
                intervened_prediction = run_generation_pass(
                    model=target_model,
                    batch=batch,
                    tokenizer=target_tokenizer,
                    max_new_tokens=max_new_tokens,
                    layer_idx=layer_idx,
                    position_policy=args.position_policy,
                    target_vector=target_vector,
                    comparison_vector=comparison_vector,
                    intervention_kind=args.intervention_kind,
                    cached_routing_by_layer=router_outputs,
                )
                baseline_scoring = score_prediction(example, baseline_prediction)
                intervened_scoring = score_prediction(example, intervened_prediction)
                intervention_records.append(
                    {
                        "record_type": "causal_intervention",
                        "example_id": example["example_id"],
                        "dataset_name": dataset_name,
                        "language": example["language"],
                        "domain": example.get("domain"),
                        "model_name": target_model_name,
                        "model_path": target_model_path,
                        "comparison_model_name": comparison_model_name,
                        "comparison_model_path": comparison_model_path,
                        "source_mode": args.source_mode,
                        "delta_basis": args.delta_basis,
                        "layer": layer_idx,
                        "position_policy": args.position_policy,
                        "intervention_kind": args.intervention_kind,
                        "delta_aggregation": args.delta_aggregation,
                        "delta_type": delta_type,
                        "delta_norm": float(torch.linalg.vector_norm(delta).item()),
                        "target_component_name": target_component_name,
                        "comparison_component_name": comparison_component_name,
                        "target_component_norm": float(torch.linalg.vector_norm(target_vector).item()),
                        "comparison_component_norm": float(torch.linalg.vector_norm(comparison_vector).item()),
                        "baseline_loss": baseline_loss,
                        "intervened_loss": intervened_loss,
                        "delta_loss": intervened_loss - baseline_loss,
                        "delta_loss_baseline": baseline_delta_loss,
                        "delta_ratio": delta_ratio,
                        "scoring_mode": example.get("scoring_mode"),
                        "reference_answer": example["reference_answer"],
                        "baseline_prediction_text": baseline_prediction,
                        "intervened_prediction_text": intervened_prediction,
                        "baseline_prediction_text_clean": baseline_scoring["prediction_text_clean"],
                        "intervened_prediction_text_clean": intervened_scoring["prediction_text_clean"],
                        "baseline_score": baseline_scoring["score"],
                        "intervened_score": intervened_scoring["score"],
                        "score_delta": intervened_scoring["score"] - baseline_scoring["score"],
                        "baseline_is_correct": baseline_scoring["is_correct"],
                        "intervened_is_correct": intervened_scoring["is_correct"],
                        "baseline_exact_match": baseline_scoring["exact_match"],
                        "intervened_exact_match": intervened_scoring["exact_match"],
                        "baseline_relaxed_match": baseline_scoring["relaxed_match"],
                        "intervened_relaxed_match": intervened_scoring["relaxed_match"],
                        "baseline_token_f1": baseline_scoring["token_f1"],
                        "intervened_token_f1": intervened_scoring["token_f1"],
                        "baseline_normalized_prediction": baseline_scoring["normalized_prediction"],
                        "intervened_normalized_prediction": intervened_scoring["normalized_prediction"],
                        "normalized_reference": baseline_scoring["normalized_reference"],
                        "baseline_kind": "random_real_diff",
                        "baseline_pair_example_id": baseline_pair_example_id,
                        "routing_frozen": True,
                        **router_metrics,
                        "metadata": {
                            "prompt_length": batch["prompt_length"],
                            "answer_length": batch["answer_length"],
                            "sequence_length": batch["sequence_length"],
                        },
                    }
                )

        summary_records = build_summary(intervention_records)
        records_path = dataset_dir / "causal_intervention_records.jsonl"
        summary_path = dataset_dir / "causal_intervention_summary.jsonl"
        write_jsonl(records_path, intervention_records)
        write_jsonl(summary_path, summary_records)

        run_manifest = {
            "dataset_name": dataset_name,
            "num_examples": len(payload["examples"]),
            "selected_layers": selected_layers,
            "position_policy": args.position_policy,
            "intervention_kind": args.intervention_kind,
            "source_mode": args.source_mode,
            "delta_aggregation": args.delta_aggregation,
            "delta_basis": args.delta_basis,
            "routing_frozen": True,
            "baseline_kind": "random_real_diff",
            "records_path": str(records_path),
            "summary_path": str(summary_path),
        }
        (dataset_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        suite_manifest["datasets"][dataset_name] = run_manifest
        print(
            f"Wrote causal intervention records for {target_model_name} vs {comparison_model_name} "
            f"on {dataset_name} ({len(payload['examples'])} examples)."
        )

    suite_manifest_path = model_output_root / "causal_intervention_suite_manifest.json"
    suite_manifest_path.write_text(json.dumps(suite_manifest, indent=2), encoding="utf-8")
    print(f"Wrote causal intervention suite manifest to {suite_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
