from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import torch

from flex_moe_toolkit.core.routing_diagnostics import compute_all_metrics, compute_offdiagonal_ratio
from flex_moe_toolkit.pipelines.flex_olmo import analyze_flex_olmo_routing, restricted_expert_mode
from flex_moe_toolkit.utils.jsonl import write_jsonl
from flex_moe_toolkit.utils.router_activity import (
    activated_expert_combination,
    count_combinations,
    flatten_topk_experts,
    layer_iou_summary,
)


DEFAULT_MULTIPLE_CHOICE_PROMPT = (
    "{context_block}"
    "{question_prefix}{question}\n"
    "Choices:\n"
    "{choice_block}\n"
    "Answer:"
)


@dataclass(frozen=True)
class FlexOlmoEvalRunSpec:
    label: str
    allowed_experts: tuple[int, ...]


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_eval_example(example: dict[str, Any], example_index: int) -> dict[str, Any]:
    example_id = example.get("id", example.get("example_id", f"example_{example_index:05d}"))
    dataset_name = example.get("dataset", example.get("benchmark", "unknown_dataset"))
    language = example.get("language", "unknown")
    task_type = example.get("task_type", "multiple_choice" if "choices" in example else "target_scoring")

    if "choices" in example:
        if "question" not in example:
            raise ValueError(
                f"Example `{example_id}` is missing `question` for multiple-choice scoring."
            )
        if "correct_choice_idx" not in example:
            raise ValueError(
                f"Example `{example_id}` is missing `correct_choice_idx` for multiple-choice scoring."
            )
        choices = example["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Example `{example_id}` must provide a non-empty `choices` list.")
        return {
            "example_id": example_id,
            "dataset": dataset_name,
            "benchmark": example.get("benchmark", dataset_name),
            "language": language,
            "task_type": task_type,
            "context": example.get("context", ""),
            "question": example["question"],
            "choices": [str(choice) for choice in choices],
            "correct_choice_idx": int(example["correct_choice_idx"]),
            "metadata": example.get("metadata", {}),
        }

    prompt = example.get("prompt", example.get("input", example.get("question")))
    target = example.get("target", example.get("answer", example.get("completion", example.get("reference"))))
    if prompt is None or target is None:
        raise ValueError(
            f"Example `{example_id}` must provide either MCQ fields "
            "(`question`, `choices`, `correct_choice_idx`) or target-scoring fields "
            "(`prompt` plus `target`/`answer`/`completion`/`reference`)."
        )

    return {
        "example_id": example_id,
        "dataset": dataset_name,
        "benchmark": example.get("benchmark", dataset_name),
        "language": language,
        "task_type": task_type,
        "prompt": str(prompt),
        "target": str(target),
        "metadata": example.get("metadata", {}),
    }


def build_multiple_choice_prompt(example: dict[str, Any]) -> tuple[str, list[str]]:
    context = str(example.get("context", "")).strip()
    context_block = f"Context: {context}\n" if context else ""
    choice_lines = [f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(example["choices"])]
    prompt = DEFAULT_MULTIPLE_CHOICE_PROMPT.format(
        context_block=context_block,
        question_prefix="Question: ",
        question=example["question"],
        choice_block="\n".join(choice_lines),
    )
    continuations = [f" {choice}" for choice in example["choices"]]
    return prompt, continuations


def build_prompt_and_targets(example: dict[str, Any]) -> tuple[str, list[str], int]:
    if "choices" in example:
        prompt, continuations = build_multiple_choice_prompt(example)
        return prompt, continuations, int(example["correct_choice_idx"])

    return str(example["prompt"]), [str(example["target"])], 0


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _encode_prompt_continuation(
    tokenizer,
    prompt: str,
    continuation: str,
    max_length: int,
) -> tuple[list[int], list[int], list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)

    if not continuation_ids:
        raise ValueError("Encountered an empty continuation after tokenization; cannot score it.")

    available_continuation_tokens = max(1, max_length - len(prompt_ids))
    if len(continuation_ids) > available_continuation_tokens:
        continuation_ids = continuation_ids[:available_continuation_tokens]

    input_ids = prompt_ids + continuation_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + continuation_ids
    return input_ids, attention_mask, labels


def tokenize_scored_candidates(
    tokenizer,
    prompt: str,
    continuations: list[str],
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded_inputs = []
    encoded_masks = []
    encoded_labels = []

    for continuation in continuations:
        input_ids, attention_mask, labels = _encode_prompt_continuation(
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=continuation,
            max_length=max_length,
        )
        encoded_inputs.append(torch.tensor(input_ids, dtype=torch.long))
        encoded_masks.append(torch.tensor(attention_mask, dtype=torch.long))
        encoded_labels.append(torch.tensor(labels, dtype=torch.long))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(
            "Tokenizer does not define `pad_token_id`. Set `tokenizer.pad_token = tokenizer.eos_token` before running."
        )

    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            encoded_inputs, batch_first=True, padding_value=pad_token_id
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            encoded_masks, batch_first=True, padding_value=0
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            encoded_labels, batch_first=True, padding_value=-100
        ),
    }


def compute_sequence_scores(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]

    token_log_probs = torch.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
    gathered = token_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = shift_labels != -100
    masked_scores = gathered * mask
    token_counts = mask.sum(dim=-1).clamp_min(1)
    return (masked_scores.sum(dim=-1) / token_counts).detach().cpu().tolist()


def ensure_router_logits_3d(router_logits: torch.Tensor) -> torch.Tensor:
    if router_logits.ndim == 2:
        return router_logits.unsqueeze(0)
    if router_logits.ndim == 4:
        return router_logits.flatten(0, 1)
    return router_logits


def build_run_specs(
    num_experts: int,
    public_expert_idx: int = 0,
    combined_active_counts: tuple[int, ...] = (2, 4, 7),
    include_individual_experts: bool = True,
    expert_order: tuple[int, ...] | None = None,
) -> list[FlexOlmoEvalRunSpec]:
    if expert_order is None:
        expert_order = tuple(range(num_experts))
    if len(expert_order) != num_experts:
        raise ValueError(
            "`expert_order` must list every expert exactly once so combined runs are reproducible."
        )

    run_specs = [FlexOlmoEvalRunSpec(label="public_only", allowed_experts=(public_expert_idx,))]

    if include_individual_experts:
        for expert_idx in range(num_experts):
            if expert_idx == public_expert_idx:
                continue
            run_specs.append(
                FlexOlmoEvalRunSpec(
                    label=f"single_expert_{expert_idx}",
                    allowed_experts=(expert_idx,),
                )
            )

    for active_count in combined_active_counts:
        if active_count > num_experts:
            continue
        run_specs.append(
            FlexOlmoEvalRunSpec(
                label=f"combined_top{active_count}",
                allowed_experts=tuple(expert_order[:active_count]),
            )
        )

    return run_specs


def evaluate_example(
    model,
    tokenizer,
    example: dict[str, Any],
    run_spec: FlexOlmoEvalRunSpec,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    prompt, continuations, correct_choice_idx = build_prompt_and_targets(example)
    candidate_inputs = tokenize_scored_candidates(
        tokenizer=tokenizer,
        prompt=prompt,
        continuations=continuations,
        max_length=max_length,
    )
    candidate_inputs = move_batch_to_device(candidate_inputs, device)

    with restricted_expert_mode(model, allowed_experts=run_spec.allowed_experts):
        with torch.no_grad():
            outputs = model(
                input_ids=candidate_inputs["input_ids"],
                attention_mask=candidate_inputs["attention_mask"],
            )
        choice_scores = compute_sequence_scores(outputs.logits, candidate_inputs["labels"])
        predicted_choice_idx = max(range(len(choice_scores)), key=lambda idx: choice_scores[idx])

        winning_inputs = {
            "input_ids": candidate_inputs["input_ids"][predicted_choice_idx : predicted_choice_idx + 1],
            "attention_mask": candidate_inputs["attention_mask"][predicted_choice_idx : predicted_choice_idx + 1],
        }
        routing = analyze_flex_olmo_routing(
            model,
            winning_inputs,
            top_k=min(len(run_spec.allowed_experts), model.config.num_experts_per_tok),
        )

    first_layer_router_logits = ensure_router_logits_3d(routing["router_logits"][0])
    metrics_top_k = min(len(run_spec.allowed_experts), 2, first_layer_router_logits.shape[-1])
    batch_metrics = compute_all_metrics(first_layer_router_logits, top_k=metrics_top_k)
    layer_batch_routing_metrics = []
    for layer_idx, layer_router_logits in enumerate(routing["router_logits"]):
        normalized_logits = ensure_router_logits_3d(layer_router_logits)
        layer_metrics = compute_all_metrics(
            normalized_logits,
            top_k=min(len(run_spec.allowed_experts), 2, normalized_logits.shape[-1]),
        )
        layer_batch_routing_metrics.append(
            {
                "layer_idx": layer_idx,
                "usage": layer_metrics["usage"],
                "entropy_mean": layer_metrics["entropy_mean"],
                "coactivation_matrix": layer_metrics["coactivation_matrix"],
                "offdiag_ratio": layer_metrics["offdiag_ratio"],
            }
        )

    layer_combos = flatten_topk_experts(routing["topk_experts"])
    global_combo = activated_expert_combination(routing["topk_experts"])
    layer_overlap = layer_iou_summary(layer_combos)
    token_combination_counts = count_token_level_combinations(routing["topk_experts"])

    record = {
        "record_type": "evaluation_example",
        "run_label": run_spec.label,
        "available_experts": run_spec.allowed_experts,
        "num_available_experts": len(run_spec.allowed_experts),
        "example_id": example["example_id"],
        "dataset": example["dataset"],
        "benchmark": example["benchmark"],
        "language": example["language"],
        "task_type": example["task_type"],
        "prompt": prompt,
        "choices": continuations,
        "correct_choice_idx": correct_choice_idx,
        "predicted_choice_idx": predicted_choice_idx,
        "is_correct": predicted_choice_idx == correct_choice_idx,
        "choice_scores": choice_scores,
        "entropy": routing["entropy"],
        "load_balance": routing["load_balance"],
        "expert_usage": routing["expert_usage"],
        "layer_expert_matrix": routing["layer_expert_matrix"],
        "batch_routing_metrics": {
            "usage": batch_metrics["usage"],
            "entropy_mean": batch_metrics["entropy_mean"],
            "coactivation_matrix": batch_metrics["coactivation_matrix"],
            "offdiag_ratio": batch_metrics["offdiag_ratio"],
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
    if "question" in example:
        record["question"] = example["question"]
        record["choices_text"] = example["choices"]
    if "target" in example:
        record["target"] = example["target"]
    return record


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


def build_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    by_run = defaultdict(list)
    for record in records:
        by_run[record["run_label"]].append(record)

    for run_label, run_records in by_run.items():
        total = len(run_records)
        correct = sum(1 for record in run_records if record["is_correct"])
        by_benchmark = defaultdict(lambda: {"examples": 0, "correct": 0})
        by_language = defaultdict(lambda: {"examples": 0, "correct": 0})

        for record in run_records:
            by_benchmark[record["benchmark"]]["examples"] += 1
            by_benchmark[record["benchmark"]]["correct"] += int(record["is_correct"])
            by_language[record["language"]]["examples"] += 1
            by_language[record["language"]]["correct"] += int(record["is_correct"])

        combination_counts = count_combinations(record["activated_experts"] for record in run_records)
        layer_pattern_counts = Counter(
            tuple(tuple(layer_combo) for layer_combo in record["layer_activated_experts"])
            for record in run_records
        )

        summaries.append(
            {
                "record_type": "evaluation_summary",
                "run_label": run_label,
                "num_examples": total,
                "num_correct": correct,
                "accuracy": correct / total if total else 0.0,
                "available_experts": run_records[0]["available_experts"],
                "activated_expert_combinations": {
                    ",".join(str(expert) for expert in combo): count
                    for combo, count in sorted(combination_counts.items())
                },
                "mean_layer_iou": (
                    sum(record["mean_layer_iou"] for record in run_records) / total if total else 0.0
                ),
                "by_benchmark": {
                    benchmark: {
                        **stats,
                        "accuracy": stats["correct"] / stats["examples"] if stats["examples"] else 0.0,
                    }
                    for benchmark, stats in by_benchmark.items()
                },
                "by_language": {
                    language: {
                        **stats,
                        "accuracy": stats["correct"] / stats["examples"] if stats["examples"] else 0.0,
                    }
                    for language, stats in by_language.items()
                },
                "layer_pattern_counts": {
                    " | ".join(
                        "{" + ",".join(str(expert) for expert in combo) + "}"
                        for combo in layer_pattern
                    ): count
                    for layer_pattern, count in layer_pattern_counts.items()
                },
            }
        )

    return summaries


def aggregate_routing_analysis(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        raise ValueError("No records were provided for routing analysis.")

    usage_sum = None
    entropy_total = 0.0
    layer_coactivation_sums = []
    batch_records = []

    for record in records:
        metrics = record["batch_routing_metrics"]
        usage = metrics["usage"].detach().cpu()
        entropy = float(metrics["entropy_mean"])

        usage_sum = usage.clone() if usage_sum is None else usage_sum + usage
        entropy_total += entropy

        for layer_metrics in record["layer_batch_routing_metrics"]:
            layer_idx = layer_metrics["layer_idx"]
            layer_coactivation = layer_metrics["coactivation_matrix"].detach().cpu()
            while len(layer_coactivation_sums) <= layer_idx:
                layer_coactivation_sums.append(None)
            layer_coactivation_sums[layer_idx] = (
                layer_coactivation.clone()
                if layer_coactivation_sums[layer_idx] is None
                else layer_coactivation_sums[layer_idx] + layer_coactivation
            )

        batch_records.append(
            {
                "record_type": "routing_batch",
                "run_label": record["run_label"],
                "example_id": record["example_id"],
                "usage": usage,
                "entropy_mean": entropy,
                "coactivation_matrix": metrics["coactivation_matrix"].detach().cpu(),
                "offdiag_ratio": metrics["offdiag_ratio"],
            }
        )

    aggregate_usage = usage_sum / usage_sum.sum()
    aggregate_entropy = entropy_total / len(records)
    non_null_layer_matrices = [matrix for matrix in layer_coactivation_sums if matrix is not None]
    if not non_null_layer_matrices:
        raise ValueError("No layer-wise coactivation matrices were collected.")

    coactivation_sum = non_null_layer_matrices[0].clone()
    for matrix in non_null_layer_matrices[1:]:
        coactivation_sum = coactivation_sum + matrix

    aggregate_record = {
        "record_type": "routing_aggregate",
        "num_batches": len(records),
        "usage": aggregate_usage,
        "entropy_mean": aggregate_entropy,
        "coactivation_matrix": coactivation_sum,
        "layer_coactivation_matrices": layer_coactivation_sums,
        "offdiag_ratio": compute_offdiagonal_ratio(coactivation_sum),
    }

    return batch_records + [aggregate_record], aggregate_record


def evaluate_dataset_across_runs(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    run_specs: list[FlexOlmoEvalRunSpec],
    max_length: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    normalized_examples = [
        normalize_eval_example(example, example_index=example_index)
        for example_index, example in enumerate(examples)
    ]

    results = {}
    for run_spec in run_specs:
        records = [
            evaluate_example(
                model=model,
                tokenizer=tokenizer,
                example=example,
                run_spec=run_spec,
                max_length=max_length,
                device=device,
            )
            for example in normalized_examples
        ]
        summaries = build_summary(records)
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
) -> dict[str, dict[str, str]]:
    output_root = Path(output_root)
    dataset_dir = output_root / sanitize_name(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for run_label, payload in run_results.items():
        run_dir = dataset_dir / sanitize_name(run_label)
        run_dir.mkdir(parents=True, exist_ok=True)

        records_path = run_dir / "eval_records.jsonl"
        summary_path = run_dir / "eval_summary.jsonl"
        routing_path = run_dir / "routing_analysis.jsonl"

        write_jsonl(payload["records"], records_path)
        write_jsonl(payload["summaries"], summary_path)
        write_jsonl(payload["routing_analysis_records"], routing_path)

        manifest[run_label] = {
            "records_path": str(records_path),
            "summary_path": str(summary_path),
            "routing_analysis_path": str(routing_path),
        }

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
