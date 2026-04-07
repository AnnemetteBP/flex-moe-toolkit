from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import seaborn as sns

from fake_test_models.fake_flex_olmo import FakeFlexOlmoModel
from flex_moe_toolkit.core.routing_diagnostics import compute_all_metrics, compute_offdiagonal_ratio
from flex_moe_toolkit.pipelines.flex_olmo import analyze_flex_olmo_routing, restricted_expert_mode
from flex_moe_toolkit.prev_analysis.plots import (
    plot_expert_combination_upset,
    plot_run_comparison_upset,
)
from flex_moe_toolkit.utils.jsonl import write_jsonl
from flex_moe_toolkit.utils.router_activity import (
    activated_expert_combination,
    count_combinations,
    flatten_topk_experts,
    layer_iou_summary,
)


DATASET_PATH = PROJECT_ROOT / "fake_test_models" / "datasets" / "fake_eval_suite.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "flex_olmo" / "combined_flex"
EVAL_RECORDS_PATH = OUTPUT_DIR / "fake_eval_router_activity.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "fake_eval_summary.jsonl"
ROUTING_ANALYSIS_PATH = OUTPUT_DIR / "routing_analysis.jsonl"
UPSET_PATH = OUTPUT_DIR / "fake_eval_expert_upset.png"
UPSET_2_PATH = OUTPUT_DIR / "fake_eval_expert_upset_top2_active.png"
UPSET_4_PATH = OUTPUT_DIR / "fake_eval_expert_upset_top4_active.png"
USAGE_PLOT_PATH = OUTPUT_DIR / "routing_usage_bar.png"
COACTIVATION_PLOT_PATH = OUTPUT_DIR / "routing_coactivation_heatmap.png"

RUN_SPECS = (
    {"run_label": "top2_active", "active_experts": (0, 1), "top_k": 2},
    {"run_label": "top4_active", "active_experts": (0, 1, 2, 3), "top_k": 4},
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_to_fake_inputs(text, hidden_size, sequence_length=12):
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(1, sequence_length, hidden_size, generator=generator)
    return {"input_ids": hidden_states}


def score_choice(model, question, choice_text, active_experts, top_k):
    prompt = f"Question: {question}\nChoice: {choice_text}"
    inputs = text_to_fake_inputs(prompt, hidden_size=model.config.hidden_size)

    with restricted_expert_mode(model, allowed_experts=active_experts):
        outputs = model(**inputs)
        routing = analyze_flex_olmo_routing(model, inputs, top_k=min(top_k, len(active_experts)))

    final_token = outputs.last_hidden_state[:, -1, :]
    score = float(final_token.mean().item())
    return score, routing


def evaluate_example(model, example, run_spec):
    choice_runs = []

    for choice_idx, choice_text in enumerate(example["choices"]):
        score, routing = score_choice(
            model=model,
            question=example["question"],
            choice_text=choice_text,
            active_experts=run_spec["active_experts"],
            top_k=run_spec["top_k"],
        )
        choice_runs.append(
            {
                "choice_idx": choice_idx,
                "choice_text": choice_text,
                "score": score,
                "routing": routing,
            }
        )

    predicted = max(choice_runs, key=lambda item: item["score"])
    predicted_idx = predicted["choice_idx"]
    predicted_routing = predicted["routing"]
    first_layer_router_logits = predicted_routing["router_logits"][0]
    if first_layer_router_logits.ndim == 2:
        first_layer_router_logits = first_layer_router_logits.unsqueeze(0)
    elif first_layer_router_logits.ndim == 4:
        first_layer_router_logits = first_layer_router_logits.flatten(0, 1)
    batch_metrics = compute_all_metrics(first_layer_router_logits, top_k=2)
    layer_combos = flatten_topk_experts(predicted_routing["topk_experts"])
    global_combo = activated_expert_combination(predicted_routing["topk_experts"])
    layer_overlap = layer_iou_summary(layer_combos)

    return {
        "record_type": "evaluation_example",
        "run_label": run_spec["run_label"],
        "available_experts": tuple(run_spec["active_experts"]),
        "num_available_experts": len(run_spec["active_experts"]),
        "example_id": example["id"],
        "benchmark": example["benchmark"],
        "language": example["language"],
        "task_type": example["task_type"],
        "question": example["question"],
        "choices": example["choices"],
        "correct_choice_idx": example["correct_choice_idx"],
        "predicted_choice_idx": predicted_idx,
        "is_correct": predicted_idx == example["correct_choice_idx"],
        "choice_scores": [run["score"] for run in choice_runs],
        "entropy": predicted_routing["entropy"],
        "load_balance": predicted_routing["load_balance"],
        "expert_usage": predicted_routing["expert_usage"],
        "layer_expert_matrix": predicted_routing["layer_expert_matrix"],
        "batch_routing_metrics": {
            "usage": batch_metrics["usage"],
            "entropy_mean": batch_metrics["entropy_mean"],
            "coactivation_matrix": batch_metrics["coactivation_matrix"],
            "offdiag_ratio": batch_metrics["offdiag_ratio"],
        },
        "layer_activated_experts": layer_combos,
        "activated_experts": global_combo,
        "layer_intersection_experts": layer_overlap["layer_intersection_experts"],
        "layer_union_experts": layer_overlap["layer_union_experts"],
        "pairwise_layer_iou": layer_overlap["pairwise_layer_iou"],
        "mean_layer_iou": layer_overlap["mean_layer_iou"],
    }


def build_summary(records):
    summaries = []
    intersection_records = []

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
        for combination, count in sorted(combination_counts.items()):
            matching_records = [
                record for record in run_records if tuple(record["activated_experts"]) == tuple(combination)
            ]
            intersection_records.append(
                {
                    "run_label": run_label,
                    "combination": combination,
                    "count": count,
                    "mean_layer_iou": sum(record["mean_layer_iou"] for record in matching_records)
                    / len(matching_records),
                }
            )

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
                "mean_layer_iou": sum(
                    record["mean_layer_iou"] for record in run_records
                )
                / total
                if total
                else 0.0,
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

    return summaries, intersection_records


def aggregate_routing_analysis(records):
    if not records:
        raise ValueError("No records were provided for routing analysis.")

    usage_sum = None
    coactivation_sum = None
    entropy_total = 0.0
    batch_records = []

    for record in records:
        metrics = record["batch_routing_metrics"]
        usage = metrics["usage"].detach().cpu()
        coactivation = metrics["coactivation_matrix"].detach().cpu()
        entropy = float(metrics["entropy_mean"])

        usage_sum = usage.clone() if usage_sum is None else usage_sum + usage
        coactivation_sum = coactivation.clone() if coactivation_sum is None else coactivation_sum + coactivation
        entropy_total += entropy

        batch_records.append(
            {
                "record_type": "routing_batch",
                "run_label": record["run_label"],
                "example_id": record["example_id"],
                "usage": usage,
                "entropy_mean": entropy,
                "coactivation_matrix": coactivation,
                "offdiag_ratio": metrics["offdiag_ratio"],
            }
        )

    aggregate_usage = usage_sum / usage_sum.sum()
    aggregate_entropy = entropy_total / len(records)
    aggregate_offdiag_ratio = compute_offdiagonal_ratio(coactivation_sum)

    aggregate_record = {
        "record_type": "routing_aggregate",
        "num_batches": len(records),
        "usage": aggregate_usage,
        "entropy_mean": aggregate_entropy,
        "coactivation_matrix": coactivation_sum,
        "offdiag_ratio": aggregate_offdiag_ratio,
    }

    return batch_records + [aggregate_record], aggregate_record


def save_usage_bar_plot(usage, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    expert_indices = list(range(len(usage)))
    ax.bar(expert_indices, usage, color="#2f6db2")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Normalized usage")
    ax.set_title("Aggregate Expert Usage")
    ax.set_xticks(expert_indices)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_coactivation_heatmap(matrix, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, cmap="viridis", ax=ax)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Expert")
    ax.set_title("Aggregate Coactivation Matrix")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    examples = load_jsonl(DATASET_PATH)
    model = FakeFlexOlmoModel(num_experts=7)

    records = []
    for run_spec in RUN_SPECS:
        for example in examples:
            records.append(evaluate_example(model, example, run_spec))

    summaries, intersection_records = build_summary(records)
    routing_analysis_records, routing_aggregate = aggregate_routing_analysis(records)

    write_jsonl(records, EVAL_RECORDS_PATH)
    write_jsonl(summaries, SUMMARY_PATH)
    write_jsonl(routing_analysis_records, ROUTING_ANALYSIS_PATH)
    by_run = defaultdict(list)
    for record in records:
        by_run[record["run_label"]].append(record)

    plot_expert_combination_upset(
        combination_counts=count_combinations(
            tuple(record["activated_experts"]) for record in by_run["top2_active"]
        ),
        path=UPSET_2_PATH,
        title="Activated Expert Combinations for 2 Active Experts",
    )
    plot_expert_combination_upset(
        combination_counts=count_combinations(
            tuple(record["activated_experts"]) for record in by_run["top4_active"]
        ),
        path=UPSET_4_PATH,
        title="Activated Expert Combinations for 4 Active Experts",
    )
    plot_run_comparison_upset(
        intersection_records=intersection_records,
        path=UPSET_PATH,
        title="Activated Expert Intersections Across 2 / 4 Active-Expert Runs",
    )
    save_usage_bar_plot(routing_aggregate["usage"], USAGE_PLOT_PATH)
    save_coactivation_heatmap(routing_aggregate["coactivation_matrix"], COACTIVATION_PLOT_PATH)

    print(f"Wrote {len(records)} evaluation records to {EVAL_RECORDS_PATH}")
    print(f"Wrote {len(summaries)} summary records to {SUMMARY_PATH}")
    print(f"Wrote routing analysis to {ROUTING_ANALYSIS_PATH}")
    print(f"Saved upset plots to {UPSET_2_PATH} and {UPSET_4_PATH}")
    print(f"Saved comparison upset plot to {UPSET_PATH}")
    print(f"Saved routing plots to {USAGE_PLOT_PATH} and {COACTIVATION_PLOT_PATH}")


if __name__ == "__main__":
    main()
