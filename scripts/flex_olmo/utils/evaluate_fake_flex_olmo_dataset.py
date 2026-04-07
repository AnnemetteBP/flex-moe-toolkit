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

from fake_test_models.fake_flex_olmo import FakeFlexOlmoModel
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
UPSET_PATH = OUTPUT_DIR / "fake_eval_expert_upset.png"
UPSET_2_PATH = OUTPUT_DIR / "fake_eval_expert_upset_top2_active.png"
UPSET_4_PATH = OUTPUT_DIR / "fake_eval_expert_upset_top4_active.png"

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


def main():
    examples = load_jsonl(DATASET_PATH)
    model = FakeFlexOlmoModel(num_experts=7)

    records = []
    for run_spec in RUN_SPECS:
        for example in examples:
            records.append(evaluate_example(model, example, run_spec))

    summaries, intersection_records = build_summary(records)

    write_jsonl(records, EVAL_RECORDS_PATH)
    write_jsonl(summaries, SUMMARY_PATH)
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

    print(f"Wrote {len(records)} evaluation records to {EVAL_RECORDS_PATH}")
    print(f"Wrote {len(summaries)} summary records to {SUMMARY_PATH}")
    print(f"Saved upset plots to {UPSET_2_PATH} and {UPSET_4_PATH}")
    print(f"Saved comparison upset plot to {UPSET_PATH}")


if __name__ == "__main__":
    main()
