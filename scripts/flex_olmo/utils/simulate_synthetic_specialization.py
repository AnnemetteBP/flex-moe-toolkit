from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd
import torch


NUM_EXPERTS = 10
NUM_BATCHES = 8
BATCH_SIZE = 6
SEQ_LEN = 20
TOP_K = 5

SEED = 13
NOISE_STD = 0.75
PREFERRED_EXPERT_BIAS = 3.0

MATH_EXPERTS = [0, 1, 2, 6, 9]
CODE_EXPERTS = [1, 2, 3, 4, 8]
TEXT_EXPERTS = [1, 5, 7, 8, 9]


def generate_logits(preferred_experts: List[int]) -> torch.Tensor:
    logits = torch.randn(NUM_BATCHES, BATCH_SIZE, SEQ_LEN, NUM_EXPERTS) * NOISE_STD
    logits[..., preferred_experts] += PREFERRED_EXPERT_BIAS
    return logits


def route_experts(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def count_expert_usage(indices: torch.Tensor) -> torch.Tensor:
    return torch.bincount(indices.reshape(-1), minlength=NUM_EXPERTS)


def select_top_k(counts: torch.Tensor, k: int) -> Set[int]:
    return set(torch.topk(counts, k=k).indices.tolist())


def build_membership_matrix(dataset_sets: Dict[str, Set[int]]) -> pd.DataFrame:
    data = {
        dataset_name: [expert_idx in dataset_set for expert_idx in range(NUM_EXPERTS)]
        for dataset_name, dataset_set in dataset_sets.items()
    }
    return pd.DataFrame(data, index=range(NUM_EXPERTS), dtype=bool)


def compute_dataset_sets() -> Dict[str, Set[int]]:
    torch.manual_seed(SEED)

    math_logits = generate_logits(MATH_EXPERTS)
    code_logits = generate_logits(CODE_EXPERTS)
    text_logits = generate_logits(TEXT_EXPERTS)

    math_indices = route_experts(math_logits)
    code_indices = route_experts(code_logits)
    text_indices = route_experts(text_logits)

    math_counts = count_expert_usage(math_indices)
    code_counts = count_expert_usage(code_indices)
    text_counts = count_expert_usage(text_indices)

    return {
        "math": select_top_k(math_counts, TOP_K),
        "code": select_top_k(code_counts, TOP_K),
        "text": select_top_k(text_counts, TOP_K),
    }


def validate_dataset_sets(dataset_sets: Dict[str, Set[int]]) -> None:
    math_set = dataset_sets["math"]
    code_set = dataset_sets["code"]
    text_set = dataset_sets["text"]

    shared_all = math_set & code_set & text_set
    shared_math_code = (math_set & code_set) - text_set
    shared_math_text = (math_set & text_set) - code_set
    shared_code_text = (code_set & text_set) - math_set

    unique_math = math_set - code_set - text_set
    unique_code = code_set - math_set - text_set
    unique_text = text_set - math_set - code_set

    assert unique_math, "Expected at least one expert unique to math."
    assert unique_code, "Expected at least one expert unique to code."
    assert unique_text, "Expected at least one expert unique to text."
    assert shared_all, "Expected at least one expert shared across all groups."
    assert shared_math_code, "Expected at least one expert shared between math and code only."
    assert shared_math_text, "Expected at least one expert shared between math and text only."
    assert shared_code_text, "Expected at least one expert shared between code and text only."


def run_synthetic_specialization() -> Dict[str, object]:
    dataset_sets = compute_dataset_sets()
    validate_dataset_sets(dataset_sets)
    membership_matrix = build_membership_matrix(dataset_sets)
    return {
        "dataset_sets": dataset_sets,
        "membership_matrix": membership_matrix,
    }
