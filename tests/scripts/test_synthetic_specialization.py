from __future__ import annotations

import math

import torch

from scripts.flex_olmo.utils.simulate_synthetic_specialization import (
    CODE_EXPERTS,
    MATH_EXPERTS,
    TEXT_EXPERTS,
    build_membership_matrix,
    generate_logits,
    run_synthetic_specialization,
)
from flex_moe_toolkit.core.routing_diagnostics import compute_all_metrics


def _validate_metric_bundle(metrics: dict) -> None:
    usage = metrics["usage"]
    entropy_mean = metrics["entropy_mean"]
    coactivation_matrix = metrics["coactivation_matrix"]
    offdiag_ratio = metrics["offdiag_ratio"]

    if not torch.isclose(usage.sum(), torch.tensor(1.0, dtype=usage.dtype), atol=1e-6):
        raise AssertionError("usage does not sum to 1")
    if not torch.isfinite(entropy_mean):
        raise AssertionError("entropy_mean is not finite")
    if coactivation_matrix.ndim != 2 or coactivation_matrix.shape[0] != coactivation_matrix.shape[1]:
        raise AssertionError("coactivation_matrix is not square")
    if not (0.0 <= offdiag_ratio <= 1.0):
        raise AssertionError("offdiag_ratio is outside [0, 1]")


def run_test() -> dict[str, object]:
    result = run_synthetic_specialization()

    dataset_sets = result["dataset_sets"]
    membership_matrix = result["membership_matrix"]

    expected_keys = {"math", "code", "text"}
    if set(dataset_sets.keys()) != expected_keys:
        raise AssertionError("dataset_sets does not contain the expected datasets")
    if membership_matrix.shape[1] != 3:
        raise AssertionError("membership_matrix does not have 3 dataset columns")
    if membership_matrix.shape[0] <= 0:
        raise AssertionError("membership_matrix has no expert rows")

    rebuilt_membership = build_membership_matrix(dataset_sets)
    if rebuilt_membership.shape != membership_matrix.shape:
        raise AssertionError("rebuilt membership matrix shape does not match")

    math_logits = generate_logits(MATH_EXPERTS).flatten(0, 1)
    code_logits = generate_logits(CODE_EXPERTS).flatten(0, 1)
    text_logits = generate_logits(TEXT_EXPERTS).flatten(0, 1)

    math_metrics = compute_all_metrics(math_logits, top_k=5)
    code_metrics = compute_all_metrics(code_logits, top_k=5)
    text_metrics = compute_all_metrics(text_logits, top_k=5)

    _validate_metric_bundle(math_metrics)
    _validate_metric_bundle(code_metrics)
    _validate_metric_bundle(text_metrics)

    return {
        "upset_valid": True,
        "num_experts": membership_matrix.shape[0],
        "num_datasets": membership_matrix.shape[1],
        "metrics_valid": True,
    }


if __name__ == "__main__":
    run_test()
