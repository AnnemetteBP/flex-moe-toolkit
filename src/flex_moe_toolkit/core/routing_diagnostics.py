from __future__ import annotations

import torch


def _normalize_k_values(top_k) -> list[int]:
    if top_k is None:
        raise ValueError("`top_k` must be provided when computing router saturation from routing outputs.")

    if isinstance(top_k, int):
        k_values = [top_k]
    else:
        k_values = [int(k) for k in top_k]

    if not k_values:
        raise ValueError("`top_k` must contain at least one value.")
    if any(k < 1 for k in k_values):
        raise ValueError(f"All `top_k` values must be at least 1, received {k_values}.")

    return sorted(set(k_values))


def _normalize_routing_tensor(routing, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(routing)

    if tensor.ndim not in (2, 3):
        raise ValueError(
            f"`{name}` must have shape `(N, M)` or `(B, T, M)`, "
            f"received {tuple(tensor.shape)}."
        )

    if tensor.ndim == 3:
        tensor = tensor.reshape(-1, tensor.shape[-1])

    if tensor.shape[-1] < 1:
        raise ValueError(f"`{name}` must contain at least one expert dimension.")

    return tensor


def _compute_router_saturation_for_k(
    topk_t: torch.Tensor,
    topk_T: torch.Tensor,
    k: int,
) -> float:
    if topk_t.shape != topk_T.shape:
        raise ValueError(
            "`topk_t` and `topk_T` must have identical shapes, "
            f"received {tuple(topk_t.shape)} and {tuple(topk_T.shape)}."
        )

    scores = []

    for token_experts_t, token_experts_T in zip(topk_t, topk_T):
        experts_t = set(token_experts_t.tolist())
        experts_T = set(token_experts_T.tolist())
        overlap_i = len(experts_t.intersection(experts_T))
        score_i = overlap_i / k
        scores.append(score_i)

    if not scores:
        raise ValueError("Router saturation requires at least one token.")

    saturation_k = sum(scores) / len(scores)

    if torch.equal(topk_t, topk_T):
        saturation_k = 1.0

    assert 0.0 <= saturation_k <= 1.0
    return float(saturation_k)


def compute_router_saturation_random_baseline(num_experts: int, top_k) -> dict[int, float]:
    k_values = _normalize_k_values(top_k)
    if num_experts < 1:
        raise ValueError(f"`num_experts` must be at least 1, received {num_experts}.")
    if any(k > num_experts for k in k_values):
        raise ValueError(
            f"All `top_k` values must be <= `num_experts`={num_experts}, received {k_values}."
        )
    return {k: k / num_experts for k in k_values}


def compute_router_saturation(
    routing_t,
    routing_T,
    top_k=None,
) -> dict[int, float]:
    """
    Compute router saturation exactly as:

    RouterSaturation(t; k) = (1 / N) * sum_i ( |E_i(t) ∩ E_i(T)| / k )

    where E_i(t) and E_i(T) are the top-k expert ids recomputed from routing
    scores at checkpoint `t` and final checkpoint `T` for token `i`.
    """

    tensor_t = _normalize_routing_tensor(routing_t, "routing_t")
    tensor_T = _normalize_routing_tensor(routing_T, "routing_T")

    if tensor_t.shape != tensor_T.shape:
        raise ValueError(
            "`routing_t` and `routing_T` must have identical shapes, "
            f"received {tuple(tensor_t.shape)} and {tuple(tensor_T.shape)}."
        )

    if top_k is None:
        k_values = [tensor_t.shape[-1]]
    else:
        k_values = _normalize_k_values(top_k)

    if not tensor_t.is_floating_point() or not tensor_T.is_floating_point():
        raise ValueError(
            "`compute_router_saturation` requires routing scores/logits so top-k can be "
            "explicitly recomputed for each requested `k` at both checkpoints. "
            "Do not pass precomputed expert-index tensors here."
        )

    if any(k > tensor_t.shape[-1] for k in k_values):
        raise ValueError(
            f"All requested `top_k` values must be <= number of experts={tensor_t.shape[-1]}, "
            f"received {k_values}."
        )

    results = {}
    for k in k_values:
        # Recompute top-k experts independently for this exact `k` at BOTH checkpoints.
        topk_t = torch.topk(tensor_t, k=k, dim=-1).indices
        topk_T = torch.topk(tensor_T, k=k, dim=-1).indices
        results[k] = _compute_router_saturation_for_k(topk_t, topk_T, k)

    return results


def compute_router_saturation_from_logits(
    router_logits_t: torch.Tensor,
    router_logits_T: torch.Tensor,
    top_k,
) -> dict[int, float]:
    """
    Convenience wrapper that computes router saturation directly from router logits.
    """

    if router_logits_t.ndim != 3 or router_logits_T.ndim != 3:
        raise ValueError(
            "`router_logits_t` and `router_logits_T` must have shape `(B, T, E)`, "
            f"received {tuple(router_logits_t.shape)} and {tuple(router_logits_T.shape)}."
        )

    if router_logits_t.shape != router_logits_T.shape:
        raise ValueError(
            "`router_logits_t` and `router_logits_T` must have identical shapes, "
            f"received {tuple(router_logits_t.shape)} and {tuple(router_logits_T.shape)}."
        )

    return compute_router_saturation(router_logits_t, router_logits_T, top_k=top_k)


def compute_expert_usage(router_logits: torch.Tensor) -> torch.Tensor:
    if router_logits.ndim != 3:
        raise ValueError(
            "`router_logits` must have shape `(B, T, E)`, "
            f"received {tuple(router_logits.shape)}."
        )

    num_experts = router_logits.shape[-1]
    top1_indices = torch.argmax(router_logits, dim=-1)
    counts = torch.bincount(top1_indices.reshape(-1), minlength=num_experts).to(router_logits.dtype)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Top-1 routing produced no expert assignments.")
    return counts / total


def compute_entropy(router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if router_logits.ndim != 3:
        raise ValueError(
            "`router_logits` must have shape `(B, T, E)`, "
            f"received {tuple(router_logits.shape)}."
        )

    probabilities = torch.softmax(router_logits, dim=-1)
    entropy_per_token = -(probabilities * torch.log(probabilities + 1e-9)).sum(dim=-1)
    entropy_mean = entropy_per_token.mean()
    return entropy_mean, entropy_per_token


def compute_coactivation(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if router_logits.ndim != 3:
        raise ValueError(
            "`router_logits` must have shape `(B, T, E)`, "
            f"received {tuple(router_logits.shape)}."
        )

    num_experts = router_logits.shape[-1]
    if top_k < 1 or top_k > num_experts:
        raise ValueError(
            f"`top_k` must be in [1, {num_experts}], received {top_k}."
        )

    probabilities = torch.softmax(router_logits, dim=-1)
    topk_indices = torch.topk(probabilities, k=top_k, dim=-1).indices

    matrix = torch.zeros(
        (num_experts, num_experts),
        dtype=router_logits.dtype,
        device=router_logits.device,
    )

    for token_experts in topk_indices.reshape(-1, top_k):
        matrix[token_experts.unsqueeze(1), token_experts.unsqueeze(0)] += 1

    return matrix


def compute_offdiagonal_ratio(M: torch.Tensor) -> float:
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(
            "`M` must have shape `(E, E)`, "
            f"received {tuple(M.shape)}."
        )

    total = M.sum()
    if total <= 0:
        return 0.0

    diagonal = torch.diagonal(M).sum()
    offdiagonal = total - diagonal
    return float((offdiagonal / total).item())


def compute_all_metrics(router_logits: torch.Tensor, top_k: int) -> dict:
    usage = compute_expert_usage(router_logits)
    entropy_mean, entropy_per_token = compute_entropy(router_logits)
    coactivation_matrix = compute_coactivation(router_logits, top_k=top_k)
    offdiag_ratio = compute_offdiagonal_ratio(coactivation_matrix)

    return {
        "usage": usage,
        "entropy_mean": entropy_mean,
        "coactivation_matrix": coactivation_matrix,
        "offdiag_ratio": offdiag_ratio,
    }
