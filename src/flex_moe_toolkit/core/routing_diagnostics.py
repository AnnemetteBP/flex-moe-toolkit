from __future__ import annotations

import torch


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
