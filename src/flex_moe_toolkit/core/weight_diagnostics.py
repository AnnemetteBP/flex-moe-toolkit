from __future__ import annotations

import torch
import torch.nn.functional as F


def flatten_expert_weights(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim < 2:
        raise ValueError(f"`weights` must have at least 2 dimensions, received {tuple(weights.shape)}.")
    return weights.flatten(start_dim=1).float()


def pairwise_cosine_similarity(weights: torch.Tensor) -> torch.Tensor:
    flat = flatten_expert_weights(weights)
    normalized = F.normalize(flat, p=2, dim=-1)
    return normalized @ normalized.transpose(0, 1)


def vector_norms(weights: torch.Tensor) -> torch.Tensor:
    flat = flatten_expert_weights(weights)
    return torch.linalg.vector_norm(flat, ord=2, dim=-1)


def rowwise_mean_abs(weights: torch.Tensor) -> torch.Tensor:
    flat = flatten_expert_weights(weights)
    return flat.abs().mean(dim=-1)


def public_distance_profile(weights: torch.Tensor, public_expert_idx: int = 0) -> torch.Tensor:
    flat = flatten_expert_weights(weights)
    if not 0 <= public_expert_idx < flat.shape[0]:
        raise ValueError(
            f"`public_expert_idx` must be in [0, {flat.shape[0] - 1}], received {public_expert_idx}."
        )
    public = flat[public_expert_idx : public_expert_idx + 1]
    return torch.linalg.vector_norm(flat - public, ord=2, dim=-1)


def summarize_similarity_matrix(similarity: torch.Tensor) -> dict[str, float]:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(
            "`similarity` must be a square matrix, "
            f"received shape {tuple(similarity.shape)}."
        )

    num_experts = similarity.shape[0]
    if num_experts == 1:
        return {
            "mean_offdiag_similarity": 1.0,
            "max_offdiag_similarity": 1.0,
            "min_offdiag_similarity": 1.0,
        }

    mask = ~torch.eye(num_experts, dtype=torch.bool, device=similarity.device)
    offdiag = similarity[mask]
    return {
        "mean_offdiag_similarity": float(offdiag.mean().item()),
        "max_offdiag_similarity": float(offdiag.max().item()),
        "min_offdiag_similarity": float(offdiag.min().item()),
    }
