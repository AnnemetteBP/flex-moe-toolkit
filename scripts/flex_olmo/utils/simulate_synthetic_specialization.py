from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch


GROUPS = ("math", "code", "text")

GROUP_PROFILE_SPECS = {
    "math": (
        {0: 5.3, 1: 4.7, 2: 4.9, 6: 4.4, 9: 4.6, 8: 1.6, 5: 0.8},
        {0: 5.0, 1: 4.6, 2: 5.1, 6: 4.2, 9: 4.4, 8: 1.4, 3: 0.6},
        {0: 5.1, 1: 4.8, 2: 4.8, 6: 4.5, 9: 4.5, 7: 1.1, 4: 0.5},
    ),
    "code": (
        {1: 5.0, 2: 4.5, 3: 5.2, 4: 4.8, 8: 4.7, 0: 1.1, 9: 0.9},
        {1: 4.8, 2: 4.7, 3: 5.0, 4: 4.9, 8: 4.6, 6: 1.0, 5: 0.7},
        {1: 5.1, 2: 4.6, 3: 4.9, 4: 5.1, 8: 4.5, 7: 0.8, 0: 0.6},
    ),
    "text": (
        {1: 4.5, 5: 4.8, 7: 4.7, 8: 4.4, 9: 4.3, 3: 1.2, 2: 1.0},
        {1: 4.4, 5: 4.9, 7: 4.6, 8: 4.5, 9: 4.2, 4: 1.1, 0: 0.9},
        {1: 4.6, 5: 4.7, 7: 4.8, 8: 4.3, 9: 4.4, 6: 1.0, 2: 0.8},
    ),
}


@dataclass(frozen=True)
class SyntheticConfig:
    num_batches: int = 8
    batch_size: int = 6
    sequence_length: int = 20
    num_experts: int = 10
    top_k_experts_per_group: int = 5
    noise_std: float = 0.55
    batch_context_std: float = 0.25
    seed: int = 13


def _profile_to_bias_tensor(profile: dict[int, float], num_experts: int) -> torch.Tensor:
    bias = torch.zeros(num_experts, dtype=torch.float32)
    for expert_idx, weight in profile.items():
        bias[expert_idx] = float(weight)
    return bias


def build_group_bias_profiles(config: SyntheticConfig) -> dict[str, torch.Tensor]:
    return {
        group: torch.stack(
            [_profile_to_bias_tensor(profile, config.num_experts) for profile in profiles],
            dim=0,
        )
        for group, profiles in GROUP_PROFILE_SPECS.items()
    }


def generate_group_router_logits(
    group: str,
    config: SyntheticConfig,
    bias_profiles: dict[str, torch.Tensor],
) -> torch.Tensor:
    if group not in bias_profiles:
        raise ValueError(f"Unknown group {group!r}. Expected one of {tuple(bias_profiles)}.")

    generator = torch.Generator().manual_seed(config.seed + GROUPS.index(group))
    profiles = bias_profiles[group]
    profile_indices = torch.arange(config.sequence_length) % profiles.shape[0]
    token_bias = profiles[profile_indices].unsqueeze(0).unsqueeze(0)
    token_bias = token_bias.expand(config.num_batches, config.batch_size, -1, -1)

    batch_context = torch.randn(
        config.num_batches,
        config.batch_size,
        1,
        config.num_experts,
        generator=generator,
    ) * config.batch_context_std
    noise = torch.randn(
        config.num_batches,
        config.batch_size,
        config.sequence_length,
        config.num_experts,
        generator=generator,
    ) * config.noise_std

    return token_bias + batch_context + noise


def generate_all_group_router_logits(
    config: SyntheticConfig,
    bias_profiles: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    resolved_profiles = bias_profiles if bias_profiles is not None else build_group_bias_profiles(config)
    return {
        group: generate_group_router_logits(group, config, resolved_profiles)
        for group in GROUPS
    }


def simulate_top1_routing(router_logits: torch.Tensor) -> torch.Tensor:
    if router_logits.ndim != 4:
        raise ValueError(
            "`router_logits` must have shape `(num_batches, batch_size, sequence_length, num_experts)`, "
            f"received {tuple(router_logits.shape)}."
        )
    return torch.argmax(router_logits, dim=-1)


def aggregate_group_expert_counts(expert_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    if expert_indices.ndim != 3:
        raise ValueError(
            "`expert_indices` must have shape `(num_batches, batch_size, sequence_length)`, "
            f"received {tuple(expert_indices.shape)}."
        )
    return torch.bincount(expert_indices.reshape(-1), minlength=num_experts)


def select_topk_experts(expert_counts: torch.Tensor, top_k: int) -> list[int]:
    if top_k < 1:
        raise ValueError(f"`top_k` must be at least 1, received {top_k}.")
    if top_k > expert_counts.numel():
        raise ValueError(f"`top_k`={top_k} exceeds the number of experts ({expert_counts.numel()}).")
    return torch.topk(expert_counts, k=top_k).indices.tolist()


def build_group_expert_sets(
    expert_counts_by_group: dict[str, torch.Tensor],
    top_k: int,
) -> dict[str, set[int]]:
    return {
        group: set(select_topk_experts(expert_counts, top_k=top_k))
        for group, expert_counts in expert_counts_by_group.items()
    }


def build_expert_membership_table(
    group_expert_sets: dict[str, set[int]],
    num_experts: int,
) -> dict[int, dict[str, bool]]:
    return {
        expert_idx: {
            group: expert_idx in group_expert_sets[group]
            for group in GROUPS
        }
        for expert_idx in range(num_experts)
    }


def summarize_membership_patterns(
    expert_membership: dict[int, dict[str, bool]],
) -> dict[str, list[int]]:
    pattern_summary = {
        "unique_math": [],
        "unique_code": [],
        "unique_text": [],
        "shared_math_code": [],
        "shared_math_text": [],
        "shared_code_text": [],
        "shared_all": [],
    }

    for expert_idx, membership in expert_membership.items():
        math = membership["math"]
        code = membership["code"]
        text = membership["text"]

        if math and not code and not text:
            pattern_summary["unique_math"].append(expert_idx)
        elif code and not math and not text:
            pattern_summary["unique_code"].append(expert_idx)
        elif text and not math and not code:
            pattern_summary["unique_text"].append(expert_idx)
        elif math and code and not text:
            pattern_summary["shared_math_code"].append(expert_idx)
        elif math and text and not code:
            pattern_summary["shared_math_text"].append(expert_idx)
        elif code and text and not math:
            pattern_summary["shared_code_text"].append(expert_idx)
        elif math and code and text:
            pattern_summary["shared_all"].append(expert_idx)

    return pattern_summary


def validate_specialization_constraints(pattern_summary: dict[str, list[int]]) -> None:
    required_patterns = (
        "unique_math",
        "unique_code",
        "unique_text",
        "shared_math_code",
        "shared_math_text",
        "shared_code_text",
        "shared_all",
    )
    missing = [pattern for pattern in required_patterns if not pattern_summary[pattern]]
    if missing:
        raise RuntimeError(f"Synthetic routing did not satisfy the required overlap structure: {missing}")


def run_synthetic_specialization_pipeline(
    config: SyntheticConfig | None = None,
) -> dict[str, object]:
    resolved_config = config if config is not None else SyntheticConfig()
    bias_profiles = build_group_bias_profiles(resolved_config)
    router_logits_by_group = generate_all_group_router_logits(
        config=resolved_config,
        bias_profiles=bias_profiles,
    )
    expert_indices_by_group = {
        group: simulate_top1_routing(router_logits)
        for group, router_logits in router_logits_by_group.items()
    }
    expert_counts_by_group = {
        group: aggregate_group_expert_counts(expert_indices, num_experts=resolved_config.num_experts)
        for group, expert_indices in expert_indices_by_group.items()
    }
    group_expert_sets = build_group_expert_sets(
        expert_counts_by_group=expert_counts_by_group,
        top_k=resolved_config.top_k_experts_per_group,
    )
    expert_membership = build_expert_membership_table(
        group_expert_sets=group_expert_sets,
        num_experts=resolved_config.num_experts,
    )
    pattern_summary = summarize_membership_patterns(expert_membership)
    validate_specialization_constraints(pattern_summary)

    return {
        "config": resolved_config,
        "bias_profiles": bias_profiles,
        "router_logits_by_group": router_logits_by_group,
        "expert_indices_by_group": expert_indices_by_group,
        "expert_counts_by_group": expert_counts_by_group,
        "group_expert_sets": group_expert_sets,
        "expert_membership": expert_membership,
        "pattern_summary": pattern_summary,
    }


def to_serializable(results: dict[str, object]) -> dict[str, object]:
    config: SyntheticConfig = results["config"]
    return {
        "config": {
            "num_batches": config.num_batches,
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "num_experts": config.num_experts,
            "top_k_experts_per_group": config.top_k_experts_per_group,
            "noise_std": config.noise_std,
            "batch_context_std": config.batch_context_std,
            "seed": config.seed,
        },
        "expert_counts_by_group": {
            group: counts.tolist()
            for group, counts in results["expert_counts_by_group"].items()
        },
        "group_expert_sets": {
            group: sorted(experts)
            for group, experts in results["group_expert_sets"].items()
        },
        "expert_membership": results["expert_membership"],
        "pattern_summary": results["pattern_summary"],
    }


def main() -> None:
    results = run_synthetic_specialization_pipeline()
    output_path = Path("outputs/flex_olmo/combined_flex/synthetic_specialization_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(results), handle, indent=2, sort_keys=True)
    print(output_path)


if __name__ == "__main__":
    main()
