from __future__ import annotations

from collections import Counter
from itertools import combinations

import torch


def extract_expert_indices(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Select the most active expert for each token from router logits.

    Args:
        router_logits:
            Tensor with shape `(batch, sequence_length, num_experts)`.

    Returns:
        Tensor with shape `(batch, sequence_length)` containing the argmax expert index
        for each token.
    """

    if not isinstance(router_logits, torch.Tensor):
        raise TypeError(f"`router_logits` must be a torch.Tensor, received {type(router_logits)!r}.")
    if router_logits.ndim != 3:
        raise ValueError(
            "`router_logits` must have shape `(batch, sequence_length, num_experts)`, "
            f"received shape {tuple(router_logits.shape)}."
        )

    return torch.argmax(router_logits, dim=-1)


def compute_expert_sets(expert_indices: torch.Tensor) -> list[set[int]]:
    """
    Convert per-token expert indices into one unique-expert set per sequence.

    Args:
        expert_indices:
            Tensor with shape `(batch, sequence_length)`.

    Returns:
        List of Python sets, one per batch element, containing the unique experts
        selected anywhere in that sequence.
    """

    if not isinstance(expert_indices, torch.Tensor):
        raise TypeError(f"`expert_indices` must be a torch.Tensor, received {type(expert_indices)!r}.")
    if expert_indices.ndim != 2:
        raise ValueError(
            "`expert_indices` must have shape `(batch, sequence_length)`, "
            f"received shape {tuple(expert_indices.shape)}."
        )

    return [
        set(torch.unique(sequence_indices).detach().cpu().tolist())
        for sequence_indices in expert_indices
    ]


def extract_topk_expert_indices(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Select the top-k active experts for each token from router logits.

    Args:
        router_logits:
            Tensor with shape `(batch, sequence_length, num_experts)`.
        top_k:
            Number of experts to keep per token.

    Returns:
        Tensor with shape `(batch, sequence_length, top_k)` containing the top-k expert
        indices for each token, ordered by descending router score.
    """

    if not isinstance(router_logits, torch.Tensor):
        raise TypeError(f"`router_logits` must be a torch.Tensor, received {type(router_logits)!r}.")
    if router_logits.ndim != 3:
        raise ValueError(
            "`router_logits` must have shape `(batch, sequence_length, num_experts)`, "
            f"received shape {tuple(router_logits.shape)}."
        )
    if top_k < 1:
        raise ValueError(f"`top_k` must be at least 1, received {top_k}.")
    if top_k > router_logits.shape[-1]:
        raise ValueError(
            f"`top_k`={top_k} exceeds the number of experts ({router_logits.shape[-1]})."
        )

    return torch.topk(router_logits, k=top_k, dim=-1).indices


def compute_topk_expert_sets(topk_expert_indices: torch.Tensor) -> list[set[int]]:
    """
    Convert per-token top-k expert indices into one unique-expert set per sequence.

    Args:
        topk_expert_indices:
            Tensor with shape `(batch, sequence_length, top_k)`.

    Returns:
        List of Python sets, one per batch element, containing the unique experts
        selected anywhere in that sequence across all top-k assignments.
    """

    if not isinstance(topk_expert_indices, torch.Tensor):
        raise TypeError(
            "`topk_expert_indices` must be a torch.Tensor, "
            f"received {type(topk_expert_indices)!r}."
        )
    if topk_expert_indices.ndim != 3:
        raise ValueError(
            "`topk_expert_indices` must have shape `(batch, sequence_length, top_k)`, "
            f"received shape {tuple(topk_expert_indices.shape)}."
        )

    return [
        set(torch.unique(sequence_indices.reshape(-1)).detach().cpu().tolist())
        for sequence_indices in topk_expert_indices
    ]


def build_upset_data(
    expert_sets: list[set[int]],
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str | int, set[int]]:
    """
    Build a simple label -> expert-set mapping suitable for UpSet-style plotting.

    Args:
        expert_sets:
            List of per-sequence expert sets.
        labels:
            Optional labels aligned with `expert_sets`. If omitted, integer indices are used.

    Returns:
        Dictionary mapping each label (or integer index) to its set of active experts.
    """

    if labels is not None and len(labels) != len(expert_sets):
        raise ValueError(
            f"`labels` must match the number of expert sets: expected {len(expert_sets)}, "
            f"received {len(labels)}."
        )

    keys = labels if labels is not None else list(range(len(expert_sets)))
    return {
        key: set(int(expert_idx) for expert_idx in expert_set)
        for key, expert_set in zip(keys, expert_sets)
    }


def build_layerwise_upset_data(
    topk_expert_indices: torch.Tensor,
    sample_labels: list[str] | tuple[str, ...] | None = None,
    layer_idx: int | None = None,
) -> dict[str, set[int]]:
    """
    Build UpSet-ready data for `(sample, layer)` expert activations.

    Args:
        topk_expert_indices:
            Tensor with shape `(batch, num_layers, sequence_length, top_k)` or
            `(batch, sequence_length, top_k)` when `layer_idx` is provided externally.
        sample_labels:
            Optional labels aligned with the batch dimension.
        layer_idx:
            Optional fixed layer index to use when the input does not include an explicit
            layer dimension.

    Returns:
        Dictionary mapping labels like `sample_0|layer_2` to sets of active experts.
    """

    if not isinstance(topk_expert_indices, torch.Tensor):
        raise TypeError(
            "`topk_expert_indices` must be a torch.Tensor, "
            f"received {type(topk_expert_indices)!r}."
        )
    if topk_expert_indices.ndim not in (3, 4):
        raise ValueError(
            "`topk_expert_indices` must have shape `(batch, sequence_length, top_k)` "
            "or `(batch, num_layers, sequence_length, top_k)`, "
            f"received shape {tuple(topk_expert_indices.shape)}."
        )

    if topk_expert_indices.ndim == 3:
        if layer_idx is None:
            raise ValueError(
                "`layer_idx` must be provided when `topk_expert_indices` has no explicit layer dimension."
            )
        topk_expert_indices = topk_expert_indices.unsqueeze(1)
        layer_indices = [layer_idx]
    else:
        layer_indices = list(range(topk_expert_indices.shape[1]))

    batch_size = topk_expert_indices.shape[0]
    if sample_labels is not None and len(sample_labels) != batch_size:
        raise ValueError(
            f"`sample_labels` must match the batch size: expected {batch_size}, "
            f"received {len(sample_labels)}."
        )

    labels = sample_labels if sample_labels is not None else [f"sample_{idx}" for idx in range(batch_size)]
    upset_data = {}

    for batch_idx, sample_label in enumerate(labels):
        for position, resolved_layer_idx in enumerate(layer_indices):
            active_experts = set(
                torch.unique(topk_expert_indices[batch_idx, position].reshape(-1)).detach().cpu().tolist()
            )
            upset_data[f"{sample_label}|layer_{resolved_layer_idx}"] = active_experts

    return upset_data


def build_layerwise_upset_data_from_router_logits(
    router_logits_by_layer: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    top_k: int,
    sample_labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, set[int]]:
    """
    Build `(sample, layer)` UpSet data directly from per-layer router logits.

    Args:
        router_logits_by_layer:
            Either a tensor with shape `(num_layers, batch, sequence_length, num_experts)`
            or a sequence of tensors with shape `(batch, sequence_length, num_experts)`.
        top_k:
            Number of active experts to keep per token.
        sample_labels:
            Optional labels aligned with the batch dimension.

    Returns:
        Dictionary mapping labels like `sample_0|layer_2` to sets of active experts.
    """

    if isinstance(router_logits_by_layer, torch.Tensor):
        if router_logits_by_layer.ndim != 4:
            raise ValueError(
                "`router_logits_by_layer` must have shape "
                "`(num_layers, batch, sequence_length, num_experts)`, "
                f"received shape {tuple(router_logits_by_layer.shape)}."
            )
        stacked_logits = router_logits_by_layer
    elif isinstance(router_logits_by_layer, (list, tuple)):
        if not router_logits_by_layer:
            raise ValueError("`router_logits_by_layer` must contain at least one layer tensor.")
        if not all(isinstance(layer, torch.Tensor) for layer in router_logits_by_layer):
            raise TypeError("All entries in `router_logits_by_layer` must be torch tensors.")
        stacked_logits = torch.stack(tuple(router_logits_by_layer), dim=0)
    else:
        raise TypeError(
            "`router_logits_by_layer` must be a tensor or a sequence of tensors, "
            f"received {type(router_logits_by_layer)!r}."
        )

    topk_per_layer = torch.stack(
        [extract_topk_expert_indices(layer_logits, top_k=top_k) for layer_logits in stacked_logits],
        dim=1,
    )
    return build_layerwise_upset_data(
        topk_expert_indices=topk_per_layer,
        sample_labels=sample_labels,
    )


def count_token_expert_combinations_by_layer(
    topk_expert_indices_by_layer: torch.Tensor,
) -> list[Counter]:
    """
    Count exact top-k expert combinations for each layer at the token level.

    Args:
        topk_expert_indices_by_layer:
            Tensor with shape `(num_layers, batch, sequence_length, top_k)`.

    Returns:
        List of Counters, one per layer, mapping sorted expert tuples to counts.
    """

    if not isinstance(topk_expert_indices_by_layer, torch.Tensor):
        raise TypeError(
            "`topk_expert_indices_by_layer` must be a torch.Tensor, "
            f"received {type(topk_expert_indices_by_layer)!r}."
        )
    if topk_expert_indices_by_layer.ndim != 4:
        raise ValueError(
            "`topk_expert_indices_by_layer` must have shape "
            "`(num_layers, batch, sequence_length, top_k)`, "
            f"received shape {tuple(topk_expert_indices_by_layer.shape)}."
        )

    counters = []
    for layer_indices in topk_expert_indices_by_layer:
        flat_combinations = layer_indices.reshape(-1, layer_indices.shape[-1]).detach().cpu().tolist()
        counters.append(
            Counter(tuple(sorted(int(expert_idx) for expert_idx in combo)) for combo in flat_combinations)
        )

    return counters


def count_token_expert_combinations_by_layer_from_router_logits(
    router_logits_by_layer: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    top_k: int,
) -> list[Counter]:
    """
    Count exact top-k expert combinations for each layer directly from router logits.
    """

    if isinstance(router_logits_by_layer, torch.Tensor):
        if router_logits_by_layer.ndim != 4:
            raise ValueError(
                "`router_logits_by_layer` must have shape "
                "`(num_layers, batch, sequence_length, num_experts)`, "
                f"received shape {tuple(router_logits_by_layer.shape)}."
            )
        stacked_logits = router_logits_by_layer
    elif isinstance(router_logits_by_layer, (list, tuple)):
        if not router_logits_by_layer:
            raise ValueError("`router_logits_by_layer` must contain at least one layer tensor.")
        if not all(isinstance(layer, torch.Tensor) for layer in router_logits_by_layer):
            raise TypeError("All entries in `router_logits_by_layer` must be torch tensors.")
        stacked_logits = torch.stack(tuple(router_logits_by_layer), dim=0)
    else:
        raise TypeError(
            "`router_logits_by_layer` must be a tensor or a sequence of tensors, "
            f"received {type(router_logits_by_layer)!r}."
        )

    topk_by_layer = torch.stack(
        [extract_topk_expert_indices(layer_logits, top_k=top_k) for layer_logits in stacked_logits],
        dim=0,
    )
    return count_token_expert_combinations_by_layer(topk_by_layer)


def flatten_topk_experts(topk_experts):
    flattened = []

    for layer in topk_experts:
        layer_experts = set()
        for batch in layer:
            batch_list = batch.tolist() if hasattr(batch, "tolist") else batch
            if not isinstance(batch_list, list):
                batch_list = [batch_list]
            for token in batch_list:
                if isinstance(token, list):
                    expert_values = token
                else:
                    expert_values = [token]
                for expert_idx in expert_values:
                    layer_experts.add(int(expert_idx))
        flattened.append(tuple(sorted(layer_experts)))

    return flattened


def activated_expert_combination(topk_experts):
    active = set()
    for layer_combo in flatten_topk_experts(topk_experts):
        active.update(layer_combo)
    return tuple(sorted(active))


def count_combinations(combinations):
    return Counter(tuple(combo) for combo in combinations)


def set_intersection_over_union(left, right):
    left_set = set(int(expert) for expert in left)
    right_set = set(int(expert) for expert in right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def layer_iou_summary(layer_activated_experts):
    normalized_layers = [tuple(sorted(int(expert) for expert in layer)) for layer in layer_activated_experts]

    if not normalized_layers:
        return {
            "layer_intersection_experts": (),
            "layer_union_experts": (),
            "pairwise_layer_iou": [],
            "mean_layer_iou": 0.0,
        }

    if len(normalized_layers) == 1:
        only_layer = normalized_layers[0]
        return {
            "layer_intersection_experts": only_layer,
            "layer_union_experts": only_layer,
            "pairwise_layer_iou": [],
            "mean_layer_iou": 1.0,
        }

    layer_sets = [set(layer) for layer in normalized_layers]
    pairwise_scores = [
        {
            "layer_left": left_idx,
            "layer_right": right_idx,
            "iou": set_intersection_over_union(normalized_layers[left_idx], normalized_layers[right_idx]),
        }
        for left_idx, right_idx in combinations(range(len(normalized_layers)), 2)
    ]

    intersection = tuple(sorted(set.intersection(*layer_sets)))
    union = tuple(sorted(set.union(*layer_sets)))

    return {
        "layer_intersection_experts": intersection,
        "layer_union_experts": union,
        "pairwise_layer_iou": pairwise_scores,
        "mean_layer_iou": sum(item["iou"] for item in pairwise_scores) / len(pairwise_scores),
    }
