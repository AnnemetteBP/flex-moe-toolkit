import pandas as pd
import torch


def ensure_layer_tensors(router_logits):
    """
    Normalize router logits/probabilities to a tuple of per-layer tensors.
    """

    if isinstance(router_logits, torch.Tensor):
        return (router_logits,)

    if isinstance(router_logits, (list, tuple)):
        normalized = tuple(router_logits)
        if not all(isinstance(layer, torch.Tensor) for layer in normalized):
            raise TypeError("All router outputs must be torch tensors.")
        return normalized

    raise TypeError(
        "Router outputs must be a tensor or a sequence of tensors, "
        f"received {type(router_logits)!r}."
    )


def selected_experts(router_logits, k):
    """
    Return top-k experts per token per layer.
    Handles router_logits returned as tuple(list) of tensors.
    """

    selected = []

    for layer_logits in ensure_layer_tensors(router_logits):
        if layer_logits.shape[-1] < k:
            raise ValueError(
                f"Requested top-{k} experts, but only {layer_logits.shape[-1]} exist."
            )

        topk = torch.topk(layer_logits, k=k, dim=-1)
        selected.append(topk.indices)

    return selected


def expert_load(selected):
    """
    Count how often each expert is used across all layers.
    """

    counts = {}

    for layer in selected:
        flat = layer.flatten()

        for expert_idx in flat:
            expert_idx = int(expert_idx)
            counts[expert_idx] = counts.get(expert_idx, 0) + 1

    return pd.Series(counts).sort_index()
