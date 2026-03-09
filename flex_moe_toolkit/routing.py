import torch
import pandas as pd



def selected_experts(router_logits, k):
    """
    Return top-k experts per token per layer.
    Handles router_logits returned as tuple(list) of tensors.
    """

    selected = []

    for layer_logits in router_logits:

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

        for e in flat:

            e = int(e)

            counts[e] = counts.get(e, 0) + 1

    return pd.Series(counts).sort_index()