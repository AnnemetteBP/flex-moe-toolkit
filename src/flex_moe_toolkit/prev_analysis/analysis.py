import numpy as np
import pandas as pd
import torch

from flex_moe_toolkit.prev_analysis.routing import ensure_layer_tensors, selected_experts


def compute_expert_usage(log_path):
    df = pd.read_json(log_path, lines=True)

    counts = {}

    for probs in df["probs"]:
        experts = np.argsort(probs)[-5:]

        for expert_idx in experts:
            counts[expert_idx] = counts.get(expert_idx, 0) + 1

    return pd.Series(counts).sort_index()


def layer_expert_matrix(log_path):
    df = pd.read_json(log_path, lines=True)

    num_experts = len(df.iloc[0]["probs"])
    num_layers = df["layer"].max() + 1
    matrix = np.zeros((num_layers, num_experts))

    for _, row in df.iterrows():
        probs = row["probs"]
        topk = np.argsort(probs)[-5:]

        for expert_idx in topk:
            matrix[row["layer"], expert_idx] += 1

    return matrix


def expert_usage_from_router_logits(router_logits, top_k=5):
    counts = {}

    for layer in selected_experts(router_logits, k=top_k):
        for expert_idx in layer.reshape(-1).tolist():
            counts[expert_idx] = counts.get(expert_idx, 0) + 1

    return pd.Series(counts, dtype="int64").sort_index()


def layer_expert_matrix_from_router_logits(router_logits, top_k=5):
    layers = ensure_layer_tensors(router_logits)

    if not layers:
        raise ValueError("No router logits were provided.")

    num_layers = len(layers)
    num_experts = layers[0].shape[-1]
    matrix = torch.zeros((num_layers, num_experts), dtype=torch.int64)

    for layer_idx, chosen in enumerate(selected_experts(layers, k=top_k)):
        flat = chosen.reshape(-1)
        counts = torch.bincount(flat, minlength=num_experts)
        matrix[layer_idx] = counts

    return matrix.numpy()
