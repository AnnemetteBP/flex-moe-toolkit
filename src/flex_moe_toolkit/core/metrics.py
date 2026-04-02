import torch

from flex_moe_toolkit.prev_analysis.routing import ensure_layer_tensors


def routing_entropy(router_logits):
    entropies = []

    for layer_logits in ensure_layer_tensors(router_logits):
        probs = torch.softmax(layer_logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropies.append(entropy.mean())

    if not entropies:
        raise ValueError("No router logits were provided.")

    return torch.stack(entropies).mean().item()


def load_balance(router_probs):
    balances = []

    for layer_probs in ensure_layer_tensors(router_probs):
        if layer_probs.ndim == 3:
            expert_usage = layer_probs.sum(dim=(0, 1))
        elif layer_probs.ndim == 2:
            expert_usage = layer_probs.sum(dim=0)
        else:
            raise ValueError(
                f"Unexpected router probability shape: {tuple(layer_probs.shape)}"
            )

        expert_usage = expert_usage / expert_usage.sum()
        uniform = torch.ones_like(expert_usage) / expert_usage.numel()
        balances.append(torch.abs(expert_usage - uniform).mean())

    if not balances:
        raise ValueError("No router probabilities were provided.")

    return torch.stack(balances).mean().item()
