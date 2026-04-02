from __future__ import annotations

from contextlib import contextmanager

import torch

from flex_moe_toolkit.adapters.flex_olmo import FlexOlmoAdapter, iter_flex_olmo_layers
from flex_moe_toolkit.pipelines.routing import analyze_model_routing


def _zero_moe_output(_module, args, _kwargs=None):
    hidden_states = args[0]
    return torch.zeros_like(hidden_states)


def _masked_router_forward(module, hidden_states, allowed_experts):
    hidden_states = hidden_states.reshape(-1, module.hidden_dim)
    router_logits = torch.nn.functional.linear(hidden_states, module.weight)

    mask = torch.full_like(router_logits, float("-inf"))
    mask[:, allowed_experts] = 0
    masked_logits = router_logits + mask
    router_probs = torch.softmax(masked_logits, dim=-1, dtype=torch.float)
    effective_top_k = min(module.top_k, len(allowed_experts))
    top_values, top_indices = torch.topk(router_probs, effective_top_k, dim=-1)

    if module.norm_topk_prob:
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)

    return router_probs, top_values.to(router_probs.dtype), top_indices


@contextmanager
def backbone_only_mode(model):
    """
    Zero out MoE block outputs while keeping the residual stream intact.
    """

    layers = list(iter_flex_olmo_layers(model))
    originals = [layer.mlp.experts.forward for layer in layers]

    try:
        for layer in layers:
            layer.mlp.experts.forward = (
                lambda hidden_states, top_k_index, top_k_weights: torch.zeros_like(hidden_states)
            )
        yield model
    finally:
        for layer, original in zip(layers, originals):
            layer.mlp.experts.forward = original


@contextmanager
def restricted_expert_mode(model, allowed_experts):
    """
    Restrict routing to a subset of experts for per-expert or public-vs-domain runs.
    """

    allowed_experts = tuple(sorted(set(int(expert_idx) for expert_idx in allowed_experts)))
    if not allowed_experts:
        raise ValueError("`allowed_experts` must contain at least one expert index.")

    layers = list(iter_flex_olmo_layers(model))
    originals = [layer.mlp.gate.forward for layer in layers]

    try:
        for layer in layers:
            gate = layer.mlp.gate
            gate.forward = lambda hidden_states, _gate=gate: _masked_router_forward(
                _gate, hidden_states, allowed_experts
            )
        yield model
    finally:
        for layer, original in zip(layers, originals):
            layer.mlp.gate.forward = original


def analyze_flex_olmo_routing(model, inputs, top_k=None):
    adapter = FlexOlmoAdapter()
    if top_k is None:
        top_k = getattr(getattr(model, "config", None), "num_experts_per_tok", 5)
    return analyze_model_routing(model=model, inputs=inputs, adapter=adapter, top_k=top_k)


def analyze_flex_olmo_modes(model, inputs, public_expert_idx=0):
    """
    Compare full routing, backbone-only routing, and public-expert-only routing.

    Assumption:
    `public_expert_idx=0` matches the common FlexOlmo convention where the public
    expert is expert 0. Override if your checkpoint uses a different ordering.
    """

    results = {
        "full": analyze_flex_olmo_routing(model, inputs),
    }

    with backbone_only_mode(model):
        results["backbone_only"] = analyze_flex_olmo_routing(model, inputs)

    public_experts = [public_expert_idx]
    with restricted_expert_mode(model, allowed_experts=public_experts):
        results["public_only"] = analyze_flex_olmo_routing(
            model,
            inputs,
            top_k=min(len(public_experts), model.config.num_experts_per_tok),
        )

    return results
