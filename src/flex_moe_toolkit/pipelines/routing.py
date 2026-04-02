from flex_moe_toolkit.core.metrics import load_balance, routing_entropy
from flex_moe_toolkit.prev_analysis.analysis import (
    expert_usage_from_router_logits,
    layer_expert_matrix_from_router_logits,
)
from flex_moe_toolkit.prev_analysis.capture import capture_router_logits
from flex_moe_toolkit.prev_analysis.routing import selected_experts


def analyze_routing(model, adapter, inputs, top_k=5):
    logits = adapter.get_router_logits(model, inputs)
    probs = adapter.get_router_probs(model, inputs)
    selected = selected_experts(logits, k=top_k)

    return {
        "entropy": routing_entropy(logits),
        "load_balance": load_balance(probs),
        "topk_experts": selected,
        "expert_usage": expert_usage_from_router_logits(logits, top_k=top_k),
        "layer_expert_matrix": layer_expert_matrix_from_router_logits(
            logits, top_k=top_k
        ),
    }


def analyze_model_routing(model, inputs, adapter, top_k=5):
    """
    Capture routing information from a model and return aggregate analysis
    statistics that match the common OLMo/FlexOlmo replication workflow.
    """

    router_logits = capture_router_logits(model, inputs, adapter=adapter)
    selected = selected_experts(router_logits, k=top_k)
    router_probs = adapter.router_logits_to_probs(router_logits)

    return {
        "router_logits": router_logits,
        "topk_experts": selected,
        "entropy": routing_entropy(router_logits),
        "load_balance": load_balance(router_probs),
        "expert_usage": expert_usage_from_router_logits(router_logits, top_k=top_k),
        "layer_expert_matrix": layer_expert_matrix_from_router_logits(
            router_logits, top_k=top_k
        ),
    }
