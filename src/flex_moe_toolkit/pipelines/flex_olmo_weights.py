from __future__ import annotations

from flex_moe_toolkit.adapters.flex_olmo import iter_flex_olmo_layers
from flex_moe_toolkit.core.weight_diagnostics import (
    pairwise_cosine_similarity,
    public_distance_profile,
    rowwise_mean_abs,
    summarize_similarity_matrix,
    vector_norms,
)


def _base_model(model):
    return getattr(model, "model", model)


def _expert_weight_tensors(layer):
    experts = layer.mlp.experts
    tensors = {}

    if hasattr(experts, "gate_up_proj"):
        tensors["gate_up_proj"] = experts.gate_up_proj.detach().cpu()
    if hasattr(experts, "down_proj"):
        tensors["down_proj"] = experts.down_proj.detach().cpu()

    if not tensors:
        raise ValueError("Could not find expert weight tensors on `layer.mlp.experts`.")

    return tensors


def _router_weight_tensor(layer):
    gate = layer.mlp.gate
    if not hasattr(gate, "weight"):
        raise ValueError("Could not find router weight tensor on `layer.mlp.gate`.")
    return gate.weight.detach().cpu().float()


def analyze_flex_olmo_weights(model, public_expert_idx: int = 0) -> dict[str, object]:
    base_model = _base_model(model)
    layer_records = []

    router_norms_by_layer = []
    gate_similarity_by_layer = []
    down_similarity_by_layer = []

    for layer_idx, layer in enumerate(iter_flex_olmo_layers(base_model)):
        router_weight = _router_weight_tensor(layer)
        router_norms = vector_norms(router_weight)
        router_mean_abs = rowwise_mean_abs(router_weight)
        router_similarity = pairwise_cosine_similarity(router_weight)

        expert_tensors = _expert_weight_tensors(layer)
        gate_up = expert_tensors["gate_up_proj"]
        gate_similarity = pairwise_cosine_similarity(gate_up)
        gate_norms = vector_norms(gate_up)
        gate_mean_abs = rowwise_mean_abs(gate_up)
        gate_public_distance = public_distance_profile(gate_up, public_expert_idx=public_expert_idx)

        layer_record = {
            "layer_idx": layer_idx,
            "router_weight_shape": tuple(router_weight.shape),
            "router_weight_norms": router_norms,
            "router_weight_mean_abs": router_mean_abs,
            "router_similarity": router_similarity,
            "router_similarity_summary": summarize_similarity_matrix(router_similarity),
            "gate_up_proj_shape": tuple(gate_up.shape),
            "gate_up_proj_norms": gate_norms,
            "gate_up_proj_mean_abs": gate_mean_abs,
            "gate_up_proj_similarity": gate_similarity,
            "gate_up_proj_similarity_summary": summarize_similarity_matrix(gate_similarity),
            "gate_up_proj_public_distance": gate_public_distance,
        }

        if "down_proj" in expert_tensors:
            down_proj = expert_tensors["down_proj"]
            down_similarity = pairwise_cosine_similarity(down_proj)
            layer_record["down_proj_shape"] = tuple(down_proj.shape)
            layer_record["down_proj_norms"] = vector_norms(down_proj)
            layer_record["down_proj_mean_abs"] = rowwise_mean_abs(down_proj)
            layer_record["down_proj_similarity"] = down_similarity
            layer_record["down_proj_similarity_summary"] = summarize_similarity_matrix(down_similarity)
            layer_record["down_proj_public_distance"] = public_distance_profile(
                down_proj,
                public_expert_idx=public_expert_idx,
            )
            down_similarity_by_layer.append(layer_record["down_proj_similarity_summary"]["mean_offdiag_similarity"])

        router_norms_by_layer.append(float(router_norms.mean().item()))
        gate_similarity_by_layer.append(layer_record["gate_up_proj_similarity_summary"]["mean_offdiag_similarity"])
        layer_records.append(layer_record)

    return {
        "record_type": "weight_analysis",
        "num_layers": len(layer_records),
        "num_experts": int(base_model.config.num_experts),
        "public_expert_idx": public_expert_idx,
        "layer_weight_analysis": layer_records,
        "summary": {
            "mean_router_weight_norm": sum(router_norms_by_layer) / len(router_norms_by_layer),
            "mean_gate_up_proj_offdiag_similarity": (
                sum(gate_similarity_by_layer) / len(gate_similarity_by_layer)
            ),
            "mean_down_proj_offdiag_similarity": (
                sum(down_similarity_by_layer) / len(down_similarity_by_layer)
                if down_similarity_by_layer
                else None
            ),
        },
    }
