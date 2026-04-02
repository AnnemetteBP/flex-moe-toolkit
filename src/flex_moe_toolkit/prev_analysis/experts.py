import torch
import torch.nn.functional as F



def expert_similarity_matrix(model, layer_idx):
    """
    Compute cosine similarity between experts in a given layer.
    """

    base_model = getattr(model, "model", model)
    layer = base_model.layers[layer_idx]
    experts = layer.mlp.experts

    if hasattr(experts, "gate_up_proj"):
        weights = experts.gate_up_proj.flatten(start_dim=1)
    else:
        weights = torch.stack([expert.gate_up_proj.weight.flatten() for expert in experts])

    sim = F.cosine_similarity(
        weights.unsqueeze(1),
        weights.unsqueeze(0),
        dim=-1
    )

    return sim
