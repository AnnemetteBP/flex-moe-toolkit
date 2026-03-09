import torch
import torch.nn.functional as F



def expert_similarity_matrix(model, layer_idx):
    """
    Compute cosine similarity between experts in a given layer.
    """

    layer = model.model.layers[layer_idx]

    experts = layer.mlp.experts

    weights = []

    for expert in experts:

        w = expert.gate_up_proj.weight.flatten()

        weights.append(w)

    weights = torch.stack(weights)

    sim = F.cosine_similarity(
        weights.unsqueeze(1),
        weights.unsqueeze(0),
        dim=-1
    )

    return sim