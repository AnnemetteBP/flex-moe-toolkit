import json
import torch



def log_routing(router_logits, path):
    """
    Save routing probabilities for each layer and token.
    Works for shapes:
    (batch, tokens, experts)
    or
    (tokens, experts)
    """

    records = []

    for layer_idx, logits in enumerate(router_logits):

        probs = torch.softmax(logits, dim=-1)

        shape = probs.shape

        # Case 1: (batch, tokens, experts)
        if len(shape) == 3:

            batch, tokens, experts = shape

            for token_idx in range(tokens):

                record = {
                    "layer": layer_idx,
                    "token": token_idx,
                    "probs": probs[0, token_idx].tolist()
                }

                records.append(record)

        # Case 2: (tokens, experts)
        elif len(shape) == 2:

            tokens, experts = shape

            for token_idx in range(tokens):

                record = {
                    "layer": layer_idx,
                    "token": token_idx,
                    "probs": probs[token_idx].tolist()
                }

                records.append(record)

        else:
            raise ValueError(f"Unexpected router logits shape: {shape}")

    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")