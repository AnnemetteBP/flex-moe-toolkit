import torch



def capture_router_logits(model, inputs):
    """
    Run model and return router logits.
    """

    model.config.output_router_logits = True

    with torch.no_grad():
        outputs = model(**inputs)

    router_logits = outputs.router_logits

    if router_logits is None:
        raise ValueError(
            "Model did not return router logits. Ensure output_router_logits=True."
        )

    return router_logits