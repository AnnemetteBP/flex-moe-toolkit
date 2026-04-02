def iter_router_modules(model):
    """
    Yield router-like modules across the model.
    """

    for name, module in model.named_modules():
        lowered = name.lower()
        if lowered.endswith("router") or ".router" in lowered or "gate" in lowered:
            yield name, module


def register_router_hook(model, hook_fn):
    """
    Register the provided forward hook on every router-like module.
    Returns the created hook handles so callers can remove them later.
    """

    handles = []

    for _, module in iter_router_modules(model):
        handles.append(module.register_forward_hook(hook_fn))

    if not handles:
        raise ValueError("No router-like modules were found on the model.")

    return handles
