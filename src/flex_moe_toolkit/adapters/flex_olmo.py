from __future__ import annotations

from typing import Iterable

import torch

from .base import BaseAdapter


def iter_flex_olmo_layers(model) -> Iterable[torch.nn.Module]:
    """
    Yield decoder layers from either a bare FlexOlmoModel or FlexOlmoForCausalLM.
    """

    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose FlexOlmo layers via `.layers` or `.model.layers`.")
    return layers


class FlexOlmoAdapter(BaseAdapter):
    """
    Capture router outputs from local FlexOlmo/FlexMoRE-style models.

    The current local router implementation returns normalized routing probabilities
    as its first output. We log-transform them so the rest of the toolkit can treat
    them like router logits and recover the original probabilities via softmax.
    """

    def _collect_router_probs(self, model, inputs):
        router_probs = []
        handles = []

        def hook_fn(_module, _args, output):
            probs = output[0] if isinstance(output, tuple) else output
            router_probs.append(probs.detach())

        try:
            for layer in iter_flex_olmo_layers(model):
                handles.append(layer.mlp.gate.register_forward_hook(hook_fn))

            with torch.no_grad():
                model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        if not router_probs:
            raise ValueError("No router outputs were captured from the FlexOlmo model.")

        return tuple(router_probs)

    def get_router_logits(self, model, inputs):
        router_probs = self._collect_router_probs(model, inputs)
        return tuple(torch.log(layer_probs.clamp_min(1e-9)) for layer_probs in router_probs)

    def get_router_probs(self, model, inputs):
        return self._collect_router_probs(model, inputs)

    def router_logits_to_probs(self, router_logits):
        return tuple(torch.softmax(layer_logits, dim=-1) for layer_logits in router_logits)
