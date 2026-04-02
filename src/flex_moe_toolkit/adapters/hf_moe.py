import torch

from flex_moe_toolkit.prev_analysis.routing import ensure_layer_tensors

from .base import BaseAdapter


class HFMoEAdapter(BaseAdapter):
    def get_router_logits(self, model, inputs):
        outputs = model(**inputs, output_router_logits=True)
        return outputs.router_logits

    def get_router_probs(self, model, inputs):
        return self.router_logits_to_probs(self.get_router_logits(model, inputs))

    def router_logits_to_probs(self, router_logits):
        return tuple(
            torch.softmax(layer_logits, dim=-1)
            for layer_logits in ensure_layer_tensors(router_logits)
        )
