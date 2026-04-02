class BaseAdapter:
    def get_router_logits(self, model, inputs):
        raise NotImplementedError

    def get_router_probs(self, model, inputs):
        return self.router_logits_to_probs(self.get_router_logits(model, inputs))

    def router_logits_to_probs(self, router_logits):
        raise NotImplementedError
