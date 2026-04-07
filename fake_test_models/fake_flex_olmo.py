from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FakeFlexOlmoConfig:
    hidden_size: int = 128
    num_hidden_layers: int = 3
    num_experts: int = 7
    num_experts_per_tok: int = 2
    output_router_logits: bool = False


class FakeFlexOlmoRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = True
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.02)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, _hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(flat_hidden, self.weight)
        router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        effective_top_k = min(self.top_k, self.num_experts)
        top_values, top_indices = torch.topk(router_probs, effective_top_k, dim=-1)

        if self.norm_topk_prob:
            top_values = top_values / top_values.sum(dim=-1, keepdim=True)

        router_probs = router_probs.view(batch_size, sequence_length, self.num_experts)
        top_values = top_values.view(batch_size, sequence_length, effective_top_k)
        top_indices = top_indices.view(batch_size, sequence_length, effective_top_k)
        return router_probs, top_values.to(router_probs.dtype), top_indices


class FakeExpert(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor):
        return self.ffn(hidden_states)


class FakeFlexOlmoExperts(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(FakeExpert(hidden_size) for _ in range(num_experts))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ):
        if top_k_index.ndim == 2:
            batch_size, sequence_length, _hidden_size = hidden_states.shape
            top_k = top_k_index.shape[-1]
            top_k_index = top_k_index.view(batch_size, sequence_length, top_k)
            top_k_weights = top_k_weights.view(batch_size, sequence_length, top_k)

        output = torch.zeros_like(hidden_states)

        for expert_idx, expert in enumerate(self.experts):
            expert_hidden = expert(hidden_states)
            matches = top_k_index == expert_idx
            if not matches.any():
                continue
            weights = (top_k_weights * matches.to(top_k_weights.dtype)).sum(dim=-1, keepdim=True)
            output = output + (expert_hidden * weights)

        return output


class FakeFlexOlmoSparseMlp(nn.Module):
    def __init__(self, config: FakeFlexOlmoConfig):
        super().__init__()
        self.gate = FakeFlexOlmoRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
        )
        self.experts = FakeFlexOlmoExperts(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
        )

    def forward(self, hidden_states: torch.Tensor):
        router_probs, top_k_weights, top_k_index = self.gate(hidden_states)
        moe_output = self.experts(hidden_states, top_k_index, top_k_weights)
        return hidden_states + moe_output, router_probs


class FakeFlexOlmoLayer(nn.Module):
    def __init__(self, config: FakeFlexOlmoConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp = FakeFlexOlmoSparseMlp(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        normed = self.layer_norm(hidden_states)
        updated, router_probs = self.mlp(normed)
        return updated, router_probs


class FakeFlexOlmoModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_experts: int = 7,
        num_hidden_layers: int = 3,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.config = FakeFlexOlmoConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )
        self.layers = nn.ModuleList(
            FakeFlexOlmoLayer(self.config, layer_idx=layer_idx)
            for layer_idx in range(num_hidden_layers)
        )

    def forward(self, input_ids=None, output_router_logits=False, **kwargs):
        if input_ids is None:
            raise ValueError("`input_ids` must be provided as fake hidden-state inputs.")

        hidden_states = input_ids
        router_probs = []

        for layer in self.layers:
            hidden_states, layer_router_probs = layer(hidden_states)
            router_probs.append(layer_router_probs)

        return SimpleNamespace(
            last_hidden_state=hidden_states,
            router_logits=tuple(router_probs) if output_router_logits else None,
        )


def build_fake_inputs(
    batch_size: int = 2,
    sequence_length: int = 10,
    hidden_size: int = 128,
    seed: int = 7,
):
    generator = torch.Generator().manual_seed(seed)
    return {"input_ids": torch.randn(batch_size, sequence_length, hidden_size, generator=generator)}


"""
Test examples:

from fake_test_models.fake_flex_olmo import FakeFlexOlmoModel, build_fake_inputs
from flex_moe_toolkit.adapters.flex_olmo import FlexOlmoAdapter
from flex_moe_toolkit.pipelines.flex_olmo import analyze_flex_olmo_routing

model = FakeFlexOlmoModel()
inputs = build_fake_inputs()

adapter = FlexOlmoAdapter()
router_logits = adapter.get_router_logits(model, inputs)
results = analyze_flex_olmo_routing(model, inputs)

print(len(router_logits))
print(results["entropy"])
"""
