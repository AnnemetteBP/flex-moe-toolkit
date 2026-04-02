# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch import nn
import weakref

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import OutputRecorder, check_model_inputs
from ..mixtral.modeling_mixtral import MixtralModel
from ..olmo2.modeling_olmo2 import Olmo2Attention, Olmo2RMSNorm, Olmo2RotaryEmbedding
from ..olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
    OlmoeForCausalLM,
    OlmoeMLP,
    OlmoeTopKRouter,
)


class FlexMoREConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlexMoREModel`]. It is used to instantiate an FlexMoRE
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [allenai/FlexOlmo-7x7B-1T](https://huggingface.co/allenai/FlexOlmo-7x7B-1T).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the FlexMoRE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FlexMoREModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 100277):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 100257):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 5):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 7):
            Number of routed experts.
        expert_ranks ('list of int', *optional*):
            List of expert ranks for mixture of experts layers. If not provided, all experts will have
            rank 0, i.e., all experts are dense.
        expert_alphas ('list of float', *optional*):
            List of expert alphas for mixture of experts layers. Effective scale will be computed as alpha / rank.
            If not provided, all experts will have alpha equal to their rank, i.e., scale will be 1.0 for all experts.
        expert_bases ('list of int', *optional*):
            List of expert bases for mixture of experts layers. If not provided, all experts will have
            base 0, i.e., they refer to expert 0 as their base expert.
        expert_status ('list of int', *optional*):
            List of expert status for mixture of experts layers. If not provided, all experts will be active.
            Status -1 means the expert is deactivated, status 0 means the expert is active, and status 1 means the expert is always on.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.01):
            The aux loss factor for the total loss.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to normalize the topk probabilities.

    ```python
    >>> from transformers import FlexMoREModel, FlexMoREConfig

    >>> # Initializing a FlexMoRE style configuration
    >>> configuration = FlexMoREConfig()

    >>> # Initializing a model from the FlexMoRE style configuration
    >>> model = FlexMoREModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flex_more"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "num_experts"}
    default_theta = 500000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 100352,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 11008,
        num_hidden_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 4096,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-06,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 100277,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = 100257,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        num_experts_per_tok: Optional[int] = 5,
        num_experts: Optional[int] = 7,
        expert_ranks: Optional[list[int]] = None,
        expert_alphas: Optional[list[float]] = None,
        expert_bases: Optional[list[int]] = None,
        expert_status: Optional[list[int]] = None,
        output_router_logits: Optional[bool] = False,
        router_aux_loss_coef: Optional[float] = 0.01,
        norm_topk_prob: Optional[bool] = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.expert_ranks = expert_ranks if expert_ranks is not None else [0] * num_experts
        assert len(self.expert_ranks) == self.num_experts, "Length of expert_ranks must be equal to num_experts"
        self.expert_alphas = expert_alphas if expert_alphas is not None else [float(rank) for rank in self.expert_ranks]  
        assert len(self.expert_alphas) == self.num_experts, "Length of expert_alphas must be equal to num_experts"
        self.expert_bases = expert_bases if expert_bases is not None else [0] * num_experts
        assert len(self.expert_bases) == self.num_experts, "Length of expert_bases must be equal to num_experts"
        self.expert_status = expert_status if expert_status is not None else [0] * num_experts
        assert len(self.expert_status) == self.num_experts, "Length of expert_status must be equal to num_experts"
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_topk_prob = norm_topk_prob
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# FlexMore RMS norm reuses Olmo2 RMS norm, which handles low precision slightly differently than the original Olmoe.
class FlexMoRERMSNorm(Olmo2RMSNorm):
    pass


# FlexMore RMS norm reuses Olmo2 RMS norm, so that the output cos and sin are returned
# as float32 rather than the input type.
class FlexMoRERotaryEmbedding(Olmo2RotaryEmbedding):
    pass


class FlexMoREMLP(OlmoeMLP):
    pass


# FlexMore uses Olmo2 attention instead of OlmoE Attention since its `apply_rotary_pos_emb`
# implementation handles lower precision more faithfully to the Olmo codebase.
class FlexMoREAttention(Olmo2Attention):
    pass


class FlexMoREExpert(nn.Module):
    """Single expert."""

    def __init__(self, config: FlexMoREConfig, rank: int, alpha: float):
        super().__init__()
        hidden_dim = config.hidden_size
        intermediate_dim = config.intermediate_size
        if rank > 0:
            self.gate_proj_a = nn.Linear(hidden_dim, rank, bias=False)
            self.gate_proj_b = nn.Linear(rank, intermediate_dim, bias=False)
            self.up_proj_a = nn.Linear(hidden_dim, rank, bias=False)
            self.up_proj_b = nn.Linear(rank, intermediate_dim, bias=False)
            self.down_proj_a = nn.Linear(intermediate_dim, rank, bias=False)
            self.down_proj_b = nn.Linear(rank, hidden_dim, bias=False)
            self.scaling = alpha / rank
        else:
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.rank = rank
        self.base_expert = None

    def _gate_project(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank > 0:
            base = self.base_expert()
            return base._gate_project(x) + self.scaling * self.gate_proj_b(self.gate_proj_a(x))
        return self.gate_proj(x)

    def _up_project(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank > 0:
            base = self.base_expert()
            return base._up_project(x) + self.scaling * self.up_proj_b(self.up_proj_a(x))
        return self.up_proj(x)

    def _down_project(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank > 0:
            base = self.base_expert()
            return base._down_project(x) + self.scaling * self.down_proj_b(self.down_proj_a(x))
        return self.down_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._gate_project(x)
        up = self._up_project(x)
        h = self.act_fn(gate) * up
        return self._down_project(h)

class FlexMoREExperts(nn.ModuleList):
    """Collection of experts."""

    def __init__(self, config: FlexMoREConfig):
        super().__init__([
            FlexMoREExpert(config, rank, alpha)
            for rank, alpha in zip(config.expert_ranks, config.expert_alphas)
        ])
        for expert, base in zip(self, config.expert_bases):
            if expert.rank > 0:
                expert.base_expert = weakref.ref(self[base])
        # validate that there is no circular dependency in expert bases
        for expert in self:
            visited = set()
            while expert.rank > 0:
                if id(expert) in visited:
                    raise ValueError("Circular dependency detected in expert bases")
                visited.add(id(expert))
                expert = expert.base_expert()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=len(self))
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == len(self):
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class FlexMoRETopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.expert_status = torch.tensor(config.expert_status)
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = hidden_states @ self.weight.T  # (seq_len, num_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        activated_router_logits = router_logits + float('inf') * self.expert_status.to(router_logits.dtype)
        _, router_indices = torch.topk(activated_router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.gather(router_logits, dim=-1, index=router_indices)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class FlexMoRESparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = FlexMoRETopKRouter(config)
        self.experts = FlexMoREExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, top_k_index, top_k_weights).reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


# FlexMore decoder layer is identical to OlmoE decoder layer except:
# - Norm is applied after attention/feedforward rather than before.
class FlexMoREDecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: FlexMoREConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.post_attention_layernorm = FlexMoRERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = FlexMoRERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = FlexMoREAttention(config=config, layer_idx=layer_idx)
        del self.input_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

@auto_docstring
class FlexMoREPreTrainedModel(PreTrainedModel):
    config: FlexMoREConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FlexMoREDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(nn.Linear, layer_name="mlp.gate", index=0),
        "hidden_states": FlexMoREDecoderLayer,
        "attentions": FlexMoREAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, FlexMoREExpert):
            if module.rank > 0:
                init.normal_(module.gate_proj_a, mean=0.0, std=std)
                init.normal_(module.gate_proj_b, mean=0.0, std=std)
                init.normal_(module.up_proj_a, mean=0.0, std=std)
                init.normal_(module.up_proj_b, mean=0.0, std=std)
                init.normal_(module.down_proj_a, mean=0.0, std=std)
                init.normal_(module.down_proj_b, mean=0.0, std=std)
            else:
                init.normal_(module.gate_proj, mean=0.0, std=std)
                init.normal_(module.up_proj, mean=0.0, std=std)
                init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, FlexMoRETopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)


# FlexMoRE uses Mixtral model as its base instead of OlmoE model since Mixtral is more up-to-date with the rest
# of the transformers library. For example, it uses the newer mechanisms of recording submodule outputs.
# FlexMoRE model is identical to Mixtral model except:
# - FlexMoRE does not use sliding window attention.
class FlexMoREModel(MixtralModel):
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class FlexMoREForCausalLM(OlmoeForCausalLM):
    pass


__all__ = [
    "FlexMoREConfig",
    "FlexMoREForCausalLM",
    "FlexMoREModel",
    "FlexMoREPreTrainedModel",
]
