from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    BaseModelOutputWithPast,
    Cache,
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ModelOutputWithPast,
    TransformersKwargs,
    Unpack,
    capture_outputs,
    create_causal_mask,
    merge_with_config_defaults,
)
from verl.utils.ulysses import all_gather_tensor


class Slice(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        global_tensor: Tensor,
        dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> Tensor:
        ctx.group = group
        ctx.dim = dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        # slice the input tensor
        dim_size = global_tensor.size(dim)
        if dim_size % sp_world_size != 0:
            raise ValueError(
                f"Cannot evenly slice tensor of size {dim_size} along dim {dim} "
                f"across {sp_world_size} ranks. This would truncate data. "
                "Ensure the dimension size is divisible by the SP world size."
            )
        parts = dim_size // sp_world_size
        slc = [slice(None)] * len(global_tensor.shape)
        slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
        return global_tensor[tuple(slc)].contiguous()

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Any:
        if ctx.grad_scaler:
            grad_outputs = grad_outputs / ctx.sp_world_size

        output = all_gather_tensor(grad_outputs, ctx.group, ctx.async_op)
        return (
            None,
            torch.cat(output.split(grad_outputs.size(0), dim=0), dim=ctx.dim).contiguous(),
            None,
            None,
            None,
            None,
        )


# TODO: may optimize this function
def ulysses_gated_delta_net_forward_decorator(func):
    @wraps(func)
    def wrapper(
        hidden_states: torch.Tensor,
        **kwargs,
    ):
        from verl.utils.ulysses import (
            gather_outputs_and_unpad,
            get_ulysses_sequence_parallel_group,
            get_ulysses_sequence_parallel_world_size,
        )

        ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
        if ulysses_sp_size > 1:
            hidden_states = gather_outputs_and_unpad(hidden_states, gather_dim=1)

        output = func(hidden_states, **kwargs)

        if ulysses_sp_size > 1:
            group = get_ulysses_sequence_parallel_group()
            output = Slice.apply(group, output, 1)
        return output

    return wrapper


@merge_with_config_defaults
@capture_outputs
def qwen35_text_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        past_key_values = Qwen3_5DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # mrope: the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    causal_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_mask = (
            linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
        )

        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


@dataclass
class Qwen3_5CausalLMOutputForPPO(Qwen3_5CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def forward_with_torch_backend(
    self: Qwen3_5ForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> tuple | Qwen3_5CausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError(
            "To use forward_with_torch_backend, either labels or input_ids must be provided."
        )

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: Qwen3_5ForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> tuple | Qwen3_5CausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError(
            "To use forward_with_triton_backend, either labels or input_ids must be provided."
        )

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
