"""Monkey patching for 'glm4v' models."""

from typing import Optional, Union

import torch
from transformers.models.glm4v.modeling_glm4v import (
    BaseModelOutputWithPast,
    Cache,
    DynamicCache,
    FlashAttentionKwargs,
    Glm4vTextModel,
    Unpack,
    create_causal_mask,
)


def glm4v_text_forward(
    self: Glm4vTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        text_position_ids = position_ids[0]
    elif position_ids.dim() == 2:
        text_position_ids = position_ids
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = layer_outputs

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )
