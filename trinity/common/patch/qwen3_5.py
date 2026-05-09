from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    Cache,
    F,
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ModelOutputWithPast,
    TransformersKwargs,
    Unpack,
    apply_mask_to_padding_states,
    can_return_tuple,
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


_in_gate_delta_net_with_sp = False


def ulysses_gate_delta_net_decorator(net, ulysses_sp_size):
    if getattr(net, "_is_patched", False):
        return

    net._is_patched = True

    # ulysses sequence parallel setup
    from verl.utils.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
        get_ulysses_sequence_parallel_group,
    )

    if ulysses_sp_size == 1:
        # no need to patch
        return

    # Patch net.forward
    original_net_forward = net.forward

    @wraps(original_net_forward)
    def new_net_forward(*args, **kwargs):
        global _in_gate_delta_net_with_sp
        _in_gate_delta_net_with_sp = True
        output = original_net_forward(*args, **kwargs)
        _in_gate_delta_net_with_sp = False
        return output

    net.forward = new_net_forward

    # Patch in_proj_qkv
    original_in_proj_qkv_forward = net.in_proj_qkv.forward

    @wraps(original_in_proj_qkv_forward)
    def new_in_proj_qkv_forward(input):
        output = original_in_proj_qkv_forward(input)
        group = get_ulysses_sequence_parallel_group()
        output = gather_seq_scatter_heads(output, seq_dim=1, head_dim=2, group=group)
        return output

    net.in_proj_qkv.forward = new_in_proj_qkv_forward

    # Patch conv1d layer
    original_conv1d_class = net.conv1d.__class__
    original_conv1d_getattr = original_conv1d_class.__getattr__

    @wraps(original_conv1d_getattr)
    def new_conv1d_getattr(self, name):
        global _in_gate_delta_net_with_sp
        attr = original_conv1d_getattr(self, name)
        # bias is None in Qwen3.5, so no need to patch for bias
        if name == "weight" and _in_gate_delta_net_with_sp:
            group = get_ulysses_sequence_parallel_group()
            return Slice.apply(group, attr, 0, True)
        return attr

    new_conv1d_class = type(
        f"UlyssesGated{original_conv1d_class.__name__}",
        (original_conv1d_class,),
        {"__getattr__": new_conv1d_getattr},
    )
    net.conv1d.__class__ = new_conv1d_class

    # Patch torch.split
    if not getattr(torch.split, "_is_patched_by_ulysses_gate_delta_net", False):
        original_split = torch.split

        @wraps(original_split)
        def new_split(tensor, split_size_or_sections, dim=0):
            global _in_gate_delta_net_with_sp
            if _in_gate_delta_net_with_sp and dim == -1 and len(split_size_or_sections) == 3:
                tensor = gather_heads_scatter_seq(tensor, seq_dim=1, head_dim=2)

            return original_split(tensor, split_size_or_sections, dim)

        torch.split = new_split
        torch.split._is_patched_by_ulysses_gate_delta_net = True

    # Patch chunk_gated_delta_rule
    original_chunk_gated_delta_rule = net.chunk_gated_delta_rule

    @wraps(original_chunk_gated_delta_rule)
    def new_chunk_gated_delta_rule(query, key, value, g, beta, **kwargs):
        query = gather_seq_scatter_heads(query, seq_dim=1, head_dim=2)
        key = gather_seq_scatter_heads(key, seq_dim=1, head_dim=2)
        value = gather_seq_scatter_heads(value, seq_dim=1, head_dim=2)
        g = gather_seq_scatter_heads(g, seq_dim=1, head_dim=2)
        beta = gather_seq_scatter_heads(beta, seq_dim=1, head_dim=2)
        output, last_recurrent_state = original_chunk_gated_delta_rule(
            query, key, value, g, beta, **kwargs
        )
        output = gather_heads_scatter_seq(output, seq_dim=1, head_dim=2)
        return output, last_recurrent_state

    net.chunk_gated_delta_rule = new_chunk_gated_delta_rule


def gate_delta_net_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Cache | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    # Set up dimensions for reshapes later
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None
        and cache_params.has_previous_state(self.layer_idx)
        and seq_len == 1
    )

    # getting projected states from cache if it exists
    if use_precomputed_states:
        conv_state = cache_params.layers[self.layer_idx].conv_states
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        # 2. Convolution sequence transformation
        # NOTE: the conv state is updated in `causal_conv1d_update`
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
        if self.causal_conv1d_fn is not None:
            seq_idx = kwargs.get("seq_idx", None)
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [
            self.key_dim,
            self.key_dim,
            self.value_dim,
        ],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        chunk_kwargs = {}
        if getattr(self.chunk_gated_delta_rule, "__module__", "").startswith("fla."):
            chunk_kwargs["cu_seqlens"] = kwargs.get("cu_seqlens", None)

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            **chunk_kwargs,
        )

    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    # Update cache
    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.FloatTensor:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Token Mixer
    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
    elif self.layer_type == "full_attention":
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


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


def qwen35_vision_fast_pos_embed_interpolate(self, grid_thw):
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    device = grid_thw.device

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

    patch_pos_embeds_permute = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    return patch_pos_embeds


@can_return_tuple
def qwen35_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3_5ModelOutputWithPast:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    vision_config = self.config.vision_config
    pixel_values_dim = (
        vision_config.in_channels
        * vision_config.temporal_patch_size
        * (vision_config.patch_size**2)
    )
    merge_size = vision_config.spatial_merge_size

    device = inputs_embeds.device
    has_mm_local = torch.tensor(
        [int(pixel_values is not None), int(pixel_values_videos is not None)], device=device
    )
    has_mm_global = has_mm_local.clone()
    if dist.is_initialized():
        dist.all_reduce(has_mm_global)
    has_mm_global = has_mm_global > 0

    # check images
    if has_mm_global[0].item():
        if not has_mm_local[0].item():
            pixel_values = torch.zeros(
                (merge_size * merge_size, pixel_values_dim), dtype=torch.float32, device=device
            )
            image_grid_thw = torch.ones((1, 3), dtype=torch.int64, device=device)
            image_grid_thw[:, 1:] = merge_size

        image_outputs: BaseModelOutputWithPooling = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = image_outputs.pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        if has_mm_local[0].item():
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:  # patched for backward
            inputs_embeds[0] = inputs_embeds[0] + image_embeds[0] * 0.0

    # check videos
    if has_mm_global[1].item():
        if not has_mm_local[1].item():
            pixel_values_videos = torch.zeros(
                (merge_size * merge_size, pixel_values_dim), dtype=torch.float32, device=device
            )
            video_grid_thw = torch.ones((1, 3), dtype=torch.int64, device=device)
            video_grid_thw[:, 1:] = merge_size

        video_outputs: BaseModelOutputWithPooling = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        video_embeds = video_outputs.pooler_output
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        if has_mm_local[1].item():
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        else:  # patched for backward
            inputs_embeds[0] = inputs_embeds[0] + video_embeds[0] * 0.0

    if position_ids is None:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    return Qwen3_5ModelOutputWithPast(
        **outputs,
        rope_deltas=self.rope_deltas,
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
