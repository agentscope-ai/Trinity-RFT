from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
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
    """Decorator to enable Ulysses Sequence Parallel for Qwen3.5 GateDeltaNet linear attention.

    This decorator patches the GateDeltaNet module to support sequence parallelism using the Ulysses
    strategy. It intercepts various operations (forward pass, projections, convolutions, and attention)
    to properly scatter/gather tensors across sequence parallel ranks.

    Args:
        net: The GateDeltaNet module to patch (typically a linear attention layer).
        ulysses_sp_size: The sequence parallel world size. If 1, no patching is performed.

    Note:
        - This function patches the module in-place and sets a `_is_patched` flag to avoid double-patching.
        - The sequence parallel operations are controlled via a global `_in_gate_delta_net_with_sp` flag.
        - The patching includes modifications to forward, in_proj_qkv, conv1d, torch.split, and chunk_gated_delta_rule.
    """
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


# removed when following PR is merged
# https://github.com/huggingface/transformers/pull/45034/changes
def gate_delta_net_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Cache | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
):
    """Forward pass for Qwen3.5 GateDeltaNet linear attention with packing support.

    This implementation of the linear attention forward pass supports packed sequences for efficient
    training, following the approach referenced in the Hugging Face transformers PR #45034.
    It handles both incremental (cached) and non-cached inference modes.

    Args:
        hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_dim).
        cache_params: Optional cache parameters for incremental decoding.
        attention_mask: Optional attention mask to mask out padding positions.
        **kwargs: Additional keyword arguments passed to sub-components (e.g., seq_idx for packed sequences).

    Returns:
        Output tensor of shape (batch_size, seq_len, hidden_dim) after linear attention computation.
    """
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


# removed when following PR is merged
# https://github.com/huggingface/transformers/pull/45034/changes
def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.FloatTensor:
    """Forward pass for a Qwen3.5 decoder layer supporting packed sequences.

    This function implements a full transformer decoder layer with support for packed sequences
    (packing training). It combines token mixing (via linear or full attention) with a feed-forward
    network, with residual connections around each sub-layer.

    Args:
        hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_dim).
        position_embeddings: Tuple of (cos_cached, sin_cached) for rotary position embeddings.
        attention_mask: Optional attention mask.
        position_ids: Optional position IDs for the sequence.
        past_key_values: Optional cache for incremental decoding.
        **kwargs: Additional arguments including:
            - layer_type: Either 'linear_attention' or 'full_attention' to determine token mixer.
            - seq_idx: Sequence indices for packed sequence training.

    Returns:
        Output hidden states of same shape as input (batch_size, seq_len, hidden_dim).
    """
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


def qwen35_vision_fast_pos_embed_interpolate(self, grid_thw):
    """Interpolate vision position embeddings for variable resolution inputs with proper device handling.

    This function performs bilinear interpolation of position embeddings to support variable spatial
    resolutions. It fixes the device handling issue that occurred during CPU offloading, ensuring all
    tensors are created and operated on the same device as the input.

    Args:
        grid_thw: Tensor of shape (num_images, 3) containing temporal, height, and width dimensions
                 for each image in the batch.

    Returns:
        Interpolated position embeddings of shape (total_patches, embedding_dim) after merging,
        where total_patches is the sum of all h*w for each image after spatial merging.

    Note:
        - The function supports batch processing of multiple images with different resolutions.
        - Spatial merging is applied based on config.spatial_merge_size.
        - All tensors are properly placed on the same device as the input grid_thw.
    """
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    device = grid_thw.device  # modified to ensure tensors are created on the correct device

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
    """Qwen3.5 model forward pass with multimodal support and gradient synchronization across ranks.

    This forward function handles multimodal training (images and/or videos) across multiple GPU ranks
    with proper synchronization. When a rank doesn't have image/video inputs but other ranks do (common in
    distributed training with different data samples), it creates dummy images/videos to maintain consistency
    and avoid hanging in collective operations.

    Args:
        input_ids: Token IDs of shape (batch_size, seq_len).
        attention_mask: Attention mask for padding tokens.
        position_ids: Position IDs for embeddings.
        past_key_values: Cached key-values for incremental decoding.
        inputs_embeds: Pre-computed input embeddings (alternative to input_ids).
        pixel_values: Image pixel values of shape (num_images, channels, height, width).
        pixel_values_videos: Video pixel values of shape (num_videos, frames, channels, height, width).
        image_grid_thw: Grid dimensions (temporal, height, width) for images.
        video_grid_thw: Grid dimensions (temporal, height, width) for videos.
        mm_token_type_ids: Token type IDs to distinguish image, video, and text tokens.
        **kwargs: Additional arguments.

    Returns:
        Qwen3_5ModelOutputWithPast containing language model outputs with rope_deltas for position embeddings.

    Note:
        - Dummy images/videos are created with shape based on spatial_merge_size when needed for gradient synchronization.
        - Uses distributed communication (dist.all_reduce) to synchronize multimodal input availability across ranks.
    """
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
    """Compute log probabilities and entropy for reinforcement learning using PyTorch backend.

    This function computes per-token log probabilities and entropy from the language model's hidden
    states using a fused PyTorch-based linear projection. It's designed for PPO and other RL algorithms
    that require per-token probability distributions over the vocabulary.

    Args:
        input_ids: Token IDs of shape (batch_size, seq_len).
        labels: Optional labels for loss computation. If None, input_ids are rolled to compute shifted targets.
        temperature: Temperature scaling for softmax (default: 1.0). Used to control probability distribution sharpness.
        **kwargs: Additional arguments passed to the model (e.g., attention_mask).

    Returns:
        Qwen3_5CausalLMOutputForPPO containing:
            - log_probs: Log probabilities of shape (batch_size, seq_len)
            - entropy: Entropy values of shape (batch_size, seq_len)
            - hidden_states: Hidden states from the model forward pass

    Raises:
        RuntimeError: If neither labels nor input_ids is provided.

    Note:
        - Uses FusedLinearForPPO for efficient torch-based computation.
        - The log probability target is computed by rolling labels (or input_ids) by -1 to create next-token prediction targets.
    """
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
    """Compute log probabilities and entropy for reinforcement learning using Triton kernel backend.

    This function computes per-token log probabilities and entropy from the language model's hidden
    states using an optimized Triton kernel (linear_cross_entropy). It provides better performance
    compared to the PyTorch backend for large vocabularies, suitable for PPO and other RL algorithms.

    Args:
        input_ids: Token IDs of shape (batch_size, seq_len).
        labels: Optional labels for loss computation. If None, input_ids are rolled to compute shifted targets.
        temperature: Temperature scaling for softmax (default: 1.0). Used to control probability distribution sharpness.
        **kwargs: Additional arguments passed to the model (e.g., attention_mask).

    Returns:
        Qwen3_5CausalLMOutputForPPO containing:
            - log_probs: Log probabilities of shape (batch_size, seq_len)
            - entropy: Entropy values of shape (batch_size, seq_len)
            - hidden_states: Hidden states from the model forward pass

    Raises:
        RuntimeError: If neither labels nor input_ids is provided.

    Note:
        - Uses the linear_cross_entropy Triton kernel from verl for highly optimized computation.
        - The log probability target is computed by rolling labels (or input_ids) by -1 to create next-token prediction targets.
        - Generally faster than forward_with_torch_backend for large vocabulary sizes.
    """
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
