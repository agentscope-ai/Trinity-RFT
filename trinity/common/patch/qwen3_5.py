from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    BaseModelOutputWithPooling,
    Cache,
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ModelOutputWithPast,
    TransformersKwargs,
    Unpack,
    can_return_tuple,
)
from verl.utils.ulysses import all_gather_tensor, slice_input_tensor


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

        # Slice attention_mask along sequence dimension to match SP-scattered hidden_states.
        # Under Ulysses SP, hidden_states is already scattered (seq_len / sp_size),
        # but attention_mask is still at full sequence length, causing a shape mismatch
        # in apply_mask_to_padding_states.
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            kwargs["attention_mask"] = slice_input_tensor(attention_mask, dim=1, padding=False)

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
    vocab_weights = self.lm_head.weight
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()
    vocab_weights = vocab_weights.to(hidden_states.device)
    hidden_states = hidden_states.to(vocab_weights.dtype)
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=vocab_weights,
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

    vocab_weights = self.lm_head.weight
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()
    vocab_weights = vocab_weights.to(hidden_states.device)
    hidden_states = hidden_states.to(vocab_weights.dtype)
    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        vocab_weights,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
