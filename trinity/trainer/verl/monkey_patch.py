import sys

import torch
from transformers.modeling_utils import PreTrainedModel

from trinity.utils.log import get_logger


# modified from verl.models.transformers.monkey_patch.apply_monkey_patch
def apply_monkey_patch(  # noqa: C901
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
    use_tiled_mlp: bool = False,
    tiled_mlp_shards: int = 4,
):
    """
    Apply monkey patch to the models for ulysses sequence parallel, fused kernel, and tiled MLP.

    In the end of this function forward function of the model is patched for fused kernel.
    If the model is not supported with fused kernel, please return after patch.

    Args:
        model: The model to apply the monkey patch.
        ulysses_sp_size: The size of ulysses sequence parallel.
        use_remove_padding: Whether to use remove padding.
        use_fused_kernels: Whether to use fused kernels.
        fused_kernels_backend: The backend to use for fused kernels.
        use_tiled_mlp: Whether to use TiledMLP for memory-efficient MLP computation.
        tiled_mlp_shards: Number of shards for TiledMLP (higher = lower memory, slightly slower).
    """
    from verl.models.transformers.monkey_patch import (
        _ulysses_flash_attention_forward,
        patch_forward_with_backends,
        patch_vlm_for_ulysses_input_slicing,
    )
    from verl.utils.import_utils import is_trl_available
    from verl.utils.transformers_compat import is_transformers_version_in_range

    logger = get_logger(__name__)

    # Apply TiledMLP monkey patch for memory-efficient MLP computation
    if use_tiled_mlp:
        from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch

        model_type = getattr(model.config, "model_type", None)
        apply_tiled_mlp_monkey_patch(num_shards=tiled_mlp_shards, model_type=model_type)

    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    module = sys.modules[model.__module__]

    try:
        num_attention_heads, num_key_value_heads = (
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
        )
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    assert (
        num_attention_heads % ulysses_sp_size == 0
    ), f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
    assert (
        num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0
    ), (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size "
        f"{ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,"
        f"kv heads are repeated to ensure correctness."
    )

    if is_trl_available():
        from trl import AutoModelForCausalLMWithValueHead  # type: ignore

        def state_dict(self, *args, **kwargs):
            return torch.nn.Module.state_dict(self, *args, **kwargs)

        AutoModelForCausalLMWithValueHead.state_dict = state_dict
        logger.info("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")

    # TODO: VLM models only, unify monkey patch to LLM models.
    if model.config.model_type in ["qwen2_5_vl", "qwen2_vl"]:
        # Step 1: patch model to support image-text mixed data
        if is_transformers_version_in_range(min_version="4.52.0"):
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLTextModel,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextModel
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLModel as Qwen2_5_VLTextModel,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLModel as Qwen2VLTextModel,
            )

        if is_transformers_version_in_range(min_version="4.53.0", max_version="4.53.3"):
            raise RuntimeError("Transformers 4.53.* is bugged. Use transformers 4.54.0 or later.")

        # Step 2: patch input for multimodal sequence parallelism
        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
            patch_vlm_for_ulysses_input_slicing(Qwen2VLTextModel)

    elif model.config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:
        # Step 1: patch model to support image-text mixed data
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextModel,
        )
        from verl.models.transformers.qwen3_vl import (
            patch_qwen3_vl_moe_sparse_moe_block_forward,
        )

        # Step 1.5: patch Qwen3VLMoeTextSparseMoeBlock to fix transformers 4.57.3 bug
        if model.config.model_type == "qwen3_vl_moe" and is_transformers_version_in_range(
            max_version="4.57.3"
        ):
            patch_qwen3_vl_moe_sparse_moe_block_forward()

        # Step 2: patch input for multimodal sequence parallelism
        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(Qwen3VLTextModel)
            patch_vlm_for_ulysses_input_slicing(Qwen3VLMoeTextModel)

    elif model.config.model_type == "glm4v":
        # Step 1: patch model to support image-text mixed data

        from transformers.models.glm4v.modeling_glm4v import Glm4vTextModel

        from trinity.common.patch.glm4v import glm4v_text_forward

        Glm4vTextModel.forward = glm4v_text_forward

        # Step 2: patch input for multimodal sequence parallelism
        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(Glm4vTextModel)

    elif model.config.model_type == "kimi_vl":
        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(module.DeepseekV3ForCausalLM)

        if use_fused_kernels:
            logger.info("Not support fused kernels for KimiVL")

    if use_remove_padding or ulysses_sp_size > 1:
        if hasattr(module, "_flash_attention_forward"):  # transformers <= 4.47.1 or legacy models
            module._flash_attention_forward = _ulysses_flash_attention_forward
            logger.info(f"Monkey patch _flash_attention_forward in {model.__module__}")
        else:
            from transformers.integrations import flash_attention

            flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
            logger.info(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")

    patch_forward_with_backends(
        model, use_fused_kernels=use_fused_kernels, fused_kernels_backend=fused_kernels_backend
    )
