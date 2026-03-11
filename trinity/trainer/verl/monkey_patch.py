import importlib
import sys
from typing import Dict, Optional, Set

import torch
from transformers.modeling_utils import PreTrainedModel

from trinity.utils.log import get_logger

# Map model types to their specific implementation modules.
# To extend support for a new model, simply add an entry here.
MODEL_TYPE_TO_MODULE_MAP: Dict[str, str] = {
    "qwen2_5_vl": "verl.models.transformers.qwen2_vl",
    "qwen2_vl": "verl.models.transformers.qwen2_vl",
    "qwen3_vl": "verl.models.transformers.qwen3_vl",
    "qwen3_vl_moe": "verl.models.transformers.qwen3_vl",
    "qwen3_5": "trinity.common.patch.qwen3_5",
    "qwen3_5_moe": "trinity.common.patch.qwen3_5",
    "glm4v": "verl.models.transformers.glm4v",
}

DEFAULT_MODULE_PATH = "verl.models.transformers.dense_common"
VALID_BACKENDS: Set[str] = {"triton", "torch"}


# modified from verl.models.transformers.monkey_patch.patch_forward_with_backends
def patch_forward_with_backends(
    model: PreTrainedModel,
    use_fused_kernels: bool = False,
    fused_kernels_backend: Optional[str] = None,
) -> None:
    """
    Monkey-patch the model's forward method with optimized backend implementations.

    Args:
        model: The model to patch.
        use_fused_kernels: Whether to enable fused kernels.
        fused_kernels_backend: The backend to use ('triton' or 'torch').
    """
    logger = get_logger(__name__)

    # 1. Validation & Early Exit
    if not use_fused_kernels:
        return

    if fused_kernels_backend not in VALID_BACKENDS:
        logger.warning(
            f"Skipping patch for {model.__class__.__name__}: "
            f"Invalid backend '{fused_kernels_backend}'. Choose from {VALID_BACKENDS}."
        )
        return

    # 2. Resolve Module Path
    model_type: str = getattr(model.config, "model_type", None)
    module_path = MODEL_TYPE_TO_MODULE_MAP.get(model_type, DEFAULT_MODULE_PATH)

    # 3. Dynamic Import
    try:
        backend_module = importlib.import_module(module_path)
    except ImportError as e:
        logger.error(f"Failed to import {module_path} for {model.__class__.__name__}: {e}")
        return

    # 4. Select and Apply Forward Function
    func_name = f"forward_with_{fused_kernels_backend}_backend"
    patched_forward = getattr(backend_module, func_name, None)

    if patched_forward is None:
        logger.error(f"Function '{func_name}' not found in {module_path}")
        return

    model.__class__.forward = patched_forward
    logger.info(f"Applied {fused_kernels_backend.upper()} backend for {model.__class__.__name__}")


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

    elif model.config.model_type in ["qwen3_5", "qwen3_5_moe"]:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextModel,
        )

        # Step 1: bug fix in transformers==5.2.0
        # see https://github.com/huggingface/transformers/pull/44382
        if "Qwen3_5TextDecoderLayer" in model._no_split_modules:
            model._no_split_modules.remove("Qwen3_5TextDecoderLayer")
            model.model._no_split_modules.remove("Qwen3_5TextDecoderLayer")
        if "Qwen3_5MoeTextDecoderLayer" in model._no_split_modules:
            model._no_split_modules.remove("Qwen3_5MoeTextDecoderLayer")
            model.model._no_split_modules.remove("Qwen3_5MoeTextDecoderLayer")

        # see https://github.com/huggingface/transformers/pull/44399
        if is_transformers_version_in_range(max_version="5.3.0"):
            from trinity.common.patch.qwen3_5 import qwen35_text_forward

            Qwen3_5TextModel.forward = qwen35_text_forward
            Qwen3_5MoeTextModel.forward = qwen35_text_forward

        # Step 2: patch input for multimodal sequence parallelism
        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(Qwen3_5TextModel)
            patch_vlm_for_ulysses_input_slicing(Qwen3_5MoeTextModel)

            from trinity.common.patch.qwen3_5 import (
                ulysses_gated_delta_net_forward_decorator,
            )

            for layer in model.model.language_model.layers:
                if layer.layer_type == "linear_attention":
                    layer.linear_attn.forward = ulysses_gated_delta_net_forward_decorator(
                        layer.linear_attn.forward
                    )

        # Step 3: patch verl.utils.flops_counter
        from verl.utils.flops_counter import ESTIMATE_FUNC, _estimate_qwen2_flops

        ESTIMATE_FUNC.update(
            {
                "qwen3_5": _estimate_qwen2_flops,
                "qwen3_5_moe": _estimate_qwen2_flops,
            }
        )

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
