import asyncio
from logging import Logger

import vllm
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version

from trinity.common.config import InferenceModelConfig


def vllm_patch():  # noqa: C901
    import transformers

    # Patch for Kimi-VL-A3B-Thinking
    if not hasattr(transformers.activations, "PytorchGELUTanh"):
        transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh

    if transformers.__version__ >= "5.0.0" and vllm.__version__ < "0.16.0":
        import copy

        from vllm.config.vllm import PretrainedConfig, VllmConfig, replace

        def with_hf_config(
            self,
            hf_config: PretrainedConfig,
            architectures: list[str] | None = None,
        ) -> "VllmConfig":
            if architectures is not None:
                hf_config = copy.deepcopy(hf_config)
                hf_config.architectures = architectures

            model_config = copy.deepcopy(self.model_config)
            if (
                model_config.is_multimodal_model
                and hasattr(model_config.hf_config, "tie_word_embeddings")
                and not hasattr(hf_config.get_text_config(), "tie_word_embeddings")
            ):
                # In Transformers v5, tie_word_embeddings belongs to the config of the class
                # that can see both layers to be tied. For example:
                #
                # SomeVLModel:
                #   self.language_model = SomeLanguageModel()
                #   self.vision_model = SomeVisionModel()
                #
                # SomeVLModelForMultimodalLM:
                #   self.model = SomeVLModel()
                #   self.lm_head = nn.Linear()
                #
                # Therefore, tie_word_embeddings is defined in SomeVLModelForMultimodalLM's
                # config and is not present in SomeVLModel's config. In vLLM, the lm_head
                # belongs to the language_model, so we must ensure that tie_word_embeddings
                # is set in the language_model's config.
                tie_word_embeddings = model_config.hf_config.tie_word_embeddings
                hf_config.get_text_config().tie_word_embeddings = tie_word_embeddings
            model_config.hf_config = hf_config
            model_config.model_arch_config = model_config.get_model_arch_config()

            return replace(self, model_config=model_config)

        VllmConfig.with_hf_config = with_hf_config

        import math

        from vllm.model_executor.models.qwen2_vl import (
            ImageSize,
            Qwen2VLImageProcessor,
            Qwen2VLProcessingInfo,
            smart_resize,
        )

        def _get_vision_info(
            self,
            *,
            image_width: int,
            image_height: int,
            num_frames: int = 1,
            do_resize: bool = True,
            image_processor: Qwen2VLImageProcessor | None,
        ) -> tuple[ImageSize, int]:
            if image_processor is None:
                image_processor = self.get_image_processor()

            hf_config = self.get_hf_config()
            vision_config = hf_config.vision_config
            patch_size = vision_config.patch_size
            merge_size = vision_config.spatial_merge_size
            temporal_patch_size = vision_config.temporal_patch_size

            if do_resize:
                resized_height, resized_width = smart_resize(
                    height=image_height,
                    width=image_width,
                    factor=patch_size * merge_size,
                    min_pixels=image_processor.size["shortest_edge"],
                    max_pixels=image_processor.size["longest_edge"],
                )
                preprocessed_size = ImageSize(width=resized_width, height=resized_height)
            else:
                preprocessed_size = ImageSize(width=image_width, height=image_height)

            # NOTE: Frames are padded to be divisible by `temporal_patch_size`
            # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
            padded_num_frames = num_frames + num_frames % temporal_patch_size

            grid_t = max(padded_num_frames // temporal_patch_size, 1)
            grid_h = preprocessed_size.height // patch_size
            grid_w = preprocessed_size.width // patch_size

            num_patches = grid_t * grid_h * grid_w
            num_vision_tokens = num_patches // (merge_size**2)

            return preprocessed_size, num_vision_tokens

        def get_image_size_with_most_features(self, max_pixels: int | None = None) -> ImageSize:
            # NOTE: Simply processing a huge size with _get_vision_info might not give a
            # size that maximizes the number of featrues, i.e., the number of (merged)
            # patches. This is because the number of patches limits the allowed aspect
            # ratios. For example, suppose the maximum number of patches is 1280. A square
            # image cannot be broken down into 1280 patches, so feeding a giant square image
            # into _get_vision_info will not yield a size that maximizes the number of
            # patches. Therefore, we directly factorize the maximum number of patches into
            # height and width. The tricky part is to avoid extreme aspect ratios (>200 for
            # qwen2-vl). If we can't find a suitable aspect ratio, we decrease the number of
            # patches and retry. This is safe because the processor does not accept extreme
            # aspect ratios, so there is no valid post-resize image with the number of
            # patches that yields extreme aspect ratios.

            hf_config = self.get_hf_config()
            vision_config = hf_config.vision_config
            patch_size = vision_config.patch_size
            merge_size = vision_config.spatial_merge_size
            if max_pixels is None:
                image_processor = self.get_image_processor()
                max_pixels = image_processor.size["longest_edge"]
            unit = patch_size * merge_size
            max_seq_len = max_pixels // (unit * unit)

            def closest_factor_pair(n: int) -> tuple[int, int]:
                # left <= right
                for d in range(math.isqrt(n), 0, -1):
                    if n % d == 0:
                        return d, n // d
                return 1, n

            height_factor, width_factor = 1, max_seq_len
            for seq_len in range(max_seq_len, 0, -1):
                height_factor, width_factor = closest_factor_pair(seq_len)
                if width_factor / height_factor <= 200:
                    break

            return ImageSize(width=unit * width_factor, height=unit * height_factor)

        Qwen2VLProcessingInfo._get_vision_info = _get_vision_info
        Qwen2VLProcessingInfo.get_image_size_with_most_features = get_image_size_with_most_features


def get_vllm_version():
    try:
        vllm_version = parse_version(vllm.__version__)
    except InvalidVersion:
        # for self-compiled vllm,
        # we cannot parse the version, trait it as the lowest version we support
        vllm_version = parse_version("0.8.5")
    return vllm_version


def get_api_server(
    async_llm,
    host: str,
    port: int,
    config: InferenceModelConfig,
    logger: Logger,
):
    vllm_version = get_vllm_version()
    if vllm_version <= parse_version("0.11.0"):
        from trinity.common.models.vllm_patch.api_patch import (
            run_api_server_in_ray_actor,
        )

    elif vllm_version == parse_version("0.12.0"):
        from trinity.common.models.vllm_patch.api_patch_v12 import (
            run_api_server_in_ray_actor_v12 as run_api_server_in_ray_actor,
        )

    else:
        from trinity.common.models.vllm_patch.api_patch_v13 import (
            run_api_server_in_ray_actor_v13 as run_api_server_in_ray_actor,
        )

    logger.info(f"Using vLLM API patch for version {vllm.__version__}")
    return asyncio.create_task(
        run_api_server_in_ray_actor(
            async_llm,
            host=host,
            port=port,
            model_path=config.model_path,  # type: ignore [arg-type]
            logger=logger,
            enable_auto_tool_choice=config.enable_auto_tool_choice,
            tool_call_parser=config.tool_call_parser,
            reasoning_parser=config.reasoning_parser,
            enable_log_requests=config.enable_log_requests,
            chat_template=config.chat_template,
        )
    )
