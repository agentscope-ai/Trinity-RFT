import asyncio
from logging import Logger

import vllm
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version

from trinity.common.config import InferenceModelConfig


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
