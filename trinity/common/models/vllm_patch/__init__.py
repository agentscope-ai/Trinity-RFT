import asyncio
import json
from logging import Logger

import vllm
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version

from trinity.common.config import InferenceModelConfig

VLLM_VERSION_0120 = parse_version("0.12.0")
VLLM_VERSION_0170 = parse_version("0.17.0")
VLLM_VERSION_0230 = parse_version("0.23.0")


def vllm_patch():
    import transformers

    # Patch for Kimi-VL-A3B-Thinking
    if not hasattr(transformers.activations, "PytorchGELUTanh"):
        transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh

    trf_version = parse_version(transformers.__version__)
    vllm_version = parse_version(vllm.__version__)
    if trf_version >= parse_version("5.0.0"):
        if vllm_version < parse_version("0.16.0"):
            raise ImportError("Please upgrade vllm to 0.16.0 or above to use transformers>=5.0.0.")

        if vllm_version < parse_version("0.19.1"):
            from transformers.configuration_utils import PreTrainedConfig

            original_init = PreTrainedConfig.__init__

            def new_init(self, *args, **kwargs):
                if "ignore_keys_at_rope_validation" in kwargs:
                    kwargs["ignore_keys_at_rope_validation"] = set(
                        kwargs["ignore_keys_at_rope_validation"]
                    )
                original_init(self, *args, **kwargs)

            PreTrainedConfig.__init__ = new_init
    if parse_version("0.20.0") <= vllm_version:
        # TODO: add upper bound when following PR is merged
        # https://github.com/vllm-project/vllm/pull/39772/changes
        from vllm.tool_parsers.qwen3coder_tool_parser import (
            FunctionCall,
            Qwen3CoderToolParser,
            ToolCall,
            find_tool_properties,
            logger,
        )

        if getattr(Qwen3CoderToolParser, "_is_patched", None) is None:

            def new_parse_xml_function_call(self, function_call_str: str) -> ToolCall | None:
                # Extract function name
                end_index = function_call_str.find(">")
                # If there's no ">" character, this is not a valid xml function call
                if end_index == -1:
                    return None
                function_name = function_call_str[:end_index]
                param_config = find_tool_properties(self.tools, function_name)
                parameters = function_call_str[end_index + 1 :]
                param_dict = {}
                for match_text in self.tool_call_parameter_regex.findall(parameters):
                    idx = match_text.find(">")
                    # Skip malformed parameters missing the name>value separator
                    # (e.g. truncated output) so other valid parameters can still
                    # be parsed.
                    if idx == -1:
                        logger.warning(
                            "Skipping malformed parameter without '>' separator "
                            "in tool call for function '%s': %r",
                            function_name,
                            match_text,
                        )
                        continue
                    param_name = match_text[:idx]
                    param_value = str(match_text[idx + 1 :])
                    # Remove prefix and trailing \n
                    if param_value.startswith("\n"):
                        param_value = param_value[1:]
                    if param_value.endswith("\n"):
                        param_value = param_value[:-1]

                    param_dict[param_name] = self._convert_param_value(
                        param_value, param_name, param_config, function_name
                    )
                return ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)
                    ),
                )

            Qwen3CoderToolParser._is_patched = True
            Qwen3CoderToolParser._parse_xml_function_call = new_parse_xml_function_call


def get_vllm_version():
    try:
        vllm_version = parse_version(vllm.__version__)
    except InvalidVersion:
        # for self-compiled vllm,
        # we cannot parse the version, trait it as the lowest version we support
        vllm_version = parse_version("0.8.5")
    return vllm_version


def _get_api_server_runner(vllm_version, *, recording: bool = False):
    if vllm_version == VLLM_VERSION_0120:
        from trinity.common.models.vllm_patch.api_patch_v12 import (
            run_api_server_in_ray_actor_v12,
        )

        return run_api_server_in_ray_actor_v12

    if VLLM_VERSION_0120 < vllm_version < VLLM_VERSION_0170:
        from trinity.common.models.vllm_patch.api_patch_v13 import (
            run_api_server_in_ray_actor_v13,
        )

        return run_api_server_in_ray_actor_v13

    if VLLM_VERSION_0170 <= vllm_version:
        # When generation recording is on, use the recording-enabled entry
        # point (a superset of api_patch_v17 that wraps engine.generate and
        # writes Experiences to the shared store). Otherwise stock api_patch_v17.
        if recording:
            from trinity.common.models.vllm_patch.recording import (
                run_api_server_with_recording as _recording_runner,
            )

            return _recording_runner

        from trinity.common.models.vllm_patch.api_patch_v17 import (
            run_api_server_in_ray_actor_v17,
        )

        return run_api_server_in_ray_actor_v17

    raise ValueError(
        f"Unsupported vLLM version: {vllm.__version__}. "
        "This patch supports vLLM 0.12.0 and above. Please upgrade vLLM to use the API server."
    )


def get_api_server(
    async_llm,
    host: str,
    port: int,
    config: InferenceModelConfig,
    logger: Logger,
):
    vllm_version = get_vllm_version()
    # Recording is driven by the config field (not env, not an engine attr):
    # when on, ``get_api_server`` selects the recording-enabled runner, which
    # wires the in-process MemoryStore + engine wrap itself — no static config
    # needs threading (the logprob width is a recorder-internal constant, and
    # the checkpoint version is read live off the engine).
    recording = bool(config.enable_history) and vllm_version >= VLLM_VERSION_0230
    if config.enable_history and not recording:
        logger.warning(
            "enable_history is on but vLLM %s < 0.23.0; recording disabled",
            vllm.__version__,
        )

    run_api_server_in_ray_actor = _get_api_server_runner(vllm_version, recording=recording)
    logger.info(f"Using vLLM API patch for version {vllm.__version__}")
    kwargs = dict(
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
    # The dynamic checkpoint version is read live off the engine instance
    # (``async_llm.trinity_model_version``, mirrored by VLLMModel), so it is
    # not part of any static config passed here.
    return asyncio.create_task(run_api_server_in_ray_actor(async_llm, **kwargs))
