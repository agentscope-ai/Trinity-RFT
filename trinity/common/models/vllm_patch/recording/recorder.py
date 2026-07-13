"""Engine-level wrap that forces top-k logprobs and records finished turns.

This is the heart of the recording patch. It follows the same instance-level
wrap pattern as ``api_patch_v17.patch_vllm_reasoning_content_alias``:
``functools.wraps`` + a ``__patched_*__`` guard attribute to stay idempotent.

Why wrap ``engine_client.generate`` instead of the serving layer?
  * The serving layer (``OpenAIServingChat``/``OpenAIServingCompletion``) is
    what decides streaming vs non-streaming and what fields to emit. vLLM does
    NOT put ``routed_experts`` into streaming responses (the streaming choice
    schemas omit it), so capturing at the HTTP layer misses it.
  * ``RequestOutput`` / ``CompletionOutput`` carry the full data regardless of
    streaming mode: ``prompt_token_ids``, ``token_ids``, ``logprobs``,
    ``routed_experts`` (raw ndarray). Wrapping at the engine boundary captures
    all four uniformly for chat / completion / responses endpoints.
  * Forcing ``sampling_params.logprobs`` here only affects engine-internal
    computation — the client response is unchanged unless the client itself
    requested logprobs. Recording stays transparent.
"""

import functools
import logging
from types import SimpleNamespace
from typing import Optional

from trinity.buffer.store import MemoryStore, RecordStore
from trinity.common.models.recording.context import (
    get_recording_record_key_from_context,
    get_recording_request_from_context,
)
from trinity.common.models.recording.recorder import (
    MODEL_VERSION_ATTR,
    TRINITY_RECORD_STORE_ATTR,
    TRINITY_RECORDER_ATTR,
    Recorder,
)
from trinity.common.models.vllm_patch.recording.models import build_experience

#: Guard attribute marking the wrapped generate, mirroring api_patch_v17 style.
_PATCHED_FLAG = "__patched_engine_recording__"
#: Force at least this many top-k logprobs per generated token so recording
#: captures the chosen token's logprob even when the caller didn't request
#: logprobs. We store ONLY the sampled token's logprob, and vLLM force-includes
#: the sampled token at ``logprobs=1``, so 1 is the only useful value — no need
#: to thread a knob through the launcher. The engine's ``max_logprobs`` cap
#: (default 1, set at engine build) already covers it.
_RECORDER_LOGPROB_WIDTH = 1
TRINITY_MM_RENDER_ATTR = "trinity_mm_render"


def _get_api_process_rank(engine_client) -> int:
    try:
        return int(engine_client.vllm_config.parallel_config._api_process_rank)
    except Exception:
        return 0


def create_vllm_recorder(
    engine_client,
    logger: logging.Logger,
    *,
    store: Optional[RecordStore] = None,
    enabled: bool = True,
) -> Recorder:
    """Create and install a vLLM-backed recorder on ``engine_client``."""
    existing = getattr(engine_client, TRINITY_RECORDER_ATTR, None)
    if existing is not None:
        return existing

    recorder = Recorder(
        store=store or MemoryStore(),
        build_experiences=build_experience,
        enabled=enabled,
        rank=_get_api_process_rank(engine_client),
        engine_client=engine_client,
    )
    patch_engine_for_recording(engine_client, recorder, logger)
    setattr(engine_client, TRINITY_RECORDER_ATTR, recorder)
    setattr(engine_client, TRINITY_RECORD_STORE_ATTR, recorder.store)
    return recorder


def _get_prompt_arg(args, kwargs):
    if "prompt" in kwargs:
        return kwargs["prompt"]
    if args:
        return args[0]
    return None


def _build_multi_modal_inputs(engine_client, prompt, output, logger: logging.Logger):
    mm_render = getattr(engine_client, TRINITY_MM_RENDER_ATTR, None)
    if mm_render is None:
        logger.warning(
            "Recording saw a possible multimodal vLLM prompt but no %s is attached to engine_client; "
            "recorded Experience will not include multi_modal_inputs.",
            TRINITY_MM_RENDER_ATTR,
        )
        return None
    try:
        if isinstance(prompt, dict):
            multi_modal_data = prompt.get("multi_modal_data")
            if multi_modal_data:
                return mm_render.build_mm_input_for_training(
                    input_ids=output.prompt_token_ids,
                    multi_modal_data=multi_modal_data,
                )
        request_info = get_recording_request_from_context()
        if request_info and request_info.get("messages") is not None:
            return mm_render.build_mm_input_for_training(
                input_ids=output.prompt_token_ids,
                messages=request_info["messages"],
                tools=request_info.get("tools"),
            )
        return None
    except Exception:
        logger.exception("Failed to build multi_modal_inputs for recorded vLLM Experience")
        return None


def _completion_index(completion, fallback: int) -> int:
    return int(getattr(completion, "index", fallback))


def _list_or_empty(value):
    if value is None:
        return []
    return list(value)


def _is_delta_output(sampling_params) -> bool:
    output_kind = getattr(sampling_params, "output_kind", None)
    return getattr(output_kind, "name", output_kind) == "DELTA"


def _concat_routed_experts(prev, cur):
    if cur is None:
        return prev
    if prev is None:
        return cur
    try:
        import numpy as np

        if isinstance(prev, np.ndarray) or isinstance(cur, np.ndarray):
            return np.concatenate([prev, cur], axis=0)
    except Exception:
        pass
    try:
        import torch

        if isinstance(prev, torch.Tensor) or isinstance(cur, torch.Tensor):
            return torch.cat([torch.as_tensor(prev), torch.as_tensor(cur)], dim=0)
    except Exception:
        pass
    try:
        return prev + cur
    except Exception:
        return cur


def _accumulate_request_output(state, output, *, is_delta_output: bool):
    if state is None:
        state = {
            "request_id": output.request_id,
            "prompt_token_ids": _list_or_empty(getattr(output, "prompt_token_ids", None)),
            "prompt": getattr(output, "prompt", None),
            "outputs": {},
            "order": [],
        }
    elif not state["prompt_token_ids"]:
        state["prompt_token_ids"] = _list_or_empty(getattr(output, "prompt_token_ids", None))
        state["prompt"] = state["prompt"] or getattr(output, "prompt", None)

    for fallback_index, completion in enumerate(list(getattr(output, "outputs", None) or [])):
        index = _completion_index(completion, fallback_index)
        if index not in state["outputs"]:
            state["outputs"][index] = {
                "token_ids": [],
                "logprobs": None,
                "text": "",
                "routed_experts": None,
            }
            state["order"].append(index)

        acc = state["outputs"][index]
        cur_token_ids = _list_or_empty(getattr(completion, "token_ids", None))
        if is_delta_output:
            if cur_token_ids:
                acc["token_ids"].extend(cur_token_ids)
        else:
            acc["token_ids"] = cur_token_ids

        cur_logprobs = getattr(completion, "logprobs", None)
        if cur_logprobs is not None:
            cur_logprobs = list(cur_logprobs)
            if not cur_logprobs:
                pass
            elif is_delta_output and acc["logprobs"] is not None:
                acc["logprobs"].extend(cur_logprobs)
            else:
                acc["logprobs"] = cur_logprobs

        cur_text = getattr(completion, "text", None) or ""
        if is_delta_output:
            if cur_text:
                acc["text"] += cur_text
        else:
            acc["text"] = cur_text

        cur_routed_experts = getattr(completion, "routed_experts", None)
        if cur_routed_experts is not None:
            if is_delta_output:
                acc["routed_experts"] = _concat_routed_experts(
                    acc["routed_experts"],
                    cur_routed_experts,
                )
            else:
                acc["routed_experts"] = cur_routed_experts

    return state


def _build_record_output(state, last):
    if state is None:
        return last
    completions = []
    for index in state["order"]:
        acc = state["outputs"][index]
        completions.append(
            SimpleNamespace(
                index=index,
                token_ids=acc["token_ids"],
                logprobs=acc["logprobs"],
                text=acc["text"],
                routed_experts=acc["routed_experts"],
            )
        )
    return SimpleNamespace(
        request_id=state["request_id"],
        prompt_token_ids=state["prompt_token_ids"],
        prompt=state["prompt"],
        outputs=completions,
        finished=getattr(last, "finished", False),
        prompt_routed_experts=getattr(last, "prompt_routed_experts", None),
    )


def patch_engine_for_recording(
    engine_client,
    recorder: "Recorder",
    logger: logging.Logger,
) -> None:
    """Wrap ``engine_client.generate`` in place to record finished turns.

    Instance-level: only this server's engine_client is affected, the global
    class is untouched. Must run before ``init_app_state`` stores the engine
    reference into the serving objects (they hold the same object, so the wrap
    is inherited).

    Args:
        engine_client: The AsyncLLM instance passed into the bootstrap.
        recorder: The ``Recorder`` that will persist turns.
        logger: Logger for the idempotency/confirmation message.

    Raises:
        RuntimeError: If ``engine_client.generate`` is missing (unexpected
            vLLM version drift).
    """
    current = getattr(engine_client, "generate", None)
    if current is None:
        raise RuntimeError("vLLM patch failed: engine_client.generate not found")
    if getattr(current, _PATCHED_FLAG, False):
        return

    @functools.wraps(current)
    async def _patched_generate(*args, **kwargs):
        # generate(prompt, sampling_params, request_id, *, ...).
        # ``engine_client.generate`` assigned as an instance attribute is NOT
        # bound, so ``self`` is absent and args map 1:1 to the protocol.
        sampling_params = kwargs.get("sampling_params")
        if sampling_params is None and len(args) >= 2:
            sampling_params = args[1]
        prompt = _get_prompt_arg(args, kwargs)

        if recorder.enabled and sampling_params is not None:
            # Ensure logprobs are computed for recording (callers may omit
            # them, e.g. on the HTTP path). See _RECORDER_LOGPROB_WIDTH.
            cur = sampling_params.logprobs
            sampling_params.logprobs = (
                max(cur, _RECORDER_LOGPROB_WIDTH) if cur is not None else _RECORDER_LOGPROB_WIDTH
            )
        model_version_start = (
            getattr(engine_client, MODEL_VERSION_ATTR, None) if recorder.enabled else None
        )
        is_delta_output = _is_delta_output(sampling_params)

        last = None
        accumulated = None
        # ``current`` is the original *bound* method captured pre-wrap, so it
        # still resolves ``self`` correctly. Yields RequestOutput unchanged.
        async for out in current(*args, **kwargs):
            last = out
            if recorder.enabled:
                accumulated = _accumulate_request_output(
                    accumulated,
                    out,
                    is_delta_output=is_delta_output,
                )
            yield out

        if recorder.enabled and last is not None and getattr(last, "finished", False):
            # Recover the record key from the request's async context (set by
            # RecordingIdentityMiddleware on the HTTP path, or by VLLMModel.chat
            # on the Ray-direct path). A missing key means the caller did not
            # opt into grouping this turn, so skip recording entirely.
            record_key = get_recording_record_key_from_context()
            if record_key is not None:
                record_output = _build_record_output(accumulated, last)
                multi_modal_inputs = _build_multi_modal_inputs(
                    engine_client,
                    prompt,
                    record_output,
                    logger,
                )
                recorder.schedule_record(
                    record_output,
                    record_key,
                    model_version_start=model_version_start,
                    multi_modal_inputs=multi_modal_inputs,
                )

    setattr(_patched_generate, _PATCHED_FLAG, True)
    engine_client.generate = _patched_generate
    logger.info("Patched vLLM engine_client.generate for generation recording")
