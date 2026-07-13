"""Engine-level wrap that records finished SGLang turns into the shared store.

Mirrors ``trinity/common/models/vllm_patch/recording/recorder.py`` but adapts to
the SGLang output path. vLLM wraps ``engine_client.generate`` (the in-process
engine boundary); SGLang's single convergence point for ``/generate``,
``/v1/chat/completions`` and ``/invocations`` is
``tokenizer_manager.generate_request`` (an async generator yielding ``ret``
dicts), so that is what we wrap here — instance-level, idempotent.

Two adaptations forced by the SGLang shape (see the plan for detail):

1. Trigger on the **finished yield**, not on generator exhaustion. SGLang's
   non-stream ``/generate`` handler pulls exactly one item via ``__anext__()``
   and never exhausts the generator, so a "record after the loop" trigger would
   never fire. We detect finished via ``ret["meta_info"]["finish_reason"] is not
   None`` and ``schedule_record`` *before* yielding that finished ``ret``.

2. Force ``obj.return_logprob = True`` and ``obj.return_prompt_token_ids = True``
   so the finished ``ret`` always carries the chosen-token logprobs and the full
   prompt token ids (the latter is stashed onto ``out_dict`` by
   ``tokenizer_manager`` only when ``return_prompt_token_ids`` is set). This is
   transparent to OpenAI clients (the chat serving layer gates its
   ``prompt_token_ids`` on the separate ``return_token_ids`` flag) and only adds
   an ignored field to ``/generate`` JSON responses.
"""

import functools
import logging
from typing import Any, List, Optional, Tuple

from trinity.buffer.store import MemoryStore, RecordStore
from trinity.common.models.recording.context import (
    get_recording_record_key_from_context,
)
from trinity.common.models.recording.recorder import (
    MODEL_VERSION_ATTR,
    TRINITY_RECORD_STORE_ATTR,
    TRINITY_RECORDER_ATTR,
    Recorder,
)
from trinity.common.models.sglang_patch.recording.models import build_sglang_experience

#: Guard attribute marking the wrapped generate_request (mirrors vLLM's style).
_PATCHED_FLAG = "__patched_sglang_recording__"


def _get_obj(args, kwargs):
    """Extract the ``GenerateReqInput``/``EmbeddingReqInput`` argument.

    ``generate_request(self, obj, request=None)`` is wrapped as an instance
    attribute, so ``self`` is absent and ``args`` map 1:1 to the protocol
    (``obj`` is ``args[0]``; ``obj`` may also be passed as ``obj=``).
    """
    if "obj" in kwargs:
        return kwargs["obj"]
    if args:
        return args[0]
    return None


def _force_record_fields(obj: Any, *, force_routed_experts: bool) -> None:
    """Force logprob + prompt-token-id (+ routed-expert) capture for recording.

    Transparent to clients: the OpenAI serving layer gates its response
    ``logprobs`` / ``prompt_token_ids`` / ``sglext.routed_experts`` emission on
    the *ChatCompletionRequest* flags (unchanged); we only flip the
    ``GenerateReqInput`` flags the scheduler reads, so the recorded ``ret`` gains
    these fields while HTTP responses stay the same.
    """
    if obj is None:
        return
    # return_logprob may be a list for batched requests; broadcast True.
    if hasattr(obj, "return_logprob"):
        cur = getattr(obj, "return_logprob", None)
        if isinstance(cur, list):
            obj.return_logprob = [True] * len(cur)
        else:
            obj.return_logprob = True
    if hasattr(obj, "return_prompt_token_ids"):
        obj.return_prompt_token_ids = True
    # The scheduler only returns routed_experts when the per-request flag is set
    # (scheduler.py: ``if recv_req.return_routed_experts``), even though the
    # model runner computes them whenever the server flag is on. The chat path
    # defaults this to False, so force it here when the server is MoE-enabled
    # (signaled by a non-None routed_experts_layout) so the recorded experience
    # carries routed_experts on every path, not just Ray-direct /generate.
    if force_routed_experts and hasattr(obj, "return_routed_experts"):
        obj.return_routed_experts = True


def _normalize_ret(out: Any) -> List[dict]:
    """A ``ret`` is a dict (n=1) or a list of dicts (n>1 / batch)."""
    if isinstance(out, list):
        return [item for item in out if isinstance(item, dict)]
    if isinstance(out, dict):
        return [out]
    return []


def _is_finished(out: Any) -> bool:
    """True if any output carries a non-None ``finish_reason``.

    ``tokenizer_manager`` sets ``meta_info["finish_reason"] =
    recv_obj.finished_reasons[i]`` and ``finished_reasons[i]`` is ``None`` until
    the request is done, so this is the reliable finished signal.
    """
    for item in _normalize_ret(out):
        meta_info = item.get("meta_info") or {}
        if meta_info.get("finish_reason") is not None:
            return True
    return False


def _monotonic_extend_list(acc: list, cur: Optional[list]) -> list:
    """Replace ``acc`` with ``cur`` if ``cur`` is cumulative (starts with ``acc``
    as a prefix), otherwise extend. Handles both cumulative-streaming chunks
    (each carries the full-so-far ids) and incremental-streaming deltas.
    """
    if not cur:
        return acc
    if not acc:
        return list(cur)
    if len(cur) >= len(acc) and list(cur[: len(acc)]) == acc:
        return list(cur)  # cumulative: cur supersedes acc
    return acc + list(cur)  # delta: extend


def _monotonic_extend_text(acc: str, cur: Optional[str]) -> str:
    """Same discipline for ``text`` (string prefix check). ``cur`` may be ``None``
    for non-incremental intermediate chunks (deferred text)."""
    if not cur:
        return acc
    if not acc:
        return cur
    if cur.startswith(acc):
        return cur  # cumulative
    return acc + cur  # delta


def _accumulate_ret(state: dict, order: list, out: Any) -> Tuple[dict, list]:
    """Merge a yielded ``ret`` into the per-output accumulator.

    ``state`` maps output index -> accumulated fields; ``order`` preserves
    first-seen order so the reconstructed ``ret`` keeps sample indexing.
    """
    items = _normalize_ret(out)
    for idx, item in enumerate(items):
        meta_info = item.get("meta_info") or {}
        if idx not in state:
            state[idx] = {
                "output_ids": [],
                "text": "",
                "output_token_logprobs": [],
                "routed_experts": None,
                "prompt_token_ids": None,
                "meta_info": {},
            }
            order.append(idx)
        acc = state[idx]
        acc["output_ids"] = _monotonic_extend_list(acc["output_ids"], item.get("output_ids"))
        acc["text"] = _monotonic_extend_text(acc["text"], item.get("text"))
        acc["output_token_logprobs"] = _monotonic_extend_list(
            acc["output_token_logprobs"],
            (meta_info.get("output_token_logprobs")),
        )
        routed = meta_info.get("routed_experts")
        if routed is not None:
            acc["routed_experts"] = routed  # latest non-None (final chunk is full)
        prompt_ids = item.get("prompt_token_ids")
        if prompt_ids:
            acc["prompt_token_ids"] = list(prompt_ids)
        # Keep the latest meta_info (carries id/finish_reason/weight_version/
        # prompt_tokens); output_token_logprobs/routed_experts are overridden
        # below from the accumulated fields.
        acc["meta_info"] = meta_info
    return state, order


def _build_ret(state: dict, order: list) -> List[dict]:
    """Reconstruct a finished ``ret`` list from accumulated per-output state."""
    out_list: List[dict] = []
    for idx in order:
        acc = state[idx]
        meta_info = dict(acc["meta_info"])
        # Override with the fully-accumulated fields (streaming deltas merged).
        meta_info["output_token_logprobs"] = acc["output_token_logprobs"]
        if acc["routed_experts"] is not None:
            meta_info["routed_experts"] = acc["routed_experts"]
        out_list.append(
            {
                "output_ids": acc["output_ids"],
                "text": acc["text"],
                "prompt_token_ids": acc["prompt_token_ids"],
                "meta_info": meta_info,
            }
        )
    return out_list


def create_sglang_recorder(
    tokenizer_manager,
    logger: logging.Logger,
    *,
    store: Optional[RecordStore] = None,
    recorder: Optional[Recorder] = None,
    enabled: bool = True,
    routed_experts_layout: Optional[Tuple[int, int]] = None,
) -> Recorder:
    """Create/accept and install a SGLang-backed recorder on ``tokenizer_manager``.

    The caller (``SGLangRolloutModel``) may pre-create and own ``recorder`` so it
    can drain it in-process via ``extract_experience_from_history``; this
    function wires it onto the ``tokenizer_manager`` (engine_client) and patches
    ``generate_request``. Idempotent.
    """
    existing = getattr(tokenizer_manager, TRINITY_RECORDER_ATTR, None)
    if existing is not None:
        return existing

    if recorder is None:
        recorder = Recorder(
            store=store or MemoryStore(),
            build_experiences=build_sglang_experience,
            enabled=enabled,
            engine_client=tokenizer_manager,
        )
    else:
        # The model owns the recorder; let it read model_version off the engine
        # if needed (build_sglang_experience prefers meta_info.weight_version).
        recorder.engine_client = tokenizer_manager
        if store is not None and recorder.store is None:
            recorder.store = store

    patch_tokenizer_manager_for_recording(
        tokenizer_manager, recorder, logger, routed_experts_layout=routed_experts_layout
    )
    setattr(tokenizer_manager, TRINITY_RECORDER_ATTR, recorder)
    setattr(tokenizer_manager, TRINITY_RECORD_STORE_ATTR, recorder.store)
    return recorder


def patch_tokenizer_manager_for_recording(
    tokenizer_manager,
    recorder: "Recorder",
    logger: logging.Logger,
    *,
    routed_experts_layout: Optional[Tuple[int, int]] = None,
) -> None:
    """Wrap ``tokenizer_manager.generate_request`` in place to record turns.

    Instance-level: only this server's tokenizer_manager is affected. Must run
    before the server starts serving (the serving objects hold the same instance,
    so the wrap is inherited).
    """
    current = getattr(tokenizer_manager, "generate_request", None)
    if current is None:
        raise RuntimeError(
            "SGLang recording patch failed: tokenizer_manager.generate_request not found"
        )
    if getattr(current, _PATCHED_FLAG, False):
        return

    @functools.wraps(current)
    async def _patched_generate_request(*args, **kwargs):
        obj = _get_obj(args, kwargs)
        if recorder.enabled and obj is not None:
            _force_record_fields(obj, force_routed_experts=routed_experts_layout is not None)

        state: dict = {}
        order: list = []
        model_version_start = (
            getattr(tokenizer_manager, MODEL_VERSION_ATTR, None) if recorder.enabled else None
        )
        # ``current`` is the original *bound* method captured pre-wrap, so it
        # still resolves ``self`` correctly. Yields each ret unchanged.
        async for out in current(*args, **kwargs):
            if recorder.enabled:
                state, order = _accumulate_ret(state, order, out)
            # Trigger on the finished yield (not on generator exhaustion): the
            # non-stream /generate consumer pulls only once via __anext__().
            if recorder.enabled and _is_finished(out):
                record_key = get_recording_record_key_from_context()
                if record_key is not None and state:
                    reconstructed = _build_ret(state, order)
                    recorder.schedule_record(
                        reconstructed,
                        record_key,
                        model_version_start=model_version_start,
                        include_routed_experts=True,
                        routed_experts_layout=routed_experts_layout,
                    )
            yield out

    setattr(_patched_generate_request, _PATCHED_FLAG, True)
    tokenizer_manager.generate_request = _patched_generate_request
    logger.info("Patched SGLang tokenizer_manager.generate_request for generation recording")
