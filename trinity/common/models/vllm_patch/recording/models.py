# -*- coding: utf-8 -*-
"""Build Trinity ``Experience`` objects from a finished vLLM ``RequestOutput``.

We record into Trinity's native ``Experience`` struct (see
``trinity.common.experience``) rather than a bespoke record, so captured data
drops straight into Trinity's RL/buffer pipeline without a conversion step.

A single ``RequestOutput`` may carry multiple completions (``n > 1``); we emit
one ``Experience`` per completion so no sample is lost.

Field mapping (captured ``RequestOutput`` fields -> ``Experience``):
  request_id        -> eid.suffix  (``EID(suffix=...)``; the vLLM engine request
                    id == the OpenAI ``response.id``. Kept for traceability;
                    ``eid.batch``/``task``/``run`` and reward are assigned from
                    record key by ``MemoryStore.update`` at consume time.)
  API key / record key -> eid.batch/task/run  (the recording identity; **the
                    group key** the MemoryStore batches experiences by, so a
                    whole reward unit's samples/turns are reward-updated and
                    consumed together.)
  sample index      -> info["sample_index"]  (position within the n-completion
                    set; orders samples/turns inside a record-key group)
  prompt_token_ids  -> tokens (prompt portion) + prompt_length
  response_token_ids-> tokens (response portion)
  logprobs          -> Experience.logprobs  -- but ONLY the *chosen* token's
                    logprob per position (flat ``[resp_length]`` tensor), per
                    the RL convention. vLLM's ``CompletionOutput.logprobs`` is a
                    top-k structure per position; we look up the actually-sampled
                    token id and take its ``.logprob``.
  routed_experts   -> Experience.routed_experts (uint8 tensor, verbatim)
  model_version    -> info["model_version"]  (which checkpoint policy served the
                    turn; read in-actor by the recorder's provider)

Plus bookkeeping (sample_index / timestamp / model_version)
stashed in ``Experience.info`` so it round-trips
with the experience through serialize/deserialize.
"""

from typing import Any, List, Optional

import torch

from trinity.buffer.store import parse_record_key
from trinity.common.experience import EID, Experience
from trinity.common.models.mm_utils import combine_output_token_ids


def _extract_chosen_logprobs(
    sample_logprobs: Any,
    response_token_ids: list[int],
) -> Optional[list[float]]:
    """Pull the sampled token's logprob at each response position.

    vLLM exposes ``CompletionOutput.logprobs`` as either a list of
    ``dict[int, Logprob]`` or the ``FlatLogprobs`` container; both support
    positional indexing returning ``dict[int, Logprob]`` for that position, so
    we treat them uniformly.

    Returns a flat ``[resp_length]`` list of floats, or None when logprobs were
    not requested/computed.

    Note: the sampled token is *always* present at each position's dict. vLLM
    force-includes it as column 0 of the reported set
    (``vllm/v1/worker/gpu/sample/logprob.py:compute_topk_logprobs``), so a
    request with ``sampling_params.logprobs = N`` reports ``{sampled} ∪
    top-N`` — the chosen token is reported even when it ranks beyond N in the
    model's distribution. There is therefore no "sampled token absent from
    top-k" case to handle here: ``pos[tid]`` always resolves, and a length
    mismatch between ``sample_logprobs`` and ``response_token_ids`` cannot
    occur in normal operation (both are indexed per generated token).

    Args:
        sample_logprobs: ``CompletionOutput.logprobs`` (may be None).
        response_token_ids: The generated token ids.

    Returns:
        Flat list of chosen-token logprobs, or None.
    """
    if not sample_logprobs:
        return None
    # One entry per generated token; sampled token is force-included per the
    # note above, so a direct lookup per position is always well-defined.
    return [float(sample_logprobs[i][tid].logprob) for i, tid in enumerate(response_token_ids)]


def _sample_suffix(request_id: str, sample_index: int, num_samples: int) -> str:
    if num_samples <= 1:
        return request_id
    return f"{request_id}:{sample_index}"


def _model_version_drift(start: Optional[Any], end: Optional[Any]) -> int:
    if start is None or end is None:
        return 0
    try:
        return int(end) - int(start)
    except (TypeError, ValueError):
        return 0


def _extract_routed_experts(
    output: Any,
    completion: Any,
    *,
    include_routed_experts: bool,
    include_prompt_routed_experts: bool,
):
    if not include_routed_experts:
        return None

    routed_experts_parts = []
    if include_prompt_routed_experts:
        prompt_routed_experts = getattr(output, "prompt_routed_experts", None)
        if prompt_routed_experts is not None:
            routed_experts_parts.append(torch.as_tensor(prompt_routed_experts, dtype=torch.uint8))

    completion_routed_experts = getattr(completion, "routed_experts", None)
    if completion_routed_experts is not None:
        routed_experts_parts.append(torch.as_tensor(completion_routed_experts, dtype=torch.uint8))

    if not routed_experts_parts:
        return None
    if len(routed_experts_parts) == 1:
        return routed_experts_parts[0]
    return torch.cat(routed_experts_parts, dim=0)


def build_experience(
    output: Any,
    record_key: Optional[str],
    *,
    timestamp: str,
    model_version: Optional[int] = None,
    model_version_start: Optional[Any] = None,
    multi_modal_inputs: Optional[dict] = None,
    prompt_text: Optional[str] = None,
    include_routed_experts: bool = True,
    include_prompt_routed_experts: bool = False,
) -> List[Experience]:
    """Build Trinity ``Experience`` objects from a finished ``RequestOutput``.

    One experience per completion (``output.outputs``), so ``n > 1`` sampling
    is captured in full. Each experience carries ``record_key`` in
    ``eid.batch/task/run`` when provided and shares ``eid.suffix = request_id``;
    ``info["sample_index"]`` distinguishes samples within the group.

    Args:
        output: A ``RequestOutput`` with ``finished == True``.
        record_key: The recording identity (API key / Ray-injected record key);
            stored in ``eid.batch/task/run`` and used as the MemoryStore group key.
        timestamp: UTC ISO-8601 string (caller-stamped to keep this pure).
        model_version: Checkpoint version the serving policy was at; stamped
            into ``info`` for RL attribution (read in-actor by the recorder).
        model_version_start: Checkpoint version captured when this generation
            entered the rollout engine. Used to compute
            ``info["model_version_drift"]``.
        multi_modal_inputs: Optional training-time multimodal tensors aligned
            with the prompt tokens. Response token type ids are appended per
            completion before storing on the ``Experience``.
        prompt_text: Optional prompt text override. Direct model calls can pass
            tokenizer-decoded prompt text when ``RequestOutput.prompt`` is not
            suitable for training records.
        include_routed_experts: Whether routed experts should be copied.
        include_prompt_routed_experts: Whether to prepend prompt routed experts
            to completion routed experts. Direct generate uses this to match
            its full-token training representation.

    Returns:
        One ``Experience`` per non-degenerate completion. Empty list if the
        request had no prompt or no completion with response tokens.
    """
    request_id = output.request_id
    # eid.suffix = request_id for traceability; batch/task/run are assigned
    # from record_key when this Experience is destined for the recording store.

    prompt_token_ids = list(output.prompt_token_ids or [])
    if not prompt_token_ids:
        return []

    completions = list(output.outputs or [])
    if not completions:
        return []

    experiences: List[Experience] = []
    for sample_index, completion in enumerate(completions):
        response_token_ids = list(completion.token_ids or [])
        # A valid single-turn experience needs both a prompt and a response;
        # Experience.__init__ asserts len(tokens) > prompt_length otherwise.
        if not response_token_ids:
            continue

        tokens = prompt_token_ids + response_token_ids
        prompt_length = len(prompt_token_ids)

        chosen_logprobs = _extract_chosen_logprobs(completion.logprobs, response_token_ids)
        routed_experts = _extract_routed_experts(
            output,
            completion,
            include_routed_experts=include_routed_experts,
            include_prompt_routed_experts=include_prompt_routed_experts,
        )

        suffix = _sample_suffix(request_id, sample_index, len(completions))
        if record_key is None:
            eid = EID(suffix=suffix)
        else:
            batch, task, run = parse_record_key(record_key)
            eid = EID(batch=batch, task=task, run=run, suffix=suffix)
        info = {
            "sample_index": sample_index,
            "timestamp": timestamp,
            "model_version": model_version,
            "model_version_drift": _model_version_drift(model_version_start, model_version),
        }

        experiences.append(
            Experience(
                eid=eid,
                tokens=tokens,
                logprobs=chosen_logprobs,
                prompt_length=prompt_length,
                routed_experts=routed_experts,
                prompt_text=prompt_text if prompt_text is not None else output.prompt,
                response_text=getattr(completion, "text", None) or "",
                multi_modal_inputs=combine_output_token_ids(
                    response_token_ids,
                    multi_modal_inputs,
                ),
                info=info,
            )
        )
    return experiences
