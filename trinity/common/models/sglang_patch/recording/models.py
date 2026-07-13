"""Build Trinity ``Experience`` objects from a finished SGLang ``ret``.

Mirrors ``trinity/common/models/vllm_patch/recording/models.py`` but for the
SGLang output shape. A SGLang ``ret`` is the dict (or list of dicts for ``n > 1``
/ batch) yielded by ``tokenizer_manager.generate_request``. Each item carries
``output_ids``, ``text`` and a ``meta_info`` dict with ``id``, ``prompt_tokens``,
``output_token_logprobs``, ``routed_experts`` and ``weight_version``.

The finished ``ret`` also carries ``prompt_token_ids`` because the recorder
wrapper forces ``obj.return_prompt_token_ids = True`` (see ``recorder.py``), so
the recorded Experience gets the real prompt tokens without reconstructing them
from the request.

Field mapping (SGLang ``ret`` -> ``Experience``):
  meta_info.id        -> eid.suffix  (traceability)
  record_key          -> eid.batch/task/run  (the MemoryStore group key)
  sample index        -> info["sample_index"]  (position within the n set)
  prompt_token_ids    -> tokens (prompt) + prompt_length
  output_ids          -> tokens (response)
  output_token_logprobs -> Experience.logprobs (flat ``[resp_length]``; SGLang
                      returns ``(logprob, *_)`` tuples per token)
  routed_experts      -> Experience.routed_experts (uint8 tensor, decoded with
                      the model's ``(num_layers, topk)`` layout when base64-str)
  meta_info.weight_version -> info["model_version"]
"""

from typing import Any, List, Optional, Tuple

import torch

from trinity.buffer.store import parse_record_key
from trinity.common.experience import EID, Experience
from trinity.common.models.sglang_model import decode_sglang_routed_experts


def _extract_output_logprobs(meta_info: dict) -> List[float]:
    """Pull the chosen-token logprob at each response position.

    SGLang ``output_token_logprobs`` is a list of ``(logprob, *_)`` tuples (one
    per generated token). Mirrors ``SGLangRolloutModel._extract_output_logprobs``.
    """
    output_token_logprobs = meta_info.get("output_token_logprobs") or []
    return [float(logprob) for logprob, *_ in output_token_logprobs]


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
    routed_experts_value: Any,
    total_tokens: int,
    routed_experts_layout: Optional[Tuple[int, int]],
) -> Optional[torch.Tensor]:
    if routed_experts_value is None:
        return None
    if isinstance(routed_experts_value, str):
        if routed_experts_layout is None:
            return None
        return decode_sglang_routed_experts(
            routed_experts_value,
            total_tokens,
            layout=routed_experts_layout,
        )
    return torch.tensor(routed_experts_value, dtype=torch.uint8)


def build_sglang_experience(
    ret: Any,
    record_key: Optional[str],
    *,
    timestamp: str,
    model_version: Optional[Any] = None,
    model_version_start: Optional[Any] = None,
    include_routed_experts: bool = True,
    routed_experts_layout: Optional[Tuple[int, int]] = None,
) -> List[Experience]:
    """Build Trinity ``Experience`` objects from a finished SGLang ``ret``.

    One experience per output (``n > 1`` / batch is captured in full). Each
    carries ``record_key`` in ``eid.batch/task/run`` and shares
    ``eid.suffix = meta_info.id``; ``info["sample_index"]`` distinguishes
    samples within the group.

    Args:
        ret: A finished SGLang result — a dict, or a list of dicts for ``n > 1``
            / batch. Each dict has ``output_ids``/``text``/``meta_info`` and
            (when the wrapper forced ``return_prompt_token_ids``) a
            ``prompt_token_ids`` list.
        record_key: The recording identity (Authorization bearer / record key);
            the MemoryStore group key.
        timestamp: UTC ISO-8601 string (caller-stamped to keep this pure).
        model_version: Checkpoint version fallback; overridden by
            ``meta_info.weight_version`` when present.
        model_version_start: Checkpoint version captured when this generation
            entered the rollout engine. Used to compute
            ``info["model_version_drift"]``.
        include_routed_experts: Whether routed experts should be copied.
        routed_experts_layout: ``(num_layers, topk)`` for decoding base64-str
            routed experts (from ``BaseInferenceModel._get_routed_experts_layout``).

    Returns:
        One ``Experience`` per non-degenerate output. Empty list if the request
        had no prompt tokens or no output with response tokens.
    """
    ret_list = ret if isinstance(ret, list) else [ret]
    if not ret_list:
        return []

    experiences: List[Experience] = []
    for sample_index, item in enumerate(ret_list):
        if not isinstance(item, dict):
            continue
        meta_info = item.get("meta_info") or {}
        prompt_token_ids = list(item.get("prompt_token_ids") or [])
        if not prompt_token_ids:
            # No prompt tokens captured (return_prompt_token_ids not honored);
            # cannot build a valid single-turn Experience.
            continue

        response_token_ids = list(item.get("output_ids") or [])
        if not response_token_ids:
            # Fall back to re-encoding text if the engine omitted output_ids.
            response_text = item.get("text") or ""
            if response_text:
                # The recorder runs in-process but has no tokenizer handle here;
                # output_ids should normally be present, so just skip otherwise.
                response_token_ids = []
            if not response_token_ids:
                continue

        prompt_length = int(meta_info.get("prompt_tokens") or len(prompt_token_ids))
        # Guard against an inconsistent count: prefer the real token list length.
        if prompt_length <= 0 or prompt_length > len(prompt_token_ids):
            prompt_length = len(prompt_token_ids)

        response_logprobs = torch.tensor(
            _extract_output_logprobs(meta_info),
            dtype=torch.float32,
        )

        routed_experts = None
        if include_routed_experts:
            routed_experts = _extract_routed_experts(
                meta_info.get("routed_experts"),
                total_tokens=len(prompt_token_ids) + len(response_token_ids),
                routed_experts_layout=routed_experts_layout,
            )

        request_id = str(meta_info.get("id") or "")
        resolved_model_version = meta_info.get("weight_version")
        if resolved_model_version is None:
            resolved_model_version = model_version

        suffix = _sample_suffix(request_id, sample_index, len(ret_list))
        if record_key is None:
            eid = EID(suffix=suffix)
        else:
            batch, task, run = parse_record_key(record_key)
            eid = EID(batch=batch, task=task, run=run, suffix=suffix)
        info = {
            "sample_index": sample_index,
            "timestamp": timestamp,
            "model_version": resolved_model_version,
            "model_version_drift": _model_version_drift(
                model_version_start,
                resolved_model_version,
            ),
        }

        experiences.append(
            Experience(
                eid=eid,
                tokens=torch.tensor(prompt_token_ids + response_token_ids, dtype=torch.int32),
                logprobs=response_logprobs,
                prompt_length=prompt_length,
                prompt_text=item.get("prompt_text"),
                response_text=item.get("text") or "",
                routed_experts=routed_experts,
                info=info,
            )
        )
    return experiences
