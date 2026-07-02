"""Prefix-based merging for recorded multi-turn experiences."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

from trinity.buffer.store import RecordStore, get_sample_id
from trinity.common.experience import Experience

MAX_HEADS_PER_STREAM = 128


class PrefixExperienceMerger:
    """Merge same-record experiences whose tokens form a strict prefix chain.

    Strategy:
      * Experiences are grouped by record key and a best-effort sample stream
        key (sample_index, then default).
      * Each stream tracks multiple latest/longest heads so interleaved
        branches sharing one record/sample stream do not evict each other.
      * A length index tries longer heads first; exact token prefix comparison
        remains the source of truth.
      * If no cached head exists yet, the store is scanned once to seed the
        stream cache from previously appended experiences.
    """

    def __init__(self, store: RecordStore) -> None:
        self.store = store
        self._heads: dict[str, dict[tuple[str, Any], _StreamHeads]] = {}

    def try_merge(self, record_key: str, exp: Experience) -> bool:
        stream_key = _sample_stream_key(exp)
        heads = self._heads.setdefault(record_key, {})
        stream_heads = heads.setdefault(stream_key, _StreamHeads())
        candidate = stream_heads.find_longest_prefix(exp)
        if candidate is None and stream_heads.is_empty():
            candidate = _find_longest_prefix_experience(self.store.get(record_key), exp)
            if candidate is not None:
                stream_heads.remember(candidate)
        if candidate is None:
            return False

        old_sample_id = get_sample_id(candidate)
        merged = _merge_prefix_experiences(candidate, exp)
        try:
            self.store.replace(record_key, old_sample_id, merged)
        except KeyError:
            stream_heads.discard_sample_id(old_sample_id)
            return False
        stream_heads.discard_sample_id(old_sample_id)
        stream_heads.remember(merged)
        return True

    def remember(self, record_key: str, exp: Experience) -> None:
        heads = self._heads.setdefault(record_key, {})
        heads.setdefault(_sample_stream_key(exp), _StreamHeads()).remember(exp)

    def forget_record(self, record_key: str) -> None:
        self._heads.pop(record_key, None)


@dataclass
class _HeadEntry:
    exp: Experience
    sequence: int
    signature: tuple[int, ...]


class _StreamHeads:
    """Small in-memory index of possible heads for one record/sample stream."""

    def __init__(self, max_heads: int = MAX_HEADS_PER_STREAM) -> None:
        self.max_heads = max_heads
        self._heads_by_sample_id: dict[str, _HeadEntry] = {}
        self._sample_ids_by_length: dict[int, set[str]] = {}
        self._sample_ids_by_fingerprint: dict[tuple[int, tuple[int, ...]], set[str]] = {}
        self._lengths_desc: list[int] = []
        self._sequence = 0

    def is_empty(self) -> bool:
        return not self._heads_by_sample_id

    def remember(self, exp: Experience) -> None:
        sample_id = get_sample_id(exp)
        self.discard_sample_id(sample_id)
        self._sequence += 1
        length = len(exp.tokens)
        signature = _prefix_signature(exp.tokens, length)
        self._heads_by_sample_id[sample_id] = _HeadEntry(
            exp=exp,
            sequence=self._sequence,
            signature=signature,
        )
        sample_ids = self._sample_ids_by_length.setdefault(length, set())
        if not sample_ids:
            self._insert_length(length)
        sample_ids.add(sample_id)
        self._sample_ids_by_fingerprint.setdefault((length, signature), set()).add(sample_id)
        self._evict_excess_heads()

    def discard_sample_id(self, sample_id: str) -> None:
        entry = self._heads_by_sample_id.pop(sample_id, None)
        if entry is None:
            return
        length = len(entry.exp.tokens)
        fingerprint_key = (length, entry.signature)
        fingerprint_sample_ids = self._sample_ids_by_fingerprint.get(fingerprint_key)
        if fingerprint_sample_ids is not None:
            fingerprint_sample_ids.discard(sample_id)
            if not fingerprint_sample_ids:
                self._sample_ids_by_fingerprint.pop(fingerprint_key, None)
        sample_ids = self._sample_ids_by_length.get(length)
        if sample_ids is None:
            return
        sample_ids.discard(sample_id)
        if not sample_ids:
            self._sample_ids_by_length.pop(length, None)
            self._lengths_desc.remove(length)

    def find_longest_prefix(self, exp: Experience) -> Optional[Experience]:
        exp_length = len(exp.tokens)
        for length in self._lengths_desc:
            if length >= exp_length:
                continue
            signature = _prefix_signature(exp.tokens, length)
            best_entry = None
            sample_ids = self._sample_ids_by_fingerprint.get((length, signature), ())
            for sample_id in sample_ids:
                entry = self._heads_by_sample_id.get(sample_id)
                if entry is None:
                    continue
                if _is_mergeable_turn_prefix(entry.exp, exp):
                    if best_entry is None or entry.sequence < best_entry.sequence:
                        best_entry = entry
            if best_entry is not None:
                return best_entry.exp
        return None

    def _insert_length(self, length: int) -> None:
        index = 0
        while index < len(self._lengths_desc) and self._lengths_desc[index] > length:
            index += 1
        self._lengths_desc.insert(index, length)

    def _evict_excess_heads(self) -> None:
        while len(self._heads_by_sample_id) > self.max_heads:
            shortest_length = self._lengths_desc[-1]
            sample_ids = self._sample_ids_by_length[shortest_length]
            oldest_sample_id = min(
                sample_ids,
                key=lambda sample_id: self._heads_by_sample_id[sample_id].sequence,
            )
            self.discard_sample_id(oldest_sample_id)


def _find_longest_prefix_experience(
    existing: Sequence[Experience],
    exp: Experience,
) -> Optional[Experience]:
    best_candidate = None
    best_length = -1
    for candidate in existing:
        candidate_length = len(candidate.tokens)
        if candidate_length <= best_length:
            continue
        if not _same_sample_stream(candidate, exp):
            continue
        if _is_mergeable_turn_prefix(candidate, exp):
            best_candidate = candidate
            best_length = candidate_length
    return best_candidate


def _same_sample_stream(left: Experience, right: Experience) -> bool:
    return _sample_stream_key(left) == _sample_stream_key(right)


def _prefix_signature(tokens: torch.Tensor, length: int) -> tuple[int, ...]:
    """Return a cheap, collision-tolerant signature for ``tokens[:length]``.

    This only narrows candidates. ``_is_strict_token_prefix`` still performs the
    exact comparison before any merge.
    """
    if length <= 0:
        return ()
    positions = {
        0,
        length // 3,
        (2 * length) // 3,
        max(0, length - 4),
        max(0, length - 3),
        max(0, length - 2),
        length - 1,
    }
    return tuple(int(tokens[position].item()) for position in sorted(positions))


def _sample_stream_key(exp: Experience) -> tuple[str, Any]:
    info = exp.info or {}
    sample_index = info.get("sample_index")
    if sample_index is not None:
        return ("sample_index", sample_index)

    return ("default", 0)


def _is_strict_token_prefix(prefix: torch.Tensor, tokens: torch.Tensor) -> bool:
    prefix_len = len(prefix)
    if prefix_len == 0 or prefix_len >= len(tokens):
        return False
    if prefix.device == tokens.device:
        return bool(torch.equal(prefix.detach(), tokens[:prefix_len].detach()))
    return bool(torch.equal(prefix.detach().cpu(), tokens[:prefix_len].detach().cpu()))


def _is_mergeable_turn_prefix(prefix_exp: Experience, final_exp: Experience) -> bool:
    prefix_len = len(prefix_exp.tokens)
    if prefix_len > final_exp.prompt_length:
        return False
    return _is_strict_token_prefix(prefix_exp.tokens, final_exp.tokens)


def _merge_prefix_experiences(prefix_exp: Experience, final_exp: Experience) -> Experience:
    prefix_len = len(prefix_exp.tokens)
    final_prompt_length = final_exp.prompt_length
    assert final_prompt_length >= prefix_len
    gap_len = final_prompt_length - prefix_len
    final_response_len = len(final_exp.tokens) - final_prompt_length

    prefix_action_mask = _response_action_mask(prefix_exp)
    final_source_mask = _response_action_mask(final_exp)
    final_action_mask = (
        final_source_mask[-final_response_len:] if final_response_len else final_source_mask[:0]
    )
    if gap_len:
        action_mask = torch.cat(
            [
                prefix_action_mask,
                torch.zeros(gap_len, dtype=torch.bool, device=prefix_action_mask.device),
                final_action_mask,
            ]
        )
    else:
        action_mask = torch.cat([prefix_action_mask, final_action_mask])

    logprobs = _merge_logprobs(prefix_exp, final_exp, gap_len, final_response_len)
    routed_experts = _merge_routed_experts(prefix_exp, final_exp, gap_len, final_response_len)
    info = _merge_info(prefix_exp, final_exp)

    return Experience(
        eid=final_exp.eid,
        tokens=final_exp.tokens,
        logprobs=logprobs,
        reward=final_exp.reward,
        token_level_reward=final_exp.token_level_reward,
        advantages=final_exp.advantages,
        returns=final_exp.returns,
        truncate_status=final_exp.truncate_status or prefix_exp.truncate_status,
        info=info,
        metrics=final_exp.metrics,
        prompt_length=prefix_exp.prompt_length,
        response_text=final_exp.response_text,
        prompt_text=prefix_exp.prompt_text,
        action_mask=action_mask,
        messages=final_exp.messages or prefix_exp.messages,
        tools=final_exp.tools or prefix_exp.tools,
        multi_modal_inputs=final_exp.multi_modal_inputs,
        teacher_logprobs=final_exp.teacher_logprobs,
        routed_experts=routed_experts,
        custom_fields=final_exp.custom_fields,
    )


def _response_action_mask(exp: Experience) -> torch.Tensor:
    response_len = len(exp.tokens) - exp.prompt_length
    if exp.action_mask is None:
        return torch.ones(response_len, dtype=torch.bool)
    return exp.action_mask.to(dtype=torch.bool)


def _merge_logprobs(
    prefix_exp: Experience,
    final_exp: Experience,
    gap_len: int,
    final_response_len: int,
) -> Optional[torch.Tensor]:
    if prefix_exp.logprobs is None or final_exp.logprobs is None:
        return None
    parts = [prefix_exp.logprobs]
    if gap_len:
        parts.append(
            torch.zeros(
                gap_len,
                dtype=prefix_exp.logprobs.dtype,
                device=prefix_exp.logprobs.device,
            )
        )
    parts.append(
        final_exp.logprobs[-final_response_len:] if final_response_len else final_exp.logprobs[:0]
    )
    return torch.cat(parts)


def _merge_routed_experts(
    prefix_exp: Experience,
    final_exp: Experience,
    gap_len: int,
    final_response_len: int,
) -> Optional[torch.Tensor]:
    prefix_routed = _response_routed_experts(prefix_exp)
    final_routed = _response_routed_experts(final_exp)
    if prefix_routed is None or final_routed is None:
        return None
    parts = [prefix_routed]
    if gap_len:
        parts.append(
            torch.zeros(
                (gap_len, *prefix_routed.shape[1:]),
                dtype=prefix_routed.dtype,
                device=prefix_routed.device,
            )
        )
    parts.append(final_routed[-final_response_len:] if final_response_len else final_routed[:0])
    return torch.cat(parts, dim=0)


def _response_routed_experts(exp: Experience) -> Optional[torch.Tensor]:
    routed = exp.routed_experts
    if routed is None:
        return None
    response_len = len(exp.tokens) - exp.prompt_length
    if len(routed) == response_len:
        return routed
    # Full-sequence routing is aligned to next-token predictions:
    # token i uses routing row i - 1, so response tokens start at prompt_length - 1.
    if len(routed) == len(exp.tokens) - 1:
        return routed[exp.prompt_length - 1 :]
    return None


def _merge_info(prefix_exp: Experience, final_exp: Experience) -> dict:
    info = dict(final_exp.info or {})

    merged_eid_suffixes = list((prefix_exp.info or {}).get("merged_eid_suffixes") or [])
    for suffix in (prefix_exp.eid.suffix, final_exp.eid.suffix):
        if suffix not in merged_eid_suffixes:
            merged_eid_suffixes.append(suffix)
    info["merged_eid_suffixes"] = merged_eid_suffixes

    merged_sample_ids = list((prefix_exp.info or {}).get("merged_sample_ids") or [])
    for sample_id in (get_sample_id(prefix_exp), get_sample_id(final_exp)):
        if sample_id not in merged_sample_ids:
            merged_sample_ids.append(sample_id)
    info["merged_sample_ids"] = merged_sample_ids
    info["merged_turn_count"] = int((prefix_exp.info or {}).get("merged_turn_count") or 1) + 1
    return info
