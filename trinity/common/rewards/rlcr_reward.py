# -*- coding: utf-8 -*-
"""A safe port of the five-component RLCR reward for math tasks."""

import math as _math
import re as _re
from collections.abc import Callable as _Callable
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass
from numbers import Real as _Real
from typing import Any as _Any

from trinity.common.rewards.eval_utils import simple_answer_parser as _simple_answer_parser
from trinity.common.rewards.eval_utils import verify_with_timeout as _verify_with_timeout
from trinity.common.rewards.reward_fn import RewardFn as _RewardFn

__all__ = ["RLCRRewardFn"]

_COMPONENTS = (
    "format",
    "accuracy",
    "brier",
    "mean_confidence",
    "confidence_one_or_zero",
)
_DEFAULT_WEIGHTS = {
    "format": 0.5,
    "accuracy": 0.5,
    "brier": 0.5,
    "mean_confidence": 1e-5,
    "confidence_one_or_zero": 1e-5,
}
_RESERVED_TAG_RE = _re.compile(r"</?(?:think|answer|analysis|confidence)>")
_CONFIDENCE_TAG_RE = _re.compile(r"</?confidence>")
_TERMINAL_TAGS = (
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "<analysis>",
    "</analysis>",
    "<confidence>",
    "</confidence>",
)


@_dataclass(frozen=True)
class _ConfidenceParseResult:
    confidence: float | None
    reason: str | None


@_dataclass(frozen=True)
class _TerminalParseResult:
    ok: bool
    answer: str | None
    confidence: float | None
    reason: str | None


def _terminal_failure(reason: str) -> _TerminalParseResult:
    return _TerminalParseResult(False, None, None, reason)


def _parse_finite_confidence(payload: str) -> _ConfidenceParseResult:
    payload = payload.strip()
    if not payload:
        return _ConfidenceParseResult(None, "confidence_empty")
    try:
        confidence = float(payload)
    except (TypeError, ValueError, OverflowError):
        return _ConfidenceParseResult(None, "confidence_non_numeric")
    if not _math.isfinite(confidence):
        return _ConfidenceParseResult(None, "confidence_non_finite")
    if not 0.0 <= confidence <= 1.0:
        return _ConfidenceParseResult(None, "confidence_out_of_range")
    return _ConfidenceParseResult(confidence, None)


def _scan_confidence(response: str) -> _ConfidenceParseResult:
    """Return the last q only when every confidence tag is balanced and unnested."""
    if not isinstance(response, str):
        return _ConfidenceParseResult(None, "response_not_string")

    open_tag: _re.Match[str] | None = None
    last_payload: str | None = None
    for tag in _CONFIDENCE_TAG_RE.finditer(response):
        if tag.group(0) == "<confidence>":
            if open_tag is not None:
                return _ConfidenceParseResult(None, "nested_tags")
            open_tag = tag
        else:
            if open_tag is None:
                return _ConfidenceParseResult(None, "unbalanced_tags")
            last_payload = response[open_tag.end() : tag.start()]
            open_tag = None

    if open_tag is not None:
        return _ConfidenceParseResult(None, "unbalanced_tags")
    if last_payload is None:
        return _ConfidenceParseResult(None, "missing_tag")
    return _parse_finite_confidence(last_payload)


def _parse_safe_confidence(response: str) -> float | None:
    """Parse the final closed confidence even if the full terminal chain is invalid."""
    return _scan_confidence(response).confidence


def _parse_terminal(response: str) -> _TerminalParseResult:
    """Validate the strict terminal tag chain and return its answer and confidence."""
    if not isinstance(response, str):
        return _terminal_failure("response_not_string")

    tags = list(_RESERVED_TAG_RE.finditer(response))
    open_name: str | None = None
    for tag in tags:
        token = tag.group(0)
        is_close = token.startswith("</")
        name = token[2:-1] if is_close else token[1:-1]
        if is_close:
            if open_name is None:
                return _terminal_failure("unbalanced_tags")
            if open_name != name:
                return _terminal_failure("crossed_tags")
            open_name = None
        else:
            if open_name is not None:
                return _terminal_failure("nested_tags")
            open_name = name
    if open_name is not None:
        return _terminal_failure("unbalanced_tags")
    if len(tags) < len(_TERMINAL_TAGS):
        return _terminal_failure("missing_tag")

    terminal = tags[-len(_TERMINAL_TAGS) :]
    if tuple(tag.group(0) for tag in terminal) != _TERMINAL_TAGS:
        return _terminal_failure("wrong_order")

    for close_index, next_open_index in ((1, 2), (3, 4), (5, 6)):
        between = response[terminal[close_index].end() : terminal[next_open_index].start()]
        if between.strip():
            return _terminal_failure("inter_tag_junk")
    if response[terminal[-1].end() :].strip():
        return _terminal_failure("trailing_junk")

    answer = response[terminal[2].end() : terminal[3].start()]
    confidence_payload = response[terminal[6].end() : terminal[7].start()]
    confidence_result = _parse_finite_confidence(confidence_payload)
    if confidence_result.reason is not None:
        return _terminal_failure(confidence_result.reason)
    return _TerminalParseResult(True, answer, confidence_result.confidence, None)


def _validate_weights(weights: _Mapping[str, _Any] | None) -> dict[str, float]:
    if weights is None:
        return dict(_DEFAULT_WEIGHTS)
    if not isinstance(weights, _Mapping):
        raise TypeError("weights must be a named mapping")

    provided = set(weights)
    expected = set(_COMPONENTS)
    missing = sorted(expected - provided)
    unknown = sorted(provided - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unknown:
            details.append(f"unknown keys: {unknown}")
        raise ValueError("invalid weights mapping; " + "; ".join(details))

    validated: dict[str, float] = {}
    for name in _COMPONENTS:
        value = weights[name]
        if isinstance(value, bool) or not isinstance(value, _Real):
            raise TypeError(f"weight '{name}' must be numeric")
        numeric = float(value)
        if not _math.isfinite(numeric):
            raise ValueError(f"weight '{name}' must be finite")
        validated[name] = numeric
    return validated


class RLCRRewardFn(_RewardFn):
    """Return already weighted RLCR format, accuracy, and calibration components."""

    def __init__(
        self,
        weights: _Mapping[str, _Any] | None = None,
        answer_parser: _Callable[[str], _Any] = _simple_answer_parser,
        verifier: _Callable[[_Any, _Any], bool] = _verify_with_timeout,
    ) -> None:
        if not callable(answer_parser):
            raise TypeError("answer_parser must be callable")
        if not callable(verifier):
            raise TypeError("verifier must be callable")
        self._weights = _validate_weights(weights)
        self._answer_parser = answer_parser
        self._verifier = verifier

    def __call__(  # type: ignore[override]
        self,
        response: str,
        prompt: str | None = None,
        truth: str | None = None,
        **kwargs: _Any,
    ) -> dict[str, float]:
        del prompt, kwargs
        safe_confidence = _parse_safe_confidence(response)
        mean_confidence = safe_confidence if safe_confidence is not None else 0.0
        confidence_one_or_zero = float(
            safe_confidence is not None and (safe_confidence < 0.01 or safe_confidence > 0.99)
        )

        terminal = _parse_terminal(response)
        format_score = float(terminal.ok)
        accuracy_score = 0.0
        brier_score = 0.0

        if terminal.ok:
            try:
                if terminal.answer is None or terminal.confidence is None or truth is None:
                    raise ValueError("answer, confidence, and truth are required for verification")
                parsed_answer = self._answer_parser(terminal.answer)
                parsed_truth = self._answer_parser(str(truth))
                accuracy_score = float(bool(self._verifier(parsed_answer, parsed_truth)))
                brier_score = 1.0 - (accuracy_score - terminal.confidence) ** 2
            except Exception:
                # An infrastructure failure is not evidence that the answer is wrong.
                accuracy_score = 0.0
                brier_score = 0.0

        return {
            "format": float(self._weights["format"] * format_score),
            "accuracy": float(self._weights["accuracy"] * accuracy_score),
            "brier": float(self._weights["brier"] * brier_score),
            "mean_confidence": float(self._weights["mean_confidence"] * mean_confidence),
            "confidence_one_or_zero": float(
                self._weights["confidence_one_or_zero"] * confidence_one_or_zero
            ),
        }
