# -*- coding: utf-8 -*-
"""A safe port of the five-component RLCR reward for math tasks."""

import math as _math
import re as _re
from collections.abc import Callable as _Callable
from collections.abc import Mapping as _Mapping
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


def _parse_finite_confidence(payload: str) -> float | None:
    payload = payload.strip()
    if not payload:
        return None
    try:
        confidence = float(payload)
    except (TypeError, ValueError, OverflowError):
        return None
    if not _math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _parse_safe_confidence(response: str) -> float | None:
    """Parse the final closed confidence even if the full terminal chain is invalid."""
    if not isinstance(response, str):
        return None

    open_tag = "<confidence>"
    close_tag = "</confidence>"
    last_open = response.rfind(open_tag)
    last_close = response.rfind(close_tag)
    if last_open < 0 or last_close < 0 or last_open > last_close:
        return None

    payload_start = last_open + len(open_tag)
    payload = response[payload_start:last_close]
    if _RESERVED_TAG_RE.search(payload):
        return None
    return _parse_finite_confidence(payload)


def _parse_terminal(response: str) -> tuple[bool, str | None, float | None]:
    """Validate the strict terminal tag chain and return its answer and confidence."""
    if not isinstance(response, str):
        return False, None, None

    tags = list(_RESERVED_TAG_RE.finditer(response))
    open_name: str | None = None
    for tag in tags:
        token = tag.group(0)
        is_close = token.startswith("</")
        name = token[2:-1] if is_close else token[1:-1]
        if is_close:
            if open_name != name:
                return False, None, None
            open_name = None
        else:
            if open_name is not None:
                return False, None, None
            open_name = name
    if open_name is not None or len(tags) < len(_TERMINAL_TAGS):
        return False, None, None

    terminal = tags[-len(_TERMINAL_TAGS) :]
    if tuple(tag.group(0) for tag in terminal) != _TERMINAL_TAGS:
        return False, None, None

    for close_index, next_open_index in ((1, 2), (3, 4), (5, 6)):
        between = response[terminal[close_index].end() : terminal[next_open_index].start()]
        if between.strip():
            return False, None, None
    if response[terminal[-1].end() :].strip():
        return False, None, None

    answer = response[terminal[2].end() : terminal[3].start()]
    confidence_payload = response[terminal[6].end() : terminal[7].start()]
    confidence = _parse_finite_confidence(confidence_payload)
    if confidence is None:
        return False, None, None
    return True, answer, confidence


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

        format_ok, answer, confidence = _parse_terminal(response)
        format_score = float(format_ok)
        accuracy_score = 0.0
        brier_score = 0.0

        if format_ok:
            try:
                if answer is None or confidence is None or truth is None:
                    raise ValueError("answer, confidence, and truth are required for verification")
                parsed_answer = self._answer_parser(answer)
                parsed_truth = self._answer_parser(str(truth))
                accuracy_score = float(bool(self._verifier(parsed_answer, parsed_truth)))
                brier_score = 1.0 - (accuracy_score - confidence) ** 2
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
