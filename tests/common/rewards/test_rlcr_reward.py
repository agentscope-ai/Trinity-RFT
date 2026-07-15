# -*- coding: utf-8 -*-
"""Behavioral contract tests for the RLCR reward function."""

import math

import pytest

from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.rewards.reward_fn import RewardFn
from trinity.common.rewards.rlcr_reward import RLCRRewardFn


def _identity(value: str) -> str:
    return value.strip()


def _equal(answer: str, truth: str) -> bool:
    return answer == truth


def _reward(**kwargs) -> RLCRRewardFn:
    return RLCRRewardFn(answer_parser=_identity, verifier=_equal, **kwargs)


def _response(
    answer: str = "42",
    confidence: str = "0.8",
    *,
    prefix: str = "",
    separator: str = "",
) -> str:
    return (
        f"{prefix}<think>work</think>{separator}"
        f"<answer>{answer}</answer>{separator}"
        f"<analysis>check</analysis>{separator}"
        f"<confidence>{confidence}</confidence>"
    )


def _assert_all_zero(scores: dict[str, float]) -> None:
    assert scores == {
        "format": 0.0,
        "accuracy": 0.0,
        "brier": 0.0,
        "mean_confidence": 0.0,
        "confidence_one_or_zero": 0.0,
    }


def test_registry_resolves_rlcr_lazily() -> None:
    reward_cls = REWARD_FUNCTIONS.get("rlcr_reward")

    assert reward_cls is RLCRRewardFn
    assert issubclass(reward_cls, RewardFn)


def test_correct_answer_returns_five_weighted_components() -> None:
    scores = _reward()(response=_response(), truth="42")

    assert scores == pytest.approx(
        {
            "format": 0.5,
            "accuracy": 0.5,
            "brier": 0.5 * (1.0 - (1.0 - 0.8) ** 2),
            "mean_confidence": 1e-5 * 0.8,
            "confidence_one_or_zero": 0.0,
        }
    )
    assert all(type(value) is float for value in scores.values())
    assert sum(scores.values()) == pytest.approx(1.480008)


def test_wrong_answer_uses_the_same_brier_formula() -> None:
    scores = _reward()(response=_response(answer="41", confidence="0.2"), truth="42")

    assert scores == pytest.approx(
        {
            "format": 0.5,
            "accuracy": 0.0,
            "brier": 0.5 * (1.0 - (0.0 - 0.2) ** 2),
            "mean_confidence": 1e-5 * 0.2,
            "confidence_one_or_zero": 0.0,
        }
    )


def test_default_parser_and_timeout_aware_verifier_path() -> None:
    scores = RLCRRewardFn()(response=_response(answer="42", confidence="1"), truth="42")

    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["brier"] == pytest.approx(0.5)


def test_parser_receives_extracted_answer_payload_and_truth_separately() -> None:
    parser_calls: list[str] = []
    verifier_calls: list[tuple[str, str]] = []

    def parser(value: str) -> str:
        parser_calls.append(value)
        return value.strip()

    def verifier(answer: str, truth: str) -> bool:
        verifier_calls.append((answer, truth))
        return True

    response = _response(
        answer="final",
        confidence="0.7",
        prefix="<answer>decoy</answer><confidence>0.1</confidence>",
    )
    scores = RLCRRewardFn(answer_parser=parser, verifier=verifier)(
        response=response,
        truth="gold",
        source=["code", "math"],
    )

    assert parser_calls == ["final", "gold"]
    assert verifier_calls == [("final", "gold")]
    assert scores["accuracy"] == pytest.approx(0.5)


def test_complete_earlier_reserved_tags_are_allowed_and_terminal_values_win() -> None:
    prefix = (
        "draft <think>old work</think><answer>wrong</answer>"
        "<analysis>old check</analysis><confidence>0.1</confidence> revised: "
    )

    scores = _reward()(response=_response(prefix=prefix, separator=" \n\t"), truth="42")

    assert scores["format"] == pytest.approx(0.5)
    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["mean_confidence"] == pytest.approx(8e-6)


@pytest.mark.parametrize(
    "response",
    [
        "work</think><answer>42</answer><analysis>x</analysis><confidence>0.8</confidence>",
        "<think>x</think><analysis>x</analysis><answer>42</answer><confidence>0.8</confidence>",
        "<think>x</think><answer>42</answer><analysis>x</analysis><confidence>0.8</confidence",
        _response() + " trailing junk",
        "<think>x<answer>nested</answer></think>" + _response(),
        "<think>x</answer><answer>42</think><analysis>x</analysis><confidence>0.8</confidence>",
        "<answer>unclosed " + _response(),
        "<think>x</think><answer>42<analysis>nested</analysis></answer>"
        "<analysis>x</analysis><confidence>0.8</confidence>",
        "<Think>x</Think><answer>42</answer><analysis>x</analysis><confidence>0.8</confidence>",
    ],
    ids=[
        "missing-think-opener",
        "wrong-order",
        "truncated-confidence",
        "trailing-junk",
        "nested-prefix",
        "crossed-tags",
        "unbalanced-prefix",
        "nested-terminal",
        "case-sensitive",
    ],
)
def test_malformed_terminal_chain_fails_the_three_main_components(response: str) -> None:
    scores = _reward()(response=response, truth="42")

    assert scores["format"] == 0.0
    assert scores["accuracy"] == 0.0
    assert scores["brier"] == 0.0


def test_only_trailing_whitespace_is_allowed_after_terminal_chain() -> None:
    scores = _reward()(response=_response() + " \r\n\t", truth="42")

    assert scores["format"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "confidence",
    ["", "not-a-number", "NaN", "Inf", "+inf", "-inf", "-0.01", "1.01"],
)
def test_invalid_confidence_zeros_every_component(confidence: str) -> None:
    scores = _reward()(response=_response(confidence=confidence), truth="42")

    _assert_all_zero(scores)


def test_safe_confidence_still_drives_micros_when_terminal_format_is_invalid() -> None:
    response = "<answer>42</answer><analysis>x</analysis><confidence>1</confidence>"

    scores = _reward()(response=response, truth="42")

    assert scores == pytest.approx(
        {
            "format": 0.0,
            "accuracy": 0.0,
            "brier": 0.0,
            "mean_confidence": 1e-5,
            "confidence_one_or_zero": 1e-5,
        }
    )


def test_a_later_truncated_confidence_makes_the_micro_confidence_unsafe() -> None:
    response = _response(confidence="0.8") + "<confidence>0.1"

    scores = _reward()(response=response, truth="42")

    _assert_all_zero(scores)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [("0", 1e-5), ("0.009", 1e-5), ("0.01", 0.0), ("0.99", 0.0), ("0.991", 1e-5), ("1", 1e-5)],
)
def test_extreme_confidence_uses_strict_open_thresholds(confidence: str, expected: float) -> None:
    scores = _reward()(response=_response(confidence=confidence), truth="42")

    assert scores["confidence_one_or_zero"] == pytest.approx(expected)


@pytest.mark.parametrize("stage", ["answer", "truth", "verify"])
def test_verification_errors_zero_accuracy_and_brier_without_low_q_reward(stage: str) -> None:
    def parser(value: str) -> str:
        if stage == "answer" and value == "42":
            raise TimeoutError("answer parse timed out")
        if stage == "truth" and value == "gold":
            raise ValueError("truth parse failed")
        return value

    def verifier(answer: str, truth: str) -> bool:
        if stage == "verify":
            raise TimeoutError("verify timed out")
        return answer == truth

    scores = RLCRRewardFn(answer_parser=parser, verifier=verifier)(
        response=_response(answer="42", confidence="0"),
        truth="gold",
    )

    assert scores == pytest.approx(
        {
            "format": 0.5,
            "accuracy": 0.0,
            "brier": 0.0,
            "mean_confidence": 0.0,
            "confidence_one_or_zero": 1e-5,
        }
    )


def test_missing_truth_is_a_verification_error_not_an_incorrect_answer() -> None:
    scores = _reward()(response=_response(confidence="0"), truth=None)

    assert scores["format"] == pytest.approx(0.5)
    assert scores["accuracy"] == 0.0
    assert scores["brier"] == 0.0


def test_custom_weights_are_matched_by_name_not_position() -> None:
    weights = {
        "confidence_one_or_zero": 5.0,
        "mean_confidence": 4.0,
        "brier": 3.0,
        "accuracy": 2.0,
        "format": 1.0,
    }

    scores = _reward(weights=weights)(response=_response(confidence="1"), truth="42")

    assert scores == pytest.approx(
        {
            "format": 1.0,
            "accuracy": 2.0,
            "brier": 3.0,
            "mean_confidence": 4.0,
            "confidence_one_or_zero": 5.0,
        }
    )
    assert sum(scores.values()) == pytest.approx(15.0)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (
            {"format": 1, "accuracy": 1, "brier": 1, "mean_confidence": 1},
            "missing",
        ),
        (
            {
                "format": 1,
                "accuracy": 1,
                "brier": 1,
                "mean_confidence": 1,
                "confidence_one_or_zero": 1,
                "extra": 1,
            },
            "unknown",
        ),
        (
            {
                "format": "0.5",
                "accuracy": 1,
                "brier": 1,
                "mean_confidence": 1,
                "confidence_one_or_zero": 1,
            },
            "numeric",
        ),
        (
            {
                "format": math.nan,
                "accuracy": 1,
                "brier": 1,
                "mean_confidence": 1,
                "confidence_one_or_zero": 1,
            },
            "finite",
        ),
        (
            {
                "format": math.inf,
                "accuracy": 1,
                "brier": 1,
                "mean_confidence": 1,
                "confidence_one_or_zero": 1,
            },
            "finite",
        ),
        (
            {
                "format": True,
                "accuracy": 1,
                "brier": 1,
                "mean_confidence": 1,
                "confidence_one_or_zero": 1,
            },
            "numeric",
        ),
    ],
)
def test_invalid_named_weight_mappings_fail_deterministically(
    weights: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _reward(weights=weights)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_callable", [None, 3, "callable"])
def test_injected_parser_and_verifier_must_be_callable(bad_callable: object) -> None:
    with pytest.raises(TypeError, match="answer_parser"):
        RLCRRewardFn(answer_parser=bad_callable, verifier=_equal)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="verifier"):
        RLCRRewardFn(answer_parser=_identity, verifier=bad_callable)  # type: ignore[arg-type]
