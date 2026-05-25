# -*- coding: utf-8 -*-
"""Reward Function with Overlong Reward Shaping described in DAPO (https://arxiv.org/pdf/2503.14476)"""
from typing import Optional

import torch

from trinity.common.rewards.naive_dapo_score import compute_score
from trinity.common.rewards.reward_fn import RewardFn


class MathDAPORewardFn(RewardFn):
    """A reward function that follows the definition in DAPO for math task."""

    def __init__(
        self,
        enable_overlong_penalty: Optional[bool] = None,
        penalty_factor: Optional[float] = None,
        max_response_length: Optional[int] = None,
        cache_length: Optional[int] = None,
    ) -> None:
        """Initialize DAPO math reward settings.

        Args:
            enable_overlong_penalty: Whether to apply overlong response shaping.
            penalty_factor: Magnitude for overlong penalties.
            max_response_length: Maximum allowed response length in tokens.
            cache_length: Soft-penalty transition window in tokens.
        """
        self.enable_overlong_penalty = enable_overlong_penalty
        self.penalty_factor = penalty_factor
        self.max_response_length = max_response_length
        self.cache_length = cache_length

    def __call__(  # type: ignore
        self,
        response: str,
        response_token: torch.Tensor,
        truth: str,
        **kwargs,
    ) -> dict[str, float]:
        """Compute DAPO reward components for one response.

        Args:
            response: Model-generated response text.
            response_token: Response token ids.
            truth: Ground-truth answer string.
            **kwargs: Extra arguments for compatibility with reward API.

        Returns:
            dict[str, float]: Reward components containing accuracy and format_score.
        """
        correct = compute_score(response, truth) >= 0.5
        # DAPO paper (Sec. 2.4): +1 / -1 rule-based outcome reward
        accuracy_score = 1.0 if correct else -1.0

        format_score = 0.0

        if self.enable_overlong_penalty:
            format_score = self.compute_overlong_penalty(response_token)

        return {
            "accuracy": accuracy_score,
            "format_score": format_score,
        }

    def compute_overlong_penalty(self, response_token):
        """Compute soft/hard penalty for long responses.

        Args:
            response_token: Response token ids.

        Returns:
            float: Length-based shaping value, where negative values penalize overlong outputs.
        """
        assert (
            self.max_response_length is not None
            and self.cache_length is not None
            and self.penalty_factor is not None
        ), "When enable_overlong_penalty = true, max_response_length, penalty_factor, cache_length must be set"
        assert (
            self.max_response_length > self.cache_length
        ), "max_response_length must be greater than cache_length"

        response_len = len(response_token)
        expected_len = self.max_response_length - self.cache_length

        if response_len < expected_len:
            return 0.0
        elif response_len > self.max_response_length:
            return -self.penalty_factor
        else:
            return (expected_len - response_len) / self.cache_length * self.penalty_factor
