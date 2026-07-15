# -*- coding: utf-8 -*-
"""Reward functions for RFT"""

from trinity.common.rewards.reward_fn import RewardFn
from trinity.utils.registry import Registry

REWARD_FUNCTIONS = Registry(
    "reward_functions",
    default_mapping={
        "math_reward": "trinity.common.rewards.math_reward.MathRewardFn",
        "math_boxed_reward": "trinity.common.rewards.math_reward.MathBoxedRewardFn",
        "format_reward": "trinity.common.rewards.format_reward.FormatReward",
        "countdown_reward": "trinity.common.rewards.countdown_reward.CountDownRewardFn",
        "accuracy_reward": "trinity.common.rewards.accuracy_reward.AccuracyReward",
        "math_dapo_reward": "trinity.common.rewards.dapo_reward.MathDAPORewardFn",
        "rlcr_reward": "trinity.common.rewards.rlcr_reward.RLCRRewardFn",
    },
)

__all__ = [
    "RewardFn",
    "REWARD_FUNCTIONS",
]
