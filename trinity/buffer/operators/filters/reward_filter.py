from typing import List, Tuple

import numpy as np

from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.metrics import aggregate_metrics


class RewardFilter(ExperienceOperator):
    """
    Filter experiences based on the reward value.

    Note: This filter assumes that the reward is already calculated and stored in the Experience object.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        """Filter experiences based on reward value."""
        filtered_exps = [exp for exp in exps if exp.reward >= self.threshold]  # type: ignore [operator]
        metrics = {"filtered_count": len(exps) - len(filtered_exps)}
        return filtered_exps, metrics


class RewardSTDFilter(ExperienceOperator):
    """
    Filter experiences based on the standard deviation of rewards within each group.

    Note: This filter assumes that the reward is already calculated and stored in the Experience object.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        """Filter experiences based on reward std."""
        metrics: dict[str, float | int] = {}
        result_exps = []
        original_count = len(exps)
        grouped_experiences = group_by(exps, id_type="task")
        metrics_list = []
        for _, group_exps in grouped_experiences.items():
            if len(group_exps) < 2:
                continue
            rewards = [exp.reward for exp in group_exps]
            variance = np.std(rewards)
            metrics_list.append(
                {
                    "reward_mean": np.mean(rewards),
                    "reward_std": variance,
                }
            )
            if variance <= self.threshold:
                continue
            result_exps.extend(group_exps)
        final_count = len(result_exps)
        metrics["operator_filtered_count"] = original_count - final_count
        metrics.update(aggregate_metrics(metrics_list, "origin_group_advantages"))
        return result_exps, metrics


class InvalidRewardFilter(ExperienceOperator):
    """
    Filters out experiences with invalid reward values.

    Note: This operator assumes that rewards are already computed and stored in the
    Experience object.Any experience with a missing (`None`) or invalid (`NaN`)
    reward is removed to prevent low-quality data from entering the training
    pipeline.
    """

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        kept = [e for e in exps if e.reward is not None and e.reward == e.reward]

        return kept, {"filtered_count": len(exps) - len(kept)}
