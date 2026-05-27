from typing import List, Tuple

import numpy as np
import torch

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


class DAPODynamicSamplingFilter(ExperienceOperator):
    """
    DAPO dynamic sampling (arXiv:2503.14476 Sec. 3.2).

    Keeps a task group only when some but not all rollouts are correct:
    0 < |{correct}| < G. Uses outcome accuracy from experience metrics, not
    length-shaped total reward.
    """

    def __init__(
        self,
        metric_key: str = "accuracy",
        correct_threshold: float = 0.0,
    ) -> None:
        """Initialize the dynamic sampling filter.

        Args:
            metric_key: Metric name used to determine rollout correctness.
            correct_threshold: Minimum score treated as correct.
        """
        self.metric_key = metric_key
        self.correct_threshold = correct_threshold

    def _outcome_score(self, exp: Experience) -> float:
        """Extract the outcome score from an experience.

        Args:
            exp: Experience to evaluate.

        Returns:
            float: Outcome score used for correctness decisions.

        Raises:
            ValueError: If neither the configured metric nor reward is available.
        """
        if exp.metrics and self.metric_key in exp.metrics:
            return float(exp.metrics[self.metric_key])
        if exp.reward is not None:
            return float(exp.reward)
        raise ValueError(
            f"Experience missing '{self.metric_key}' in metrics and has no reward."
        )

    def _is_correct(self, exp: Experience) -> bool:
        """Determine whether an experience is correct.

        Args:
            exp: Experience to evaluate.

        Returns:
            bool: True when the outcome score exceeds the threshold.
        """
        return self._outcome_score(exp) > self.correct_threshold

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        """Keep only mixed-correctness groups for DAPO training.

        Args:
            exps: Experiences grouped by task id during filtering.

        Returns:
            Tuple[List[Experience], dict]: Filtered experiences and filtering metrics.
        """
        result_exps = []
        original_count = len(exps)
        dropped_all_correct = 0
        dropped_all_wrong = 0
        grouped_experiences = group_by(exps, id_type="task")
        for _, group_exps in grouped_experiences.items():
            if len(group_exps) < 2:
                continue
            num_correct = sum(1 for exp in group_exps if self._is_correct(exp))
            group_size = len(group_exps)
            if num_correct == 0:
                dropped_all_wrong += group_size
                continue
            if num_correct == group_size:
                dropped_all_correct += group_size
                continue
            result_exps.extend(group_exps)
        metrics = {
            "filtered_count": original_count - len(result_exps),
            "dropped_all_correct": dropped_all_correct,
            "dropped_all_wrong": dropped_all_wrong,
        }
        return result_exps, metrics


class MaskResponseTruncatedOperator(ExperienceOperator):
    """
    DAPO overlong filtering stage 1 (Sec. 3.4): exclude truncated responses from loss.

    Zeros action_mask so truncated rollouts do not contribute to the policy gradient.
    """

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        """Mask action positions for truncated responses.

        Args:
            exps: Experiences to process.

        Returns:
            Tuple[List[Experience], dict]: Original experiences and masking metrics.
        """
        masked_count = 0
        for exp in exps:
            if exp.truncate_status == "response_truncated" and exp.action_mask is not None:
                exp.action_mask = torch.zeros_like(exp.action_mask, dtype=torch.bool)
                masked_count += 1
        return exps, {"masked_truncated_count": masked_count}


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
