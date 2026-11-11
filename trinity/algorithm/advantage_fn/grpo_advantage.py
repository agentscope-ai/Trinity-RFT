"""GRPO advantage computation
"""

import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn, GroupAdvantage
from trinity.common.experience import Experience, group_by
from trinity.utils.annotations import Deprecated
from trinity.utils.metrics import aggregate_metrics


@Deprecated
class GRPOAdvantageFn(AdvantageFn):
    """GRPO advantage computation"""

    def __init__(
        self,
        epsilon: float = 1e-6,
    ) -> None:
        self.epsilon = epsilon

    def __call__(
        self,
        exps: "DataProto",
        **kwargs,
    ) -> Tuple["DataProto", Dict]:
        """
        Compute advantage for GRPO, operating only on Outcome reward
        (with only one scalar reward for each response).
        Ref: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py

            token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            eos_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            scores: `(torch.Tensor)`
                shape: (bs, response_length)
        """
        token_level_rewards = exps.batch["token_level_rewards"]
        eos_mask = exps.batch["response_mask"]
        index = exps.non_tensor_batch["uid"]
        epsilon = self.epsilon

        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx], dtype=torch.float32))
                    id2std[idx] = torch.std(torch.tensor(id2score[idx], dtype=torch.float32))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        exps.batch["advantages"] = scores
        exps.batch["returns"] = scores

        metrics = {
            # TODO: add meaningful metrics
        }

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 1e-6,
        }


class GRPOGroupedAdvantage(GroupAdvantage):
    """An advantage class that calculates GRPO advantages."""

    def __init__(
        self,
        epsilon: float = 1e-6,
        std_threshold: Optional[float] = None,
        duplicate_experiences: bool = False,
        rank_penalty: Optional[float] = None,
        std_cal_level: str = "group",  # "group" or "batch"
    ) -> None:
        """Initialize the GRPO advantage function.

        Args:
            epsilon (float): A small value to avoid division by zero.
            std_threshold (Optional[float]): If provided, groups with a reward standard deviation equal
                or below this threshold will be skipped.
            duplicate_experiences (bool): If True, allows duplicate experiences to keep the original experience
                count. Only used when `std_threshold` is not None (https://hkunlp.github.io/blog/2025/Polaris).
            rank_penalty (Optional[float]): A penalty applied to the rank of rewards to correct for bias
                (https://arxiv.org/pdf/2506.02355).
            std_cal_level (str): The scope for calculating the reward standard deviation for normalization.
                Can be 'group' (default, std is calculated per group) or 'batch' (std is calculated
                across the entire batch). The mean is always calculated per group.
                Calculating the mean at the local (group) level and the standard deviation at the global (batch)
                level enables more robust reward shaping(https://arxiv.org/pdf/2508.08221v1).
        """
        self.epsilon = epsilon
        self.std_threshold = std_threshold
        self.duplicate_experiences = duplicate_experiences
        self.rank_penalty = rank_penalty
        self.std_cal_level = std_cal_level
        if self.std_cal_level not in ["group", "batch"]:
            raise ValueError("std_cal_level must be either 'group' or 'batch'")

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self,
        group_id: str,
        exps: List[Experience],
        precomputed_std: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Experience], Dict]:
        metrics = {}
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
            if len(exps) == 1:
                group_reward_mean = torch.tensor(
                    0.0
                )  # check this value (use exps[0].reward may be better)
                group_reward_std = torch.tensor(1.0)  # set to 1.0 to avoid division by zero
                if self.std_threshold is not None:
                    metrics["skipped_count"] = 1
                    exps.clear()  # Clear experiences if only one experience
            else:
                if self.rank_penalty is not None:
                    # Correct bias by adjusting rewards based on their ranks
                    old_log_probs = torch.tensor(
                        [torch.mean(exp.logprobs, axis=-1) for exp in exps]
                    )
                    group_ranks = torch.argsort(torch.argsort(old_log_probs))
                    group_ranks = group_ranks / len(group_ranks)
                    rewards = rewards * (1 - group_ranks * self.rank_penalty)

                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)

                # If the reward standard deviation is below a threshold, skip the group
                if self.std_threshold is not None and group_reward_std <= self.std_threshold:
                    metrics["skipped_count"] = len(exps)
                    exps.clear()

            # use the (possibly rank-penalized) rewards the group statistics were computed from
            for exp, reward in zip(exps, rewards):
                if self.std_cal_level == "batch" and precomputed_std is not None:
                    score = (reward - group_reward_mean) / (precomputed_std + self.epsilon)
                else:
                    score = (reward - group_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics["reward_mean"] = group_reward_mean.item()
            metrics["reward_std"] = group_reward_std.item()

        return exps, metrics

    def _duplicate_experiences(self, exp_groups: Dict[str, List[Experience]]) -> List[Experience]:
        original_group_num = len(exp_groups)
        valid_groups = [group_exps for group_exps in exp_groups.values() if len(group_exps) > 0]
        skipped_group_num = original_group_num - len(valid_groups)

        if skipped_group_num == 0:
            return [exp for group in valid_groups for exp in group]
        elif skipped_group_num == original_group_num:
            # All groups are skipped, return an empty list
            return []
        else:
            idx = torch.randint(0, len(valid_groups), (skipped_group_num,)).tolist()
            duplicated_groups = [copy.deepcopy(valid_groups[i]) for i in idx]
            duplicated_exps = [exp for group in duplicated_groups for exp in group]
            exps = [exp for group in valid_groups for exp in group]
            exps.extend(duplicated_exps)
            return exps

    def process(self, exps):
        exp_groups = self.group_experiences(exps)
        metric_list = []
        precomputed_std = None
        if self.std_cal_level == "batch":
            all_rewards = torch.tensor(
                [exp.reward for exp in exps], dtype=torch.float32
            )  # All rewards in the batch
            if len(all_rewards) <= 1:
                precomputed_std = torch.tensor(1.0)
            else:
                precomputed_std = torch.std(all_rewards)
        for group_id, group_exps in exp_groups.items():
            group_exps, group_metrics = self.calculate_group_advantage(
                group_id, group_exps, precomputed_std=precomputed_std
            )
            metric_list.append(group_metrics)

        # Update the filtered_count metric
        filtered_count = sum(metric.pop("skipped_count", 0) for metric in metric_list)
        metrics = aggregate_metrics(metric_list, prefix="group_advantages")
        metrics["filtered_count"] = filtered_count
        if self.duplicate_experiences and self.std_threshold is not None:
            exps = self._duplicate_experiences(exp_groups)
        else:
            exps = [exp for group in exp_groups.values() for exp in group]  # Flatten the list
        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {
            "epsilon": 1e-6,
            "std_threshold": None,
            "duplicate_experiences": False,
            "rank_penalty": None,
            "std_cal_level": "group",
        }
