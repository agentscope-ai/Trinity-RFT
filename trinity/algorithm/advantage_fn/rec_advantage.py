"""REC advantage computation
"""

from typing import Dict, List, Optional, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import GroupAdvantage
from trinity.common.experience import Experience, group_by


class RECGroupedAdvantage(GroupAdvantage):
    """An advantage class that calculates REC advantages."""

    def __init__(
        self,
        epsilon: float = 1e-6,
        std_normalize: Optional[bool] = False,
        drop: Optional[str] = None,
    ) -> None:
        """Initialize the REC advantage function.

        Args:
            epsilon (float): A small value to avoid division by zero.
            std_normalize (Optional[bool]): If provided, normalize the advantage with group-level reward standard deviation.
            drop (Optional[str]): Strategy to drop experiences. Options are "balance" or None.
        """
        self.epsilon = epsilon
        self.std_normalize = std_normalize
        self.drop = drop
        assert self.drop in [None, "balance"], f"Invalid drop: {self.drop}"

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        # Initialize masks and metrics
        N = len(exps)
        metrics = {}
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)  # set to 1.0 to avoid division by zero
            else:
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)

            is_pos = rewards >= group_reward_mean
            pos_count = is_pos.sum().item()
            neg_count = len(exps) - pos_count

            drop_idx = torch.tensor([], dtype=torch.long)
            drop_frac = 0.0
            if self.drop == "balance" and neg_count > pos_count:
                extra_neg = neg_count - pos_count
                neg_idx = (~is_pos).nonzero(as_tuple=True)[0]
                perm = torch.randperm(len(neg_idx))[:extra_neg]
                drop_idx = neg_idx[perm]
                drop_frac = float(extra_neg) / float(max(N, 1))
            metrics["drop_balance"] = drop_frac
            keep_mask = torch.ones(N, dtype=torch.bool)
            if drop_idx.numel() > 0:
                keep_mask[drop_idx] = False

            if keep_mask.sum().item() <= 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)  # avoid divide-by-zero
            else:
                sel_rewards = rewards[keep_mask]
                group_reward_mean = sel_rewards.mean()
                group_reward_std = sel_rewards.std(unbiased=False)

            for i, exp in enumerate(exps):
                if not keep_mask[i]:
                    adv = torch.tensor(0.0)
                else:
                    if getattr(self, "std_normalize", False):
                        adv = (rewards[i] - group_reward_mean) / (group_reward_std + self.epsilon)
                    else:
                        adv = rewards[i] - group_reward_mean

                exp.advantages = adv * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics["reward_mean"] = group_reward_mean.item()
            metrics["reward_std"] = group_reward_std.item()

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {
            "epsilon": 1e-6,
            "std_normalize": False,
            "drop": None,
        }
