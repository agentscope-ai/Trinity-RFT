"""GiGPO (Group-in-Group Policy Optimization) advantage computation.

Ref: Feng et al., arXiv:2505.10978 — episode-level and anchor-state step-level advantages.
"""

from typing import Dict, List, Literal, Optional, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn
from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.metrics import aggregate_metrics


class GiGPOAdvantageFn(AdvantageFn, ExperienceOperator):
    """Hierarchical critic-free advantages: A = A^E(τ) + ω · A^S(a|s̃)."""

    def __init__(
        self,
        omega: float = 1.0,
        gamma: float = 1.0,
        fnorm: Literal["std", "none"] = "none",
        epsilon: float = 1e-6,
        step_reward_key: str = "step_reward",
        env_state_hash_key: str = "env_state_hash",
        **kwargs,
    ) -> None:
        self.omega = omega
        self.gamma = gamma
        self.epsilon = epsilon
        self.step_reward_key = step_reward_key
        self.env_state_hash_key = env_state_hash_key
        if fnorm not in ("std", "none"):
            raise ValueError("fnorm must be 'std' or 'none'")
        self.fnorm = fnorm

    @staticmethod
    def _sort_run_steps(run_steps: List[Experience]) -> List[Experience]:
        return sorted(run_steps, key=lambda exp: exp.eid.step)

    def _step_rewards(self, run_steps: List[Experience]) -> List[float]:
        """Immediate per-step rewards r_t."""
        sorted_steps = self._sort_run_steps(run_steps)
        has_step_reward = any(self.step_reward_key in (exp.info or {}) for exp in sorted_steps)
        if has_step_reward:
            return [float((exp.info or {}).get(self.step_reward_key, 0.0)) for exp in sorted_steps]
        # Sparse fallback: terminal reward only on last step (RewardPropagationWorkflow).
        rewards: List[float] = []
        for idx, exp in enumerate(sorted_steps):
            if idx == len(sorted_steps) - 1 and exp.reward is not None:
                rewards.append(float(exp.reward))
            else:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def _discounted_returns(rewards: List[float], gamma: float) -> List[float]:
        """R_t = sum_{k>=t} gamma^{k-t} r_k."""
        n = len(rewards)
        if n == 0:
            return []
        returns = [0.0] * n
        running = 0.0
        for t in range(n - 1, -1, -1):
            running = rewards[t] + gamma * running
            returns[t] = running
        return returns

    def _normalize(
        self, values: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (normalized_scores, mean, denom) aligned with values length."""
        tensor = torch.tensor(values, dtype=torch.float32)
        if len(values) == 1:
            mean = torch.tensor(0.0)
            denom = torch.tensor(1.0)
        else:
            mean = torch.mean(tensor)
            if self.fnorm == "std":
                denom = torch.std(tensor)
            else:
                denom = torch.tensor(1.0)
        scores = (tensor - mean) / (denom + self.epsilon)
        return scores, mean, denom

    def _apply_mask(self, exp: Experience, scalar: float) -> None:
        if exp.action_mask is not None:
            exp.advantages = exp.action_mask * scalar
        else:
            exp.advantages = torch.tensor(scalar, dtype=torch.float32)
        exp.returns = exp.advantages.clone()

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        if len(exps) == 0:
            return [], {}

        episode_metric_list: List[Dict] = []
        anchor_groups_total = 0
        anchor_groups_size_gt1 = 0
        abs_a_e: List[float] = []
        abs_a_s: List[float] = []

        # --- Episode-level advantages (per task, per run) ---
        run_to_a_e: Dict[str, float] = {}
        exp_to_discounted_return: Dict[int, float] = {}

        task_exps = group_by(exps, "task")
        for task_exp in task_exps.values():
            run_exps = group_by(task_exp, "run")
            episode_returns: List[float] = []
            run_ids: List[str] = []

            for run_id, run_steps in run_exps.items():
                sorted_steps = self._sort_run_steps(run_steps)
                rewards = self._step_rewards(sorted_steps)
                disc_returns = self._discounted_returns(rewards, self.gamma)
                episode_return = sum(rewards)
                episode_returns.append(episode_return)
                run_ids.append(run_id)
                for exp, r_t in zip(sorted_steps, disc_returns):
                    exp_to_discounted_return[id(exp)] = r_t

            scores, mean, denom = self._normalize(episode_returns)
            episode_metric_list.append(
                {
                    "episode_reward_mean": mean.item(),
                    "episode_reward_std": denom.item(),
                }
            )
            for run_id, a_e in zip(run_ids, scores.tolist()):
                run_to_a_e[run_id] = a_e

        # --- Step-level anchor advantages (across full batch) ---
        anchor_buckets: Dict[str, List[Tuple[Experience, float]]] = {}
        for exp in exps:
            info = exp.info or {}
            state_hash = info.get(self.env_state_hash_key)
            if state_hash is None:
                continue
            r_t = exp_to_discounted_return[id(exp)]
            anchor_buckets.setdefault(str(state_hash), []).append((exp, r_t))

        exp_to_a_s: Dict[int, float] = {id(exp): 0.0 for exp in exps}
        for members in anchor_buckets.values():
            anchor_groups_total += 1
            if len(members) < 2:
                continue
            anchor_groups_size_gt1 += 1
            disc_values = [r for _, r in members]
            scores, _, _ = self._normalize(disc_values)
            for (exp, _), a_s in zip(members, scores.tolist()):
                exp_to_a_s[id(exp)] = a_s

        # --- Combine and assign ---
        result_exps: List[Experience] = []
        for exp in exps:
            a_e = run_to_a_e.get(exp.eid.rid, 0.0)
            a_s = exp_to_a_s.get(id(exp), 0.0)
            combined = a_e + self.omega * a_s
            self._apply_mask(exp, combined)
            abs_a_e.append(abs(a_e))
            abs_a_s.append(abs(a_s))
            result_exps.append(exp)

        metrics = aggregate_metrics(episode_metric_list, prefix="gigpo")
        metrics["gigpo/anchor_groups_total"] = anchor_groups_total
        metrics["gigpo/anchor_groups_size_gt1"] = anchor_groups_size_gt1
        if anchor_groups_total > 0:
            metrics["gigpo/anchor_group_hit_ratio"] = (
                anchor_groups_size_gt1 / anchor_groups_total
            )
        else:
            metrics["gigpo/anchor_group_hit_ratio"] = 0.0
        if abs_a_e:
            metrics["gigpo/mean_abs_A_E"] = sum(abs_a_e) / len(abs_a_e)
        if abs_a_s:
            metrics["gigpo/mean_abs_A_S"] = sum(abs_a_s) / len(abs_a_s)
        metrics["experience_count"] = len(result_exps)

        return result_exps, metrics

    def __call__(self, exps, **kwargs):
        return self.process(exps)

    @classmethod
    def compute_in_trainer(cls) -> bool:
        return False

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "omega": 1.0,
            "gamma": 1.0,
            "fnorm": "none",
            "epsilon": 1e-6,
            "step_reward_key": "step_reward",
            "env_state_hash_key": "env_state_hash",
        }
