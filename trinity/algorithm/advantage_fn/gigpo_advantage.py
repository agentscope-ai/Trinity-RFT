"""GiGPO (Group-in-Group Policy Optimization) advantage computation.

Reference:
    Feng et al., "Group-in-Group Policy Optimization for LLM Agent Training",
    arXiv:2505.10978.
"""

from typing import Dict, List, Literal, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn
from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.metrics import aggregate_metrics


class GiGPOAdvantageFn(AdvantageFn, ExperienceOperator):
    """Compute hierarchical GiGPO advantages for multi-turn agent experiences.

    GiGPO combines episode-level relative advantages (GRPO-style over full
    trajectories) with step-level relative advantages within anchor-state groups.
    The combined scalar advantage is ``A = A_E + omega * A_S``, then broadcast
    to tokens via ``action_mask``.

    Workflows must set ``experience.info[env_state_hash_key]`` for step-level
    grouping and should set ``experience.info[step_reward_key]`` for per-step
    immediate rewards. See ``examples/gigpo_alfworld/README.md``.

    Attributes:
        omega: Weight on step-level advantage A_S.
        gamma: Discount factor for discounted step returns R_t.
        fnorm: Normalization mode, ``"std"`` (GRPO) or ``"none"`` (RLOO-style).
        epsilon: Small constant added to the normalization denominator.
        step_reward_key: Key in ``experience.info`` for immediate reward r_t.
        env_state_hash_key: Key in ``experience.info`` for anchor state identity.
    """

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
        """Initialize GiGPO advantage computation.

        Args:
            omega: Weight on step-level advantage A_S in the combined advantage.
            gamma: Discount factor for R_t = sum_{k>=t} gamma^{k-t} r_k.
            fnorm: Group normalization. ``"std"`` divides by standard deviation;
                ``"none"`` uses F_norm = 1 (paper default for agent benchmarks).
            epsilon: Stabilizer when dividing by std or 1.
            step_reward_key: ``experience.info`` field for immediate step reward.
            env_state_hash_key: ``experience.info`` field for anchor-state hash.
            **kwargs: Ignored; accepted for registry compatibility.
        """
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
        """Sort experiences within a run by ``eid.step``.

        Args:
            run_steps: Experiences belonging to one run.

        Returns:
            List[Experience]: Steps ordered by increasing step index.
        """
        return sorted(run_steps, key=lambda exp: exp.eid.step)

    def _step_rewards(self, run_steps: List[Experience]) -> List[float]:
        """Extract immediate per-step rewards r_t for one trajectory.

        Uses ``info[step_reward_key]`` when present. Otherwise applies a sparse
        fallback: zero on non-terminal steps and ``exp.reward`` on the last
        step only (for ``RewardPropagationWorkflow`` that copies terminal reward).

        Args:
            run_steps: Experiences belonging to one run.

        Returns:
            List[float]: Immediate rewards aligned with sorted steps.
        """
        sorted_steps = self._sort_run_steps(run_steps)
        has_step_reward = any(self.step_reward_key in (exp.info or {}) for exp in sorted_steps)
        if has_step_reward:
            return [float((exp.info or {}).get(self.step_reward_key, 0.0)) for exp in sorted_steps]
        rewards: List[float] = []
        for idx, exp in enumerate(sorted_steps):
            if idx == len(sorted_steps) - 1 and exp.reward is not None:
                rewards.append(float(exp.reward))
            else:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def _discounted_returns(rewards: List[float], gamma: float) -> List[float]:
        """Compute discounted returns R_t = sum_{k>=t} gamma^{k-t} r_k.

        Args:
            rewards: Immediate rewards r_t along a trajectory.
            gamma: Discount factor.

        Returns:
            List[float]: Discounted returns, same length as ``rewards``.
        """
        n = len(rewards)
        if n == 0:
            return []
        returns = [0.0] * n
        running = 0.0
        for t in range(n - 1, -1, -1):
            running = rewards[t] + gamma * running
            returns[t] = running
        return returns

    def _normalize(self, values: List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize a list of scalars within a group (episode or anchor).

        Singleton groups use mean=0 and denom=1, matching GRPO convention.

        Args:
            values: Scalar values to normalize (e.g. episode returns or R_t).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Normalized scores,
                group mean, and normalization denominator (std or 1).
        """
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
        """Write scalar advantage and returns onto an experience.

        Args:
            exp: Experience to update in place.
            scalar: Combined advantage A_E + omega * A_S.
        """
        if exp.action_mask is not None:
            exp.advantages = exp.action_mask * scalar
        else:
            exp.advantages = torch.tensor(scalar, dtype=torch.float32)
        exp.returns = exp.advantages.clone()

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        """Compute GiGPO advantages for a batch of multi-step experiences.

        Episode-level: group by task, compare trajectory returns R(tau) across
        runs (GRPO-style). Step-level: group by ``env_state_hash`` across the
        batch, compare discounted returns R_t; singleton anchors get A_S = 0.

        Args:
            exps: Multi-step experiences with ``eid.task``, ``eid.run``,
                ``eid.step``, and optional anchor metadata in ``info``.

        Returns:
            Tuple[List[Experience], Dict]: Experiences with ``advantages`` and
                ``returns`` set, plus logging metrics prefixed with ``gigpo/``.
        """
        if len(exps) == 0:
            return [], {}

        episode_metric_list: List[Dict] = []
        anchor_groups_total = 0
        anchor_groups_size_gt1 = 0
        abs_a_e: List[float] = []
        abs_a_s: List[float] = []

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
            metrics["gigpo/anchor_group_hit_ratio"] = anchor_groups_size_gt1 / anchor_groups_total
        else:
            metrics["gigpo/anchor_group_hit_ratio"] = 0.0
        if abs_a_e:
            metrics["gigpo/mean_abs_A_E"] = sum(abs_a_e) / len(abs_a_e)
        if abs_a_s:
            metrics["gigpo/mean_abs_A_S"] = sum(abs_a_s) / len(abs_a_s)
        metrics["experience_count"] = len(result_exps)

        return result_exps, metrics

    def __call__(self, exps, **kwargs):
        """Callable entry point; delegates to :meth:`process`."""
        return self.process(exps)

    @classmethod
    def compute_in_trainer(cls) -> bool:
        """Whether advantages are computed in the trainer loop.

        Returns:
            bool: ``False``; GiGPO runs in the experience pipeline.
        """
        return False

    @classmethod
    def default_args(cls) -> Dict:
        """Return default ``advantage_fn_args`` for GiGPO.

        Returns:
            Dict: Default hyperparameters for ``GiGPOAdvantageFn``.
        """
        return {
            "omega": 1.0,
            "gamma": 1.0,
            "fnorm": "none",
            "epsilon": 1e-6,
            "step_reward_key": "step_reward",
            "env_state_hash_key": "env_state_hash",
        }
