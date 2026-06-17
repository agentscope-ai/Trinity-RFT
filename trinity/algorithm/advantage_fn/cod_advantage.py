"""CoD advantage computation."""
from typing import Dict, List, Optional, Tuple
import math

import torch

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn
from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.monitor import gather_metrics


def helper_get_reward(exp: Experience, reward_field: str):
    if reward_field == "reward":
        return exp.reward
    elif reward_field == "total_reward":
        return exp.info["total_reward"]
    else:
        raise ValueError(f"Invalid reward_field {reward_field}")


class CoDAdvantageFn(AdvantageFn, ExperienceOperator):
    """An advantage function dedicated for CoD."""

    def __init__(
        self,
        epsilon: float = 1e-6,
        enable_step_norm: bool = False,
        std_cal_level: str = "group",  # 'group' (task-level) or 'batch' or 'none'
        std_threshold: Optional[float] = None,
        iterative_hint_e2e_causal: bool = False,
        e2e_causal_returns_style: str = "mean",
        e2e_causal_returns_window: int = -1,
        e2e_causal_returns_gamma: float = 1.0,
        e2e_causal_baseline: str = "group-mean",
        mask_format_issue_exp: bool = False,
        red_weight_temp: Optional[float] = None,
        red_weight_adaptive_temp: bool = False,
        red_weight_adaptive_version: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the CoD advantage function.

        Args:
            --- original multi-step grpo advantage ---
            epsilon (float): A small value to avoid division by zero.
            enable_step_norm (bool): If True, normalize advantages by trajectory length.
            std_cal_level (str): The scope for calculating reward standard deviation.
                'group' (default): Std is calculated per task group.
                'batch': Std is calculated across all last-step rewards in the entire batch.
                'none': no Std calculation or advantage normalization.
                The mean is always calculated per task group.
            std_threshold (Optional[float]): If provided, task groups with a reward standard deviation
                equal or below this threshold will be skipped.

            --- CoD-specific ---

            iterative_hint_e2e_causal (bool): If True, use iterative hint with end-to-end meta-rl and advantages that respect causality. Options:
                - {"style": "mean"}
                - {"style": "sliding_window_mean", "window": int}
                - {"style": "discounted", "gamma": float}
                - e2e_causal_baseline: "group-mean"
                - mask_format_issue_exp (bool): if true, mask exp with format issue & positive score
        """
        self.epsilon = epsilon
        self.enable_step_norm = enable_step_norm
        self.std_cal_level = std_cal_level
        self.std_threshold = std_threshold
        if self.std_cal_level not in ["group", "batch", "none"]:
            raise ValueError("std_cal_level must be either 'group' or 'batch' or 'none'")
        self.iterative_hint_e2e_causal = iterative_hint_e2e_causal
        self.e2e_causal_returns_style = e2e_causal_returns_style
        self.e2e_causal_returns_window = e2e_causal_returns_window
        self.e2e_causal_returns_gamma = e2e_causal_returns_gamma
        self.e2e_causal_baseline = e2e_causal_baseline
        self.mask_format_issue_exp = mask_format_issue_exp
        self.red_weight_temp = red_weight_temp
        self.red_weight_adaptive_temp = red_weight_adaptive_temp
        self.red_weight_adaptive_version = red_weight_adaptive_version


    def calculate_last_step_advantage(
        self,
        exps: Dict[str, Experience],
        precomputed_std: Optional[torch.Tensor] = None,
        reward_field: str = "reward",
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        """Calculate group advantage for a given group of experiences.

        Args:
            exps (Dict[str, Experience]): One experience per run, keyed by run ID.
            precomputed_std (Optional[torch.Tensor]): Precomputed standard deviation for batch-level calculation.
            reward_field: which field to use as reward (for score and advantage calculation).

        Returns:
            Dict[str, float]: Scores for each run.
            Dict[str, float]: Metrics for logging.
            bool: Whether this group should be skipped.
        """
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([helper_get_reward(exp, reward_field) for exp in exps.values()], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)

            # Determine if this group should be skipped based on std_threshold
            should_skip = False
            if self.std_threshold is not None:
                if len(exps) == 1 or group_reward_std <= self.std_threshold:
                    should_skip = True

            scores = {}
            for rid, exp in exps.items():
                exp_reward = helper_get_reward(exp, reward_field)
                if self.std_cal_level == "batch" and precomputed_std is not None:
                    score = (exp_reward - group_reward_mean) / (precomputed_std + self.epsilon)
                elif self.std_cal_level == "group":
                    score = (exp_reward - group_reward_mean) / (group_reward_std + self.epsilon)
                elif self.std_cal_level == "none":
                    score = exp_reward - group_reward_mean
                else:
                    raise ValueError(f"Invalid std_cal_level '{self.std_cal_level}'.")
                scores[rid] = score.item()

            # Use standard task rewards for metrics
            standard_rewards = torch.tensor([exp.reward for exp in exps.values()], dtype=torch.float32)
            standard_reward_mean = torch.mean(standard_rewards)
            standard_reward_std = torch.std(standard_rewards)
            metrics = {
                "reward_mean": standard_reward_mean.item(),
                "reward_std": standard_reward_std.item(),
            }

        return scores, metrics, should_skip

    def broadcast_advantages(
        self, run_exps: Dict[str, List[Experience]], scores: Dict[str, float]
    ) -> Dict[str, List[Experience]]:
        """Broadcast the calculated advantages to all previous steps in each run.

        Args:
            run_exps (Dict[str, List[Experience]]): Experiences grouped by run ID.
            scores (Dict[str, float]): Calculated scores for each run.

        Returns:
            Dict[str, List[Experience]]: Updated experiences with advantages broadcasted.
        """
        for run_id, exps in run_exps.items():
            score = scores[run_id]
            traj_length = len(exps)
            for exp in exps:
                exp.advantages = exp.action_mask * score  # type: ignore [operator]
                if self.enable_step_norm:
                    exp.advantages /= traj_length
                exp.returns = exp.advantages.clone()
        return run_exps

    def process_standard_grpo_custom_score(
            self, 
            exps: List[Experience],
            reward_field: str = "reward",
        ) -> Tuple[List[Experience], Dict]:
        """Standard GRPO with custom field as reward, e.g., per-task reward or total reward in end-to-end meta-rl.
        
        reward_field: which field to use as reward (for score and advantage calculation)
            - "reward": use exp.reward 
            - "total_reward": use exp.info["total_reward"], tailored to iterative_hint_e2e
        """
        if len(exps) == 0:
            return [], {}
        cnt = 0
        metric_list = []
        filtered_count = 0
        # Step 1: split the experiences into sub-groups by task
        task_exps = group_by(exps, "task")

        # --- Pre-computation step for batch-level standard deviation ---
        precomputed_std = None
        if self.std_cal_level == "batch":
            all_laststep_rewards = []
            for task_exp in task_exps.values():
                # First, group all experiences by run to find the last step of each run
                task_run_exps = group_by(task_exp, "run")
                # Collect rewards from the last step of every run in the entire batch
                last_step_rewards = [
                    helper_get_reward(run_steps[-1], reward_field) for run_steps in task_run_exps.values() if run_steps
                ]
                all_laststep_rewards.extend(last_step_rewards)

            if len(all_laststep_rewards) <= 1:
                precomputed_std = torch.tensor(1.0)
            else:
                precomputed_std = torch.std(torch.tensor(all_laststep_rewards, dtype=torch.float32))
        # --- End of pre-computation ---

        # Step 2: further split each task's experiences into sub-groups by run
        result_exps = []
        total_task_groups = len(task_exps)
        skipped_task_groups = 0

        for task_exp in task_exps.values():
            run_exps = group_by(task_exp, "run")

            # Step3: extract the last experience (last step) from each run and calculate scores
            last_step_exps = {run_id: step_exps[-1] for run_id, step_exps in run_exps.items()}
            scores, metrics, should_skip = self.calculate_last_step_advantage(
                last_step_exps, 
                precomputed_std=precomputed_std,
                reward_field=reward_field,
            )

            # Skip this task group if std is below threshold
            if should_skip:
                # Count all experiences in this task group as filtered
                task_exp_count = sum(len(step_exps) for step_exps in run_exps.values())
                filtered_count += task_exp_count
                skipped_task_groups += 1
                metric_list.append(metrics)
                continue

            metric_list.append(metrics)

            # Step 4: broadcast the advantages to all previous steps
            run_exps = self.broadcast_advantages(run_exps, scores)
            for exps in run_exps.values():
                cnt += len(exps)
                result_exps.extend(exps)

        metrics = gather_metrics(metric_list, "group_advantages")
        metrics["experience_count"] = cnt
        metrics["filtered_count"] = filtered_count

        # Calculate the ratio of skipped task groups
        if total_task_groups > 0:
            metrics["skipped_group_ratio"] = skipped_task_groups / total_task_groups
        else:
            metrics["skipped_group_ratio"] = 0.0

        return result_exps, metrics
    
    def process_standard_grpo(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        """Standard GRPO advantage, using exp.reward as reward field."""
        return self.process_standard_grpo_custom_score(exps, "reward")

    def process_iterative_hint_e2e_causal(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        """Refined advantage for cross-task e2e case, respecting causal relationship.

        Assume each exp has exp.info fields:
            - "exp_type": "solve_task" or "gen_hint"
            - "task_idx" (0, 1, 2, ...) or "hint_idx" (0, 1, ...)
            - "task_rewards": List[float], list of task/episode-wise outcome rewards within the same meta-trajectory
        """
        metrics = dict()
        metrics["experience_count"] = len(exps)

        # Drop trajectories the judge flagged (own- or trajectory-level).
        def _is_bad(e: Experience) -> bool:
            return (
                e.info.get("judge_format_error", False)
                or e.info.get("judge_format_error_in_trajectory", False)
            )

        bad_count = sum(1 for e in exps if _is_bad(e))
        exps = [e for e in exps if not _is_bad(e)]
        metrics["judge_format_error_count"] = bad_count

        # Step 1. For each exp, calculate exp.info["returns_norm"], plus concise record for Step 2
        record_returns = dict()

        for exp in exps:
            task_rewards = exp.info["task_rewards"]  # List[float]
            exp_type = exp.info["exp_type"]

            # Get rewards-to-go (for gen_hint exp, also include current exp's own reward)
            if exp_type == "solve_task":
                idx = int(exp.info["task_idx"])
                rewards_to_go = task_rewards[idx:]
            elif exp_type == "gen_hint":
                idx = int(exp.info["hint_idx"])
                rewards_to_go = [exp.reward] + task_rewards[(idx + 1) : ]
            else:
                raise ValueError(f"Invalid exp_type {exp_type}, something went wrong.")
            rewards_to_go = [float(r) for r in rewards_to_go]  # mitigate dtype error in torch.tensor/sum below

            # Calculate returns
            e2e_causal_returns_style = self.e2e_causal_returns_style
            if e2e_causal_returns_style == "mean":
                exp_returns_norm = torch.mean(torch.tensor(rewards_to_go)).item()
            elif e2e_causal_returns_style == "sliding_window_mean":
                window = int(self.e2e_causal_returns_window)
                assert window >= 1, f"Expect sliding-window size >= 1, get {window}."
                rewards_to_go_window = rewards_to_go[ : min(window, len(rewards_to_go))]
                exp_returns_norm = torch.mean(torch.tensor(rewards_to_go_window)).item()
            elif e2e_causal_returns_style == "discounted":
                gamma = float(self.e2e_causal_returns_gamma)
                assert 0.0 <= gamma <= 1.0, f"Expect gamma within range [0, 1], get {gamma}."
                rewards_weights = [gamma ** i for i in range(len(rewards_to_go))]
                exp_returns_norm = torch.sum(torch.tensor(rewards_to_go) * torch.tensor(rewards_weights)).item()
            else:
                raise ValueError(f"Invalid e2e_causal_returns_style: {e2e_causal_returns_style}.")

            exp.info["returns_norm"] = exp_returns_norm

            key_for_record = "_".join([str(exp.eid.task), exp_type, str(idx)])
            exp.info["key_for_record"] = key_for_record
            if key_for_record not in record_returns:
                record_returns[key_for_record] = dict()
            if exp.eid.run not in record_returns[key_for_record]:
                record_returns[key_for_record][exp.eid.run] = exp_returns_norm

        # Step 2. Calculate episode-wise baseline within each meta-task, then calculate advantage for each exp
        episode_wise_baselines = dict()
        for key, value in record_returns.items():
            if self.e2e_causal_baseline == "group-mean":
                baseline = torch.mean(torch.tensor(list(value.values()))).item()
            else:
                raise ValueError(f"Invalid e2e_causal_baseline: {self.e2e_causal_baseline}")
            episode_wise_baselines[key] = baseline
        
        mask_format_issue_count = 0

        # Compute per-exp score (= returns_norm - baseline)
        lst_resp_len = [torch.sum(exp.action_mask).item() for exp in exps]
        lst_scores = []
        for exp in exps:
            key = exp.info["key_for_record"]
            baseline = episode_wise_baselines[key]
            lst_scores.append(exp.info["returns_norm"] - baseline)

        # Decide the re-weight temperature: adaptive (set so that token-wise mean
        # advantage is non-negative) or the fixed red_weight_temp.
        if self.red_weight_adaptive_temp:

            def estimate_adv_sum(lst_scores, lst_resp_len, temp):
                if temp is None:
                    return sum([lst_scores[i] * lst_resp_len[i] for i in range(len(lst_scores))])
                return sum(
                    [lst_scores[i] * math.exp(lst_scores[i] / temp) * lst_resp_len[i] for i in range(len(lst_scores))]
                )

            weight_temp = None
            if self.red_weight_adaptive_version == "bisection":
                # refined implementation, by bisection
                if estimate_adv_sum(lst_scores, lst_resp_len, None) < 0:
                    temp_low = 0.8  # avoid too aggressive weighting
                    temp_high = 100.0
                    for _ in range(20):
                        temp_mid = (temp_low + temp_high) / 2
                        if estimate_adv_sum(lst_scores, lst_resp_len, temp_mid) < 0:
                            temp_high = temp_mid
                        else:
                            temp_low = temp_mid
                        if temp_high - temp_low < 0.05:
                            break
                    weight_temp = temp_mid
            else:
                raise ValueError(f"Invalid red_weight_adaptive_version {self.red_weight_adaptive_version}.")
        else:
            weight_temp = self.red_weight_temp
            if weight_temp:
                assert weight_temp > 0.0, "red_weight_temp must be positive float."

        # Unified loop: optional format-issue masking + re-weight, then set advantage/returns
        for i, exp in enumerate(exps):
            score = lst_scores[i]
            if self.mask_format_issue_exp:
                if exp.info.get("early_termination_by_format_issue", False) and (score > 0.0):
                    score = 0.0
                    mask_format_issue_count += 1
            if weight_temp:
                score *= math.exp(score / weight_temp)
            exp.advantages = exp.action_mask * score
            exp.returns = exp.action_mask * exp.info["returns_norm"]

        metrics["mask_format_issue_count"] = mask_format_issue_count

        return exps, metrics

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        if self.iterative_hint_e2e_causal:
            return self.process_iterative_hint_e2e_causal(exps)
        else:
            return self.process_standard_grpo(exps)

    def __call__(self, exps, **kwargs):
        """Not used, `process` is the method being called."""
        if self.iterative_hint_e2e_causal:
            return self.process_iterative_hint_e2e_causal(exps)
        else:
            return self.process_standard_grpo(exps)

    @classmethod
    def compute_in_trainer(cls) -> bool:
        """Whether the advantage should be computed in the trainer loop."""
        return False

    @classmethod
    def default_args(cls) -> Dict:
        """Return the default configuration for this strategy."""
        return {
            "epsilon": 1e-6,
            "enable_step_norm": False,
            "std_threshold": None,
            "std_cal_level": "group",
            "iterative_hint_e2e_causal": False,
            "e2e_causal_returns_style": "mean",
            "e2e_causal_returns_window": -1,
            "e2e_causal_returns_gamma": 1.0,
            "e2e_causal_baseline": "group-mean",
            "mask_format_issue_exp": False,
            "red_weight_temp": None,
            "red_weight_adaptive_temp": False,
            "red_weight_adaptive_version": None,
        }
