# -*- coding: utf-8 -*-
"""On-Policy Distillation advantage computation.

Reference: Tinker library's on-policy distillation.

advantages = -(student_logprobs - teacher_logprobs)
           = teacher_logprobs - student_logprobs
"""

from typing import Dict, Tuple

from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn


class OnPolicyDistillAdvantage(AdvantageFn):
    """Advantage function for on-policy distillation.

    Computes: advantages = kl_coef * (teacher_logprobs - student_logprobs)

    The teacher_logprobs should be stored in Experience.teacher_logprobs
    by the workflow during exploration.
    """

    def __init__(self, kl_coef: float = 1.0) -> None:
        self.kl_coef = kl_coef

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Compute advantages from teacher and student logprobs.

        Args:
            exps: DataProto containing:
                - old_log_probs: student's sampling logprobs [batch, seq]
                - teacher_log_probs: teacher's logprobs [batch, seq]
                - response_mask: mask for response tokens [batch, seq]

        Returns:
            exps: DataProto with advantages and returns added
            metrics: Dict with kl and advantage statistics
        """
        metrics = {}

        old_log_probs = exps.batch["old_log_probs"]  # student sampling logprobs
        teacher_log_probs = exps.batch["teacher_logprobs"]
        response_mask = exps.batch["response_mask"]

        # advantages = -(student - teacher) = teacher - student
        advantages = self.kl_coef * (teacher_log_probs - old_log_probs)

        # Apply mask
        advantages = advantages * response_mask

        exps.batch["advantages"] = advantages
        exps.batch["returns"] = advantages.clone()

        # Metrics
        kl_per_token = old_log_probs - teacher_log_probs
        kl_sum = (kl_per_token * response_mask).sum(dim=-1)
        metrics["kl/mean"] = kl_sum.mean().item()
        metrics["kl/std"] = kl_sum.std().item() if kl_sum.numel() > 1 else 0.0
        metrics["advantages/mean"] = advantages.sum(dim=-1).mean().item()

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {"kl_coef": 1.0}
