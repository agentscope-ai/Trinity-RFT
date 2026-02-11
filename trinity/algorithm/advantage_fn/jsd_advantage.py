# -*- coding: utf-8 -*-
"""Jensen-Shannon Divergence (JSD) advantage computation.

JSD(P||Q) = beta * KL(teacher||M) + (1-beta) * KL(student||M), where M = beta*teacher + (1-beta)*student.
When beta=0.5, this gives the standard symmetric JSD. All computations in log-space (no exp).
Aligned with SWIFT: beta=0/1 yield pure KL; temperature and optional chunking supported.
"""

from typing import Dict, Optional, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn


class JSDAdvantage(AdvantageFn):
    """Advantage function using Jensen-Shannon Divergence (log-space, SWIFT-aligned).

    Computes JSD in log-space only:
    - beta=0: JSD = KL(student || teacher)  [pure KL]
    - beta=1: JSD = KL(teacher || student)  [pure KL]
    - else: JSD = beta*KL(teacher||M) + (1-beta)*KL(student||M), M = mixture in log-space.

    The teacher_logprobs should be stored in Experience.teacher_logprobs
    by the workflow during exploration.
    """

    def __init__(
        self,
        lambda_coef: float = 0.5,
        kl_coef: float = 1.0,
        temperature: float = 1.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        """Initialize JSD advantage function.

        Args:
            lambda_coef: Weight beta for mixture. JSD = beta*KL(teacher||M) + (1-beta)*KL(student||M).
                         beta=0 => KL(student||teacher), beta=1 => KL(teacher||student). Range: [0, 1].
            kl_coef: Overall scaling coefficient for advantages.
            temperature: Temperature scaling for log-probs (log_probs / temperature). 1.0 = no scaling.
            chunk_size: If set, process flattened valid tokens in chunks to reduce peak memory; None = no chunking.
        """
        self.lambda_coef = lambda_coef
        self.kl_coef = kl_coef
        self.temperature = temperature
        self.chunk_size = chunk_size

    def _js_divergence_per_token(
        self,
        student_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute JSD per token in log-space only (no exp).
        - beta=0: KL(student || teacher) = student_logprobs - teacher_logprobs
        - beta=1: KL(teacher || student) = teacher_logprobs - student_logprobs
        - else: mixture log_m = logsumexp([s+log(1-beta), t+log(beta)]); JSD = beta*(t-log_m) + (1-beta)*(s-log_m).
        """
        beta = self.lambda_coef
        s = student_logprobs
        t = teacher_logprobs

        if beta == 0:
            # Pure KL(student || teacher)
            jsd = s - t
        elif beta == 1:
            # Pure KL(teacher || student)
            jsd = t - s
        else:
            # Precompute log coefficients once
            dtype, device = s.dtype, s.device
            beta_t = torch.tensor(beta, dtype=dtype, device=device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)

            # log(mixture) = log(beta*teacher + (1-beta)*student)
            mixture_log = torch.logsumexp(
                torch.stack([s + log_1_minus_beta, t + log_beta]),
                dim=0,
            )
            # KL(teacher || mixture) = t - log_m, KL(student || mixture) = s - log_m (log-space, no exp)
            kl_teacher = t - mixture_log
            kl_student = s - mixture_log
            jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

        if mask is not None:
            jsd = jsd * mask
        return jsd

    def _js_divergence_per_token_chunked(
        self,
        student_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute JSD per token with optional chunking over valid positions (for memory)."""
        flat_s = student_logprobs.reshape(-1)
        flat_t = teacher_logprobs.reshape(-1)
        flat_mask = response_mask.reshape(-1)
        valid = flat_mask > 0
        n_valid = valid.sum().item()
        if n_valid == 0:
            return (flat_s * 0).reshape_as(response_mask)

        s_valid = flat_s[valid]
        t_valid = flat_t[valid]
        chunk_size = self.chunk_size or n_valid
        beta = self.lambda_coef

        if beta == 0:
            jsd_valid = s_valid - t_valid
        elif beta == 1:
            jsd_valid = t_valid - s_valid
        else:
            dtype, device = s_valid.dtype, s_valid.device
            beta_t = torch.tensor(beta, dtype=dtype, device=device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
            jsd_valid = s_valid.new_zeros(s_valid.shape)
            for start in range(0, n_valid, chunk_size):
                end = min(start + chunk_size, n_valid)
                s_chunk = s_valid[start:end]
                t_chunk = t_valid[start:end]
                mixture_log = torch.logsumexp(
                    torch.stack([s_chunk + log_1_minus_beta, t_chunk + log_beta]),
                    dim=0,
                )
                kl_t = t_chunk - mixture_log
                kl_s = s_chunk - mixture_log
                jsd_valid[start:end] = beta_t * kl_t + (1 - beta_t) * kl_s

        out = flat_s.new_zeros(flat_s.shape)
        out[valid] = jsd_valid
        return out.reshape_as(response_mask)

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Compute advantages using JSD.

        Advantages are computed directly from JSD: advantages = -kl_coef * JSD
        Since we want to minimize JSD, we use negative JSD as advantage.
        Lower JSD (better alignment with teacher) → higher advantage.
        The advantage guides the policy gradient to reduce JSD.

        Args:
            exps: DataProto containing:
                - old_log_probs: student's sampling logprobs [batch, seq]
                - teacher_logprobs: teacher's logprobs [batch, seq]
                - response_mask: mask for response tokens [batch, seq]

        Returns:
            exps: DataProto with advantages and returns added
            metrics: Dict with jsd and advantage statistics
        """
        metrics = {}

        old_log_probs = exps.batch["old_log_probs"]  # student sampling logprobs
        teacher_log_probs = exps.batch["teacher_logprobs"]
        response_mask = exps.batch["response_mask"]

        # Temperature scaling (align with SWIFT: logits / T => apply to log-probs here)
        if self.temperature != 1.0:
            old_log_probs = old_log_probs / self.temperature
            teacher_log_probs = teacher_log_probs / self.temperature

        # Compute JSD per token (with optional chunking for memory)
        if self.chunk_size is not None:
            jsd_per_token = self._js_divergence_per_token_chunked(
                old_log_probs, teacher_log_probs, response_mask
            )
        else:
            jsd_per_token = self._js_divergence_per_token(
                old_log_probs, teacher_log_probs, mask=response_mask
            )

        # For advantage function, use JSD directly
        # Since we want to minimize JSD, we use negative JSD as advantage
        advantages = -self.kl_coef * jsd_per_token
        advantages = advantages * response_mask

        exps.batch["advantages"] = advantages
        exps.batch["returns"] = advantages.clone()

        # JSD metrics (over valid tokens)
        jsd_sum = (jsd_per_token * response_mask).sum(dim=-1)
        metrics["jsd/mean"] = jsd_sum.mean().item()
        metrics["jsd/std"] = jsd_sum.std().item() if jsd_sum.numel() > 1 else 0.0

        metrics["advantages/mean"] = advantages.sum(dim=-1).mean().item()

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "lambda_coef": 0.5,
            "kl_coef": 1.0,
            "temperature": 1.0,
            "chunk_size": None,
        }
