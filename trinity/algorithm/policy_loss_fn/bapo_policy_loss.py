"""BAPO policy loss function.

Balanced Policy Optimization with Adaptive Clipping (BAPO) keeps the PPO-clip
surrogate but chooses the clipping bounds for every update, so that tokens with
positive advantages contribute at least a target share of the policy gradient.

Refer to the BAPO paper for details. https://arxiv.org/abs/2510.18927
"""

import math
from typing import Dict, List, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


class BAPOPolicyLossFn(PolicyLossFn):
    """BAPO policy loss (arXiv:2510.18927, Eq. 8 and Algorithm 1).

    The clipping bounds are **explicit importance-ratio values** (e.g. ``0.6`` and
    ``1.2``), not the ``1 ± clip_range`` epsilons used by ``PPOPolicyLossFn``.

    For every call, the bounds start at ``c_low = lower_bound_start`` (a-) and
    ``c_high = upper_bound_start`` (a+). While the positive contribution ratio
    ``rho`` is below ``target_positive_ratio`` (rho_0), ``c_high`` is raised by
    ``upper_bound_step`` (delta_1) up to ``upper_bound_max`` (b+), and then ``c_low``
    is raised by ``lower_bound_step`` (delta_2) up to ``lower_bound_max`` (b-). If the
    target is never met, the widest bounds reached are used. This matches the loop in
    Algorithm 1 whenever ``lower_bound_start + lower_bound_step <= lower_bound_max``
    (true for the paper's settings); when the lower range is empty, the upper bound is
    still searched instead of skipping the search altogether.

    ``rho`` is the share of the policy gradient that comes from positive-advantage
    tokens. A token contributes ``ratio * |advantage|`` (the magnitude of its
    gradient coefficient on ``log pi``) when it is not clipped, and nothing otherwise:

    - a positive-advantage token is clipped when ``ratio > c_high``;
    - a negative-advantage token is clipped when ``ratio < c_low`` or, with the
      dual-clip, when ``ratio > clip_ratio_c``.

    Raising ``c_high`` therefore lets more positive tokens contribute, and raising
    ``c_low`` removes low-ratio negative tokens, so both steps increase ``rho``.
    When no negative-advantage token contributes, ``rho`` is taken as ``1``.

    The bounds are searched independently for every call, i.e. per micro-batch on
    each data-parallel rank.
    """

    def __init__(
        self,
        backend: str = "verl",
        lower_bound_start: float = 0.6,
        lower_bound_max: float = 0.9,
        lower_bound_step: float = 0.02,
        upper_bound_start: float = 1.2,
        upper_bound_max: float = 3.0,
        upper_bound_step: float = 0.05,
        target_positive_ratio: float = 0.4,
        clip_ratio_c: float = 3.0,
        loss_agg_mode: str = "token-mean",
    ) -> None:
        """Initialize the BAPO policy loss function.

        Args:
            backend: The training framework/backend to use (e.g., "verl").
            lower_bound_start: Initial lower clipping bound ``a-``.
            lower_bound_max: Largest allowed lower clipping bound ``b-``.
            lower_bound_step: Step ``delta_2`` used to raise the lower bound.
            upper_bound_start: Initial upper clipping bound ``a+``.
            upper_bound_max: Largest allowed upper clipping bound ``b+``.
            upper_bound_step: Step ``delta_1`` used to raise the upper bound.
            target_positive_ratio: Target share ``rho_0`` of the policy gradient
                contributed by positive-advantage tokens, in ``[0, 1]``.
            clip_ratio_c: Dual-clip constant for negative-advantage tokens, as in PPO.
            loss_agg_mode: Mode for aggregating the loss across tokens.
        """
        super().__init__(backend=backend)
        assert 0.0 < lower_bound_start <= lower_bound_max <= upper_bound_start <= upper_bound_max, (
            "BAPO bounds must satisfy 0 < lower_bound_start <= lower_bound_max "
            "<= upper_bound_start <= upper_bound_max, got "
            f"[{lower_bound_start}, {lower_bound_max}] and [{upper_bound_start}, {upper_bound_max}]."
        )
        assert lower_bound_step >= 0.0, "lower_bound_step must be non-negative."
        assert upper_bound_step >= 0.0, "upper_bound_step must be non-negative."
        assert 0.0 <= target_positive_ratio <= 1.0, "target_positive_ratio must be in [0, 1]."
        assert clip_ratio_c > 1.0, "clip_ratio_c must be greater than 1.0."
        self.lower_bound_start = lower_bound_start
        self.lower_bound_max = lower_bound_max
        self.lower_bound_step = lower_bound_step
        self.upper_bound_start = upper_bound_start
        self.upper_bound_max = upper_bound_max
        self.upper_bound_step = upper_bound_step
        self.target_positive_ratio = target_positive_ratio
        self.clip_ratio_c = clip_ratio_c
        self.loss_agg_mode = loss_agg_mode
        self.bound_schedule = self._build_bound_schedule()

    @staticmethod
    def _num_steps(start: float, end: float, step: float) -> int:
        """Number of whole steps of size ``step`` that fit between ``start`` and ``end``."""
        if step <= 0.0:
            return 0
        return int(math.floor((end - start) / step + 1e-6))

    def _build_bound_schedule(self) -> List[Tuple[float, float]]:
        """Candidate ``(c_low, c_high)`` pairs in the order Algorithm 1 visits them."""
        num_high = self._num_steps(
            self.upper_bound_start, self.upper_bound_max, self.upper_bound_step
        )
        num_low = self._num_steps(
            self.lower_bound_start, self.lower_bound_max, self.lower_bound_step
        )
        schedule = [
            (self.lower_bound_start, round(self.upper_bound_start + k * self.upper_bound_step, 8))
            for k in range(num_high + 1)
        ]
        final_high = schedule[-1][1]
        schedule.extend(
            (round(self.lower_bound_start + k * self.lower_bound_step, 8), final_high)
            for k in range(1, num_low + 1)
        )
        return schedule

    def positive_contribution_ratio(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
        clip_low: torch.Tensor,
        clip_high: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``rho`` for one or more candidate bounds.

        Args:
            ratio: Importance ratio ``pi_theta / pi_old``, shape ``[batch, seq_len]``.
            advantages: Advantages, shape ``[batch, seq_len]``.
            action_mask: Mask of valid tokens, shape ``[batch, seq_len]``.
            clip_low: Lower bounds, shape ``[num_candidates]``.
            clip_high: Upper bounds, shape ``[num_candidates]``.

        Returns:
            torch.Tensor: ``rho`` for every candidate, shape ``[num_candidates]``.
        """
        valid = action_mask.bool()
        r = ratio.detach()[valid].unsqueeze(0)  # [1, num_tokens]
        adv = advantages.detach()[valid].unsqueeze(0)
        weight = r * adv.abs()
        pos_active = (adv > 0) & (r <= clip_high.unsqueeze(1))
        neg_active = (adv < 0) & (r >= clip_low.unsqueeze(1)) & (r <= self.clip_ratio_c)
        pos_contrib = (weight * pos_active).sum(dim=-1)
        neg_contrib = (weight * neg_active).sum(dim=-1)
        total = (pos_contrib + neg_contrib).clamp(min=torch.finfo(weight.dtype).tiny)
        return torch.where(neg_contrib > 0, pos_contrib / total, torch.ones_like(total))

    def select_bounds(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Run the bound search of Algorithm 1.

        Returns:
            Tuple[float, float, float]: The selected ``c_low``, ``c_high`` and the
            resulting positive contribution ratio ``rho``.
        """
        device = ratio.device
        dtype = torch.float32 if ratio.dtype in (torch.float16, torch.bfloat16) else ratio.dtype
        clip_low = torch.tensor([b[0] for b in self.bound_schedule], device=device, dtype=dtype)
        clip_high = torch.tensor([b[1] for b in self.bound_schedule], device=device, dtype=dtype)
        rho = self.positive_contribution_ratio(
            ratio.to(dtype), advantages.to(dtype), action_mask, clip_low, clip_high
        )
        met = rho >= self.target_positive_ratio
        # Index of the first candidate that meets the target. A sentinel appended at the
        # end makes argmax fall back to the last (widest) candidate when none does.
        num_candidates = len(self.bound_schedule)
        index_t = torch.cat([met, met.new_ones(1)]).int().argmax().clamp(max=num_candidates - 1)
        index, rho_value = torch.stack([index_t.to(rho.dtype), rho[index_t]]).tolist()
        c_low, c_high = self.bound_schedule[int(index)]
        return c_low, c_high, rho_value

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        negative_approx_kl = torch.clamp(logprob - old_logprob, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = masked_mean(-negative_approx_kl, action_mask)

        clip_low, clip_high, rho = self.select_bounds(ratio, advantages, action_mask)

        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, clip_low, clip_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_clip_frac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), action_mask)

        pg_losses3 = -advantages * self.clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = masked_mean(
            torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), action_mask
        )
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

        pg_loss = aggregate_loss(pg_losses, action_mask, loss_agg_mode=self.loss_agg_mode)
        metrics = {
            "pg_clipfrac": pg_clip_frac.detach().item(),
            "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_loss": pg_loss.detach().item(),
            "bapo_clip_low": clip_low,
            "bapo_clip_high": clip_high,
            "bapo_positive_ratio": rho,
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        """Default arguments, following Sec. 5.1 of the BAPO paper."""
        return {
            "lower_bound_start": 0.6,
            "lower_bound_max": 0.9,
            "lower_bound_step": 0.02,
            "upper_bound_start": 1.2,
            "upper_bound_max": 3.0,
            "upper_bound_step": 0.05,
            "target_positive_ratio": 0.4,
            "clip_ratio_c": 3.0,
            "loss_agg_mode": "token-mean",
        }
