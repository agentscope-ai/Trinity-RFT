"""Second-Moment Trust Policy Optimization (M2PO) policy loss.

Implemented from the final M2PO formulation in:
https://openreview.net/forum?id=IIgl5MWelz

The implementation is independently adapted to Trinity-RFT's policy-loss API.
The authors' Apache-2.0 reference repository is available at:
https://github.com/Infini-AI-Lab/M2PO
"""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


def _compute_m2po_mask(
    log_ratio: torch.Tensor,
    action_mask: torch.Tensor,
    advantages: torch.Tensor,
    m2_threshold: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build the token mask from Algorithm 1 of the M2PO paper.

    M2PO applies its trust-region constraint only where PPO clipping would be
    active: ``advantage > 0 and ratio > 1`` or ``advantage < 0 and ratio < 1``.
    Within those tokens, the largest squared log-ratios are removed until the
    mean second moment of the remaining trust-region tokens is at most the
    configured threshold.

    Args:
        log_ratio: ``log(pi_theta / pi_behavior)`` for every token.
        action_mask: Mask selecting generated tokens that contribute to loss.
        advantages: Token-aligned policy advantages.
        m2_threshold: Upper bound on the mean squared log-ratio.

    Returns:
        The final boolean loss mask and masking diagnostics.
    """
    valid_mask = action_mask.bool()

    with torch.no_grad():
        detached_log_ratio = log_ratio.detach()
        detached_advantages = advantages.detach()
        second_moment = detached_log_ratio.float().square()

        trust_region_mask = valid_mask & (
            ((detached_advantages > 0) & (detached_log_ratio > 0))
            | ((detached_advantages < 0) & (detached_log_ratio < 0))
        )
        final_mask = valid_mask.clone()
        flat_final_mask = final_mask.reshape(-1)

        flat_trust_indices = torch.nonzero(trust_region_mask.reshape(-1), as_tuple=False).squeeze(
            -1
        )
        trust_values = second_moment.reshape(-1)[flat_trust_indices]

        trust_count = int(trust_values.numel())
        m2_before = trust_values.mean().item() if trust_count else 0.0

        if trust_count and m2_before > m2_threshold:
            sorted_values, order = torch.sort(trust_values)
            prefix_counts = torch.arange(
                1,
                trust_count + 1,
                device=sorted_values.device,
                dtype=sorted_values.dtype,
            )
            prefix_means = torch.cumsum(sorted_values, dim=0) / prefix_counts
            keep_count = int((prefix_means <= m2_threshold).sum().item())

            masked_trust_indices = flat_trust_indices[order[keep_count:]]
            flat_final_mask[masked_trust_indices] = False

        final_mask = flat_final_mask.view_as(valid_mask)

        kept_trust_mask = trust_region_mask & final_mask
        kept_trust_values = second_moment[kept_trust_mask]
        m2_after = kept_trust_values.mean().item() if kept_trust_values.numel() else 0.0

        valid_count = int(valid_mask.sum().item())
        masked_count = int((valid_mask & ~final_mask).sum().item())

    metrics = {
        "m2_before": m2_before,
        "m2_after": m2_after,
        "masked_fraction": masked_count / valid_count if valid_count else 0.0,
        "trust_region_fraction": trust_count / valid_count if valid_count else 0.0,
    }
    return final_mask, metrics


class M2POPolicyLossFn(PolicyLossFn):
    """M2PO loss for stable policy optimization with stale rollouts."""

    def __init__(
        self,
        backend: str = "verl",
        m2_threshold: float = 0.04,
        loss_agg_mode: str = "token-mean",
        log_ratio_clip: float = 20.0,
    ) -> None:
        """Initialize M2PO with its second-moment threshold."""
        super().__init__(backend=backend)
        if m2_threshold < 0:
            raise ValueError("m2_threshold must be non-negative.")
        if log_ratio_clip <= 0:
            raise ValueError("log_ratio_clip must be positive.")
        self.m2_threshold = m2_threshold
        self.loss_agg_mode = loss_agg_mode
        self.log_ratio_clip = log_ratio_clip

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the final-paper M2PO objective.

        The mask is non-differentiable. Following footnote 2 of the paper, the
        masked objective is normalized by all valid tokens, not only the tokens
        that remain after M2PO masking.
        """
        log_ratio = logprob - old_logprob
        valid_log_ratio = log_ratio[action_mask.bool()]
        if valid_log_ratio.numel() and not torch.isfinite(valid_log_ratio).all():
            raise ValueError("M2PO received non-finite log-probability ratios.")

        m2po_mask, m2_metrics = _compute_m2po_mask(
            log_ratio=log_ratio,
            action_mask=action_mask,
            advantages=advantages,
            m2_threshold=self.m2_threshold,
        )

        stable_log_ratio = torch.clamp(log_ratio, min=-self.log_ratio_clip, max=self.log_ratio_clip)
        ratio = torch.exp(stable_log_ratio)
        pg_losses = -advantages * ratio
        masked_pg_losses = torch.where(m2po_mask, pg_losses, torch.zeros_like(pg_losses))
        pg_loss = aggregate_loss(
            masked_pg_losses,
            action_mask,
            loss_agg_mode=self.loss_agg_mode,
        )

        metrics = {
            "pg_loss": pg_loss.detach().item(),
            "ppo_kl": masked_mean(-stable_log_ratio, action_mask).detach().item(),
            "ratio/mean": masked_mean(ratio, action_mask).detach().item(),
            **{f"m2po/{name}": value for name, value in m2_metrics.items()},
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        """Return the final paper's default threshold and stable loss settings."""
        return {
            "m2_threshold": 0.04,
            "loss_agg_mode": "token-mean",
            "log_ratio_clip": 20.0,
        }
