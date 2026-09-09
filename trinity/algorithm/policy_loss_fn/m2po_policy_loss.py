"""Second-Moment Trust Policy Optimization (M2PO) policy loss.

Implemented from the final M2PO formulation in:
https://openreview.net/forum?id=IIgl5MWelz

The implementation is independently adapted to Trinity-RFT's policy-loss API.
The authors' Apache-2.0 reference repository is available at:
https://github.com/Infini-AI-Lab/M2PO
"""

import math
from collections.abc import Mapping
from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


def find_active_stochastic_settings(config: object) -> list[str]:
    """Return nonzero model settings that make train/eval log-probs differ."""
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        config = to_dict()

    active: list[str] = []
    stochastic_terms = (
        "dropout",
        "layerdrop",
        "jitter",
        "noise",
        "drop_path",
        "stochastic_depth",
    )

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                visit(child, child_path)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        normalized_path = path.lower()
        is_qat_enable = normalized_path.endswith("qat.enable")
        if not is_qat_enable and not any(term in normalized_path for term in stochastic_terms):
            return

        numeric_value: float | None = None
        if isinstance(value, bool):
            numeric_value = float(value)
        elif isinstance(value, (int, float)):
            numeric_value = float(value)
        elif isinstance(value, str):
            try:
                numeric_value = float(value)
            except ValueError:
                pass
        if numeric_value is not None and numeric_value != 0.0:
            active.append(f"{path}={value}")

    visit(config, "")
    return sorted(active)


def find_r2_router_replay_settings(config: object) -> list[str]:
    """Return every nested router-replay setting that selects R2 mode."""
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        config = to_dict()

    active: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if (
                    str(key).lower() == "mode"
                    and path.lower().endswith("router_replay")
                    and str(child).upper() == "R2"
                ):
                    active.append(f"{child_path}={child}")
                visit(child, child_path)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(config, "")
    return sorted(active)


def find_active_stochastic_modules(model: torch.nn.Module) -> list[str]:
    """Return modules with known train/eval behavior differences."""
    active: list[str] = []
    for name, module in model.named_modules():
        module_path = name or "<root>"
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            active.append(f"{module_path}:{type(module).__name__}")
        elif isinstance(module, torch.nn.modules.dropout._DropoutNd) and float(module.p) != 0.0:
            active.append(f"{module_path}:{type(module).__name__}(p={module.p})")
    return sorted(active)


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
        kept_trust = torch.ones(trust_count, dtype=torch.bool, device=trust_values.device)

        if trust_count and m2_before > m2_threshold:
            sorted_values, order = torch.sort(trust_values, stable=True)
            prefix_counts = torch.arange(
                1,
                trust_count + 1,
                device=sorted_values.device,
                dtype=sorted_values.dtype,
            )
            prefix_means = torch.cumsum(sorted_values, dim=0) / prefix_counts
            keep_count = int((prefix_means <= m2_threshold).sum().item())

            kept_trust[order[keep_count:]] = False

        flat_final_mask[flat_trust_indices[~kept_trust]] = False

        final_mask = flat_final_mask.view_as(valid_mask)

        kept_trust_values = trust_values[kept_trust]
        m2_after = kept_trust_values.mean().item() if kept_trust_values.numel() else 0.0

        valid_count = int(valid_mask.sum().item())
        masked_count = trust_count - int(kept_trust.sum().item())

    metrics = {
        "m2_before": m2_before,
        "m2_after": m2_after,
        "masked_fraction": masked_count / valid_count if valid_count else 0.0,
        "trust_region_fraction": trust_count / valid_count if valid_count else 0.0,
    }
    return final_mask, metrics


def compute_m2po_mask(
    logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    action_mask: torch.Tensor,
    advantages: torch.Tensor,
    m2_threshold: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute one M2PO mask over a complete optimizer mini-batch.

    This function deliberately has no distributed or micro-batch fallback. The
    caller must first collect the current-policy log probabilities for the
    complete global mini-batch, because Algorithm 1 sorts all training tokens
    together and the selection is nonlinear in the batch partition.
    """
    if not math.isfinite(m2_threshold) or m2_threshold < 0:
        raise ValueError("m2_threshold must be finite and non-negative.")
    if any(tensor.shape != logprob.shape for tensor in (old_logprob, action_mask, advantages)):
        raise ValueError("M2PO inputs must have identical token-aligned shapes.")

    log_ratio = logprob - old_logprob
    valid_log_ratio = log_ratio[action_mask.bool()]
    if valid_log_ratio.numel() and not torch.isfinite(valid_log_ratio).all().item():
        raise ValueError("M2PO received non-finite log-probability ratios.")

    return _compute_m2po_mask(
        log_ratio=log_ratio,
        action_mask=action_mask,
        advantages=advantages,
        m2_threshold=m2_threshold,
    )


class M2POPolicyLossFn(PolicyLossFn):
    """M2PO loss for stable policy optimization with stale rollouts."""

    _runtime_keys = {"batch_num_tokens", "dp_size"}

    def __init__(
        self,
        backend: str = "verl",
        m2_threshold: float = 0.04,
        loss_agg_mode: str = "token-mean",
        log_ratio_clip: float = 20.0,
    ) -> None:
        """Initialize M2PO with its second-moment threshold."""
        super().__init__(backend=backend)
        if not math.isfinite(m2_threshold) or m2_threshold < 0:
            raise ValueError("m2_threshold must be finite and non-negative.")
        if not math.isfinite(log_ratio_clip) or log_ratio_clip <= 0:
            raise ValueError("log_ratio_clip must be finite and positive.")
        if loss_agg_mode != "token-mean":
            raise ValueError(
                "M2PO requires loss_agg_mode='token-mean' to preserve the paper's "
                "original valid-token denominator."
            )
        self.m2_threshold = m2_threshold
        self.loss_agg_mode = loss_agg_mode
        self.log_ratio_clip = log_ratio_clip

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        m2po_mask: torch.Tensor,
        batch_num_tokens: Optional[int] = None,
        dp_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the final-paper M2PO objective.

        The mask is non-differentiable. Following footnote 2 of the paper, the
        masked objective is normalized by all valid tokens, not only the tokens
        that remain after M2PO masking.
        """
        if any(
            tensor.shape != logprob.shape
            for tensor in (old_logprob, action_mask, advantages, m2po_mask)
        ):
            raise ValueError("M2PO inputs must have identical token-aligned shapes.")
        if bool((m2po_mask.bool() & ~action_mask.bool()).any().item()):
            raise ValueError("m2po_mask cannot select tokens outside action_mask.")

        log_ratio = logprob - old_logprob
        valid_log_ratio = log_ratio[action_mask.bool()]
        if valid_log_ratio.numel() and not torch.isfinite(valid_log_ratio).all().item():
            raise ValueError("M2PO received non-finite log-probability ratios.")

        clamped_log_ratio = torch.clamp(
            log_ratio, min=-self.log_ratio_clip, max=self.log_ratio_clip
        )
        stable_log_ratio = log_ratio + (clamped_log_ratio - log_ratio).detach()
        ratio = torch.exp(stable_log_ratio)
        pg_losses = -advantages * ratio
        masked_pg_losses = torch.where(m2po_mask.bool(), pg_losses, torch.zeros_like(pg_losses))
        if batch_num_tokens is None:
            pg_loss = aggregate_loss(
                masked_pg_losses,
                action_mask,
                loss_agg_mode=self.loss_agg_mode,
            )
        else:
            global_token_count = float(batch_num_tokens)
            data_parallel_size = 1 if dp_size is None else int(dp_size)
            if not math.isfinite(global_token_count) or global_token_count < 0:
                raise ValueError("batch_num_tokens must be finite and non-negative.")
            if data_parallel_size <= 0:
                raise ValueError("dp_size must be positive.")
            pg_loss = masked_pg_losses.sum() / max(global_token_count, 1e-8) * data_parallel_size

        metrics = {
            "pg_loss": pg_loss.detach().item(),
            "ppo_kl": masked_mean(-stable_log_ratio, action_mask).detach().item(),
            "ratio/mean": masked_mean(ratio, action_mask).detach().item(),
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
