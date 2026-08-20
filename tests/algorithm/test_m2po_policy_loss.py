"""Unit tests for the M2PO policy loss."""

import pytest
import torch

from trinity.algorithm import ALGORITHM_TYPE, POLICY_LOSS_FN
from trinity.algorithm.policy_loss_fn.m2po_policy_loss import (
    M2POPolicyLossFn,
    _compute_m2po_mask,
)


def test_masks_largest_trust_region_token_until_m2_is_below_threshold():
    """The largest active token is removed until the M2 budget is met."""
    log_ratio = torch.tensor([[0.1, 0.2, 0.3]])
    action_mask = torch.ones_like(log_ratio, dtype=torch.bool)
    advantages = torch.ones_like(log_ratio)

    mask, metrics = _compute_m2po_mask(log_ratio, action_mask, advantages, 0.04)

    assert torch.equal(mask, torch.tensor([[True, True, False]]))
    assert metrics["m2_before"] == pytest.approx((0.01 + 0.04 + 0.09) / 3)
    assert metrics["m2_after"] == pytest.approx(0.025)
    assert metrics["masked_fraction"] == pytest.approx(1 / 3)


def test_only_constrains_the_ppo_trust_region_quadrants():
    """High-M2 tokens outside PPO's active clipping quadrants remain usable."""
    log_ratio = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    advantages = torch.tensor([[-1.0, 1.0, 1.0, -1.0]])
    action_mask = torch.ones_like(log_ratio, dtype=torch.bool)

    mask, _ = _compute_m2po_mask(log_ratio, action_mask, advantages, 0.1)

    assert torch.equal(mask, torch.tensor([[True, True, False, False]]))


def test_vectorized_mask_matches_algorithm_one_reference():
    """The optimized sort implementation matches the paper's iterative loop."""
    threshold = 0.04
    for seed in range(10):
        generator = torch.Generator().manual_seed(seed)
        log_ratio = 0.8 * torch.randn(4, 7, generator=generator)
        advantages = torch.randn(4, 7, generator=generator)
        action_mask = torch.rand(4, 7, generator=generator) > 0.2

        actual, _ = _compute_m2po_mask(
            log_ratio,
            action_mask,
            advantages,
            threshold,
        )

        expected = action_mask.clone()
        trust_region = action_mask & (
            ((advantages > 0) & (log_ratio > 0)) | ((advantages < 0) & (log_ratio < 0))
        )
        active = torch.nonzero(trust_region.reshape(-1), as_tuple=False).squeeze(-1)
        m2 = log_ratio.float().square().reshape(-1)
        while active.numel() and m2[active].mean() > threshold:
            largest = torch.argmax(m2[active])
            expected.reshape(-1)[active[largest]] = False
            active = torch.cat((active[:largest], active[largest + 1 :]))

        assert torch.equal(actual, expected)


def test_loss_uses_the_original_token_denominator_and_masks_gradients():
    """Masked tokens keep the paper's original denominator but have no gradient."""
    logprob = torch.tensor([[0.0, 1.0]], requires_grad=True)
    old_logprob = torch.zeros_like(logprob)
    action_mask = torch.ones_like(logprob, dtype=torch.bool)
    advantages = torch.ones_like(logprob)
    loss_fn = M2POPolicyLossFn(m2_threshold=0.1)

    loss, metrics = loss_fn(
        logprob=logprob,
        old_logprob=old_logprob,
        action_mask=action_mask,
        advantages=advantages,
    )
    loss.backward()

    assert loss.item() == pytest.approx(-0.5)
    assert torch.allclose(logprob.grad, torch.tensor([[-0.5, 0.0]]))
    assert metrics["m2po/masked_fraction"] == pytest.approx(0.5)


def test_empty_action_mask_returns_zero_loss_and_metrics():
    """An empty generated-token mask is handled without division errors."""
    logprob = torch.tensor([[1.0, -1.0]], requires_grad=True)
    loss_fn = M2POPolicyLossFn()

    loss, metrics = loss_fn(
        logprob=logprob,
        old_logprob=torch.zeros_like(logprob),
        action_mask=torch.zeros_like(logprob, dtype=torch.bool),
        advantages=torch.ones_like(logprob),
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["m2po/masked_fraction"] == pytest.approx(0.0)
    assert metrics["m2po/trust_region_fraction"] == pytest.approx(0.0)


def test_rejects_invalid_threshold_and_non_finite_ratios():
    """Invalid settings and numerically unsafe policy ratios fail early."""
    with pytest.raises(ValueError, match="non-negative"):
        M2POPolicyLossFn(m2_threshold=-0.01)

    loss_fn = M2POPolicyLossFn()
    with pytest.raises(ValueError, match="non-finite"):
        loss_fn(
            logprob=torch.tensor([[float("nan")]]),
            old_logprob=torch.zeros(1, 1),
            action_mask=torch.ones(1, 1, dtype=torch.bool),
            advantages=torch.ones(1, 1),
        )


def test_m2po_is_registered_with_paper_defaults():
    """M2PO is available as a complete Trinity-RFT algorithm preset."""
    assert POLICY_LOSS_FN.get("m2po") is M2POPolicyLossFn
    algorithm = ALGORITHM_TYPE.get("m2po")
    config = algorithm.default_config()

    assert not algorithm.use_reference
    assert config["policy_loss_fn"] == "m2po"
    assert config["advantage_fn"] == "grpo"
    assert config["repeat_times"] == 8
    assert config["kl_penalty_fn"] == "none"
    assert config["kl_loss_fn"] == "none"
