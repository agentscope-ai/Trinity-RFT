"""Unit tests for the M2PO policy loss."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from trinity.algorithm import ALGORITHM_TYPE, POLICY_LOSS_FN
from trinity.algorithm.algorithm import M2POAlgorithm
from trinity.algorithm.policy_loss_fn.m2po_policy_loss import (
    M2POPolicyLossFn,
    _compute_m2po_mask,
    compute_m2po_mask,
    find_active_stochastic_modules,
    find_active_stochastic_settings,
    find_r2_router_replay_settings,
)
from trinity.common.config import AlgorithmConfig, Config
from trinity.common.config_validator import AlgorithmConfigValidator
from trinity.trainer.verl.losses import TrinityPolicyLoss


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
        m2po_mask=torch.tensor([[True, False]]),
    )
    loss.backward()

    assert loss.item() == pytest.approx(-0.5)
    assert torch.allclose(logprob.grad, torch.tensor([[-0.5, 0.0]]))
    assert metrics["pg_loss"] == pytest.approx(-0.5)


def test_empty_action_mask_returns_zero_loss_and_metrics():
    """An empty generated-token mask is handled without division errors."""
    logprob = torch.tensor([[1.0, -1.0]], requires_grad=True)
    loss_fn = M2POPolicyLossFn()

    loss, metrics = loss_fn(
        logprob=logprob,
        old_logprob=torch.zeros_like(logprob),
        action_mask=torch.zeros_like(logprob, dtype=torch.bool),
        advantages=torch.ones_like(logprob),
        m2po_mask=torch.zeros_like(logprob, dtype=torch.bool),
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["pg_loss"] == pytest.approx(0.0)


def test_rejects_invalid_threshold_and_non_finite_ratios():
    """Invalid settings and numerically unsafe policy ratios fail early."""
    with pytest.raises(ValueError, match="non-negative"):
        M2POPolicyLossFn(m2_threshold=-0.01)
    with pytest.raises(ValueError, match="finite"):
        M2POPolicyLossFn(m2_threshold=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        M2POPolicyLossFn(log_ratio_clip=float("inf"))
    with pytest.raises(ValueError, match="token-mean"):
        M2POPolicyLossFn(loss_agg_mode="seq-mean-token-mean")

    with pytest.raises(ValueError, match="non-finite"):
        compute_m2po_mask(
            logprob=torch.tensor([[float("nan")]]),
            old_logprob=torch.zeros(1, 1),
            action_mask=torch.ones(1, 1, dtype=torch.bool),
            advantages=torch.ones(1, 1),
            m2_threshold=0.04,
        )


def test_numerical_clamp_preserves_the_policy_gradient():
    """The finite forward surrogate must not turn a large valid ratio into a zero gradient."""
    logprob = torch.tensor([[25.0]], requires_grad=True)
    loss_fn = M2POPolicyLossFn(log_ratio_clip=20.0)

    loss, _ = loss_fn(
        logprob=logprob,
        old_logprob=torch.zeros_like(logprob),
        action_mask=torch.ones_like(logprob, dtype=torch.bool),
        advantages=-torch.ones_like(logprob),
        m2po_mask=torch.ones_like(logprob, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == pytest.approx(torch.exp(torch.tensor(20.0)).item())
    assert logprob.grad.item() == pytest.approx(torch.exp(torch.tensor(20.0)).item())


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
    assert config["entropy_loss_fn"] == "none"
    assert "process_group" not in M2POPolicyLossFn().select_keys
    assert "batch_num_tokens" not in M2POPolicyLossFn().select_keys
    assert "m2po_mask" in M2POPolicyLossFn().select_keys


def test_m2po_requires_rollout_behavior_logprobs():
    """A proximal-policy denominator is a different algorithm and must fail closed."""
    trainer = SimpleNamespace(trainer_type="verl")
    valid_config = SimpleNamespace(
        trainer=trainer,
        algorithm=SimpleNamespace(bypass_old_logprobs=True),
    )
    M2POAlgorithm.check_config(valid_config)

    invalid_config = SimpleNamespace(
        trainer=trainer,
        algorithm=SimpleNamespace(bypass_old_logprobs=False),
    )
    with pytest.raises(ValueError, match="rollout behavior policy"):
        M2POAlgorithm.check_config(invalid_config)


def test_m2po_rejects_backends_without_a_global_prepass(monkeypatch):
    """Legacy veRL and Tinker cannot safely construct the optimizer-batch mask."""
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer_type="verl"),
        algorithm=SimpleNamespace(bypass_old_logprobs=True),
    )
    monkeypatch.setattr("trinity.trainer.trainer.is_verl_legacy", lambda: True)
    with pytest.raises(ValueError, match="veRL >= 0.8"):
        M2POAlgorithm.check_config(config)

    monkeypatch.setattr("trinity.trainer.trainer.is_verl_legacy", lambda: False)
    config.trainer.trainer_type = "tinker"
    with pytest.raises(ValueError, match="veRL >= 0.8"):
        M2POAlgorithm.check_config(config)


def test_validator_cannot_bypass_m2po_guards_with_composed_config(monkeypatch):
    """Selecting the loss directly must preserve every M2PO runtime invariant."""
    monkeypatch.setattr("trinity.trainer.trainer.is_verl_legacy", lambda: False)
    validator = AlgorithmConfigValidator()

    tinker_config = Config()
    tinker_config.trainer.trainer_type = "tinker"
    tinker_config.algorithm.algorithm_type = "ppo"
    tinker_config.algorithm.policy_loss_fn = "m2po"
    with pytest.raises(ValueError, match="veRL >= 0.8"):
        validator.validate(tinker_config)

    denominator_config = Config()
    denominator_config.algorithm.algorithm_type = "ppo"
    denominator_config.algorithm.policy_loss_fn = "m2po"
    denominator_config.algorithm.bypass_old_logprobs = False
    with pytest.raises(ValueError, match="rollout behavior policy"):
        validator.validate(denominator_config)

    mismatched_preset = Config()
    mismatched_preset.algorithm.algorithm_type = "m2po"
    mismatched_preset.algorithm.policy_loss_fn = "ppo"
    with pytest.raises(ValueError, match="requires.*policy_loss_fn='m2po'"):
        validator.validate(mismatched_preset)


def test_stochastic_train_eval_settings_are_detected():
    """The two-pass mask must reject settings that change policy log-probs."""
    assert (
        find_active_stochastic_settings(
            {
                "attention_dropout": 0.0,
                "hidden_dropout": "0",
                "router": {"router_jitter_noise": 0},
            }
        )
        == []
    )
    assert find_active_stochastic_settings(
        {
            "attention_dropout": 0.1,
            "decoder": {"layerdrop": 0.2},
            "router_jitter_noise": 0.01,
            "vision_config": {
                "drop_path_rate": 0.15,
                "stochastic_depth_prob": 0.25,
            },
            "megatron": {"qat": {"enable": True}},
        }
    ) == [
        "attention_dropout=0.1",
        "decoder.layerdrop=0.2",
        "megatron.qat.enable=True",
        "router_jitter_noise=0.01",
        "vision_config.drop_path_rate=0.15",
        "vision_config.stochastic_depth_prob=0.25",
    ]


def test_nested_r2_router_replay_settings_are_detected():
    """Engine-specific overrides must not bypass the M2PO R2 guard."""
    assert find_r2_router_replay_settings(
        {
            "router_replay": {"mode": "disabled"},
            "megatron": {"router_replay": {"mode": "r2"}},
            "fsdp": {"router_replay": {"mode": "R3"}},
        }
    ) == ["megatron.router_replay.mode=r2"]
    assert find_r2_router_replay_settings({"router_replay": {"mode": "R3"}}) == []


def test_stochastic_train_eval_modules_are_detected():
    """Model code cannot hide common train/eval randomness from config guards."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Dropout(p=0.1),
        torch.nn.BatchNorm1d(2),
        torch.nn.Dropout(p=0.0),
    )

    assert find_active_stochastic_modules(model) == [
        "1:Dropout(p=0.1)",
        "2:BatchNorm1d",
    ]


def test_full_batch_mask_is_not_recomputed_per_microbatch():
    """Algorithm 1 must select once before a nonlinear micro-batch split."""
    log_ratio = torch.sqrt(torch.tensor([[0.01, 0.09, 0.05]]))
    action_mask = torch.ones_like(log_ratio, dtype=torch.bool)
    advantages = torch.ones_like(log_ratio)

    full_mask, _ = compute_m2po_mask(
        logprob=log_ratio,
        old_logprob=torch.zeros_like(log_ratio),
        action_mask=action_mask,
        advantages=advantages,
        m2_threshold=0.04,
    )
    wrong_split_mask = torch.cat(
        [
            _compute_m2po_mask(log_ratio[:, :2], action_mask[:, :2], advantages[:, :2], 0.04)[0],
            _compute_m2po_mask(log_ratio[:, 2:], action_mask[:, 2:], advantages[:, 2:], 0.04)[0],
        ],
        dim=1,
    )

    assert torch.equal(full_mask, torch.tensor([[True, False, True]]))
    assert torch.equal(wrong_split_mask, torch.tensor([[True, False, False]]))


def test_engine_wrapper_uses_precomputed_mask_and_global_token_denominator():
    """The engine callback consumes the attached mask and preserves global scaling."""
    parameter = torch.tensor(0.0, requires_grad=True)
    coefficients = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    response_length = coefficients.shape[1]
    data = TensorDict(
        {
            "prompts": torch.ones(1, 1, dtype=torch.long),
            "responses": torch.ones(1, response_length, dtype=torch.long),
            "attention_mask": torch.ones(1, response_length + 1, dtype=torch.bool),
            "response_mask": torch.ones(1, response_length, dtype=torch.bool),
            "old_log_probs": torch.zeros(1, response_length),
            "advantages": torch.ones(1, response_length),
            "m2po_mask": torch.ones(1, response_length, dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, batch_num_tokens=4, dp_size=1)
    loss_wrapper = TrinityPolicyLoss(
        AlgorithmConfig(
            policy_loss_fn="m2po",
            policy_loss_fn_args=M2POPolicyLossFn.default_args(),
            kl_loss_fn="none",
            kl_loss_fn_args={"adaptive": False, "kl_coef": 0.0},
            entropy_loss_fn="none",
            entropy_loss_fn_args={"entropy_coef": 0.0},
            loss_agg_mode="token-mean",
        )
    )
    full_sequence_logprob = torch.cat(
        [coefficients.reshape(-1) * parameter, parameter.reshape(1) * 0.0]
    )

    loss, _ = loss_wrapper(
        model_output={"log_probs": full_sequence_logprob},
        data=data,
    )
    loss.backward()

    assert parameter.grad.item() == pytest.approx(-2.5)


def test_loss_fails_closed_without_a_precomputed_mask():
    """A loss callback cannot silently substitute a micro-batch-local mask."""
    loss_fn = M2POPolicyLossFn()
    with pytest.raises(TypeError, match="m2po_mask"):
        loss_fn(
            logprob=torch.zeros(1, 1),
            old_logprob=torch.zeros(1, 1),
            action_mask=torch.ones(1, 1, dtype=torch.bool),
            advantages=torch.ones(1, 1),
        )
