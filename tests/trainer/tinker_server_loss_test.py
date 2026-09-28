"""CPU tests for the optional Tinker server-side PPO loss."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from trinity.algorithm.entropy_loss_fn.entropy_loss_fn import DummyEntropyLossFn
from trinity.algorithm.kl_fn.kl_fn import DummyKLFn, K1Fn, K2Fn
from trinity.algorithm.policy_loss_fn.ppo_policy_loss import PPOPolicyLossFn
from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.trainer.tinker.server_loss import (
    trinity_ppo_metrics,
    with_trinity_ppo_inputs,
)
from trinity.trainer.tinker.tinker_trainer import TinkerTrainerWrapper
from trinity.trainer.tinker.utils import to_tinker_input


def make_wrapper():
    """Construct a trainer without Ray actors or remote clients."""
    wrapper = TinkerTrainerWrapper.__new__(TinkerTrainerWrapper)
    wrapper.config = Config()
    wrapper.config.model.tinker.server_loss_fn = "trinity_ppo"
    wrapper.algorithm = SimpleNamespace(use_reference=True, compute_advantage_in_trainer=False)
    wrapper.policy_loss_fn = PPOPolicyLossFn(backend="tinker", clip_range=0.2)
    wrapper.kl_loss_fn = K2Fn(kl_coef=0.001)
    wrapper.entropy_loss_fn = DummyEntropyLossFn(entropy_coef=0)
    wrapper.loss_agg_mode = "token-mean"
    wrapper.do_fix_actor_microbatch_loss_scale = False
    return wrapper


def make_experience(mask=(1, 0, 1)):
    return Experience(
        tokens=torch.tensor([10, 11, 12, 13, 14]),
        prompt_length=2,
        action_mask=torch.tensor(mask, dtype=torch.bool),
        logprobs=torch.tensor([-0.2, -0.3, -0.4]),
        advantages=torch.tensor([2.0, 3.0, -1.0]),
        reward=1.0,
    )


class ServerLossInputTest(unittest.TestCase):
    def setUp(self):
        self.batch, _, self.inputs = to_tinker_input([make_experience()], MagicMock())
        self.inputs[0]["ref_logprob"] = torch.tensor([-0.1, -0.5, -0.2])

    def test_alignment_preserves_masked_response_tokens(self):
        result = with_trinity_ppo_inputs(self.batch, self.inputs, kl_coef=0.001)[0]
        expected = {
            "target_tokens": [11, 12, 13, 14],
            "weights": [0, 1, 0, 1],
            "logprobs": [0, -0.2, 0, -0.4],
            "advantages": [0, 2, 0, -1],
            "ref_logprobs": [0, -0.1, 0, -0.2],
        }
        for key, values in expected.items():
            torch.testing.assert_close(
                result.loss_fn_inputs[key].to_torch().float(), torch.tensor(values).float()
            )
        self.assertNotIn("logprobs", self.batch[0].loss_fn_inputs)

    def test_scalar_advantage_and_zero_kl_without_reference(self):
        self.inputs[0]["advantages"] = torch.tensor(2.0)
        del self.inputs[0]["ref_logprob"]
        result = with_trinity_ppo_inputs(self.batch, self.inputs, kl_coef=0)[0]
        self.assertNotIn("ref_logprobs", result.loss_fn_inputs)
        self.assertEqual(result.loss_fn_inputs["advantages"].data, [0, 2, 0, 2])

    def test_empty_mask_contributes_zero_and_ignores_masked_nonfinite_values(self):
        self.inputs[0]["action_mask"] = torch.zeros(3)
        self.inputs[0]["old_logprob"] = torch.tensor([float("-inf")] * 3)
        result = with_trinity_ppo_inputs(self.batch, self.inputs, kl_coef=0.001)[0]
        for key in ("weights", "logprobs", "advantages", "ref_logprobs"):
            self.assertEqual(result.loss_fn_inputs[key].data, [0, 0, 0, 0])

    def test_missing_required_values_and_invalid_shapes_are_rejected(self):
        for key in ("old_logprob", "advantages", "ref_logprob"):
            with self.subTest(key=key):
                inputs = [dict(self.inputs[0])]
                del inputs[0][key]
                with self.assertRaisesRegex(ValueError, key):
                    with_trinity_ppo_inputs(self.batch, inputs, kl_coef=0.001)
        self.inputs[0]["old_logprob"] = torch.tensor([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "per response token"):
            with_trinity_ppo_inputs(self.batch, self.inputs, kl_coef=0)

    def test_nonbinary_masks_and_active_nonfinite_values_are_rejected(self):
        self.inputs[0]["action_mask"] = torch.tensor([1, 0.5, 1])
        with self.assertRaisesRegex(ValueError, "binary"):
            with_trinity_ppo_inputs(self.batch, self.inputs, kl_coef=0)
        self.inputs[0]["action_mask"] = torch.ones(3)
        self.inputs[0]["advantages"] = torch.tensor([1.0, float("nan"), 2.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            with_trinity_ppo_inputs(self.batch, self.inputs, kl_coef=0)

    def test_metrics_use_additive_token_counts(self):
        metrics = {
            "loss:sum": 0.3,
            "trinity/response_tokens:sum": 4,
            "trinity/ratio_sum:sum": 6,
            "trinity/ratio_squared_sum:sum": 10,
            "trinity/clipped_tokens:sum": 1,
            "trinity/kl_sum:sum": 0.8,
        }
        self.assertEqual(
            trinity_ppo_metrics(metrics),
            {
                "actor/final_loss": 0.3,
                "actor/ratio_mean": 1.5,
                "actor/ratio_var": 0.25,
                "actor/ratio_clip_fraction": 0.25,
                "actor/kl_token_mean": 0.2,
            },
        )
        self.assertEqual(trinity_ppo_metrics({"trinity/response_tokens:sum": 0}), {})


class ServerLossConfigTest(unittest.TestCase):
    def test_supported_objective_and_disabled_kl(self):
        wrapper = make_wrapper()
        self.assertEqual(
            wrapper._server_loss_fn_config(),
            {"clip_range": 0.2, "clip_ratio_c": 3.0, "kl_coef": 0.001},
        )
        wrapper.kl_loss_fn = DummyKLFn()
        self.assertEqual(wrapper._server_loss_fn_config()["kl_coef"], 0)

    def test_unsupported_policy_options_fail_before_training(self):
        for attribute, value in (
            ("clip_range_high", 0.3),
            ("enable_sequence_masking", True),
            ("fallback_to_policy_gradient", True),
            ("loss_agg_mode", "seq-mean-token-sum"),
        ):
            with self.subTest(attribute=attribute):
                wrapper = make_wrapper()
                setattr(wrapper.policy_loss_fn, attribute, value)
                with self.assertRaisesRegex(ValueError, "does not support"):
                    wrapper._server_loss_fn_config()

    def test_unsupported_loss_options_fail_before_training(self):
        for attribute, value in (
            ("kl_loss_fn", K1Fn()),
            ("entropy_loss_fn", object()),
            ("policy_loss_fn", object()),
            ("loss_agg_mode", "seq-mean-token-sum"),
            ("do_fix_actor_microbatch_loss_scale", True),
        ):
            with self.subTest(attribute=attribute):
                wrapper = make_wrapper()
                setattr(wrapper, attribute, value)
                with self.assertRaises(ValueError):
                    wrapper._server_loss_fn_config()
        wrapper = make_wrapper()
        wrapper.algorithm.use_reference = False
        with self.assertRaisesRegex(ValueError, "reference"):
            wrapper._server_loss_fn_config()


class ServerLossTrainStepTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        wrapper = make_wrapper()
        wrapper.logger = MagicMock()
        wrapper.server_loss_config = wrapper._server_loss_fn_config()
        wrapper._train_step_num = 0
        wrapper.algorithm_config = wrapper.config.algorithm
        wrapper.lr_scheduler_type = "constant"
        wrapper.num_warmup_steps = 0
        wrapper.total_steps = 10
        wrapper.min_lr_ratio = 0
        wrapper.ref_client = SimpleNamespace(
            compute_logprobs_async=AsyncMock(return_value=[None, -0.1, -0.2, -0.3, -0.4])
        )
        calls = []

        async def forward_backward(batch, loss_fn, loss_config):
            calls.append(("forward_backward", len(batch), loss_fn, loss_config))
            result = asyncio_future(SimpleNamespace(metrics={"loss:sum": 0.3}))
            return result

        async def optim_step(params):
            calls.append(("optim_step",))
            return asyncio_future(SimpleNamespace(metrics={}))

        wrapper.actor_client = SimpleNamespace(
            forward_backward_async=AsyncMock(side_effect=forward_backward),
            forward_backward_custom_async=AsyncMock(),
            optim_step_async=AsyncMock(side_effect=optim_step),
        )
        self.wrapper = wrapper
        self.calls = calls

    async def train(self):
        with patch(
            "trinity.trainer.tinker.tinker_trainer.compute_throughout_metrics", return_value={}
        ):
            return await self.wrapper.train_step([make_experience(), make_experience()])

    async def test_one_optimizer_step_and_full_batch_denominator(self):
        metrics = await self.train()
        wrapper, calls = self.wrapper, self.calls
        self.assertEqual(calls[0][1:3], (2, "trinity_ppo"))
        self.assertEqual(calls[0][3]["num_total_datums"], 2)
        self.assertEqual(calls[1], ("optim_step",))
        self.assertEqual(len(calls), 2)
        wrapper.actor_client.forward_backward_custom_async.assert_not_called()
        self.assertEqual(metrics["actor/final_loss"], 0.3)

    async def test_zero_kl_skips_reference_requests(self):
        self.wrapper.server_loss_config["kl_coef"] = 0
        await self.train()
        self.wrapper.ref_client.compute_logprobs_async.assert_not_called()

    async def test_default_path_keeps_client_loss_callback(self):
        self.wrapper.server_loss_config = None
        future = asyncio_future(SimpleNamespace(metrics={"custom_loss": 0.5}))
        self.wrapper.actor_client.forward_backward_custom_async.return_value = future
        metrics = await self.train()
        self.wrapper.actor_client.forward_backward_async.assert_not_called()
        self.wrapper.actor_client.forward_backward_custom_async.assert_awaited_once()
        self.assertEqual(metrics["custom_loss"], 0.5)

    async def test_failed_server_request_does_not_step_optimizer(self):
        # A failed asynchronous result models server-side validation failures.
        future = asyncio.get_running_loop().create_future()
        future.set_exception(ValueError("unsupported loss"))
        self.wrapper.actor_client.forward_backward_async.side_effect = None
        self.wrapper.actor_client.forward_backward_async.return_value = future
        with self.assertRaisesRegex(ValueError, "unsupported loss"):
            await self.train()
        self.wrapper.actor_client.optim_step_async.assert_not_called()


def asyncio_future(result):
    """Return an awaitable result, like a completed SDK API future."""
    future = asyncio.get_running_loop().create_future()
    future.set_result(result)
    return future
