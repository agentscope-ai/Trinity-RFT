# -*- coding: utf-8 -*-
"""Tests for the BAPO policy loss function."""

import unittest

import torch

from trinity.algorithm import ALGORITHM_TYPE
from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN
from trinity.algorithm.policy_loss_fn.bapo_policy_loss import BAPOPolicyLossFn


def reference_positive_ratio(ratio, advantages, mask, c_low, c_high, clip_ratio_c):
    """Token-by-token computation of the positive contribution ratio."""
    pos, neg = 0.0, 0.0
    for r, a, m in zip(ratio.flatten().tolist(), advantages.flatten().tolist(), mask.flatten()):
        if not m:
            continue
        if a > 0 and r <= c_high:
            pos += r * a
        elif a < 0 and c_low <= r <= clip_ratio_c:
            neg += r * -a
    if neg <= 0:
        return 1.0
    return pos / (pos + neg)


def reference_algorithm_1(ratio, advantages, mask, args):
    """Literal transcription of the bound search in Algorithm 1 of the BAPO paper."""
    c_low, c_high = args["lower_bound_start"], args["upper_bound_start"]

    def rho():
        return reference_positive_ratio(
            ratio, advantages, mask, c_low, c_high, args["clip_ratio_c"]
        )

    tol = 1e-6  # float tolerance for the "<=" comparisons
    while (
        rho() < args["target_positive_ratio"]
        and c_low + args["lower_bound_step"] <= args["lower_bound_max"] + tol
    ):
        if c_high + args["upper_bound_step"] <= args["upper_bound_max"] + tol:
            c_high += args["upper_bound_step"]
        else:
            c_low += args["lower_bound_step"]
    return c_low, c_high, rho()


class BAPOPolicyLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        shape = (6, 24)
        self.logprob = 2 * torch.rand(shape) - 1
        self.old_logprob = 2 * torch.rand(shape) - 1
        self.mask = torch.rand(shape) > 0.3
        self.advantages = 2 * torch.rand(shape) - 1
        self.args = BAPOPolicyLossFn.default_args()

    def _call(self, fn, advantages=None):
        return fn(
            log_prob=self.logprob,
            old_log_probs=self.old_logprob,
            response_mask=self.mask,
            advantages=self.advantages if advantages is None else advantages,
        )

    def _ratio(self):
        return torch.exp(self.logprob - self.old_logprob)

    def test_registered(self):
        self.assertIs(POLICY_LOSS_FN.get("bapo"), BAPOPolicyLossFn)
        algorithm = ALGORITHM_TYPE.get("bapo")
        self.assertEqual(algorithm.default_config()["policy_loss_fn"], "bapo")
        self.assertEqual(algorithm.default_config()["advantage_fn"], "grpo")
        self.assertFalse(algorithm.use_critic)

    def test_bound_schedule_follows_algorithm_1(self):
        fn = BAPOPolicyLossFn(**self.args)
        schedule = fn.bound_schedule
        # 1 initial pair + 36 upper steps (1.2 -> 3.0) + 15 lower steps (0.6 -> 0.9)
        self.assertEqual(len(schedule), 52)
        self.assertEqual(schedule[0], (0.6, 1.2))
        self.assertEqual(schedule[1], (0.6, 1.25))
        self.assertEqual(schedule[36], (0.6, 3.0))
        self.assertEqual(schedule[37], (0.62, 3.0))
        self.assertEqual(schedule[-1], (0.9, 3.0))

    def test_matches_literal_algorithm_1(self):
        for seed in range(20):
            torch.manual_seed(seed)
            self.logprob = 2 * torch.rand(self.logprob.shape) - 1
            self.old_logprob = 2 * torch.rand(self.logprob.shape) - 1
            self.advantages = 2 * torch.rand(self.logprob.shape) - 1.2  # mostly negative
            for target in (0.2, 0.4, 0.5, 0.7):
                args = dict(self.args, target_positive_ratio=target)
                fn = BAPOPolicyLossFn(**args)
                c_low, c_high, rho = fn.select_bounds(self._ratio(), self.advantages, self.mask)
                ref_low, ref_high, ref_rho = reference_algorithm_1(
                    self._ratio(), self.advantages, self.mask, args
                )
                self.assertAlmostEqual(c_low, ref_low, places=6)
                self.assertAlmostEqual(c_high, ref_high, places=6)
                self.assertAlmostEqual(rho, ref_rho, places=5)

    def test_no_adjustment_when_target_already_met(self):
        fn = BAPOPolicyLossFn(**dict(self.args, target_positive_ratio=0.0))
        _, metrics = self._call(fn)
        self.assertEqual(metrics["bapo_clip_low"], 0.6)
        self.assertEqual(metrics["bapo_clip_high"], 1.2)

        # all advantages positive: rho is 1 for the initial bounds
        fn = BAPOPolicyLossFn(**self.args)
        _, metrics = self._call(fn, advantages=self.advantages.abs() + 0.1)
        self.assertEqual(metrics["bapo_clip_low"], 0.6)
        self.assertEqual(metrics["bapo_clip_high"], 1.2)
        self.assertEqual(metrics["bapo_positive_ratio"], 1.0)

    def test_upper_bound_raised_first(self):
        # One positive token with ratio 2.0 (clipped at 1.2) and one negative token.
        ratio = torch.tensor([[2.0, 1.0]])
        advantages = torch.tensor([[1.0, -1.0]])
        mask = torch.ones_like(ratio, dtype=torch.bool)
        fn = BAPOPolicyLossFn(**self.args)
        c_low, c_high, rho = fn.select_bounds(ratio, advantages, mask)
        self.assertEqual(c_low, 0.6)
        self.assertAlmostEqual(c_high, 2.0)
        self.assertAlmostEqual(rho, 2.0 / 3.0, places=6)

    def test_lower_bound_raised_after_upper_saturates(self):
        # Positive token is never clipped; negative tokens with small ratios dominate.
        ratio = torch.tensor([[1.0, 0.65, 0.7, 0.95]])
        advantages = torch.tensor([[0.5, -1.0, -1.0, -0.2]])
        mask = torch.ones_like(ratio, dtype=torch.bool)
        fn = BAPOPolicyLossFn(**self.args)
        c_low, c_high, rho = fn.select_bounds(ratio, advantages, mask)
        # 0.5 / (0.5 + 0.65 + 0.7 + 0.19) < 0.4: raising c_high cannot help, so c_low
        # rises past 0.65 (rho = 0.5 / 1.39 < 0.4) and then past 0.7 (rho = 0.5 / 0.69).
        self.assertAlmostEqual(c_high, 3.0)
        self.assertAlmostEqual(c_low, 0.72)
        self.assertAlmostEqual(rho, 0.5 / 0.69, places=5)

    def test_target_unreachable_uses_widest_bounds(self):
        fn = BAPOPolicyLossFn(**self.args)
        _, metrics = self._call(fn, advantages=-(self.advantages.abs() + 0.1))
        self.assertAlmostEqual(metrics["bapo_clip_low"], 0.9)
        self.assertAlmostEqual(metrics["bapo_clip_high"], 3.0)
        self.assertEqual(metrics["bapo_positive_ratio"], 0.0)

    def test_equivalent_to_ppo_with_fixed_bounds(self):
        bapo = BAPOPolicyLossFn(
            lower_bound_start=0.8,
            lower_bound_max=0.8,
            upper_bound_start=1.28,
            upper_bound_max=1.28,
            target_positive_ratio=1.0,
        )
        ppo = POLICY_LOSS_FN.get("ppo")(clip_range_low=0.2, clip_range_high=0.28)
        bapo_loss, bapo_metrics = self._call(bapo)
        ppo_loss, ppo_metrics = self._call(ppo)
        self.assertTrue(torch.allclose(bapo_loss, ppo_loss))
        for key in ("pg_clipfrac", "pg_clipfrac_lower", "ppo_kl", "pg_loss"):
            self.assertAlmostEqual(bapo_metrics[key], ppo_metrics[key], places=6)

    def test_loss_is_differentiable(self):
        self.logprob.requires_grad_(True)
        fn = BAPOPolicyLossFn(**self.args)
        loss, _ = self._call(fn)
        loss.backward()
        self.assertIsNotNone(self.logprob.grad)
        self.assertTrue(torch.isfinite(self.logprob.grad).all())

    def test_empty_mask(self):
        fn = BAPOPolicyLossFn(**self.args)
        loss, metrics = fn(
            log_prob=self.logprob,
            old_log_probs=self.old_logprob,
            response_mask=torch.zeros_like(self.mask),
            advantages=self.advantages,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["bapo_clip_low"], 0.6)
        self.assertEqual(metrics["bapo_clip_high"], 1.2)

    def test_invalid_args(self):
        with self.assertRaises(AssertionError):
            BAPOPolicyLossFn(lower_bound_start=0.9, lower_bound_max=0.6)
        with self.assertRaises(AssertionError):
            BAPOPolicyLossFn(lower_bound_max=1.3, upper_bound_start=1.2)
        with self.assertRaises(AssertionError):
            BAPOPolicyLossFn(upper_bound_step=-0.1)
        with self.assertRaises(AssertionError):
            BAPOPolicyLossFn(target_positive_ratio=1.5)


if __name__ == "__main__":
    unittest.main()
