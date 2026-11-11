# -*- coding: utf-8 -*-
"""Test for KL functions"""

import unittest

import torch

from trinity.algorithm.kl_fn import KL_FN


class KLFnTest(unittest.TestCase):
    def setUp(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        shape = (4, 10)
        self.logprob = 2 * torch.rand(shape) - 1
        self.ref_logprob = 2 * torch.rand(shape) - 1
        self.old_logprob = 2 * torch.rand(shape) - 1
        self.response_mask = torch.rand(shape) > 0.5

    def test_k1_kl_fn(self):
        kl_fn_cls = KL_FN.get("k1")
        kl_fn = kl_fn_cls(kl_coef=0.01)
        kl = kl_fn.calculate_kl(self.logprob, self.ref_logprob)
        expected_kl = self.logprob - self.ref_logprob
        self.assertTrue(torch.allclose(kl, expected_kl))

    def test_k2_kl_fn(self):
        kl_fn_cls = KL_FN.get("k2")
        kl_fn = kl_fn_cls(kl_coef=0.01)
        kl = kl_fn.calculate_kl(self.logprob, self.ref_logprob)
        expected_kl = (self.logprob - self.ref_logprob).square() * 0.5
        self.assertTrue(torch.allclose(kl, expected_kl))

    def test_k3_kl_fn(self):
        kl_fn_cls = KL_FN.get("k3")
        kl_fn = kl_fn_cls(kl_coef=0.01)
        kl = kl_fn.calculate_kl(self.logprob, self.ref_logprob)
        logr = self.ref_logprob - self.logprob
        expected_kl = logr.exp() - 1 - logr
        self.assertTrue(torch.allclose(kl, expected_kl))

    def test_abs_kl_fn(self):
        kl_fn_cls = KL_FN.get("abs")
        kl_fn = kl_fn_cls(kl_coef=0.01)
        kl = kl_fn.calculate_kl(self.logprob, self.ref_logprob)
        expected_kl = torch.abs(self.logprob - self.ref_logprob)
        self.assertTrue(torch.allclose(kl, expected_kl))

    def test_low_var_kl_fn(self):
        kl_fn_cls = KL_FN.get("low_var_kl")
        kl_fn = kl_fn_cls(kl_coef=0.01)
        kl = kl_fn.calculate_kl(self.logprob, self.ref_logprob)
        kl_intermediate = self.ref_logprob - self.logprob
        kl_intermediate = torch.clamp(kl_intermediate, min=-20, max=20)
        ratio = torch.exp(kl_intermediate)
        expected_kl = torch.clamp((ratio - kl_intermediate - 1).contiguous(), min=-10, max=10)
        self.assertTrue(torch.allclose(kl, expected_kl))

    def test_dummy_kl_fn(self):
        kl_fn_cls = KL_FN.get("none")
        kl_fn = kl_fn_cls(kl_coef=0.01)
        kl = kl_fn.calculate_kl(self.logprob, self.ref_logprob)
        expected_kl = torch.zeros_like(self.logprob)
        self.assertTrue(torch.allclose(kl, expected_kl))

    def test_corrected_k3_fallback(self):
        k3_fn = KL_FN.get("k3")(kl_coef=0.01)
        corrected_k3_fn = KL_FN.get("corrected_k3")(kl_coef=0.01)
        kl_standard = k3_fn.calculate_kl(self.logprob, self.ref_logprob)
        kl_corrected_no_old = corrected_k3_fn.calculate_kl(
            self.logprob, self.ref_logprob, old_logprob=None
        )
        self.assertTrue(torch.allclose(kl_standard, kl_corrected_no_old))

    def test_corrected_k3_with_old_logprob(self):
        corrected_k3_fn = KL_FN.get("corrected_k3")(kl_coef=0.01)
        kl_corrected = corrected_k3_fn.calculate_kl(
            self.logprob, self.ref_logprob, self.old_logprob
        )
        logr = self.ref_logprob - self.logprob
        kl_standard = logr.exp() - 1 - logr
        log_ratio_is = self.logprob - self.old_logprob
        ratio_is = log_ratio_is.exp()
        ratio_is = torch.clamp(ratio_is, min=0.0, max=2.0)
        expected_kl = ratio_is * kl_standard
        self.assertTrue(torch.allclose(kl_corrected, expected_kl))

    def test_corrected_k3_same_policy(self):
        k3_fn = KL_FN.get("k3")(kl_coef=0.01)
        corrected_k3_fn = KL_FN.get("corrected_k3")(kl_coef=0.01)
        kl_standard = k3_fn.calculate_kl(self.logprob, self.ref_logprob)
        kl_corrected = corrected_k3_fn.calculate_kl(self.logprob, self.ref_logprob, self.logprob)
        self.assertTrue(torch.allclose(kl_standard, kl_corrected, rtol=1e-4, atol=1e-6))

    def test_corrected_k3_loss(self):
        corrected_k3_fn = KL_FN.get("corrected_k3")(kl_coef=0.01)
        kl_loss, metrics = corrected_k3_fn.calculate_kl_loss(
            logprob=self.logprob,
            ref_logprob=self.ref_logprob,
            response_mask=self.response_mask,
            loss_agg_mode="token-mean",
            old_logprob=self.old_logprob,
        )
        self.assertEqual(kl_loss.dim(), 0)
        self.assertIn("kl_loss", metrics)
        self.assertIn("kl_coef", metrics)
        self.assertEqual(metrics["kl_coef"], 0.01)

    def test_kl_loss_aggregation_modes(self):
        corrected_k3_fn = KL_FN.get("corrected_k3")(kl_coef=0.01)
        kl_loss_mean, _ = corrected_k3_fn.calculate_kl_loss(
            logprob=self.logprob,
            ref_logprob=self.ref_logprob,
            response_mask=self.response_mask,
            loss_agg_mode="token-mean",
            old_logprob=self.old_logprob,
        )
        kl_loss_sum, _ = corrected_k3_fn.calculate_kl_loss(
            logprob=self.logprob,
            ref_logprob=self.ref_logprob,
            response_mask=self.response_mask,
            loss_agg_mode="seq-mean-token-sum",
            old_logprob=self.old_logprob,
        )
        self.assertGreater(kl_loss_sum.item(), kl_loss_mean.item())

    def test_adaptive_kl_penalty_updates_coef(self):
        from verl import DataProto

        batch_size, prompt_len, resp_len = 4, 3, 5
        response_mask = torch.ones(batch_size, resp_len)
        batch = DataProto.from_dict(
            {
                "responses": torch.zeros(batch_size, resp_len, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, prompt_len + resp_len),
                "response_mask": response_mask,
                "token_level_scores": torch.zeros(batch_size, resp_len),
                "old_log_probs": torch.zeros(batch_size, resp_len),
                "ref_log_prob": torch.full((batch_size, resp_len), -0.5),
            }
        )
        # k1 KL is 0.5 per token, so the proportional error is clipped to +0.2 / -0.2
        for target_kl, multiplier in [(0.1, 1.2), (1.0, 0.8)]:
            kl_fn = KL_FN.get("k1")(adaptive=True, kl_coef=0.1, target_kl=target_kl, horizon=10)
            _, metrics = kl_fn.apply_kl_penalty_to_reward(batch)
            self.assertAlmostEqual(metrics["kl"], 0.5, places=6)
            self.assertAlmostEqual(metrics["kl_coef"], 0.1)
            # verl AdaptiveKLController: coef *= 1 + clip(kl / target - 1, -0.2, 0.2) * n / horizon
            expected = 0.1 * (1 + (multiplier - 1) * batch_size / 10)
            self.assertAlmostEqual(kl_fn.kl_coef, expected, places=6)
