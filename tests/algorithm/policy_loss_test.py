# -*- coding: utf-8 -*-
"""Test for policy loss functions"""

import unittest

import torch
from verl import DataProto

from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN


class VerlPolicyLossTest(unittest.TestCase):
    def setUp(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        shape = (5, 20)
        self.logprob = 2 * torch.rand(shape) - 1
        self.input_data = DataProto.from_dict(
            {
                "old_log_probs": 2 * torch.rand(shape) - 1,
                "ref_log_prob": 2 * torch.rand(shape) - 1,
                "response_mask": torch.rand(shape) > 0.5,
                "advantages": 2 * torch.rand(shape) - 1,
                "expert_mask": torch.rand(shape[0]) > 0.5,
            }
        )

    def test_ppo_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("ppo")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        ppo_loss = torch.tensor(0.26889559626579285)
        pg_clipfrac = torch.tensor(0.3541666567325592)
        ppo_kl = torch.tensor(-0.21663446724414825)
        pg_clipfrac_lower = torch.tensor(0.0625)
        self.assertTrue(torch.allclose(loss, ppo_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_clipfrac"]), pg_clipfrac))
        self.assertTrue(torch.allclose(torch.tensor(metrics["ppo_kl"]), ppo_kl))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_loss"]), ppo_loss))
        self.assertTrue(
            torch.allclose(torch.tensor(metrics["pg_clipfrac_lower"]), pg_clipfrac_lower)
        )

    def test_gspo_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("gspo")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        gspo_loss_expected = torch.tensor(0.27235108613967896)
        pg_clipfrac_expected = torch.tensor(0.375)
        ppo_kl_seq_expected = torch.tensor(-0.21027061343193054)
        ppo_kl_expected = torch.tensor(-0.21663446724414825)
        self.assertTrue(torch.allclose(loss, gspo_loss_expected))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_clipfrac"]), pg_clipfrac_expected))
        self.assertTrue(torch.allclose(torch.tensor(metrics["ppo_kl_seq"]), ppo_kl_seq_expected))
        self.assertTrue(torch.allclose(torch.tensor(metrics["ppo_kl"]), ppo_kl_expected))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_loss"]), gspo_loss_expected))

    def test_sft_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("sft")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        sft_loss = torch.tensor(-0.07560186833143234)
        self.assertTrue(torch.allclose(loss, sft_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["sft_loss"]), sft_loss))

    def test_dpo_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("dpo")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        dpo_loss = torch.tensor(0.5406752228736877)
        chosen_reward = torch.tensor(0.7082431316375732)
        rejected_reward = torch.tensor(0.3757950782775879)
        accuracy_mean = torch.tensor(1.0)
        self.assertTrue(torch.allclose(loss, dpo_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["chosen_reward"]), chosen_reward))
        self.assertTrue(torch.allclose(torch.tensor(metrics["rejected_reward"]), rejected_reward))
        self.assertTrue(torch.allclose(torch.tensor(metrics["accuracy_mean"]), accuracy_mean))
        self.assertTrue(torch.allclose(torch.tensor(metrics["dpo_loss"]), dpo_loss))

    def test_opmd_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("opmd")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        opmd_loss = torch.tensor(-0.009589947760105133)
        self.assertTrue(torch.allclose(loss, opmd_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["opmd_loss"]), opmd_loss))

    def test_mix_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("mix")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        mix_loss = torch.tensor(0.6298247575759888)
        pg_clipfrac = torch.tensor(0.7777777910232544)
        ppo_kl = torch.tensor(-1.0737695693969727)
        pg_loss = torch.tensor(0.6921210885047913)
        sft_loss = torch.tensor(0.06915830634534359)
        pg_clipfrac_lower = torch.tensor(0.2222222238779068)
        self.assertTrue(torch.allclose(loss, mix_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["usual/pg_clipfrac"]), pg_clipfrac))
        self.assertTrue(torch.allclose(torch.tensor(metrics["usual/ppo_kl"]), ppo_kl))
        self.assertTrue(torch.allclose(torch.tensor(metrics["usual/pg_loss"]), pg_loss))
        self.assertTrue(
            torch.allclose(torch.tensor(metrics["usual/pg_clipfrac_lower"]), pg_clipfrac_lower)
        )
        self.assertTrue(torch.allclose(torch.tensor(metrics["expert/sft_loss"]), sft_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["loss"]), mix_loss))

    def test_ppo_policy_loss_with_sequence_masking(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("ppo")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn_args["enable_sequence_masking"] = True
        policy_loss_fn_args["delta_sequence_masking"] = 0.1
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        ppo_loss_masked = torch.tensor(0.22175675630569458)
        pg_clipfrac = torch.tensor(0.3541666567325592)
        ppo_kl = torch.tensor(-0.21663446724414825)
        pg_clipfrac_lower = torch.tensor(0.0625)
        masked_tokens = torch.tensor(0.16666666666631944)
        mean_sequence_kl = torch.tensor(-0.21027061343193054)
        self.assertTrue(torch.allclose(loss, ppo_loss_masked))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_clipfrac"]), pg_clipfrac))
        self.assertTrue(torch.allclose(torch.tensor(metrics["ppo_kl"]), ppo_kl))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_loss"]), ppo_loss_masked))
        self.assertTrue(
            torch.allclose(torch.tensor(metrics["pg_clipfrac_lower"]), pg_clipfrac_lower)
        )
        self.assertTrue(
            torch.allclose(torch.tensor(metrics["seq_mask/masked_tokens"]), masked_tokens)
        )
        self.assertTrue(
            torch.allclose(torch.tensor(metrics["seq_mask/mean_sequence_kl"]), mean_sequence_kl)
        )

    def test_sapo_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("sapo")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        sapo_loss = torch.tensor(0.24060271680355072)
        ppo_kl = torch.tensor(-0.21663446724414825)
        avg_soft_gate = torch.tensor(2.3191137313842773)
        avg_ratio = torch.tensor(1.630766749382019)
        pos_adv_frac = torch.tensor(0.3958333432674408)
        self.assertTrue(torch.allclose(loss, sapo_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["sapo_loss"]), sapo_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["ppo_kl"]), ppo_kl))
        self.assertTrue(torch.allclose(torch.tensor(metrics["avg_soft_gate"]), avg_soft_gate))
        self.assertTrue(torch.allclose(torch.tensor(metrics["avg_ratio"]), avg_ratio))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pos_adv_frac"]), pos_adv_frac))

    def test_sapo_policy_loss_gradient(self):
        # SAPO (arXiv:2511.20347, Eq. 7-8): d/dlogpi of -f(r) A is -w r A with w = 4 p (1 - p),
        # p = sigmoid(tau (r - 1)); the weight is 1 at r = 1 and decays away from it.
        policy_loss_fn_cls = POLICY_LOSS_FN.get("sapo")
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_cls.default_args())
        batch = self.input_data.batch
        logprob = self.logprob.clone().requires_grad_(True)
        loss, _ = policy_loss_fn(log_prob=logprob, **batch)
        loss.backward()

        mask = batch["response_mask"].float()
        advantages = batch["advantages"]
        ratio = torch.exp(self.logprob - batch["old_log_probs"])
        tau = torch.where(advantages > 0, torch.tensor(1.0), torch.tensor(1.05))
        p = torch.sigmoid(tau * (ratio - 1))
        expected = -4 * p * (1 - p) * ratio * advantages * mask / mask.sum()
        self.assertTrue(torch.allclose(logprob.grad, expected, atol=1e-6))
