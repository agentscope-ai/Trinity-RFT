"""Tests for GiGPO advantage computation and algorithm registration."""

import unittest

import torch

from trinity.algorithm import ADVANTAGE_FN, ALGORITHM_TYPE
from trinity.algorithm.advantage_fn.gigpo_advantage import GiGPOAdvantageFn
from trinity.algorithm.algorithm import AlgorithmType, GiGPOAlgorithm
from trinity.common.experience import EID, Experience


def _make_exp(
    task: int,
    run: int,
    step: int,
    reward: float = 0.0,
    *,
    step_reward: float | None = None,
    env_state_hash: str | None = None,
    action_mask: torch.Tensor | None = None,
) -> Experience:
    info = {}
    if step_reward is not None:
        info["step_reward"] = step_reward
    if env_state_hash is not None:
        info["env_state_hash"] = env_state_hash
    mask = action_mask if action_mask is not None else torch.tensor([1.0, 1.0, 1.0])
    return Experience(
        eid=EID(batch=0, task=task, run=run, step=step),
        tokens=torch.zeros(5),
        prompt_length=2,
        reward=reward,
        info=info,
        action_mask=mask,
    )


class TestGiGPORegistry(unittest.TestCase):
    def test_advantage_fn_registry(self):
        cls = ADVANTAGE_FN.get("gigpo")
        self.assertIs(cls, GiGPOAdvantageFn)

    def test_algorithm_registry(self):
        cls = ALGORITHM_TYPE.get("gigpo")
        self.assertIs(cls, GiGPOAlgorithm)
        self.assertTrue(issubclass(cls, AlgorithmType))


class TestGiGPOAlgorithm(unittest.TestCase):
    def test_default_config(self):
        config = GiGPOAlgorithm.default_config()
        self.assertEqual(config["repeat_times"], 8)
        self.assertEqual(config["advantage_fn"], "gigpo")
        self.assertEqual(config["policy_loss_fn"], "ppo")
        self.assertEqual(config["policy_loss_fn_args"]["loss_agg_mode"], "token-mean")
        self.assertEqual(config["advantage_fn_args"]["omega"], 1.0)
        self.assertEqual(config["advantage_fn_args"]["fnorm"], "none")


class TestGiGPOAdvantageFn(unittest.TestCase):
    def test_episode_advantage_broadcast(self):
        """All steps in a run share the same episode-level component."""
        fn = GiGPOAdvantageFn(omega=0.0, fnorm="none")
        exps = []
        for run, terminal in enumerate([1.0, 0.0]):
            for step in range(3):
                exps.append(
                    _make_exp(
                        task=0,
                        run=run,
                        step=step,
                        reward=terminal,
                        step_reward=terminal if step == 2 else 0.0,
                        env_state_hash=f"unique_{run}_{step}",
                    )
                )
        processed, _ = fn.process(exps)
        run0_adv = [e.advantages for e in processed if e.eid.run == 0]
        run1_adv = [e.advantages for e in processed if e.eid.run == 1]
        for adv in run0_adv[1:]:
            self.assertTrue(torch.allclose(adv, run0_adv[0]))
        for adv in run1_adv[1:]:
            self.assertTrue(torch.allclose(adv, run1_adv[0]))
        # run0 return 1, run1 return 0 -> A^E(0) > A^E(1) with fnorm=none
        self.assertGreater(run0_adv[0].sum().item(), run1_adv[0].sum().item())

    def test_anchor_state_grouping(self):
        """Shared env_state_hash yields non-zero step-level advantage differences."""
        fn = GiGPOAdvantageFn(omega=1.0, gamma=1.0, fnorm="none")
        shared_hash = "anchor_state_A"
        exps = [
            _make_exp(0, 0, 0, step_reward=0.0, env_state_hash=shared_hash),
            _make_exp(0, 0, 1, step_reward=1.0, env_state_hash="other_0_1"),
            _make_exp(0, 1, 0, step_reward=0.0, env_state_hash=shared_hash),
            _make_exp(0, 1, 1, step_reward=0.0, env_state_hash="other_1_1"),
        ]
        for e in exps:
            e.reward = 1.0 if e.eid.run == 0 else 0.0
        processed, metrics = fn.process(exps)
        step0_run0 = next(e for e in processed if e.eid.run == 0 and e.eid.step == 0)
        step0_run1 = next(e for e in processed if e.eid.run == 1 and e.eid.step == 0)
        # Same anchor, different future returns -> different A^S
        self.assertNotAlmostEqual(
            step0_run0.advantages.sum().item(),
            step0_run1.advantages.sum().item(),
            places=5,
        )
        self.assertGreater(metrics["gigpo/anchor_groups_size_gt1"], 0)

    def test_singleton_anchor_zero_step_adv(self):
        """Unique anchor hashes -> A^S = 0; only episode term remains."""
        fn_none = GiGPOAdvantageFn(omega=1.0, fnorm="none")
        fn_zero = GiGPOAdvantageFn(omega=0.0, fnorm="none")
        exps = [
            _make_exp(0, 0, 0, step_reward=0.0, env_state_hash="only_0"),
            _make_exp(0, 1, 0, step_reward=1.0, env_state_hash="only_1"),
        ]
        for e in exps:
            e.reward = float(e.eid.run)
        p_full, _ = fn_none.process(exps)
        p_ep, _ = fn_zero.process(exps)
        for a, b in zip(
            sorted(p_full, key=lambda e: e.eid.run), sorted(p_ep, key=lambda e: e.eid.run)
        ):
            self.assertTrue(torch.allclose(a.advantages, b.advantages))

    def test_omega_zero(self):
        """omega=0 disables step term in combined advantage but still computes A^S metrics."""
        shared = "shared"

        def build():
            exps = [
                _make_exp(0, 0, 0, step_reward=0.0, env_state_hash=shared),
                _make_exp(0, 1, 0, step_reward=1.0, env_state_hash=shared),
            ]
            for e in exps:
                e.reward = float(e.eid.run)
            return exps

        p_zero, metrics = GiGPOAdvantageFn(omega=0.0, fnorm="none").process(build())
        p_one, _ = GiGPOAdvantageFn(omega=1.0, fnorm="none").process(build())
        self.assertGreater(metrics["gigpo/mean_abs_A_E"], 0.0)
        self.assertGreater(metrics["gigpo/mean_abs_A_S"], 0.0)
        z0 = next(e for e in p_zero if e.eid.run == 0).advantages.sum().item()
        z1 = next(e for e in p_zero if e.eid.run == 1).advantages.sum().item()
        o0 = next(e for e in p_one if e.eid.run == 0).advantages.sum().item()
        o1 = next(e for e in p_one if e.eid.run == 1).advantages.sum().item()
        self.assertNotAlmostEqual(z0, o0, places=5)
        self.assertNotAlmostEqual(z1, o1, places=5)

    def test_fnorm_none_vs_std(self):
        shared = "anchor"

        def build():
            return [
                _make_exp(0, 0, 0, step_reward=0.0, env_state_hash=shared),
                _make_exp(0, 1, 0, step_reward=10.0, env_state_hash=shared),
            ]

        fn_none = GiGPOAdvantageFn(omega=1.0, fnorm="none")
        fn_std = GiGPOAdvantageFn(omega=1.0, fnorm="std", epsilon=1e-6)
        p_none, _ = fn_none.process(build())
        p_std, _ = fn_std.process(build())
        a_none = next(e for e in p_none if e.eid.run == 1).advantages.sum().item()
        a_std = next(e for e in p_std if e.eid.run == 1).advantages.sum().item()
        self.assertNotAlmostEqual(a_none, a_std, places=5)

    def test_sparse_reward_fallback(self):
        """Without step_reward in info, use terminal reward on last step only."""
        fn = GiGPOAdvantageFn(omega=0.0, fnorm="none")
        exps = [
            Experience(
                eid=EID(batch=0, task=0, run=0, step=0),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=1.0,
            ),
            Experience(
                eid=EID(batch=0, task=0, run=0, step=1),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=1.0,
            ),
            Experience(
                eid=EID(batch=0, task=0, run=1, step=0),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=0.0,
            ),
            Experience(
                eid=EID(batch=0, task=0, run=1, step=1),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=0.0,
            ),
        ]
        processed, _ = fn.process(exps)
        run0 = [e for e in processed if e.eid.run == 0]
        self.assertEqual(len(run0), 2)
        # Episode return 1 vs 0 -> positive advantage for run 0
        self.assertGreater(run0[0].advantages.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
