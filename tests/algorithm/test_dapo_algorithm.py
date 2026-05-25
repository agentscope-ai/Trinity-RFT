"""Tests for DAPO algorithm registration and paper-aligned pipeline operators."""

import unittest
from unittest.mock import patch

import torch

from trinity.algorithm import ALGORITHM_TYPE
from trinity.algorithm.algorithm import AlgorithmType, DAPOAlgorithm, GRPOAlgorithm
from trinity.buffer.operators.filters.reward_filter import (
    DAPODynamicSamplingFilter,
    MaskResponseTruncatedOperator,
)
from trinity.common.experience import EID, Experience
from trinity.common.rewards.dapo_reward import MathDAPORewardFn


class TestDAPOAlgorithm(unittest.TestCase):
    def test_registry(self):
        cls = ALGORITHM_TYPE.get("dapo")
        self.assertIs(cls, DAPOAlgorithm)
        self.assertTrue(issubclass(cls, AlgorithmType))

    def test_default_config(self):
        config = DAPOAlgorithm.default_config()
        self.assertEqual(config["repeat_times"], 16)
        self.assertEqual(config["advantage_fn"], "grpo")
        self.assertEqual(config["policy_loss_fn"], "ppo")
        self.assertEqual(config["kl_penalty_fn"], "none")
        self.assertEqual(config["kl_loss_fn"], "none")
        self.assertEqual(config["sample_strategy"], "default")

    def test_no_reference_policy(self):
        self.assertFalse(DAPOAlgorithm.use_reference)
        self.assertTrue(GRPOAlgorithm.use_reference)


class TestDAPODynamicSamplingFilter(unittest.TestCase):
    def _exp(self, task: int, run: int, accuracy: float, reward: float | None = None) -> Experience:
        return Experience(
            eid=EID(task=task, run=run),
            tokens=torch.zeros(5),
            prompt_length=2,
            reward=reward if reward is not None else accuracy,
            metrics={"accuracy": accuracy},
        )

    def test_drops_all_correct(self):
        filt = DAPODynamicSamplingFilter()
        exps = [self._exp(0, i, 1.0, reward=1.0) for i in range(4)]
        kept, metrics = filt.process(exps)
        self.assertEqual(len(kept), 0)
        self.assertEqual(metrics["dropped_all_correct"], 4)

    def test_drops_all_wrong(self):
        filt = DAPODynamicSamplingFilter()
        exps = [self._exp(0, i, -1.0, reward=-1.0) for i in range(4)]
        kept, metrics = filt.process(exps)
        self.assertEqual(len(kept), 0)
        self.assertEqual(metrics["dropped_all_wrong"], 4)

    def test_keeps_mixed_correctness(self):
        filt = DAPODynamicSamplingFilter()
        exps = [
            self._exp(0, 0, 1.0, reward=1.0),
            self._exp(0, 1, -1.0, reward=-1.0),
            self._exp(0, 2, 1.0, reward=0.5),
        ]
        kept, metrics = filt.process(exps)
        self.assertEqual(len(kept), 3)
        self.assertEqual(metrics["filtered_count"], 0)

    def test_keeps_group_when_overlong_differs_but_accuracy_mixed(self):
        """All-correct with different total rewards must still drop (accuracy-only)."""
        filt = DAPODynamicSamplingFilter()
        exps = [
            self._exp(0, 0, 1.0, reward=1.0),
            self._exp(0, 1, 1.0, reward=0.5),
        ]
        kept, metrics = filt.process(exps)
        self.assertEqual(len(kept), 0)
        self.assertEqual(metrics["dropped_all_correct"], 2)


class TestMaskResponseTruncatedOperator(unittest.TestCase):
    def test_zeros_action_mask(self):
        op = MaskResponseTruncatedOperator()
        exp = Experience(
            eid=EID(task=0, run=0),
            tokens=torch.zeros(5),
            prompt_length=2,
            truncate_status="response_truncated",
            action_mask=torch.ones(3, dtype=torch.bool),
        )
        kept, metrics = op.process([exp])
        self.assertEqual(len(kept), 1)
        self.assertTrue(torch.all(kept[0].action_mask == 0))
        self.assertEqual(metrics["masked_truncated_count"], 1)


class TestMathDAPORewardFn(unittest.TestCase):
    @patch("trinity.common.rewards.dapo_reward.compute_score")
    def test_symmetric_accuracy(self, mock_compute_score):
        mock_compute_score.side_effect = [1.0, 0.0]
        fn = MathDAPORewardFn(enable_overlong_penalty=False)
        good = fn(response="x", response_token=torch.zeros(10), truth="42")
        bad = fn(response="y", response_token=torch.zeros(10), truth="42")
        self.assertEqual(good["accuracy"], 1.0)
        self.assertEqual(bad["accuracy"], -1.0)


if __name__ == "__main__":
    unittest.main()
