import unittest
from typing import List, Optional

import torch

from trinity.buffer.operators.experience_operator import create_operators
from trinity.buffer.operators.filters.repetition_filter import (
    RepetitionFilter,
    compute_repeat_ratio,
)
from trinity.common.config import OperatorConfig
from trinity.common.experience import EID, Experience

PROMPT = [1, 2, 3, 4]


def make_exp(
    response: List[int],
    task: int = 0,
    run: int = 0,
    action_mask: Optional[List[int]] = None,
) -> Experience:
    """Build an experience with a fixed prompt and the given response tokens."""
    return Experience(
        eid=EID(task=task, run=run),
        tokens=torch.tensor(PROMPT + response, dtype=torch.long),
        prompt_length=len(PROMPT),
        reward=1.0,
        action_mask=None if action_mask is None else torch.tensor(action_mask, dtype=torch.bool),
    )


def diverse(length: int) -> List[int]:
    return list(range(100, 100 + length))


def repetitive(length: int) -> List[int]:
    pattern = [7, 8, 9]
    return (pattern * (length // len(pattern) + 1))[:length]


class TestComputeRepeatRatio(unittest.TestCase):
    def test_diverse_sequence_has_zero_ratio(self):
        self.assertEqual(compute_repeat_ratio(diverse(40), 4), 0.0)

    def test_repetitive_sequence_has_high_ratio(self):
        # 60 tokens -> 57 4-grams, only 3 of them unique
        ratio = compute_repeat_ratio([7, 8, 9] * 20, 4)
        self.assertAlmostEqual(ratio, 1 - 3 / 57)

    def test_sequence_shorter_than_ngram_is_zero(self):
        self.assertEqual(compute_repeat_ratio([1, 2], 4), 0.0)
        self.assertEqual(compute_repeat_ratio([], 4), 0.0)

    def test_single_duplicate(self):
        # 5 tokens -> 4 bigrams: (1,1) (1,1) (1,2) (2,3) -> 3 unique
        self.assertAlmostEqual(compute_repeat_ratio([1, 1, 1, 2, 3], 2), 0.25)


class TestRepetitionFilter(unittest.IsolatedAsyncioTestCase):
    async def test_keeps_diverse_and_drops_repetitive(self):
        op = RepetitionFilter(ngram_size=4, max_repeat_ratio=0.5, min_response_tokens=32)
        exps = [
            make_exp(diverse(64), task=0),
            make_exp(repetitive(64), task=1),
            make_exp(diverse(64), task=2),
        ]
        kept, metrics = await op.process(exps)
        self.assertEqual([e.eid.task for e in kept], [0, 2])
        self.assertEqual(metrics["filtered_count"], 1)
        self.assertEqual(metrics["skipped_empty_count"], 0)
        self.assertEqual(metrics["skipped_short_count"], 0)
        self.assertGreater(metrics["repeat_ratio/max"], 0.9)
        self.assertAlmostEqual(metrics["repeat_ratio/mean"], metrics["repeat_ratio/max"] / 3)

    async def test_empty_and_short_responses_are_kept(self):
        op = RepetitionFilter(ngram_size=4, max_repeat_ratio=0.5, min_response_tokens=32)
        exps = [
            make_exp([], action_mask=[]),  # multi-turn experience with no response tokens
            make_exp(repetitive(64), action_mask=[0] * 64),  # all tokens are observations
            make_exp(repetitive(8)),  # repetitive but too short to judge
            make_exp([5]),  # single token
        ]
        kept, metrics = await op.process(exps)
        self.assertEqual(len(kept), 4)
        self.assertEqual(metrics["filtered_count"], 0)
        self.assertEqual(metrics["skipped_empty_count"], 2)
        self.assertEqual(metrics["skipped_short_count"], 2)
        self.assertNotIn("repeat_ratio/mean", metrics)

    async def test_threshold_is_strict(self):
        # 8 tokens with ngram_size=2 -> 7 bigrams; [1,2,1,2,1,2,3,4] has 4 unique -> 3/7
        tokens = [1, 2, 1, 2, 1, 2, 3, 4]
        ratio = compute_repeat_ratio(tokens, 2)
        op_equal = RepetitionFilter(ngram_size=2, max_repeat_ratio=ratio, min_response_tokens=0)
        kept, _ = await op_equal.process([make_exp(tokens)])
        self.assertEqual(len(kept), 1)
        op_below = RepetitionFilter(
            ngram_size=2, max_repeat_ratio=ratio - 1e-6, min_response_tokens=0
        )
        kept, _ = await op_below.process([make_exp(tokens)])
        self.assertEqual(len(kept), 0)

    async def test_only_generated_tokens_are_inspected(self):
        op = RepetitionFilter(ngram_size=4, max_repeat_ratio=0.5, min_response_tokens=32)
        # Repetitive prompt must not affect the decision.
        exp = Experience(
            eid=EID(task=0),
            tokens=torch.tensor(repetitive(64) + diverse(64), dtype=torch.long),
            prompt_length=64,
            reward=1.0,
        )
        kept, metrics = await op.process([exp])
        self.assertEqual(len(kept), 1)
        self.assertEqual(metrics["repeat_ratio/max"], 0.0)

        # Multi-turn: repetitive observation tokens are masked out, generated tokens diverse.
        response = repetitive(64) + diverse(64)
        mask = [0] * 64 + [1] * 64
        kept, metrics = await op.process([make_exp(response, action_mask=mask)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(metrics["repeat_ratio/max"], 0.0)

        # Multi-turn: diverse observation tokens, repetitive generated tokens -> dropped.
        response = diverse(64) + repetitive(64)
        kept, metrics = await op.process([make_exp(response, action_mask=mask)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(metrics["filtered_count"], 1)

    async def test_disable_with_ratio_one(self):
        op = RepetitionFilter(max_repeat_ratio=1.0, min_response_tokens=0)
        kept, metrics = await op.process([make_exp(repetitive(64))])
        self.assertEqual(len(kept), 1)
        self.assertEqual(metrics["filtered_count"], 0)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            RepetitionFilter(ngram_size=0)
        with self.assertRaises(ValueError):
            RepetitionFilter(max_repeat_ratio=1.5)
        with self.assertRaises(ValueError):
            RepetitionFilter(min_response_tokens=-1)

    async def test_create_from_registry(self):
        ops = create_operators(
            [
                OperatorConfig(
                    name="repetition_filter",
                    args={"ngram_size": 3, "max_repeat_ratio": 0.6, "min_response_tokens": 16},
                )
            ]
        )
        self.assertEqual(len(ops), 1)
        self.assertIsInstance(ops[0], RepetitionFilter)
        self.assertEqual(ops[0].ngram_size, 3)
        self.assertEqual(ops[0].max_repeat_ratio, 0.6)
        self.assertEqual(ops[0].min_response_tokens, 16)
        kept, metrics = await ops[0].process([make_exp(diverse(32)), make_exp(repetitive(32))])
        self.assertEqual(len(kept), 1)
        self.assertEqual(metrics["filtered_count"], 1)


if __name__ == "__main__":
    unittest.main()
