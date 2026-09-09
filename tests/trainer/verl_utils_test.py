import unittest
from unittest import mock

import pytest
import torch

from trinity.common.experience import EID, Experience
from trinity.trainer.verl.utils import to_data_proto as engine_to_data_proto
from trinity.trainer.verl_legacy.utils import to_data_proto


class TestToDataProtoRoutedExperts(unittest.TestCase):
    def test_to_data_proto_pads_routed_experts_to_full_sequence(self):
        exp1_routed_experts = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
                [[13, 14], [15, 16]],
            ],
            dtype=torch.uint8,
        )
        exp2_routed_experts = torch.tensor(
            [
                [[21, 22], [23, 24]],
                [[25, 26], [27, 28]],
            ],
            dtype=torch.uint8,
        )
        experiences = [
            Experience(
                eid=EID(batch=1, task=1, run=1, step=1),
                tokens=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
                prompt_length=2,
                routed_experts=exp1_routed_experts,
            ),
            Experience(
                eid=EID(batch=1, task=2, run=1, step=1),
                tokens=torch.tensor([20, 21, 22], dtype=torch.int32),
                prompt_length=1,
                routed_experts=exp2_routed_experts,
            ),
        ]

        batch = to_data_proto(experiences, pad_token_id=0, model=object(), logger=mock.Mock())

        self.assertIn("routed_experts", batch.batch)
        routed_experts = batch.batch["routed_experts"]
        self.assertEqual(routed_experts.dtype, torch.uint8)
        self.assertEqual(tuple(routed_experts.shape), (2, 5, 2, 2))

        expected_exp1 = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
                [[13, 14], [15, 16]],
                [[0, 0], [0, 0]],
            ],
            dtype=torch.uint8,
        )
        expected_exp2 = torch.tensor(
            [
                [[0, 0], [0, 0]],
                [[21, 22], [23, 24]],
                [[25, 26], [27, 28]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            dtype=torch.uint8,
        )
        self.assertTrue(torch.equal(routed_experts[0], expected_exp1))
        self.assertTrue(torch.equal(routed_experts[1], expected_exp2))


@pytest.mark.parametrize("logprob_count", [None, 0, 1], ids=["missing", "empty", "partial"])
def test_to_data_proto_rejects_incomplete_rollout_logprobs(logprob_count):
    """Missing behavior logprobs must not be silently padded into a valid training batch."""
    logprobs = None if logprob_count is None else torch.zeros(logprob_count)
    experience = Experience(
        tokens=torch.tensor([10, 11, 12], dtype=torch.int32),
        prompt_length=1,
        reward=1.0,
        logprobs=logprobs,
    )

    with pytest.raises(ValueError, match="rollout logprobs|one value"):
        engine_to_data_proto([experience], pad_token_id=0, model=object(), logger=mock.Mock())


def test_to_data_proto_only_pads_between_complete_experiences():
    """Batch padding remains valid after each experience passes the behavior-logprob invariant."""
    experiences = [
        Experience(
            tokens=torch.tensor([10, 11, 12], dtype=torch.int32),
            prompt_length=1,
            reward=1.0,
            logprobs=torch.tensor([-0.1, -0.2]),
        ),
        Experience(
            tokens=torch.tensor([20, 21, 22], dtype=torch.int32),
            prompt_length=2,
            reward=0.0,
            logprobs=torch.tensor([-0.3]),
        ),
    ]

    batch = engine_to_data_proto(experiences, pad_token_id=0, model=object(), logger=mock.Mock())

    assert torch.allclose(
        batch.batch["rollout_log_probs"],
        torch.tensor([[-0.1, -0.2], [-0.3, 0.0]]),
    )
    assert torch.equal(
        batch.batch["response_mask"],
        torch.tensor([[True, True], [True, False]]),
    )
