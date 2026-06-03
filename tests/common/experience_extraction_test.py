import io
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import pybase64
import torch

from trinity.common.models.experience_extraction import convert_api_output_to_experience


class TestExperienceExtraction(TestCase):
    def test_convert_completion_output_extracts_sglang_routed_experts(self):
        routed_experts = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ],
            dtype=torch.int32,
        )
        routed_experts_b64 = pybase64.b64encode(routed_experts.numpy().tobytes()).decode("utf-8")
        output = SimpleNamespace(
            model="mock-moe-model",
            prompt_token_ids=[10, 11],
            sglext={"routed_experts": routed_experts_b64},
            choices=[
                SimpleNamespace(
                    token_ids=[12, 13],
                    message=SimpleNamespace(content="done"),
                    logprobs=SimpleNamespace(
                        content=[SimpleNamespace(logprob=-0.1), SimpleNamespace(logprob=-0.2)]
                    ),
                )
            ],
        )

        experiences = convert_api_output_to_experience(output, routed_experts_layout=(2, 2))

        self.assertEqual(len(experiences), 1)
        exp = experiences[0]
        self.assertEqual(exp.prompt_length, 2)
        self.assertEqual(exp.response_text, "done")
        self.assertTrue(torch.equal(exp.logprobs, torch.tensor([-0.1, -0.2], dtype=torch.float32)))
        self.assertIsNotNone(exp.routed_experts)
        self.assertEqual(exp.routed_experts.dtype, torch.uint8)
        self.assertEqual(tuple(exp.routed_experts.shape), (3, 2, 2))
        self.assertTrue(torch.equal(exp.routed_experts, routed_experts.to(torch.uint8)))

    def test_convert_completion_output_ignores_invalid_routed_experts_shape(self):
        output = SimpleNamespace(
            model="mock-moe-model",
            prompt_token_ids=[10, 11],
            sglext={"routed_experts": "aW52YWxpZA=="},
            choices=[
                SimpleNamespace(
                    token_ids=[12, 13],
                    message=SimpleNamespace(content="done"),
                    logprobs=None,
                )
            ],
        )

        experiences = convert_api_output_to_experience(output, routed_experts_layout=(2, 2))

        self.assertEqual(len(experiences), 1)
        self.assertIsNone(experiences[0].routed_experts)

    def test_convert_completion_output_extracts_vllm_routed_experts(self):
        routed_experts = np.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ],
            dtype=np.uint8,
        )
        buffer = io.BytesIO()
        np.save(buffer, routed_experts)
        routed_experts_b64 = pybase64.b64encode(buffer.getvalue()).decode("utf-8")
        output = SimpleNamespace(
            model="mock-moe-model",
            prompt_token_ids=[10, 11],
            choices=[
                SimpleNamespace(
                    token_ids=[12, 13],
                    message=SimpleNamespace(content="done"),
                    logprobs=SimpleNamespace(
                        content=[SimpleNamespace(logprob=-0.1), SimpleNamespace(logprob=-0.2)]
                    ),
                    routed_experts=routed_experts_b64,
                )
            ],
        )

        experiences = convert_api_output_to_experience(output, routed_experts_layout=(2, 2))

        self.assertEqual(len(experiences), 1)
        exp = experiences[0]
        self.assertIsNotNone(exp.routed_experts)
        self.assertEqual(exp.routed_experts.dtype, torch.uint8)
        self.assertEqual(tuple(exp.routed_experts.shape), (3, 2, 2))
        self.assertTrue(torch.equal(exp.routed_experts, torch.tensor(routed_experts)))
