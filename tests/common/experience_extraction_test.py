import asyncio
import io
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pybase64
import torch

from trinity.common.models.experience_extraction import (
    HistoryRecordingStream,
    convert_api_output_to_experience,
)
from trinity.common.models.vllm_model import vLLMRolloutModel


class TestExperienceExtraction(TestCase):
    def test_vllm_generate_maps_finish_reason_per_output(self):
        def logprob(value: float):
            return {0: SimpleNamespace(logprob=value)}

        model = vLLMRolloutModel.__new__(vLLMRolloutModel)
        model.tokenizer = MagicMock()
        model.tokenizer.decode.return_value = "prompt"
        model._handle_prompt_truncation = MagicMock(return_value=([1, 2], True))
        model._extract_routed_experts = MagicMock(return_value=None)
        model._generate_internal = AsyncMock(
            return_value=SimpleNamespace(
                prompt_token_ids=[1, 2],
                outputs=[
                    SimpleNamespace(
                        token_ids=[3],
                        logprobs=[logprob(-0.1)],
                        text="truncated",
                        finish_reason="length",
                    ),
                    SimpleNamespace(
                        token_ids=[4],
                        logprobs=[logprob(-0.2)],
                        text="complete",
                        finish_reason="stop",
                    ),
                ],
            )
        )

        experiences = asyncio.run(model.generate("prompt", n=2))

        self.assertEqual(experiences[0].truncate_status, "response_truncated")
        self.assertIsNone(experiences[1].truncate_status)

    def test_convert_completion_output_maps_finish_reason(self):
        output = SimpleNamespace(
            prompt_token_ids=[1, 2],
            choices=[
                SimpleNamespace(
                    token_ids=[3],
                    logprobs=None,
                    message=SimpleNamespace(content="truncated"),
                    finish_reason="length",
                    routed_experts=None,
                ),
                SimpleNamespace(
                    token_ids=[4],
                    logprobs=None,
                    message=SimpleNamespace(content="complete"),
                    finish_reason="stop",
                    routed_experts=None,
                ),
            ],
        )

        experiences = convert_api_output_to_experience(output)

        self.assertEqual(experiences[0].truncate_status, "response_truncated")
        self.assertIsNone(experiences[1].truncate_status)

    def test_stream_conversion_preserves_final_finish_reason(self):
        chunks = [
            SimpleNamespace(
                prompt_token_ids=[1, 2],
                choices=[
                    SimpleNamespace(
                        index=0,
                        token_ids=[3],
                        logprobs=None,
                        delta=SimpleNamespace(content="go"),
                        finish_reason=None,
                    )
                ],
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        token_ids=None,
                        logprobs=None,
                        delta=SimpleNamespace(content=None),
                        finish_reason="length",
                    )
                ]
            ),
        ]
        history = []

        list(HistoryRecordingStream(iter(chunks), history))

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].response_text, "go")
        self.assertEqual(history[0].truncate_status, "response_truncated")

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
