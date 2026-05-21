# -*- coding: utf-8 -*-
"""Tests for model utils tokenization helpers."""

import unittest

import torch
import transformers

from tests.tools import get_model_path, get_vision_language_model_path
from trinity.common.models.utils import tokenize_and_mask_messages_default


class TestTokenizeAndMaskMessagesDefault(unittest.TestCase):
    """Update this class if you change the default model."""

    def setUp(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(get_model_path())
        self.processor = transformers.AutoProcessor.from_pretrained(
            get_vision_language_model_path()
        )
        return super().setUp()

    def test_normal_conversation_data(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": "greeting"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine.", "reasoning_content": "answering"},
        ]

        token_ids, assistant_mask, prompt_length = tokenize_and_mask_messages_default(
            tokenizer=self.tokenizer,
            messages=messages,
            enable_thinking=True,
        )

        if "Qwen3.5" in get_model_path():
            # For Qwen3.5
            expected_mask = torch.tensor([0] * 26 + [1] * 12, dtype=torch.int)
            expected_prompt_length = 26
        else:
            # For Qwen3
            expected_mask = torch.tensor([0] * 24 + [1] * 14, dtype=torch.int)
            expected_prompt_length = 24

        self.assertTrue(
            torch.equal(
                assistant_mask,
                expected_mask,
            )
        )
        self.assertEqual(prompt_length, expected_prompt_length)

    def test_messages_empty(self):
        with self.assertRaises(ValueError):
            tokenize_and_mask_messages_default(tokenizer=self.tokenizer, messages=[])

    def test_no_assistant_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Still user"},
        ]

        token_ids, assistant_mask, prompt_length = tokenize_and_mask_messages_default(
            tokenizer=self.tokenizer,
            messages=messages,
            enable_thinking=True,
        )

        self.assertTrue(torch.equal(assistant_mask, torch.zeros(13, dtype=torch.int)))
        self.assertEqual(prompt_length, 0)

    def test_first_message_is_assistant(self):
        messages = [
            {"role": "assistant", "content": "I start first.", "reasoning_content": "starting"},
            {"role": "user", "content": "Then me."},
            {"role": "assistant", "content": "Final reply.", "reasoning_content": "ending"},
        ]

        token_ids, assistant_mask, prompt_length = tokenize_and_mask_messages_default(
            tokenizer=self.tokenizer,
            messages=messages,
            enable_thinking=True,
        )

        if "Qwen3.5" in get_model_path():
            # For Qwen3.5
            expected_mask = torch.tensor([0] * 22 + [1] * 9, dtype=torch.int)
            expected_prompt_length = 22
        else:
            # For Qwen3
            expected_mask = torch.tensor([0] * 20 + [1] * 11, dtype=torch.int)
            expected_prompt_length = 20

        self.assertTrue(
            torch.equal(
                assistant_mask,
                expected_mask,
            )
        )
        self.assertEqual(prompt_length, expected_prompt_length)

    def test_mm_messages(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) $\\frac{2}{9}$\n(B) $\\sqrt{5}$\n(C) $0.8 \\cdot \\pi$\n(D) 2.5\n(E) $1+\\sqrt{2}$",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "The user wants me to solve a geometry problem based on an image.",
            },
        ]
        if self.processor is None:
            return

        token_ids, assistant_mask, prompt_length = tokenize_and_mask_messages_default(
            tokenizer=self.processor,
            messages=messages,
            enable_thinking=True,
        )

        if "Qwen3.5" in get_vision_language_model_path():
            # For Qwen3.5
            expected_mask = torch.tensor([0] * 271 + [1] * 18, dtype=torch.int)
            expected_prompt_length = 271

        self.assertTrue(
            torch.equal(
                assistant_mask,
                expected_mask,
            )
        )
        self.assertEqual(prompt_length, expected_prompt_length)
