# -*- coding: utf-8 -*-
"""Tests for model utils tokenization helpers."""

import unittest

import torch
import transformers

from tests.tools import get_model_path
from trinity.common.models.utils import tokenize_and_mask_messages_default


class TestTokenizeAndMaskMessagesDefault(unittest.TestCase):
    def setUp(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(get_model_path())
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

        self.assertTrue(
            torch.equal(
                assistant_mask,
                torch.tensor([0] * 24 + [1] * 14, dtype=torch.int),
            )
        )
        self.assertEqual(prompt_length, 24)

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
            {"role": "assistant", "content": "I start first.", "reasoning": "starting"},
            {"role": "user", "content": "Then me."},
            {"role": "assistant", "content": "Final reply.", "reasoning": "ending"},
        ]

        token_ids, assistant_mask, prompt_length = tokenize_and_mask_messages_default(
            tokenizer=self.tokenizer,
            messages=messages,
            enable_thinking=True,
        )

        self.assertTrue(
            torch.equal(
                assistant_mask,
                torch.tensor([0] * 20 + [1] * 9, dtype=torch.int),
            )
        )
        self.assertEqual(prompt_length, 20)


if __name__ == "__main__":
    unittest.main()
