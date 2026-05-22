# -*- coding: utf-8 -*-
"""Tests for model utils tokenization helpers."""

import unittest

import torch
import transformers

from tests.tools import (
    CHAT_TEMPLATE_QWEN2_5,
    get_model_path,
    get_vision_language_model_path,
)
from trinity.common.models.utils import (
    tokenize_and_mask_messages_default,
    tokenize_and_mask_messages_hf,
)


class TestTokenizer(unittest.TestCase):
    def test_action_mask(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"},
            {
                "role": "assistant",
                "content": "I'm sorry, but as an AI language model, I don't have access to real-time weather information. To get accurate weather information for your location, you can check a weather website or app, or look outside if possible.",
            },
            {"role": "user", "content": "OK, thanks!"},
            {
                "role": "assistant",
                "content": "You're welcome! If you have any other questions, feel free to ask.",
            },
        ]
        tokenizer = transformers.AutoTokenizer.from_pretrained(get_model_path())
        inputs = tokenize_and_mask_messages_default(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=CHAT_TEMPLATE_QWEN2_5,
        )
        inputs_hf = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=CHAT_TEMPLATE_QWEN2_5,
        )
        self.assertEqual(inputs["input_ids"].shape, inputs_hf["input_ids"].shape)
        self.assertEqual(inputs["assistant_masks"].shape, inputs_hf["assistant_masks"].shape)
        self.assertTrue(torch.equal(inputs["input_ids"], inputs_hf["input_ids"]))
        self.assertTrue(torch.equal(inputs["assistant_masks"], inputs_hf["assistant_masks"]))

    def test_action_mask_with_tools(self):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to various tools. Use them when needed to help users.",
            },
            {"role": "user", "content": "What's the weather like in Beijing today?"},
            {
                "role": "assistant",
                "content": "Let me get the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Beijing", "unit": "celsius"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"temperature": 22, "condition": "sunny", "humidity": 45}',
                "tool_call_id": "call_abc123",
            },
            {
                "role": "assistant",
                "content": "The weather in Beijing today is sunny with a temperature of 22°C and humidity at 45%. It's a pleasant day!",
            },
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        tokenizer = transformers.AutoTokenizer.from_pretrained(get_model_path())
        inputs = tokenize_and_mask_messages_default(
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
            chat_template=CHAT_TEMPLATE_QWEN2_5,
        )
        inputs_hf = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
            chat_template=CHAT_TEMPLATE_QWEN2_5,
        )
        self.assertEqual(inputs["input_ids"].shape, inputs_hf["input_ids"].shape)
        self.assertEqual(inputs["assistant_masks"].shape, inputs_hf["assistant_masks"].shape)
        self.assertTrue(torch.equal(inputs["input_ids"], inputs_hf["input_ids"]))
        self.assertTrue(torch.equal(inputs["assistant_masks"], inputs_hf["assistant_masks"]))


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

        inputs = tokenize_and_mask_messages_default(
            tokenizer=self.tokenizer,
            messages=messages,
            enable_thinking=True,
        )
        assistant_mask = inputs["assistant_masks"][0]
        prompt_length = assistant_mask.argmax().item()

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

        inputs = tokenize_and_mask_messages_default(
            tokenizer=self.tokenizer,
            messages=messages,
            enable_thinking=True,
        )
        assistant_mask = inputs["assistant_masks"][0]
        prompt_length = assistant_mask.argmax().item()

        self.assertTrue(torch.equal(assistant_mask, torch.zeros(13, dtype=torch.int)))
        self.assertEqual(prompt_length, 0)

    def test_first_message_is_assistant(self):
        messages = [
            {"role": "assistant", "content": "I start first.", "reasoning_content": "starting"},
            {"role": "user", "content": "Then me."},
            {"role": "assistant", "content": "Final reply.", "reasoning_content": "ending"},
        ]

        inputs = tokenize_and_mask_messages_default(
            tokenizer=self.tokenizer,
            messages=messages,
            enable_thinking=True,
        )
        assistant_mask = inputs["assistant_masks"][0]
        prompt_length = assistant_mask.argmax().item()

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

        inputs = tokenize_and_mask_messages_default(
            tokenizer=self.processor,
            messages=messages,
            enable_thinking=True,
        )
        assistant_mask = inputs["assistant_masks"][0]
        prompt_length = assistant_mask.argmax().item()
        mm_token_type_ids = inputs["mm_token_type_ids"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        if "Qwen3.5" in get_vision_language_model_path():
            # For Qwen3.5
            expected_mask = torch.tensor([0] * 271 + [1] * 18, dtype=torch.int)
            expected_prompt_length = 271
            expected_mm_token_type_ids = torch.tensor(
                [[0] * 4 + [1] * 156 + [0] * 129], dtype=torch.int
            )
            expected_pixel_values_shape = (624, 1536)
            expected_image_grid_thw = torch.tensor([[1, 24, 26]], dtype=torch.int)

        self.assertTrue(
            torch.equal(
                assistant_mask,
                expected_mask,
            )
        )
        self.assertEqual(prompt_length, expected_prompt_length)
        self.assertTrue(
            torch.equal(
                mm_token_type_ids,
                expected_mm_token_type_ids,
            )
        )
        self.assertEqual(pixel_values.shape, expected_pixel_values_shape)
        self.assertTrue(
            torch.equal(
                image_grid_thw,
                expected_image_grid_thw,
            )
        )
