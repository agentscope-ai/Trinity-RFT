import unittest

from datasets import load_dataset
from transformers import AutoTokenizer

from tests.tools import (
    get_model_path,
    get_unittest_dataset_config,
    get_vision_language_model_path,
)
from trinity.buffer.schema import FORMATTER
from trinity.common.config import FormatConfig, StorageConfig
from trinity.common.constants import PromptType
from trinity.common.experience import Experience


class TestFormatter(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(get_model_path())

    def test_sft_messages_formatter(self):
        config = FormatConfig(
            prompt_type=PromptType.MESSAGES,
            messages_key="message_list",
        )
        formatter = FORMATTER.get("sft")(tokenizer_path=get_model_path(), format_config=config)
        sample = {
            "message_list": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }

        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        self.assertTrue(exp.prompt_length < len(exp.tokens))
        sequence = self.tokenizer.decode(exp.tokens)

        self.assertIn("Hi", sequence)
        self.assertIn("Hello", sequence)

        # test tool

        sample = {
            "messages": [
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
            ],
            "tools": [
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
            ],
        }
        config = FormatConfig(
            prompt_type=PromptType.MESSAGES,
            messages_key="messages",
            tools_key="tools",
            enable_concatenated_multi_turn=False,
        )
        formatter = FORMATTER.get("sft")(tokenizer_path=get_model_path(), format_config=config)
        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        self.assertTrue(exp.prompt_length < len(exp.tokens))
        self.assertIsNotNone(exp.action_mask)
        self.assertEqual(len(exp.action_mask) + exp.prompt_length, len(exp.tokens))
        # assert action mask is all true
        self.assertTrue(all(exp.action_mask.tolist()))
        sequence = self.tokenizer.decode(exp.tokens)
        self.assertIn("What's the weather like in Beijing today?", sequence)
        self.assertIn("Let me get the weather for you.", sequence)
        self.assertIn(
            "The weather in Beijing today is sunny with a temperature of 22°C and humidity at 45%. It's a pleasant day!",
            sequence,
        )
        self.assertIn("get_weather", sequence)

        config = FormatConfig(
            prompt_type=PromptType.MESSAGES,
            messages_key="messages",
            tools_key="tools",
            enable_concatenated_multi_turn=True,
        )
        formatter = FORMATTER.get("sft")(tokenizer_path=get_model_path(), format_config=config)
        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        self.assertTrue(exp.prompt_length < len(exp.tokens))
        self.assertIsNotNone(exp.action_mask)
        self.assertEqual(len(exp.action_mask) + exp.prompt_length, len(exp.tokens))
        self.assertTrue(any(exp.action_mask.tolist()) and not all(exp.action_mask.tolist()))
        prompt = self.tokenizer.decode(exp.tokens[: exp.prompt_length])
        response = self.tokenizer.decode(exp.tokens[exp.prompt_length :])
        self.assertIn("What's the weather like in Beijing today?", prompt)
        self.assertNotIn("Let me get the weather for you.", prompt)
        self.assertIn("Let me get the weather for you.", response)
        self.assertNotIn(
            "The weather in Beijing today is sunny with a temperature of 22°C and humidity at 45%. It's a pleasant day!",
            prompt,
        )
        self.assertIn(
            "The weather in Beijing today is sunny with a temperature of 22°C and humidity at 45%. It's a pleasant day!",
            response,
        )

    def test_sft_plaintext_formatter(self):
        # with system prompt key
        config = FormatConfig(
            prompt_type=PromptType.PLAINTEXT,
            system_prompt_key="system",
            system_prompt="You are a programmer.",  # has lower priority than system_prompt_key
            prompt_key="prompt",
            response_key="response",
        )
        formatter = FORMATTER.get("sft")(tokenizer_path=get_model_path(), format_config=config)
        sample = {
            "system": "You are a helpful assistant.",
            "prompt": "What is 2+2?",
            "response": "2+2=4",
        }
        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        self.assertTrue(exp.prompt_length < len(exp.tokens))
        # detokenize exp.tokens into text
        sequence = self.tokenizer.decode(exp.tokens)
        self.assertIn("You are a helpful assistant.", sequence)
        self.assertIn("What is 2+2?", sequence)
        self.assertIn("2+2=4", sequence)

        # with system prompt
        config = FormatConfig(
            prompt_type=PromptType.PLAINTEXT,
            system_prompt="You are a programmer.",
            prompt_key="prompt",
            response_key="response",
        )
        formatter = FORMATTER.get("sft")(tokenizer_path=get_model_path(), format_config=config)

        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        self.assertTrue(exp.prompt_length < len(exp.tokens))
        # detokenize exp.tokens into text
        sequence = self.tokenizer.decode(exp.tokens)
        self.assertIn("You are a programmer.", sequence)
        self.assertIn("What is 2+2?", sequence)
        self.assertIn("2+2=4", sequence)

    def test_dpo_plaintext_formatter(self):
        config = FormatConfig(
            prompt_type=PromptType.PLAINTEXT,
            prompt_key="prompt",
            chosen_key="chosen",
            rejected_key="rejected",
        )
        formatter = FORMATTER.get("dpo")(tokenizer_path=get_model_path(), format_config=config)
        sample = {"prompt": "What is 2+2?", "chosen": "2+2=4", "rejected": "2+2=5"}
        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.chosen)
        self.assertIsNotNone(exp.rejected)
        self.assertIsNotNone(exp.prompt_length)
        prompt = self.tokenizer.decode(exp.tokens)
        chosen = self.tokenizer.decode(exp.chosen)
        rejected = self.tokenizer.decode(exp.rejected)
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("2+2=4", chosen)
        self.assertIn("2+2=5", rejected)
        self.assertNotIn("What is 2+2?", chosen)
        self.assertNotIn("What is 2+2?", rejected)
        self.assertNotIn("2+2=4", prompt)
        self.assertNotIn("2+2=5", prompt)

    def test_dpo_messages_formatter(self):
        config = FormatConfig(
            prompt_type=PromptType.MESSAGES,
            messages_key="messages",
            chosen_key="chosen",
            rejected_key="rejected",
        )
        formatter = FORMATTER.get("dpo")(tokenizer_path=get_model_path(), format_config=config)
        sample = {
            "messages": [
                {"role": "user", "content": "What is your name?"},
            ],
            "chosen": [
                {"role": "assistant", "content": "My name is Assistant."},
            ],
            "rejected": [{"role": "assistant", "content": "I don't have a favorite color."}],
        }
        exp = formatter.format(sample)
        self.assertIsInstance(exp, Experience)
        self.assertIsNotNone(exp.tokens)
        self.assertIsNotNone(exp.prompt_length)
        # detokenize exp.tokens into text
        prompt = self.tokenizer.decode(exp.tokens)
        chosen = self.tokenizer.decode(exp.chosen)
        rejected = self.tokenizer.decode(exp.rejected)
        self.assertIn("What is your name?", prompt)
        self.assertIn("My name is Assistant.", chosen)
        self.assertIn("I don't have a favorite color.", rejected)

    def test_task_formatter(self):
        sample = {
            "question": "1+1=",
            "answer": "2",
            "workflow": "math_rm_workflow",
            "reward": "math_boxed_reward",
        }
        config = StorageConfig(
            is_eval=True,
            default_workflow_type="math_boxed_workflow",
            workflow_args={"use_base": True, "with_think": True},
        )
        formatter = FORMATTER.get("task")(config=config)
        task = formatter.format(sample)
        from trinity.common.workflows.customized_math_workflows import MathBoxedWorkflow

        self.assertEqual(task.workflow, MathBoxedWorkflow)
        self.assertTrue(task.workflow_args.get("use_base"))
        self.assertTrue(task.workflow_args.get("with_think"))
        self.assertEqual(task.raw_task, sample)

        config = StorageConfig(
            is_eval=False,
            default_workflow_type="math_workflow",
            default_reward_fn_type="math_reward",
            workflow_args={"use_base": False, "with_think": True},
        )
        formatter = FORMATTER.get("task")(config=config)
        task = formatter.format(sample)
        from trinity.common.rewards.math_reward import MathRewardFn
        from trinity.common.workflows.workflow import MathWorkflow

        self.assertEqual(task.workflow, MathWorkflow)
        self.assertEqual(task.reward_fn, MathRewardFn)
        self.assertFalse(task.workflow_args.get("use_base"))
        self.assertTrue(task.workflow_args.get("with_think"))
        self.assertEqual(task.raw_task, sample)

        config = StorageConfig(
            is_eval=False,
            default_workflow_type="math_workflow",
            workflow_args={"use_base": True, "with_think": False},
            format=FormatConfig(
                workflow_key="workflow",
                reward_fn_key="reward",
            ),
        )
        formatter = FORMATTER.get("task")(config=config)
        task = formatter.format(sample)
        from trinity.common.rewards.math_reward import MathBoxedRewardFn
        from trinity.common.workflows.math_rm_workflow import MathRMWorkflow

        self.assertEqual(task.workflow, MathRMWorkflow)
        self.assertEqual(task.reward_fn, MathBoxedRewardFn)
        self.assertTrue(task.workflow_args.get("use_base"))
        self.assertFalse(task.workflow_args.get("with_think"))
        self.assertEqual(task.raw_task, sample)

    def test_multi_modal_sft_formatter(self):
        storage_config = get_unittest_dataset_config("geometry")

        formatter = FORMATTER.get("sft")(
            tokenizer_path=get_vision_language_model_path(), format_config=storage_config.format
        )
        self.assertIsNotNone(formatter.processor)
        IMAGE_TOKEN_ID = formatter.processor.image_token_id
        ds = load_dataset(storage_config.path, split=storage_config.split)
        count = 0
        for sample in ds:
            exp = formatter.format(sample)
            self.assertIsInstance(exp, Experience)
            self.assertIsNotNone(exp.tokens)
            self.assertIn(IMAGE_TOKEN_ID, exp.tokens)
            self.assertIsNotNone(exp.prompt_length)
            self.assertTrue(exp.prompt_length < len(exp.tokens))
            self.assertIsNotNone(exp.multi_modal_inputs)
            self.assertTrue(len(exp.multi_modal_inputs) > 0)
            count += 1
        self.assertEqual(count, 8)  # there are total 8 samples in geometry dataset
