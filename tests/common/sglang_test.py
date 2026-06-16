import asyncio

import torch
from parameterized import parameterized_class
from transformers import AutoConfig, AutoTokenizer

from tests.tools import (
    CHAT_TEMPLATE,
    RayUnittestBaseAsync,
    get_model_path,
    get_moe_model_path,
    get_template_config,
)
from trinity.common.models.allocator import Allocator


async def prepare_engines(engines, auxiliary_engines):
    prepare_model_refs = []
    for engine in engines:
        prepare_model_refs.append(engine.prepare.remote())
    for engines in auxiliary_engines:
        for engine in engines:
            prepare_model_refs.append(engine.prepare.remote())
    await asyncio.gather(*prepare_model_refs)


def assert_experience_tokens_match_text(test_case, tokenizer, exp, prompt_contents, response_text):
    full_text = tokenizer.decode(exp.tokens.tolist(), skip_special_tokens=False)
    prompt_text = tokenizer.decode(
        exp.tokens[: exp.prompt_length].tolist(), skip_special_tokens=False
    )
    decoded_response_text = tokenizer.decode(
        exp.tokens[exp.prompt_length :].tolist(), skip_special_tokens=False
    )

    for prompt_content in prompt_contents:
        test_case.assertIn(prompt_content, full_text)
        test_case.assertIn(prompt_content, prompt_text)
    test_case.assertIn(response_text, full_text)
    test_case.assertIn(response_text, decoded_response_text)


def _get_text_config(model_path: str):
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return getattr(hf_config, "text_config", hf_config)


def _assert_routed_experts_shape(test_case, exp, expected_layers: int, expected_topk: int):
    test_case.assertIsNotNone(exp.routed_experts)
    routed_experts = exp.routed_experts
    test_case.assertEqual(routed_experts.dtype, torch.uint8)
    test_case.assertEqual(routed_experts.ndim, 3)
    test_case.assertEqual(
        tuple(routed_experts.shape),
        (len(exp.tokens) - 1, expected_layers, expected_topk),
    )


@parameterized_class(
    (
        "tensor_parallel_size",
        "data_parallel_size",
        "pipeline_parallel_size",
        "engine_num",
        "nnodes",
        "enable_history",
        "enable_return_routed_experts",
    ),
    [
        (2, 2, 1, 1, 2, True, True),
        (1, 4, 1, 1, 2, True, True),
        (2, 1, 2, 1, 2, True, False),
        (1, 1, 1, 2, 1, False, False),
    ],
)
class TestSGLangOpenAIAPI(RayUnittestBaseAsync):
    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = (
            get_moe_model_path() if self.enable_return_routed_experts else get_model_path()
        )
        if self.enable_return_routed_experts:
            self.text_config = _get_text_config(self.config.model.model_path)
            self.expected_routed_experts_layers = int(self.text_config.num_hidden_layers)
            self.expected_routed_experts_topk = int(self.text_config.num_experts_per_tok)
        else:
            self.expected_routed_experts_layers = 0
            self.expected_routed_experts_topk = 0
        self.config.explorer.rollout_model.engine_type = "sglang"
        self.config.explorer.rollout_model.engine_num = self.engine_num
        self.config.explorer.rollout_model.tensor_parallel_size = self.tensor_parallel_size
        self.config.explorer.rollout_model.data_parallel_size = self.data_parallel_size
        self.config.explorer.rollout_model.pipeline_parallel_size = self.pipeline_parallel_size
        self.config.explorer.rollout_model.enable_expert_parallel = (
            self.enable_return_routed_experts
        )
        self.config.explorer.rollout_model.nnodes = self.nnodes
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_history = self.enable_history
        self.config.explorer.rollout_model.base_port = 13000
        self.config.algorithm.enable_router_replay = self.enable_return_routed_experts
        self.config.check_and_update()
        allocator = Allocator(self.config.explorer)
        rollout_models, _ = await allocator.create_all_models()
        self.model_wrapper = rollout_models[0]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_path,
            trust_remote_code=self.config.explorer.rollout_model.trust_remote_code,
        )

    def _assert_experience_matches_text(self, exp, prompt_contents, response_text):
        self.assertGreater(exp.prompt_length, 0)
        self.assertGreater(len(exp.tokens), exp.prompt_length)
        assert_experience_tokens_match_text(
            self, self.tokenizer, exp, prompt_contents, response_text
        )

    def _assert_history_matches_responses(self, expected_count, prompt_contents, response_texts):
        if not self.enable_history:
            self.assertEqual(len(self.model_wrapper.history), 0)
            return []

        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), expected_count)
        for exp, response_text in zip(exps, response_texts):
            self.assertEqual(exp.response_text, response_text)
            self._assert_experience_matches_text(exp, prompt_contents, response_text)
            if self.enable_return_routed_experts:
                _assert_routed_experts_shape(
                    self,
                    exp,
                    self.expected_routed_experts_layers,
                    self.expected_routed_experts_topk,
                )
        return exps

    def _assert_openai_response_routed_experts(self, response):
        if not self.enable_return_routed_experts:
            return
        self.assertTrue(hasattr(response, "sglext"))
        self.assertIsNotNone(response.sglext)
        self.assertTrue("routed_experts" in response.sglext)
        self.assertIsInstance(response.sglext["routed_experts"], str)
        self.assertGreater(len(response.sglext["routed_experts"]), 0)

    def _get_tool_call_case(self):
        tool_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Use the weather tool result to answer what the weather is in Boston.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_weather_boston",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location":"Boston","unit":"fahrenheit"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_boston",
                "content": "The weather in Boston is 72 F.",
            },
        ]
        tool_prompt_contents = [
            tool_messages[0]["content"],
            tool_messages[1]["content"],
            tool_messages[3]["content"],
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        return tool_messages, tool_prompt_contents, tools

    async def _collect_response_texts(self, response):
        response_texts = []
        for choice in response.choices:
            self.assertIsNotNone(choice.message.content)
            self.assertGreater(len(choice.message.content), 0)
            response_texts.append(choice.message.content)
        return response_texts

    async def _collect_stream_contents(self, stream_response, n):
        contents = ["" for _ in range(n)]
        async for chunk in stream_response:
            for choice in chunk.choices:
                if choice.delta.content is not None:
                    contents[choice.index] += choice.delta.content
        return contents

    async def test_chat_completions(self):
        self.assertEqual(self.model_wrapper.model_path, self.config.model.model_path)
        self.assertIsNotNone(self.model_wrapper.api_address)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write one short sentence about Boston."},
        ]
        prompt_contents = [message["content"] for message in messages]

        openai_client = self.model_wrapper.get_openai_async_client()
        response = await openai_client.chat.completions.create(
            model=openai_client.model_path,
            messages=messages,
            n=1,
            temperature=0.7,
            max_tokens=32,
        )

        self.assertEqual(len(response.choices), 1)
        self._assert_openai_response_routed_experts(response)
        response_texts = await self._collect_response_texts(response)
        self._assert_history_matches_responses(1, prompt_contents, response_texts)

        tool_messages, tool_prompt_contents, tools = self._get_tool_call_case()
        tool_response = await openai_client.chat.completions.create(
            model=openai_client.model_path,
            messages=tool_messages,
            tools=tools,
            tool_choice="none",
            temperature=0.7,
            max_tokens=32,
        )

        self.assertEqual(len(tool_response.choices), 1)
        self._assert_openai_response_routed_experts(tool_response)
        tool_response_texts = await self._collect_response_texts(tool_response)
        self._assert_history_matches_responses(1, tool_prompt_contents, tool_response_texts)

        if not self.enable_history:
            stream_response = await openai_client.chat.completions.create(
                model=openai_client.model_path,
                messages=messages,
                n=2,
                stream=True,
                temperature=0.7,
                max_tokens=32,
            )
            stream_contents = await self._collect_stream_contents(stream_response, 2)

            self.assertEqual(len(stream_contents), 2)
            for content in stream_contents:
                self.assertGreater(len(content), 0)
            self._assert_history_matches_responses(2, prompt_contents, stream_contents)

            stream_tool_response = await openai_client.chat.completions.create(
                model=openai_client.model_path,
                messages=tool_messages,
                tools=tools,
                tool_choice="none",
                n=1,
                stream=True,
                temperature=0.7,
                max_tokens=32,
            )
            stream_tool_contents = await self._collect_stream_contents(stream_tool_response, 1)

            self.assertEqual(len(stream_tool_contents), 1)
            self.assertGreater(len(stream_tool_contents[0]), 0)
            self._assert_history_matches_responses(1, tool_prompt_contents, stream_tool_contents)

        chat_exps = await self.model_wrapper.chat_async(
            messages,
            n=2,
            temperature=0.7,
            max_tokens=32,
        )

        self.assertEqual(len(chat_exps), 2)
        for exp in chat_exps:
            self.assertGreater(len(exp.response_text), 0)
            self.assertGreater(exp.prompt_length, 0)
            self.assertGreater(len(exp.tokens), exp.prompt_length)
            if self.enable_return_routed_experts:
                _assert_routed_experts_shape(
                    self,
                    exp,
                    self.expected_routed_experts_layers,
                    self.expected_routed_experts_topk,
                )

        if self.enable_history:
            chat_history = self.model_wrapper.extract_experience_from_history()
            self.assertEqual(len(chat_history), 2)
            for exp, recorded_exp in zip(chat_exps, chat_history):
                self.assertEqual(recorded_exp.response_text, exp.response_text)
                self.assertEqual(recorded_exp.prompt_length, exp.prompt_length)
                self._assert_experience_matches_text(
                    recorded_exp, prompt_contents, exp.response_text
                )
                if self.enable_return_routed_experts:
                    _assert_routed_experts_shape(
                        self,
                        recorded_exp,
                        self.expected_routed_experts_layers,
                        self.expected_routed_experts_topk,
                    )
        else:
            self.assertEqual(len(self.model_wrapper.history), 0)

        generate_prompt = "Write one short sentence about Boston."
        generate_exps = await self.model_wrapper.generate_async(
            [generate_prompt],
            n=2,
            temperature=0.7,
            max_tokens=32,
        )

        self.assertEqual(len(generate_exps), 2)
        for exp in generate_exps:
            self.assertEqual(exp.prompt_text, generate_prompt)
            self.assertGreater(len(exp.response_text), 0)
            self.assertGreater(exp.prompt_length, 0)
            self.assertGreater(len(exp.tokens), exp.prompt_length)
            if self.enable_return_routed_experts:
                _assert_routed_experts_shape(
                    self,
                    exp,
                    self.expected_routed_experts_layers,
                    self.expected_routed_experts_topk,
                )

        if self.enable_history:
            generate_history = self.model_wrapper.extract_experience_from_history()
            self.assertEqual(len(generate_history), 2)
            for exp, recorded_exp in zip(generate_exps, generate_history):
                self.assertEqual(recorded_exp.response_text, exp.response_text)
                self.assertEqual(recorded_exp.prompt_text, exp.prompt_text)
                self._assert_experience_matches_text(
                    recorded_exp, [generate_prompt], exp.response_text
                )
                if self.enable_return_routed_experts:
                    _assert_routed_experts_shape(
                        self,
                        recorded_exp,
                        self.expected_routed_experts_layers,
                        self.expected_routed_experts_topk,
                    )
        else:
            self.assertEqual(len(self.model_wrapper.history), 0)
