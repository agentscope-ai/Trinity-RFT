import asyncio
import os
import unittest

import ray
import torch
from openai import BadRequestError
from parameterized import parameterized_class
from transformers import AutoTokenizer

from tests.tools import (
    RayUnittestBaseAsync,
    get_api_model_path,
    get_model_path,
    get_template_config,
)
from trinity.common.config import Config
from trinity.common.models import create_explorer_models
from trinity.common.models.model import ModelWrapper
from trinity.common.models.utils import (
    tokenize_and_mask_messages_default,
    tokenize_and_mask_messages_hf,
)
from trinity.manager.synchronizer import Synchronizer

DEBUG = False


def print_debug(*args):
    if DEBUG:
        print(*args)


async def prepare_engines(engines, auxiliary_engines):
    prepare_model_refs = []
    for engine in engines:
        prepare_model_refs.append(engine.prepare.remote())
    for engines in auxiliary_engines:
        for engine in engines:
            prepare_model_refs.append(engine.prepare.remote())
    await asyncio.gather(*prepare_model_refs)


# Qwen2.5 chat template with {% generation %} mark
CHAT_TEMPLATE = r"""
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n'}}{% generation %}{{- message.content + '<|im_end|>' + '\n' }}{% endgeneration %}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n'}}{% generation %}
        {%- if message.content %}
            {{- message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>\n' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}{% endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


@parameterized_class(
    (
        "tensor_parallel_size",
        "engine_num",
        "repeat_times",
        "enable_history",
        "use_async",
    ),
    [
        (2, 2, 2, True, False),
        (1, 2, 1, False, True),
        (2, 1, 3, True, True),
    ],
)
class ModelWrapperTest(RayUnittestBaseAsync):
    def setUp(self):
        # configure the model
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.model.custom_chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.engine_num = self.engine_num
        self.config.explorer.rollout_model.tensor_parallel_size = self.tensor_parallel_size
        self.config.algorithm.repeat_times = self.repeat_times
        self.config.explorer.rollout_model.enable_history = self.enable_history
        self.config.check_and_update()

        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(
            self.engines[0], engine_type="vllm", enable_history=self.enable_history
        )

    async def test_generate(self):
        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.model_wrapper.prepare()
        self.assertEqual(self.model_wrapper.model_path, self.config.model.model_path)
        prompts = ["Hello, world!", "Hello, my name is"]
        n = self.config.algorithm.repeat_times
        if self.use_async:
            generate_results = await self.model_wrapper.generate_async(
                prompts, n=n, temperature=1.0
            )
        else:
            generate_results = self.model_wrapper.generate(prompts, n=n, temperature=1.0)
        self.assertEqual(len(generate_results), len(prompts) * n)
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history(
                clear_history=False
            )
            self.assertEqual(len(history_experiences), len(generate_results))
            for exp, history_exp in zip(generate_results, history_experiences):
                self.assertEqual(exp.response_text, history_exp.response_text)
                self.assertEqual(exp.tokens.tolist(), history_exp.tokens.tolist())
                self.assertEqual(exp.prompt_length, history_exp.prompt_length)
                self.assertEqual(exp.logprobs.tolist(), history_exp.logprobs.tolist())
        else:
            with self.assertRaises(ValueError):
                self.model_wrapper.extract_experience_from_history(clear_history=False)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"},
            {
                "role": "assistant",
                "content": "I'm sorry, but as an AI language model, I don't have access to real-time weather information. To get accurate weather information for your location, you can check a weather website or app, or look outside if possible.",
            },
            {"role": "user", "content": "OK, thanks!"},
        ]
        if self.use_async:
            results = await self.model_wrapper.chat_async(messages, n=n, temperature=1.0)
        else:
            results = self.model_wrapper.chat(messages, n=n, temperature=1.0)
        self.assertEqual(len(results), n)
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertEqual(len(history_experiences) - len(generate_results), len(results))
            for exp, history_exp in zip(results, history_experiences[len(generate_results) :]):
                self.assertEqual(exp.response_text, history_exp.response_text)
                self.assertEqual(exp.tokens.tolist(), history_exp.tokens.tolist())
                self.assertEqual(exp.prompt_length, history_exp.prompt_length)
                self.assertEqual(exp.logprobs.tolist(), history_exp.logprobs.tolist())
        for result in results:
            self.assertTrue(torch.any(result.logprobs != 0))
        if self.use_async:
            logprobs = await self.model_wrapper.logprobs_async(results[0].tokens.tolist())
        else:
            logprobs = self.model_wrapper.logprobs(results[0].tokens.tolist())
        self.assertEqual(logprobs.shape[0], results[0].tokens.shape[0] - 1)
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertTrue(len(history_experiences) == 0)
        messages.append(
            {
                "role": "assistant",
                "content": results[0].response_text,
            }
        )
        exp = self.model_wrapper.convert_messages_to_experience(messages)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
        result_dict = tokenizer.apply_chat_template(
            messages,
            chat_template=CHAT_TEMPLATE,
            add_generation_prompt=False,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        prompt_length = torch.argmax(result_dict["assistant_masks"][0]).item()
        self.assertTrue(
            torch.equal(result_dict["assistant_masks"][0][prompt_length:], exp.action_mask)
        )
        self.assertTrue(exp.logprobs.shape[0] == exp.tokens.shape[0] - prompt_length)
        self.assertTrue(torch.equal(result_dict["input_ids"][0], exp.tokens))
        self.assertRaises(ValueError, self.model_wrapper.get_openai_client)
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertTrue(len(history_experiences) == 0)


@parameterized_class(
    (
        "max_model_len",
        "max_prompt_tokens",
        "max_response_tokens",
    ),
    [
        (20, 19, None),
        (20, None, 1),
        (20, 5, 15),
    ],
)
class TestModelLen(RayUnittestBaseAsync):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.model.max_model_len = self.max_model_len
        self.config.model.max_prompt_tokens = self.max_prompt_tokens
        self.config.model.max_response_tokens = self.max_response_tokens
        self.config.model.enable_prompt_truncation = True
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.check_and_update()

        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)

    async def test_model_len(self):
        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.model_wrapper.prepare()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"},
        ]

        def _check_experience(exp):
            # check prompt content and length
            encoded_prompt = self.tokenizer.encode(exp.prompt_text, add_special_tokens=False)
            self.assertEqual(len(encoded_prompt), exp.prompt_length)
            self.assertLessEqual(exp.prompt_length, self.config.model.max_prompt_tokens)
            # check response content and length
            encoded_response = self.tokenizer.encode(exp.response_text, add_special_tokens=False)
            self.assertEqual(len(encoded_response), len(exp.tokens) - exp.prompt_length)
            self.assertLessEqual(
                len(exp.tokens) - exp.prompt_length, self.config.model.max_response_tokens
            )
            # check full sequence
            self.assertLessEqual(len(exp.tokens), self.config.model.max_model_len)

        # For vllm engine, max_prompt_tokens and max_response_tokens work
        response = self.model_wrapper.chat(messages)
        self.assertEqual(len(response), 1)
        if self.max_prompt_tokens == 5:
            self.assertEqual(response[0].truncate_status, "prompt_truncated")
        _check_experience(response[0])

        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        _check_experience(exps[0])

        # For openai api, max_prompt_tokens and max_response_tokens do not work
        openai_client = self.model_wrapper.get_openai_client()
        model_id = openai_client.models.list().data[0].id
        with self.assertRaises(BadRequestError):
            # the prompt is longer than max_model_len
            openai_client.chat.completions.create(model=model_id, messages=messages, n=1)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 0)

        response = openai_client.chat.completions.create(model=model_id, messages=messages[1:], n=1)
        self.assertEqual(len(response.choices), 1)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        # only generate max_response_tokens tokens
        self.assertLessEqual(
            len(exps[0].tokens) - response.usage.prompt_tokens,
            self.config.model.max_response_tokens,
        )


class TestModelLenWithoutPromptTruncation(RayUnittestBaseAsync):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.model.max_model_len = 20
        self.config.model.max_prompt_tokens = 1
        self.config.model.max_response_tokens = None
        self.config.model.enable_prompt_truncation = False
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.check_and_update()

        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)

    async def test_model_len(self):
        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.model_wrapper.prepare()
        messages = [
            {"role": "user", "content": "How are you?"},
        ]

        # For vllm engine, max_prompt_tokens and max_response_tokens work
        response = self.model_wrapper.chat(messages)
        self.assertEqual(len(response), 1)
        self.assertLessEqual(
            len(response[0].tokens) - response[0].prompt_length,
            self.config.model.max_response_tokens,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertLessEqual(
            len(exps[0].tokens) - exps[0].prompt_length,
            self.config.model.max_response_tokens,
        )

        # For openai api
        openai_client = self.model_wrapper.get_openai_client()
        model_id = openai_client.models.list().data[0].id
        response = openai_client.chat.completions.create(model=model_id, messages=messages, n=1)
        self.assertEqual(len(response.choices), 1)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertLessEqual(
            len(exps[0].tokens) - response.usage.prompt_tokens,
            self.config.model.max_response_tokens,
        )


class TestMessageProcess(RayUnittestBaseAsync):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.model.max_model_len = 100
        self.config.model.max_prompt_tokens = 50
        self.config.model.max_response_tokens = 50
        self.config.model.enable_prompt_truncation = True
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)

    async def test_truncation_status(self):
        """Test truncation status for multi-turn conversations."""
        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.model_wrapper.prepare()

        # Case: "prompt_truncated"
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "A very long prompt." * 20},
            {"role": "assistant", "content": "OK"},
        ]
        converted_experience = self.model_wrapper.convert_messages_to_experience(
            messages,
        )
        self._check_experience(converted_experience, "prompt_truncated")

        # Case: No truncation
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me about weather."},
            {"role": "assistant", "content": "OK"},
        ]
        converted_experience = self.model_wrapper.convert_messages_to_experience(
            messages,
        )
        self._check_experience(converted_experience, None)

    async def test_no_prompt_truncation(self):
        """Test truncation status for multi-turn conversations in workflow."""
        self.config.model.enable_prompt_truncation = False
        self.config.check_and_update()
        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.model_wrapper.prepare()

        # Case: No truncation
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me about weather."},
        ]
        converted_experience = self.model_wrapper.convert_messages_to_experience(messages)
        self._check_experience(converted_experience, None)

        # Case: "response_truncated"
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me about weather."},
            {"role": "assistant", "content": "A very long response" * 20},
        ]
        converted_experience = self.model_wrapper.convert_messages_to_experience(messages)
        self._check_experience(converted_experience, "response_truncated")

    def _check_experience(self, exp, target_truncate_status):
        self.assertIsNotNone(exp)
        model_len = len(exp.tokens)
        prompt_length = exp.prompt_length
        self.assertEqual(exp.truncate_status, target_truncate_status)
        self.assertLessEqual(prompt_length, self.config.model.max_prompt_tokens)
        self.assertLessEqual(model_len, self.config.model.max_model_len)


class TestAPIServer(RayUnittestBaseAsync):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)
        self.model_wrapper_no_history = ModelWrapper(
            self.engines[0], engine_type="vllm", enable_history=False
        )

    async def test_api(self):
        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.model_wrapper.prepare()
        await self.model_wrapper_no_history.prepare()
        openai_client = self.model_wrapper.get_openai_client()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ]
        model_id = openai_client.models.list().data[0].id
        response = openai_client.chat.completions.create(model=model_id, messages=messages, n=1)
        self.assertEqual(1, len(response.choices))
        self.assertTrue(len(response.choices[0].message.content) > 0)
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=2,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
        )
        self.assertEqual(2, len(response.choices))
        self.assertTrue(response.choices[0].logprobs is not None)
        self.assertEqual(0, len(response.choices[0].logprobs.content[2].top_logprobs))
        # here we check the 3rd token logprob, because the first two tokens (`<think>`,`\n` usually have zero logprob)
        self.assertTrue(response.choices[0].logprobs.content[2].logprob < 0)
        self.assertTrue(hasattr(response, "prompt_token_ids"))
        self.assertTrue(len(response.prompt_token_ids) > 0)
        self.assertTrue(hasattr(response.choices[0], "token_ids"))
        self.assertTrue(len(response.choices[0].token_ids) > 0)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 3)
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=4,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 4)
        for exp in exps:
            self.assertTrue(len(exp.tokens) > 0)
            self.assertTrue(len(exp.logprobs) > 0)
            self.assertTrue(exp.prompt_length + len(exp.logprobs) == len(exp.tokens))
        self.assertEqual(len(self.model_wrapper.extract_experience_from_history()), 0)
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertTrue(len(exps[0].tokens) > 0)
        self.assertTrue(len(exps[0].logprobs) > 0)
        self.assertTrue(exps[0].prompt_length + len(exps[0].logprobs) == len(exps[0].tokens))
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            logprobs=False,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertTrue(len(exps[0].logprobs) == 0)
        response = self.model_wrapper_no_history.get_openai_client().chat.completions.create(
            model=model_id, messages=messages, n=2
        )
        self.assertEqual(2, len(response.choices))
        self.assertTrue(hasattr(response.choices[0], "token_ids"))
        self.assertTrue(response.choices[0].token_ids is None)
        with self.assertRaises(ValueError):
            self.model_wrapper_no_history.extract_experience_from_history()
        self.assertEqual(len(self.model_wrapper_no_history.history), 0)


SYSTEM_PROMPT = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each iteration in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

USER_PROMPT = """Current Observation (0):
 _ 	 G 	 _
 _ 	 _ 	 _
 P 	 O 	 O
You have not achieved the goal, P has not reached G yet. Please give the next action.
The maximum number of steps remaining is 10.
"""


class TestLogprobs(RayUnittestBaseAsync):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_log_requests = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)

    async def test_logprobs_api(self):
        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.model_wrapper.prepare()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        # Test openai api logprobs with different temperature

        self.model_client = self.model_wrapper.get_openai_async_client()
        _ = await self.model_client.chat.completions.create(
            model=self.model_client.model_path,
            messages=messages,
            n=1,
            temperature=1.0,
            logprobs=True,
            max_tokens=15,
        )
        response_1 = self.model_wrapper.extract_experience_from_history()[0]
        _ = await self.model_client.chat.completions.create(
            model=self.model_client.model_path,
            messages=messages,
            n=1,
            temperature=0.8,
            logprobs=True,
            max_tokens=15,
        )
        response_2 = self.model_wrapper.extract_experience_from_history()[0]
        self.assertTrue(response_1.logprobs is not None)
        self.assertTrue(len(response_1.logprobs) > 0)
        self.assertTrue(response_2.logprobs is not None)
        self.assertTrue(len(response_2.logprobs) > 0)
        logprobs_1 = self.model_wrapper.logprobs(response_1.tokens.tolist(), temperature=1.0)
        logprobs_2 = self.model_wrapper.logprobs(response_1.tokens.tolist(), temperature=0.8)
        logprobs_3 = self.model_wrapper.logprobs(response_2.tokens.tolist(), temperature=1.0)
        logprobs_4 = self.model_wrapper.logprobs(response_2.tokens.tolist(), temperature=0.8)
        self.assertEqual(logprobs_1.shape, logprobs_2.shape)
        self.assertEqual(logprobs_3.shape, logprobs_4.shape)
        self.assertFalse(torch.allclose(logprobs_1, logprobs_2, rtol=0.3, atol=1e-3))
        self.assertFalse(torch.allclose(logprobs_3, logprobs_4, rtol=0.3, atol=1e-3))
        logprobs_1_prompt = logprobs_1[: response_1.prompt_length - 1]
        logprobs_2_prompt = logprobs_2[: response_1.prompt_length - 1]
        logprobs_3_prompt = logprobs_3[: response_2.prompt_length - 1]
        logprobs_4_prompt = logprobs_4[: response_2.prompt_length - 1]
        self.assertEqual(logprobs_1_prompt.shape, logprobs_2_prompt.shape)
        self.assertFalse(torch.allclose(logprobs_1_prompt, logprobs_2_prompt, rtol=0.3, atol=1e-3))
        self.assertFalse(torch.allclose(logprobs_3_prompt, logprobs_4_prompt, rtol=0.3, atol=1e-3))
        self.assertTrue(torch.allclose(logprobs_1_prompt, logprobs_3_prompt, rtol=0.3, atol=1e-3))
        self.assertTrue(torch.allclose(logprobs_2_prompt, logprobs_4_prompt, rtol=0.3, atol=1e-3))
        logprobs_1_response = logprobs_1[response_1.prompt_length - 1 :]
        logprobs_2_response = logprobs_2[response_1.prompt_length - 1 :]
        logprobs_3_response = logprobs_3[response_2.prompt_length - 1 :]
        logprobs_4_response = logprobs_4[response_2.prompt_length - 1 :]
        self.assertEqual(logprobs_1_response.shape, logprobs_2_response.shape)
        self.assertEqual(logprobs_3_response.shape, logprobs_4_response.shape)
        self.assertEqual(logprobs_1_response.shape, logprobs_2_response.shape)
        self.assertEqual(response_1.logprobs.shape, logprobs_1_response.shape)
        self.assertTrue(
            torch.allclose(response_1.logprobs, logprobs_1_response, rtol=0.3, atol=1e-3)
        )
        self.assertFalse(
            torch.allclose(response_1.logprobs, logprobs_2_response, rtol=0.3, atol=1e-3)
        )
        self.assertTrue(
            torch.allclose(response_2.logprobs, logprobs_4_response, rtol=0.5, atol=1e-2)
        )
        self.assertFalse(
            torch.allclose(response_2.logprobs, logprobs_3_response, rtol=0.3, atol=1e-3)
        )

        # test vllm engine logprobs with different temperature
        response_1 = self.model_wrapper.chat(
            messages, n=1, temperature=1.0, logprobs=True, max_tokens=15
        )[0]
        response_2 = self.model_wrapper.chat(
            messages, n=1, temperature=0.8, logprobs=True, max_tokens=15
        )[0]
        self.assertTrue(response_1.logprobs is not None)
        self.assertTrue(len(response_1.logprobs) > 0)
        self.assertTrue(response_2.logprobs is not None)
        self.assertTrue(len(response_2.logprobs) > 0)
        logprobs_1 = self.model_wrapper.logprobs(response_1.tokens.tolist(), temperature=1.0)
        logprobs_2 = self.model_wrapper.logprobs(response_1.tokens.tolist(), temperature=0.8)
        logprobs_3 = self.model_wrapper.logprobs(response_2.tokens.tolist(), temperature=1.0)
        logprobs_4 = self.model_wrapper.logprobs(response_2.tokens.tolist(), temperature=0.8)
        self.assertEqual(logprobs_1.shape, logprobs_2.shape)
        self.assertEqual(logprobs_3.shape, logprobs_4.shape)
        self.assertFalse(torch.allclose(logprobs_1, logprobs_2, rtol=0.3, atol=1e-3))
        self.assertFalse(torch.allclose(logprobs_3, logprobs_4, rtol=0.3, atol=1e-3))
        logprobs_1_prompt = logprobs_1[: response_1.prompt_length - 1]
        logprobs_2_prompt = logprobs_2[: response_1.prompt_length - 1]
        logprobs_3_prompt = logprobs_3[: response_2.prompt_length - 1]
        logprobs_4_prompt = logprobs_4[: response_2.prompt_length - 1]
        self.assertEqual(logprobs_1_prompt.shape, logprobs_2_prompt.shape)
        self.assertFalse(torch.allclose(logprobs_1_prompt, logprobs_2_prompt, rtol=0.3, atol=1e-3))
        self.assertFalse(torch.allclose(logprobs_3_prompt, logprobs_4_prompt, rtol=0.3, atol=1e-3))
        self.assertTrue(torch.allclose(logprobs_1_prompt, logprobs_3_prompt, rtol=0.3, atol=1e-3))
        self.assertTrue(torch.allclose(logprobs_2_prompt, logprobs_4_prompt, rtol=0.3, atol=1e-3))
        logprobs_1_response = logprobs_1[response_1.prompt_length - 1 :]
        logprobs_2_response = logprobs_2[response_1.prompt_length - 1 :]
        logprobs_3_response = logprobs_3[response_2.prompt_length - 1 :]
        logprobs_4_response = logprobs_4[response_2.prompt_length - 1 :]
        self.assertEqual(logprobs_1_response.shape, logprobs_2_response.shape)
        self.assertEqual(logprobs_3_response.shape, logprobs_4_response.shape)
        self.assertEqual(logprobs_1_response.shape, logprobs_2_response.shape)
        self.assertEqual(response_1.logprobs.shape, logprobs_1_response.shape)
        self.assertTrue(
            torch.allclose(response_1.logprobs, logprobs_1_response, rtol=0.3, atol=1e-3)
        )
        self.assertFalse(
            torch.allclose(response_1.logprobs, logprobs_2_response, rtol=0.3, atol=1e-3)
        )
        self.assertTrue(
            torch.allclose(response_2.logprobs, logprobs_4_response, rtol=0.5, atol=1e-2)
        )
        self.assertFalse(
            torch.allclose(response_2.logprobs, logprobs_3_response, rtol=0.3, atol=1e-3)
        )

        # test openai api and vllm engine logprobs consistency
        await self.model_wrapper.clean_workflow_state()
        _ = await self.model_client.chat.completions.create(
            model=self.model_client.model_path,
            messages=messages,
            n=1,
            temperature=1.0,
            logprobs=0,
            max_tokens=1,
        )
        response_openai_1 = self.model_wrapper.extract_experience_from_history()[0]
        _ = await self.model_client.chat.completions.create(
            model=self.model_client.model_path,
            messages=messages,
            n=1,
            temperature=0.8,
            logprobs=0,
            max_tokens=1,
        )
        response_openai_2 = self.model_wrapper.extract_experience_from_history()[0]
        response_vllm_1 = self.model_wrapper.chat(
            messages,
            n=1,
            temperature=1.0,
            logprobs=0,
            max_tokens=1,
        )[0]
        response_vllm_2 = self.model_wrapper.chat(
            messages,
            n=1,
            temperature=0.8,
            logprobs=0,
            max_tokens=1,
        )[0]
        self.assertEqual(len(response_openai_1.tokens), len(response_vllm_1.tokens))
        self.assertTrue(
            torch.allclose(
                response_openai_1.logprobs,
                response_vllm_1.logprobs,
                rtol=0.1,
            )
        )
        self.assertTrue(
            torch.allclose(
                response_openai_2.logprobs,
                response_vllm_2.logprobs,
                rtol=0.1,
            )
        )


class TestAsyncAPIServer(RayUnittestBaseAsync):
    engine_type: str = "vllm"
    model_path: str = get_model_path()

    async def asyncSetUp(self):
        self.config = get_template_config()
        self._update_config()
        await self._setup_engines()

    def _update_config(self):
        self.config.mode = "explore"
        self.config.model.model_path = self.model_path
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True

        self.config.check_and_update()

    async def _setup_engines(self):
        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(
            self.engines[0], engine_type=self.engine_type, enable_history=True
        )
        self.model_wrapper_no_history = ModelWrapper(
            self.engines[0], engine_type=self.engine_type, enable_history=False
        )

    async def test_api_async(self):
        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.model_wrapper.prepare()
        await self.model_wrapper_no_history.prepare()
        openai_client = self.model_wrapper.get_openai_async_client()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ]
        model_id = openai_client.model_path
        response = await openai_client.chat.completions.create(
            model=model_id, messages=messages, n=1
        )
        self.assertEqual(1, len(response.choices))
        self.assertTrue(len(response.choices[0].message.content) > 0)
        response = await openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=2,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
        )
        self.assertEqual(2, len(response.choices))
        self.assertTrue(response.choices[0].logprobs is not None)
        self.assertEqual(0, len(response.choices[0].logprobs.content[2].top_logprobs))
        # here we check the 3rd token logprob, because the first two tokens (`<think>`,`\n` usually have zero logprob)
        if "Instruct" not in self.model_path:
            self.assertTrue(response.choices[0].logprobs.content[2].logprob < 0)
        self.assertTrue(hasattr(response, "prompt_token_ids"))
        self.assertTrue(len(response.prompt_token_ids) > 0)
        self.assertTrue(hasattr(response.choices[0], "token_ids"))
        self.assertTrue(len(response.choices[0].token_ids) > 0)
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 3)
        response = await openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=4,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 4)
        for exp in exps:
            self.assertTrue(len(exp.tokens) > 0)
            self.assertTrue(len(exp.logprobs) > 0)
            self.assertTrue(exp.prompt_length + len(exp.logprobs) == len(exp.tokens))
        self.assertEqual(len(self.model_wrapper.extract_experience_from_history()), 0)
        response = await openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertTrue(len(exps[0].tokens) > 0)
        self.assertTrue(len(exps[0].logprobs) > 0)
        self.assertTrue(exps[0].prompt_length + len(exps[0].logprobs) == len(exps[0].tokens))
        response = await openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            logprobs=False,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.assertTrue(len(exps[0].logprobs) == 0)
        response = (
            await self.model_wrapper_no_history.get_openai_async_client().chat.completions.create(
                model=model_id, messages=messages, n=2
            )
        )
        self.assertEqual(2, len(response.choices))
        self.assertTrue(hasattr(response.choices[0], "token_ids"))
        self.assertTrue(response.choices[0].token_ids is None)
        with self.assertRaises(ValueError):
            self.model_wrapper_no_history.extract_experience_from_history()
        self.assertEqual(len(self.model_wrapper_no_history.history), 0)


@unittest.skipIf("TINKER_API_KEY" not in os.environ, "TINKER_API_KEY is not set")
class TestTinkerAsyncAPIServer(TestAsyncAPIServer):
    engine_type: str = "tinker"
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    # llama model in Tinker does not support chat template

    def _update_config(self):
        self.config.model.tinker.enable = True
        self.config.algorithm.algorithm_type = "grpo"
        super()._update_config()

    async def _setup_engines(self):
        @ray.remote
        class FakeTrainer:
            def __init__(self, config: Config):
                self.config = config
                self.synchronizer = Synchronizer.get_actor(config)

            async def is_alive(self):
                return True

        fake_trainer = FakeTrainer.remote(self.config)
        await fake_trainer.__ray_ready__.remote()
        await super()._setup_engines()

    async def test_api_async(self):
        await super().test_api_async()


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
        tokenizer = AutoTokenizer.from_pretrained(get_model_path())
        token_ids, action_mask, prompt_length = tokenize_and_mask_messages_default(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=CHAT_TEMPLATE,
        )
        token_ids_hf, action_mask_hf, prompt_length_hf = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=CHAT_TEMPLATE,
        )
        self.assertEqual(token_ids.shape, token_ids_hf.shape)
        self.assertEqual(action_mask.shape, action_mask_hf.shape)
        self.assertTrue(torch.equal(token_ids, token_ids_hf))
        self.assertTrue(torch.equal(action_mask, action_mask_hf))
        self.assertEqual(prompt_length, prompt_length_hf)

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
        tokenizer = AutoTokenizer.from_pretrained(get_model_path())
        token_ids, action_mask, prompt_length = tokenize_and_mask_messages_default(
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
            chat_template=CHAT_TEMPLATE,
        )
        token_ids_hf, action_mask_hf, prompt_length_hf = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
            chat_template=CHAT_TEMPLATE,
        )
        self.assertEqual(token_ids.shape, token_ids_hf.shape)
        self.assertEqual(action_mask.shape, action_mask_hf.shape)
        self.assertTrue(torch.equal(token_ids, token_ids_hf))
        self.assertTrue(torch.equal(action_mask, action_mask_hf))
        self.assertEqual(prompt_length, prompt_length_hf)


@parameterized_class(
    ("enable_thinking", "reasoning_parser"),
    [
        (True, "deepseek_r1"),
        (False, None),
    ],
)
class TestAPIServerToolCall(RayUnittestBaseAsync):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_api_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True
        # added for toolcalls
        self.config.explorer.rollout_model.enable_auto_tool_choice = True
        self.config.explorer.rollout_model.tool_call_parser = "hermes"
        self.config.explorer.rollout_model.enable_thinking = self.enable_thinking
        self.config.explorer.rollout_model.reasoning_parser = self.reasoning_parser

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)
        self.model_wrapper_no_history = ModelWrapper(
            self.engines[0], engine_type="vllm", enable_history=False
        )

    async def test_api_tool_calls(self):
        """Tests the full conversation flow of a tool call via the OpenAI API.
        Note: This test require a model that supports tool calls and thinking mode, e.g. Qwen3-1.7B.
        """
        import json
        import time

        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.model_wrapper.prepare()
        await self.model_wrapper_no_history.prepare()
        tokenizer = AutoTokenizer.from_pretrained(get_api_model_path())
        print_debug("\n\n" + "=" * 30 + " Running test_api_tool_calls " + "=" * 30)
        start_time = time.time()

        # --- Step 0: Get OpenAI Client ---
        print_debug(f"[{time.time() - start_time:.2f}s] Getting OpenAI client...")
        openai_client = self.model_wrapper.get_openai_client()
        model_id = openai_client.models.list().data[0].id
        print_debug(
            f"[{time.time() - start_time:.2f}s] Successfully got client. Model ID: {model_id}"
        )

        # --- Step 1: Define Tools and Messages ---
        print_debug(f"[{time.time() - start_time:.2f}s] Defining tools and initial message...")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
        print_debug(
            f"[{time.time() - start_time:.2f}s] Initial user message: {messages[0]['content']}"
        )
        print_debug("-" * 80)

        # --- Step 2: First API Call (Expecting a tool call) ---
        print_debug(f"[{time.time() - start_time:.2f}s] Making first API call to the model...")
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking
                },  # default to True
            },
        )
        print_debug(f"[{time.time() - start_time:.2f}s] First API call completed.")

        # --- Step 3: Assert and Print the Tool Call Response ---
        print_debug(f"[{time.time() - start_time:.2f}s] Asserting response is a tool call...")
        self.assertEqual(len(response.choices), 1)
        choice = response.choices[0]
        print_debug(f"    > Finish Reason: {choice.finish_reason}")
        self.assertEqual(choice.finish_reason, "tool_calls")
        if self.enable_thinking:
            self.assertIsNotNone(choice.message.reasoning_content)
        self.assertIsNotNone(choice.message.tool_calls)
        self.assertEqual(len(choice.message.tool_calls), 1)

        tool_call = choice.message.tool_calls[0]
        print_debug(f"    > Tool Call ID: {tool_call.id}")
        print_debug(f"    > Function Name: {tool_call.function.name}")
        print_debug(f"    > Function Arguments: {tool_call.function.arguments}")
        self.assertEqual(tool_call.type, "function")
        self.assertEqual(tool_call.function.name, "get_current_weather")
        self.assertIn("Boston", tool_call.function.arguments)
        print_debug(f"[{time.time() - start_time:.2f}s] Assertions for tool call passed.")
        print_debug("-" * 80)

        # --- Step 4: Check Experience History ---
        print_debug(f"[{time.time() - start_time:.2f}s] Checking experience history...")
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        # The response text in the experience should contain the tool call info
        print_debug(f"    > Recorded experience response_text: {exps[0].response_text}")
        print_debug(f"    > Recorded experience: {exps[0]}")
        print_debug(f"    > message: {choice.message}")

        exp = exps[0]
        print_debug("\n" + "-" * 15 + " Decoding Experience Tokens " + "-" * 15)

        full_decoded_text = tokenizer.decode(exp.tokens, skip_special_tokens=False)
        print_debug(
            f"    > Full Decoded Text ({len(exp.tokens)} tokens):\n---\n{full_decoded_text}\n---"
        )

        prompt_length = exp.prompt_length
        prompt_tokens = exp.tokens[:prompt_length]
        response_tokens = exp.tokens[prompt_length:]

        prompt_decoded_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        response_decoded_text = tokenizer.decode(response_tokens, skip_special_tokens=False)

        print_debug(
            f"    > Decoded Prompt Part ({len(prompt_tokens)} tokens):\n---\n{prompt_decoded_text}\n---"
        )
        print_debug(
            f"    > Decoded Response Part ({len(response_tokens)} tokens):\n---\n{response_decoded_text}\n---"
        )

        action_mask = getattr(exp, "action_mask", None)
        if action_mask is not None:
            print_debug(f"\n    > Action Mask (Length: {len(action_mask)}):")
            masked_tokens_info = []
            for i, token_id in enumerate(response_tokens):
                token_text = tokenizer.decode([token_id])
                mask_value = action_mask[i] if i < len(action_mask) else "N/A"
                masked_tokens_info.append(f"({repr(token_text)}, Mask: {mask_value})")

            print_debug("      " + " ".join(masked_tokens_info))

            self.assertTrue(
                abs(len(action_mask) - len(response_tokens)) <= 1,
                f"Length of action_mask ({len(action_mask)}) does not match "
                f"length of response_tokens ({len(response_tokens)})",
            )
        else:
            print_debug("    > Action Mask: Not found in experience.")

        print_debug("-" * 52 + "\n")

        # pass this part
        # self.assertIn("get_current_weather", exps[0].response_text)

        self.assertEqual(
            len(self.model_wrapper.extract_experience_from_history()), 0
        )  # Verify cleared
        print_debug(f"[{time.time() - start_time:.2f}s] Experience history check passed.")
        print_debug("-" * 80)

        # --- Step 5: Second API Call (Providing tool result) ---
        print_debug(
            f"[{time.time() - start_time:.2f}s] Preparing for the second API call with tool result..."
        )
        messages.append(response.choices[0].message)  # Add assistant's tool call message

        # Mock the result of our tool
        tool_response_content = json.dumps(
            {"location": "Boston", "temperature": "72", "unit": "fahrenheit"}
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response_content,
            }
        )
        print_debug(f"[{time.time() - start_time:.2f}s] Full message list for second call:")
        for msg in messages:
            print_debug(f"    - {msg}")

        print_debug(f"[{time.time() - start_time:.2f}s] Making second API call...")
        second_response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools,
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking
                },  # default to True
            },
        )
        print_debug(f"[{time.time() - start_time:.2f}s] Second API call completed.")

        # --- Step 6: Assert and Print the Final Response ---
        print_debug(
            f"[{time.time() - start_time:.2f}s] Asserting final natural language response..."
        )
        self.assertEqual(len(second_response.choices), 1)
        final_choice = second_response.choices[0]
        print_debug(f"    > Final Finish Reason: {final_choice.finish_reason}")
        print_debug(f"    > Final Message Content: {final_choice.message.content}")
        print_debug(f"    > Final Message: {final_choice.message}")
        self.assertEqual(final_choice.finish_reason, "stop")
        # self.assertIsNone(final_choice.message.tool_calls)
        self.assertEqual(final_choice.message.tool_calls, [])
        self.assertIsNotNone(final_choice.message.content)
        # Check if the model used the information from the tool response
        self.assertIn("72", final_choice.message.content)
        self.assertIn("Boston", final_choice.message.content)
        print_debug(f"[{time.time() - start_time:.2f}s] Assertions for final response passed.")
        print_debug("-" * 80)

        # --- Step 7: Check Final Experience History ---
        print_debug(f"[{time.time() - start_time:.2f}s] Checking final experience history...")
        final_exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(final_exps), 1)
        print_debug(f"    > Final recorded experience response_text: {final_exps[0].response_text}")
        self.assertEqual(final_exps[0].response_text, final_choice.message.content)
        print_debug(f"[{time.time() - start_time:.2f}s] Final experience history check passed.")

        exp = final_exps[0]
        print_debug("\n" + "-" * 15 + " Decoding Experience Tokens " + "-" * 15)

        full_decoded_text = tokenizer.decode(exp.tokens, skip_special_tokens=False)
        print_debug(
            f"    > Full Decoded Text ({len(exp.tokens)} tokens):\n---\n{full_decoded_text}\n---"
        )

        prompt_length = exp.prompt_length
        prompt_tokens = exp.tokens[:prompt_length]
        response_tokens = exp.tokens[prompt_length:]

        prompt_decoded_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        response_decoded_text = tokenizer.decode(response_tokens, skip_special_tokens=False)

        print_debug(
            f"    > Decoded Prompt Part ({len(prompt_tokens)} tokens):\n---\n{prompt_decoded_text}\n---"
        )
        print_debug(
            f"    > Decoded Response Part ({len(response_tokens)} tokens):\n---\n{response_decoded_text}\n---"
        )

        action_mask = getattr(exp, "action_mask", None)
        if action_mask is not None:
            print_debug(f"\n    > Action Mask (Length: {len(action_mask)}):")
            masked_tokens_info = []
            for i, token_id in enumerate(response_tokens):
                token_text = tokenizer.decode([token_id])
                mask_value = action_mask[i] if i < len(action_mask) else "N/A"
                masked_tokens_info.append(f"({repr(token_text)}, Mask: {mask_value})")

            print_debug("      " + " ".join(masked_tokens_info))

            self.assertTrue(
                abs(len(action_mask) - len(response_tokens)) <= 1,
                f"Length of action_mask ({len(action_mask)}) does not match "
                f"length of response_tokens ({len(response_tokens)})",
            )
        else:
            print_debug("    > Action Mask: Not found in experience.")

        total_time = time.time() - start_time
        print_debug(
            "\n" + "=" * 28 + f" test_api_tool_calls PASSED in {total_time:.2f}s " + "=" * 28 + "\n"
        )


class TestSuperLongGeneration(RayUnittestBaseAsync):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.model.max_model_len = 81920
        self.config.model.max_prompt_tokens = 61440
        self.config.model.max_response_tokens = 20480
        self.config.model.rope_scaling = {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 40960,
        }
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)

    async def test_generate(self):
        base_dir = os.path.dirname(__file__)
        target_dir = os.path.join(base_dir, "..", "..", "trinity", "trainer", "verl")
        with open(os.path.join(target_dir, "fsdp_workers.py")) as f:
            fsdp_code = f.read()
        with open(os.path.join(target_dir, "megatron_workers.py")) as f:
            megatron_code = f.read()
        target_dir = os.path.join(base_dir, "..", "..", "trinity", "common")
        with open(os.path.join(target_dir, "config.py")) as f:
            config_code = f.read()
        target_dir = os.path.join(base_dir, "..", "..", "trinity", "manager")
        with open(os.path.join(target_dir, "config_manager.py")) as f:
            config_manager_code = f.read()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": """# Please add comments and documentation for these following code, """
                """make sure the code is well-structured and easy to read, """
                """and the complete code must be shown, do not omit any parts.\n"""
                f"""## fsdp_workers.py\n{fsdp_code}\n"""
                f"""## megatron_workers.py\n{megatron_code}\n"""
                f"""## config.py\n{config_code}\n"""
                f"""## config_manager.py\n{config_manager_code}\n""",
            },
        ]
        response = self.model_wrapper.chat(messages, n=1, temperature=0.7, logprobs=True)[0]
        self.assertGreater(
            response.prompt_length, 40960
        )  # If not long enough, please add more files to prompt
        self.assertGreater(response.logprobs.shape[0], 1000)


class TestTinkerAPI(RayUnittestBaseAsync):
    """Test the Tinker API integration with the vLLM engine."""

    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_lora = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_explorer_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)

    async def test_tinker_api(self):
        from tinker import types
        from transformers import AutoTokenizer

        engine = self.engines[0]
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ]
        result_dict = tokenizer.apply_chat_template(
            messages,
            chat_template=CHAT_TEMPLATE,
            add_generation_prompt=False,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        prompt = types.ModelInput.from_ints(
            result_dict["input_ids"][0].tolist(),
        )
        # sample api without prompt logprobs
        num_samples = 4
        response = await engine.sample.remote(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=types.SamplingParams(temperature=0.7),  # no limit on length
        )
        self.assertEqual(len(response.sequences), num_samples)
        for sequence in response.sequences:
            self.assertEqual(len(sequence.tokens), len(sequence.logprobs))
            self.assertEqual(sequence.stop_reason, "stop")
        self.assertIsNone(response.prompt_logprobs)
        self.assertIsNone(response.topk_prompt_logprobs)
        # sample api with prompt logprobs
        num_samples = 2
        topk_prompt_logprobs = 3
        response = await engine.sample.remote(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=types.SamplingParams(temperature=0.7, max_tokens=8),
            include_prompt_logprobs=True,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )
        self.assertEqual(len(response.sequences), num_samples)
        for sequence in response.sequences:
            self.assertEqual(len(sequence.tokens), len(sequence.logprobs))
            self.assertEqual(sequence.stop_reason, "length")
        self.assertEqual(len(response.prompt_logprobs), len(prompt.to_ints()))
        self.assertIsNone(response.prompt_logprobs[0])
        self.assertEqual(len(response.topk_prompt_logprobs), len(prompt.to_ints()))
        self.assertIsNone(response.topk_prompt_logprobs[0])
        for topk_logprobs in response.topk_prompt_logprobs[1:]:
            self.assertIsNotNone(topk_logprobs)
            self.assertEqual(len(topk_logprobs), topk_prompt_logprobs)
        # compute_logprob api
        response = await engine.sample.remote(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
        )
        self.assertEqual(len(response.sequences), 1)
        self.assertEqual(response.sequences[0].stop_reason, "length")
        self.assertEqual(len(prompt.to_ints()), len(response.prompt_logprobs))
        self.assertIsNone(response.topk_prompt_logprobs)

        # test add remove lora
        from vllm.lora.request import LoRARequest

        # create a dummy lora adapter with all zero weights
        lora_path_1 = os.path.join(self.config.checkpoint_job_dir, "adapter_1")
        lora_path_2 = os.path.join(self.config.checkpoint_job_dir, "adapter_2")
        _create_adapter(self.config.model.model_path, lora_path_1, "adapter_1")
        _create_adapter(self.config.model.model_path, lora_path_2, "adapter_2")
        lora_1 = LoRARequest(
            lora_name="test_adapter_1",
            lora_int_id=1,
            lora_path=os.path.join(lora_path_1, "adapter_1"),
        )
        lora_2 = LoRARequest(
            lora_name="test_adapter_2",
            lora_int_id=2,
            lora_path=os.path.join(lora_path_2, "adapter_2"),
        )
        response = await engine.sample.remote(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            lora_request=lora_1,
        )
        ids = await engine.list_lora_adapters.remote()
        self.assertEqual(ids, [1])
        self.assertEqual(len(response.sequences), 1)
        self.assertEqual(response.sequences[0].stop_reason, "length")
        self.assertEqual(len(prompt.to_ints()), len(response.prompt_logprobs))
        self.assertIsNone(response.topk_prompt_logprobs)
        response = await engine.sample.remote(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            lora_request=lora_2,
        )
        self.assertEqual(len(response.sequences), 1)
        self.assertEqual(response.sequences[0].stop_reason, "length")
        self.assertEqual(len(prompt.to_ints()), len(response.prompt_logprobs))
        self.assertIsNone(response.topk_prompt_logprobs)
        await engine.remove_lora_adapter.remote(lora_id=1)
        await engine.remove_lora_adapter.remote(lora_id=2)
        ids = await engine.list_lora_adapters.remote()
        self.assertEqual(ids, [])


def _create_adapter(model_path: str, lora_path: str, name: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
    )
    lora_model = get_peft_model(model, lora_config, adapter_name=name)
    lora_model.save_pretrained(lora_path)
