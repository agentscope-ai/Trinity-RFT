import asyncio
import json
import os
import shutil
import unittest
from copy import deepcopy
from typing import cast

import ray
import torch
from openai import BadRequestError
from parameterized import parameterized_class
from ray.util import remove_placement_group
from ray.util.placement_group import placement_group_table
from transformers import AutoConfig, AutoTokenizer

from tests.tools import (
    CHAT_TEMPLATE,
    RayUnittestBaseAsync,
    get_checkpoint_path,
    get_model_path,
    get_moe_model_path,
    get_template_config,
    get_vision_language_model_path,
)
from trinity.common.config import Config, GenerationConfig
from trinity.common.constants import ROLLOUT_WEIGHT_SYNC_GROUP_NAME, SyncMethod
from trinity.common.models.allocator import Allocator
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.thinking_budget_workflow import ThinkingBudgetWorkflow
from trinity.common.workflows.workflow import Task
from trinity.manager.synchronizer import Synchronizer

DEBUG = False


def print_debug(*args):
    if DEBUG:
        print(*args)


async def create_test_models(config: Config):
    allocator = Allocator(config.explorer)
    try:
        engines, auxiliary_engines = await allocator.create_all_models()
    except BaseException:
        if hasattr(allocator, "pg"):
            remove_placement_group(allocator.pg)
        raise

    for wrapper in engines:
        setattr(wrapper, "_test_placement_group", allocator.pg)
        setattr(
            wrapper,
            "_test_actor_names",
            tuple(allocator.bundle_result.actor_bundle_map),
        )
    for wrappers in auxiliary_engines:
        for wrapper in wrappers:
            setattr(wrapper, "_test_placement_group", allocator.pg)
            setattr(
                wrapper,
                "_test_actor_names",
                tuple(allocator.bundle_result.actor_bundle_map),
            )
    return engines, auxiliary_engines


def clone_wrapper(wrapper: ModelWrapper, enable_history: bool) -> ModelWrapper:
    config = deepcopy(wrapper.config)
    config.enable_history = enable_history
    return ModelWrapper(
        models=cast(list, wrapper.models),
        config=config,
        api_address=wrapper.api_address,
    )


def _get_text_config(model_path: str):
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return getattr(hf_config, "text_config", hf_config)


def _count_moe_layers(hf_config) -> int:
    layers_block_type = getattr(hf_config, "layers_block_type", None)
    if layers_block_type is not None:
        return layers_block_type.count("moe")
    num_layers = int(hf_config.num_hidden_layers)
    mlp_only_layers = getattr(hf_config, "mlp_only_layers", None) or []
    decoder_sparse_step = getattr(hf_config, "decoder_sparse_step", 1) or 1
    if decoder_sparse_step > 1:
        return sum(
            1
            for layer_id in range(num_layers)
            if (layer_id + 1) % decoder_sparse_step == 0 and layer_id not in mlp_only_layers
        )
    return num_layers - sum(1 for layer_id in mlp_only_layers if 0 <= layer_id < num_layers)


def _assert_routed_experts_shape(test_case, exp, expected_layers: int, expected_topk: int):
    test_case.assertIsNotNone(exp.routed_experts)
    routed_experts = exp.routed_experts
    test_case.assertEqual(routed_experts.dtype, torch.uint8)
    test_case.assertEqual(routed_experts.ndim, 3)
    test_case.assertEqual(
        tuple(routed_experts.shape),
        (len(exp.tokens) - 1, expected_layers, expected_topk),
    )


def _load_gsm8k_questions() -> list[str]:
    """Load the diverse math questions from the GSM8K training set."""
    path = os.path.join(os.path.dirname(__file__), "..", "template", "data", "gsm8k", "train.jsonl")
    questions: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line)["question"])
    return questions


class VLLMTestBase(RayUnittestBaseAsync):
    async def asyncTearDown(self):
        wrappers = []
        for attr in ("engines", "auxiliary_engines"):
            value = getattr(self, attr, None)
            if value is None:
                continue
            if attr == "engines":
                wrappers.extend(value)
            else:
                for model_list in value:
                    wrappers.extend(model_list)

        if not wrappers:
            return

        placement_groups = {
            getattr(wrapper, "_test_placement_group").id: getattr(wrapper, "_test_placement_group")
            for wrapper in wrappers
            if hasattr(wrapper, "_test_placement_group")
        }
        actors = [actor for wrapper in wrappers for actor in wrapper.models]
        actor_names = {
            name for wrapper in wrappers for name in getattr(wrapper, "_test_actor_names", ())
        }
        namespace = self.config.explorer.rollout_model.ray_namespace
        try:
            await asyncio.gather(*[wrapper.shutdown() for wrapper in wrappers])
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
            for pg in placement_groups.values():
                remove_placement_group(pg)
        for _ in range(100):
            tables = [placement_group_table(pg) for pg in placement_groups.values()]
            live_actor_names = set()
            for name in actor_names:
                try:
                    ray.get_actor(name, namespace=namespace)
                    live_actor_names.add(name)
                except ValueError:
                    pass
            if (
                all(not table or table.get("state") == "REMOVED" for table in tables)
                and not live_actor_names
            ):
                break
            await asyncio.sleep(0.1)
        else:
            self.fail("Timed out while removing vLLM test actors or placement groups.")


@parameterized_class(
    (
        "tensor_parallel_size",
        "data_parallel_size",
        "pipeline_parallel_size",
        "engine_num",
        "nnodes",
        "repeat_times",
        "enable_history",
        "use_async",
        "enable_return_routed_experts",
    ),
    [
        (2, 1, 1, 2, 1, 1, True, False, True),
        (1, 2, 1, 2, 1, 1, True, False, False),
        (1, 1, 2, 2, 1, 1, False, True, False),
        (4, 1, 1, 1, 2, 4, True, True, True),
    ],
)
class ModelWrapperTest(VLLMTestBase):
    async def asyncSetUp(self):
        # configure the model
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = (
            get_model_path() if not self.enable_return_routed_experts else get_moe_model_path()
        )
        if self.enable_return_routed_experts:
            self.text_config = _get_text_config(self.config.model.model_path)
            self.expected_routed_experts_layers = _count_moe_layers(self.text_config)
            self.expected_routed_experts_topk = int(self.text_config.num_experts_per_tok)
        else:
            self.expected_routed_experts_layers = 0
            self.expected_routed_experts_topk = 0
        self.config.model.custom_chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.engine_num = self.engine_num
        self.config.explorer.rollout_model.nnodes = self.nnodes
        self.config.explorer.rollout_model.tensor_parallel_size = self.tensor_parallel_size
        self.config.explorer.rollout_model.data_parallel_size = self.data_parallel_size
        self.config.explorer.rollout_model.pipeline_parallel_size = self.pipeline_parallel_size
        self.config.explorer.rollout_model.enable_expert_parallel = (
            self.enable_return_routed_experts
        )
        self.config.algorithm.repeat_times = self.repeat_times
        self.config.explorer.rollout_model.enable_history = self.enable_history
        self.config.explorer.rollout_model.enable_openai_api = self.enable_return_routed_experts
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.extra_engine_args = {"max_num_seqs": 24}
        if self.enable_return_routed_experts:
            self.config.explorer.rollout_model.extra_engine_args["moe_backend"] = "triton"
            self.config.explorer.rollout_model.extra_engine_args["gdn_prefill_backend"] = "triton"
        self.config.algorithm.enable_router_replay = self.enable_return_routed_experts
        self.config.check_and_update()

        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]

    def _assert_openai_response_routed_experts(self, response, expected_choices: int):
        self.assertEqual(len(response.choices), expected_choices)
        if not self.enable_return_routed_experts:
            return
        for choice in response.choices:
            self.assertTrue(hasattr(choice, "routed_experts"))
            self.assertIsInstance(choice.routed_experts, str)
            self.assertGreater(len(choice.routed_experts), 0)

    async def test_generate(self):  # noqa: C901
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
        if self.enable_return_routed_experts:
            for exp in generate_results:
                _assert_routed_experts_shape(
                    self,
                    exp,
                    self.expected_routed_experts_layers,
                    self.expected_routed_experts_topk,
                )
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
                if self.enable_return_routed_experts:
                    _assert_routed_experts_shape(
                        self,
                        history_exp,
                        self.expected_routed_experts_layers,
                        self.expected_routed_experts_topk,
                    )
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
        if self.enable_return_routed_experts:
            for exp in results:
                _assert_routed_experts_shape(
                    self,
                    exp,
                    self.expected_routed_experts_layers,
                    self.expected_routed_experts_topk,
                )
        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertEqual(len(history_experiences) - len(generate_results), len(results))
            for exp, history_exp in zip(results, history_experiences[len(generate_results) :]):
                self.assertEqual(exp.response_text, history_exp.response_text)
                self.assertEqual(exp.tokens.tolist(), history_exp.tokens.tolist())
                self.assertEqual(exp.prompt_length, history_exp.prompt_length)
                self.assertEqual(exp.logprobs.tolist(), history_exp.logprobs.tolist())
                if self.enable_return_routed_experts:
                    _assert_routed_experts_shape(
                        self,
                        history_exp,
                        self.expected_routed_experts_layers,
                        self.expected_routed_experts_topk,
                    )
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
        if self.enable_return_routed_experts:
            self.assertIsNotNone(self.model_wrapper.get_openai_client())
        else:
            self.assertRaises(ValueError, self.model_wrapper.get_openai_client)

        if self.enable_return_routed_experts:
            openai_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write one short sentence about Boston."},
            ]
            prompt_contents = [message["content"] for message in openai_messages]
            if self.use_async:
                openai_client = self.model_wrapper.get_openai_async_client()
                openai_response = await openai_client.chat.completions.create(
                    model=self.model_wrapper.model_path,
                    messages=openai_messages,
                    n=n,
                    temperature=0.7,
                    max_tokens=32,
                )
            else:
                openai_client = self.model_wrapper.get_openai_client()
                openai_response = openai_client.chat.completions.create(
                    model=self.model_wrapper.model_path,
                    messages=openai_messages,
                    n=n,
                    temperature=0.7,
                    max_tokens=32,
                )

            self._assert_openai_response_routed_experts(openai_response, n)

            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertEqual(len(history_experiences), n)
            for choice, history_exp in zip(openai_response.choices, history_experiences):
                self.assertIsNotNone(choice.message.content)
                self.assertEqual(history_exp.response_text, choice.message.content)
                self.assertGreater(history_exp.prompt_length, 0)
                self.assertGreater(len(history_exp.tokens), history_exp.prompt_length)
                prompt_text = tokenizer.decode(
                    history_exp.tokens[: history_exp.prompt_length].tolist(),
                    skip_special_tokens=False,
                )
                for prompt_content in prompt_contents:
                    self.assertIn(prompt_content, prompt_text)
                _assert_routed_experts_shape(
                    self,
                    history_exp,
                    self.expected_routed_experts_layers,
                    self.expected_routed_experts_topk,
                )

        if self.config.explorer.rollout_model.enable_history:
            history_experiences = self.model_wrapper.extract_experience_from_history()
            self.assertTrue(len(history_experiences) == 0)


class TestMultiModal(VLLMTestBase):
    async def asyncSetUp(self):
        # configure the model
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_vision_language_model_path()
        self.config.model.custom_chat_template = CHAT_TEMPLATE
        self.config.algorithm.repeat_times = 4
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.check_and_update()

        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]

    async def test_generate(self):  # noqa: C901
        n = self.config.algorithm.repeat_times
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"},
            {
                "role": "assistant",
                "content": "I'm sorry, but as an AI language model, I don't have access to real-time weather information. To get accurate weather information for your location, you can check a weather website or app, or look outside if possible.",
            },
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
        ]
        results = await self.model_wrapper.chat_async(messages, n=n, temperature=1.0)
        self.assertEqual(len(results), n)
        for exp in results:  # test multi-modal
            self.assertSetEqual(
                set(exp.multi_modal_inputs.keys()),
                {"mm_token_type_ids", "pixel_values", "image_grid_thw"},
            )
            self.assertEqual(len(exp.tokens), exp.multi_modal_inputs["mm_token_type_ids"].size(1))


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
class TestModelLen(VLLMTestBase):
    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.model.max_model_len = self.max_model_len
        self.config.model.max_prompt_tokens = self.max_prompt_tokens
        self.config.model.max_response_tokens = self.max_response_tokens
        self.config.model.enable_prompt_truncation = True
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_history = True
        self.config.check_and_update()

        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)

    async def test_model_len(self):
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
            if exp.truncate_status == "prompt_truncated":
                self.assertEqual(
                    exp.response_text, "[This experience is masked out due to overlong prompt]"
                )
                self.assertEqual(exp.prompt_text, self.tokenizer.decode(exp.tokens[:-1]))
                self.assertEqual(len(exp.tokens), self.config.model.max_prompt_tokens + 1)
                self.assertEqual(exp.prompt_length, self.config.model.max_prompt_tokens)
                self.assertTrue(torch.equal(exp.logprobs, torch.zeros(1, dtype=torch.float32)))
            else:
                encoded_response = self.tokenizer.encode(
                    exp.response_text, add_special_tokens=False
                )
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

        # test prompt truncation branch in generate
        if self.max_prompt_tokens == 5:
            prompt = "This is a deliberately long prompt for truncation coverage."
            prompt_token_ids = self.tokenizer(prompt, truncation=False, return_tensors="pt")[
                "input_ids"
            ][0].tolist()
            self.assertGreater(len(prompt_token_ids), self.config.model.max_prompt_tokens)

            responses = self.model_wrapper.generate([prompt], n=2)
            self.assertEqual(len(responses), 2)

            for response in responses:
                self.assertEqual(response.truncate_status, "prompt_truncated")
                _check_experience(response)

            exps = self.model_wrapper.extract_experience_from_history()
            self.assertEqual(len(exps), 2)
            for exp in exps:
                self.assertEqual(exp.truncate_status, "prompt_truncated")
                _check_experience(exp)


class TestModelLenWithoutPromptTruncation(VLLMTestBase):
    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.model.max_model_len = 20
        self.config.model.max_prompt_tokens = 1
        self.config.model.max_response_tokens = None
        self.config.model.enable_prompt_truncation = False
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_history = True
        self.config.check_and_update()

        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]

    async def test_model_len(self):
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


class TestMessageProcess(VLLMTestBase):
    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.model.max_model_len = 100
        self.config.model.max_prompt_tokens = 50
        self.config.model.max_response_tokens = 50
        self.config.model.enable_prompt_truncation = True
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.check_and_update()
        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]

    async def test_truncation_status(self):
        """Test truncation status for multi-turn conversations."""
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


class TestAPIServerCommon(VLLMTestBase):
    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_auto_tool_choice = True
        self.config.explorer.rollout_model.tool_call_parser = "qwen3_coder"
        self.config.explorer.rollout_model.enable_history = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]
        self.model_wrapper_no_history = clone_wrapper(self.model_wrapper, enable_history=False)

    async def test_api(self):
        openai_client = self.model_wrapper.get_openai_client()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ]
        model_id = openai_client.models.list().data[0].id
        response = openai_client.chat.completions.create(
            model=model_id, messages=messages, n=1, stream=True
        )
        content = ""
        for chunk in response:
            content += chunk.choices[0].delta.content or ""
            self.assertTrue(len(chunk.choices) == 1)
        self.assertTrue(len(content) > 0)
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
        self.assertEqual(exps[0].response_text, content)
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


class TestQwen35APIServerReasoning(VLLMTestBase):
    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_vision_language_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_auto_tool_choice = True
        self.config.explorer.rollout_model.tool_call_parser = "qwen3_coder"
        self.config.explorer.rollout_model.reasoning_parser = "qwen3"
        self.config.explorer.rollout_model.enable_history = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]

    async def test_reasoning_content(self):
        openai_client = self.model_wrapper.get_openai_client()
        model_id = openai_client.models.list().data[0].id
        # test reasoning content
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please give me all available agents."},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "\n\n"}],
                "tool_calls": [
                    {
                        "id": "call_9ab63a0fa4fd4c398339e229",
                        "type": "function",
                        "function": {"name": "list_agents", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "Use `list_agents` tool to get the list of agents.",
            },
            {
                "role": "tool",
                "tool_call_id": "call_9ab63a0fa4fd4c398339e229",
                "content": '{"agents": ["agent_1", "agent_2", "agent_3"]}',
            },
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_agents",
                    "parameters": {
                        "properties": {
                            "base_url": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            }
                        },
                        "type": "object",
                    },
                    "description": "List all configured agents from the QwenPaw service.",
                },
            },
        ]
        _ = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=1,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
            tools=tools,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
        text = self.tokenizer.decode(exps[0].tokens.tolist())
        self.assertIn("Use `list_agents` tool to get the list of agents.", text)

    async def test_thinking_token_budget_action_mask(self):
        thinking_token_budget = 3
        messages = [{"role": "user", "content": "Which is larger, 9.11 or 9.8? Explain."}]
        task = Task(
            raw_task={"messages": messages},
            workflow_args={"thinking_token_budget": thinking_token_budget},
            rollout_args=GenerationConfig(n=1, max_tokens=32, temperature=0.0),
        )

        exps = ThinkingBudgetWorkflow(task=task, model=self.model_wrapper).run()

        self.assertEqual(len(exps), 1)
        exp = exps[0]
        response_token_ids = exp.tokens[exp.prompt_length :].tolist()
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
        reasoning_start_token_ids = tokenizer.encode("<think>", add_special_tokens=False)
        reasoning_end_token_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.assertGreater(len(reasoning_start_token_ids), 0)
        self.assertGreater(len(reasoning_end_token_ids), 0)

        def find_subsequence(token_ids, subsequence):
            return next(
                (
                    index
                    for index in range(len(token_ids) - len(subsequence) + 1)
                    if token_ids[index : index + len(subsequence)] == subsequence
                ),
                -1,
            )

        reasoning_start = find_subsequence(response_token_ids, reasoning_start_token_ids)
        reasoning_end_start = find_subsequence(response_token_ids, reasoning_end_token_ids)
        self.assertGreaterEqual(reasoning_end_start, 0)
        reasoning_content_start = (
            reasoning_start + len(reasoning_start_token_ids) if reasoning_start >= 0 else 0
        )
        self.assertGreaterEqual(reasoning_end_start, reasoning_content_start)
        self.assertLessEqual(
            reasoning_end_start - reasoning_content_start,
            thinking_token_budget,
        )

        reasoning_end_stop = reasoning_end_start + len(reasoning_end_token_ids)
        self.assertTrue(torch.all(exp.action_mask[:reasoning_end_start]))
        self.assertTrue(torch.all(~exp.action_mask[reasoning_end_start:reasoning_end_stop]))
        self.assertTrue(torch.all(exp.action_mask[reasoning_end_stop:]))


class TestQwen35APIServerMultiModal(VLLMTestBase):
    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_vision_language_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_auto_tool_choice = True
        self.config.explorer.rollout_model.tool_call_parser = "qwen3_coder"
        self.config.explorer.rollout_model.enable_history = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]

    async def test_multi_modal_content(self):
        openai_client = self.model_wrapper.get_openai_client()
        model_id = openai_client.models.list().data[0].id
        # test multi-modal content
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
        ]
        _ = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=1,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
        )
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 1)
        exp = exps[0]
        self.assertSetEqual(
            set(exp.multi_modal_inputs.keys()),
            {"mm_token_type_ids", "pixel_values", "image_grid_thw"},
        )
        self.assertEqual(exp.multi_modal_inputs["mm_token_type_ids"].size(1), len(exp.tokens))


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


class TestLogprobs(VLLMTestBase):
    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_history = True
        self.config.explorer.rollout_model.enable_log_requests = True

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]

    async def test_logprobs_api(self):
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


class TestAsyncAPIServer(VLLMTestBase):
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
        self.config.explorer.rollout_model.enable_history = True

        self.config.check_and_update()

    async def _setup_engines(self):
        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]
        self.model_wrapper_no_history = clone_wrapper(self.model_wrapper, enable_history=False)

    async def test_api_async(self):
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
            stream=True,
            temperature=0.5,
            logprobs=True,
            top_logprobs=0,
            max_tokens=10,
        )
        contents = ["", "", "", ""]
        async for chunk in response:
            for choice in chunk.choices:
                contents[choice.index] += choice.delta.content or ""
        exps = self.model_wrapper.extract_experience_from_history()
        self.assertEqual(len(exps), 4)
        for exp in exps:
            self.assertTrue(len(exp.tokens) > 0)
            self.assertTrue(len(exp.logprobs) > 0)
            self.assertTrue(exp.prompt_length + len(exp.logprobs) == len(exp.tokens))
            self.assertTrue(exp.response_text in contents)
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


class TestConcurrentSyncWeights(VLLMTestBase):
    """The vLLM engine must keep serving OpenAI chat completions while a weight
    sync is in progress.

    When ``sync_model_weights`` is invoked it pauses generation, swaps weights and
    resumes. Requests that were already submitted must not be dropped or crash —
    after the sync finishes they should still produce normal content, and the
    engine should keep accepting new requests.
    """

    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 4
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_history = True
        # Use the checkpoint-based weight sync. The checkpoint holds the *same*
        # weights as the running model, so the swap is a semantic no-op and the
        # test isolates the concurrency behavior rather than the weight values.
        self.config.explorer.rollout_model.sync_method = SyncMethod.CHECKPOINT
        self.config.explorer.rollout_model.extra_engine_args = {
            "attention_backend": "FLASHINFER",
        }
        # A throwaway checkpoint root for this test run.
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.check_and_update()

        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]
        self.openai_client = self.model_wrapper.get_openai_async_client()
        self.model_id = self.openai_client.model_path
        master_addr, master_port = await self.model_wrapper.get_available_address_async()
        # Stand up the weight-transfer process group for the single inference rank
        # (this mirrors the single-engine deployment path in Explorer).
        await self.model_wrapper.init_process_group(
            master_address=master_addr,
            master_port=master_port,
            rank_offset=0,
            world_size=4,
            group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
        )

        # Materialise an identical checkpoint at the path the engine expects:
        #   <checkpoint_job_dir>/global_step_<version>/actor/huggingface/
        self._target_version = 1
        huggingface_dir = os.path.join(
            self.config.get_checkpoint_job_dir(),
            f"global_step_{self._target_version}",
            "actor",
            "huggingface",
        )
        os.makedirs(huggingface_dir, exist_ok=True)
        for entry in os.listdir(self.config.model.model_path):
            link = os.path.join(huggingface_dir, entry)
            if not os.path.exists(link):
                os.symlink(os.path.join(self.config.model.model_path, entry), link)

    async def asyncTearDown(self):
        try:
            await self.model_wrapper.teardown_process_group()
        except Exception:
            pass
        await super().asyncTearDown()
        checkpoint_path = get_checkpoint_path()
        shutil.rmtree(os.path.join(checkpoint_path, "unittest"), ignore_errors=True)

    async def test_chat_during_weight_sync(self):  # noqa: C901
        # Use the diverse GSM8K math questions as prompts so each request hits
        # different content rather than repeating one fixed prompt.
        questions = _load_gsm8k_questions()
        self.assertGreater(len(questions), 0)

        # CHECKPOINT sync reloads the full model, so it is slow. To faithfully
        # exercise the concurrency invariant we keep a pool of chat completions
        # in flight for the *entire* sync window — topping up as requests finish
        # until `sync_model_weights` returns.
        concurrency = 4
        temperature = 1.0
        contents: list[str] = []
        interrupted_contents: list[dict] = []  # responses that spanned the weight sync boundary
        errors: list[BaseException] = []
        sync_done = asyncio.Event()
        submit_idx = {"i": 0}

        def next_messages() -> list[dict]:
            question = questions[submit_idx["i"] % len(questions)]
            submit_idx["i"] += 1
            return [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Solve the problem step by step.",
                },
                {"role": "user", "content": question},
            ]

        async def one_request() -> dict:
            messages = next_messages()
            version_before = await self.model_wrapper.model_version_async
            response = await self.openai_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=2048,
            )
            version_after = await self.model_wrapper.model_version_async
            content = response.choices[0].message.content
            return {
                "content": content,
                "question": messages[-1]["content"],
                "version_before": version_before,
                "version_after": version_after,
                "interrupted": version_before != version_after,
            }

        async def submitter():
            in_flight: set[asyncio.Task] = set()
            while True:
                # Keep the pool full while the sync is still running.
                while len(in_flight) < concurrency and not sync_done.is_set():
                    in_flight.add(asyncio.create_task(one_request()))
                if not in_flight:
                    break  # sync finished and nothing left to drain
                done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        result = task.result()
                        contents.append(result["content"])
                        if result["interrupted"]:
                            interrupted_contents.append(result)
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)

        submitter_task = asyncio.create_task(submitter())

        # Let a couple of requests actually start before triggering the sync.
        await asyncio.sleep(10)
        # Trigger the (slow) weight sync while chat requests keep flowing.
        await self.model_wrapper.sync_model_weights(self._target_version, SyncMethod.CHECKPOINT)
        sync_done.set()

        await submitter_task

        # Extract all experiences from history to match with interrupted requests
        all_experiences = self.model_wrapper.extract_experience_from_history()

        # Print interrupted responses that spanned the weight sync boundary
        if interrupted_contents:
            print(
                f"\n{'=' * 60}\n"
                f" {len(interrupted_contents)} request(s) interrupted by weight sync "
                f"(model_version changed during generation)\n"
                f"{'=' * 60}"
            )

            # Match interrupted requests with their experiences and verify logprobs consistency
            for idx, item in enumerate(interrupted_contents):
                print(
                    f"\n--- Interrupted Request {idx + 1} ---\n"
                    f"  model_version: {item['version_before']} -> {item['version_after']}\n"
                    f"  Question: {item['question'][:120]}...\n"
                    f"  Response: {item['content']}\n"
                )

                # Find matching experience by comparing response text
                matching_exp = None
                for exp in all_experiences:
                    if exp.response_text == item["content"]:
                        matching_exp = exp
                        break

                if matching_exp:
                    print("  Original Experience Data:")
                    print(f"    - logprobs_shape: {matching_exp.logprobs.shape}")
                    print(f"    - prompt_length: {matching_exp.prompt_length}")
                    print(f"    - total_tokens: {len(matching_exp.tokens)}")

                    # Recompute logprobs on the original tokens (prompt + response) using
                    # the post-sync model. This verifies that the weight sync did not
                    # corrupt the model: the same token sequence should yield nearly
                    # identical logprobs before and after the sync.
                    print("  Recomputing logprobs on original tokens after weight sync...")
                    recomputed_logprobs = self.model_wrapper.logprobs(
                        matching_exp.tokens.tolist(), temperature=temperature
                    )
                    # logprobs() returns shape (num_tokens - 1,), where logprobs[i] is
                    # the log-probability of token[i+1] given token[:i+1].
                    # The experience stores only the response portion, i.e. from index
                    # (prompt_length - 1) onwards.
                    original_response_logprobs = matching_exp.logprobs
                    recomputed_response_logprobs = recomputed_logprobs[
                        matching_exp.prompt_length - 1 :
                    ]

                    print("  Logprobs Comparison:")

                    self.assertEqual(
                        original_response_logprobs.shape,
                        recomputed_response_logprobs.shape,
                        "logprobs shape mismatch between original and recomputed",
                    )

                    # Use torch.allclose with tolerances similar to test_logprobs_api
                    logprobs_similar = torch.allclose(
                        original_response_logprobs,
                        recomputed_response_logprobs,
                        rtol=0.4,
                        atol=1e-2,
                    )
                    print(f"    - logprobs_similar (rtol=0.4, atol=1e-2): {logprobs_similar}")

                    if logprobs_similar:
                        print("    ✓ Logprobs are consistent after weight sync")
                    else:
                        print("    ✗ Logprobs differ after weight sync")
                        abs_diff = torch.abs(
                            original_response_logprobs - recomputed_response_logprobs
                        )
                        mean_diff = torch.mean(abs_diff).item()
                        max_diff = torch.max(abs_diff).item()
                        print(f"    - mean_abs_diff: {mean_diff:.6f}")
                        print(f"    - max_abs_diff: {max_diff:.6f}")

                        # Find positions where the difference exceeds tolerance
                        # torch.allclose uses: |a - b| <= atol + rtol * |b|
                        tolerance = 1e-2 + 0.4 * torch.abs(recomputed_response_logprobs)
                        mismatch_mask = abs_diff > tolerance
                        mismatch_indices = torch.where(mismatch_mask)[0]

                        print(
                            f"    - num_mismatched_positions: {len(mismatch_indices)} / {len(original_response_logprobs)}"
                        )

                        if len(mismatch_indices) > 0:
                            # Load tokenizer to decode mismatched tokens
                            _tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
                            # response tokens start at prompt_length in matching_exp.tokens
                            response_tokens = matching_exp.tokens[matching_exp.prompt_length :]

                            print("    - Top 5 largest mismatches:")
                            # Get top 5 largest differences
                            top_k = min(5, len(mismatch_indices))
                            top_diffs, top_indices = torch.topk(abs_diff[mismatch_mask], top_k)

                            for i, (diff_val, idx) in enumerate(
                                zip(top_diffs, mismatch_indices[top_indices])
                            ):
                                orig_val = original_response_logprobs[idx].item()
                                recomp_val = recomputed_response_logprobs[idx].item()
                                tol_val = tolerance[idx].item()
                                # logprobs[i] is the log-prob of token[i+1],
                                # so the mismatched token is response_tokens[idx]
                                token_id = response_tokens[idx].item()
                                token_text = _tokenizer.decode([token_id])
                                # ANSI red: \033[91m ... \033[0m
                                red_token = f"\033[91m{repr(token_text)}\033[0m"
                                print(
                                    f"      [{i + 1}] position={idx.item()}, "
                                    f"token={red_token} (id={token_id}): "
                                    f"original={orig_val:.6f}, recomputed={recomp_val:.6f}, "
                                    f"diff={diff_val.item():.6f}, tolerance={tol_val:.6f}"
                                )

                            # Print the full response text with mismatched tokens highlighted in red
                            print("    - Response text with mismatched tokens highlighted in red:")
                            highlighted_parts = []
                            mismatch_set = set(mismatch_indices.tolist())
                            for pos in range(len(response_tokens)):
                                token_text = _tokenizer.decode([response_tokens[pos].item()])
                                if pos in mismatch_set:
                                    highlighted_parts.append(f"\033[91m{token_text}\033[0m")
                                else:
                                    highlighted_parts.append(token_text)
                            print(f"      {''.join(highlighted_parts)}")

                    self.assertTrue(
                        logprobs_similar,
                        f"Logprobs for interrupted request {idx + 1} are not consistent "
                        f"after weight sync (mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}, "
                        f"num_mismatched={len(mismatch_indices) if not logprobs_similar else 0})"
                        if not logprobs_similar
                        else "",
                    )
                else:
                    print("  [WARNING] No matching experience found in history")

            print(f"{'=' * 60}\n")
        else:
            print(
                "\n[INFO] No requests were interrupted by weight sync "
                "(model_version did not change during any generation)."
            )

        self.assertEqual(errors, [], f"some chat requests failed: {errors!r}")
        self.assertGreater(len(contents), 0)
        for content in contents:
            self.assertIsNotNone(content)
            self.assertGreater(len(content), 0)


@parameterized_class(
    ("enable_thinking", "reasoning_parser"),
    [
        (True, "deepseek_r1"),
        (False, None),
    ],
)
class TestAPIServerToolCall(VLLMTestBase):
    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_vision_language_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_history = True
        # added for toolcalls
        self.config.explorer.rollout_model.enable_auto_tool_choice = True
        self.config.explorer.rollout_model.tool_call_parser = "hermes"
        self.config.explorer.rollout_model.enable_thinking = self.enable_thinking
        self.config.explorer.rollout_model.reasoning_parser = self.reasoning_parser

        self.config.check_and_update()
        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]
        self.model_wrapper_no_history = clone_wrapper(self.model_wrapper, enable_history=False)

    async def test_api_tool_calls(self):
        """Tests the full conversation flow of a tool call via the OpenAI API.
        Note: This test require a model that supports tool calls and thinking mode, e.g. Qwen3-1.7B.
        """
        import json
        import time

        tokenizer = AutoTokenizer.from_pretrained(get_vision_language_model_path())
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
            self.assertIsNotNone(choice.message.reasoning)
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


class TestSuperLongGeneration(VLLMTestBase):
    async def asyncSetUp(self):
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
        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]

    async def test_generate(self):
        base_dir = os.path.dirname(__file__)
        target_dir = os.path.join(base_dir, "..", "..", "trinity", "trainer", "verl_legacy")
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


class TestTinkerAPI(VLLMTestBase):
    """Test the Tinker API integration with the vLLM engine."""

    async def asyncSetUp(self):
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
        self.engines, self.auxiliary_engines = await create_test_models(self.config)
        self.model_wrapper = self.engines[0]

    async def test_tinker_api(self):
        from tinker import types
        from transformers import AutoTokenizer

        engine = self.model_wrapper.model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ]
        result_dict = tokenizer.apply_chat_template(
            messages,
            chat_template=CHAT_TEMPLATE,
            add_generation_prompt=True,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
            enable_thinking=False,
        )
        prompt = types.ModelInput.from_ints(
            result_dict["input_ids"][0].tolist(),
        )
        # sample api without prompt logprobs
        num_samples = 4
        response = await engine.sample.remote(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=types.SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=20,
            ),  # no limit on length
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
