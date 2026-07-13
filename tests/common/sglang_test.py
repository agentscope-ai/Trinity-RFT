import asyncio

import httpx
import openai
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
from trinity.buffer.store import get_record_key
from trinity.common.experience import Experience
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
        (2, 1, 1, 2, 1, True, True),
        (2, 1, 2, 1, 2, False, False),
        (4, 1, 1, 1, 2, True, True),
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
        self.config.explorer.rollout_model.enable_history = self.enable_history
        allocator = Allocator(self.config.explorer)
        rollout_models, _ = await allocator.create_all_models()
        self.model_wrapper = rollout_models[0]
        self.record_key = "0/sglang_openai_api/0"
        if self.enable_history:
            self.model_wrapper.set_api_key(self.record_key)
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
            with self.assertRaises(ValueError):
                self.model_wrapper.extract_experience_from_history()
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
            enable_recording=self.enable_history,
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
            with self.assertRaises(ValueError):
                self.model_wrapper.extract_experience_from_history()

        generate_prompt = "Write one short sentence about Boston."
        generate_exps = await self.model_wrapper.generate_async(
            [generate_prompt],
            enable_recording=self.enable_history,
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
            with self.assertRaises(ValueError):
                self.model_wrapper.extract_experience_from_history()


class TestRecording(RayUnittestBaseAsync):
    """Correctness of the in-SGLang generation recording flow (``enable_history``).

    Mirrors ``tests/common/vllm_test.py::TestRecording``. Verifies that every
    call path lands its finished turn in the in-process ``MemoryStore`` under
    the right ``record_key``, and that actor-side reward update + drain APIs
    stamp and return recorded experiences.

    Paths covered (all async):
      * Ray-direct ``generate`` / ``chat`` — SGLang's Ray-direct path is over
        HTTP (unlike vLLM's in-process call), so ``record_key`` travels as the
        ``Authorization: Bearer <record_key>`` header.
      * OpenAI HTTP regular / streaming / tool-augmented — same bearer path.

    Recording disables SGLang's api_key auth middleware (Option A, see
    ``sglang_patch/server_patch.py``), so the bearer is used purely as the
    per-task ``record_key`` (captured by ``RecordingIdentityMiddleware``),
    matching vLLM (which sets no api_key auth in recording mode).

    ``enable_router_replay`` (mirrored to ``enable_return_routed_experts`` by
    ``check_and_update``) is on, so this test uses a MoE checkpoint
    (``get_moe_model_path``) and asserts routed_experts shapes.
    """

    async def asyncSetUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        # enable_router_replay drives enable_return_routed_experts (see
        # ``config_validator``) -> needs a MoE model (otherwise routed_experts
        # is absent and the shape asserts below would fail). Use a Qwen3-MoE
        # checkpoint.
        self.config.model.model_path = get_moe_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_path,
            trust_remote_code=True,
        )
        self.text_config = _get_text_config(self.config.model.model_path)
        self.expected_routed_experts_layers = int(self.text_config.num_hidden_layers)
        self.expected_routed_experts_topk = int(self.text_config.num_experts_per_tok)
        self.config.model.custom_chat_template = CHAT_TEMPLATE
        self.config.explorer.rollout_model.engine_type = "sglang"
        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 2
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        # enable_history requires the OpenAI API server (the recording runner).
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.explorer.rollout_model.enable_history = True
        self.config.explorer.rollout_model.enable_expert_parallel = True
        # enable_router_replay is mirrored to enable_return_routed_experts by
        # ``check_and_update`` (config_validator); it is NOT implied by
        # enable_history. The routed-experts asserts below require it on, so
        # the in-SGLang recorder captures routed_experts on every path.
        self.config.algorithm.enable_router_replay = True
        # Tool-call parsing coverage (qwen3_coder matches the Qwen3.5 chat
        # template). SGLang enables tool calling via tool_call_parser (no
        # separate enable_auto_tool_choice flag); enable_auto_tool_choice is
        # set for parity with the vLLM TestRecording config.
        self.config.explorer.rollout_model.enable_auto_tool_choice = True
        self.config.explorer.rollout_model.tool_call_parser = "qwen3_coder"
        self.config.explorer.rollout_model.enable_thinking = False
        # The in-SGLang recorder is the subject.
        self.config.explorer.rollout_model.base_port = 13400
        self.config.check_and_update()

        allocator = Allocator(self.config.explorer)
        rollout_models, _ = await allocator.create_all_models()
        self.model_wrapper = rollout_models[0]
        self.api_address = self.model_wrapper.api_address
        self._http = httpx.AsyncClient(timeout=120.0)
        self._model_id = None

    async def asyncTearDown(self):
        await self._http.aclose()
        await self.model_wrapper.shutdown()
        await super().asyncTearDown()

    # -- actor-side recording store helpers -----------------------------------

    async def _consume(self, record_key: str, reward: float) -> list[Experience]:
        await self.model_wrapper.update_experience_reward_async(record_key, reward=reward)
        payload = await self.model_wrapper.drain_experience_records_bytes_async(record_key)
        return Experience.deserialize_many(payload)

    async def _openai_client(self, record_key: str) -> openai.AsyncOpenAI:
        # record_key travels as the Bearer api_key -> RecordingIdentityMiddleware.
        return openai.AsyncOpenAI(base_url=f"{self.api_address}/v1", api_key=record_key)

    async def _get_model_id(self, client: openai.AsyncOpenAI) -> str:
        if self._model_id is None:
            self._model_id = (await client.models.list()).data[0].id
        return self._model_id  # type: ignore [return-value]

    # -- per-recorded-experience invariants -----------------------------------

    def _assert_recorded_experience(self, exp: Experience, record_key: str):
        self.assertEqual(get_record_key(exp), record_key)
        self.assertTrue(exp.eid.suffix)
        # SGLang stamps meta_info.weight_version ("default" until a weight sync);
        # unlike vLLM it is a server-tracked string, not the model_version int.
        self.assertIsNotNone(exp.info.get("model_version"))
        self.assertGreater(len(exp.tokens), exp.prompt_length)  # type: ignore [arg-type]
        # The recorder forces return_logprob=True even when the client omitted it.
        self.assertGreater(len(exp.logprobs), 0)  # type: ignore [arg-type]
        self.assertEqual(len(exp.logprobs), len(exp.tokens) - exp.prompt_length)  # type: ignore [arg-type]
        # SGLang's ret does not carry prompt text, so prompt_text is None on the
        # recording hot path (decode token ids lazily where a check is needed).
        if exp.prompt_text is not None:
            self.assertGreater(len(exp.prompt_text), 0)
        self.assertGreater(len(exp.response_text), 0)

    def _assert_recorded_routed_experts(self, exp: Experience):
        # enable_router_replay -> enable_return_routed_experts is on for this test.
        self.assertIsNotNone(exp.routed_experts)
        re = exp.routed_experts
        self.assertEqual(re.dtype, torch.uint8)
        self.assertEqual(re.ndim, 3)
        self.assertEqual(re.shape[1], self.expected_routed_experts_layers)
        self.assertEqual(re.shape[2], self.expected_routed_experts_topk)

    async def test_record(self):  # noqa: C901
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one short sentence."},
        ]

        # ===== 1. Ray-direct generate (record_key via Authorization bearer) =====
        rk_gen = "0/t_gen/1"
        await self.model_wrapper.generate_async(
            ["Hello, world!"], n=1, temperature=1.0, max_tokens=16, key=rk_gen
        )
        consumed = await self._consume(rk_gen, reward=0.5)
        self.assertEqual(len(consumed), 1)
        self.assertEqual(consumed[0].reward, 0.5)
        self.assertEqual(consumed[0].eid.run, 1)
        self.assertEqual(consumed[0].eid.task, "t_gen")
        self._assert_recorded_experience(consumed[0], rk_gen)
        self._assert_recorded_routed_experts(consumed[0])

        # ===== 2. Ray-direct chat, n=2 (one record-key group, two samples) =====
        rk_chat = "0/t_chat/2"
        chat_exps = await self.model_wrapper.chat_async(
            messages, n=2, temperature=1.0, max_tokens=16, key=rk_chat
        )
        self.assertEqual(len(chat_exps), 2)
        consumed = await self._consume(rk_chat, reward=0.8)
        self.assertEqual(len(consumed), 2)
        # SGLang expands n=2 parallel sampling into two scheduler requests.
        # The list position becomes sample_index (0, 1) to order the two
        # samples within the record-key group.
        self.assertEqual(sorted(exp.info["sample_index"] for exp in consumed), [0, 1])
        self.assertEqual(len({exp.eid.suffix for exp in consumed}), 2)
        for exp in consumed:
            self.assertEqual(exp.reward, 0.8)
            self.assertEqual(exp.eid.run, 2)
            self.assertEqual(exp.eid.task, "t_chat")
            self._assert_recorded_experience(exp, rk_chat)
            self._assert_recorded_routed_experts(exp)

        # ===== 3. OpenAI regular (HTTP; record_key = Bearer api_key) =====
        rk_oai = "0/t_oai/3"
        client = await self._openai_client(rk_oai)
        model_id = await self._get_model_id(client)
        resp = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=1,
            temperature=0.7,
            max_tokens=32,
        )
        consumed = await self._consume(rk_oai, reward=0.3)
        self.assertEqual(len(consumed), 1)
        self._assert_recorded_experience(consumed[0], rk_oai)
        self._assert_recorded_routed_experts(consumed[0])
        # No reasoning_parser is configured, so message.content == ret.text.
        self.assertEqual(consumed[0].response_text, resp.choices[0].message.content)

        # ===== 4. OpenAI streaming (HTTP) =====
        rk_str = "0/t_str/4"
        sclient = await self._openai_client(rk_str)
        stream = await sclient.chat.completions.create(
            model=model_id,
            messages=messages,
            n=1,
            stream=True,
            temperature=0.7,
            max_tokens=32,
        )
        content = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                content += delta
        self.assertGreater(len(content), 0)
        consumed = await self._consume(rk_str, reward=0.1)
        self.assertEqual(len(consumed), 1)
        self._assert_recorded_experience(consumed[0], rk_str)
        self._assert_recorded_routed_experts(consumed[0])
        response_token_ids = consumed[0].tokens[consumed[0].prompt_length :].tolist()
        decoded_content = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)
        self.assertEqual(decoded_content, content)
        self.assertEqual(consumed[0].response_text, content)

        # ===== 5. OpenAI tool-call parsing (HTTP) =====
        rk_tool = "0/t_tool/5"
        tclient = await self._openai_client(rk_tool)
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
                                "description": "The city and state, e.g. San Francisco, CA",
                                "type": "string",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        tool_messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
        no_think = {"chat_template_kwargs": {"enable_thinking": False}}
        tresp = await tclient.chat.completions.create(
            model=model_id,
            messages=tool_messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=64,
            extra_body=no_think,
        )
        consumed = await self._consume(rk_tool, reward=1.0)
        self.assertEqual(len(consumed), 1)
        self._assert_recorded_experience(consumed[0], rk_tool)
        self._assert_recorded_routed_experts(consumed[0])
        # tool_choice != "none" -> SGLang renders the tool defs into the prompt
        # (serving_chat._process_messages), so the recorded prompt tokens carry
        # the tool name. SGLang's ret does not carry prompt text, so decode.
        decoded = self.tokenizer.decode(consumed[0].tokens.tolist(), skip_special_tokens=False)
        self.assertIn("get_current_weather", decoded)
        # If the model emitted a tool call, its function name is in the raw
        # recorded response text (ret.text), which the qwen3_coder parser also
        # surfaces as choice.message.tool_calls.
        choice = tresp.choices[0]
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                self.assertIn(tc.function.name, consumed[0].response_text)

        # ===== global: every group consumed -> store is drained =====
        await self.model_wrapper.delete_experience_records_async("0")
