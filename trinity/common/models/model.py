# -*- coding: utf-8 -*-
"""Base Model Class"""
import asyncio
import copy
import socket
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx
import numpy as np
import openai
import ray
import torch
from PIL import Image
from torch import Tensor

from trinity.common.config import InferenceModelConfig
from trinity.common.constants import RunningStatus
from trinity.common.experience import Experience
from trinity.common.models.utils import get_action_mask_method
from trinity.utils.log import get_logger


class InferenceModel(ABC):
    """A model for high performance for rollout inference."""

    def __init__(self, config: InferenceModelConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    async def generate(self, prompt: str, **kwargs) -> Sequence[Experience]:
        """Generate a responses from a prompt in async."""
        raise NotImplementedError

    async def chat(self, messages: List[dict], **kwargs) -> Sequence[Experience]:
        """Generate experiences from a list of history chat messages in async."""
        raise NotImplementedError

    async def logprobs(self, token_ids: List[int], **kwargs) -> Tensor:
        """Generate logprobs for a list of tokens in async."""
        raise NotImplementedError

    async def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        """Convert a list of messages into an experience in async."""
        raise NotImplementedError

    async def prepare(self) -> None:
        """Prepare the model before inference."""
        pass

    @abstractmethod
    async def sync_model(self, model_version: int) -> int:
        """Sync the model with the latest model_version."""

    @abstractmethod
    def get_model_version(self) -> int:
        """Get the checkpoint version."""

    def get_available_address(self) -> Tuple[str, int]:
        """Get the address of the actor."""
        address = ray.util.get_node_ip_address()
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return address, port

    def get_api_server_url(self) -> Optional[str]:
        """Get the API server URL if available."""
        return None

    def get_api_key(self) -> str:
        """Get the API key."""
        return "EMPTY"

    def get_model_config(self) -> InferenceModelConfig:
        """Get the model configuration."""
        return self.config

    def get_model_path(self) -> Optional[str]:
        """Get the model path"""
        return self.config.model_path


class BaseInferenceModel(InferenceModel):
    """Base class for inference models containing common logic."""

    def __init__(self, config: InferenceModelConfig) -> None:
        super().__init__(config)
        self.tokenizer = None
        self.chat_template = None
        if self.config.chat_template:
            self.chat_template = self.config.chat_template
        self.action_mask_method = get_action_mask_method(self.chat_template)
        self.enable_thinking = config.enable_thinking

    def apply_chat_template(
        self,
        tokenizer_or_processor,
        messages: List[dict],
    ) -> str:
        assert tokenizer_or_processor is not None, "tokenizer_or_processor must be provided."

        if messages[-1]["role"] == "assistant":
            prompt = tokenizer_or_processor.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True,
                chat_template=self.chat_template,
            )
        else:
            prompt = tokenizer_or_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template,
                enable_thinking=self.enable_thinking,
            )
        return prompt

    def _handle_prompt_truncation(self, prompt: str, **kwargs) -> Tuple[Sequence, bool]:
        """Handle prompt truncation if needed."""
        # Tokenize once without truncation to check if truncation is needed
        token_ids = self.tokenizer(  # type: ignore
            prompt,
            truncation=False,
            return_tensors="pt",
        )[
            "input_ids"
        ][0].tolist()

        # Check if truncation is needed and apply it
        if (
            self.config.enable_prompt_truncation
            and self.config.max_prompt_tokens is not None
            and len(token_ids) > self.config.max_prompt_tokens
        ):
            self.logger.warning(f"Prompt was truncated to {self.config.max_prompt_tokens} tokens")
            token_ids = token_ids[: self.config.max_prompt_tokens + 1]  # leave one for response
            return [
                Experience(
                    tokens=token_ids,
                    logprobs=torch.zeros(1, dtype=torch.float32),
                    prompt_length=len(token_ids) - 1,
                    prompt_text=self.tokenizer.decode(token_ids[:-1]),
                    response_text=self.tokenizer.decode(token_ids[-1]),
                    truncate_status="prompt_truncated",
                    reward=0.0,
                )
                for _ in range(kwargs.get("n", 1))
            ], False
        return token_ids, True

    async def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        """Convert a list of messages into an experience in async.

        Args:
            messages: List of message dictionaries
            tools: Optional list of tools
            temperature: Optional temperature for logprobs calculation
        """
        if self.tokenizer is None:
            await self._initialize_tokenizer()
        token_ids, action_mask, prompt_length = self.action_mask_method(
            tokenizer=self.tokenizer,
            messages=messages,
            tools=tools,
            chat_template=self.chat_template,
            enable_thinking=self.enable_thinking,
        )  # (seq_length, ), (seq_length, )

        assert token_ids is not None
        truncate_status = None
        # Truncate prompt if it exceeds max_prompt_tokens
        if (
            self.config.enable_prompt_truncation
            and self.config.max_prompt_tokens is not None
            and prompt_length > self.config.max_prompt_tokens
        ):
            truncate_status = "prompt_truncated"
            self.logger.warning(
                f"Warning: {prompt_length=} exceeds the length limit {self.config.max_prompt_tokens}, "
                f"this experience will be not counted in the loss computation."
            )
            return Experience(
                tokens=token_ids[: self.config.max_prompt_tokens + 1],
                logprobs=torch.zeros(1, dtype=torch.float32),
                prompt_length=self.config.max_prompt_tokens,  # Use truncated length
                action_mask=torch.zeros(1, dtype=torch.bool),  # ignored in loss computation
                messages=messages,  # messages are not truncated
                truncate_status=truncate_status,
            )

        # Truncate response if it exceeds max_model_len
        max_model_len = self.config.max_model_len
        if max_model_len is not None and len(token_ids) > max_model_len - 1:
            truncate_status = "response_truncated"
            self.logger.warning(
                f"Warning: {len(token_ids)=} exceeds the length limit {(max_model_len - 1)=}"
            )
            token_ids = token_ids[: max_model_len - 1]
            action_mask = action_mask[: max_model_len - 1]

        temperature = temperature if temperature is not None else self.config.temperature
        logprobs = await self.logprobs(
            token_ids=token_ids.tolist(), temperature=temperature
        )  # (seq_length - 1,)

        return Experience(
            tokens=token_ids,
            logprobs=logprobs[prompt_length - 1 :],
            prompt_length=prompt_length,
            action_mask=action_mask[prompt_length:],  # Exclude the prompt tokens
            messages=messages,
            truncate_status=truncate_status,
        )


def _history_recorder(func):
    """Decorator to record history of the model calls."""

    async def async_wrapper(self, *args, **kwargs):
        result = await func(self, *args, **kwargs)
        if self.enable_history:
            self._record_history(result)
        return result

    def sync_wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.enable_history:
            self._record_history(result)
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class ModelWrapper:
    """A wrapper for the InferenceModel Ray Actor"""

    def __init__(
        self,
        model: InferenceModel,
        engine_type: str = "vllm",
        enable_lora: bool = False,
        enable_history: bool = False,
    ):
        """Initialize the ModelWrapper.

        Args:
            model (InferenceModel): The inference model Ray actor.
            engine_type (str): The type of the model engine. Default to "vllm".
            enable_lora (bool): Whether to enable LoRA. Default to False.
            enable_history (bool): Whether to enable history recording. Default to False.
        """
        assert (
            engine_type.startswith("vllm") or engine_type == "tinker"
        ), "Only vLLM and tinker model is supported for now."
        self.model = model
        self.engine_type = engine_type
        self.config: InferenceModelConfig = None  # init during prepare
        self._model_name: str = None
        self.api_address: str = None
        self._api_key: str = None
        self.openai_client: openai.OpenAI = None
        self.openai_async_client: openai.AsyncOpenAI = None
        self.logger = get_logger(__name__)
        self.enable_lora = enable_lora
        self.enable_history = enable_history
        self.history = []
        self.status = RunningStatus.RUNNING
        self.workflow_state: Dict = {}
        self.request_count = 0
        self.state_lock = asyncio.Lock()

    async def prepare(self) -> None:
        """Prepare the model wrapper."""
        self.config = await self.model.get_model_config.remote()
        self._model_name = self.config.name
        self._api_key = await self.model.get_api_key.remote()
        self._generate_kwargs = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_response_tokens,
        }
        if self.config.enable_thinking is not None:
            self._generate_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking}
            }
        self.api_address = await self.model.get_api_server_url.remote()
        if self.api_address is None:
            self.logger.info("API server is not enabled for inference model.")
            return
        if self.engine_type == "tinker":
            return
        max_retries = 30
        interval = 2  # seconds
        for i in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(self.api_address + "/health", timeout=5)
                    if response.status_code == 200:
                        return
            except Exception as e:
                self.logger.info(f"API server not ready (attempt {i + 1}/{max_retries}): {e}")
            await asyncio.sleep(interval)
        raise RuntimeError(
            f"API server at {self.api_address} not ready after {max_retries} attempts."
        )

    def _record_history(self, exps: Union[Experience, List[Experience]]) -> None:
        """Record experiences to history."""
        if isinstance(exps, Experience):
            self.history.append(exps)
        elif isinstance(exps, list):
            self.history.extend(exps)
        else:
            raise TypeError("Expected Experience or List[Experience], got {}".format(type(exps)))

    @_history_recorder
    def generate(self, prompts: List[str], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of prompts."""
        lora_request = self.get_lora_request()
        results = ray.get(
            [self.model.generate.remote(prompt, lora_request, **kwargs) for prompt in prompts]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    async def generate_async(self, prompts: List[str], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of prompts in async."""
        lora_request = await self.get_lora_request_async()
        results = await asyncio.gather(
            *[self.model.generate.remote(prompt, lora_request, **kwargs) for prompt in prompts]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    def generate_mm(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        videos: List[List[np.ndarray]],
        **kwargs,
    ) -> List[Experience]:
        """Generate a list of experiences from a list of prompts and multi-modal data."""
        results = ray.get(
            [
                self.model.generate_mm.remote(prompt, images=img, videos=vid, **kwargs)
                for prompt, img, vid in zip(prompts, images, videos)
            ]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    async def generate_mm_async(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        videos: List[List[np.ndarray]],
        **kwargs,
    ) -> List[Experience]:
        results = await asyncio.gather(
            *[
                self.model.generate_mm.remote(p, images=img, videos=vid, **kwargs)
                for p, img, vid in zip(prompts, images, videos)
            ]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    def chat(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of messages."""
        lora_request = self.get_lora_request()
        return ray.get(self.model.chat.remote(messages, lora_request=lora_request, **kwargs))

    @_history_recorder
    async def chat_async(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of messages in async."""
        lora_request = await self.get_lora_request_async()
        return await self.model.chat.remote(messages, lora_request=lora_request, **kwargs)

    @_history_recorder
    def chat_mm(
        self, messages: List[dict], images: List[Image.Image], videos: List[np.ndarray], **kwargs
    ) -> List[Experience]:
        return ray.get(self.model.chat_mm.remote(messages, images=images, videos=videos, **kwargs))

    @_history_recorder
    async def chat_mm_async(
        self, messages: List[dict], images: List[Image.Image], videos: List[np.ndarray], **kwargs
    ) -> List[Experience]:
        return await self.model.chat_mm.remote(messages, images=images, videos=videos, **kwargs)

    def logprobs(self, tokens: List[int], temperature: Optional[float] = None) -> Tensor:
        """Calculate the logprobs of the given tokens."""
        return ray.get(self.model.logprobs.remote(tokens, temperature=temperature))

    async def logprobs_async(
        self, tokens: List[int], temperature: Optional[float] = None
    ) -> Tensor:
        """Calculate the logprobs of the given tokens in async."""
        return await self.model.logprobs.remote(tokens, temperature=temperature)

    def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        """Convert a list of messages into an experience."""
        return ray.get(
            self.model.convert_messages_to_experience.remote(
                messages, tools=tools, temperature=temperature
            )
        )

    async def convert_messages_to_experience_async(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        """Convert a list of messages into an experience in async."""
        return await self.model.convert_messages_to_experience.remote(
            messages, tools=tools, temperature=temperature
        )

    @property
    def api_key(self) -> str:
        """Get the API key."""
        return self._api_key

    @property
    def model_version(self) -> int:
        """Get the version of the model."""
        return ray.get(self.model.get_model_version.remote())

    @property
    async def model_version_async(self) -> int:
        """Get the version of the model."""
        return await self.model.get_model_version.remote()

    @property
    def model_path(self) -> str:
        """
        Returns the path to the model files based on the current engine type.

        - For 'vllm' engine: returns the model path from the configuration (`config.model_path`)
        - For 'tinker' engine: returns the path to the most recent sampler weights
        """
        return ray.get(self.model.get_model_path.remote())

    @property
    async def model_path_async(self) -> str:
        """
        Returns the path to the model files based on the current engine type.

        - For 'vllm' engine: returns the model path from the configuration (`config.model_path`)
        - For 'tinker' engine: returns the path to the most recent sampler weights
        """
        return await self.model.get_model_path.remote()

    @property
    def model_name(self) -> Optional[str]:
        """Get the name of the model."""
        return self._model_name

    @property
    def model_config(self) -> InferenceModelConfig:
        """Get the model config."""
        return self.config

    @property
    def generate_kwargs(self) -> Dict[str, Any]:
        """Get the generation kwargs for openai client."""
        return self._generate_kwargs

    def get_lora_request(self) -> Any:
        if self.enable_lora:
            return ray.get(self.model.get_lora_request.remote())
        else:
            return None

    async def get_lora_request_async(self) -> Any:
        if self.enable_lora:
            return await self.model.get_lora_request.remote()
        else:
            return None

    async def get_message_token_len(self, messages: List[dict]) -> int:
        return await self.model.get_message_token_len.remote(messages)

    def get_openai_client(self) -> openai.OpenAI:
        """Get the openai client.

        Returns:
            openai.OpenAI: The openai client. And `model_path` is added to the client which refers to the model path.
        """
        if self.openai_client is not None:
            setattr(self.openai_client, "model_path", self.model_path)
            return self.openai_client
        if not self.api_address:
            raise ValueError(
                "API server is not enabled for this model. OpenAI client is unavailable."
            )
        self.openai_client = openai.OpenAI(
            base_url=f"{self.api_address}/v1",
            api_key=self._api_key,
        )
        if self.engine_type == "tinker":
            # ! TODO: because tinker's OpenAI API interface is in beta,
            # we need to use original API in thinker instead.
            def chat_completions(*args, **kwargs):
                messages = kwargs.pop("messages")
                chat_response = ray.get(
                    self.model.chat.remote(
                        messages=messages,
                        with_chat_completion=True,
                        return_token_ids=self.enable_history,
                        **kwargs,
                    )
                )
                response = chat_response.pop()
                if self.enable_history:
                    self.history.extend(chat_response)
                return response

            self.openai_client.chat.completions.create = chat_completions
        elif self.enable_history:
            # add a decorator to the openai client to record history

            ori_create = self.openai_client.chat.completions.create

            def record_chat_completions(*args, **kwargs):
                logprobs = kwargs.pop("logprobs", True)
                extra_body = kwargs.pop("extra_body", {})
                if self.config.enable_thinking is not None:
                    if "chat_template_kwargs" not in extra_body:
                        extra_body["chat_template_kwargs"] = {}
                    extra_body["chat_template_kwargs"][
                        "enable_thinking"
                    ] = self.config.enable_thinking
                extra_body["return_token_ids"] = True
                response = ori_create(*args, extra_body=extra_body, logprobs=logprobs, **kwargs)
                self.history.extend(convert_api_output_to_experience(response))
                return response

            self.openai_client.chat.completions.create = record_chat_completions
        setattr(self.openai_client, "model_path", self.model_path)
        return self.openai_client

    def get_openai_async_client(self) -> openai.AsyncOpenAI:
        """Get the async openai client.

        Returns:
            openai.AsyncOpenAI: The async openai client. And `model_path` is added to the client which refers to the model path.
        """
        if self.openai_async_client is not None:
            setattr(self.openai_async_client, "model_path", self.model_path)
            return self.openai_async_client
        if not self.api_address:
            raise ValueError(
                "API server is not enabled for this model. OpenAI async client is unavailable."
            )
        # first make sure that we have the sync openai client
        self.openai_async_client = openai.AsyncOpenAI(
            base_url=f"{self.api_address}/v1",
            api_key=self._api_key,
        )

        if self.engine_type == "tinker":
            # ! TODO: because tinker's OpenAI API interface is in beta,
            # we need to use original API in thinker instead.
            async def chat_completions(*args, **kwargs):
                messages = kwargs.pop("messages")
                chat_response = await self.model.chat.remote(
                    messages=messages,
                    with_chat_completion=True,
                    return_token_ids=self.enable_history,
                    **kwargs,
                )
                response = chat_response.pop()
                if self.enable_history:
                    self.history.extend(chat_response)
                return response

            self.openai_async_client.chat.completions.create = chat_completions
        elif self.enable_history:
            # add a decorator to the openai client to record history

            ori_create = self.openai_async_client.chat.completions.create

            async def record_chat_completions(*args, **kwargs):
                logprobs = kwargs.pop("logprobs", True)
                extra_body = kwargs.pop("extra_body", {})
                if self.config.enable_thinking is not None:
                    if "chat_template_kwargs" not in extra_body:
                        extra_body["chat_template_kwargs"] = {}
                    extra_body["chat_template_kwargs"][
                        "enable_thinking"
                    ] = self.config.enable_thinking
                extra_body["return_token_ids"] = True
                response = await ori_create(
                    *args, extra_body=extra_body, logprobs=logprobs, **kwargs
                )
                self.history.extend(convert_api_output_to_experience(response))
                return response

            self.openai_async_client.chat.completions.create = record_chat_completions
        # get model_path from the sync openai client to avoid async call here
        setattr(self.openai_async_client, "model_path", self.model_path)
        return self.openai_async_client

    async def get_current_load(self) -> int:
        """Get the current load metrics of the model."""
        if not self.api_address:
            raise ValueError(
                "API server is not enabled for this model. Load metrics is unavailable."
            )
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_address}/load")
            data = response.json()
            return data["server_load"]

    async def sync_model_weights(self, model_version: int) -> None:
        """Sync the model weights"""
        await self.model.sync_model.remote(model_version)

    def extract_experience_from_history(self, clear_history: bool = True) -> List[Experience]:
        """Extract experiences from the history."""
        if not self.enable_history:
            raise ValueError("History recording is not enabled.")
        exps = [exp for exp in self.history]
        if clear_history:
            self.history.clear()
        return exps

    # Workflow state management methods
    async def set_workflow_state(self, state: Dict) -> None:
        """Set the state of workflow using the model."""
        async with self.state_lock:
            self.workflow_state.update(state)

    async def clean_workflow_state(self) -> None:
        """Clean the state of workflow using the model."""
        async with self.state_lock:
            self.workflow_state = {}
            self.history.clear()

    async def get_workflow_state(self) -> Dict:
        """Get the state of workflow using the model."""
        async with self.state_lock:
            return self.workflow_state.copy()

    def clone_with_isolated_history(self) -> "ModelWrapper":
        """Clone the current ModelWrapper with isolated history."""
        new_wrapper = copy.copy(self)
        new_wrapper.openai_async_client = None
        new_wrapper.openai_client = None
        new_wrapper.history = []
        return new_wrapper


def convert_api_output_to_experience(
    output,
) -> List[Experience]:
    """Convert the API output to a list of experiences."""
    return [
        Experience(
            tokens=torch.cat(
                (
                    torch.tensor(output.prompt_token_ids, dtype=torch.int32),
                    torch.tensor(choice.token_ids, dtype=torch.int32),
                )
            ),
            logprobs=extract_logprobs(choice),
            prompt_length=len(output.prompt_token_ids),
            response_text=choice.message.content,
        )
        for choice in output.choices
    ]


def extract_logprobs(choice) -> Tensor:
    """Extract logprobs from a list of logprob dictionaries."""
    if not hasattr(choice, "logprobs") or choice.logprobs is None:
        return torch.tensor([], dtype=torch.float32)
    return torch.tensor(
        [logprob.logprob for logprob in choice.logprobs.content],
        dtype=torch.float32,
    )
