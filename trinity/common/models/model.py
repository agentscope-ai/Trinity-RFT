# -*- coding: utf-8 -*-
"""Base Model Class"""

import asyncio
import copy
import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx
import ray
import torch
from ray.actor import ActorHandle
from torch import Tensor
from transformers import AutoConfig

from trinity.common.config import InferenceModelConfig
from trinity.common.constants import RunningStatus, SyncMethod
from trinity.common.experience import Experience
from trinity.common.models.experience_extraction import (
    HistoryRecordingStream,
    convert_api_output_to_experience,
    get_routed_experts_layout,
)
from trinity.common.models.mm_utils import should_use_processor, vLLMMultiModalRender
from trinity.common.models.utils import get_action_mask_method
from trinity.utils.log import get_logger

if TYPE_CHECKING:
    import openai


class InferenceModel(ABC):
    """A model for high performance for rollout inference."""

    def __init__(self, config: InferenceModelConfig) -> None:
        self.config = config
        self.ray_actor_name = config.ray_actor_name
        self.logger = get_logger(self.ray_actor_name or __name__, in_ray_actor=True)
        self._prepared = False
        self.master_addr: Optional[str] = None
        self.master_port: Optional[int] = None

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

    async def ready(self) -> bool:
        """Check if the model is ready for inference."""
        return self._prepared

    async def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        explorer_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        state_dict_meta: Optional[List] = None,
    ):
        """Initialize the process group for model weight synchronization."""
        pass

    @abstractmethod
    async def sync_model_weights(
        self, model_version: int, method: SyncMethod, timeout: float = 1200
    ) -> int:
        """Sync the model with the latest model_version."""

    @abstractmethod
    def get_model_version(self) -> int:
        """Get the checkpoint version."""

    def get_available_address(self, random_port: bool = False) -> Tuple[str, int]:
        """Get an available address on the current actor node.

        Args:
            random_port: Whether to skip the configured ``base_port`` convention and
                allocate an ephemeral port on the current node directly.
        """
        address = ray.util.get_node_ip_address()
        if not random_port and self.config.base_port is not None:
            configured_port = self.config.base_port + self.config.engine_id
            with socket.socket() as s:
                try:
                    s.bind(("", configured_port))
                    return address, configured_port
                except OSError:
                    self.logger.warning(
                        "Configured port %s is unavailable for engine %s; falling back to an ephemeral port.",
                        configured_port,
                        self.config.engine_id,
                    )
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return address, port

    def set_master_addr_port(self, master_addr: str, master_port: int):
        """For multi node setup, set the master address and port for distributed communication."""
        self.master_addr = master_addr
        self.master_port = master_port

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

    async def shutdown(self) -> None:
        """Shutdown the model and release resources."""
        pass


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
        self._routed_experts_layout: Optional[Tuple[int, int, Optional[int]]] = None

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

    def _get_routed_experts_layout(self) -> Tuple[int, int, Optional[int]]:
        """Read and memoize the MoE routing layout ``(num_layers, topk, num_experts)``."""
        if self._routed_experts_layout is None:
            model_path = self.config.model_path
            if model_path is None:
                raise ValueError("model_path must be provided to read routed_experts layout.")
            hf_config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=self.config.trust_remote_code
            )
            text_config = getattr(hf_config, "text_config", hf_config)
            num_layers = getattr(text_config, "num_hidden_layers", None)
            topk = getattr(text_config, "num_experts_per_tok", None)
            # Qwen* exposes ``num_experts``; DeepSeek* exposes ``n_routed_experts``.
            num_experts = getattr(text_config, "num_experts", None) or getattr(
                text_config, "n_routed_experts", None
            )
            if num_layers is None or topk is None:
                raise ValueError(
                    "Model config must expose num_hidden_layers and num_experts_per_tok "
                    "to use routed_experts."
                )
            self._routed_experts_layout = (
                int(num_layers),
                int(topk),
                int(num_experts) if num_experts is not None else None,
            )
        return self._routed_experts_layout

    def _build_dummy_routed_experts(self) -> torch.Tensor:
        """Build routed_experts for dummy (prompt-truncated) experiences.
        These tokens are fully masked from the loss but still flow through the MoE
        forward during router replay, so the indices must be valid and spread across
        experts.
        """
        num_layers, topk, num_experts = self._get_routed_experts_layout()
        if num_experts is None:
            raise ValueError(
                "Model config must expose num_experts (or n_routed_experts) to build "
                "dummy routed_experts when enable_return_routed_experts is True."
            )
        seq_len = self.config.max_prompt_tokens
        if seq_len is None:
            raise ValueError(
                "max_prompt_tokens must be set to build dummy routed_experts for truncated prompts."
            )
        idx = torch.arange(seq_len * num_layers * topk, dtype=torch.int64) % num_experts
        return idx.reshape(seq_len, num_layers, topk).to(torch.uint8)

    def _handle_prompt_truncation(self, prompt: str, **kwargs) -> Tuple[Sequence, bool]:
        """Handle prompt truncation if needed."""
        # Tokenize once without truncation to check if truncation is needed
        prompt_token_ids = self.tokenizer(  # type: ignore
            prompt, truncation=False, return_tensors="pt"
        )["input_ids"][0].tolist()

        # Check if truncation is needed and apply it
        if (
            self.config.enable_prompt_truncation
            and self.config.max_prompt_tokens is not None
            and len(prompt_token_ids) > self.config.max_prompt_tokens
        ):
            self.logger.warning(f"Prompt was truncated to {self.config.max_prompt_tokens} tokens")

            dummy_response = "[This experience is masked out due to overlong prompt]"

            token_ids = prompt_token_ids[: self.config.max_prompt_tokens + 1]

            routed_experts = None
            if getattr(self.config, "enable_return_routed_experts", False):
                routed_experts = self._build_dummy_routed_experts()

            return [
                Experience(
                    tokens=token_ids,
                    logprobs=torch.zeros(1, dtype=torch.float32),
                    prompt_length=self.config.max_prompt_tokens,  # Use truncated length
                    prompt_text=self.tokenizer.decode(token_ids[:-1]),
                    response_text=dummy_response,
                    truncate_status="prompt_truncated",
                    reward=0.0,
                    routed_experts=routed_experts,
                )
                for _ in range(kwargs.get("n", 1))
            ], False  # If prompt truncation is activated, return a list of dummy experiences & False
        return prompt_token_ids, True  # Otherwise, return prompt_token_ids & True

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
        inputs = self.action_mask_method(
            tokenizer=self.tokenizer,
            messages=messages,
            tools=tools,
            chat_template=self.chat_template,
            enable_thinking=self.enable_thinking,
        )
        token_ids = inputs["input_ids"][0]  # (seq_length, )
        action_mask = inputs["assistant_masks"][0]  # (seq_length, )
        prompt_length = action_mask.argmax().item()

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

            routed_experts = None
            if getattr(self.config, "enable_return_routed_experts", False):
                routed_experts = self._build_dummy_routed_experts()

            return Experience(
                tokens=token_ids[: self.config.max_prompt_tokens + 1],
                logprobs=torch.zeros(1, dtype=torch.float32),
                prompt_length=self.config.max_prompt_tokens,  # Use truncated length
                action_mask=torch.zeros(1, dtype=torch.bool),  # ignored in loss computation
                messages=messages,  # messages are not truncated
                truncate_status=truncate_status,
                routed_experts=routed_experts,
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
        model: Optional[ActorHandle[InferenceModel]] = None,
        models: Optional[List[ActorHandle[InferenceModel]]] = None,
        config: Optional[InferenceModelConfig] = None,
        api_address: Optional[str] = None,
    ):
        """Initialize the ModelWrapper.

        Args:
            model (InferenceModel): The inference model Ray actor.
            models (List[InferenceModel]): A list of inference model Ray actors for ensemble. The first model will be used as the main model for generation and other models will be used for auxiliary purposes such as logprobs calculation. If `model` is provided, `models` will be ignored.
            config (InferenceModelConfig): The configuration for the inference model.
            api_address (str, optional): The API address for the model. Required if `enable_openai_api` is True in the config.
        """
        if config is None:
            raise ValueError("Model config must be provided.")
        if model is None and models is None and config.engine_type != "external":
            raise ValueError("Either model or models must be provided.")
        if model is not None:
            self.model = model
            self.models = [model]
        elif models is not None and len(models) > 0:
            self.model = models[0]
            self.models = models
        else:
            self.model = None
            self.models = []
        self.config: InferenceModelConfig = config
        if self.config.model_path is None:
            raise ValueError("model_path must be provided in the config.")
        self._model_path = self.config.model_path
        self._engine_type = config.engine_type
        self._generate_kwargs = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_response_tokens,
        }
        if self.config.enable_thinking is not None:
            self._generate_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking}
            }
        self.api_address: Optional[str] = api_address
        self._api_key: str = self.config.api_key
        self.openai_client: openai.OpenAI = None
        self.openai_async_client: openai.AsyncOpenAI = None
        self.logger = get_logger(__name__)
        self.enable_lora = config.enable_lora
        self.enable_history = config.enable_history
        self.history = []
        self.status = RunningStatus.RUNNING
        self.workflow_state: Dict = {}
        self.request_count = 0
        self.state_lock = asyncio.Lock()
        self._routed_experts_layout: Optional[Tuple[int, int]] = None
        self._mm_render = None

    async def prepare(self) -> None:
        """Prepare some necessary information for the model before inference."""
        if not self.config.enable_openai_api:
            return
        if (
            self.config.enable_return_routed_experts
            and self.config.engine_type == "sglang"
            and self._routed_experts_layout is None
        ):
            self._routed_experts_layout = get_routed_experts_layout(
                self.model_path,
                trust_remote_code=self.config.trust_remote_code,
            )
        if self.api_address is None:
            if self.model is None:
                raise ValueError("Cannot get API address from the model.")
            self.api_address = await self.model.get_api_server_url.remote()
            if self.api_address is None:
                raise ValueError(
                    "Cannot get API address from the model. API server might not be enabled for this model."
                )

        if self.config.engine_type in {"tinker", "external"}:
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

    def _assert_openai_routed_experts_request_supported(
        self, extra_body: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> None:
        """Validate routed_experts constraints for OpenAI-compatible backends."""
        requested_routed_experts = self.config.enable_return_routed_experts or bool(
            extra_body.get("return_routed_experts", False)
        )
        if requested_routed_experts:
            if self.config.engine_type not in {"sglang", "vllm"}:
                raise ValueError("Routed experts can only be returned from SGLang or vLLM.")
            if kwargs.get("stream", False):
                raise ValueError("Routed experts cannot be returned for streaming requests.")
            if self.config.engine_type == "sglang" and kwargs.get("n", 1) != 1:
                raise ValueError(
                    "SGLang OpenAI API returns routed_experts at response level only; "
                    "set n=1 when requesting routed_experts."
                )

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
    def chat(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of messages."""
        lora_request = self.get_lora_request()
        return ray.get(self.model.chat.remote(messages, lora_request=lora_request, **kwargs))

    @_history_recorder
    async def chat_async(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of messages in async."""
        lora_request = await self.get_lora_request_async()
        return await self.model.chat.remote(messages, lora_request=lora_request, **kwargs)

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
        return self._model_path

    @property
    def model_name(self) -> str:
        """Get the name of the model."""
        return self.config.model_path  # type: ignore [return-value]

    @property
    def model_config(self) -> InferenceModelConfig:
        """Get the model config."""
        return self.config

    @property
    def generate_kwargs(self) -> Dict[str, Any]:
        """Get the generation kwargs for openai client."""
        return self._generate_kwargs

    async def get_available_address_async(self, random_port: bool = False) -> Tuple[str, int]:
        return await self.model.get_available_address.remote(random_port=random_port)

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

    def _get_multi_modal_inputs(
        self,
        *,
        messages: List[dict] = None,
        tools: Optional[List[dict]] = None,
        input_ids: Optional[List[int]] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        if should_use_processor(self.model_path):
            if self._mm_render is None:
                self._mm_render = vLLMMultiModalRender(  # TODO: support sglang
                    self.model_path,
                )
            return self._mm_render.build_mm_input_for_training(
                messages=messages, tools=tools, input_ids=input_ids
            )
        return None

    def get_openai_client(self) -> "openai.OpenAI":
        """Get the openai client.

        Returns:
            openai.OpenAI: The openai client. And `model_path` is added to the client which refers to the model path.
        """
        import openai

        if not self.config.enable_openai_api:
            raise ValueError(
                "OpenAI API is not enabled for this model. OpenAI client is unavailable."
            )

        if self.openai_client is not None:
            setattr(self.openai_client, "model_path", self.config.model_path)
            return self.openai_client
        if not self.api_address:
            if self.model is None:
                raise ValueError("Cannot get API address from the model.")
            self.api_address = ray.get(self.model.get_api_server_url.remote())
            if self.api_address is None:
                raise ValueError(
                    "Cannot get API address from the model. API server might not be enabled for this model."
                )
        self.openai_client = openai.OpenAI(
            base_url=f"{self.api_address}/v1",
            api_key=self._api_key,
        )
        if self._engine_type == "tinker":
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
                extra_body = dict(kwargs.pop("extra_body", {}))
                if self.config.enable_thinking is not None:
                    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
                    chat_template_kwargs["enable_thinking"] = self.config.enable_thinking
                    extra_body["chat_template_kwargs"] = chat_template_kwargs
                extra_body["return_token_ids"] = True
                if self.config.enable_return_routed_experts:
                    extra_body["return_routed_experts"] = True
                self._assert_openai_routed_experts_request_supported(extra_body, kwargs)
                response = ori_create(*args, extra_body=extra_body, logprobs=logprobs, **kwargs)
                if kwargs.get("stream", False):
                    return HistoryRecordingStream(response, self.history, is_async=False)
                messages = args[-2] if len(args) > 2 else kwargs.get("messages")
                tools = kwargs.get("tools", None)
                multi_modal_inputs = self._get_multi_modal_inputs(
                    messages=messages, tools=tools, input_ids=response.prompt_token_ids
                )
                self.history.extend(
                    convert_api_output_to_experience(
                        response,
                        multi_modal_inputs=multi_modal_inputs,
                        routed_experts_layout=self._routed_experts_layout,
                    )
                )
                return response

            self.openai_client.chat.completions.create = record_chat_completions
        setattr(self.openai_client, "model_path", self.config.model_path)
        return self.openai_client

    def get_openai_async_client(self) -> "openai.AsyncOpenAI":
        """Get the async openai client.

        Returns:
            openai.AsyncOpenAI: The async openai client. And `model_path` is added to the client which refers to the model path.
        """
        import openai

        if self.openai_async_client is not None:
            setattr(self.openai_async_client, "model_path", self.config.model_path)
            return self.openai_async_client
        if not self.api_address:
            if self.model is None:
                raise ValueError("Cannot get API address from the model.")
            self.api_address = ray.get(self.model.get_api_server_url.remote())
            if self.api_address is None:
                raise ValueError(
                    "Cannot get API address from the model. API server might not be enabled for this model."
                )
        # first make sure that we have the sync openai client
        self.openai_async_client = openai.AsyncOpenAI(
            base_url=f"{self.api_address}/v1",
            api_key=self._api_key,
        )

        if self._engine_type == "tinker":
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
                extra_body = dict(kwargs.pop("extra_body", {}))
                if self.config.enable_thinking is not None:
                    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
                    chat_template_kwargs["enable_thinking"] = self.config.enable_thinking
                    extra_body["chat_template_kwargs"] = chat_template_kwargs
                extra_body["return_token_ids"] = True
                if self.config.enable_return_routed_experts:
                    extra_body["return_routed_experts"] = True
                self._assert_openai_routed_experts_request_supported(extra_body, kwargs)
                response = await ori_create(
                    *args, extra_body=extra_body, logprobs=logprobs, **kwargs
                )
                if kwargs.get("stream", False):
                    return HistoryRecordingStream(response, self.history, is_async=True)
                messages = args[-2] if len(args) > 2 else kwargs.get("messages")
                tools = kwargs.get("tools", None)
                multi_modal_inputs = self._get_multi_modal_inputs(
                    messages=messages, tools=tools, input_ids=response.prompt_token_ids
                )
                self.history.extend(
                    convert_api_output_to_experience(
                        response,
                        multi_modal_inputs=multi_modal_inputs,
                        routed_experts_layout=self._routed_experts_layout,
                    )
                )
                return response

            self.openai_async_client.chat.completions.create = record_chat_completions
        # get model_path from the sync openai client to avoid async call here
        setattr(self.openai_async_client, "model_path", self.config.model_path)
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

    async def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        explorer_name: str,
        timeout: int = 1200,
        state_dict_meta: Optional[List] = None,
    ):
        """Initialize the process group for model weight synchronization."""

        await self.model.init_process_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            explorer_name=explorer_name,
            backend="nccl",
            timeout=timeout,
            state_dict_meta=state_dict_meta,
        )

    async def sync_model_weights(
        self, model_version: int, method: SyncMethod, timeout: int = 1200
    ) -> None:
        """Sync the model weights"""
        await self.model.sync_model_weights.remote(model_version, method, timeout=timeout)
        if self._engine_type == "tinker":
            # update the model path after syncing weights for tinker engine
            self._model_path = await self.model.get_model_path.remote()

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

    async def shutdown(self) -> None:
        """Shutdown all underlying model actors cleanly."""
        try:
            await asyncio.gather(*[model.shutdown.remote() for model in self.models])
        except Exception as e:
            self.logger.error(
                f"Error during model {self.config.model_path}[{self.config.engine_id}:{self.config.node_rank}] shutdown: {e}"
            )

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
