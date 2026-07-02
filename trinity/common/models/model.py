# -*- coding: utf-8 -*-
"""Base Model Class"""

import asyncio
import copy
import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import httpx
import ray
import torch
from ray.actor import ActorHandle
from torch import Tensor
from transformers import AutoConfig

from trinity.buffer.store import ExperienceUpdate
from trinity.common.config import InferenceModelConfig
from trinity.common.constants import RunningStatus, SyncMethod
from trinity.common.experience import Experience
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
        backend: str = "nccl",
        timeout: int = 1200,
    ):
        """Initialize the process group for model weight synchronization."""
        pass

    async def teardown_process_group(self):
        """Destroy the process group for model weight synchronization."""
        pass

    async def set_state_dict_meta(self, state_dict_meta: List):
        """Set the state_dict meta for NCCL weight sync."""
        pass

    @abstractmethod
    async def sync_model_weights(
        self,
        model_version: int,
        method: SyncMethod,
        timeout: float = 1200,
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

    def get_api_server_exit_reason(self) -> Optional[str]:
        """Return API server exit reason if the background server task has exited."""
        return None

    def get_api_key(self) -> str:
        """Get the API key."""
        return "EMPTY"

    async def extract_experience_from_history(
        self, key: str, clear_history: bool = True
    ) -> List[Experience]:
        """Extract recorded experiences by record key from the in-process store.

        Both vLLM and SGLang keep the recorder and its store in-process (the
        engine / embedded HTTP server runs in the same event loop as the model),
        so extraction is a direct store lookup with no HTTP hop. Subclasses that
        enable recording must set ``self.recorder`` (a ``Recorder`` whose
        ``.store`` is a ``RecordStore``); this base implementation is shared.
        """
        return await self._collect_experiences(
            key,
            remove=clear_history,
        )

    async def update_experience_reward(
        self,
        key: str,
        reward: float,
        info: Optional[dict] = None,
        sample_ids: Optional[List[str]] = None,
    ) -> None:
        """Update reward and optional info on recorded experiences."""
        await self.update_experience_records(
            key=key,
            update=ExperienceUpdate(reward=reward, info=info),
            sample_ids=sample_ids,
        )

    async def update_experience_records(
        self,
        key: str,
        update: ExperienceUpdate,
        sample_ids: Optional[List[str]] = None,
    ) -> None:
        """Patch recorded experiences with generation-time training signals."""
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            raise ValueError("Recording is not enabled for this model.")
        await recorder.flush()
        if not recorder.store.get(key):
            return
        recorder.store.update(
            key=key,
            update=update,
            sample_ids=sample_ids,
        )

    async def overwrite_history_experiences(self, key: str, payload: bytes) -> None:
        """Overwrite recorded experiences under one complete record key."""
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            raise ValueError("Recording is not enabled for this model.")
        await recorder.flush()
        recorder.store.overwrite(key, Experience.deserialize_many(payload))
        recorder.forget_record(key)

    async def _drain_experience_records(self, prefix: str) -> List[Experience]:
        """Remove and return recorded experiences matching a key or prefix."""
        return await self._collect_experiences(
            prefix,
            remove=True,
        )

    async def _collect_experiences(
        self,
        key: str,
        *,
        remove: bool,
    ) -> List[Experience]:
        """Collect recorded experiences by exact key or store-supported prefix."""
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            raise ValueError("Recording is not enabled for this model.")
        await recorder.flush()
        if remove:
            exps = recorder.store.remove(key)
            recorder.forget_record(key)
            return exps
        return recorder.store.get(key)

    async def drain_experience_records_bytes(self, prefix: str) -> bytes:
        """Remove matching recorded experiences and return serialized bytes."""
        return Experience.serialize_many(await self._drain_experience_records(prefix))

    async def delete_experience_records(self, prefix: str) -> None:
        """Remove recorded experiences matching a key or prefix."""
        await self._drain_experience_records(prefix)

    async def block_experience_records(self, prefix: str) -> None:
        """Block future writes for the given batch prefix on this rollout rank.

        Sets the block flag before flushing the recorder so that any in-flight
        experiences still queued in the recorder are dropped by ``MemoryStore``
        rather than written back as orphans. ``prefix`` is the batch segment
        of the store key (``str(batch_id)``).
        """
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            return
        recorder.store.block_prefix(prefix)
        await recorder.flush()

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
        # TODO(recording): when the in-vLLM recorder is active, this is
        # redundant — it re-tokenizes messages and runs an extra logprobs
        # forward (and fakes routed_experts), all of which build_experience
        # already captured at generation time into the MemoryStore. Redirect to
        # a store lookup by the call's record_key once it's threaded here.
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
        self.status = RunningStatus.RUNNING
        self.request_count = 0

    async def prepare(self) -> None:
        """Prepare some necessary information for the model before inference."""
        # The OpenAI API server is always enabled for vLLM/SGLang models; only the
        # Tinker and external backends skip the HTTP probe — Tinker has no real
        # API server (its OpenAI client is a Ray-remote shim), and external's
        # address comes from the environment. This short-circuit is intentionally
        # based on engine type, not on the deprecated ``enable_openai_api`` flag.
        if self.config.engine_type in {"tinker", "external"}:
            return
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
            reason = await self.model.get_api_server_exit_reason.remote()
            if reason is not None:
                raise RuntimeError(
                    f"API server at {self.api_address} exited before becoming ready: {reason}."
                )
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

    def generate(
        self, prompts: List[str], enable_recording: bool = False, **kwargs
    ) -> List[Experience]:
        """Generate a list of experiences from a list of prompts."""
        lora_request = self.get_lora_request()
        if self.config.enable_history and enable_recording:
            kwargs["key"] = self._api_key
        results = ray.get(
            [self.model.generate.remote(prompt, lora_request, **kwargs) for prompt in prompts]
        )
        return [exp for exps in results for exp in exps]

    async def generate_async(
        self, prompts: List[str], enable_recording: bool = False, **kwargs
    ) -> List[Experience]:
        """Generate a list of experiences from a list of prompts in async."""
        lora_request = await self.get_lora_request_async()
        if self.config.enable_history and enable_recording:
            kwargs["key"] = self._api_key
        results = await asyncio.gather(
            *[self.model.generate.remote(prompt, lora_request, **kwargs) for prompt in prompts]
        )
        return [exp for exps in results for exp in exps]

    def chat(
        self, messages: List[dict], enable_recording: bool = False, **kwargs
    ) -> List[Experience]:
        """Generate a list of experiences from a list of messages."""
        lora_request = self.get_lora_request()
        if self.config.enable_history and enable_recording:
            kwargs["key"] = self._api_key
        return ray.get(self.model.chat.remote(messages, lora_request=lora_request, **kwargs))

    async def chat_async(
        self, messages: List[dict], enable_recording: bool = False, **kwargs
    ) -> List[Experience]:
        """Generate a list of experiences from a list of messages in async."""
        lora_request = await self.get_lora_request_async()
        if self.config.enable_history and enable_recording:
            kwargs["key"] = self._api_key
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
    def base_url(self) -> str:
        """Get the base URL of the API server."""
        if not self.api_address:
            raise ValueError("API address is not set. Cannot get base URL.")
        return f"{self.api_address}/v1"

    @property
    def api_key(self) -> str:
        """Get the API key."""
        return self._api_key

    def set_api_key(self, api_key: str) -> None:
        """Set the API key used by existing and future OpenAI clients."""
        self._api_key = api_key
        if self.openai_client is not None:
            self.openai_client.api_key = api_key
        if self.openai_async_client is not None:
            self.openai_async_client.api_key = api_key

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

    def get_openai_client(self) -> "openai.OpenAI":
        """Get the openai client.

        Returns:
            openai.OpenAI: The openai client. And `model_path` is added to the client which refers to the model path.
        """
        import openai

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
                        record_key=(self._api_key if self.enable_history else None),
                        **kwargs,
                    )
                )
                response = chat_response.pop()
                return response

            self.openai_client.chat.completions.create = chat_completions
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
                    record_key=(self._api_key if self.enable_history else None),
                    **kwargs,
                )
                response = chat_response.pop()
                return response

            self.openai_async_client.chat.completions.create = chat_completions
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
        timeout: int = 1200,
    ):
        """Initialize the process group for model weight synchronization."""

        await self.model.init_process_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend="nccl",
            timeout=timeout,
        )

    async def teardown_process_group(self):
        """Destroy the process group for model weight synchronization."""
        await self.model.teardown_process_group.remote()

    async def set_state_dict_meta(self, state_dict_meta: List):
        """Set the state_dict meta for NCCL weight sync."""
        await self.model.set_state_dict_meta.remote(state_dict_meta)

    async def sync_model_weights(
        self,
        model_version: int,
        method: SyncMethod,
        timeout: float = 1200,
    ) -> None:
        """Sync the model weights"""
        await self.model.sync_model_weights.remote(model_version, method, timeout=timeout)
        if self._engine_type == "tinker":
            # update the model path after syncing weights for tinker engine
            self._model_path = await self.model.get_model_path.remote()

    def extract_experience_from_history(
        self, clear_history: bool = True, key: Optional[str] = None
    ) -> List[Experience]:
        """Extract experiences from the history."""
        if not self.enable_history:
            raise ValueError("History recording is not enabled.")
        if self.model is None:
            raise ValueError("Recording extraction requires an inference model actor.")
        key = key or self._api_key
        if key is None:
            raise ValueError("key is required when recording is enabled.")
        exps = ray.get(
            self.model.extract_experience_from_history.remote(
                key=key,
                clear_history=clear_history,
            )
        )
        return exps

    async def update_experience_reward_async(
        self,
        key: str,
        reward: float,
        info: Optional[dict] = None,
        sample_ids: Optional[List[str]] = None,
    ) -> None:
        """Update reward and optional info on recorded experiences."""
        await self.update_experience_records_async(
            key=key,
            update=ExperienceUpdate(reward=reward, info=info),
            sample_ids=sample_ids,
        )

    async def update_experience_records_async(
        self,
        key: str,
        update: ExperienceUpdate,
        sample_ids: Optional[List[str]] = None,
    ) -> None:
        """Patch recorded experiences with generation-time training signals."""
        if not self.enable_history:
            raise ValueError("History recording is not enabled.")
        if self.model is None:
            raise ValueError("Recording update requires an inference model actor.")
        await self.model.update_experience_records.remote(
            key=key,
            update=update,
            sample_ids=sample_ids,
        )

    def update_experience_reward(
        self,
        key: str,
        reward: float,
        info: Optional[dict] = None,
        sample_ids: Optional[List[str]] = None,
    ) -> None:
        """Update reward and optional info on recorded experiences."""
        self.update_experience_records(
            key=key,
            update=ExperienceUpdate(reward=reward, info=info),
            sample_ids=sample_ids,
        )

    def update_experience_records(
        self,
        key: str,
        update: ExperienceUpdate,
        sample_ids: Optional[List[str]] = None,
    ) -> None:
        """Patch recorded experiences with generation-time training signals."""
        if not self.enable_history:
            raise ValueError("History recording is not enabled.")
        if self.model is None:
            raise ValueError("Recording update requires an inference model actor.")
        ray.get(
            self.model.update_experience_records.remote(
                key=key,
                update=update,
                sample_ids=sample_ids,
            )
        )

    async def overwrite_history_experiences_async(
        self, experiences: List[Experience], key: str
    ) -> None:
        """Overwrite recorded experiences under one complete record key."""
        if not self.enable_history:
            raise ValueError("History recording is not enabled.")
        if self.model is None:
            raise ValueError("Recording overwrite requires an inference model actor.")
        await self.model.overwrite_history_experiences.remote(
            key=key,
            payload=Experience.serialize_many(experiences),
        )

    async def drain_experience_records_bytes_async(self, prefix: str) -> bytes:
        """Remove matching recorded experiences and return serialized bytes."""
        if not self.enable_history:
            raise ValueError("History recording is not enabled.")
        if self.model is None:
            raise ValueError("Recording drain requires an inference model actor.")
        return await self.model.drain_experience_records_bytes.remote(prefix=prefix)

    async def delete_experience_records_async(self, prefix: str) -> None:
        """Remove recorded experiences matching a key or prefix."""
        if not self.enable_history:
            raise ValueError("History recording is not enabled.")
        if self.model is None:
            raise ValueError("Recording delete requires an inference model actor.")
        await self.model.delete_experience_records.remote(prefix=prefix)

    async def block_experience_records_async(self, prefix: str) -> None:
        """Block future writes for the given batch prefix on the rollout actor."""
        if not self.enable_history:
            raise ValueError("History recording is not enabled.")
        if self.model is None:
            raise ValueError("Recording block requires an inference model actor.")
        await self.model.block_experience_records.remote(prefix=prefix)

    async def shutdown(self) -> None:
        """Shutdown all underlying model actors cleanly."""
        try:
            await asyncio.gather(*[model.shutdown.remote() for model in self.models])
        except Exception as e:
            self.logger.error(
                f"Error during model {self.config.model_path}[{self.config.engine_id}:{self.config.node_rank}] shutdown: {e}"
            )

    def clone_with_isolated_state(self) -> "ModelWrapper":
        """Clone the current ModelWrapper with isolated state."""
        new_wrapper = copy.copy(self)
        new_wrapper.openai_async_client = None
        new_wrapper.openai_client = None
        return new_wrapper
