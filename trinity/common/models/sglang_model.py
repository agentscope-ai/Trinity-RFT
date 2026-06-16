from __future__ import annotations

import asyncio
import operator
import os
import traceback
from functools import reduce
from logging import Logger
from typing import Any, List, Literal, Optional, Sequence, Tuple

import httpx
import torch
from transformers import AutoTokenizer

from trinity.common.config import InferenceModelConfig
from trinity.common.constants import ROLLOUT_WEIGHT_SYNC_GROUP_NAME, SyncMethod
from trinity.common.experience import Experience
from trinity.common.models.experience_extraction import decode_sglang_routed_experts
from trinity.common.models.model import BaseInferenceModel
from trinity.manager.synchronizer import Synchronizer


class SGLangClient:
    """A simple http client to interact with the SGLang API server."""

    def __init__(self, server_url: str, api_key: Optional[str], logger: Logger):
        self.server_url = server_url
        self.api_key = api_key
        self.logger = logger

    async def _server_call(
        self,
        method: Literal["GET", "POST"],
        endpoint: str,
        payload: Optional[dict] = None,
        timeout: float = 60,
    ) -> dict:
        async with httpx.AsyncClient(
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            }
        ) as client:
            url = f"{self.server_url}{endpoint}"
            self.logger.debug(
                f"Making {method} request to SGLang API server at {url} with payload: {payload}"
            )
            try:
                if method == "GET":
                    response = await client.get(url, timeout=timeout)
                elif method == "POST":
                    response = await client.post(url, json=payload or {}, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                response.raise_for_status()
                return response.json()
            except Exception:
                self.logger.debug(
                    f"Error during {method} request to SGLang API server at {url}:\n{traceback.format_exc()}"
                )
                return {"error": traceback.format_exc()}

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                }
            ) as client:
                response = await client.get(f"{self.server_url}/health", timeout=5)
                return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"SGLang API server health check failed: {e}")
            return False

    async def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
    ) -> bool:
        payload = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": rank_offset,
            "world_size": world_size,
            "group_name": group_name,
            "backend": backend,
        }
        response = await self._server_call(
            "POST", "/init_weights_update_group", payload, timeout=timeout
        )
        success = response.get("success", False)
        if not success:
            self.logger.error(
                f"Failed to initialize weights update group in SGLang API server: {response.get('message')}"
            )
        return success

    async def destroy_weights_update_group(self, group_name: str) -> bool:
        payload = {"group_name": group_name}
        response = await self._server_call(
            "POST", "/destroy_weights_update_group", payload, timeout=5
        )
        success = response.get("success", False)
        if not success:
            self.logger.error(
                f"Failed to destroy weights update group in SGLang API server: {response.get('message')}"
            )
        return success

    async def flush_cache(self, timeout: float = 30) -> bool:
        """Flush KV and Mamba cache to free GPU memory before weight sync."""
        import httpx

        url = f"{self.server_url}/flush_cache"
        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.api_key}" if self.api_key else ""}
            ) as client:
                response = await client.post(url, timeout=timeout)
                if response.status_code == 200:
                    return True
                self.logger.warning(
                    f"flush_cache returned status {response.status_code}: {response.text}"
                )
                return False
        except Exception as e:
            self.logger.warning(f"Failed to flush cache: {e}")
            return False

    async def update_weights_from_distributed(
        self,
        state_dict_meta_list: List[Tuple[str, str, Tuple]],
        group_name: str,
        flush_cache: bool = True,
        abort_all_requests: bool = True,
        weight_version: Optional[str] = None,
        timeout: float = 300,
    ) -> bool:
        names = [meta[0] for meta in state_dict_meta_list]
        dtypes = [meta[1] for meta in state_dict_meta_list]
        shapes = [meta[2] for meta in state_dict_meta_list]
        payload = {
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
            "abort_all_requests": abort_all_requests,
            "weight_version": weight_version,
            "torch_empty_cache": True,
        }
        response = await self._server_call(
            "POST", "/update_weights_from_distributed", payload, timeout=timeout
        )
        success = response.get("success", False)
        if not success:
            self.logger.error(
                f"Failed to update weights from distributed in SGLang API server: {response.get('message')}"
            )
        return success

    async def update_weights_from_disk(
        self,
        model_path: str,
        abort_all_requests: bool = True,
        weight_version: Optional[str] = None,
        is_async: bool = False,
        timeout: float = 300,
    ) -> bool:
        payload = {
            "model_path": model_path,
            "abort_all_requests": abort_all_requests,
            "weight_version": weight_version,
            "is_async": is_async,
            "torch_empty_cache": True,
        }
        response = await self._server_call(
            "POST", "/update_weights_from_disk", payload, timeout=timeout
        )
        success = response.get("success", False)
        if not success:
            self.logger.error(
                f"Failed to update weights from disk in SGLang API server: {response.get('message')}"
            )
        return success

    async def generate(self, input_ids: List[int], **kwargs) -> Sequence[dict[str, Any]]:
        sampling_params = {
            "n": kwargs.get("n", 1),
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "max_new_tokens": kwargs.get("max_tokens"),
            "min_new_tokens": kwargs.get("min_tokens"),
            "repetition_penalty": kwargs.get("repetition_penalty"),
            "stop": kwargs.get("stop"),
            "ignore_eos": kwargs.get("ignore_eos"),
        }
        sampling_params = {k: v for k, v in sampling_params.items() if v is not None}

        payload: dict[str, Any] = {
            "sampling_params": sampling_params,
            "return_logprob": kwargs.get("return_logprob", False),
            "return_routed_experts": kwargs.get("return_routed_experts", False),
            "top_logprobs_num": kwargs.get("top_logprobs_num", 0),
            "return_text_in_logprobs": False,
            "input_ids": input_ids,
        }

        response = await self._server_call(
            "POST",
            "/generate",
            payload,
            timeout=kwargs.get("timeout", 300),
        )
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(f"Failed to generate with SGLang: {response['error']}")
        if isinstance(response, dict):
            return [response]
        if isinstance(response, list):
            return response
        raise TypeError(f"Unexpected SGLang generate response type: {type(response)!r}")


class SGLangRolloutModel(BaseInferenceModel):
    """Wrapper around the SGLang engine to handle async requests.

    Args:
        config (Config): The config.
    """

    def __init__(
        self,
        config: InferenceModelConfig,
    ) -> None:
        super().__init__(config)
        if config.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        if not self.config.enable_openai_api:
            self.logger.warning("SGLangRolloutModel requires OpenAI API to be enabled.")
            self.config.enable_openai_api = True
        os.environ["SGLANG_GRPC_PORT"] = "12345"  # a dummy port not actually used
        os.environ["SGLANG_ENABLE_GRPC"] = "0"
        self.api_server_host: Optional[str] = None
        self.api_server_port: Optional[int] = None
        self.api_server: Optional[asyncio.Task[None]] = None
        self.api_client: Optional[SGLangClient] = None
        self.synchronizer = None
        self.state_dict_meta: List[Tuple[str, str, Tuple]] = []
        self.model_version = 0
        self._prepared = False
        self._has_weight_update_group = False
        self.async_lock = asyncio.Lock()
        self.group_name = ROLLOUT_WEIGHT_SYNC_GROUP_NAME

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
        if self.config.node_rank != 0:
            self.logger.warning(
                "init_process_group should only be called on the main node (node_rank=0). "
                f"Current node_rank={self.config.node_rank}, skipping initialization and returning."
            )
            return
        if self.api_client is None:
            raise RuntimeError("API client must be initialized before calling init_process_group")
        if not self.synchronizer:
            self.synchronizer = Synchronizer.get_actor(namespace=self.config.ray_namespace)
        self.logger.info(
            "SGLang starting init_process_group:\n"
            f"  > address={master_address}:{master_port}\n"
            f"  > rank_offset={rank_offset}\n"
            f"  > world_size={world_size}\n"
            f"  > group_name={group_name}\n"
        )
        self.group_name = group_name
        resp = await self.api_client.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
            timeout=timeout,
        )
        self.logger.info("SGLang init_process_group finished.")
        self._has_weight_update_group = resp
        return resp

    async def set_state_dict_meta(self, state_dict_meta: List[Tuple[str, str, Tuple]]):
        """Set the state_dict meta for NCCL weight sync."""
        self.state_dict_meta = state_dict_meta or []

    async def teardown_process_group(self):
        """Destroy the weight update group via the SGLang API.

        Only the main node (node_rank=0) issues the API call; other nodes
        just clear local state.
        """
        if (
            self.config.node_rank == 0
            and self._has_weight_update_group
            and self.api_client is not None
        ):
            await self.api_client.destroy_weights_update_group(group_name=self.group_name)
        self._has_weight_update_group = False
        self.state_dict_meta = []

    async def _initialize_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.tokenizer.truncation_side = "left"

    async def prepare(self) -> None:
        async with self.async_lock:
            if self._prepared:
                return
            if self.config.enable_return_routed_experts:
                self._get_routed_experts_layout()
            await self.run_api_server()
            self._prepared = True

    @staticmethod
    def _extract_output_logprobs(meta_info: dict[str, Any]) -> List[float]:
        output_token_logprobs = meta_info.get("output_token_logprobs") or []
        return [float(logprob) for logprob, *_ in output_token_logprobs]

    def _normalize_chat_messages(self, messages: List[dict]) -> List[dict]:
        normalized_messages = []
        for message in messages:
            normalized_message = dict(message)
            content = normalized_message.get("content")
            if isinstance(content, list):
                text_parts = [item["text"] for item in content if item.get("type") == "text"]
                normalized_message["content"] = "".join(text_parts)
            normalized_messages.append(normalized_message)
        return normalized_messages

    def _extract_routed_experts(self, routed_experts_str: str, total_tokens: int) -> torch.Tensor:
        # decode only needs (num_layers, topk); num_experts is used by dummy experiences.
        num_layers, topk, _ = self._get_routed_experts_layout()
        routed_experts = decode_sglang_routed_experts(
            routed_experts_str,
            total_tokens,
            layout=(num_layers, topk),
        )
        assert routed_experts is not None
        return routed_experts

    async def generate(self, prompt: str, lora_request=None, **kwargs) -> Sequence[Experience]:
        assert self.api_client is not None, "API client must be initialized before calling generate"
        if self.tokenizer is None:
            await self._initialize_tokenizer()

        returned_seq, is_valid = self._handle_prompt_truncation(prompt, **kwargs)
        if not is_valid:
            return returned_seq
        prompt_token_ids = list(returned_seq)

        logprobs = kwargs.get("logprobs", self.config.logprobs)
        return_logprob = logprobs is not None and logprobs is not False
        responses = await self.api_client.generate(
            input_ids=prompt_token_ids,
            n=kwargs.get("n", 1),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            max_tokens=kwargs.get("max_tokens", self.config.max_response_tokens),
            min_tokens=kwargs.get("min_tokens", self.config.min_response_tokens),
            repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
            stop=kwargs.get("stop"),
            ignore_eos=kwargs.get("ignore_eos", self.config.ignore_eos),
            return_logprob=return_logprob,
            return_routed_experts=self.config.enable_return_routed_experts,
            timeout=kwargs.get("timeout", 300),
        )

        prompt_text = self.tokenizer.decode(prompt_token_ids)
        experiences = []
        for response in responses:
            response_token_ids = response.get("output_ids") or []
            response_text = response.get("text") or ""
            if not response_token_ids and response_text:
                response_token_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

            meta_info = response.get("meta_info") or {}
            prompt_length = int(meta_info.get("prompt_tokens") or len(prompt_token_ids))
            if return_logprob:
                response_logprobs = torch.tensor(
                    self._extract_output_logprobs(meta_info),
                    dtype=torch.float32,
                )
            else:
                response_logprobs = torch.tensor([], dtype=torch.float32)

            routed_experts = None
            routed_experts_value = meta_info.get("routed_experts", None)
            if self.config.enable_return_routed_experts and routed_experts_value is not None:
                if isinstance(routed_experts_value, str):
                    routed_experts = self._extract_routed_experts(
                        routed_experts_value,
                        total_tokens=len(prompt_token_ids) + len(response_token_ids),
                    )
                else:
                    routed_experts = torch.tensor(routed_experts_value, dtype=torch.uint8)

            experiences.append(
                Experience(
                    tokens=torch.tensor(prompt_token_ids + response_token_ids, dtype=torch.int32),
                    logprobs=response_logprobs,
                    prompt_length=prompt_length,
                    prompt_text=prompt_text,
                    response_text=response_text,
                    routed_experts=routed_experts,
                )
            )
        return experiences

    async def chat(self, messages: List[dict], lora_request=None, **kwargs) -> Sequence[Experience]:
        if self.tokenizer is None:
            await self._initialize_tokenizer()

        normalized_messages = self._normalize_chat_messages(messages)
        prompt = self.apply_chat_template(self.tokenizer, normalized_messages)
        return await self.generate(prompt=prompt, lora_request=lora_request, **kwargs)

    async def logprobs(self, token_ids: List[int], **kwargs) -> torch.Tensor:
        raise NotImplementedError("SGLangRolloutModel does not support logprobs.")

    async def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools=None,
        temperature: Optional[float] = None,
    ) -> Experience:
        raise NotImplementedError(
            "SGLangRolloutModel does not support convert_messages_to_experience."
        )

    def _get_api_server_exit_reason(self) -> Optional[str]:
        if self.api_server is None or not self.api_server.done():
            return None
        if self.api_server.cancelled():
            return "cancelled"
        exc = self.api_server.exception()
        return "unknown error" if exc is None else repr(exc)

    async def _wait_until_server_ready(self, server_url: str) -> None:
        max_retries = 100
        assert self.api_client is not None
        for _ in range(max_retries):
            reason = self._get_api_server_exit_reason()
            if reason is not None:
                raise RuntimeError(f"SGLang API server exited before becoming ready: {reason}.")
            if await self.api_client.health_check():
                self.logger.info(f"SGLang API server at {server_url} is ready.")
                return
            self.logger.debug(f"SGLang API server at {server_url} not ready yet, retrying...")
            await asyncio.sleep(2)
        self.logger.error(
            f"SGLang API server at {server_url} not ready after {max_retries} attempts."
        )
        await self.shutdown()
        raise RuntimeError(
            f"SGLang API server at {server_url} not ready after {max_retries} attempts."
        )

    async def run_api_server(self) -> bool:
        from trinity.common.models.sglang_patch import get_api_server

        if self.api_server_host is None or self.api_server_port is None:
            self.api_server_host, self.api_server_port = self.get_available_address()
        self.api_server = get_api_server(
            host=self.api_server_host,
            port=self.api_server_port,
            model_path=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            data_parallel_size=self.config.data_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            enable_expert_parallel=self.config.enable_expert_parallel,
            extra_engine_args=self.config.extra_engine_args,
            dtype=self.config.dtype,
            served_model_name=self.config.name or self.config.model_path,
            mem_fraction_static=self.config.gpu_memory_utilization,
            trust_remote_code=self.config.trust_remote_code,
            context_length=self.config.max_model_len,
            enable_multimodal=self.config.enable_multimodal,
            api_key=self.config.api_key,
            nnodes=self.config.nnodes,
            node_rank=self.config.node_rank,
            master_addr=self.master_addr,
            master_port=self.master_port,
            enable_return_routed_experts=self.config.enable_return_routed_experts,
            logger=self.logger,
        )
        server_url = f"http://{self.api_server_host}:{self.api_server_port}"
        self.api_client = SGLangClient(
            server_url=server_url,
            api_key=self.config.api_key,
            logger=self.logger,
        )
        await self._wait_until_server_ready(server_url)
        return True

    async def shutdown(self) -> None:
        if self.api_server is not None:
            try:
                if (
                    self.config.node_rank == 0
                    and self._has_weight_update_group
                    and self.api_client is not None
                ):
                    await self.api_client.destroy_weights_update_group(group_name=self.group_name)
                self.api_server.cancel()
                await self.api_server
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.error("Error while shutting down SGLang API server: %s", e)
            reason = self._get_api_server_exit_reason()
            if reason not in {None, "cancelled"}:
                self.logger.warning("Embedded SGLang HTTP server exited with error: %s", reason)
            self.api_server = None
            self.api_client = None
            self._has_weight_update_group = False

    async def sync_model_weights(
        self,
        model_version: int,
        method: SyncMethod,
        timeout: float = 1200,
    ) -> int:
        if self.config.node_rank != 0:
            self.logger.warning(
                "sync_model_weights should only be called on the main node (node_rank=0). "
                f"Current node_rank={self.config.node_rank}, skipping sync and returning version {model_version}."
            )
            return model_version
        assert (
            self.api_client is not None
        ), "API client must be initialized before calling sync_model_weights"
        assert (
            self.synchronizer is not None
        ), "Synchronizer must be initialized before calling sync_model_weights"
        self.logger.info(f"Synchronizing model to version {model_version} using method {method}...")
        if method == SyncMethod.NCCL:
            assert self.state_dict_meta, "state_dict_meta must be initialized for NCCL sync"
            # Flush KV + Mamba cache to free GPU memory for receive buffers
            await self.api_client.flush_cache()
            self.logger.info("Flushed KV/Mamba cache before NCCL weight sync")
            batches = self._partition_state_dict_meta(self.state_dict_meta)
            self.logger.info(
                f"NCCL weight sync: {len(self.state_dict_meta)} tensors in {len(batches)} batches "
                f"(buffer_size={self.config.weight_sync_buffer_size} MB)"
            )
            for i, batch in enumerate(batches):
                is_last = i == len(batches) - 1
                await self.api_client.update_weights_from_distributed(
                    state_dict_meta_list=batch,
                    group_name=self.group_name,
                    weight_version=str(model_version),
                    flush_cache=is_last,
                    timeout=timeout,
                )
            self.model_version = model_version
        elif method == SyncMethod.CHECKPOINT:
            # TODO: this branch is buggy, which only supports hf format checkpoints
            model_path = await self.synchronizer.get_latest_model_path.remote(use_huggingface=True)
            if model_path is not None:
                await self.api_client.update_weights_from_disk(
                    model_path=model_path,
                    weight_version=str(model_version),
                    timeout=timeout,
                )
                self.model_version = model_version
        else:
            raise ValueError(f"Unsupported sync method for SGLang: {method}")
        self.logger.info("Synchronization finished.")
        return model_version

    def _partition_state_dict_meta(
        self, meta_list: "List[Tuple[str, str, Tuple]]"
    ) -> "List[List[Tuple[str, str, Tuple]]]":
        """Partition state_dict_meta into batches that fit within weight_sync_buffer_size.

        This prevents OOM during NCCL weight sync by ensuring SGLang only allocates
        receive buffers for one batch at a time, rather than all tensors simultaneously.
        """
        buffer_size = self.config.weight_sync_buffer_size * 1024 * 1024  # convert MB to bytes

        batches: list = []
        current_batch: list = []
        current_size = 0

        for item in meta_list:
            name, dtype_str, shape = item
            dtype = getattr(torch, dtype_str, torch.bfloat16)
            elem_size = torch.tensor([], dtype=dtype).element_size()
            tensor_bytes = reduce(operator.mul, shape, 1) * elem_size

            if current_size + tensor_bytes > buffer_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0

            current_batch.append(item)
            current_size += tensor_bytes

        if current_batch:
            batches.append(current_batch)

        return batches

    def get_model_version(self) -> int:
        return self.model_version

    def get_api_server_url(self) -> Optional[str]:
        if not self._prepared:
            raise RuntimeError("Model is not prepared. Please call `prepare()` first.")
        return f"http://{self.api_server_host}:{self.api_server_port}"
