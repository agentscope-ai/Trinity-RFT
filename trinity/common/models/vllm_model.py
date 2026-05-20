"""A wrapper around the vllm.AsyncEngine to handle async requests."""

import asyncio
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from packaging.version import parse as parse_version
from transformers import AutoProcessor

from trinity.common.config import InferenceModelConfig
from trinity.common.constants import SyncMethod
from trinity.common.experience import Experience
from trinity.common.models.mm_utils import (
    build_mm_input_for_training,
    build_multi_modal_data,
    has_multi_modal_content,
)
from trinity.common.models.model import BaseInferenceModel
from trinity.common.models.vllm_patch import get_vllm_version


# V0 engine is deprecated since vLLM v0.10.2, related code will be removed in the future.
class vLLMRolloutModel(BaseInferenceModel):
    """Wrapper around the vLLM engine to handle async requests.

    Args:
        config (Config): The config.
    """

    def __init__(
        self,
        config: InferenceModelConfig,
    ) -> None:
        super().__init__(config)
        if config.cuda_visible_devices:
            # only for colocate mode
            os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        import vllm
        from vllm.sampling_params import RequestOutputKind

        self.vllm_version = get_vllm_version()
        self.use_v1 = config.use_v1
        if config.tensor_parallel_size != 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = config.bundle_indices
        if self.vllm_version <= parse_version("0.11.0") and not vllm.envs.is_set("VLLM_USE_V1"):
            self.logger.info(f"Using vLLM v{int(config.use_v1)} engine")
            os.environ["VLLM_USE_V1"] = str(int(config.use_v1))
        if config.use_v1:
            os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(int(config.use_v1))
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        if self.vllm_version >= parse_version("0.11.0"):
            os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        if self.config.enable_runtime_lora_updating:
            os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        self.tokenization_kwargs = {
            "truncate_prompt_tokens": config.max_prompt_tokens
            if config.enable_prompt_truncation
            else None
        }
        self.default_sampling_params = vllm.SamplingParams(
            n=1,
            temperature=config.temperature,
            max_tokens=config.max_response_tokens,
            min_tokens=config.min_response_tokens,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            logprobs=config.logprobs,
            top_p=config.top_p,
            top_k=config.top_k,
            ignore_eos=config.ignore_eos,
            **(self.tokenization_kwargs if self.vllm_version <= parse_version("0.16.0") else {}),
        )
        self.ray_namespace = config.ray_namespace
        self.request_id = 0
        self.enable_lora = config.enable_lora
        self.default_lora_path = config.lora_kwargs.pop("default_lora_path", None)
        self.logprobs_no_prefix_cache = True
        self.processor = None
        self.state_dict_meta = None
        self.model_version = 0  # TODO: resume the value from the checkpoint
        self.api_server_host = None
        self.api_server_port = None
        self.api_server = None
        self._prepared = False
        self.async_llm = None
        self.headless_executor = None
        self.async_lock = asyncio.Lock()

    async def _initialize_tokenizer(self):
        if self.tokenizer is None:
            if self.vllm_version >= parse_version("0.15.0"):
                self.tokenizer = self.async_llm.get_tokenizer()
            else:
                self.tokenizer = await self.async_llm.get_tokenizer()
        self.tokenizer.truncation_side = "left"

    async def _initialize_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        await self._initialize_tokenizer()

    async def prepare(self) -> None:
        """Prepare the model for inference."""
        import vllm

        async with self.async_lock:
            if self._prepared:
                return
            rope_params = defaultdict(dict)
            if self.config.rope_scaling is not None:
                rope_params["rope_parameters"] = self.config.rope_scaling
            if self.config.rope_theta is not None:
                rope_params["rope_parameters"]["rope_theta"] = self.config.rope_theta
            if len(rope_params) > 0:
                rope_kwargs = {"hf_overrides": rope_params}
            else:
                rope_kwargs = {}
            engine_args = vllm.AsyncEngineArgs(
                model=self.config.model_path,
                enforce_eager=self.config.enforce_eager,
                worker_extension_cls="trinity.common.models.vllm_worker.WorkerExtension",
                tensor_parallel_size=self.config.tensor_parallel_size,
                seed=self.config.seed,
                distributed_executor_backend="mp",
                max_model_len=self.config.max_model_len,
                enable_prefix_caching=self.config.enable_prefix_caching,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
                dtype=self.config.dtype,
                trust_remote_code=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                override_generation_config={  # TODO: find a way to unittest this
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "max_new_tokens": self.config.max_response_tokens,
                    "repetition_penalty": self.config.repetition_penalty,
                },
                reasoning_parser=self.config.reasoning_parser,
                disable_log_stats=True,
                enable_log_requests=self.config.enable_log_requests,
                enable_lora=self.config.enable_lora,
                logprobs_mode="processed_logprobs",
                nnodes=self.config.nnodes,
                node_rank=self.config.node_rank,
                async_scheduling=False,
                **rope_kwargs,
                **self.config.lora_kwargs,
            )
            if self.master_addr is not None and self.master_port is not None:
                engine_args.master_addr = self.master_addr
                engine_args.master_port = self.master_port
            if self.config.node_rank == 0:
                self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
                await self._collective_rpc("apply_patches")
                await self.run_api_server()
            else:
                # use run_headless
                from vllm.v1.executor.multiproc_executor import MultiprocExecutor

                vllm_config = engine_args.create_engine_config(headless=True)
                self.headless_executor = MultiprocExecutor(vllm_config, monitor_workers=False)
                self.headless_executor.start_worker_monitor()
            self._prepared = True

    async def chat(self, messages: List[Dict], lora_request=None, **kwargs) -> Sequence[Experience]:
        """Chat with the model with a list of messages in async.

        Args:
            messages (List[dict]): The input history messages.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        is_mm_message = has_multi_modal_content(messages)
        if is_mm_message:
            if self.processor is None:
                await self._initialize_processor()
            tokenizer_or_processor = self.processor
        else:
            if self.tokenizer is None:
                await self._initialize_tokenizer()
            tokenizer_or_processor = self.tokenizer

        prompt = self.apply_chat_template(tokenizer_or_processor, messages)
        if is_mm_message:
            multi_modal_data = build_multi_modal_data(self.processor, messages)
            prompt = {
                "prompt": prompt,
                "multi_modal_data": multi_modal_data,
            }
        return await self.generate(prompt=prompt, lora_request=lora_request, **kwargs)

    async def generate(
        self, prompt: Union[str, Dict], lora_request=None, **kwargs
    ) -> Sequence[Experience]:
        """Generate a response from the provided prompt in async.

        Args:
            prompt (str): The input prompt.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        if isinstance(prompt, str):  # pure text
            if self.tokenizer is None:
                await self._initialize_tokenizer()

            returned_seq, is_valid = self._handle_prompt_truncation(prompt, **kwargs)
            if not is_valid:
                return (
                    returned_seq  # is_valid is False: returned_seq is a list of dummy experiences
                )
            prompt = {
                "prompt_token_ids": returned_seq
            }  # is_valid is True: returned_seq is token_ids
            multi_modal_inputs = None
        else:  # multi modal
            multi_modal_inputs = build_mm_input_for_training(self.processor, **prompt)
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)

        output = await self._generate_internal(prompt=prompt, lora_request=lora_request, **kwargs)
        experiences = [
            Experience(
                tokens=torch.cat(
                    (
                        torch.tensor(output.prompt_token_ids, dtype=torch.int32),
                        torch.tensor(output.outputs[i].token_ids, dtype=torch.int32),
                    )
                ),
                logprobs=torch.cat(
                    (
                        torch.tensor(
                            [
                                list(logprob_dict.values())[0].logprob
                                for logprob_dict in output.outputs[i].logprobs
                            ],
                            dtype=torch.float32,
                        ),
                    )
                ),
                prompt_length=len(output.prompt_token_ids),
                prompt_text=self.tokenizer.decode(output.prompt_token_ids),
                response_text=output.outputs[i].text,
                multi_modal_inputs=multi_modal_inputs,
            )
            for i in range(len(output.outputs))
        ]
        return experiences

    async def logprobs(  # type: ignore [override]
        self,
        token_ids: List[int],
        lora_request=None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Calculate the logprobs of the given tokens in async. Please slice the result carefully
        to align with the actual response length.

        Args:
            token_ids (List[int]): The input token ids (seq_length). Please make sure the length of
                it does not exceed `max_model_len - 1`.
            lora_request (LoRARequest, optional): The LoRA request. Defaults to None.
            temperature (float): The temperature for scaling logits.

        Returns:
            A tensor of logprobs (seq_length - 1).
        """
        temperature = temperature if temperature is not None else self.config.temperature
        if temperature is None:
            temperature = 1.0
        kwargs = {
            "n": 1,
            "max_tokens": 1,
            "prompt_logprobs": 0,  # vLLM return `prompt_logprobs + 1` logrpobs for each token
            "temperature": temperature,
        }
        # avoid using prefix cache when calculating logprobs, only for vLLM >= 0.12.0
        if self.logprobs_no_prefix_cache:
            kwargs["skip_reading_prefix_cache"] = True
        output = await self._generate_internal(
            prompt={"prompt_token_ids": token_ids},
            lora_request=lora_request,
            **kwargs,
        )
        return torch.tensor(
            [list(logprob_dict.values())[0].logprob for logprob_dict in output.prompt_logprobs[1:]],
            dtype=torch.float32,
        )

    async def add_lora_adapter(self, lora_request: Any) -> int:
        """Add a LoRA adapter to the vLLM engine.

        Args:
            lora_request (LoRARequest): The LoRA request.

        Returns:
            lora_id (int): The LoRA adapter ID.
        """
        lora_id = await self.async_llm.add_lora(lora_request)
        return lora_id

    async def remove_lora_adapter(self, lora_id: int) -> None:
        """Remove a LoRA adapter from the vLLM engine.

        Args:
            lora_id (int): The LoRA adapter ID.
        """
        await self.async_llm.remove_lora(lora_id)

    async def list_lora_adapters(self) -> Sequence[int]:
        """List all LoRA adapter IDs in the vLLM engine.

        Returns:
            lora_ids (List[int]): The list of LoRA adapter IDs.
        """
        lora_ids = await self.async_llm.list_loras()
        return list(lora_ids)

    async def sample(
        self,
        prompt: Any,
        num_samples: int,
        sampling_params: Any,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_request: Optional[Any] = None,
    ) -> Any:
        """Tinker compatible sampling interface.

        Args:
            prompt (ModelInput): The input prompt.
            num_samples (int): The number of samples to generate.
            sampling_params (SamplingParams): The sampling parameters.
            include_prompt_logprobs (bool): Whether to include prompt logprobs.
            topk_prompt_logprobs (int): The top-k prompt logprobs to include.
            lora_request (LoRARequest, optional): The LoRA request. Defaults to None.
        Returns:
            SampleResponse: The sample response.
        """
        from tinker.types import SampledSequence, SampleResponse

        params = {
            "max_tokens": (
                sampling_params.max_tokens
                if sampling_params.max_tokens is not None
                else self.config.max_response_tokens
            ),
            "seed": sampling_params.seed if sampling_params.seed is not None else self.config.seed,
            "top_k": sampling_params.top_k,
            "top_p": sampling_params.top_p,
            "temperature": sampling_params.temperature,
            "n": num_samples,
            "prompt_logprobs": (topk_prompt_logprobs if include_prompt_logprobs else None),
            # in vLLM, 0 means only return the chosen token's logprob
            "logprobs": 0,
        }
        if include_prompt_logprobs and self.logprobs_no_prefix_cache:
            params["skip_reading_prefix_cache"] = True
        if sampling_params.stop is not None:
            params["stop"] = sampling_params.stop
        req_output = await self._generate_internal(
            prompt={"prompt_token_ids": prompt.to_ints()},
            lora_request=lora_request,
            **params,
        )
        sequences = []
        # vLLM's prompt_logprobs output does not include a value for the first token.
        # Initialize with [None] to align with the prompt tokens.
        topk_prompt_logprobs_list: List[Optional[List[Tuple[int, float]]]] = [None]
        prompt_logprobs: List[Optional[float]] = [None]

        # collect prompt logprobs
        if include_prompt_logprobs:
            for logprob_dict in req_output.prompt_logprobs[1:]:
                prompt_logprobs.append(next(iter(logprob_dict.values())).logprob)
                if topk_prompt_logprobs > 0:
                    # collect top-k prompt logprobs
                    # logprob_dict: {token_id: Logprob(logprob, rank, ...), ...}
                    logprob_items = list(logprob_dict.items())
                    # sort by Logprob.rank
                    logprob_items_sorted = sorted(logprob_items, key=lambda x: x[1].rank)
                    # pick topk
                    topk = logprob_items_sorted[:topk_prompt_logprobs]
                    # record as (token_id, logprob)
                    topk_prompt_logprobs_list.append(
                        [(token_id, logprob.logprob) for token_id, logprob in topk]
                    )
        # collect response sequences
        for seq_output in req_output.outputs:
            seq = SampledSequence(
                stop_reason="length" if seq_output.finish_reason == "length" else "stop",
                tokens=seq_output.token_ids,
                logprobs=[
                    next(iter(logprob_dict.values())).logprob
                    for logprob_dict in seq_output.logprobs
                ],
            )
            sequences.append(seq)
        return SampleResponse(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs if include_prompt_logprobs else None,
            topk_prompt_logprobs=(
                topk_prompt_logprobs_list
                if include_prompt_logprobs and topk_prompt_logprobs > 0
                else None
            ),
        )

    async def _generate_internal(self, prompt: Any, lora_request=None, **kwargs) -> Any:
        # Send the request to the LLM engine.
        self.request_id += 1
        generate_kwargs = (
            {"tokenization_kwargs": self.tokenization_kwargs}
            if self.vllm_version > parse_version("0.16.0")
            else {}
        )
        stream = self.async_llm.generate(
            request_id=str(self.request_id),
            prompt=prompt,
            sampling_params=self._create_sampling_params(**kwargs),
            lora_request=lora_request,
            **generate_kwargs,
        )

        # Consume the stream until the request is finished.
        async for request_output in stream:
            if request_output.finished:
                # Bypass the original full prompt.
                # request_output.prompt = request.prompt
                return request_output

        raise RuntimeError("[vLLM] The request is not finished. This should not happen.")

    async def shutdown(self):
        """Shutdown the vLLM v1 engine. This kills child processes forked
        by the vLLM engine. If not called, the child processes will be
        orphaned and will not be killed when the parent process exits,
        and they won't be able to be tracked by Ray anymore.
        """
        if self.api_server is not None:
            self.api_server.cancel()
            try:
                await self.api_server
            except asyncio.CancelledError:
                pass
            self.api_server = None
        if self.headless_executor is not None:
            self.logger.info("Shutting down headless executor")
            self.headless_executor.shutdown()
            self.headless_executor = None
        if self.async_llm is not None:
            self.logger.info("Shutting down vLLM engine")
            self.async_llm.shutdown()

    def _create_sampling_params(self, **kwargs):
        """Create sampling params."""
        if len(kwargs) == 0:
            return self.default_sampling_params
        params = self.default_sampling_params.clone()
        for k, v in kwargs.items():
            if hasattr(params, k):
                setattr(params, k, v)
        return params

    async def _collective_rpc(
        self,
        method: str,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        if self.use_v1:
            return await self.async_llm.collective_rpc(method, timeout, args, kwargs)
        else:
            return self.async_llm.engine.model_executor.collective_rpc(
                method, timeout, args, kwargs
            )

    async def sync_model_weights(
        self, model_version: int, sync_method: SyncMethod, timeout: float = 1200
    ) -> int:
        """Sync model weights to vLLM."""
        if self.config.node_rank != 0:
            self.logger.warning(
                "sync_model_weights should only be called on the main node (node_rank=0). "
                f"Current node_rank={self.config.node_rank}, skipping sync and returning version {model_version}."
            )
            return model_version
        if self.enable_lora:
            # Revise the lora path; no need to sync weights manually.
            self.default_lora_path = self.default_lora_path.replace(
                f"global_step_{self.model_version}", f"global_step_{model_version}"
            )
            self.logger.info(
                f"Redirect `lora_path` from old_model_version={self.model_version} to {model_version=} successfully."
            )
            lora_int_ids = await self.async_llm.list_loras()
            for lora_id in lora_int_ids:
                await self.async_llm.remove_lora(lora_id)
            await self.async_llm.add_lora(self.get_lora_request(self.default_lora_path))
            self.model_version = model_version
            return model_version
        await self.async_llm.reset_prefix_cache(reset_running_requests=True)
        await self._collective_rpc("update_weight", timeout=timeout)
        self.logger.info(
            f"Synchronized model to version {model_version} using method {sync_method}."
        )
        self.model_version = model_version
        return model_version

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
        if self.config.node_rank != 0:
            self.logger.warning(
                "init_process_group should only be called on the main node (node_rank=0). "
                f"Current node_rank={self.config.node_rank}, skipping initialization and returning."
            )
            return
        return await self._collective_rpc(
            "init_process_group",
            args=(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
                timeout,
                state_dict_meta,
                explorer_name,
                self.ray_namespace,
            ),
        )

    async def run_api_server(self) -> bool:
        """Run the OpenAI API server in a Ray actor.

        Returns:
            success (bool): Whether the API server is started successfully.
        """
        if not self.config.enable_openai_api:
            self.logger.info("OpenAI API server is not enabled. Skipping...")
            return False  # Not enabled

        if self.api_server_host is not None and self.api_server_port is not None:
            self.logger.info("OpenAI API server is already running. Skipping...")
            return True  # already running

        api_server_host, api_server_port = self.get_available_address()
        from trinity.common.models.vllm_patch import get_api_server

        self.api_server = get_api_server(
            self.async_llm,
            host=api_server_host,
            port=api_server_port,
            config=self.config,
            logger=self.logger,
        )
        self.api_server_host = api_server_host
        self.api_server_port = api_server_port
        return True

    def get_api_server_url(self) -> Optional[str]:
        """Get the URL of the OpenAI API server.

        Returns:
            api_url (str): The URL of the OpenAI API server.
        """
        if not self._prepared:
            raise RuntimeError("Model is not prepared. Please call `prepare()` first.")
        if self.api_server_host is None or self.api_server_port is None:
            # openai api is not enabled
            return None
        return f"http://{self.api_server_host}:{self.api_server_port}"

    async def reset_prefix_cache(self) -> None:
        await self.async_llm.reset_prefix_cache(reset_running_requests=True)

    def get_model_version(self) -> int:
        return self.model_version

    def get_lora_request(self, lora_path: Optional[str] = None) -> Any:
        from vllm.lora.request import LoRARequest

        assert self.config.lora_modules is not None
        lora_request = LoRARequest(**self.config.lora_modules[0])
        if lora_path is not None:
            self.config.lora_modules[0]["lora_path"] = lora_path  # for consistency
            lora_request.lora_path = lora_path
        return lora_request

    async def get_message_token_len(self, messages) -> int:
        if self.tokenizer is None:
            await self._initialize_tokenizer()
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self.chat_template,
            enable_thinking=self.enable_thinking,
        )
        prompt_token = self.tokenizer(  # type: ignore
            prompt, truncation=False, return_tensors="pt"
        )["input_ids"][0].tolist()
        return len(prompt_token)

    async def sleep(self, level: int = 1) -> None:
        await self.async_llm.sleep(level=level)

    async def wake_up(self) -> None:
        await self.async_llm.wake_up()
