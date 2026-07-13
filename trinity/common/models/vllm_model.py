"""A wrapper around the vllm.AsyncEngine to handle async requests."""

import asyncio
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from packaging.version import parse as parse_version
from transformers import AutoProcessor

from trinity.buffer.store import parse_record_key
from trinity.common.config import InferenceModelConfig
from trinity.common.constants import SyncMethod
from trinity.common.experience import Experience
from trinity.common.models.mm_utils import vLLMMultiModalRender
from trinity.common.models.model import BaseInferenceModel
from trinity.common.models.recording.context import (
    RecordingContext,
    recording_ctx,
    skip_recording_ctx,
)
from trinity.common.models.vllm_patch import get_vllm_version
from trinity.common.models.vllm_patch.recording.models import build_experience


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
        if config.gpu_per_engine != 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = config.bundle_indices
        if self.vllm_version >= parse_version("0.22.0"):
            # Force vLLM to use V1 for now
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
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
            "truncate_prompt_tokens": (
                config.max_prompt_tokens if config.enable_prompt_truncation else None
            )
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
        self.mm_render = None
        self.state_dict_meta = None
        self.model_version = 0  # TODO: resume the value from the checkpoint
        self.api_server_host = None
        self.api_server_port = None
        self.api_server = None
        self.recorder = None
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

    def _use_data_parallel_mode(self) -> bool:
        return self.config.data_parallel_size > 1 and self.config.nnodes > 1

    async def prepare(self) -> None:
        """Prepare the model for inference.

        Branches by launch mode:
        - SINGLE_NODE: create full AsyncLLMEngine with all DP/TP/PP via mp backend.
          When nnodes>1 and DP>1, each actor is a self-contained engine (dp_size=1).
        - HEADLESS: node_rank=0 creates full engine, node_rank>0 runs headless executor.
        """
        import vllm
        from vllm.config import WeightTransferConfig

        async with self.async_lock:
            if self._prepared:
                return

            weight_transfer_config = WeightTransferConfig(
                backend="nccl" if self.config.sync_method == SyncMethod.NCCL else "checkpoint"
            )

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
                worker_cls="trinity.common.models.vllm_worker.TrinityGPUWorker",
                tensor_parallel_size=self.config.tensor_parallel_size,
                pipeline_parallel_size=self.config.pipeline_parallel_size,
                data_parallel_size=self.config.data_parallel_size,
                enable_expert_parallel=self.config.enable_expert_parallel,
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
                enable_return_routed_experts=self.config.enable_return_routed_experts,
                reasoning_parser=self.config.reasoning_parser,
                disable_log_stats=True,
                enable_log_requests=self.config.enable_log_requests,
                enable_lora=self.config.enable_lora,
                logprobs_mode="processed_logprobs",
                nnodes=self.config.nnodes,
                node_rank=self.config.node_rank,
                async_scheduling=True,
                weight_transfer_config=weight_transfer_config,
                **rope_kwargs,
                **self.config.lora_kwargs,
                **self.config.extra_engine_args,
            )

            # Cross-node TP/PP: primary node vs headless nodes
            if self.config.tensor_parallel_size > 1 and self.config.nnodes > 1:
                engine_args.compilation_config.pass_config.fuse_allreduce_rms = False

            if self.master_addr is not None and self.master_port is not None:
                engine_args.master_addr = self.master_addr
                engine_args.master_port = self.master_port
            if self.config.node_rank == 0:
                self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
                # Expose the current checkpoint version on the engine instance so
                # the in-vLLM recorder (which only sees `engine_client`) can
                # attribute experiences to the right policy without an extra
                # launch-time parameter. Updated in sync_model_weights.
                self.async_llm.trinity_model_version = self.model_version
                if self.config.enable_history:
                    from trinity.common.models.vllm_patch.recording.recorder import (
                        TRINITY_MM_RENDER_ATTR,
                        create_vllm_recorder,
                    )

                    if self.mm_render is None:
                        self.mm_render = vLLMMultiModalRender(
                            model_path=self.config.model_path,  # type: ignore
                        )
                    setattr(self.async_llm, TRINITY_MM_RENDER_ATTR, self.mm_render)
                    self.recorder = create_vllm_recorder(self.async_llm, self.logger)
                    self.recorder.start()
                await self._collective_rpc("apply_patches")
                await self.run_api_server()
            else:
                # Headless executor for cross-node TP/PP
                from vllm.v1.executor.multiproc_executor import MultiprocExecutor

                vllm_config = engine_args.create_engine_config(headless=True)
                self.headless_executor = MultiprocExecutor(vllm_config, monitor_workers=False)
                self.headless_executor.start_worker_monitor()
            self._prepared = True

    async def chat(
        self,
        messages: List[Dict],
        lora_request=None,
        key: Optional[str] = None,
        **kwargs,
    ) -> Sequence[Experience]:
        """Chat with the model with a list of messages in async.

        Args:
            messages (List[dict]): The input history messages.
            key (Optional[str]): Recording identity for the in-vLLM
                recorder (the MemoryStore group key). Propagated to
                ``generate`` via ``recording_ctx`` so the recorder stamps it
                into ``Experience.eid`` without an HTTP hop. None skips
                recording.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        if self.mm_render is None:
            self.mm_render = vLLMMultiModalRender(
                model_path=self.config.model_path,  # type: ignore
            )
        prompt_messages, multi_modal_data = await self.mm_render.process_messages_async(messages)
        if multi_modal_data is not None:
            if self.processor is None:
                await self._initialize_processor()
            tokenizer_or_processor = self.processor
        else:
            if self.tokenizer is None:
                await self._initialize_tokenizer()
            tokenizer_or_processor = self.tokenizer

        prompt = self.apply_chat_template(tokenizer_or_processor, prompt_messages)
        if multi_modal_data is not None:
            prompt = {
                "prompt": prompt,
                "multi_modal_data": multi_modal_data or {},
            }
        return await self.generate(prompt=prompt, lora_request=lora_request, key=key, **kwargs)

    async def generate(
        self,
        prompt: Union[str, Dict],
        lora_request=None,
        key: Optional[str] = None,
        **kwargs,
    ) -> Sequence[Experience]:
        """Generate a response from the provided prompt in async.

        Args:
            prompt (str): The input prompt.
            key (Optional[str]): Recording identity propagated to the
                in-vLLM recorder via ``recording_ctx`` (see ``chat``).
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        is_mm_prompt = not isinstance(prompt, str)
        if not is_mm_prompt:  # pure text
            if self.tokenizer is None:
                await self._initialize_tokenizer()

            returned_seq, is_valid = self._handle_prompt_truncation(prompt, **kwargs)  # type: ignore
            if not is_valid:
                # Prompt was truncated: ``_handle_prompt_truncation`` returns
                # dummy (masked) experiences and we skip real generation. The
                # engine-level recorder only captures actual generations, so
                # persist these dummies directly under the record_key — masked
                # experiences must still be tracked for history extraction and
                # the buffer/trainer (they are popped by record_key on consume).
                if self.recorder is not None and key is not None:
                    batch, task, run = parse_record_key(key)
                    for exp in returned_seq:
                        exp.eid.batch = batch
                        exp.eid.task = task
                        exp.eid.run = run
                        exp.info["model_version"] = self.model_version
                        self.recorder.store.add(key, [exp])
                return returned_seq
            prompt = {
                "prompt_token_ids": returned_seq
            }  # is_valid is True: returned_seq is token_ids
            multi_modal_inputs = None

        # Propagate the recording identity to the engine-level recorder (same
        # async task, same process) so the recorded experience is grouped under
        # this record key in the MemoryStore.
        record_key_token = recording_ctx.set(RecordingContext(record_key=key))
        try:
            output = await self._generate_internal(
                prompt=prompt, lora_request=lora_request, **kwargs
            )
        finally:
            recording_ctx.reset(record_key_token)
        if is_mm_prompt:
            if self.mm_render is None:
                self.mm_render = vLLMMultiModalRender(
                    model_path=self.config.model_path,  # type: ignore
                )
            multi_modal_inputs = self.mm_render.build_mm_input_for_training(
                input_ids=output.prompt_token_ids,
                multi_modal_data=prompt.get("multi_modal_data", {}),
            )
        if self.tokenizer is None:
            await self._initialize_tokenizer()
        return build_experience(
            output,
            record_key=None,
            timestamp="",
            multi_modal_inputs=multi_modal_inputs,
            model_version=self.model_version,
            prompt_text=self.tokenizer.decode(output.prompt_token_ids),
            include_routed_experts=self.config.enable_return_routed_experts,
            include_prompt_routed_experts=True,
        )

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
        # This is an auxiliary 1-token forward, not a real turn — keep it out
        # of the recording store so it doesn't pollute task-id groups.
        skip_token = skip_recording_ctx.set(True)
        try:
            output = await self._generate_internal(
                prompt={"prompt_token_ids": token_ids},
                lora_request=lora_request,
                **kwargs,
            )
        finally:
            skip_recording_ctx.reset(skip_token)
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
        from tinker.types.topk_prompt_logprobs import TopkPromptLogprobs

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
        prompt_token_ids = prompt.to_ints()
        req_output = await self._generate_internal(
            prompt={"prompt_token_ids": prompt_token_ids},
            lora_request=lora_request,
            **params,
        )
        sequences = []
        prompt_logprobs_np = None
        topk_prompt_logprobs_np = None

        # collect prompt logprobs
        if include_prompt_logprobs:
            prompt_logprobs_np = np.full(len(prompt_token_ids), np.nan, dtype=np.float32)
            if topk_prompt_logprobs > 0:
                topk_token_ids = np.zeros(
                    (len(prompt_token_ids), topk_prompt_logprobs), dtype=np.int32
                )
                topk_logprobs = np.full(
                    (len(prompt_token_ids), topk_prompt_logprobs),
                    -99999.0,  # align with tinker's TopkPromptLogprobs
                    dtype=np.float32,
                )

            for prompt_idx, logprob_dict in enumerate(req_output.prompt_logprobs[1:], start=1):
                prompt_logprobs_np[prompt_idx] = next(iter(logprob_dict.values())).logprob
                if topk_prompt_logprobs > 0:
                    logprob_items = list(logprob_dict.items())
                    logprob_items_sorted = sorted(logprob_items, key=lambda x: x[1].rank)
                    topk = logprob_items_sorted[:topk_prompt_logprobs]
                    for topk_idx, (token_id, logprob) in enumerate(topk):
                        topk_token_ids[prompt_idx, topk_idx] = token_id
                        topk_logprobs[prompt_idx, topk_idx] = logprob.logprob

            if topk_prompt_logprobs > 0:
                topk_prompt_logprobs_np = TopkPromptLogprobs(
                    token_ids=topk_token_ids,
                    logprobs=topk_logprobs,
                )
        # collect response sequences
        for seq_output in req_output.outputs:
            seq = SampledSequence(
                stop_reason="length" if seq_output.finish_reason == "length" else "stop",
                tokens_np=np.asarray(seq_output.token_ids, dtype=np.int32),
                logprobs_np=np.asarray(
                    [
                        next(iter(logprob_dict.values())).logprob
                        for logprob_dict in seq_output.logprobs
                    ],
                    dtype=np.float32,
                ),
            )
            sequences.append(seq)
        return SampleResponse(
            sequences=sequences,
            prompt_logprobs_np=prompt_logprobs_np,
            topk_prompt_logprobs_np=topk_prompt_logprobs_np,
        )

    async def _generate_internal(self, prompt: Any, lora_request=None, **kwargs) -> Any:
        # Send the request to the LLM engine.
        self.request_id += 1
        generate_kwargs = {"tokenization_kwargs": self.tokenization_kwargs}
        stream = self.async_llm.generate(
            request_id=str(self.request_id),
            prompt=prompt,
            sampling_params=self._create_sampling_params(**kwargs),
            lora_request=lora_request,
            **generate_kwargs,
        )

        # Consume the stream to completion so engine-level recording runs only
        # after the full generation stream has ended.
        finished_output = None
        async for request_output in stream:
            if request_output.finished:
                # Bypass the original full prompt.
                # request_output.prompt = request.prompt
                finished_output = request_output

        if finished_output is None:
            raise RuntimeError("[vLLM] The request is not finished. This should not happen.")
        return finished_output

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
        if self.recorder is not None:
            await self.recorder.stop()
            self.recorder = None
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
        self,
        model_version: int,
        method: SyncMethod,
        timeout: float = 1200,
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
            self.async_llm.trinity_model_version = model_version
            return model_version

        from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

        await self.async_llm.pause_generation(mode="keep", clear_cache=False)
        await self.async_llm.reset_prefix_cache(reset_running_requests=True)

        await self.async_llm.start_weight_update(is_checkpoint_format=True)
        update_info = {}
        if method == SyncMethod.NCCL:
            update_info = dict(
                names=[meta[0] for meta in self.state_dict_meta],
                dtype_names=[meta[1] for meta in self.state_dict_meta],
                shapes=[meta[2] for meta in self.state_dict_meta],
                packed=True,
            )
        elif method == SyncMethod.CHECKPOINT:
            checkpoint_path = os.path.join(
                self.config.checkpoint_job_dir, f"global_step_{model_version}", "actor"  # type: ignore
            )
            update_info = dict(checkpoint_path=checkpoint_path)
        await self.async_llm.update_weights(WeightTransferUpdateRequest(update_info=update_info))
        await self.async_llm.finish_weight_update()
        await self.async_llm.resume_generation()
        self.model_version = model_version
        self.async_llm.trinity_model_version = model_version
        return model_version

    async def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: float = 1200,
    ):
        from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

        if self.config.node_rank != 0:
            self.logger.warning(
                "init_process_group should only be called on the main node (node_rank=0). "
                f"Current node_rank={self.config.node_rank}, skipping initialization and returning."
            )
            return
        self.logger.info(
            "vLLM starting init_process_group:\n"
            f"  > address={master_address}:{master_port}\n"
            f"  > rank_offset={rank_offset}\n"
            f"  > world_size={world_size}\n"
            f"  > group_name={group_name}\n"
        )
        init_info = dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
        )
        if self.config.sync_method != SyncMethod.NCCL:
            init_info["namespace"] = self.ray_namespace
            init_info["sync_method"] = self.config.sync_method.value
        await self.async_llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info=init_info)
        )
        self.logger.info("vLLM init_process_group finished.")

    async def set_state_dict_meta(self, state_dict_meta: List):
        """Set the state_dict meta for NCCL weight sync."""
        self.state_dict_meta = state_dict_meta

    async def run_api_server(self) -> bool:
        """Run the OpenAI API server in a Ray actor.

        Returns:
            success (bool): Whether the API server is started successfully.
        """
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

    def get_api_server_exit_reason(self) -> Optional[str]:
        if self.api_server is None or not self.api_server.done():
            return None
        if self.api_server.cancelled():
            return "cancelled"
        exc = self.api_server.exception()
        return "unknown error" if exc is None else repr(exc)

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
