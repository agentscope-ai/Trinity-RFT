# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm.
Modified from https://github.com/volcengine/verl/blob/v0.7.0/verl/workers/megatron_workers.py
"""

import datetime
import os
import sys
import time

import psutil
import ray
import torch
import torch.distributed
import vllm  # noqa: F401 ; import vllm to set NCCL_CUMEM_ENABLE automatically.
from codetiming import Timer
from megatron.core import parallel_state as mpu
from omegaconf import DictConfig, OmegaConf, open_dict

from trinity.utils.log import get_logger

try:
    from verl.workers.engine.mindspeed.transformer_impl import repatch
except ImportError:
    repatch = None

# start of patch for verl to support transformers v5
if not hasattr(sys.modules["transformers"], "AutoModelForVision2Seq"):
    setattr(
        sys.modules["transformers"],
        "AutoModelForVision2Seq",
        sys.modules["transformers"].AutoModelForImageTextToText,
    )
    sys.modules["transformers"].__all__.append("AutoModelForVision2Seq")

    import accelerate

    setattr(accelerate, "init_empty_weights", lambda: torch.device("cpu"))
# end of patch for verl to support transformers v5

from verl import DataProto
from verl.models.mcore import get_mcore_weight_converter
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.distributed import set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.megatron.router_replay_patch import (
    RouterReplay,
    RouterReplayAction,
    apply_router_replay_patch,
)
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    per_tensor_generator,
    register_megatron_training_hooks,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import (
    get_hf_model_path,
    load_mcore_dist_weights,
    load_megatron_gptmodel_weights,
)
from verl.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    GPUMemoryLogger,
    ProfilerConfig,
    log_gpu_memory_usage,
)
from verl.workers.config import McoreCriticConfig
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.workers.megatron_workers import logger, set_random_seed

from trinity.common.config import AlgorithmConfig
from trinity.common.constants import ROLLOUT_WEIGHT_SYNC_GROUP_NAME, SyncMethod
from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.verl.megatron_actor import MegatronPPOActor
from trinity.trainer.verl.megatron_checkpoint_manager import MegatronCheckpointManager
from trinity.utils.distributed import init_process_group


class MegatronWorker(Worker):
    def _init_hf_config_and_tf_config(  # noqa: C901
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
        megatron_config=None,
    ):
        from transformers import AutoConfig
        from verl.models.mcore import hf_to_mcore_config
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.model import update_model_config

        # Step 1: initialize the tokenizer
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(self.local_path, trust_remote_code=trust_remote_code)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = hf_tokenizer(
                copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code
            )
            self.processor = hf_processor(
                copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code
            )
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))

        # patch for rope
        if self.config.model.rope_scaling is not None:
            hf_config.rope_scaling = OmegaConf.to_container(self.config.model.rope_scaling)
        if self.config.model.rope_theta is not None:
            hf_config.rope_theta = self.config.model.rope_theta

        # start of patch for verl to support transformers v5
        if not hasattr(hf_config, "rope_theta"):
            rope_theta = hf_config.rope_parameters.get("rope_theta", None)
            if rope_theta is not None:
                setattr(hf_config, "rope_theta", rope_theta)
        # end of patch for verl to support transformers v5

        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            self.logger.info(f"Model config after override: {hf_config}")

        from verl.models.mcore.config_converter import mapping_string_to_attn_backend

        # todo: remove this line after mcore adopt mbridge 0.15, now for compatibility
        override_transformer_config = mapping_string_to_attn_backend(override_transformer_config)
        fp16 = dtype == torch.float16
        bf16 = dtype == torch.bfloat16
        if fp16:
            assert megatron_config.use_mbridge, "fp16 mode requires use_mbridge to be True"

        self.provider = None
        self.vanilla_bridge = megatron_config.get("vanilla_mbridge", True)
        if megatron_config.use_mbridge:
            if self.vanilla_bridge:
                from verl.models.mcore.mbridge import AutoBridge

                bridge = AutoBridge.from_config(hf_config, dtype=dtype)
                bridge.set_extra_args(**override_transformer_config)
                tf_config = bridge.config
                tf_config.fp16 = fp16
                tf_config.bf16 = bf16
            else:
                from verl.models.mcore.bridge import AutoBridge

                # Use Megatron-Bridge to convert HF config to Megatron config
                bridge = AutoBridge.from_hf_pretrained(
                    self.local_path, trust_remote_code=trust_remote_code
                )
                # Get Megatron provider and configure it
                provider = bridge.to_megatron_provider(load_weights=False)

                # In case of invalid overrides, we need to make sure some critical params are set correctly
                provider.params_dtype = dtype

                # Pass distributed info
                provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
                provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
                provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
                provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
                provider.virtual_pipeline_model_parallel_size = (
                    megatron_config.virtual_pipeline_model_parallel_size
                )
                provider.context_parallel_size = megatron_config.context_parallel_size
                provider.sequence_parallel = megatron_config.sequence_parallel

                # Match verl implementation (need variable_seq_lengths)
                from megatron.core.transformer.enums import AttnBackend

                provider.attention_backend = AttnBackend.flash
                provider.variable_seq_lengths = True
                provider.moe_token_dispatcher_type = "alltoall"
                provider.moe_router_load_balancing_type = "none"

                # Apply transformer config overrides
                for key, value in override_transformer_config.items():
                    setattr(provider, key, value)

                provider.finalize()
                self.provider = provider
                tf_config = None  # Will be set after model creation
            self.bridge = bridge
        else:
            tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)
            self.bridge = None

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                self.logger.info(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config

        # Get PEFT config from model.lora if specified
        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.config.model, bridge=self.bridge, provider=self.provider, dtype=dtype
        )


class ActorRolloutRefWorker(MegatronWorker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        if repatch is not None:
            # NPU MindSpeed patch, will be refactored with MindSpeedEngine.
            repatch(self.config.actor.megatron.get("override_transformer_config", {}))

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            if self._is_actor or self._is_ref:
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                    pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                    virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                    use_sharp=False,
                    context_parallel_size=self.config.actor.megatron.context_parallel_size,
                    expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
                    expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
                    nccl_communicator_config_path=None,
                )
        self.logger = get_logger(f"{role}_{self.rank}", in_ray_actor=True)

        if self._is_actor or self._is_ref:
            is_collect = (
                mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank()
                == mpu.get_pipeline_model_parallel_world_size() - 1
                and mpu.get_context_parallel_rank() == 0
            )
            self._register_dispatch_collect_info(
                mesh_name="actor", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
            )

        self.enable_routing_replay = False
        if self._is_actor:
            self.router_replay = self.config.actor.router_replay
            self.enable_routing_replay = self.router_replay.mode != "disabled"

        if self.enable_routing_replay:
            apply_router_replay_patch()

        set_random_seed(seed=self.config.actor.megatron.seed)

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(
            omega_profiler_config, dataclass_type=ProfilerConfig
        )
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # normalize config
        if self._is_actor:
            # note: no need to conduct `ppo_mini_batch_size *= rollout_n` anymore
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            if self.config.actor.get("ppo_micro_batch_size", None):
                self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.actor.ppo_micro_batch_size_per_gpu = (
                    self.config.actor.ppo_micro_batch_size
                )
                self.config.rollout.log_prob_micro_batch_size_per_gpu = (
                    self.config.rollout.log_prob_micro_batch_size
                )

            self._is_offload_param = self.config.actor.megatron.get("param_offload", False)
            self._is_offload_grad = self.config.actor.megatron.get("grad_offload", False)
            self._is_offload_optimizer = self.config.actor.megatron.get("optimizer_offload", False)
        elif self._is_ref:
            if self.config.ref.get("log_prob_micro_batch_size", None):
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size_per_gpu = (
                    self.config.ref.log_prob_micro_batch_size
                )
            else:
                assert self.config.ref.get("log_prob_micro_batch_size_per_gpu", None) is not None, (
                    "Please note that in the ref policy configuration, `log_prob_micro_batch_size_per_gpu` and "
                    "`log_prob_micro_batch_size` should not be None at the same time."
                )
            self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)

    def _build_model_optimizer(
        self,
        model_path,
        optim_config,
        override_model_config,
        override_transformer_config,
        override_ddp_config=None,
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )
        from verl.utils.megatron_utils import (
            McoreModuleWrapperConfig,
            make_megatron_module,
        )
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.get("tokenizer_path") or model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.actor.megatron if not self._is_ref else self.config.ref.megatron,
        )
        self.generation_config = get_generation_config(
            self.local_path,
            self.config.model.get("trust_remote_code", False),
        )

        if self._is_actor:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # actor is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            actor_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
                override_ddp_config=override_ddp_config,
                peft_cls=self.peft_cls,
                peft_config=self.config.model.get("lora", None),
            )
            self.tf_config = updated_tf_config
            self.logger.info(f"actor_module: {len(actor_module)}")
            if self.config.actor.load_weight:
                if self.config.actor.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        actor_module,
                        self.config.actor.megatron.dist_checkpointing_path,
                        is_value_model=False,
                        prefix=self.config.actor.megatron.dist_checkpointing_prefix,
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        if self.vanilla_bridge:
                            self.bridge.load_weights(actor_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(actor_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config,
                            self.hf_config,
                            actor_module,
                            params_dtype=self.dtype,
                            is_value_model=False,
                        )

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After MegatronPPOActor init", logger=self.logger)
        elif self._is_ref:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # ref is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )
            ref_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
            )
            self.tf_config = updated_tf_config
            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                self.logger.info("load ref weight start")
                if self.config.ref.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        ref_module,
                        self.config.ref.megatron.dist_checkpointing_path,
                        is_value_model=False,
                        prefix=self.config.ref.megatron.dist_checkpointing_prefix,
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        if self.vanilla_bridge:
                            self.bridge.load_weights(ref_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(ref_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config,
                            self.hf_config,
                            ref_module,
                            params_dtype=self.dtype,
                            is_value_model=False,
                        )
            log_gpu_memory_usage("After ref module init", logger=self.logger)
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config_megatron = init_megatron_optim_config(
                optim_config,
                use_distributed_optimizer=wrap_config.use_distributed_optimizer,
                fp16=self.dtype == torch.float16,
            )
            actor_optimizer = get_megatron_optimizer(
                model=actor_module, config=optim_config_megatron
            )
            actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
                optimizer=actor_optimizer, config=optim_config
            )
        else:
            optim_config = None
            actor_optimizer = None
            actor_optimizer_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=self.logger)

        register_megatron_training_hooks(actor_module, actor_optimizer)

        return (
            actor_module,
            actor_optimizer,
            actor_optimizer_scheduler,
            self.hf_config,
            optim_config,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        if self._is_actor:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_transformer_config", {}))
            )
            if self.enable_routing_replay:
                override_transformer_config["enable_routing_replay"] = True
            override_ddp_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_ddp_config", {}))
            )
        elif self._is_ref:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.ref.megatron.get("override_transformer_config", {}))
            )
        else:
            override_transformer_config = {}
        self.param_dtype = PrecisionType.to_dtype(self.config.actor.megatron.dtype)
        log_gpu_memory_usage("Before init actor model and optimizer", logger=self.logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        if self._is_actor:
            # we need the model for actor
            optim_config = self.config.actor.optim if self._is_actor else None
            (
                self.actor_module,
                self.actor_optimizer,
                self.actor_optimizer_scheduler,
                self.actor_model_config,
                self.actor_optim_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
                override_ddp_config=override_ddp_config,
            )
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage(
                    "After offload actor params and grad during init", logger=self.logger
                )
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=self.logger
                )

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                use_fused_kernels = self.config.model.get("use_fused_kernels", False)
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = MegatronPPOActor(
                config=self.config.actor,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
            )
            self.logger.info(f"routing replay layers: {len(RouterReplay.router_instances)}")
            log_gpu_memory_usage("After MegatronPPOActor init", logger=self.logger)

        if self._is_ref:
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After ref model init", logger=self.logger)
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage("After offload ref params during init", logger=self.logger)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                checkpoint_config=self.config.actor.checkpoint,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                optimizer=self.actor_optimizer,
                optimizer_scheduler=self.actor_optimizer_scheduler,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
                bridge=self.bridge,
                provider=self.provider,
                use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
                peft_cls=self.peft_cls,
                ray_namespace=self.config.synchronizer.ray_namespace,
            )

            self.layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            self.weight_converter = None
            if not self.config.actor.megatron.use_mbridge:
                self.weight_converter = get_mcore_weight_converter(
                    self.actor_model_config, self.dtype
                )

        self.synchronizer = Synchronizer.get_actor(namespace=self.config.synchronizer.ray_namespace)
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=self.logger)

    def _get_tensor_generator(self):
        """
        This part of the code is written by referring to the initialization of the `MegatronVLLMShardingManager` class
        in `verl.workers.megatron_workers.ActorRolloutRefWorker._build_rollout` and its `__enter__` method.
        When the version of verl changes, please check the related code.
        """
        if self.bridge is not None:
            if self.vanilla_bridge:
                per_tensor_param = self.bridge.export_weights(self.actor.actor_module)
            else:
                per_tensor_param = self.bridge.export_hf_weights(self.actor.actor_module)
        else:
            per_tensor_param = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                self.weight_converter,
                self.tf_config,
                self.layer_name_mapping,
            )
        return per_tensor_param

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def setup_weight_sync_group(self):
        if self.config.synchronizer.sync_method == SyncMethod.NCCL:
            aggressive_empty_cache(force_sync=True)
            set_expandable_segments(False)
            self.state_dict_meta = []

            if self._is_offload_param:
                load_megatron_model_to_gpu(self.actor_module)
            for name, weight in self._get_tensor_generator():
                self.state_dict_meta.append((name, str(weight.dtype), tuple(weight.shape)))
                del weight
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            torch.distributed.barrier()
            torch.cuda.empty_cache()

            if torch.distributed.get_rank() == 0:
                master_address, master_port = self.get_availale_master_addr_port()
                world_size = self.config.synchronizer.explorer_world_size + 1
                self.logger.info(
                    f"Trainer init_process_group {master_address}:{master_port} ({world_size})."
                )
                synchronizer = Synchronizer.get_actor(
                    namespace=self.config.synchronizer.ray_namespace
                )
                setup_ref = synchronizer.setup_weight_sync_group.remote(
                    master_address, master_port, self.state_dict_meta
                )
                timeout = self.config.synchronizer.sync_timeout

                self._model_update_group = init_process_group(
                    host=master_address,
                    port=master_port,
                    group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                    backend="nccl",
                    timeout=timeout,
                    world_size=world_size,
                    rank=0,
                )
                torch.distributed.barrier(group=self._model_update_group)
                ray.get(setup_ref)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_weight(self):
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        for name, weight in self._get_tensor_generator():
            if torch.distributed.get_rank() == 0:
                torch.distributed.broadcast(weight, 0, group=self._model_update_group)
            del weight
        if torch.distributed.get_rank() == 0:
            torch.distributed.barrier(group=self._model_update_group)
            torch.cuda.synchronize()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def upload_state_dict(self, trainer_step: int):
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        state_dict = {}
        for name, weight in self._get_tensor_generator():
            if torch.distributed.get_rank() == 0:
                state_dict[name] = weight.cpu().detach()
            del weight
        if torch.distributed.get_rank() == 0:
            ray.get(self.synchronizer.set_model_state_dict.remote(state_dict, trainer_step))
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_algorithm(self, algo_config: AlgorithmConfig):
        self.actor.set_algorithm(algo_config)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage(
                "After load actor params and grad during update_actor", logger=self.logger
            )
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage(
                "After load actor optimizer during update_actor", logger=self.logger
            )

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            global_num_tokens, delta_time
        )
        metrics["perf/mfu/actor"] = (
            estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        )
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (
            1024**3
        )
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (
            1024**3
        )
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage(
                "After offload actor params and grad during update_actor", logger=self.logger
            )
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage(
                "After offload actor optimizer during update_actor", logger=self.logger
            )

        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        if self._ref_is_offload_param:
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)
            log_gpu_memory_usage(
                "After load ref params and grad during compute_ref_log_prob", logger=self.logger
            )
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        output, _, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
        output = output.to("cpu")
        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)
            log_gpu_memory_usage(
                "After offload ref params and grad during compute_ref_log_prob", logger=self.logger
            )
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_gpu_memory_usage(
                "After load actor params and grad during compute_log_prob", logger=self.logger
            )
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        if self.enable_routing_replay and self.config.actor.router_replay.mode == "R2":
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

        if self.enable_routing_replay and self.config.actor.router_replay.mode == "R3":
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        output, entropys, layers_topk_idx = self.actor.compute_log_prob(
            data=data, calculate_entropy=True
        )
        output = DataProto.from_dict(
            tensors={"old_log_probs": output, "entropys": entropys},
            meta_info={"temperature": self.config.rollout.temperature},
        )
        if self.config.actor.router_replay.mode == "R2":
            output.batch["routed_experts"] = layers_topk_idx

        if self.config.actor.router_replay.mode in ["R2", "R3"]:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()

        output = output.to("cpu")
        # clear kv cache
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage(
                "After offload actor params and grad during compute_log_prob", logger=self.logger
            )
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        # No checkpoint to load, just offload the model and optimizer to CPU
        if checkpoint_path is None:
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage(
                "After offload actor params and optimizer during load_checkpoint",
                logger=self.logger,
            )
            return

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_state_dict(
        self,
        checkpoint_path,
        global_step=0,
    ):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.save_state_dict(
            local_path=checkpoint_path,
            global_step=global_step,
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        checkpoint_path,
        global_step=0,
        max_ckpt_to_keep=None,
        save_as_hf=False,
    ):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        if self.checkpoint_mananager.checkpoint_config.async_save and self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            save_as_hf=save_as_hf,
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self.checkpoint_mananager.checkpoint_config.async_save and self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def async_calls_finalize_fn_exec(self, blocking=False):
        from megatron.core.dist_checkpointing.strategies.base import async_calls

        async_calls.maybe_finalize_async_calls(blocking=blocking)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(
                        out_dir=out_dir, tag=tag, sub_dir=sub_dir
                    )
            except Exception as e:
                # Log a warning if memory snapshot fails. This might be expected if the profiler doesn't support it.
                self.logger.warning(f"Failed to dump memory snapshot: {e}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def wait_on_save_thread(self) -> None:
        self.async_calls_finalize_fn_exec(blocking=True)


class CriticWorker(MegatronWorker, DistProfilerExtension):
    def __init__(self, config: McoreCriticConfig):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(
            omega_profiler_config, dataclass_type=ProfilerConfig
        )
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        self.config: McoreCriticConfig = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )
        self.logger = get_logger(f"critic_{self.rank}", in_ray_actor=True)

        is_collect = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank()
            == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )
        self._register_dispatch_collect_info(
            mesh_name="critic", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
        )

        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        # note: no need to conduct `ppo_mini_batch_size *= rollout_n` anymore
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.get("ppo_micro_batch_size", None):
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        # TODO(sgm): support critic model offload

    def _build_critic_model_optimizer(
        self,
        model_path,
        optim_config,
        override_model_config,
        override_transformer_config,
        override_ddp_config,
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )
        from verl.utils.megatron_utils import (
            McoreModuleWrapperConfig,
            make_megatron_module,
        )
        from verl.utils.model import print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.get("tokenizer_path") or model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.megatron,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,  # critic is value model
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        critic_module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.config.model.get("lora", None),
        )
        self.tf_config = updated_tf_config
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            t0 = time.time()
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    critic_module,
                    self.config.megatron.dist_checkpointing_path,
                    is_value_model=True,
                    prefix=self.config.megatron.dist_checkpointing_prefix,
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    if self.vanilla_bridge:
                        self.bridge.load_weights(critic_module, local_model_path)
                    else:
                        self.bridge.load_hf_weights(
                            critic_module,
                            local_model_path,
                            allowed_mismatched_params=["output_layer.weight"],
                        )
                else:
                    load_megatron_gptmodel_weights(
                        self.config,
                        self.hf_config,
                        critic_module,
                        params_dtype=self.dtype,
                        is_value_model=True,
                    )
            t1 = time.time()
            if torch.distributed.get_rank() == 0:
                self.logger.info(f"critic load_weight time: {t1 - t0}")
        if self.rank == 0:
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        optim_config_megatron = init_megatron_optim_config(
            optim_config,
            use_distributed_optimizer=wrap_config.use_distributed_optimizer,
            fp16=self.dtype == torch.float16,
        )
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
        critic_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=critic_optimizer, config=optim_config
        )
        get_torch_device().empty_cache()

        register_megatron_training_hooks(critic_module, critic_optimizer)

        return (
            critic_module,
            critic_optimizer,
            critic_optimizer_scheduler,
            self.hf_config,
            optim_config,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        override_transformer_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_transformer_config", {}))
        )
        override_ddp_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_ddp_config", {}))
        )
        self.param_dtype = PrecisionType.to_dtype(self.config.megatron.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        (
            self.critic_module,
            self.critic_optimizer,
            self.critic_optimizer_scheduler,
            self.critic_model_config,
            critic_optimizer_config,
        ) = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.checkpoint,
            model_config=self.critic_model_config,
            transformer_config=self.tf_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.critic_optimizer,
            optimizer_scheduler=self.critic_optimizer_scheduler,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            provider=self.provider,
            use_dist_checkpointing=self.config.megatron.use_dist_checkpointing,
            peft_cls=self.peft_cls,
            ray_namespace=self.config.ray_namespace,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="compute_values")
    def compute_values(self, data: DataProto):
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink", role="critic_update")
    def update_critic(self, data: DataProto):
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.critic_optimizer)

        dataloader = self.critic.make_minibatch_iterator(data)
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self.critic.update_critic(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            global_num_tokens, delta_time
        )
        metrics["perf/mfu/critic"] = (
            estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
        )
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["critic/lr"] = get_megatron_last_lr(self.critic_optimizer)
        self.critic_optimizer_scheduler.step(1)

        output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        checkpoint_path,
        global_step=0,
        max_ckpt_to_keep=None,
        save_as_hf=False,
    ):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            save_as_hf=save_as_hf,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def async_calls_finalize_fn_exec(self, blocking=False):
        from megatron.core.dist_checkpointing.strategies.base import async_calls

        async_calls.maybe_finalize_async_calls(blocking=blocking)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def wait_on_save_thread(self) -> None:
        self.async_calls_finalize_fn_exec(blocking=True)
