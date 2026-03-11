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
Megatron Checkpoint Manager.
Modified from https://github.com/volcengine/verl/blob/v0.7.0/verl/utils/checkpoint/megatron_checkpoint_manager.py
"""

import json
import os
from collections.abc import Callable
from dataclasses import asdict

import megatron
import ray
import torch
import torch.distributed
from megatron.core import dist_checkpointing, mpu
from megatron.core.transformer.enums import AttnBackend
from packaging import version
from transformers import GenerationConfig
from verl.utils.checkpoint.megatron_checkpoint_manager import (
    MegatronCheckpointManager as OldMegatronCheckpointManager,
)
from verl.utils.checkpoint.megatron_checkpoint_manager import (
    is_non_local,
    load_dist_checkpointing,
    logger,
)
from verl.utils.fs import local_mkdir_safe
from verl.utils.logger import log_with_rank
from verl.utils.megatron.dist_checkpointing import (
    FullyParallelSaveStrategyWrapper,
    get_default_save_sharded_strategy,
)
from verl.utils.megatron_utils import (
    get_dist_checkpoint_path,
    get_hf_model_checkpoint_path,
    get_transformer_config_checkpoint_path,
)

from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.verl.verl_trainer import CheckpointMonitor
from trinity.utils.log import get_logger

mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
if not mcore_ge_014:
    logger.warning(
        "Detected megatron.core %s, recommend upgrading to >= 0.14.0 for better checkpoint compatibility",
        megatron.core.__version__,
    )


# TODO: removed after upgrading verl > 0.7.0; https://github.com/verl-project/verl/pull/5154
def save_dist_checkpointing(
    sharded_state_dict,
    ckpt_path,
    async_save=False,
    content_metadata=None,
):
    validate_sharding_integrity = True
    # Get checkpointing strategies
    save_strategy = get_default_save_sharded_strategy("torch_dist")
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/optimizer/distrib_optimizer.py#L1109-L1123
    mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
    # Save model sharded state dicts
    save_kwargs = dict(
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )
    if content_metadata is not None:
        if mcore_ge_014:
            save_kwargs["content_metadata"] = content_metadata

    return dist_checkpointing.save(sharded_state_dict, ckpt_path, **save_kwargs)


class MegatronCheckpointManager(OldMegatronCheckpointManager):
    """
    An enhanced version of the original Megatron checkpoint manager that:

    1. Uploads model state dicts to a remote Synchronizer actor (either directly or via checkpoints).
    """

    def __init__(
        self,
        *args,
        ray_namespace: str = "",
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.logger = get_logger()
        self.synchronizer = Synchronizer.get_actor(namespace=ray_namespace)
        self.checkpoint_monitor = CheckpointMonitor.get_actor(
            namespace=ray_namespace,
        )
        self.latest_model_save_step = None
        self.latest_tokenizer_save_step = None
        self.latest_extra_state_save_step = None
        self.latest_hf_model_save_step = None

    # TODO: removed after upgrading verl > 0.7.0; https://github.com/verl-project/verl/pull/5154
    def generate_state_dict(
        self,
        generate_model: bool = True,
        generate_optimizer: bool = True,
        generate_extra: bool = True,
        is_loading: bool = False,
        metadata: dict | None = None,
    ):
        # For save dist checkpointing
        state_dict = {}
        base_metadata = metadata or self._build_sharded_state_dict_metadata()

        # Should always generate model state dict
        # All ranks Save Model to reduce memory pressure
        # Get sharded state dict, notice that state_dict will collect among dp groups, causing memory pressure
        for vpp_rank, model in enumerate(self.model):
            if len(self.model) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                key = f"model{vpp_rank}" if len(self.model) > 1 else "model"
            else:
                key = "model"
            if hasattr(model, "module"):
                model = model.module

            # GPTModel's sharded_state_dict function when having mtp requires metadata['dp_cp_group']
            model_metadata = dict(base_metadata)
            model_metadata["dp_cp_group"] = mpu.get_data_parallel_group(with_context_parallel=True)
            kwargs = {"metadata": model_metadata}
            state_dict[key] = model.sharded_state_dict(**kwargs)

        # Optimizer State Dict
        if generate_optimizer:
            torch.distributed.barrier()
            sharded_state_dict_kwargs = {"is_loading": is_loading}
            if base_metadata is not None:
                # https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/optimizer/distrib_optimizer.py#L1109-L1123
                if mcore_ge_014:
                    sharded_state_dict_kwargs["metadata"] = base_metadata
            optimizer_sharded_states = self.optimizer.sharded_state_dict(
                state_dict, **sharded_state_dict_kwargs
            )
            state_dict["optimizer"] = optimizer_sharded_states

            if self.lr_scheduler is not None:
                lr_state_dict = self.lr_scheduler.state_dict()
                state_dict["lr_scheduler"] = lr_state_dict

        if not generate_model:
            state_dict.pop("model", None)

        # RNG States State Dict
        if generate_extra:
            torch.distributed.barrier()
            rng_state = self.get_rng_state()
            state_dict["rng_state"] = rng_state

        return state_dict

    # TODO: removed after upgrading verl > 0.7.0; https://github.com/verl-project/verl/pull/5154
    def _build_sharded_state_dict_metadata(self) -> dict:
        """Builds metadata used for sharded_state_dict versioning.


        The whole content metadata is passed to ``sharded_state_dict`` model and optimizer methods
        and therefore affects only the logic behind sharded_state_dict creation.
        The content metadata should be minimalistic, ideally flat (or with a single nesting level)
        and with semantically meaningful flag names (e.g. `distrib_optim_sharding_type`).
        In particular, a simple integer (or SemVer) versioning flag (e.g. `metadata['version'] = 3.4`)
        is discouraged, because the metadata serves for all models and optimizers and it's practically
        impossible to enforce a linearly increasing versioning for this whole space.
        """
        metadata: dict = {}

        if not mcore_ge_014:
            # For backward compatibility with Megatron core < v0.14.0
            if self.use_distributed_optimizer:
                metadata["distrib_optim_sharding_type"] = "fully_sharded_model_space"
            return metadata

        if self.use_distributed_optimizer:
            megatron_config = getattr(self.config, self.role, self.config).megatron
            dist_ckpt_optim_fully_reshardable = megatron_config.dist_ckpt_optim_fully_reshardable
            distrib_optim_fully_reshardable_mem_efficient = (
                megatron_config.distrib_optim_fully_reshardable_mem_efficient
            )
            if dist_ckpt_optim_fully_reshardable:
                metadata["distrib_optim_sharding_type"] = "fully_reshardable"
                metadata[
                    "distrib_optim_fully_reshardable_mem_efficient"
                ] = distrib_optim_fully_reshardable_mem_efficient
            else:
                metadata["distrib_optim_sharding_type"] = "dp_reshardable"

        metadata["singleton_local_shards"] = False
        metadata["chained_optim_avoid_prefix"] = True
        return metadata

    # TODO: removed after upgrading verl > 0.7.0; https://github.com/verl-project/verl/pull/5154
    def load_checkpoint(  # noqa: C901
        self, local_path: str, hdfs_path: str = None, del_local_after_load=False
    ):
        if local_path is not None:
            assert os.path.exists(local_path), f"Checkpoint path {local_path} does not exist."

        # For load optimizer dist_ckpt
        try:
            import transformer_engine

            torch.serialization.add_safe_globals([torch.optim.AdamW])
            torch.serialization.add_safe_globals(
                [transformer_engine.pytorch.optimizers.fused_adam.FusedAdam]
            )
        except Exception:
            pass

        dist_checkpoint_path = get_dist_checkpoint_path(local_path)

        load_content_metadata = getattr(dist_checkpointing, "load_content_metadata", None)
        if load_content_metadata is None:
            # For backward compatibility
            sharded_sd_metadata = None
        else:
            sharded_sd_metadata = load_content_metadata(checkpoint_dir=dist_checkpoint_path)
        if sharded_sd_metadata is None:
            if self.use_distributed_optimizer:
                # Backward-compatibility with old checkpoints which don't have content versioning
                # Can be removed after ending support for MLM optimizer checkpoints with MCore < v0.13
                # (for MCore v0.13+ checkpoints `sharded_sd_metadata is not None`)
                sharded_sd_metadata = {
                    "distrib_optim_sharding_type": "fully_sharded_model_space",
                }
            else:
                sharded_sd_metadata = self._build_sharded_state_dict_metadata()

        # Get State Dict for loading
        sharded_state_dict = self.generate_state_dict(
            self.should_load_model and self.use_dist_checkpointing,
            self.should_load_optimizer,
            self.should_load_extra,
            is_loading=True,
            metadata=sharded_sd_metadata,
        )
        log_with_rank(
            f"Generated state dict for loading: {sharded_state_dict.keys()}",
            rank=self.rank,
            logger=logger,
        )

        # Load Dist Checkpointing
        state_dict = load_dist_checkpointing(
            sharded_state_dict=sharded_state_dict,
            ckpt_dir=dist_checkpoint_path,
        )

        if self.should_load_model and self.use_dist_checkpointing:
            assert "model" in state_dict or any(
                f"model{vpp_rank}" in state_dict for vpp_rank in range(len(self.model))
            ), f"Model state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            for vpp_rank, model in enumerate(self.model):
                if len(self.model) == 1:
                    model_state_dict = state_dict["model"]
                else:
                    assert (
                        f"model{vpp_rank}" in state_dict
                    ), f"model{vpp_rank} not found in state_dict"
                    model_state_dict = state_dict[f"model{vpp_rank}"]
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                self.model[vpp_rank].load_state_dict(model_state_dict)
            log_with_rank(
                f"Loaded sharded model checkpoint from {local_path}", rank=self.rank, logger=logger
            )

        # Skip HF checkpoint loading if PEFT is used
        elif self.should_load_model and self.use_hf_checkpoint and self.peft_cls is None:
            hf_model_path = get_hf_model_checkpoint_path(local_path)
            if self.vanilla_bridge:
                self.bridge.load_weights(self.model, hf_model_path)
            else:
                self.bridge.load_hf_weights(self.model, hf_model_path)
            log_with_rank(
                f"Loaded HF model checkpoint from {hf_model_path} with bridge",
                rank=self.rank,
                logger=logger,
            )
        # Load PEFT adapter checkpoint if available
        if self.should_load_model and self.peft_cls is not None:
            adapter_ckpt_path = os.path.join(local_path, "adapter_checkpoint")
            if os.path.exists(adapter_ckpt_path):
                from verl.utils.megatron_peft_utils import load_adapter_checkpoint

                # TODO: a better format for adapter checkpoint, waiting megatron-bridge support

                load_adapter_checkpoint(
                    self.model,
                    adapter_ckpt_path,
                )
                log_with_rank(
                    f"Loaded adapter checkpoint from {adapter_ckpt_path}",
                    rank=self.rank,
                    logger=logger,
                )
            else:
                log_with_rank(
                    f"PEFT config is set but no adapter checkpoint found at {adapter_ckpt_path}",
                    rank=self.rank,
                    logger=logger,
                )

        if self.should_load_optimizer:
            assert (
                "optimizer" in state_dict
            ), f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            optimizer_state_dict = state_dict["optimizer"]
            self.optimizer.load_state_dict(optimizer_state_dict)
            log_with_rank(
                f"Loaded optimizer checkpoint from {local_path}", rank=self.rank, logger=logger
            )
            if self.use_checkpoint_opt_param_scheduler:
                assert "lr_scheduler" in state_dict, (
                    f"LR scheduler state dict not found in {state_dict.keys()}. Please check the checkpoint file "
                    f"{local_path}."
                )
                lr_scheduler_state_dict = state_dict["lr_scheduler"]
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                    log_with_rank(
                        f"Loaded LR scheduler checkpoint from {local_path}",
                        rank=self.rank,
                        logger=logger,
                    )

        if self.should_load_extra:
            assert (
                "rng_state" in state_dict
            ), f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            rng_state = state_dict["rng_state"]
            self.load_rng_states(rng_state)
            log_with_rank(f"Loaded RNG states from {local_path}", rank=self.rank, logger=logger)

        if del_local_after_load:
            try:
                os.remove(local_path) if is_non_local(local_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

    def _save_state_dict(self, local_path, global_step) -> bool:
        """
        Save the model state dict to the specified local path.

        Args:
            local_path (str): The local path where the model state dict should be saved.
            global_step (int): The current training step number.

        Returns:
            bool: True if the model save operation was initiated, False if a save for
                  this global_step has already been performed.
        """
        if self.latest_model_save_step == global_step:
            return False

        dist_checkpoint_path = get_dist_checkpoint_path(local_path)
        hf_ckpt_path = get_hf_model_checkpoint_path(local_path)

        # Note that model weights, optimizer states, and extra states are generated
        # together in a state dict, we save them in one time
        if self.use_dist_checkpointing:
            # Generate state dict for saving
            sharded_sd_metadata = self._build_sharded_state_dict_metadata()
            state_dict = self.generate_state_dict(
                self.should_save_model,
                self.should_save_optimizer,
                self.should_save_extra,
                metadata=sharded_sd_metadata,
            )
            # log_with_rank(f"Generated state dict for saving: {state_dict.keys()}", rank=self.rank, logger=logger)
            # for vpp_rank, model in enumerate(self.model):
            #     if len(self.model) > 1:
            #         model_i_keys = state_dict[f"model{vpp_rank}"].keys()
            #         log_with_rank(f"Generated state dict for saving: {model_i_keys}", rank=self.rank, logger=logger)
            #     else:
            #         log_with_rank(
            #             f"Generated state dict for saving: {state_dict['model'].keys()}", rank=self.rank, logger=logger
            #         )
            # Start Async save if enabled
            async_save_request = save_dist_checkpointing(
                sharded_state_dict=state_dict,
                ckpt_path=dist_checkpoint_path,
                async_save=self.checkpoint_config.async_save,
                content_metadata=sharded_sd_metadata,
            )

            # Synchronize all async save requests
            if not self.checkpoint_config.async_save:
                assert (
                    async_save_request is None
                ), "Async save request should be None when not using async save."
                torch.distributed.barrier()
        else:
            assert (
                self.use_hf_checkpoint
            ), "When not using distributed checkpointing, use_hf_checkpoint should be True."
            # Generate optimizer and exra state dicts
            sharded_sd_metadata = self._build_sharded_state_dict_metadata()
            state_dict = self.generate_state_dict(
                generate_model=False,
                generate_optimizer=self.should_save_optimizer,
                generate_extra=self.should_save_extra,
                metadata=sharded_sd_metadata,
            )
            # Save optimizer and extra states to local path
            # Start Async save if enabled
            async_save_request = save_dist_checkpointing(
                sharded_state_dict=state_dict,
                ckpt_path=dist_checkpoint_path,
                async_save=self.checkpoint_config.async_save,
                content_metadata=sharded_sd_metadata,
            )

            # Synchronize all async save requests
            if not self.checkpoint_config.async_save:
                assert (
                    async_save_request is None
                ), "Async save request should be None when not using async save."
                torch.distributed.barrier()

        if self.should_save_model:
            # Save adapter-only checkpoint if PEFT is enabled
            if self.peft_cls is not None:
                from verl.utils.megatron_peft_utils import save_adapter_checkpoint

                adapter_ckpt_path = os.path.join(local_path, "adapter_checkpoint")

                # Save adapter weights only (much smaller than full model)
                save_adapter_checkpoint(
                    self.model,
                    adapter_ckpt_path,
                    self.rank,
                )

                log_with_rank(
                    f"Saved adapter-only checkpoint to {adapter_ckpt_path}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
            if self.use_hf_checkpoint:
                # Use mbridge to save HF model checkpoint
                log_with_rank(
                    f"Saving HF model checkpoint to {local_path} with bridge",
                    rank=self.rank,
                    logger=logger,
                )
                hf_ckpt_path = get_hf_model_checkpoint_path(local_path)
                if self.vanilla_bridge:
                    self.bridge.save_weights(
                        self.model, hf_ckpt_path, distributed_filesystem=True, memory_efficient=True
                    )
                else:
                    self.bridge.save_hf_weights(self.model, hf_ckpt_path)

                log_with_rank(
                    f"Saved bridge checkpoint to {hf_ckpt_path}", rank=self.rank, logger=logger
                )

        def finalize_save_fn():
            # Rank 0 uploads checkpoint to HDFS if hdfs_path is provided
            runtime_context = ray.get_runtime_context()
            node_id = runtime_context.get_node_id()
            job_id = runtime_context.get_job_id()
            ray.get(self.checkpoint_monitor.notify_started.remote(node_id=node_id, job_id=job_id))
            log_with_rank(
                f"Dist checkpointing save completed for {dist_checkpoint_path}",
                rank=self.rank,
                logger=logger,
            )
            ray.get(self.checkpoint_monitor.notify_finished.remote(global_step, True))

        if self.checkpoint_config.async_save:
            assert (
                async_save_request is not None
            ), "Async save request should not be None when using async save."
            async_save_request.add_finalize_fn(finalize_save_fn)
            from megatron.core.dist_checkpointing.strategies.base import async_calls

            async_calls.schedule_async_request(async_save_request)
        else:
            finalize_save_fn()

        self.latest_model_save_step = global_step
        return True

    def _save_tokenizer(self, local_path, global_step) -> bool:
        """
        Save the tokenizer class to the specified local path.

        Args:
            local_path (str): The local path where the tokenizer class should be saved.
            global_step (int): The current training step number.

        Returns:
            bool: True if the tokenizer save operation was initiated, False if a save for
                  this global_step has already been performed.
        """
        if self.latest_tokenizer_save_step == global_step:
            return False

        if self.should_save_model:
            # Only rank 0 saves the hf config and tokenizer to huggingface path
            # No matter whether we save hf model or not
            if self.rank == 0:
                # Save tokenizer
                hf_config_tokenizer_path = get_hf_model_checkpoint_path(local_path)
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(hf_config_tokenizer_path)
                # Save huggingface config
                self.hf_config.save_pretrained(hf_config_tokenizer_path)
                if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
                    try:
                        generation_config = GenerationConfig.from_pretrained(
                            self.hf_config.name_or_path
                        )
                        generation_config.save_pretrained(hf_config_tokenizer_path)
                    except Exception:
                        # if the generation config isn't available, we don't save it
                        pass
                log_with_rank(
                    f"Saved Huggingface config and tokenizer to {hf_config_tokenizer_path}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )

        self.latest_tokenizer_save_step = global_step
        return self.rank == 0

    def _save_extra_state(self, local_path, global_step) -> bool:
        """
        Save the extra state dict to the specified local path.

        Args:
            local_path (str): The local path where the extra state dict should be saved.
            global_step (int): The current training step number.

        Returns:
            bool: True if the extra state dict save operation was initiated, False if a save for
                  this global_step has already been performed.
        """
        if self.latest_extra_state_save_step == global_step:
            return False

        if self.rank == 0:
            # Save transformer config
            print(self.transformer_config)
            bypass_keys = [
                "finalize_model_grads_func",
                "grad_scale_func",
                "no_sync_func",
                "grad_sync_func",
                "param_sync_func",
                "generation_config",
            ]
            backup = {}
            for k in bypass_keys:
                if hasattr(self.transformer_config, k):
                    backup[k] = getattr(self.transformer_config, k, None)
                    delattr(self.transformer_config, k)
            transformer_config_dict = asdict(self.transformer_config)
            for k in backup:
                setattr(self.transformer_config, k, backup[k])
            to_convert_types = {torch.dtype: str, AttnBackend: str}
            ignore_types = [Callable]
            pop_keys = []
            for key, value in transformer_config_dict.items():
                if type(value) in to_convert_types:
                    transformer_config_dict[key] = to_convert_types[type(value)](value)
                if type(value) in ignore_types:
                    pop_keys.append(key)
                if callable(value):
                    pop_keys.append(key)
            for key in pop_keys:
                transformer_config_dict.pop(key)
            transformer_config_path = get_transformer_config_checkpoint_path(local_path)
            with open(transformer_config_path, "w") as f:
                json.dump(transformer_config_dict, f, indent=2)

        return self.rank == 0

    def _save_hf_model(self, local_path, global_step) -> bool:
        """
        Save the Huggingface model to the specified local path.

        Args:
            local_path (str): The local path where the Huggingface model should be saved.
            global_step (int): The current training step number.

        Returns:
            bool: True if the Huggingface model save operation was initiated, False if a save for
                  this global_step has already been performed.
        """
        if self.latest_hf_model_save_step == global_step:
            return False

        try:
            # wait for everyone to dump to local
            if self.bridge is not None:
                hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                if self.vanilla_bridge:
                    self.bridge.save_weights(
                        self.model,
                        hf_model_ckpt_path,
                        distributed_filesystem=True,
                        memory_efficient=True,
                    )
                else:
                    self.bridge.save_hf_weights(self.model, hf_model_ckpt_path)
            else:
                state_dict = self.weight_saver(
                    self.model,
                    self.hf_config,
                    dtype=self.param_dtype,
                    is_value_model=self.is_value_model,
                    tie_word_embeddings=self.share_embeddings_and_output_weights,
                )

                torch.distributed.barrier()
                if self.rank == 0:
                    hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                    import warnings

                    from accelerate import init_empty_weights

                    # TODO: Switch to get_hf_auto_model_class
                    with init_empty_weights(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if "mistral7b-rm" in self.config.model.path:
                            from transformers import MistralForSequenceClassification

                            model = MistralForSequenceClassification.from_pretrained(
                                self.config.model.path
                            )  # use score head instead of lm_head
                            state_dict["score.weight"] = state_dict["score.weight"]
                        else:
                            from transformers import AutoModelForCausalLM

                            model = AutoModelForCausalLM.from_pretrained(
                                self.config.model.path, torch_dtype="auto"
                            )
                    model.save_pretrained(hf_model_ckpt_path, state_dict=state_dict)
                    log_with_rank(
                        f"Saved Huggingface config and tokenizer to {hf_model_ckpt_path}",
                        rank=self.rank,
                        logger=logger,
                        log_only_rank_0=True,
                    )

        except Exception:
            logger.error(
                f"Failed to save Huggingface model to {local_path}, you can try to set `use_mbridge=true` to save it.",
                exc_info=True,
            )

        self.latest_hf_model_save_step = global_step
        return self.rank == 0

    def save_state_dict(  # noqa: C901
        self,
        local_path: str,
        global_step: int = 0,
    ):
        if self.latest_model_save_step is None:
            # First sync in trainer.prepare
            self.latest_model_save_step = global_step
            return
        elif self.latest_model_save_step == global_step:
            # No need to save for sync again
            return

        local_path = local_mkdir_safe(local_path)
        self._save_state_dict(local_path, global_step)
        self._save_tokenizer(local_path, global_step)
        ray.get(
            self.checkpoint_monitor.register_thread_count.remote(
                global_step, state_dict_thread_count=1
            )
        )

    def save_checkpoint(
        self,
        local_path: str,
        global_step: int = 0,
        max_ckpt_to_keep=None,
        save_as_hf: bool = False,
    ):
        # record the previous global step
        self.previous_global_step = global_step
        local_path = local_mkdir_safe(local_path)

        # remove previous local_path
        if (
            not self.checkpoint_config.async_save
            and max_ckpt_to_keep
            and isinstance(max_ckpt_to_keep, int)
            and max_ckpt_to_keep > 0
            and len(self.previous_saved_paths) >= max_ckpt_to_keep  # type: ignore
            and local_path != self.previous_saved_paths[-1]  # type: ignore
        ):  # last step may save twice
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1  # type: ignore
            self.logger.info(
                "Checkpoint manager is removing previous checkpoints at "
                + str(self.previous_saved_paths[:keep_start])  # type: ignore
            )
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])  # type: ignore
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]  # type: ignore

        torch.distributed.barrier()

        state_dict_thread_count = 0
        if self._save_state_dict(local_path, global_step):
            state_dict_thread_count += 1

        self._save_tokenizer(local_path, global_step)

        if self.should_save_extra:
            self._save_extra_state(local_path, global_step)

        if (self.should_save_hf_model or save_as_hf) and not self.use_hf_checkpoint:
            self._save_hf_model(local_path, global_step)

        ray.get(
            self.checkpoint_monitor.register_thread_count.remote(
                global_step, state_dict_thread_count=state_dict_thread_count
            )
        )
        if (
            len(self.previous_saved_paths) == 0 or local_path != self.previous_saved_paths[-1]
        ):  # last step may save twice
            self.previous_saved_paths.append(local_path)
