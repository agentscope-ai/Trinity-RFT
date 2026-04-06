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
Modified from https://github.com/volcengine/verl/blob/v0.7.1/verl/utils/checkpoint/megatron_checkpoint_manager.py
"""

import inspect
import json
import os
from collections.abc import Callable
from dataclasses import asdict
from typing import Optional

import ray
import torch
import torch.distributed
from megatron.core.transformer.enums import AttnBackend
from transformers import GenerationConfig
from verl.utils.checkpoint.megatron_checkpoint_manager import (
    MegatronCheckpointManager as OldMegatronCheckpointManager,
)
from verl.utils.checkpoint.megatron_checkpoint_manager import logger
from verl.utils.fs import local_mkdir_safe
from verl.utils.logger import log_with_rank
from verl.utils.megatron.dist_checkpointing import save_dist_checkpointing
from verl.utils.megatron_utils import (
    get_dist_checkpoint_path,
    get_hf_model_checkpoint_path,
    get_transformer_config_checkpoint_path,
)

from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.verl.verl_trainer import CheckpointMonitor
from trinity.utils.log import get_logger


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

    def _is_latest_registered_checkpoint(self, path: str) -> bool:
        if not self.previous_saved_paths:
            return False
        return os.path.abspath(path) == os.path.abspath(self.previous_saved_paths[-1])

    def register_checkpoint(self, new_path: str, max_ckpt_to_keep: Optional[int] = None):
        if self._is_latest_registered_checkpoint(new_path):
            return
        super().register_checkpoint(new_path, max_ckpt_to_keep)

    def _save_state_dict(self, local_path, global_step, max_ckpt_to_keep=None) -> bool:
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
            elif self.use_hf_checkpoint:
                # Use mbridge to save HF model checkpoint
                log_with_rank(
                    f"Saving HF model checkpoint to {local_path} with bridge",
                    rank=self.rank,
                    logger=logger,
                )
                hf_ckpt_path = get_hf_model_checkpoint_path(local_path)
                if self.vanilla_bridge:
                    extended_args = {}
                    mbridge_config = getattr(self.checkpoint_config, "mbridge_config", None) or {}
                    for sig in inspect.signature(self.bridge.save_weights).parameters:
                        if sig == "weights_path" or sig == "models":
                            continue
                        if sig in mbridge_config:
                            extended_args[sig] = mbridge_config[sig]
                    self.bridge.save_weights(self.model, hf_ckpt_path, **extended_args)
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
                "_pg_collection",
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
                    extended_args = {}
                    mbridge_config = getattr(self.checkpoint_config, "mbridge_config", None) or {}
                    for sig in inspect.signature(self.bridge.save_weights).parameters:
                        if sig == "weights_path" or sig == "models":
                            continue
                        if sig in mbridge_config:
                            extended_args[sig] = mbridge_config[sig]
                    self.bridge.save_weights(self.model, hf_model_ckpt_path, **extended_args)
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

        skip_retention_rotation = self._is_latest_registered_checkpoint(local_path)

        if not self.checkpoint_config.async_save and not skip_retention_rotation:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)

        torch.distributed.barrier()

        state_dict_thread_count = 0
        if self._save_state_dict(local_path, global_step, max_ckpt_to_keep):
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

        if self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)
