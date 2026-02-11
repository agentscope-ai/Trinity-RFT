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
FSDP Checkpoint Manager.
Modified from https://github.com/volcengine/verl/blob/v0.7.0/verl/utils/checkpoint/fsdp_checkpoint_manager.py
"""

import json
import os
import threading
import warnings
from dataclasses import asdict
from typing import Optional, Union

import ray
import torch
from accelerate import init_empty_weights
from torch.distributed.fsdp import (
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from transformers import GenerationConfig
from transformers.dynamic_module_utils import custom_object_save
from verl.utils.checkpoint.fsdp_checkpoint_manager import (
    FSDPCheckpointManager as OldFSDPCheckpointManager,
)
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPConfig, logger
from verl.utils.device import is_cuda_available
from verl.utils.fs import local_mkdir_safe
from verl.utils.fsdp_utils import (
    fsdp_version,
    get_fsdp_full_state_dict,
    get_fsdp_state_ctx,
)
from verl.utils.logger import log_with_rank

from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.verl.utils import get_model_class
from trinity.trainer.verl_trainer import CheckpointMonitor
from trinity.utils.log import get_logger


class FSDPCheckpointManager(OldFSDPCheckpointManager):
    """
    An enhanced version of the original FSDP checkpoint manager that:

    1. Uploads model state dicts to a remote Synchronizer actor (either directly or via checkpoints).
    2. Offloads saving operations (model, optimizer, extra states) into background threads to avoid blocking the training loop.

    This class is useful in distributed training scenarios where synchronization and non-blocking I/O are important.
    """

    def __init__(self, *args, ray_namespace: str = "", trust_remote_code: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger()
        self.synchronizer = Synchronizer.get_actor(namespace=ray_namespace)
        self.checkpoint_monitor = CheckpointMonitor.get_actor(
            namespace=ray_namespace,
        )
        self.trust_remote_code = trust_remote_code

        # Threads for asynchronous saving of different components
        self._model_state_dict_thread = None
        self._optimizer_state_dict_thread = None
        self._extra_state_dict_thread = None
        self._save_model_thread = None
        self.latest_model_save_step = None
        self.latest_optimizer_save_step = None
        self.latest_extra_state_save_step = None
        self.latest_hf_model_save_step = None
        self.latest_tokenizer_save_step = None

    def _upload_state_dict(self, state_dict: Union[dict, None], global_step: int):
        """
        Internal method to upload a state dict to the Synchronizer actor.

        Args:
            state_dict (dict or None): The model state dictionary to upload.
            global_step (int): The current training step number.
        """
        if self.rank == 0:
            ray.get(self.synchronizer.set_model_state_dict.remote(state_dict, global_step))

    def upload_state_dict(self, global_step: int):
        """
        Uploads the full model state dictionary to the synchronizer actor for remote access.

        Args:
            global_step (int): The current training step number.
        """
        assert self.synchronizer is not None
        state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
            state_dict = self.model.state_dict()
        state_dict = {key: value.to("cpu") for key, value in state_dict.items()}
        self._upload_state_dict(state_dict, global_step)

    def _save_with_thread(
        self,
        obj,
        local_path: str,
        prefix: str,
        thread_name: str,
        global_step: int,
        is_state_dict: bool = False,
    ):
        path = os.path.join(
            local_path, f"{prefix}_world_size_{self.world_size}_rank_{self.rank}.pt"
        )
        thread = getattr(self, thread_name)
        if thread is not None:
            thread.join()

        def _save():
            runtime_context = ray.get_runtime_context()
            node_id = runtime_context.get_node_id()
            job_id = runtime_context.get_job_id()
            ray.get(self.checkpoint_monitor.notify_started.remote(node_id=node_id, job_id=job_id))
            torch.save(obj, path)
            log_with_rank(
                f"Saved {prefix} to {os.path.abspath(path)}",
                rank=self.rank,
                logger=logger,
            )
            ray.get(self.checkpoint_monitor.notify_finished.remote(global_step, is_state_dict))

        thread = threading.Thread(
            target=_save,
        )
        thread.start()
        setattr(self, thread_name, thread)

    def _save_model(self, local_path, global_step) -> bool:
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

        model_state_dict = self.model.state_dict()
        self._save_with_thread(
            model_state_dict, local_path, "model", "_model_state_dict_thread", global_step, True
        )
        self.latest_model_save_step = global_step
        return True

    def _save_optimizer(self, local_path, global_step) -> bool:
        """
        Save the optimizer state dict to the specified local path.

        Args:
            local_path (str): The local path where the optimizer state dict should be saved.
            global_step (int): The current training step number.

        Returns:
            bool: True if the optimizer save operation was initiated, False if a save for
                  this global_step has already been performed.
        """
        if self.latest_optimizer_save_step == global_step:
            return False

        optimizer_state_dict = self.optimizer.state_dict()
        self._save_with_thread(
            optimizer_state_dict, local_path, "optim", "_optimizer_state_dict_thread", global_step
        )
        self.latest_optimizer_save_step = global_step
        return True

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

        lr_scheduler_state_dict = (
            self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        )
        extra_state_dict = {
            "lr_scheduler": lr_scheduler_state_dict,
            "rng": self.get_rng_state(),
        }
        self._save_with_thread(
            extra_state_dict, local_path, "extra_state", "_extra_state_dict_thread", global_step
        )
        self.latest_extra_state_save_step = global_step
        return True

    def _get_unwrap_model_and_config(self):
        if fsdp_version(self.model) == 1:
            unwrap_model = self.model._fsdp_wrapped_module
        else:
            unwrap_model = self.model

        model_config = unwrap_model.config
        if (
            unwrap_model.can_generate()
            and hasattr(model_config, "name_or_path")
            and model_config.name_or_path
        ):
            try:
                # Some model's name_or_path is empty if not initialized from pretrained,
                # in this cases, we don't save generation config.
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
            except Exception:
                # if the generation config isn't available, we don't save it
                generation_config = None
        else:
            generation_config = None

        if hasattr(model_config, "auto_map") and None in model_config.auto_map:
            model_config.auto_map = {
                k: v for k, v in model_config.auto_map.items() if k is not None
            }
        return unwrap_model, model_config, generation_config

    def _save_tokenizer(self, local_path, global_step):
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

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)

            unwrap_model, model_config, generation_config = self._get_unwrap_model_and_config()
            model_config.save_pretrained(hf_config_tokenizer_path)
            if generation_config is not None:
                generation_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)

        # wait for everyone to dump to local
        torch.distributed.barrier()
        self.latest_tokenizer_save_step = global_step

        return self.rank == 0

    def _save_hf_model(self, local_path, global_step) -> bool:
        """
        Save the HuggingFace model to the specified local path.

        Args:
            local_path (str): The local path where the HuggingFace model should be saved.
            global_step (int): The current training step number.

        Returns:
            bool: True if the HuggingFace model save operation was initiated, False if a save for
                  this global_step has already been performed.
        """

        if self.latest_hf_model_save_step == global_step:
            return False

        # Only rank 0 will save hf model and,
        # offload to cpu to save LLMs which may be too large to fit in one GPU
        state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

        if self.rank == 0:
            hf_local_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)

            _, model_config, generation_config = self._get_unwrap_model_and_config()
            auto_model_cls = get_model_class(model_config)

            with init_empty_weights():
                save_model = auto_model_cls.from_config(
                    model_config,
                    dtype=torch.bfloat16,
                    trust_remote_code=self.trust_remote_code,
                )
            save_model.to_empty(device="cpu")

            if save_model.can_generate():
                if generation_config is not None:
                    save_model.generation_config = generation_config
                else:
                    logger.warning(
                        f"{self.__class__.__name__}.save_checkpoint: Generation config file not found in, "
                        "using a generation config created from the model config when saving hf_model."
                    )

            if self._save_model_thread is not None:
                self._save_model_thread.join()

            state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            def _save_hf_model_thread_target():
                runtime_context = ray.get_runtime_context()
                node_id = runtime_context.get_node_id()
                job_id = runtime_context.get_job_id()
                ray.get(
                    self.checkpoint_monitor.notify_started.remote(node_id=node_id, job_id=job_id)
                )
                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                ray.get(self.checkpoint_monitor.notify_finished.remote(global_step))

            self._save_model_thread = threading.Thread(
                target=_save_hf_model_thread_target,
            )
            self._save_model_thread.start()

        # wait for rank0 to dump hf_model to local
        torch.distributed.barrier()
        self.latest_hf_model_save_step = global_step

        return self.rank == 0

    def save_state_dict(
        self,
        local_path: str,
        global_step: int = 0,
    ):
        if self.latest_model_save_step is None:
            # First sync in trainer.prepare
            self.latest_model_save_step = global_step
            self._upload_state_dict(None, global_step)
            return
        elif self.latest_model_save_step == global_step:
            # No need to save for sync again
            return
        if local_path is None:
            return

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(
                self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
            ):
                self._save_model(local_path, global_step)
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
        max_ckpt_to_keep: Optional[int] = None,
        save_as_hf: bool = False,
    ):
        """
        Modified from verl.utils.checkpoint.fsdp_checkpoint_manager.py:save_checkpoint

        Saves the model checkpoint to disk, optionally uploads it to a remote Synchronizer,
        and uses background threads to prevent blocking the main training loop.

        Main improvements over the base class:
        - Uses separate threads for saving model/optimizer/extras.
        - Implements synchronization with a remote actor. If the model is not trained (`global_step == 0`) or continues from a breakpoint, `Synchonizer` will be notified and the model will not be saved.

        Args:
            local_path (str): Local directory path to save the checkpoint.
            hdfs_path (str, optional): HDFS path for saving the checkpoint (not implemented here).
            global_step (int): Current training step.
            max_ckpt_to_keep (int, optional): Maximum number of checkpoints to keep locally.
            save_as_hf (bool): Whether to force save the model in Hugging Face format.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step
        local_path = local_mkdir_safe(local_path)

        # remove previous local_path, only rank 0 should do this
        if (
            self.rank == 0
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

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert (
                self.model is not None
            ), "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert (
                self.optimizer is not None
            ), "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"

        state_dict_thread_count = 0
        checkpoint_thread_count = 0

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(
                self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
            ):
                if self.should_save_model:
                    if self._save_model(local_path, global_step):
                        state_dict_thread_count += 1

                if self.should_save_optimizer:
                    if self._save_optimizer(local_path, global_step):
                        checkpoint_thread_count += 1

                if self.should_save_extra:
                    if self._save_extra_state(local_path, global_step):
                        checkpoint_thread_count += 1

        self._save_tokenizer(local_path, global_step)

        if self.should_save_hf_model or save_as_hf:
            if self._save_hf_model(local_path, global_step):
                checkpoint_thread_count += 1

        ray.get(
            self.checkpoint_monitor.register_thread_count.remote(
                global_step,
                state_dict_thread_count=state_dict_thread_count,
                checkpoint_thread_count=checkpoint_thread_count,
            )
        )
        if (
            len(self.previous_saved_paths) == 0 or local_path != self.previous_saved_paths[-1]
        ):  # last step may save twice
            self.previous_saved_paths.append(local_path)

    def wait_on_save_thread(self) -> None:
        """
        Wait for all background saving threads to complete.
        """
        if self._model_state_dict_thread is not None:
            self._model_state_dict_thread.join()
        if self._optimizer_state_dict_thread is not None:
            self._optimizer_state_dict_thread.join()
        if self._extra_state_dict_thread is not None:
            self._extra_state_dict_thread.join()
        if self._save_model_thread is not None:
            self._save_model_thread.join()
