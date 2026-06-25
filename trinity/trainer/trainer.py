# -*- coding: utf-8 -*-
"""
Trainer Class
"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ray

from trinity.algorithm import SAMPLE_STRATEGY
from trinity.algorithm.sample_strategy.sample_strategy import SampleStrategy
from trinity.common.config import Config
from trinity.common.constants import RunningStatus, SyncMethod, SyncStyle
from trinity.common.experience import Experience
from trinity.manager.state_manager import StateManager
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR
from trinity.utils.plugin_loader import load_plugins
from trinity.utils.timer import Timer


class Trainer:
    """Consume the experience and train the model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(config.trainer.name, in_ray_actor=True)
        load_plugins()
        self.synchronizer = Synchronizer.get_actor(config)
        self.engine = get_trainer_wrapper(config)
        self.state = StateManager(
            path=config.checkpoint_job_dir, trainer_name=config.trainer.name, config=config
        )
        trainer_state = self.state.load_trainer()
        self.monitor = MONITOR.get(config.monitor.monitor_type)(
            project=config.project,
            group=self.config.group,
            name=config.name,
            role=config.trainer.name,
            config=config,
        )
        self._sample_exps_to_log = []
        self.sample_strategy: SampleStrategy = SAMPLE_STRATEGY.get(
            config.algorithm.sample_strategy
        )(
            buffer_config=config.buffer,
            **config.algorithm.sample_strategy_args,
        )
        if "latest_exp_index" in trainer_state:
            sample_strategy_state = {"current_index": trainer_state["latest_exp_index"]}
        else:
            sample_strategy_state = trainer_state.get("sample_strategy_state", {})
        self.sample_strategy.load_state_dict(sample_strategy_state)
        self.save_interval = config.trainer.save_interval
        self.last_sync_step = 0
        self.last_sync_time = None
        self.sync_interval: int = config.synchronizer.trainer_sync_interval
        self.sync_method = config.synchronizer.sync_method
        self.sync_style = config.synchronizer.sync_style
        self.total_steps = config.trainer.total_steps or float("inf")
        self.save_hf_checkpoint = config.trainer.save_hf_checkpoint

    async def prepare(self) -> None:
        """Prepare the trainer."""
        await self.engine.prepare()
        self.last_sync_step = self.train_step_num
        await self.synchronizer.set_trainer_status.remote(RunningStatus.RUNNING)
        self.logger.info("Trainer is ready.")

    async def get_weight_sync_info(self) -> Optional[Tuple[str, int, List]]:
        """Get rendezvous info for NCCL weight sync group setup.

        Returns (master_address, master_port, state_dict_meta) from the
        trainer's GPU worker rank 0. Called by Synchronizer before
        coordinating NCCL group creation.
        """
        return await self.engine.get_weight_sync_info()

    async def setup_weight_sync_group(
        self,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        timeout: int,
    ) -> None:
        """Join the NCCL weight sync group. Called by Synchronizer."""
        await self.engine.setup_weight_sync_group(
            master_address, master_port, world_size, group_name, timeout
        )

    async def teardown_weight_sync_group(self) -> None:
        """Destroy the NCCL weight sync group. Called by Synchronizer."""
        await self.engine.teardown_weight_sync_group()

    async def train(self) -> str:
        """Train the model."""
        while self.train_step_num < self.total_steps:
            try:
                metrics = {}
                # sample may be blocked due to explorer does not generate enough data
                self.logger.info(f"Sample data for step {self.train_step_num + 1} started.")
                sample_task = asyncio.create_task(self._sample_data())
                while not sample_task.done():
                    # sync weight to make sure the explorer can continue to explore and generate enough data
                    if await self.need_sync():
                        metrics.update(await self.sync_weight())
                    await asyncio.sleep(1)
                exps, sample_metrics, repr_samples = await sample_task
                metrics.update(sample_metrics)
                self.logger.info(f"Sample data for step {self.train_step_num + 1} finished.")
                metrics.update(await self.train_step(exps))
                need_sync = await self.need_sync()
                need_save = self.need_save()
                # For CHECKPOINT sync, save_checkpoint is a superset of
                # save_state_dict — skip the latter to avoid redundant writes
                # to the same directory.
                if need_sync and not (need_save and self.sync_method == SyncMethod.CHECKPOINT):
                    metrics.update(await self.sync_weight())
                if need_save:
                    # Only block for the final step (when total_steps is
                    # finite) to avoid stalling the training loop on earlier
                    # saves. The HF-model write for save_hf_checkpoint="last"
                    # is handled unconditionally by the post-loop save below.
                    is_final_step = self.train_step_num >= self.total_steps
                    save_as_hf = self.save_hf_checkpoint == "always" or (
                        is_final_step and self.save_hf_checkpoint == "last"
                    )
                    metrics.update(
                        await self.save_checkpoint(
                            block_until_saved=is_final_step,
                            save_as_hf=save_as_hf,
                        )
                    )
                    if need_sync:
                        # Update sync bookkeeping even though sync_weight was
                        # skipped — save_checkpoint already wrote the weights
                        # and updated latest_state_dict_iteration.txt.
                        self.last_sync_step = self.train_step_num
                        self.last_sync_time = time.time()
                if self.config.trainer.enable_preview:
                    self._log_experiences(repr_samples)
                self.monitor.log(metrics, self.train_step_num)
            except StopAsyncIteration:
                self.logger.info("No more samples to train. Stopping training.")
                break
            except Exception:
                self.logger.error(f"Error in Trainer:\n{traceback.format_exc()}")
                break

        # Always perform a final save to guarantee:
        # 1. A checkpoint exists even when the loop never triggered
        #    need_save() at the final step.
        # 2. The HF model is saved for save_hf_checkpoint="last" even
        #    when config.trainer.total_steps is None (total_steps becomes
        #    float("inf"), so is_final_step is never True inside the loop
        #    and save_as_hf stays False there). The loop may exit via
        #    StopAsyncIteration after the last step has already been saved
        #    *without* the HF model; this call fills that gap.
        # Checkpoint managers deduplicate saves by tracking the latest
        # saved global_step per component, so re-saving an already-written
        # component is a safe no-op.
        await self.save_checkpoint(
            block_until_saved=True, save_as_hf=self.save_hf_checkpoint != "never"
        )
        await self.synchronizer.set_trainer_status.remote(RunningStatus.STOPPED)
        self.logger.info("--------------------\n> Trainer finished.\n--------------------")
        return self.config.trainer.name

    async def train_step(self, exps: List[Experience]) -> Dict:
        """Train one step.

        Returns:
            bool: Whether to continue training.
            Dict: Metrics of the training step.
        """
        self.logger.info(f"Training at step {self.train_step_num + 1} started.")
        metrics = {}
        with Timer(metrics, "time/train_step"):
            train_metrics = await self.engine.train_step(exps)
        self.logger.info(f"Training at step {self.train_step_num} finished.")
        metrics.update(train_metrics)
        return metrics

    async def _sample_data(self) -> Tuple[List[Experience], Dict, List[Dict]]:
        """Sample a batch of experiences.

        Returns:
            List[Experience]: A batch of experiences.
            Dict: Metrics of the sampling step.
            List[Dict]: A list of representative samples for logging.
        """
        batch, metrics, repr_samples = await self.sample_strategy.sample(self.train_step_num + 1)
        metrics["sample/task_count"] = len(set(exp.eid.tid for exp in batch))
        return batch, metrics, repr_samples

    async def need_sync(self) -> bool:
        """Whether to sync the model weight."""
        if self.sync_style in {SyncStyle.FIXED, SyncStyle.TRAINER_DRIVEN, SyncStyle.FULLY_ASYNC}:
            return (
                self.last_sync_step != self.train_step_num
                and self.train_step_num % self.sync_interval == 0
            )
        else:  # explorer driven
            # for memory & checkpoint; TODO: apply to nccl sync
            if self.last_sync_step == self.train_step_num and self.sync_method != SyncMethod.NCCL:
                await self.synchronizer.notify_no_new_model_state_dict.remote()
                return False
            return await self.synchronizer.explorer_requires_sync.remote()

    def need_save(self) -> bool:
        """Whether to save the checkpoint."""
        return self.save_interval > 0 and self.train_step_num % self.save_interval == 0

    async def sync_weight(self) -> Dict:
        """Sync the model weight."""
        self.logger.info(f"Trainer sync_weights at step {self.train_step_num} started.")
        metrics = {}
        if self.last_sync_time is not None:
            metrics["time/trainer_sync_interval"] = time.time() - self.last_sync_time
        with Timer(metrics, "time/sync_weight"):
            if self.sync_method == SyncMethod.NCCL:
                result = await self.synchronizer.ready_to_nccl_sync.remote(
                    "trainer", self.train_step_num
                )
                if result is None:
                    self.logger.warning(
                        "NCCL weight sync skipped: Explorer has stopped or is unreachable."
                    )
                else:
                    try:
                        self.engine.sync_weight_nccl()
                    except Exception:
                        self.logger.warning(
                            "NCCL weight sync failed (Explorer may have exited);"
                            f" continuing with stale weights:\n{traceback.format_exc()}"
                        )
            elif self.train_step_num > 0:
                if self.sync_method == SyncMethod.CHECKPOINT:
                    await self.engine.save_state_dict()
                elif self.sync_method == SyncMethod.MEMORY:
                    await self.engine.upload_state_dict()
            self.last_sync_step = self.train_step_num
            self.last_sync_time = time.time()
        self.logger.info(f"Trainer sync_weights at step {self.train_step_num} finished.")
        return metrics

    def _log_experiences(self, samples: List[Dict]) -> None:
        self._sample_exps_to_log.extend(samples)
        if self.train_step_num % self.config.synchronizer.sync_interval == 0:
            self.monitor.log_table(
                "rollout_examples", pd.DataFrame(self._sample_exps_to_log), self.train_step_num
            )
            self._sample_exps_to_log.clear()

    async def save_checkpoint(
        self, block_until_saved: bool = False, save_as_hf: bool = False
    ) -> Dict:
        metrics = {}
        with Timer(metrics, "time/save_checkpoint"):
            self.logger.info(f"Saving checkpoint at step {self.train_step_num}...")
            await self.engine.save_checkpoint(
                block_until_saved=block_until_saved, save_as_hf=save_as_hf
            )
            self.state.save_trainer(
                current_step=self.train_step_num,
                sample_strategy_state=self.sample_strategy.state_dict(),
            )
        return metrics

    async def shutdown(self) -> None:
        self.monitor.close()

    @property
    def train_step_num(self) -> int:
        """Get the current training step number."""
        return self.engine.train_step_num

    async def is_alive(self) -> bool:
        """Check if the trainer is alive."""
        return True

    @classmethod
    def get_actor(cls, config: Config):
        """Get a Ray actor for the trainer."""
        return (
            ray.remote(cls)
            .options(name=config.trainer.name, namespace=config.ray_namespace)
            .remote(config)
        )


class TrainEngineWrapper(ABC):
    """A wrapper class to wrap various training engines."""

    @abstractmethod
    async def prepare(self) -> None:
        """Do some preparation before training started."""

    @property
    @abstractmethod
    def train_step_num(self) -> int:
        """Get the current training step number."""

    @abstractmethod
    async def train_step(self, batch_exps: List[Experience]) -> Dict:
        """Training one step.

        Args:
            batch_exps (List[Experience]): A batch of experiences to train.

        Returns:
            Dict: Metrics of the training step.
        """

    @abstractmethod
    async def save_checkpoint(
        self, block_until_saved: bool = False, save_as_hf: bool = False
    ) -> None:
        """Save the whole checkpoint (Including model, optimizer, and other states)."""

    async def wait_for_save(self) -> None:
        """Wait for any pending background save operations to complete.

        Default implementation is a no-op. Override in subclasses that use
        background save threads to ensure the checkpoint iteration file is
        written before the trainer exits.
        """
        pass

    @abstractmethod
    def sync_weight_nccl(self) -> None:
        """Sync the model weight by NCCL. (For `NCCL` sync method)"""

    @abstractmethod
    async def upload_state_dict(self) -> None:
        """Upload the state dict to Synchronizer. (For `MEMORY` sync method)"""

    @abstractmethod
    async def save_state_dict(self) -> None:
        """Only save the model state dict for Synchronizer.  (For `CHECKPOINT` sync method)"""

    @abstractmethod
    async def get_weight_sync_info(self) -> Optional[Tuple[str, int, List]]:
        """Get (master_address, master_port, state_dict_meta) for NCCL group setup."""

    @abstractmethod
    async def setup_weight_sync_group(
        self,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        timeout: int,
    ) -> None:
        """Join the NCCL weight sync group."""

    @abstractmethod
    async def teardown_weight_sync_group(self) -> None:
        """Tear down the NCCL weight sync group."""


def is_verl_legacy() -> bool:
    """Return True when the installed verl package is < 0.8 (legacy backend)."""
    from packaging.version import parse as parse_version

    try:
        import verl

        ver = getattr(verl, "__version__", "0.0.0")
    except ImportError:
        return False
    return parse_version(ver) < parse_version("0.8.0")


def get_latest_hf_checkpoint_path(config: Config) -> str | None:
    """Return the latest HF checkpoint path for a verl trainer config."""
    if config.trainer.trainer_type != "verl":
        raise ValueError("This function is only for verl trainer.")

    from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

    checkpoint_dir = find_latest_ckpt_path(config.checkpoint_job_dir)
    if checkpoint_dir is None:
        return None

    hf_checkpoint_dir = os.path.join(checkpoint_dir, "actor", "huggingface")
    if not os.path.exists(hf_checkpoint_dir):
        return None
    return hf_checkpoint_dir


def get_trainer_wrapper(config: Config) -> TrainEngineWrapper:
    """Get a trainer wrapper."""
    if config.trainer.trainer_type == "verl":
        if is_verl_legacy():
            from trinity.trainer.verl_legacy.verl_trainer import VerlPPOTrainerWrapper

            return VerlPPOTrainerWrapper(config)
        else:
            from trinity.trainer.verl.trainer import VERLTrainer

            return VERLTrainer(config)
    elif config.trainer.trainer_type == "tinker":
        from trinity.trainer.tinker.tinker_trainer import TinkerTrainerWrapper

        return TinkerTrainerWrapper(config)
    else:
        raise NotImplementedError
