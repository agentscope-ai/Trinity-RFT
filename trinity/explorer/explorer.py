# -*- coding: utf-8 -*-
"""The explorer module"""

from __future__ import annotations

import asyncio
import math
import os
import time
import traceback
from collections import deque
from typing import List, Optional

import ray
import torch

from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.task_scheduler import get_taskset_scheduler
from trinity.common.config import Config
from trinity.common.constants import RunningStatus, SyncMethod, SyncStyle
from trinity.common.models.allocator import Allocator
from trinity.common.models.model import ModelWrapper
from trinity.explorer.rollout_coordinator import RolloutCoordinator
from trinity.manager.state_manager import StateManager
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.annotations import Experimental
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR
from trinity.utils.plugin_loader import load_plugins
from trinity.utils.timer import Timer


class Explorer:
    """Responsible for exploring the taskset."""

    def __init__(self, config: Config):
        self.logger = get_logger(config.explorer.name, in_ray_actor=True)
        load_plugins()
        self.state = StateManager(
            path=config.checkpoint_job_dir, explorer_name=config.explorer.name, config=config
        )
        explorer_state = self.state.load_explorer()
        self.explore_step_num = explorer_state.get("latest_iteration", 0)
        self.last_monitored_step = self.explore_step_num
        self.synchronizer = Synchronizer.get_actor(config)
        self.config = config
        self.model_type = config.explorer.rollout_model.engine_type
        self.model_allocator = Allocator(config.explorer)
        self.models: List[ModelWrapper] = []
        self.auxiliary_models: List[List[ModelWrapper]] = []
        self.taskset = (
            get_taskset_scheduler(explorer_state=explorer_state, config=config)
            if self.config.mode not in {"bench", "serve"}
            else None
        )
        self.monitor = MONITOR.get(self.config.monitor.monitor_type)(
            project=self.config.project,
            group=self.config.group,
            name=self.config.name,
            role=self.config.explorer.name,
            config=config,
        )
        self.detailed_stats = config.monitor.detailed_stats
        if config.explorer.over_rollout.ratio > 0.0:
            self.min_wait_num = math.ceil(
                config.buffer.batch_size * (1 - config.explorer.over_rollout.ratio)
            )
            self.logger.info(
                f"Over rollout is enabled. Explorer will only wait for {self.min_wait_num} tasks in each step."
            )
        else:
            self.min_wait_num = None
        self.rollout_coordinator = None
        self.use_nccl_sync = self.config.synchronizer.sync_method == SyncMethod.NCCL
        self.pending_eval_tasks = deque()

        # For checkpoint weights update
        # Use explorer to periodically load the latest model weights and
        # boradcast to all rollout models
        self.enable_lora = self.config.explorer.rollout_model.enable_lora
        self.model_version = -1
        self.sync_offset = config.synchronizer.sync_offset
        self.sync_interval = config.synchronizer.explorer_sync_interval
        self.sync_method = config.synchronizer.sync_method
        self.sync_style = config.synchronizer.sync_style
        self.eval_start_time = None
        self.explore_start_time = None
        # FULLY_ASYNC: background sync watcher task
        self._async_watch_stopped: bool = False
        self._async_watch_task: Optional[asyncio.Task] = None
        # Used to drain the oldest batch when the in-flight window is full and to drain
        # all remaining batches at sync time via finish_current_steps().
        self._inflight_train_steps: deque[int] = deque()
        self.logger.info("Finished initializing Explorer.")

    async def _wait_for_models_ready(self) -> None:
        """Wait until rollout models are created before using them."""
        if self.models:
            return

        timeout = max(self.config.synchronizer.sync_timeout, 1)
        deadline = time.monotonic() + timeout
        self.logger.info("Waiting for rollout models to finish initialization.")
        while not self.models:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for rollout models before initializing the weight sync group."
                )
            await asyncio.sleep(0.1)
        self.logger.info("Rollout models are ready. Continue weight sync initialization.")

    async def setup_weight_sync_group(
        self,
        master_address: str,
        master_port: int,
        world_size: int = None,
        group_name: str = None,
        timeout: int = None,
    ):
        await self._wait_for_models_ready()
        base_offset = 1 if self.use_nccl_sync else 0
        gpu_per_engine: int = self.config.explorer.rollout_model.gpu_per_engine
        world_size = world_size or len(self.models) * gpu_per_engine + base_offset
        timeout = timeout or self.config.synchronizer.sync_timeout
        group_name = group_name or self.config.synchronizer.group_name
        self.logger.info(
            f"Initialize process group for weight synchronization, "
            f"master_address={master_address}, master_port={master_port}, "
            f"world_size={world_size}, rank_offset={base_offset}"
        )

        refs = [
            model.init_process_group(
                master_address=master_address,
                master_port=master_port,
                rank_offset=i * gpu_per_engine + base_offset,
                world_size=world_size,
                group_name=group_name,
                timeout=timeout,
            )
            for i, model in enumerate(self.models)
        ]
        await asyncio.gather(*refs)

    async def set_state_dict_meta(self, state_dict_meta: List):
        """Set the state_dict meta on all model workers for NCCL weight sync.

        Must be called after setup_weight_sync_group and before the first
        sync_model_weights call.
        """
        refs = [model.set_state_dict_meta(state_dict_meta) for model in self.models]
        await asyncio.gather(*refs)

    async def teardown_weight_sync_group(self):
        """Destroy the NCCL process group on all model workers."""
        refs = [model.teardown_process_group() for model in self.models]
        await asyncio.gather(*refs)

    async def setup_model_level_weight_sync_group(self):
        """Setup process group for each model, only used in serve mode."""
        await self._wait_for_models_ready()
        refs = []
        world_size = self.config.explorer.rollout_model.gpu_per_engine
        for model in self.models:
            master_address, master_port = await model.get_available_address_async(random_port=True)
            self.logger.info(
                f"Initialize process group for model weight synchronization, "
                f"master_address={master_address}, master_port={master_port}, "
                f"world_size={world_size}"
            )
            refs.append(
                model.init_process_group(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=0,
                    world_size=world_size,
                    group_name=self.config.synchronizer.group_name,
                    timeout=self.config.synchronizer.sync_timeout,
                )
            )
        await asyncio.gather(*refs)

    async def _checkpoint_weights_update(self, step_num: Optional[int] = None) -> int:
        self.logger.info(f"Start to update model weights from checkpoint at step {step_num}.")
        if step_num is None:
            step_num = await self.synchronizer.get_latest_model_version.remote()
        if step_num is None or step_num <= self.model_version:
            self.logger.warning(
                f"No new checkpoint found for step {step_num}. Current model version: {self.model_version}."
            )
            return self.model_version
        await asyncio.gather(
            *[
                model.sync_model_weights(
                    step_num,
                    self.config.synchronizer.sync_method,
                    timeout=self.config.synchronizer.sync_timeout,
                )
                for model in self.models
            ]
        )
        self.logger.info(f"Model weights updated to checkpoint at step {step_num}.")
        return step_num  # type: ignore

    async def _pull_latest_weights(self):
        self.logger.info("Start to pull latest model weights.")
        new_version = await self.synchronizer.wait_new_model_state_dict.remote(
            current_version=self.model_version,
        )
        if new_version > self.model_version:
            if self.model_version != -1 or new_version > 0:
                self.logger.info(f"New model weights version: {new_version}")
                await asyncio.gather(
                    *[
                        model.sync_model_weights(
                            new_version,
                            self.config.synchronizer.sync_method,
                            timeout=self.config.synchronizer.sync_timeout,
                        )
                        for model in self.models
                    ]
                )
            self.model_version = new_version
        else:
            self.logger.warning(
                f"No new model weights found, current version: {self.model_version}"
            )

    async def _nccl_weights_update(self):
        new_version = await self.synchronizer.ready_to_nccl_sync.remote(
            "explorer", self.model_version
        )
        if new_version is None:
            self.logger.info("Trainer is not ready to sync weight. Skipping sync weight.")
            return
        self.model_version = new_version
        await asyncio.gather(
            *[
                model.sync_model_weights(
                    self.model_version,
                    self.config.synchronizer.sync_method,
                    timeout=self.config.synchronizer.sync_timeout,
                )
                for model in self.models
            ]
        )

    async def prepare(self) -> None:
        """Preparation before running."""
        try:
            # make sure all rollout models are ready
            self.models, self.auxiliary_models = await self.model_allocator.create_all_models()
            self.logger.info("All models are ready.")
            if not self.use_nccl_sync and self.model_type not in {"tinker", "external"}:
                if self.config.mode == "serve":
                    # In serving mode, each engine will setup its own process group
                    await self.setup_model_level_weight_sync_group()
                else:
                    master_address, master_port = await self.models[0].get_available_address_async(
                        random_port=True
                    )
                    await self.setup_weight_sync_group(master_address, master_port)

            self.rollout_coordinator = RolloutCoordinator.get_actor(self.config)
            await self.rollout_coordinator.prepare.remote()
            self.logger.info("Rollout coordinator is ready.")
            if self.config.explorer.eval_on_startup and self.explore_step_num == 0:
                await self.eval()

            await self.synchronizer.set_explorer_status.remote(RunningStatus.RUNNING)
            if self.sync_style == SyncStyle.FULLY_ASYNC:
                self._async_watch_task = asyncio.create_task(self._watch_trainer_sync_signal())
                self.logger.info("Trainer sync watcher task started.")
            self.logger.info("Explorer is ready.")
        except Exception as e:
            self.logger.error(f"Error during explorer preparation: {traceback.format_exc()}")
            await self.shutdown()
            raise e

    async def get_weight(self, name: str) -> torch.Tensor:
        """Get the weight of the loaded model (For checkpoint weights update)."""
        return self.state_dict[name]

    async def explore(self) -> str:
        """
        The timeline of the exploration process:
                 | <--------------------------------- one period -------------------------------------> |
        explorer | <---------------- step_1 --------------> |                                           |
                 |   | <---------------- step_2 --------------> |                                       |
                 |      ...                                                                             |
                 |          | <---------------- step_n ---------------> |                               |
                 |                  | <---------------------- eval --------------------> | <-- sync --> |
                 |--------------------------------------------------------------------------------------|
        trainer  | <-- idle --> | <-- step_1 --> | <-- step_2 --> | ... | <-- step_n --> | <-- sync --> |
        """
        while True:
            try:
                self.logger.info(f"Explore step {self.explore_step_num + 1} started.")
                explore_contionue = await self.explore_step()
                if not explore_contionue:
                    # TODO: support eval on last checkpoint
                    break
                if self.need_eval():
                    await self.eval()
                if await self.need_sync():
                    await self.sync_weight()
            except Exception:
                self.logger.error(f"Error in Explorer: {traceback.format_exc()}")
                break
        self.logger.info(
            f"--------------------\n> Explorer ({self.config.explorer.name}) finished.\n--------------------"
        )
        return self.config.explorer.name

    async def explore_step(self) -> bool:
        if self.explore_start_time is None:
            self.explore_start_time = time.time()
        try:
            tasks = await self.taskset.read()
        except StopAsyncIteration:
            self.logger.warning("No more tasks to explore. Stop exploring.")
            await self.finish_current_steps()
            await self.save_checkpoint()
            await self.synchronizer.set_explorer_status.remote(
                RunningStatus.STOPPED,
                old_status=RunningStatus.RUNNING,
            )
            await self.shutdown()
            return False
        self.explore_step_num += 1
        if self.rollout_coordinator is None:
            return False
        # FULLY_ASYNC: when the in-flight window is full, wait for the oldest batch to
        # finish before submitting a new one.  This provides rolling backpressure without
        # waiting for a sync-interval boundary.
        if (
            self.sync_style == SyncStyle.FULLY_ASYNC
            and len(self._inflight_train_steps) >= self.config.explorer.max_inflight_batches
        ):
            oldest_step = self._inflight_train_steps.popleft()
            self.logger.debug(
                f"FULLY_ASYNC: at capacity, draining oldest batch (step {oldest_step})."
            )
            await self._finish_explore_step(step=oldest_step)
            self.last_monitored_step = oldest_step
        await self.rollout_coordinator.submit_batch.remote(
            batch_id=self.explore_step_num,
            tasks=tasks,
            batch_type="train",
            min_wait_num=self.min_wait_num,
        )
        if self.sync_style == SyncStyle.FULLY_ASYNC:
            self._inflight_train_steps.append(self.explore_step_num)
        return True

    async def finish_current_steps(self) -> None:
        if self.rollout_coordinator is not None:
            await self._finish_steps(self.last_monitored_step + 1, self.explore_step_num)
            self.last_monitored_step = self.explore_step_num

    async def _watch_trainer_sync_signal(self) -> None:
        """Background polling task for FULLY_ASYNC and TRAINER_DRIVEN+NCCL modes.

        Polls the Synchronizer at a fixed interval and sets _pending_async_sync when
        the Trainer signals readiness for weight synchronization.  The caller-side
        polling approach keeps the Synchronizer methods simple and non-blocking.
        """
        while not self._async_watch_stopped:
            try:
                if self.sync_method == SyncMethod.NCCL:
                    ready = await self.synchronizer.trainer_requires_weight_sync.remote()
                    if ready is None:
                        break  # trainer stopped
                    if ready:
                        await self._nccl_weights_update()
                        await self.save_checkpoint()
                else:
                    # Non-NCCL (FULLY_ASYNC only): detect an incremented model version.
                    new_version = await self.synchronizer.get_latest_model_version.remote()
                    if new_version > self.model_version:
                        await self._pull_latest_weights()
                        await self.save_checkpoint()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.error(f"Trainer sync watcher task error:\n{traceback.format_exc()}")
            await asyncio.sleep(0.5)

    async def need_sync(self) -> bool:
        if self.sync_style == SyncStyle.FULLY_ASYNC:
            # Fully async mode calls weight sync directly, without need_sync check
            return False
        if self.explore_step_num <= self.sync_offset:
            return False
        if (self.explore_step_num - self.sync_offset) % self.sync_interval == 0:
            await self.finish_current_steps()
            if self.sync_style == SyncStyle.TRAINER_DRIVEN and self.sync_method == SyncMethod.NCCL:
                require_sync = bool(await self.synchronizer.trainer_requires_weight_sync.remote())
            else:
                require_sync = True
            return require_sync
        return False

    def need_eval(self) -> bool:
        return self.explore_step_num % self.config.explorer.eval_interval == 0

    async def eval(self):
        """Evaluation on all evaluation data samples."""
        if len(self.config.buffer.explorer_input.eval_tasksets) == 0:
            self.logger.info("No evaluation data samples. Skip evaluation.")
            return

        self.eval_start_time = time.time()
        self.logger.info(f"Evaluation at step {self.explore_step_num} started.")

        if self.config.buffer.explorer_input.default_eval_workflow_type:
            self.logger.info(
                f"Use '{self.config.buffer.explorer_input.default_eval_workflow_type}' for evaluation."
            )

        for eval_taskset_config in self.config.buffer.explorer_input.eval_tasksets:
            self.logger.info(
                f"Evaluation on {eval_taskset_config.name} at step {self.explore_step_num} started."
            )
            eval_taskset = get_buffer_reader(eval_taskset_config)
            eval_batch_id = f"{self.explore_step_num}/{eval_taskset_config.name}"
            self.pending_eval_tasks.append((self.explore_step_num, eval_taskset_config.name))
            eval_tasks = []
            while True:
                try:
                    eval_tasks.extend(await eval_taskset.read())
                except StopAsyncIteration:
                    break
            assert (
                self.rollout_coordinator is not None
            ), "Rollout coordinator must be prepared first."
            await self.rollout_coordinator.submit_batch.remote(
                batch_id=eval_batch_id,
                tasks=eval_tasks,
                batch_type="eval",
            )

    async def benchmark(self) -> bool:
        """Benchmark the model checkpoints."""
        # benchmark on the latest checkpoint
        if self.config.explorer.bench_on_latest_checkpoint:
            self.explore_step_num = await self._checkpoint_weights_update()
            await self.eval()
            await self._finish_eval_step(prefix="bench")
            return True

        # benchmark on base model
        if self.config.explorer.eval_on_startup:
            await self._finish_eval_step(prefix="bench")

        # benchmark on all checkpoints
        all_ckp_steps = sorted(
            [
                int(ckp.split("global_step_")[-1])
                for ckp in os.listdir(self.config.checkpoint_job_dir)
                if os.path.isdir(os.path.join(self.config.checkpoint_job_dir, ckp))
                and ckp.startswith("global_step_")
            ]
        )
        for step_num in all_ckp_steps:
            if step_num <= self.explore_step_num:
                continue
            self.explore_step_num = await self._checkpoint_weights_update(step_num=step_num)
            await self.eval()
            await self._finish_eval_step(prefix="bench")
        return True

    async def save_checkpoint(self) -> None:
        # save explore checkpoint
        self.state.save_explorer(
            current_step=self.explore_step_num,
            taskset_states=self.taskset.state_dict() if self.taskset else [],
        )

    async def sync_weight(self) -> None:
        """Synchronize model weights."""
        # call this method before training start to load the latest model weights
        if self.rollout_coordinator is not None and self.explore_step_num == 0:
            await self._finish_eval_step(step=0)

        self.logger.info(f"Explorer sync_weights at step {self.explore_step_num} started.")
        if self.use_nccl_sync:
            await self._nccl_weights_update()
        else:  # pull weights from Synchronizer
            await self._pull_latest_weights()
        self.logger.info(
            f"Explorer sync_weights at step {self.explore_step_num} finished, model version = {self.model_version}."
        )

        await self.save_checkpoint()

    async def _finish_steps(self, start_step: int, end_step: int) -> None:
        for step in range(start_step, end_step + 1):
            self.logger.info(f"Waiting for step {step}")
            await self._finish_explore_step(step=step)
            await self._finish_eval_step(step=step)

        # Record the time: read_task + explore_step (>=1) + eval (if any)
        if self.explore_start_time is not None:
            metric = {"explore/time/sync_interval": time.time() - self.explore_start_time}
            self.explore_start_time = None
            if self.monitor is not None:
                self.monitor.log(metric, step=end_step)

    async def _finish_explore_step(self, step: int) -> None:
        if self.rollout_coordinator is None:
            return

        metric = {"rollout/model_version": self.model_version}
        with Timer(metric, "explorer/time/wait_explore_step"):
            result = await self.rollout_coordinator.finalize_train_batch.remote(step)
        if self.taskset is not None:
            self.taskset.feedback(result["metrics"])
        metric.update(result["metrics"])
        if result["finished_task_count"] > 0 and self.monitor is not None:
            self.monitor.log(metric, step=step)

    async def _finish_eval_step(self, step: Optional[int] = None, prefix: str = "eval") -> None:
        if not self.pending_eval_tasks:
            return
        step = step or self.explore_step_num
        metric = {}
        while self.pending_eval_tasks:
            eval_step, eval_task_name = self.pending_eval_tasks[0]
            if eval_step != step:
                return
            self.pending_eval_tasks.popleft()
            assert (
                self.rollout_coordinator is not None
            ), "Rollout coordinator must be prepared first."
            result = await self.rollout_coordinator.finalize_eval_batch.remote(
                f"{step}/{eval_task_name}"
            )
            batch_metrics = result["metrics"]
            if prefix != "eval":
                batch_metrics = {
                    key.replace("eval/", f"{prefix}/", 1) if key.startswith("eval/") else key: value
                    for key, value in batch_metrics.items()
                }
            metric.update(batch_metrics)
        if self.eval_start_time is not None:
            metric.update({f"time/{prefix}": time.time() - self.eval_start_time})
            self.eval_start_time = None
        if self.monitor is not None:
            self.monitor.log(metric, step)

    async def shutdown(self) -> None:
        # Stop the FULLY_ASYNC background watcher before tearing down models.
        self._async_watch_stopped = True
        if self._async_watch_task is not None and not self._async_watch_task.done():
            self._async_watch_task.cancel()
            try:
                await self._async_watch_task
            except asyncio.CancelledError:
                pass
        if self.rollout_coordinator:
            await self.rollout_coordinator.shutdown.remote()
            self.rollout_coordinator = None
        if self.monitor:
            self.monitor.close()
            self.monitor = None
        handlers = []
        for model in self.models:
            handlers.append(model.shutdown())
        for auxiliary_model_list in self.auxiliary_models:
            for model in auxiliary_model_list:
                handlers.append(model.shutdown())
        await asyncio.gather(*handlers)
        self.logger.info(
            f"Explorer ({self.config.explorer.name}) shutdown successfully at step {self.explore_step_num}."
        )

    async def is_alive(self) -> bool:
        """Check if the explorer is alive."""
        return True

    @Experimental
    async def serve(self) -> None:
        """Run the explorer in serving mode.

        In serving mode, the explorer starts an OpenAI compatible server to handle requests.
        Agent applications can be deployed separately and interact with the explorer via the API.


        .. code-block:: python

            import openai


            client = openai.OpenAI(
                base_url=f"{explorer_server_url}/v1",
                api_key="EMPTY",
            )
            response = client.chat.completions.create(
                model=config.model.model_path,
                messages=[{"role": "user", "content": "Hello!"}]
            )
        """
        from trinity.explorer.proxy.service import ExplorerService

        self.service = ExplorerService(
            self,
            listen_address=self.config.explorer.listen_address,
            port=self.config.explorer.proxy_port,
        )
        await self.service.serve()
        self.server_url = f"http://{ray.util.get_node_ip_address()}:{self.service.port}"
        self.logger.info(
            "======================================================\n"
            f"Starting Trinity Service on {self.server_url}\n"
            "======================================================"
        )
        self.state.save_explorer_server_url(self.server_url)
        while True:
            await asyncio.sleep(self.config.explorer.service_status_check_interval)
            # get the latest checkpoint
            model_version = await self.synchronizer.get_latest_model_version.remote()
            self.service.set_latest_model_version(model_version)

    @classmethod
    def get_actor(cls, config: Config):
        """Get a Ray actor for the explorer."""
        return (
            ray.remote(cls)
            .options(
                name=config.explorer.name,
                namespace=config.ray_namespace,
            )
            .remote(config)
        )
