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
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.pipelines.experience_pipeline import ExperiencePipeline
from trinity.buffer.task_scheduler import get_taskset_scheduler
from trinity.common.config import Config
from trinity.common.constants import (
    ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
    RunningStatus,
    SyncMethod,
    SyncStyle,
)
from trinity.common.models import create_inference_models
from trinity.explorer.scheduler import Scheduler
from trinity.manager.state_manager import StateManager
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.annotations import Experimental
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR, gather_eval_metrics, gather_metrics
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
        self.models, self.auxiliary_models = create_inference_models(config)
        self.experience_pipeline = self._init_experience_pipeline()
        self.taskset = (
            get_taskset_scheduler(explorer_state=explorer_state, config=config)
            if self.config.mode not in {"bench", "serve"}
            else None
        )
        self.scheduler = None
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
        self.use_nccl_sync = self.config.synchronizer.sync_method == SyncMethod.NCCL
        self.pending_eval_tasks = deque()

        # For checkpoint weights update
        # Use explorer to periodically load the latest model weights and
        # boradcast to all rollout models
        self.enable_lora = self.config.explorer.rollout_model.enable_lora
        self.model_version = -1
        self.sync_offset = config.synchronizer.sync_offset
        self.sync_interval = config.synchronizer.sync_interval
        self.sync_method = config.synchronizer.sync_method
        self.sync_style = config.synchronizer.sync_style
        self.eval_start_time = None
        self.explore_start_time = None
        self.logger.info("Finished initializing Explorer.")

    async def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        base_offset = 1 if self.use_nccl_sync else 0
        world_size = (
            len(self.models) * self.config.explorer.rollout_model.tensor_parallel_size + base_offset
        )
        self.logger.info(
            f"Initialize process group for weight synchronization, "
            f"master_address={master_address}, master_port={master_port}, "
            f"world_size={world_size}, rank_offset={base_offset}"
        )
        # TODO: save state_dict in models
        refs = [
            model.init_process_group.remote(
                master_address=master_address,
                master_port=master_port,
                rank_offset=i * self.config.explorer.rollout_model.tensor_parallel_size
                + base_offset,
                world_size=world_size,
                group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                explorer_name=self.config.explorer.name,
                timeout=self.config.synchronizer.sync_timeout,
                state_dict_meta=state_dict_meta,
            )
            for i, model in enumerate(self.models)
        ]
        await asyncio.gather(*refs)

    async def setup_model_level_weight_sync_group(self):
        """Setup process group for each model, only used in serve mode."""
        refs = []
        world_size = self.config.explorer.rollout_model.tensor_parallel_size
        for model in self.models:
            master_address, master_port = await model.get_available_address.remote()
            self.logger.info(
                f"Initialize process group for model weight synchronization, "
                f"master_address={master_address}, master_port={master_port}, "
                f"world_size={world_size}"
            )
            refs.append(
                model.init_process_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=0,
                    world_size=world_size,
                    group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                    explorer_name=self.config.explorer.name,
                    timeout=self.config.synchronizer.sync_timeout,
                )
            )
        await asyncio.gather(*refs)

    async def _checkpoint_weights_update(self, step_num: Optional[int] = None) -> int:
        self.logger.info(f"Start to update model weights from checkpoint at step {step_num}.")
        step_num = await self.synchronizer.set_model_state_dict_with_step_num.remote(step_num)
        await asyncio.gather(*[model.sync_model.remote(step_num) for model in self.models])
        self.logger.info(f"Model weights updated to checkpoint at step {step_num}.")
        return step_num  # type: ignore

    async def _pull_latest_weights(self):
        self.logger.info("Start to pull latest model weights.")
        new_version = await self.synchronizer.wait_new_model_state_dict.remote(
            current_version=self.model_version,
        )
        if new_version > self.model_version:
            if self.model_version != -1:
                self.logger.info(f"New model weights version: {new_version}")
                await asyncio.gather(
                    *[model.sync_model.remote(new_version) for model in self.models]
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
            *[model.sync_model.remote(self.model_version) for model in self.models]
        )

    async def prepare(self) -> None:
        """Preparation before running."""
        try:
            # prepare experience pipeline
            if self.experience_pipeline:
                await self.experience_pipeline.prepare.remote()
            self.logger.info("Experience pipeline is ready.")
            # make sure all rollout models are ready
            run_api_ref = [model.prepare.remote() for model in self.models]
            run_api_ref.extend(
                model.prepare.remote() for models in self.auxiliary_models for model in models
            )
            await asyncio.gather(*run_api_ref)
            self.logger.info("All models are ready.")

            if not self.use_nccl_sync and self.model_type != "tinker":
                if self.config.mode == "serve":
                    # In serving mode, each engine will setup its own process group
                    await self.setup_model_level_weight_sync_group()
                else:
                    master_address, master_port = await self.models[
                        0
                    ].get_available_address.remote()
                    await self.setup_weight_sync_group(master_address, master_port)
            if self.config.mode != "serve":
                self.scheduler = Scheduler(self.config, self.models, self.auxiliary_models)
                await self.scheduler.start()
            if self.config.explorer.eval_on_startup and self.explore_step_num == 0:
                await self.eval()

            await self.synchronizer.set_explorer_status.remote(RunningStatus.RUNNING)
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
            tasks = await self.taskset.read_async()
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
        self.scheduler.schedule(tasks, batch_id=self.explore_step_num)
        return True

    async def finish_current_steps(self) -> None:
        if self.scheduler:
            await self._finish_steps(
                self.last_monitored_step + 1, self.explore_step_num, self.model_version
            )
            self.last_monitored_step = self.explore_step_num

    async def need_sync(self) -> bool:
        if self.explore_step_num <= self.sync_offset:
            return False
        require_sync = False
        if (self.explore_step_num - self.sync_offset) % self.sync_interval == 0:
            await self.finish_current_steps()
            if (
                self.sync_style == SyncStyle.DYNAMIC_BY_TRAINER
                and self.sync_method == SyncMethod.NCCL
            ):
                require_sync = await (
                    self.synchronizer.get_trainer_status.remote() == RunningStatus.REQUIRE_SYNC
                )
            else:
                require_sync = True
        return require_sync

    def need_eval(self) -> bool:
        return self.explore_step_num % self.config.explorer.eval_interval == 0

    async def eval(self):
        """Evaluation on all evaluation data samples."""
        self.eval_start_time = time.time()
        if len(self.config.buffer.explorer_input.eval_tasksets) == 0:
            self.logger.warning("No evaluation data samples. Skip evaluation.")
            return
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
            while True:
                try:
                    data = await eval_taskset.read_async()
                    self.scheduler.schedule(data, batch_id=eval_batch_id)
                except StopAsyncIteration:
                    break

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
        if self.scheduler and self.explore_step_num == 0:
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

    async def _finish_steps(self, start_step: int, end_step: int, model_version: int) -> None:
        for step in range(start_step, end_step + 1):
            self.logger.info(f"Waiting for step {step}")
            await self._finish_explore_step(step=step, model_version=model_version)
            await self._finish_eval_step(step=step)

        # Record the time: read_task + explore_step (>=1) + eval (if any)
        if self.explore_start_time is not None:
            metric = {"time/explorer_sync_interval": time.time() - self.explore_start_time}
            self.explore_start_time = None
            self.monitor.log(metric, step=end_step)

    async def _finish_explore_step(self, step: int, model_version: int) -> None:
        metric = {"rollout/model_version": model_version}
        with Timer(metric, "time/wait_explore_step"):
            statuses, exps = await self.scheduler.get_results(
                batch_id=step, min_num=self.min_wait_num
            )
        pipeline_metrics = await self.experience_pipeline.process.remote(exps)
        self.taskset.feedback(pipeline_metrics)
        metric.update(pipeline_metrics)
        if statuses:
            metric.update(gather_metrics([status.metrics[0] for status in statuses], "rollout"))
            metric["rollout/finished_task_count"] = len(statuses)
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
            statuses, _ = await self.scheduler.get_results(batch_id=f"{step}/{eval_task_name}")
            metric[f"{prefix}/{eval_task_name}/finished_task_count"] = len(statuses)
            metric.update(
                gather_eval_metrics(
                    [status.metrics[0] for status in statuses],
                    f"{prefix}/{eval_task_name}",
                    detailed_stats=self.detailed_stats,
                )
            )
        if self.eval_start_time is not None:
            metric.update({"time/eval": time.time() - self.eval_start_time})
            self.eval_start_time = None
        self.monitor.log(metric, step)

    async def shutdown(self) -> None:
        if self.scheduler:
            await self.scheduler.stop()
            self.scheduler = None
        if self.experience_pipeline:
            await self.experience_pipeline.close.remote()
            # reserve `experience_pipeline.output` for trainer
            # TODO: refactor the lifecycle of buffer actor
            self._old_experience_pipeline = self.experience_pipeline
            self.experience_pipeline = None
        if self.monitor:
            self.monitor.close()
            self.monitor = None
        self.logger.info(
            f"Explorer ({self.config.explorer.name}) shutdown successfully at step {self.explore_step_num}."
        )

    async def is_alive(self) -> bool:
        """Check if the explorer is alive."""
        return True

    def _init_experience_pipeline(self) -> ray.actor.ActorHandle:
        """Init experience pipeline for the explorer."""
        if self.config.mode == "bench":
            return None
        node_id = ray.get_runtime_context().get_node_id()
        return (
            ray.remote(ExperiencePipeline)
            .options(
                name=f"{self.config.explorer.name}_pipeline",
                namespace=self.config.ray_namespace,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
            )
            .remote(self.config)
        )

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
