# -*- coding: utf-8 -*-
"""Test cases for Synchronizer modules."""

import asyncio
import multiprocessing
import os
import shutil
import time
import unittest
from copy import deepcopy
from datetime import datetime
from multiprocessing import Process
from typing import Dict, List

import ray
from parameterized import parameterized_class

from tests.tools import (
    TensorBoardParser,
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.cli.launcher import both, explore, train
from trinity.common.config import Config, ExperienceBufferConfig
from trinity.common.constants import RunningStatus, StorageType, SyncMethod, SyncStyle
from trinity.explorer.explorer import Explorer
from trinity.trainer.trainer import Trainer
from trinity.utils.log import get_logger

logger = get_logger(__name__)


def trainer_monkey_patch(sample_time_list: List[int], train_step_time_list: List[int]):
    async def new_sample_data(self: Trainer):
        self.logger.info(f"Sample data for step {self.train_step_num + 1} started.")
        await asyncio.sleep(sample_time_list[self.engine.global_steps - 1])
        self.logger.info(f"Sample data for step {self.train_step_num + 1} finished.")
        return [], {}, []

    async def new_train_step(self: Trainer, exps) -> Dict:
        self.engine.global_steps += 1
        self.logger.info(f"Training at step {self.engine.global_steps} started.")
        time.sleep(train_step_time_list[self.engine.global_steps - 1])
        metrics = {"actor/step": self.engine.global_steps}
        self.logger.info(f"Training at step {self.engine.global_steps} finished.")
        return metrics

    Trainer.train_step = new_train_step
    Trainer._sample_data = new_sample_data


def explorer_monkey_patch(explore_step_time_list: List[int]):
    async def new_explore_step(self: Explorer):
        if self.explore_step_num >= len(explore_step_time_list):
            await self.finish_current_steps()
            await self.save_checkpoint()
            await self.synchronizer.set_explorer_status.remote(
                RunningStatus.STOPPED,
                old_status=RunningStatus.RUNNING,
            )
            await self.shutdown()
            return False
        self.explore_step_num += 1
        return True

    async def new_finish_explore_step(self: Explorer, step: int, model_version: int) -> None:
        metric = {"rollout/model_version": model_version}
        await asyncio.sleep(explore_step_time_list[step - 1])
        self.monitor.log(metric, step=step)

    Explorer.explore_step = new_explore_step
    Explorer._finish_explore_step = new_finish_explore_step


def run_trainer(
    config: Config, sample_time_list: List[int], train_step_time_list: List[int]
) -> None:
    ray.init(ignore_reinit_error=True, namespace=config.ray_namespace)
    trainer_monkey_patch(sample_time_list, train_step_time_list)
    train(config)
    ray.shutdown()


def run_explorer(config: Config, explore_step_time_list: List[int]) -> None:
    ray.init(ignore_reinit_error=True, namespace=config.ray_namespace)
    explorer_monkey_patch(explore_step_time_list)
    explore(config)
    ray.shutdown()


def run_both(
    config: Config,
    sample_time_list: List[int],
    train_step_time_list: List[int],
    explore_step_time_list: List[int],
) -> None:
    ray.init(ignore_reinit_error=True, namespace=config.ray_namespace)
    trainer_monkey_patch(sample_time_list, train_step_time_list)
    explorer_monkey_patch(explore_step_time_list)
    both(config)
    ray.shutdown()


class BaseTestSynchronizer(unittest.TestCase):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        self.process_list = []

        self.config = get_template_config()
        self.config.project = "unittest"
        self.config.name = f"test_synchronizer_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.buffer.total_epochs = 1
        self.config.buffer.batch_size = 4
        self.config.cluster.gpu_per_node = 2
        self.config.cluster.node_num = 1
        self.config.model.model_path = get_model_path()
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        experience_buffer = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE.value,
        )
        self.config.buffer.trainer_input.experience_buffer = deepcopy(experience_buffer)
        self.config.synchronizer.sync_method = getattr(self, "sync_method", SyncMethod.NCCL)
        self.config.synchronizer.sync_style = self.sync_style
        self.config.synchronizer.sync_interval = 2
        self.config.monitor.monitor_type = "tensorboard"

        assert len(self.sample_time_list) == len(self.train_step_time_list), (
            f"sample_time_list and train_step_time_list must have equal lengths, "
            f"but got {len(self.sample_time_list)} and {len(self.train_step_time_list)}"
        )
        self.config.trainer.total_steps = len(self.sample_time_list)
        self.config.trainer.save_interval = 100
        self.config.buffer.train_batch_size = 4

        self.config.explorer.rollout_model.engine_num = 1
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.buffer.explorer_output = deepcopy(experience_buffer)

    def _start_process(self, target_func, *args) -> Process:
        process = Process(target=target_func, args=args)
        process.start()
        self.process_list.append(process)
        return process

    def start_train_process(self, config: Config) -> Process:
        return self._start_process(
            run_trainer, config, self.sample_time_list, self.train_step_time_list
        )

    def start_explore_process(self, config: Config, explore_step_time_list: List[int]) -> Process:
        return self._start_process(run_explorer, config, explore_step_time_list)

    def start_both_process(self, config: Config, explore_step_time_list: List[int]) -> Process:
        return self._start_process(
            run_both,
            config,
            self.sample_time_list,
            self.train_step_time_list,
            explore_step_time_list,
        )

    def join_process(self, process: Process, process_name: str, timeout: int = 200):
        process.join(timeout=timeout)
        if process.is_alive():
            self.fail(f"Process [{process_name}] is still alive after timeout")

    def wait_trainer_started(self, ray_namespace: str):
        ray.init(ignore_reinit_error=True)
        while True:
            try:
                synchronizer = ray.get_actor("synchronizer", namespace=ray_namespace)
                break
            except ValueError:
                print("waiting for trainer to start.")
                time.sleep(5)
        return synchronizer

    def _check_metrics(
        self,
        config: Config,
        module: str,
        metric_check_dict: Dict[str, float],
    ):
        parser = TensorBoardParser(os.path.join(config.monitor.cache_dir, "tensorboard", module))
        for metric_name, metric_value in metric_check_dict.items():
            metric_list = parser.metric_list(metric_name)
            self.assertEqual(parser.metric_max_step(metric_list[0]), metric_value)

    def tearDown(self):
        ray.shutdown(_exiting_interpreter=True)
        if os.path.exists(self.config.checkpoint_root_dir):
            shutil.rmtree(self.config.checkpoint_root_dir, ignore_errors=True)
        for process in self.process_list:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()


@parameterized_class(
    [
        {
            "sync_method": SyncMethod.NCCL,  # will be converted to CHECKPOINT
        },
        {
            "sync_method": SyncMethod.MEMORY,
        },
    ]
)
class TestSynchronizerExit(BaseTestSynchronizer):
    def setUp(self):
        self.sync_style = SyncStyle.FIXED
        self.sample_time_list = [2, 1, 2, 1, 2, 1, 2, 1]
        self.train_step_time_list = [2, 1, 2, 1, 2, 1, 2, 1]
        self.explore_step_time_list = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
        super().setUp()

    def test_synchronizer(self):
        trainer_config = deepcopy(self.config)
        trainer_config.cluster.gpu_per_node = 1
        trainer_config.mode = "train"
        trainer_config.check_and_update()

        explorer_config = deepcopy(self.config)
        explorer_config.mode = "explore"
        explorer_config.check_and_update()

        trainer_process = self.start_train_process(trainer_config)
        synchronizer = self.wait_trainer_started(trainer_config.ray_namespace)

        explorer_process = self.start_explore_process(explorer_config, self.explore_step_time_list)
        self.assertEqual(
            synchronizer, ray.get_actor("synchronizer", namespace=trainer_config.ray_namespace)
        )
        for _ in range(12):  # Wait for up to 60 seconds
            try:
                explorer = ray.get_actor("explorer", namespace=trainer_config.ray_namespace)
                ray.get(explorer.is_alive.remote())
                break
            except ValueError:
                print("waiting for explorer to start.")
                time.sleep(5)

        self.join_process(trainer_process, "trainer")
        self.assertEqual(
            synchronizer, ray.get_actor("synchronizer", namespace=trainer_config.ray_namespace)
        )

        self.join_process(explorer_process, "explorer")
        time.sleep(6)
        with self.assertRaises(ValueError):
            ray.get_actor("synchronizer", namespace=trainer_config.ray_namespace)


@parameterized_class(
    [
        {
            "sync_method": SyncMethod.CHECKPOINT,
            "sync_style": SyncStyle.FIXED,
            "sample_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "train_step_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "explore_step_time_lists": [
                [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
        },
        {
            "sync_method": SyncMethod.CHECKPOINT,
            "sync_style": SyncStyle.DYNAMIC_BY_EXPLORER,
            "sample_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "train_step_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "explore_step_time_lists": [
                [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
        },
        {
            "sync_method": SyncMethod.CHECKPOINT,
            "sync_style": SyncStyle.DYNAMIC_BY_TRAINER,
            "sample_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "train_step_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "explore_step_time_lists": [
                [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
        },
        {
            "sync_method": SyncMethod.MEMORY,
            "sync_style": SyncStyle.FIXED,
            "sample_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "train_step_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "explore_step_time_lists": [
                [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
        },
        {
            "sync_method": SyncMethod.MEMORY,
            "sync_style": SyncStyle.DYNAMIC_BY_EXPLORER,
            "sample_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "train_step_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "explore_step_time_lists": [
                [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
        },
        {
            "sync_method": SyncMethod.MEMORY,
            "sync_style": SyncStyle.DYNAMIC_BY_TRAINER,
            "sample_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "train_step_time_list": [2, 1, 2, 1, 2, 1, 2, 1],
            "explore_step_time_lists": [
                [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
        },
    ]
)
class TestStateDictBasedSynchronizer(BaseTestSynchronizer):
    def test_synchronizer(self):
        trainer_config = deepcopy(self.config)
        trainer_config.mode = "train"
        trainer_config.check_and_update()

        trainer_process = self.start_train_process(trainer_config)
        _ = self.wait_trainer_started(trainer_config.ray_namespace)

        explorer_configs, explorer_processes = [], []
        for i, explore_step_time_list in enumerate(self.explore_step_time_lists):
            explorer_config = deepcopy(self.config)
            explorer_config.mode = "explore"
            explorer_config.explorer.name = f"explorer_{i}"
            explorer_config.explorer.rollout_model.engine_num = 1
            explorer_config.explorer.rollout_model.tensor_parallel_size = 1
            explorer_config.check_and_update()
            explorer_configs.append(explorer_config)
            explorer_processes.append(
                self.start_explore_process(explorer_config, explore_step_time_list)
            )

        self.join_process(trainer_process, "trainer")
        for i, explore_process in enumerate(explorer_processes):
            self.join_process(explore_process, f"explorer_{i}")

        # check the tensorboard
        self._check_metrics(trainer_config, "trainer", {"actor": len(self.sample_time_list)})
        for i, (explorer_config, explore_step_time_list) in enumerate(
            zip(explorer_configs, self.explore_step_time_lists)
        ):
            self._check_metrics(
                explorer_config, f"explorer_{i}", {"rollout": len(explore_step_time_list)}
            )


@parameterized_class(
    [
        {
            "sync_style": SyncStyle.FIXED,
            "sample_time_list": [2, 2, 1, 1, 2, 2, 1, 1],
            "train_step_time_list": [2, 2, 1, 1, 2, 2, 1, 1],
            "explore_step_time_list": [1, 1, 2, 2, 1, 1, 2, 2],
        },
        {
            "sync_style": SyncStyle.DYNAMIC_BY_EXPLORER,
            "sample_time_list": [2, 2, 2, 1, 2, 1, 2, 1],
            "train_step_time_list": [2, 2, 2, 1, 2, 1, 2, 1],
            "explore_step_time_list": [0.5, 0.5, 0.5, 0.5, 0],
        },
        {
            "sync_style": SyncStyle.DYNAMIC_BY_TRAINER,
            "sample_time_list": [2, 2, 2, 1, 2, 1, 2, 1],
            "train_step_time_list": [2, 2, 2, 1, 2, 1, 2, 1],
            "explore_step_time_list": [0.5, 0.5, 0.5, 0.5, 0],
        },
    ],
)
class TestNCCLBasedSynchronizer(BaseTestSynchronizer):
    def test_synchronizer(self):
        self.config.mode = "both"
        self.config.check_and_update()

        # TODO: test more interval cases
        both_process = self.start_both_process(self.config, self.explore_step_time_list)
        self.join_process(both_process, "both")

        # check the tensorboard
        self._check_metrics(self.config, "trainer", {"actor": len(self.sample_time_list)})
        self._check_metrics(self.config, "explorer", {"rollout": len(self.explore_step_time_list)})
