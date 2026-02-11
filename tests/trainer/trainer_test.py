"""Tests for trainer."""

import asyncio
import json
import math
import multiprocessing
import os
import shutil
import time
import unittest
from copy import deepcopy
from datetime import datetime
from logging import Logger
from typing import Dict
from unittest import mock

import ray
from parameterized import parameterized_class

from tests.tools import (
    RayUnittestBase,
    RayUnittestBaseAsync,
    TensorBoardParser,
    get_checkpoint_path,
    get_lora_config,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
    get_vision_language_model_path,
)
from trinity.buffer import get_buffer_reader
from trinity.cli.launcher import bench, both, convert, explore, run, serve, train
from trinity.common.config import (
    AlgorithmConfig,
    BufferConfig,
    Config,
    ExperienceBufferConfig,
    ExplorerInput,
    StageConfig,
    TaskSelectorConfig,
    TrainerInput,
)
from trinity.common.constants import (
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    StorageType,
    SyncMethod,
    SyncStyle,
)
from trinity.common.models.utils import get_checkpoint_dir_with_step_num
from trinity.explorer.proxy.client import TrinityClient
from trinity.manager.state_manager import StateManager
from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.tinker_trainer import TinkerTrainerWrapper


class BaseTrainerCase(RayUnittestBase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.buffer.total_epochs = 2
        self.config.buffer.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm_async"
        self.config.algorithm.repeat_times = 3
        self.config.project = "Trainer-unittest"
        self.config.name = f"trainer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.monitor.monitor_type = "tensorboard"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 2
        self.config.synchronizer.sync_method = SyncMethod.NCCL
        self.config.explorer.eval_interval = 4


@parameterized_class(
    ("strategy",),
    [
        ("fsdp",),
        ("megatron",),
    ],
)
class TestTrainerCountdown(BaseTrainerCase):
    def test_trainer(self):
        """Test the both and bench mode."""
        # test both mode
        self.config.model.rope_scaling = {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 16384,
        }
        self.config.model.rope_theta = 10000
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.buffer.explorer_input.taskset.task_selector = TaskSelectorConfig(
            selector_type="shuffle", seed=42
        )
        eval_tasksets = self.config.buffer.explorer_input.eval_tasksets
        eval_tasksets.append(get_unittest_dataset_config("countdown", "test"))
        eval_tasksets.append(get_unittest_dataset_config("copy_countdown", "test"))
        eval_tasksets[0].repeat_times = 4
        eval_tasksets[1].repeat_times = 4
        self.config.trainer.save_interval = 4
        self.config.trainer.save_hf_checkpoint = "never"
        if self.strategy == "megatron":
            self.config.trainer.trainer_strategy = "megatron"
        self.config.check_and_update()
        _trainer_config = self.config.trainer.trainer_config
        if self.strategy == "megatron":
            _trainer_config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = 2
            _trainer_config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = 2
            _trainer_config.critic.strategy = "megatron"
            _trainer_config.critic.megatron.tensor_model_parallel_size = 2
        _trainer_config.trainer.max_actor_ckpt_to_keep = 2
        _trainer_config.trainer.max_critic_ckpt_to_keep = 2
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        eval_metrics = parser.metric_list("eval")
        self.assertGreater(len(eval_metrics), 0)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 8)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 8)
        actor_kl_metrics = parser.metric_list("actor/kl")
        self.assertGreater(len(actor_kl_metrics), 0)
        actor_kl_loss = parser.metric_values("actor/kl_loss")
        self.assertEqual(actor_kl_loss[0], 0.0)
        critic_kl_metrics = parser.metric_list("critic/kl")
        self.assertGreater(len(critic_kl_metrics), 0)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 8)
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint
        checkpoint_step_4, _ = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
            step_num=4,
        )
        # check save lastest checkpoint
        checkpoint_step_8, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertGreater(len(os.listdir(os.path.join(checkpoint_step_4, "actor"))), 0)
        self.assertGreater(len(os.listdir(os.path.join(checkpoint_step_8, "actor"))), 0)
        hf_dir_step_4 = os.listdir(os.path.join(checkpoint_step_4, "actor", "huggingface"))
        hf_dir_step_8 = os.listdir(os.path.join(checkpoint_step_8, "actor", "huggingface"))
        self.assertGreater(len(hf_dir_step_4), 0)
        self.assertGreater(len(hf_dir_step_8), 0)
        self.assertNotIn("model.safetensors", hf_dir_step_4)
        self.assertNotIn("model.safetensors", hf_dir_step_8)
        # test checkpoint convert
        convert(self.config.checkpoint_job_dir)
        hf_dir_step_4 = os.listdir(os.path.join(checkpoint_step_4, "actor", "huggingface"))
        hf_dir_step_8 = os.listdir(os.path.join(checkpoint_step_8, "actor", "huggingface"))
        self.assertIn("model.safetensors", hf_dir_step_4)
        self.assertIn("model.safetensors", hf_dir_step_8)
        self.assertEqual(step_num, 8)
        ray.init(ignore_reinit_error=True, namespace=self.config.ray_namespace)
        # test bench mode
        self.config.mode = "bench"
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.explorer.bench_on_latest_checkpoint = False
        self.config.buffer.explorer_input.taskset = None
        self.config.buffer.explorer_input.tasksets = []
        self.config.buffer.trainer_input.experience_buffer = None
        self.config.check_and_update()
        bench(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        for prefix in ["eval", "bench"]:
            for taskset_name in ["countdown", "copy_countdown"]:
                metrics = parser.metric_list(f"{prefix}/{taskset_name}")
                self.assertGreater(len(metrics), 0, f"{prefix}/{taskset_name} metrics not found")
                repeat_times, k_list = 4, [2, 4]
                expected_stat_suffixes = [f"mean@{repeat_times}", f"std@{repeat_times}"]
                for k in k_list:
                    expected_stat_suffixes.extend([f"best@{k}", f"worst@{k}"])
                for stat_suffix in expected_stat_suffixes:
                    metric_name = f"{prefix}/{taskset_name}/score/{stat_suffix}"
                    metric_steps = parser.metric_steps(metric_name)
                    self.assertEqual(metric_steps, [0, 4, 8])

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


class TestStepAheadAsyncRL(BaseTrainerCase):
    def test_trainer(self):
        """Test the explore step ahead trainer."""
        # train 4 step, sync_offset=1, sync_interval=2
        # Explorer:
        # | 1 | 2 | 3 |sync| 4 |
        # |---|---|---|sync|---|
        # Trainer:
        #     | 1 | 2 |sync| 3 | 4 |
        #     |---|---|sync|---|---|
        self.config.buffer.batch_size = 6
        self.config.buffer.total_steps = 4
        # use 3 GPU in a 2 x 2 cluster, the trainer only have 1 GPU
        self.config.explorer.rollout_model.engine_num = 3
        self.config.explorer.rollout_model.tensor_parallel_size = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.trainer.save_interval = 4
        self.config.synchronizer.sync_interval = 2
        self.config.synchronizer.sync_offset = 1
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 1
        self.config.trainer.trainer_config.trainer.max_critic_ckpt_to_keep = 1

        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)
        actor_kl_metrics = parser.metric_list("actor/kl")
        self.assertGreater(len(actor_kl_metrics), 0)
        critic_kl_metrics = parser.metric_list("critic/kl")
        self.assertGreater(len(critic_kl_metrics), 0)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint

        checkpoint_step_4, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertEqual(step_num, 4)
        self.assertTrue(os.path.exists(checkpoint_step_4))

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


@parameterized_class(
    ("fsdp_strategy", "offloading"),
    [
        ("fsdp", False),
        ("fsdp2", False),
        ("fsdp", True),
        ("fsdp2", True),
    ],
)
class TestTrainerGSM8K(BaseTrainerCase):
    def test_trainer(self):
        """Test GSM8K."""
        # test both mode
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.repeat_times = 4
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {
            "epsilon": 1e-6,
        }
        # self.config.algorithm.repeat_times = 8  # TODO: used for real testing
        # self.config.buffer.batch_size = 96  # TODO: used for real testing
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.trainer.trainer_strategy = self.fsdp_strategy
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        actor_rollout_ref = self.config.trainer.trainer_config.actor_rollout_ref
        actor_rollout_ref.actor.optim.lr = 1e-5
        if self.fsdp_strategy == "fsdp":
            actor_rollout_ref.actor.fsdp_config.param_offload = self.offloading
            actor_rollout_ref.actor.fsdp_config.optimizer_offload = self.offloading
            actor_rollout_ref.ref.fsdp_config.param_offload = self.offloading
            actor_rollout_ref.ref.fsdp_config.optimizer_offload = self.offloading
        else:  # fsdp2
            actor_rollout_ref.actor.fsdp_config.offload_policy = self.offloading
            actor_rollout_ref.ref.fsdp_config.offload_policy = self.offloading
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        pipeline_metrics = parser.metric_list("experience_pipeline")
        self.assertGreater(len(pipeline_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        # TODO: used for real testing
        # rewards = parser.metric_values("critic/rewards/mean")
        # self.assertTrue(0.4 < rewards[0] < 0.55)
        # self.assertTrue(0.4 < rewards[1] < 0.55)
        # self.assertTrue(0.6 < rewards[2] < 0.7)
        # self.assertTrue(0.6 < rewards[3] < 0.7)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


class TestTrainerSFTWarmupGSM8K(BaseTrainerCase):
    @mock.patch("trinity.cli.launcher.load_config")
    def test_trainer(self, mock_load):
        """Test GSM8K With SFT."""
        # test both mode
        self.config.synchronizer.sync_interval = 1
        self.config.trainer.save_interval = 8
        self.config.stages = [
            StageConfig(
                stage_name="sft_warmup",
                mode="train",
                algorithm=AlgorithmConfig(algorithm_type="sft"),
                buffer=BufferConfig(
                    total_steps=3,
                    train_batch_size=4,
                    trainer_input=TrainerInput(
                        experience_buffer=get_unittest_dataset_config("sft_for_gsm8k")
                    ),
                ),
            ),
            StageConfig(
                stage_name="grpo",
                mode="both",
                algorithm=AlgorithmConfig(
                    algorithm_type="grpo",
                    repeat_times=4,
                ),
                buffer=BufferConfig(
                    batch_size=4,
                    explorer_input=ExplorerInput(taskset=get_unittest_dataset_config("gsm8k")),
                    trainer_input=TrainerInput(
                        experience_buffer=ExperienceBufferConfig(
                            name="test_queue_storage",
                            max_read_timeout=20,
                            storage_type=StorageType.QUEUE.value,
                            max_retry_times=10,
                        )
                    ),
                    total_epochs=1,
                ),
            ),
        ]
        self.config.check_and_update()
        old_taskset_path = self.config.stages[1].buffer.explorer_input.taskset.path
        self.config.stages[1].buffer.explorer_input.taskset.path = "/invalid/path"

        mock_load.return_value = deepcopy(self.config)

        with self.assertRaises(Exception):
            run(config_path="dummy.yaml")
        ray.shutdown(_exiting_interpreter=True)

        stage_configs = [cfg.check_and_update() for cfg in deepcopy(self.config)]

        # sft warmup stage
        sft_config = stage_configs[0]
        self.assertEqual(
            sft_config.synchronizer.sync_interval,
            sft_config.trainer.save_interval,
        )
        parser = TensorBoardParser(os.path.join(sft_config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertEqual(len(rollout_metrics), 0)
        sft_metrics = parser.metric_list("actor/sft")
        self.assertGreater(len(sft_metrics), 0)
        self.assertEqual(parser.metric_max_step(sft_metrics[0]), 3)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_min_step(response_metrics[0]), 1)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 3)

        self.config.stages[1].buffer.explorer_input.taskset.path = old_taskset_path
        mock_load.return_value = deepcopy(self.config)
        ray.init(ignore_reinit_error=True, namespace=self.config.ray_namespace)
        run(config_path="dummy.yaml")

        # grpo stage
        grpo_config = stage_configs[1]
        parser = TensorBoardParser(os.path.join(grpo_config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        sft_metrics = parser.metric_list("actor/sft")
        self.assertEqual(len(sft_metrics), 0)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_min_step(response_metrics[0]), 1)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        # test save checkpoint when sft finish
        for i in range(3):
            self.assertFalse(
                os.path.exists(os.path.join(sft_config.checkpoint_job_dir, f"global_step_{i}"))
            )
        self.assertEqual(
            get_checkpoint_dir_with_step_num(
                checkpoint_root_path=sft_config.checkpoint_job_dir, trainer_type="verl", step_num=3
            )[1],
            3,
        )
        # test save checkpoint at last step
        checkpoint_dir, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=grpo_config.checkpoint_job_dir,
            trainer_type="verl",
        )
        self.assertEqual(step_num, 4)
        self.assertGreater(len(os.listdir(os.path.join(checkpoint_dir, "actor"))), 0)

    def tearDown(self):
        # TODO: remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


class TestTrainerDPO(BaseTrainerCase):
    def test_trainer(self):
        """Test DPO."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = "dpo"
        self.config.algorithm.policy_loss_fn = "dpo"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.buffer.total_epochs = 2
        self.config.buffer.total_steps = 4  # step has higher priority than epoch
        self.config.synchronizer.sync_interval = 4
        self.config.buffer.train_batch_size = 8
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config("dpo")
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        self.config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr = 5e-7
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


class TestTrainerSFT(BaseTrainerCase):
    def test_trainer(self):
        """Test SFT."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = "sft"
        self.config.algorithm.policy_loss_fn = "sft"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.entropy_loss_fn = "none"
        self.config.synchronizer.sync_interval = 4
        self.config.buffer.train_batch_size = 4
        self.config.buffer.total_epochs = 2
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config(
            "sft_for_gsm8k"
        )
        self.config.check_and_update()
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


class TestTrainerToolsSFT(BaseTrainerCase):
    def test_trainer_tools(self):
        """Test SFT with tools."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = "sft"
        self.config.algorithm.policy_loss_fn = "sft"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.entropy_loss_fn = "none"
        self.config.synchronizer.sync_interval = 4
        self.config.buffer.train_batch_size = 4
        self.config.buffer.total_epochs = 4
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config(
            "sft_with_tools"
        )
        self.config.check_and_update()
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


def run_trainer(config: Config, stop_event=None) -> None:
    ray.init(
        namespace=config.ray_namespace,
        runtime_env={
            "env_vars": {
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "INFO",
            }
        },
    )
    try:
        train(config)
    finally:
        if stop_event:
            stop_event.set()
        ray.shutdown()


def run_explorer(config: Config, stop_event=None) -> None:
    ray.init(
        namespace=config.ray_namespace,
        runtime_env={
            "env_vars": {
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "INFO",
            }
        },
    )
    try:
        explore(config)
    finally:
        if stop_event:
            stop_event.set()
        ray.shutdown()


def run_both(config: Config, stop_event=None) -> None:
    ray.init(
        namespace=config.ray_namespace,
        runtime_env={
            "env_vars": {
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "INFO",
            }
        },
    )
    try:
        both(config)
    finally:
        if stop_event:
            stop_event.set()
        ray.shutdown()


def run_serve(config: Config, stop_event=None) -> None:
    ray.init(
        namespace=config.ray_namespace,
        runtime_env={
            "env_vars": {
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "INFO",
            }
        },
    )
    try:
        serve(config)
    finally:
        if stop_event:
            stop_event.set()
        ray.shutdown()


@parameterized_class(
    ("use_priority_queue", "strategy"),
    [(False, "fsdp"), (True, "fsdp"), (True, "megatron")],
)
class TestFullyAsyncMode(unittest.TestCase):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        self.process_list = []

    def test_fully_async_mode(self):
        config = get_template_config()
        config.project = "unittest"
        config.name = f"fully_async_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        config.checkpoint_root_dir = get_checkpoint_path()
        config.buffer.total_epochs = 1
        config.buffer.batch_size = 4
        config.cluster.gpu_per_node = 2
        config.cluster.node_num = 1
        config.model.model_path = get_model_path()
        config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE.value,
        )
        config.buffer.trainer_input.experience_buffer.replay_buffer.enable = self.use_priority_queue
        config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        config.synchronizer.sync_style = SyncStyle.EXPLORER_DRIVEN
        config.synchronizer.sync_interval = 8
        config.monitor.monitor_type = "tensorboard"
        trainer_config = deepcopy(config)
        trainer_config.mode = "train"
        trainer_config.buffer.train_batch_size = 4
        if self.strategy == "megatron":
            trainer_config.trainer.trainer_strategy = "megatron"
        trainer_config.check_and_update()
        if self.strategy == "megatron":
            _trainer_config = trainer_config.trainer.trainer_config
            _trainer_config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = 2
            _trainer_config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = 2
            _trainer_config.critic.strategy = "megatron"
            _trainer_config.critic.megatron.tensor_model_parallel_size = 2

        explorer1_config = deepcopy(config)
        explorer1_config.trainer = deepcopy(trainer_config.trainer)
        explorer1_config.mode = "explore"
        explorer1_config.explorer.name = "explorer1"
        config.cluster.gpu_per_node = 1
        config.cluster.node_num = 1
        explorer1_config.explorer.rollout_model.engine_num = 1
        explorer1_config.explorer.rollout_model.tensor_parallel_size = 1
        explorer1_config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE.value,
        )
        explorer2_config = deepcopy(explorer1_config)
        explorer2_config.trainer = deepcopy(trainer_config.trainer)
        explorer1_config.check_and_update()

        trainer_stop_event = multiprocessing.Event()
        trainer_process = multiprocessing.Process(
            target=run_trainer, args=(trainer_config, trainer_stop_event)
        )
        trainer_process.start()
        self.process_list.append(trainer_process)

        ray.init(ignore_reinit_error=True)
        while True:
            try:
                ray.get_actor("queue-exp_buffer", namespace=trainer_config.ray_namespace)
                break
            except ValueError:
                print("waiting for trainer to start.")
                time.sleep(5)

        explorer1_stop_event = multiprocessing.Event()
        explorer_process_1 = multiprocessing.Process(
            target=run_explorer, args=(explorer1_config, explorer1_stop_event)
        )
        explorer_process_1.start()
        self.process_list.append(explorer_process_1)

        time.sleep(5)
        explorer2_config.explorer.name = "explorer2"
        explorer2_config.check_and_update()
        explorer2_stop_event = multiprocessing.Event()
        explorer_process_2 = multiprocessing.Process(
            target=run_explorer, args=(explorer2_config, explorer2_stop_event)
        )
        explorer_process_2.start()
        self.process_list.append(explorer_process_2)

        explorer_process_1.join(timeout=300)
        if explorer_process_1.is_alive():
            self.fail("explorer1 process is still alive")
        explorer_process_2.join(timeout=300)
        if explorer_process_2.is_alive():
            self.fail("explorer2 process is still alive")

        # wait for trainer process to finish.
        trainer_process.join(timeout=200)
        if trainer_process.is_alive():
            self.fail("trainer process is still alive")

        # check the tensorboard
        parser = TensorBoardParser(
            os.path.join(trainer_config.monitor.cache_dir, "tensorboard", "trainer")
        )
        actor_metrics = parser.metric_list("actor")
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 8)
        parser = TensorBoardParser(
            os.path.join(explorer1_config.monitor.cache_dir, "tensorboard", "explorer1")
        )
        rollout_metrics = parser.metric_list("rollout")
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        parser = TensorBoardParser(
            os.path.join(explorer2_config.monitor.cache_dir, "tensorboard", "explorer2")
        )
        rollout_metrics = parser.metric_list("rollout")
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        # check the checkpoint
        explorer1_cache = StateManager(
            path=explorer1_config.checkpoint_job_dir,
            trainer_name=None,
            explorer_name="explorer1",
            config=explorer1_config,
        )
        cache = explorer1_cache.load_explorer()
        self.assertEqual(cache["latest_iteration"], 4)
        explorer2_cache = StateManager(
            path=explorer2_config.checkpoint_job_dir,
            trainer_name=None,
            explorer_name="explorer2",
            config=explorer2_config,
        )
        cache = explorer2_cache.load_explorer()
        self.assertEqual(cache["latest_iteration"], 4)
        trainer_cache = StateManager(
            path=trainer_config.checkpoint_job_dir,
            trainer_name=trainer_config.trainer.name,
            config=trainer_config,
        )
        cache = trainer_cache.load_trainer()
        self.assertEqual(cache["latest_iteration"], 8)
        # check the lastest checkpoint
        self.assertEqual(
            get_checkpoint_dir_with_step_num(
                checkpoint_root_path=explorer1_config.checkpoint_job_dir,
                trainer_type="verl",
            )[1],
            8,
        )
        self.assertEqual(
            get_checkpoint_dir_with_step_num(
                checkpoint_root_path=explorer2_config.checkpoint_job_dir,
                trainer_type="verl",
            )[1],
            8,
        )
        log_files = os.listdir(os.path.join(explorer1_config.checkpoint_job_dir, "log"))
        self.assertIn("trainer.log", log_files)
        self.assertIn("synchronizer.log", log_files)
        self.assertIn("explorer1.log", log_files)
        self.assertIn("explorer2.log", log_files)
        self.assertIn("explorer1_runner_0.log", log_files)
        self.assertIn("explorer1_runner_7.log", log_files)
        self.assertIn("explorer2_runner_0.log", log_files)
        self.assertIn("explorer2_runner_7.log", log_files)
        self.assertIn("explorer1_experience_pipeline.log", log_files)
        self.assertIn("explorer2_experience_pipeline.log", log_files)
        files_to_check = ["trainer.log", "synchronizer.log", "explorer1.log", "explorer2.log"]
        for file_name in files_to_check:
            with open(os.path.join(explorer1_config.checkpoint_job_dir, "log", file_name)) as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 0, f"{file_name} is empty")
        ray.shutdown()

    def tearDown(self):
        checkpoint_path = get_checkpoint_path()
        shutil.rmtree(os.path.join(checkpoint_path, "unittest"), ignore_errors=True)
        for process in self.process_list:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()


@parameterized_class(
    ("strategy",),
    [
        ("fsdp",),
        ("megatron",),
    ],
)
class TestTrainerCheckpointSave(unittest.TestCase):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        self.config = get_template_config()
        self.config.buffer.total_steps = 6
        self.config.buffer.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm_async"
        self.config.algorithm.repeat_times = 3
        self.config.project = "Trainer-unittest"
        self.config.name = f"trainer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.monitor.monitor_type = "tensorboard"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 1
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.explorer.eval_interval = 4
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.trainer.save_interval = 2
        self.config.trainer.save_hf_checkpoint = "last"
        self.config.trainer.trainer_strategy = self.strategy
        self.config.trainer.max_checkpoints_to_keep = 2
        self.config.check_and_update()
        self.process_list = []

    def test_trainer(self):  # noqa: C901
        """Test the checkpoint saving."""
        _trainer_config = self.config.trainer.trainer_config
        if self.strategy == "megatron":
            _trainer_config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = 2
            _trainer_config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = 2
            _trainer_config.critic.megatron.tensor_model_parallel_size = 2

        stop_event = multiprocessing.Event()
        trainer_process = multiprocessing.Process(target=run_both, args=(self.config, stop_event))
        trainer_process.start()
        self.process_list.append(trainer_process)

        default_local_dir = _trainer_config.trainer.default_local_dir
        state_dict_iteration = checkpoint_iteration = 0
        state_dict_iteration_file = os.path.join(
            default_local_dir, "latest_state_dict_iteration.txt"
        )
        checkpoint_iteration_file = os.path.join(
            default_local_dir, "latest_checkpointed_iteration.txt"
        )

        megatron_dist_ckpt_items = {
            "__0_1.distcp",
            "__1_0.distcp",
            "common.pt",
            ".metadata",
            "metadata.json",
            "__1_1.distcp",
            "__0_0.distcp",
        }
        start_time = time.time()
        while not stop_event.is_set() and time.time() - start_time < 60 * 10:
            time.sleep(10)

            if os.path.exists(state_dict_iteration_file):
                try:
                    with open(state_dict_iteration_file, "r") as f:
                        state_dict_iteration = int(f.read().strip())
                except (IOError, ValueError):
                    pass
            if os.path.exists(checkpoint_iteration_file):
                try:
                    with open(checkpoint_iteration_file, "r") as f:
                        checkpoint_iteration = int(f.read().strip())
                except (IOError, ValueError):
                    pass

            if state_dict_iteration > 0:
                iteration_dir = os.path.join(
                    default_local_dir, f"global_step_{state_dict_iteration}", "actor"
                )
                if self.strategy == "fsdp":
                    items = os.listdir(iteration_dir)
                    self.assertIn("model_world_size_2_rank_0.pt", items)
                    self.assertIn("model_world_size_2_rank_1.pt", items)
                else:  # megatron
                    dist_ckpt_dir = os.path.join(iteration_dir, "dist_ckpt")
                    self.assertEqual(
                        set(os.listdir(dist_ckpt_dir)),
                        megatron_dist_ckpt_items,
                    )
                    huggingface_dir = os.path.join(iteration_dir, "huggingface")
                    items = os.listdir(huggingface_dir)
                    self.assertIn("config.json", items)
                    self.assertIn("generation_config.json", items)
                # print(f"State dict check at {state_dict_iteration} iteration passed.")  # for debug

            if checkpoint_iteration > 0:
                flag_file = os.path.join(
                    default_local_dir, f"global_step_{checkpoint_iteration}", ".full_checkpoint"
                )
                self.assertTrue(os.path.exists(flag_file))
                for sub_dir_name in ["critic", "actor"]:
                    iteration_dir = os.path.join(
                        default_local_dir, f"global_step_{checkpoint_iteration}", sub_dir_name
                    )
                    if self.strategy == "fsdp":
                        self.assertEqual(
                            set(os.listdir(iteration_dir)),
                            {
                                "model_world_size_2_rank_0.pt",
                                "model_world_size_2_rank_1.pt",
                                "optim_world_size_2_rank_1.pt",
                                "optim_world_size_2_rank_0.pt",
                                "extra_state_world_size_2_rank_0.pt",
                                "extra_state_world_size_2_rank_1.pt",
                                "huggingface",
                                "fsdp_config.json",
                            },
                        )
                    else:  # megatron
                        dist_ckpt_dir = os.path.join(iteration_dir, "dist_ckpt")
                        self.assertEqual(
                            set(os.listdir(dist_ckpt_dir)),
                            megatron_dist_ckpt_items,
                        )
                    huggingface_dir = os.path.join(iteration_dir, "huggingface")
                    huggingface_dir_files = os.listdir(huggingface_dir)
                    self.assertEqual(
                        set(huggingface_dir_files)
                        - {"generation_config.json", "model.safetensors"},
                        {
                            "vocab.json",
                            "merges.txt",
                            "added_tokens.json",
                            "tokenizer.json",
                            "config.json",
                            "chat_template.jinja",
                            "tokenizer_config.json",
                            "special_tokens_map.json",
                        },
                    )
                # print(f"Checkpoint check at {checkpoint_iteration} iteration passed.")  # for debug
        if not stop_event.is_set():
            self.fail("Training process failed to stop.")
        # check only full checkpoint dirs are kept
        for sync_step in [1, 3, 5]:
            state_dict_dir = os.path.join(default_local_dir, f"global_step_{sync_step}")
            self.assertFalse(
                os.path.exists(state_dict_dir),
                f"Found unexpected state dict dir at step {sync_step}",
            )
        for checkpoint_step in [4, 6]:
            checkpoint_dir = os.path.join(default_local_dir, f"global_step_{checkpoint_step}")
            self.assertTrue(
                os.path.exists(checkpoint_dir),
                f"Missing expected checkpoint dir at step {checkpoint_step}",
            )
            actor_checkpoint_dir = os.path.join(checkpoint_dir, "actor")
            self.assertTrue(os.path.exists(actor_checkpoint_dir))
        # check step 2 should have no checkpoint
        checkpoint_dir = os.path.join(default_local_dir, "global_step_2")
        self.assertTrue(os.path.exists(checkpoint_dir))
        actor_checkpoint_dir = os.path.join(checkpoint_dir, "actor")
        self.assertFalse(os.path.exists(actor_checkpoint_dir))
        critic_checkpoint_dir = os.path.join(checkpoint_dir, "critic")
        self.assertFalse(os.path.exists(critic_checkpoint_dir))
        trainer_process.join(timeout=10)
        self.assertIn("model.safetensors", huggingface_dir_files)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)
        for process in self.process_list:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()


class TestTrainerMIX(BaseTrainerCase):
    def test_trainer(self):
        """Test MIX algorithm."""
        # gsm8k has 16 tasks, sft_for_gsm8k has 8 tasks
        # total 4 steps, each step: read 4 tasks from gsm8k, 16 tasks from sft_for_gsm8k
        self.config.algorithm.algorithm_type = "mix"
        self.config.algorithm.repeat_times = 4
        self.config.algorithm.sample_strategy = "mix"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.sample_strategy_args = {"expert_data_ratio": 0.5}  # rft=4*4 : sft=16
        self.config.algorithm.policy_loss_fn = "mix"
        self.config.buffer.batch_size = 4
        self.config.buffer.train_batch_size = 32
        self.config.buffer.total_steps = 2
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.synchronizer.sync_interval = 1
        self.config.trainer.save_interval = 1
        self.config.buffer.trainer_input.auxiliary_buffers[
            "sft_dataset"
        ] = get_unittest_dataset_config("sft_for_gsm8k")
        self.config.buffer.trainer_input.auxiliary_buffers[
            "sft_dataset"
        ].total_epochs = 8  # test this works
        self.config.check_and_update()
        self.config.buffer.trainer_input.experience_buffer.max_read_timeout = 20
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        both(self.config)
        ray.shutdown(_exiting_interpreter=True)

        # check trainer resume metadata
        trainer_meta_file = os.path.join(self.config.checkpoint_job_dir, "trainer_meta.json")
        with open(trainer_meta_file) as f:
            trainer_meta = json.load(f)
        self.assertEqual(trainer_meta["latest_iteration"], 2)
        self.assertEqual(
            trainer_meta["sample_strategy_state"]["expert_buffer"]["current_index"], 32
        )

        self.config.buffer.total_steps = None
        self.config.buffer.total_epochs = 1
        self.config.check_and_update()
        ray.init(ignore_reinit_error=True, namespace=self.config.ray_namespace)
        both(self.config)

        # check trainer resume metadata
        with open(trainer_meta_file) as f:
            trainer_meta = json.load(f)
        self.assertEqual(trainer_meta["latest_iteration"], 4)
        self.assertEqual(
            trainer_meta["sample_strategy_state"]["expert_buffer"]["current_index"], 64
        )

        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))

        # test rollout metrics
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        self.assertEqual(
            parser.metric_values("experience_pipeline/experience_count")[1], 16
        )  # 16 rft experiences
        # test actor metrics
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        expert_metrics = parser.metric_list("actor/expert/")
        self.assertEqual(parser.metric_max_step(expert_metrics[0]), 4)  # SFT
        usual_metrics = parser.metric_list("actor/usual/")
        self.assertEqual(parser.metric_max_step(usual_metrics[0]), 4)  # RFT
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_min_step(response_metrics[0]), 1)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        # test save checkpoint at last step
        checkpoint_dir, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type="verl",
        )
        self.assertEqual(step_num, 4)
        self.assertGreater(len(os.listdir(os.path.join(checkpoint_dir, "actor"))), 0)

    def tearDown(self):
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


async def run_math_workflow(serve_url: str, task: dict):
    from trinity.common.rewards.math_reward import MathRewardFn

    proxy_client = TrinityClient(serve_url)
    openai_client = proxy_client.get_openai_async_client()

    query = task["question"]
    truth = task["answer"]

    reward_fn = MathRewardFn()

    system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    models = await openai_client.models.list()
    model = models.data[0].id

    response = await openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    answer = response.choices[0].message.content
    reward = reward_fn(response=answer, truth=truth, prompt=query)
    await proxy_client.feedback_async(sum(reward.values()), [response.id])


class TestServeWithTrainer(RayUnittestBaseAsync):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        checkpoint_path = get_checkpoint_path()
        shutil.rmtree(os.path.join(checkpoint_path, "unittest"), ignore_errors=True)

        config = get_template_config()
        config.project = "unittest"
        config.name = f"serve_with_trainer_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        config.checkpoint_root_dir = get_checkpoint_path()
        config.model.model_path = get_model_path()
        config.buffer.batch_size = 4
        config.buffer.train_batch_size = 4
        config.algorithm.algorithm_type = "ppo"
        config.algorithm.repeat_times = 1
        config.cluster.gpu_per_node = 2
        config.cluster.node_num = 1
        config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.SQL.value,
            schema_type="experience",
        )
        config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        config.buffer.total_steps = 3
        config.trainer.save_interval = 1
        config.synchronizer.sync_interval = 1
        config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        config.explorer.rollout_model.engine_num = 2
        config.explorer.rollout_model.enable_openai_api = True
        config.explorer.rollout_model.tensor_parallel_size = 1
        config.explorer.service_status_check_interval = 5
        self.config = config
        self.process_list = []

    async def test_serve_with_trainer(self):  # noqa: C901
        trainer_config = deepcopy(self.config)
        trainer_config.mode = "train"
        trainer_config.check_and_update()
        trainer_config.trainer.max_actor_ckpt_to_keep = 10

        trainer_process = multiprocessing.Process(target=run_trainer, args=(trainer_config,))
        trainer_process.start()
        self.process_list.append(trainer_process)

        ray.init(ignore_reinit_error=True)
        while True:
            try:
                ray.get_actor("sql-exp_buffer", namespace=trainer_config.ray_namespace)
                break
            except ValueError:
                print("waiting for trainer to start.")
                await asyncio.sleep(5)
        serve_config = deepcopy(self.config)
        serve_config.mode = "serve"
        serve_config.check_and_update()
        serve_process = multiprocessing.Process(target=run_serve, args=(serve_config,))
        serve_process.start()
        self.process_list.append(serve_process)

        state_manager = StateManager(
            path=serve_config.checkpoint_job_dir,
            explorer_name=serve_config.explorer.name,
        )

        # wait for explorer initialization
        for i in range(30):
            try:
                server_url = state_manager.load_explorer_server_url()
            except Exception:
                server_url = None
            if server_url:
                break
            await asyncio.sleep(3)
        if not server_url:
            raise RuntimeError("Explorer server URL not found.")
        proxy_client = TrinityClient(server_url)
        # wait for server setup
        for i in range(10):
            if proxy_client.alive():
                print("Proxy server is alive.")
                break
            await asyncio.sleep(2)

        self.config.buffer.explorer_input.taskset.batch_size = 4
        reader = get_buffer_reader(self.config.buffer.explorer_input.taskset)

        for i in range(3):
            tasks = reader.read()
            await asyncio.gather(*(run_math_workflow(server_url, task.raw_task) for task in tasks))
            await proxy_client.commit_async()
            # wait for synchronizer started
            end_time = time.time()
            find_checkpoint = False
            while time.time() - end_time < 100:
                _, step_num = get_checkpoint_dir_with_step_num(
                    checkpoint_root_path=serve_config.checkpoint_job_dir,
                    raise_error=False,
                )
                if step_num >= i + 1:  # checkpoint has been generated
                    find_checkpoint = True
                    break
                await asyncio.sleep(1)
            self.assertTrue(find_checkpoint, f"Checkpoint at step {i + 1} not found in time.")
            metrics = await proxy_client.get_metrics_async()
            self.assertEqual(metrics["rollout/total_experience_count"], 4 * (i + 1))
            self.assertEqual(metrics["rollout/ready_experience_count"], 4 * (i + 1))
            self.assertGreater(metrics["rollout/model_0/total_request_count"], 0)
            self.assertGreater(metrics["rollout/model_1/total_request_count"], 0)
            if i > 1:
                self.assertGreater(metrics["rollout/model_0/model_version"], 0)
                self.assertGreater(metrics["rollout/model_1/model_version"], 0)
        metrics = await proxy_client.get_metrics_async()
        self.assertEqual(metrics["rollout/total_experience_count"], 12)
        self.assertEqual(metrics["rollout/ready_experience_count"], 12)
        self.assertGreater(metrics["rollout/model_0/total_request_count"], 0)
        self.assertGreater(metrics["rollout/model_1/total_request_count"], 0)
        self.assertEqual(
            metrics["rollout/model_0/total_request_count"]
            + metrics["rollout/model_1/total_request_count"],
            metrics["rollout/total_experience_count"],
        )
        # at least updated to version 1
        await asyncio.sleep(5)  # wait for model version update
        self.assertGreaterEqual(metrics["rollout/model_0/model_version"], 1)
        self.assertGreaterEqual(metrics["rollout/model_1/model_version"], 1)
        # check final checkpoint
        _, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=serve_config.checkpoint_job_dir,
            step_num=3,
        )

    def tearDown(self):
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)
        for process in self.process_list:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
        super().tearDown()


class TestMultiModalGRPO(BaseTrainerCase):
    def test_trainer(self):
        """Test both mode with multi-modal data."""
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config(
            "geometry"
        )  # Total 8 tasks
        self.config.model.model_path = get_vision_language_model_path()
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.repeat_times = 4
        self.config.buffer.batch_size = 4
        self.config.buffer.total_epochs = 1
        self.config.trainer.save_interval = 2
        self.config.check_and_update()
        both(self.config)
        # check metrics are available
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 2)
        # check save lastest checkpoint
        checkpoint_step_2, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertGreater(len(os.listdir(os.path.join(checkpoint_step_2, "actor"))), 0)
        self.assertEqual(step_num, 2)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


class TestMultiModalSFT(BaseTrainerCase):
    def test_trainer(self):
        """Test SFT mode with multi-modal data."""
        self.config.mode = "train"
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config(
            "geometry"
        )  # Total 8 tasks
        self.config.model.model_path = get_vision_language_model_path()
        self.config.algorithm.algorithm_type = "sft"
        self.config.algorithm.policy_loss_fn = "sft"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.entropy_loss_fn = "none"
        self.config.buffer.train_batch_size = 4
        self.config.buffer.total_epochs = 1
        self.config.trainer.save_interval = 2
        self.config.check_and_update()
        train(self.config)
        # check metrics are available
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 2)
        # check save lastest checkpoint
        checkpoint_step_2, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertGreater(len(os.listdir(os.path.join(checkpoint_step_2, "actor"))), 0)
        self.assertEqual(step_num, 2)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


class TestTrainerLoRA(BaseTrainerCase):
    def test_trainer(self):
        """Test both mode with LoRA request."""
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("gsm8k", "test")
        )
        self.config.buffer.explorer_input.eval_tasksets[0].repeat_times = 8
        self.config.model.model_path = get_model_path()
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.repeat_times = 4
        self.config.buffer.batch_size = 4
        self.config.buffer.total_steps = 2
        self.config.cluster.node_num = 1
        self.config.cluster.gpu_per_node = 4
        self.config.explorer.eval_interval = 2
        self.config.model.lora_configs = [get_lora_config()]
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.synchronizer.sync_interval = 2
        self.config.trainer.save_interval = 2
        self.config.check_and_update()
        both(self.config)
        # check metrics are available
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 2)
        ray.shutdown(_exiting_interpreter=True)
        # check save lastest checkpoint
        checkpoint_step_2, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertGreater(len(os.listdir(os.path.join(checkpoint_step_2, "actor"))), 0)
        self.assertGreater(
            len(os.listdir(os.path.join(checkpoint_step_2, "actor", "lora_adapter"))), 0
        )
        self.assertEqual(step_num, 2)

        # test bench mode
        ray.init(ignore_reinit_error=True, namespace=self.config.ray_namespace)
        self.config.mode = "bench"
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.explorer.bench_on_latest_checkpoint = False
        self.config.check_and_update()
        bench(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        for prefix in ["eval", "bench"]:
            gsm8k_metrics = parser.metric_list(f"{prefix}/gsm8k")
            self.assertGreater(len(gsm8k_metrics), 0, f"{prefix}/gsm8k metrics not found")
            repeat_times, k_list = 8, [2, 4, 8]
            expected_stat_suffixes = [f"mean@{repeat_times}", f"std@{repeat_times}"]
            for k in k_list:
                expected_stat_suffixes.extend([f"best@{k}", f"worst@{k}"])
            for stat_suffix in expected_stat_suffixes:
                metric_name = f"{prefix}/gsm8k/accuracy/{stat_suffix}"
                metric_steps = parser.metric_steps(metric_name)
                self.assertEqual(metric_steps, [0, 2])

    def tearDown(self):
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


class TestOverRollout(BaseTrainerCase):
    def test_trainer(self):
        self.config.algorithm.repeat_times = 4
        self.config.buffer.batch_size = 4
        self.config.buffer.total_steps = 2
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config(
            "countdown", "train"
        )
        self.config.buffer.explorer_input.eval_tasksets = [
            get_unittest_dataset_config("countdown", "test")
        ]
        self.config.buffer.eval_interval = 4  # only eval on start
        self.config.name = f"explore-over-rollout-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.explorer.over_rollout.ratio = 0.5  # set over rollout rate to 50%, which means only wait for 2 (4 * 50%) tasks in each steps
        self.config.explorer.over_rollout.wait_after_min = 0
        self.config.explorer.dynamic_timeout.enable = True
        self.config.explorer.dynamic_timeout.ratio = 2
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {
            "epsilon": 1e-6,
        }
        self.config.synchronizer.sync_style = SyncStyle.EXPLORER_DRIVEN
        self.config.synchronizer.sync_interval = 1
        self.config.check_and_update()
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        eval_metrics = parser.metric_list("eval")
        self.assertGreater(len(eval_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)
        self.assertTrue(parser.metric_exist("experience_pipeline/experience_count"))
        experience_counts = parser.metric_values("experience_pipeline/experience_count")
        self.assertEqual(len(experience_counts), 2)
        for count in experience_counts:
            self.assertGreaterEqual(
                count, 2 * 4
            )  # at least process 2 tasks in each step, repeat_times is 4
        pg_loss = parser.metric_values("actor/pg_loss")
        self.assertGreaterEqual(len(pg_loss), 1)  # trainer only has at least 1 step
        exp_save_path = self.config.buffer.trainer_input.experience_buffer.path
        with open(exp_save_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreaterEqual(
                len(lines), 2 * 4 * 2
            )  # at least contain total_steps * repeat_times * batch_size * min_waited_tasks

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


class TestTrainerPromptTruncation(BaseTrainerCase):
    def test_trainer(self):
        self.config.model.max_model_len = 20
        self.config.model.max_prompt_tokens = 5
        self.config.model.max_response_tokens = 15
        self.config.model.enable_prompt_truncation = True
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.repeat_times = 2
        self.config.buffer.batch_size = 4
        self.config.buffer.total_steps = 2
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.check_and_update()
        both(self.config)

        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)
        max_prompt_length = parser.metric_values("prompt_length/max")
        self.assertEqual(max(max_prompt_length), 5)
        min_prompt_length = parser.metric_values("prompt_length/min")
        self.assertEqual(min(min_prompt_length), 5)
        max_response_length = parser.metric_values("response_length/max")
        self.assertEqual(max(max_response_length), 1)
        min_response_length = parser.metric_values("response_length/min")
        self.assertEqual(min(min_response_length), 1)
        final_loss = parser.metric_values("actor/final_loss")
        self.assertEqual(final_loss[0], 0.0)
        grad_norm = parser.metric_values("actor/grad_norm")
        self.assertEqual(grad_norm[0], 0.0)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


@unittest.skipIf("TINKER_API_KEY" not in os.environ, "TINKER_API_KEY is not set")
class TestTinkerTrainer(BaseTrainerCase):
    def test_trainer(self):
        """Test GSM8K on tinker."""
        # test both mode
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.repeat_times = 4
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {
            "epsilon": 1e-6,
        }
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.model.tinker.enable = True
        self.config.model.model_path = "Qwen/Qwen3-4B-Instruct-2507"
        self.config.check_and_update()
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        pipeline_metrics = parser.metric_list("experience_pipeline")
        self.assertGreater(len(pipeline_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)

    def test_trainer_class(self):
        total_steps = 100
        lr_warmup_steps = 10
        self.config.algorithm.algorithm_type = "grpo"
        self.config.model.tinker.enable = True
        self.config.model.model_path = "Qwen/Qwen3-4B-Instruct-2507"
        self.config.trainer.total_steps = total_steps
        self.config.algorithm.optimizer.lr_warmup_steps = lr_warmup_steps
        self.config.algorithm.optimizer.lr_scheduler_type = "cosine"
        self.config.check_and_update()
        lr = self.config.algorithm.optimizer.lr

        @ray.remote
        class FakeExplorer:
            def __init__(self, config: Config):
                self.config = config
                self.synchronizer = Synchronizer.get_actor(config)

            async def is_alive(self):
                return True

        fake_explorer = FakeExplorer.remote(self.config)
        ray.get(fake_explorer.__ray_ready__.remote())

        tinker_trainer = TinkerTrainerWrapper(self.config)
        tinker_trainer._train_step_num = 5
        self.assertEqual(tinker_trainer.current_learning_rate, lr * 0.5)
        tinker_trainer._train_step_num = 50
        self.assertEqual(
            tinker_trainer.current_learning_rate,
            lr
            * (
                0.5
                * (1 + math.cos((50 - lr_warmup_steps) / (total_steps - lr_warmup_steps) * math.pi))
            ),
        )

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)


@unittest.skip("Require agentscope >= 1.0.12")
class AgentScopeTunerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        ray.init(ignore_reinit_error=True)

    def tearDown(self) -> None:
        ray.shutdown(_exiting_interpreter=True)

    def test_agentscope_tuner(self):
        from agentscope.agent import ReActAgent
        from agentscope.formatter import OpenAIChatFormatter
        from agentscope.message import Msg
        from agentscope.model import ChatModelBase
        from agentscope.tuner import (
            AlgorithmConfig,
            DatasetConfig,
            JudgeOutput,
            TunerModelConfig,
            WorkflowOutput,
            tune,
        )

        async def workflow_func(
            task: Dict,
            model: ChatModelBase,
            auxiliary_models: Dict[str, ChatModelBase],
            logger: Logger,
        ) -> WorkflowOutput:
            assert isinstance(model, ChatModelBase)
            assert "judge_model" in auxiliary_models
            assert isinstance(auxiliary_models["judge_model"], ChatModelBase)
            agent = ReActAgent(
                name="test_agent",
                model=model,
                sys_prompt="You are a helpful assistant.",
                formatter=OpenAIChatFormatter(),
            )
            st = time.time()
            response = await agent.reply(Msg("user", task["question"], role="user"))
            et = time.time()
            logger.info(f"Question: {task['question']}\nAnswer: {response.get_text_content()}")
            return WorkflowOutput(response=response, metrics={"workflow_time": et - st})

        async def judge_func(
            task: Dict, response: Msg, auxiliary_models: Dict[str, ChatModelBase], logger: Logger
        ) -> JudgeOutput:
            assert "judge_model" in auxiliary_models
            judge_model = auxiliary_models["judge_model"]
            assert isinstance(judge_model, ChatModelBase)
            agent = ReActAgent(
                name="judge_agent",
                model=judge_model,
                sys_prompt="You are a judge to evaluate the correctness of answers.",
                formatter=OpenAIChatFormatter(),
            )
            workflow_text_response = response.get_text_content()
            st = time.time()
            judge_response = await agent.reply(
                Msg(
                    "user",
                    f"Question: {task['question']}\nAnswer: {workflow_text_response}\nIs the answer correct? Reply with 'Yes' or 'No'.",
                    role="user",
                )
            )
            et = time.time()
            logger.info(f"Judge Response: {judge_response.get_text_content()}")
            judge_response = judge_response.get_text_content()
            if judge_response is not None and "yes" in judge_response.lower():
                is_correct = True
            else:
                is_correct = False
            return JudgeOutput(
                reward=float(is_correct),
                metrics={"judge_time": et - st},
            )

        gsm8k_dataset = get_unittest_dataset_config("gsm8k")

        dataset = DatasetConfig(
            path=gsm8k_dataset.path,
            split="train",
            total_steps=2,
        )
        eval_dataset = DatasetConfig(
            path=gsm8k_dataset.path,
            split="test",
        )

        model = TunerModelConfig(
            model_path=get_model_path(),
            max_model_len=4096,
            max_tokens=2048,
            inference_engine_num=2,
        )

        auxiliary_models = {
            "judge_model": TunerModelConfig(
                model_path=get_model_path(),
                max_model_len=8192,
                max_tokens=2048,
                inference_engine_num=2,
            )
        }

        algorithm = AlgorithmConfig(
            algorithm_type="multi_step_grpo",
            batch_size=4,
            group_size=4,
            eval_interval_steps=2,
            save_interval_steps=2,
        )

        tune(
            workflow_func=workflow_func,
            judge_func=judge_func,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            model=model,
            auxiliary_models=auxiliary_models,
            algorithm=algorithm,
        )
        # check checkpoint dir in `./checkpoints/AgentScope/Experiment-<timestamp>`
        self.assertTrue(os.path.exists("./checkpoints/AgentScope"))
        exp_dirs = os.listdir("./checkpoints/AgentScope")
        self.assertGreaterEqual(len(exp_dirs), 1)
        latest_exp_dir = sorted(exp_dirs)[-1]
        exp_dir_path = os.path.join("./checkpoints/AgentScope", latest_exp_dir)
        _, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=exp_dir_path,
            trainer_type="verl",
        )
        self.assertEqual(step_num, 2)
        # check tensorboard
        parser = TensorBoardParser(os.path.join(exp_dir_path, "monitor", "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertIn("rollout/workflow_time/mean", rollout_metrics)
        self.assertIn("rollout/judge_time/mean", rollout_metrics)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)
        eval_metrics = parser.metric_list("eval")
        self.assertGreater(len(eval_metrics), 0)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 2)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)


class ColocateModeTest(RayUnittestBase):
    def setUp(self) -> None:
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.mode = "colocate"
        self.config.cluster.node_num = 1
        self.config.cluster.gpu_per_node = 1
        self.config.project = "Trainer-unittest"
        self.config.name = f"trainer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.model.model_path = get_model_path()
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.explorer.rollout_model.engine_num = 1
        self.config.trainer.ulysses_sequence_parallel_size = 1
        self.config.synchronizer.sync_method = SyncMethod.MEMORY

    def test_trainer(self):
        """Test colocate mode with both trainer and explorer."""
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("gsm8k", "test")
        )
        self.config.buffer.explorer_input.eval_tasksets[0].repeat_times = 4
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.repeat_times = 4
        self.config.buffer.batch_size = 8
        self.config.explorer.eval_interval = 2
        self.config.buffer.total_steps = 2
        self.config.trainer.save_interval = 2
        self.config.synchronizer.sync_interval = 1
        self.config.check_and_update()
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertGreater(len(rollout_metrics), 0)
        pipeline_metrics = parser.metric_list("experience_pipeline")
        self.assertGreater(len(pipeline_metrics), 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)
        actor_metrics = parser.metric_list("actor")
        self.assertGreater(len(actor_metrics), 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)
        response_metrics = parser.metric_list("response_length")
        self.assertGreater(len(response_metrics), 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 2)
        eval_metrics = parser.metric_list("eval")
        self.assertGreater(len(eval_metrics), 0)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 2)
