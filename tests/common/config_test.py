# -*- coding: utf-8 -*-
"""Test cases for Config modules."""
import datetime
import math
import os
import shutil
import unittest

import torch

from tests.tools import get_template_config, get_unittest_dataset_config
from trinity.common.config import InferenceModelConfig, load_config

CHECKPOINT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "temp_checkpoint_dir")


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        config = get_template_config()
        config.buffer.batch_size = 8
        config.algorithm.repeat_times = 10
        config.model.model_path = "Qwen/Qwen3-1.7B"
        config.cluster.gpu_per_node = 8
        config.cluster.node_num = 2
        config.explorer.rollout_model.engine_num = 2
        config.explorer.rollout_model.tensor_parallel_size = 2
        config.explorer.auxiliary_models.append(
            InferenceModelConfig(model_path="Qwen/Qwen3-32B", tensor_parallel_size=4, engine_num=1),
        )
        config.check_and_update()
        self.assertIsNotNone(config.trainer.trainer_config)
        self.assertEqual(config.trainer.trainer_config.trainer.n_gpus_per_node, 8)
        self.assertEqual(config.trainer.trainer_config.trainer.nnodes, 1)
        self.assertEqual(config.trainer.trainer_config.trainer.project_name, config.project)
        self.assertEqual(config.trainer.trainer_config.trainer.experiment_name, config.name)
        self.assertEqual(
            config.buffer.explorer_input.tasksets[0].repeat_times, config.algorithm.repeat_times
        )
        self.assertEqual(config.model.model_path, config.model.critic_model_path)
        self.assertEqual(config.model.model_path, config.explorer.rollout_model.model_path)

    def test_all_examples_are_valid(self):
        example_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        for example_name in os.listdir(example_dir):
            for filename in os.listdir(os.path.join(example_dir, example_name)):
                if filename.endswith(".yaml") and not (
                    filename.startswith("train_")
                    or filename.startswith("verl_")
                    or filename.startswith("dj_")
                    or filename.startswith("tinker")
                ):
                    print(f"Checking config: {filename}")
                    config_path = os.path.join(example_dir, example_name, filename)
                    try:
                        config = load_config(config_path)
                        config.checkpoint_root_dir = "./.cache/"
                        config.ignore_validator_suggestions = True
                        config.check_and_update()
                    except Exception as e:
                        print(f"Error loading config {config_path}: {e}")
                        raise e

    def test_continue_from_checkpoint_is_valid(self):
        config = get_template_config()
        config.name = "test"
        config.project = "unittest"
        config.checkpoint_root_dir = CHECKPOINT_ROOT_DIR

        dir_path = os.path.join(config.checkpoint_root_dir, config.project, config.name)
        os.makedirs(os.path.join(dir_path, "global_step_1"))

        config.continue_from_checkpoint = True
        config.check_and_update()
        self.assertEqual(config.name, "test")

        config.continue_from_checkpoint = False
        config.check_and_update()
        self.assertTrue(config.name.startswith("test_"))
        timestamp = config.name.split("_")[-1]
        self.assertTrue(datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S"))

    def test_config_flatten(self):
        config = get_template_config()
        flat_config = config.flatten()
        self.assertIsInstance(flat_config, dict)
        for key, value in flat_config.items():
            self.assertIsInstance(key, str)
            self.assertNotIsInstance(value, dict)

    def test_update_config_from_ray_cluster(self):
        config = get_template_config()
        config.cluster.node_num = None
        config.cluster.gpu_per_node = None

        config.check_and_update()
        self.assertEqual(config.cluster.node_num, 2)
        self.assertEqual(config.cluster.gpu_per_node, 2)

    def test_default_workflow(self):
        config = get_template_config()
        config.buffer.explorer_input.default_workflow_type = "simple_workflow"
        config.buffer.explorer_input.default_eval_workflow_type = "math_boxed_workflow"
        config.buffer.explorer_input.eval_tasksets.append(get_unittest_dataset_config("gsm8k"))
        st = get_unittest_dataset_config("countdown")
        st.default_workflow_type = None
        config.buffer.explorer_input.eval_tasksets.append(st)
        config.check_and_update()
        self.assertEqual(
            config.buffer.explorer_input.eval_tasksets[0].default_workflow_type,
            "math_workflow",
        )
        self.assertEqual(
            config.buffer.explorer_input.eval_tasksets[1].default_workflow_type,
            "math_boxed_workflow",
        )
        self.assertEqual(
            config.buffer.explorer_input.tasksets[0].default_workflow_type,
            "simple_workflow",
        )

    def test_max_token_len_per_gpu_set_correctly(self):
        config = get_template_config()
        config.model.max_model_len = 8192
        config.trainer.ulysses_sequence_parallel_size = 2
        config.trainer.max_token_len_per_gpu = None
        config.check_and_update()
        self.assertIsNotNone(config.trainer.trainer_config)
        expected_max_token_len = math.ceil(
            (2 * config.model.max_model_len) / config.trainer.ulysses_sequence_parallel_size
        )
        self.assertEqual(
            config.trainer.trainer_config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
            expected_max_token_len,
        )
        self.assertEqual(
            config.trainer.trainer_config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu,
            expected_max_token_len,
        )
        self.assertEqual(
            config.trainer.trainer_config.critic.ppo_max_token_len_per_gpu,
            expected_max_token_len,
        )

    def test_optimizer_config_propagation(self):
        config = get_template_config()
        config.algorithm.optimizer.lr = 1e-4
        config.algorithm.optimizer.weight_decay = 0.05
        config.algorithm.optimizer.clip_grad = 2.0
        config.trainer.total_steps = 1000
        config.algorithm.optimizer.lr_scheduler_type = "cosine"
        config.algorithm.optimizer.min_lr_ratio = 1e-2
        config.check_and_update()
        self.assertEqual(config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr, 1e-4)
        self.assertEqual(
            config.trainer.trainer_config.actor_rollout_ref.actor.optim.weight_decay, 0.05
        )
        self.assertEqual(config.trainer.trainer_config.actor_rollout_ref.actor.optim.clip_grad, 2.0)
        self.assertEqual(
            config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr_decay_steps, 1000
        )
        self.assertEqual(
            config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr_decay_style, "cosine"
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr_warmup_init
                ),
                torch.tensor(1e-6),
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(config.trainer.trainer_config.actor_rollout_ref.actor.optim.min_lr),
                torch.tensor(1e-6),
            )
        )
        # critic optimizer should not be affected
        self.assertEqual(config.trainer.trainer_config.critic.optim.lr, 1e-5)
        self.assertEqual(config.trainer.trainer_config.critic.optim.weight_decay, 0.01)
        self.assertEqual(config.trainer.trainer_config.critic.optim.lr_decay_style, "constant")
        self.assertEqual(config.trainer.trainer_config.critic.optim.clip_grad, 1.0)

    def test_chat_template_path(self):
        config = get_template_config()
        config.model.chat_template_path = "tests/template/custom_chat_template.j2"
        config.check_and_update()
        self.assertIsNotNone(config.model.custom_chat_template)
        self.assertEqual(
            config.model.custom_chat_template,
            config.buffer.explorer_input.tasksets[0].format.chat_template,
        )
        self.assertEqual(
            config.model.custom_chat_template, config.explorer.rollout_model.chat_template
        )

    def tearDown(self):
        if os.path.exists(CHECKPOINT_ROOT_DIR):
            shutil.rmtree(CHECKPOINT_ROOT_DIR, ignore_errors=True)
