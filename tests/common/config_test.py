# -*- coding: utf-8 -*-
"""Test cases for Config modules."""
import datetime
import math
import os
import re
import shutil
import socket
import unittest
from unittest.mock import patch

import torch

from tests.tools import get_template_config, get_unittest_dataset_config
from trinity.common.config import (
    InferenceModelConfig,
    LaunchMode,
    infer_launch_mode,
    load_config,
)
from trinity.common.constants import SyncMethod
from trinity.common.models.model import InferenceModel
from trinity.trainer.trainer import is_verl_legacy
from trinity.trainer.verl.config import build_verl_config

CHECKPOINT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "temp_checkpoint_dir")


class DummyInferenceModel(InferenceModel):
    async def generate(self, prompt: str, **kwargs):
        raise NotImplementedError

    async def chat(self, messages, **kwargs):
        raise NotImplementedError

    async def logprobs(self, token_ids, **kwargs):
        raise NotImplementedError

    async def convert_messages_to_experience(self, messages, tools=None, temperature=None):
        raise NotImplementedError

    async def sync_model_weights(
        self, model_version: int, sync_method: SyncMethod, timeout: float = 1200
    ) -> int:
        return model_version

    def get_model_version(self) -> int:
        return 0


class TestConfig(unittest.TestCase):
    def test_inference_model_base_port_uses_engine_id(self):
        model = DummyInferenceModel(InferenceModelConfig(base_port=9000, engine_id=3))

        _, port = model.get_available_address()

        self.assertEqual(port, 9003)

    def test_inference_model_base_port_falls_back_when_unavailable(self):
        requested_port = 9004
        model = DummyInferenceModel(InferenceModelConfig(base_port=9000, engine_id=4))

        with socket.socket() as occupied_socket:
            occupied_socket.bind(("", requested_port))

            with patch.object(model.logger, "warning") as mock_warning:
                _, port = model.get_available_address()

        self.assertNotEqual(port, requested_port)
        self.assertGreater(port, 0)
        mock_warning.assert_called_once_with(
            "Configured port %s is unavailable for engine %s; falling back to an ephemeral port.",
            requested_port,
            4,
        )

    def test_inference_model_without_base_port_uses_ephemeral_port(self):
        model = DummyInferenceModel(InferenceModelConfig())

        _, port = model.get_available_address()

        self.assertGreater(port, 0)

    def test_inference_model_random_port_ignores_base_port(self):
        requested_port = 9005
        model = DummyInferenceModel(InferenceModelConfig(base_port=9000, engine_id=5))

        _, port = model.get_available_address(random_port=True)

        self.assertNotEqual(port, requested_port)
        self.assertGreater(port, 0)

    def test_inference_model_random_port_can_use_port_reserved_by_api_server(self):
        requested_port = 9006
        model = DummyInferenceModel(InferenceModelConfig(base_port=9000, engine_id=6))

        with socket.socket() as occupied_socket:
            occupied_socket.bind(("", requested_port))

            _, port = model.get_available_address(random_port=True)

        self.assertNotEqual(port, requested_port)
        self.assertGreater(port, 0)

    def test_multinode_vllm_config_is_valid(self):
        config = get_template_config()
        config.mode = "explore"
        config.cluster.node_num = 2
        config.cluster.gpu_per_node = 4
        config.explorer.rollout_model.engine_type = "vllm"
        config.explorer.rollout_model.engine_num = 1
        config.explorer.rollout_model.tensor_parallel_size = 8
        config.explorer.rollout_model.nnodes = 2

        config.check_and_update()

        self.assertEqual(config.explorer.rollout_model.nnodes, 2)

    def test_multinode_vllm_requires_nnodes_within_cluster_size(self):
        config = get_template_config()
        config.mode = "explore"
        config.cluster.node_num = 2
        config.cluster.gpu_per_node = 4
        config.explorer.rollout_model.engine_type = "vllm"
        config.explorer.rollout_model.engine_num = 1
        config.explorer.rollout_model.tensor_parallel_size = 8
        config.explorer.rollout_model.nnodes = 3

        with self.assertRaisesRegex(ValueError, "cannot exceed cluster.node_num"):
            config.check_and_update()

    def test_multinode_vllm_requires_full_node_occupancy(self):
        config = get_template_config()
        config.mode = "explore"
        config.cluster.node_num = 3
        config.cluster.gpu_per_node = 4
        config.explorer.rollout_model.engine_type = "vllm"
        config.explorer.rollout_model.engine_num = 1
        config.explorer.rollout_model.tensor_parallel_size = 6
        config.explorer.rollout_model.nnodes = 2

        with self.assertRaisesRegex(ValueError, "integer multiple of cluster.gpu_per_node"):
            config.check_and_update()

    def test_multinode_vllm_requires_matching_nnodes_for_full_nodes(self):
        config = get_template_config()
        config.mode = "explore"
        config.cluster.node_num = 4
        config.cluster.gpu_per_node = 4
        config.explorer.rollout_model.engine_type = "vllm"
        config.explorer.rollout_model.engine_num = 1
        config.explorer.rollout_model.tensor_parallel_size = 8
        config.explorer.rollout_model.nnodes = 4

        with self.assertRaisesRegex(
            ValueError,
            re.escape("In HEADLESS mode, `nnodes` (4) must equal (TP*PP) // gpu_per_node (2)."),
        ):
            config.check_and_update()

    def test_multinode_inference_is_rejected_for_non_vllm_sglang_engines(self):
        config = get_template_config()
        config.mode = "explore"
        config.cluster.node_num = 2
        config.cluster.gpu_per_node = 4
        config.explorer.rollout_model.engine_type = "tinker"
        config.explorer.rollout_model.engine_num = 1
        config.explorer.rollout_model.tensor_parallel_size = 4
        config.explorer.rollout_model.nnodes = 2

        with self.assertRaisesRegex(ValueError, "only supported for"):
            config.check_and_update()

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
        self.assertEqual(
            config.buffer.explorer_input.tasksets[0].repeat_times, config.algorithm.repeat_times
        )
        self.assertEqual(config.model.model_path, config.model.critic_model_path)
        self.assertEqual(config.model.model_path, config.explorer.rollout_model.model_path)

        if is_verl_legacy():
            self.assertIsNotNone(config.trainer.trainer_config)
            self.assertEqual(config.trainer.trainer_config.trainer.n_gpus_per_node, 8)
            self.assertEqual(config.trainer.trainer_config.trainer.nnodes, 1)
            self.assertEqual(config.trainer.trainer_config.trainer.project_name, config.project)
            self.assertEqual(config.trainer.trainer_config.trainer.experiment_name, config.name)
            return

        verl_config = build_verl_config(config)
        self.assertEqual(verl_config.model.path, config.model.model_path)
        self.assertEqual(verl_config.actor.strategy, config.trainer.trainer_strategy)
        self.assertEqual(verl_config.actor.ppo_mini_batch_size, config.buffer.train_batch_size)
        self.assertEqual(verl_config.actor.rollout_n, config.algorithm.repeat_times)
        self.assertEqual(verl_config.rollout.n, config.algorithm.repeat_times)

    def test_all_examples_are_valid(self):
        example_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        for example_name in os.listdir(example_dir):
            for filename in os.listdir(os.path.join(example_dir, example_name)):
                if filename.endswith(".yaml") and not (
                    filename.startswith("train_")
                    or filename.startswith("verl_")
                    or filename.startswith("dj_")
                    or filename.startswith("tinker")
                    or filename.startswith("external")
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
        expected_max_token_len = math.ceil(
            (2 * config.model.max_model_len) / config.trainer.ulysses_sequence_parallel_size
        )
        self.assertEqual(config.trainer.max_token_len_per_gpu, expected_max_token_len)

        if is_verl_legacy():
            self.assertIsNotNone(config.trainer.trainer_config)
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
            return

        verl_config = build_verl_config(config)
        self.assertEqual(verl_config.actor.ppo_max_token_len_per_gpu, expected_max_token_len)
        self.assertEqual(verl_config.actor.ppo_infer_max_token_len_per_gpu, expected_max_token_len)
        self.assertEqual(verl_config.ref.log_prob_max_token_len_per_gpu, expected_max_token_len)
        self.assertEqual(verl_config.rollout.log_prob_max_token_len_per_gpu, expected_max_token_len)
        self.assertEqual(verl_config.critic.ppo_max_token_len_per_gpu, expected_max_token_len)
        self.assertEqual(
            verl_config.critic.ppo_infer_max_token_len_per_gpu,
            expected_max_token_len,
        )
        self.assertEqual(
            verl_config.critic.forward_max_token_len_per_gpu,
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
        if is_verl_legacy():
            self.assertEqual(config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr, 1e-4)
            self.assertEqual(
                config.trainer.trainer_config.actor_rollout_ref.actor.optim.weight_decay, 0.05
            )
            self.assertEqual(
                config.trainer.trainer_config.actor_rollout_ref.actor.optim.clip_grad, 2.0
            )
            self.assertEqual(
                config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr_decay_steps, 1000
            )
            self.assertEqual(
                config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr_decay_style,
                "cosine",
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
                    torch.tensor(
                        config.trainer.trainer_config.actor_rollout_ref.actor.optim.min_lr
                    ),
                    torch.tensor(1e-6),
                )
            )
            # critic optimizer should not be affected
            self.assertEqual(config.trainer.trainer_config.critic.optim.lr, 1e-5)
            self.assertEqual(config.trainer.trainer_config.critic.optim.weight_decay, 0.01)
            self.assertEqual(config.trainer.trainer_config.critic.optim.lr_decay_style, "constant")
            self.assertEqual(config.trainer.trainer_config.critic.optim.clip_grad, 1.0)
            return

        verl_config = build_verl_config(config)
        self.assertEqual(verl_config.actor.optim.lr, 1e-4)
        self.assertEqual(verl_config.actor.optim.weight_decay, 0.05)
        self.assertEqual(verl_config.actor.optim.clip_grad, 2.0)
        self.assertEqual(verl_config.actor.optim.total_training_steps, 1000)

        if config.trainer.trainer_strategy.startswith("fsdp"):
            self.assertEqual(verl_config.actor.optim.lr_scheduler_type, "cosine")
            self.assertEqual(verl_config.actor.optim.min_lr_ratio, 1e-2)
            self.assertEqual(verl_config.critic.optim.lr_scheduler_type, "constant")
            self.assertEqual(verl_config.critic.optim.min_lr_ratio, 0.01)
        else:
            self.assertEqual(verl_config.actor.optim.lr_decay_steps, 1000)
            self.assertEqual(verl_config.actor.optim.lr_decay_style, "cosine")
            self.assertTrue(
                torch.allclose(
                    torch.tensor(verl_config.actor.optim.lr_warmup_init), torch.tensor(1e-6)
                )
            )
            self.assertTrue(
                torch.allclose(torch.tensor(verl_config.actor.optim.min_lr), torch.tensor(1e-6))
            )
            self.assertEqual(verl_config.critic.optim.lr_decay_style, "constant")
            self.assertTrue(
                torch.allclose(torch.tensor(verl_config.critic.optim.min_lr), torch.tensor(0.0))
            )

        self.assertEqual(verl_config.critic.optim.lr, 1e-5)
        self.assertEqual(verl_config.critic.optim.weight_decay, 0.01)
        self.assertEqual(verl_config.critic.optim.clip_grad, 1.0)

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


class TestLaunchMode(unittest.TestCase):
    """Test cases for LaunchMode enum and infer_launch_mode function."""

    def test_single_node_default(self):
        """nnodes=1 should always be SINGLE_NODE regardless of DP size."""
        self.assertEqual(infer_launch_mode(1, 1), LaunchMode.SINGLE_NODE)
        self.assertEqual(infer_launch_mode(1, 2), LaunchMode.SINGLE_NODE)
        self.assertEqual(infer_launch_mode(1, 4), LaunchMode.SINGLE_NODE)

    def test_headless_mode(self):
        """nnodes>1 with DP=1 should be HEADLESS (cross-node TP/PP)."""
        self.assertEqual(infer_launch_mode(2, 1), LaunchMode.HEADLESS)
        self.assertEqual(infer_launch_mode(4, 1), LaunchMode.HEADLESS)

    def test_dp_with_multinode_is_single_node(self):
        """nnodes>1 with DP>1 should be SINGLE_NODE (DP expansion handled by allocator)."""
        self.assertEqual(infer_launch_mode(2, 2), LaunchMode.SINGLE_NODE)
        self.assertEqual(infer_launch_mode(4, 4), LaunchMode.SINGLE_NODE)
        self.assertEqual(infer_launch_mode(2, 3), LaunchMode.SINGLE_NODE)

    def test_launch_mode_values(self):
        """LaunchMode enum values should be stable strings."""
        self.assertEqual(LaunchMode.SINGLE_NODE.value, "single_node")
        self.assertEqual(LaunchMode.HEADLESS.value, "headless")

    def test_launch_mode_string_comparison(self):
        """LaunchMode should support string comparison since it inherits from str."""
        self.assertEqual(LaunchMode.SINGLE_NODE, "single_node")
        self.assertEqual(LaunchMode.HEADLESS, "headless")

    def test_inference_model_config_has_launch_mode_fields(self):
        """InferenceModelConfig should have launch_mode and data_parallel_rank fields."""
        config = InferenceModelConfig()
        self.assertIsNone(config.launch_mode)
        self.assertEqual(config.data_parallel_rank, 0)


class TestMultinodeValidation(unittest.TestCase):
    """Test cases for multi-node inference validation with LaunchMode."""

    def _make_config(
        self, nnodes=1, dp=1, tp=1, pp=1, engine_type="vllm", gpu_per_node=8, node_num=4
    ):
        config = get_template_config()
        config.mode = "explore"
        config.cluster.gpu_per_node = gpu_per_node
        config.cluster.node_num = node_num
        config.explorer.rollout_model.engine_type = engine_type
        config.explorer.rollout_model.nnodes = nnodes
        config.explorer.rollout_model.data_parallel_size = dp
        config.explorer.rollout_model.tensor_parallel_size = tp
        config.explorer.rollout_model.pipeline_parallel_size = pp
        config.explorer.rollout_model.engine_num = 1
        return config

    def test_single_node_no_validation_error(self):
        """nnodes=1 should pass validation without errors."""
        config = self._make_config(nnodes=1, dp=2, tp=2, pp=1)
        config.check_and_update()
        self.assertEqual(config.explorer.rollout_model.launch_mode, LaunchMode.SINGLE_NODE.value)

    def test_headless_mode_valid(self):
        """HEADLESS mode with correct nnodes should pass."""
        config = self._make_config(nnodes=2, dp=1, tp=8, pp=2, gpu_per_node=8)
        config.check_and_update()
        self.assertEqual(config.explorer.rollout_model.launch_mode, LaunchMode.HEADLESS.value)

    def test_headless_mode_wrong_nnodes(self):
        """HEADLESS mode with wrong nnodes should fail."""
        config = self._make_config(nnodes=3, dp=1, tp=8, pp=1, gpu_per_node=8, node_num=8)
        with self.assertRaises(ValueError):
            config.check_and_update()

    def test_dp_expansion_valid(self):
        """SINGLE_NODE with nnodes=dp_size and TP*PP<=gpu_per_node should pass."""
        config = self._make_config(nnodes=2, dp=2, tp=4, pp=1, gpu_per_node=8)
        config.check_and_update()
        self.assertEqual(config.explorer.rollout_model.launch_mode, LaunchMode.SINGLE_NODE.value)

    def test_dp_expansion_wrong_nnodes(self):
        """SINGLE_NODE with nnodes != dp_size should fail."""
        config = self._make_config(nnodes=3, dp=2, tp=2, pp=1, gpu_per_node=8)
        with self.assertRaises(ValueError):
            config.check_and_update()

    def test_dp_expansion_tp_too_large(self):
        """SINGLE_NODE with TP*PP > gpu_per_node should fail."""
        config = self._make_config(nnodes=2, dp=2, tp=16, pp=1, gpu_per_node=8)
        with self.assertRaises(ValueError):
            config.check_and_update()

    def test_sglang_cross_node_dp_rejected(self):
        """SGLang with nnodes>1 and DP>1 should be rejected."""
        config = self._make_config(nnodes=2, dp=2, tp=2, pp=1, engine_type="sglang")
        with self.assertRaises(ValueError) as ctx:
            config.check_and_update()
        self.assertIn("SGLang", str(ctx.exception))
