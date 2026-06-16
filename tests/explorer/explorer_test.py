"""Tests for explorer."""
import asyncio
import json
import multiprocessing
import os
import random
import shutil
import time
import unittest
from collections import deque
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import ray

from tests.tools import (
    RayUnittestBase,
    RayUnittestBaseAsync,
    TensorBoardParser,
    get_api_model_path,
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.buffer import get_buffer_reader
from trinity.buffer.operators import ExperienceOperatorV1
from trinity.cli.launcher import explore, run_stage
from trinity.common.config import (
    ExperienceBufferConfig,
    InferenceModelConfig,
    OperatorConfig,
)
from trinity.common.constants import StorageType
from trinity.explorer.explorer import Explorer
from trinity.explorer.proxy.client import TrinityClient
from trinity.manager.state_manager import StateManager


def _build_fake_coordinator_explorer():
    class FakeRemoteMethod:
        def __init__(self, func):
            self.func = func

        async def remote(self, *args, **kwargs):
            return await self.func(*args, **kwargs)

    class FakeCoordinator:
        def __init__(self):
            self.submit_calls = []
            self.finalize_train_calls = []
            self.finalize_eval_calls = []
            self.shutdown_calls = 0
            self.submit_batch = FakeRemoteMethod(self._submit_batch)
            self.finalize_train_batch = FakeRemoteMethod(self._finalize_train_batch)
            self.finalize_eval_batch = FakeRemoteMethod(self._finalize_eval_batch)
            self.shutdown = FakeRemoteMethod(self._shutdown)

        async def _submit_batch(self, **kwargs):
            self.submit_calls.append(kwargs)

        async def _finalize_train_batch(self, batch_id):
            self.finalize_train_calls.append(batch_id)
            return {
                "batch_id": batch_id,
                "batch_type": "train",
                "finished_task_count": 1,
                "metrics": {
                    "experience_pipeline/experience_count": 2.0,
                    "rollout/run_metrics/mean": float(batch_id),
                    "rollout/finished_task_count": 1.0,
                },
            }

        async def _finalize_eval_batch(self, batch_id):
            self.finalize_eval_calls.append(batch_id)
            eval_name = batch_id.split("/", 1)[1]
            return {
                "batch_id": batch_id,
                "batch_type": "eval",
                "finished_task_count": 2,
                "metrics": {
                    f"eval/{eval_name}/accuracy": 0.5,
                    f"eval/{eval_name}/finished_task_count": 2.0,
                },
            }

        async def _shutdown(self):
            self.shutdown_calls += 1

    class FakeMonitor:
        def __init__(self):
            self.logged = []

        def log(self, metric, step):
            self.logged.append((step, metric))

    feedback_calls = []

    async def read():
        return [SimpleNamespace(is_eval=False), SimpleNamespace(is_eval=False)]

    def record_feedback(metrics):
        feedback_calls.append(metrics)

    explorer = Explorer.__new__(Explorer)
    explorer.logger = MagicMock()
    explorer.rollout_coordinator = FakeCoordinator()
    explorer.monitor = FakeMonitor()
    explorer.taskset = SimpleNamespace(read=read, feedback=record_feedback)
    explorer.min_wait_num = None
    explorer.pending_eval_tasks = deque()
    explorer.explore_start_time = None
    explorer.eval_start_time = None
    explorer.last_monitored_step = 0
    explorer.explore_step_num = 0
    explorer.model_version = 7
    explorer.detailed_stats = False
    explorer.config = SimpleNamespace(
        explorer=SimpleNamespace(
            over_rollout=SimpleNamespace(return_partial_tasks=False),
            eval_interval=1,
        )
    )
    return explorer, feedback_calls


class BaseExplorerCase(RayUnittestBase):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.buffer.total_epochs = 2
        self.config.buffer.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.algorithm.repeat_times = 2
        self.config.monitor.monitor_type = "tensorboard"
        self.config.project = "Trinity-unittest"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 2
        self.config.explorer.eval_interval = 4
        self.config.monitor.detailed_stats = False


class TestExplorerCountdownEval(BaseExplorerCase):
    def test_explorer(self):
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        eval_tasksets = self.config.buffer.explorer_input.eval_tasksets
        eval_tasksets.extend(
            [
                get_unittest_dataset_config("countdown", "test"),
                get_unittest_dataset_config("eval_short"),
                get_unittest_dataset_config("eval_long"),
            ]
        )
        eval_tasksets[1].repeat_times = 6
        eval_tasksets[2].repeat_times = 10
        self.config.name = f"explore-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 8)
        for eval_taskset, k_list in zip(eval_tasksets, [[1], [2, 4, 6], [2, 4, 8, 10]]):
            metric_name = "score" if eval_taskset.name == "countdown" else "accuracy"
            repeat_times = k_list[-1]
            expected_stat_suffixes = [f"mean@{repeat_times}", f"std@{repeat_times}"]
            for k in k_list:
                if k == 1:
                    continue
                expected_stat_suffixes.extend([f"best@{k}", f"worst@{k}"])
            # only return the mean of the column
            for stat_suffix in expected_stat_suffixes:
                self.assertIn(
                    f"eval/{eval_taskset.name}/{metric_name}/{stat_suffix}",
                    eval_metrics,
                )


class TestExplorerEvalDetailedStats(BaseExplorerCase):
    def test_explorer(self):
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.monitor.detailed_stats = True
        eval_taskset = get_unittest_dataset_config("eval_short")
        eval_taskset.repeat_times = 6
        self.config.buffer.explorer_input.eval_tasksets = [eval_taskset]
        self.config.name = f"explore-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 8)
        metric_name, repeat_times, k_list = "accuracy", 6, [2, 4, 6]
        expected_stat_suffixes = [f"mean@{repeat_times}", f"std@{repeat_times}"]
        for k in k_list:  # k_list does not include 1
            expected_stat_suffixes.extend([f"best@{k}", f"worst@{k}"])
        # test detailed stats
        for stat_suffix in expected_stat_suffixes:
            for stats in ["mean", "std", "max", "min"]:
                self.assertIn(
                    f"eval/{eval_taskset.name}/{metric_name}/{stat_suffix}/{stats}",
                    eval_metrics,
                )


class DummyOperatorWithAuxiliaryModel(ExperienceOperatorV1):
    def __init__(self) -> None:
        super().__init__()

    async def prepare(self) -> None:
        import openai

        await super().prepare()
        # make sure the auxiliary model wrapper is correctly passed
        assert len(self.auxiliary_models) == 1
        assert "aux_model" in self.auxiliary_models
        assert len(self.auxiliary_models["aux_model"]) == 2
        assert isinstance(self.auxiliary_models["aux_model"][0], openai.AsyncOpenAI)

    async def process(self, exps: list) -> tuple:
        # call the auxiliary model to make sure the model wrapper is correctly passed to the operator
        messages = [{"role": "user", "content": "Hello"}]
        responses = []
        for model in self.auxiliary_models["aux_model"]:
            response = await model.chat.completions.create(
                model=model.model_path, messages=messages
            )
            responses.append(response)
        return exps, {}


class TestExplorerGSM8KRULERNoEval(BaseExplorerCase):
    def test_explorer(self):
        self.config.explorer.rollout_model.engine_num = 2
        self.config.explorer.auxiliary_models = [
            InferenceModelConfig(
                name="aux_model",
                model_path=get_api_model_path(),
                tensor_parallel_size=1,
                engine_num=2,
            )
        ]
        self.config.algorithm.repeat_times = 2
        self.config.buffer.total_steps = 2
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k_ruler")
        self.config.name = f"explore-no-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {
            "std_threshold": 0.0001,
        }
        self.config.data_processor.experience_pipeline.operators.append(
            OperatorConfig(
                name="tests.explorer.explorer_test.DummyOperatorWithAuxiliaryModel",
                args={},
            )
        )
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) == 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)


class TestExplorerGSM8k(BaseExplorerCase):
    def test_explorer(self):
        self.config.algorithm.repeat_times = 2
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.name = f"explore-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # some step may be skipped due to same reward
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {
            "epsilon": 1e-6,
        }
        self.config.model.max_model_len = 10240
        self.config.model.max_response_tokens = 8192
        self.config.model.min_response_tokens = 8192
        self.config.explorer.rollout_model.ignore_eos = True
        self.config.check_and_update()
        explorer = Explorer.get_actor(self.config)
        ray.get(explorer.prepare.remote())
        ray.get(explorer.sync_weight.remote())
        ray.get(explorer.explore.remote())
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) == 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        self.assertTrue(parser.metric_exist("experience_pipeline/experience_count"))
        experience_counts = parser.metric_values("experience_pipeline/experience_count")
        self.assertTrue(len(experience_counts) == 4)
        for count in experience_counts:
            self.assertTrue(count >= 0)
            self.assertTrue(count <= 2 * 4)  # repeat_times * batch_size
            self.assertTrue(count % 2 == 0)  # should be multiple of repeat_times
        exp_save_path = self.config.buffer.trainer_input.experience_buffer.path
        with open(exp_save_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertTrue(len(lines) <= 4 * 2 * 4)  # step * repeat_times * batch_size
            self.assertTrue(len(lines) % (2 * 4) == 0)
            exp = json.loads(lines[0])
            self.assertEqual(exp["response_length"], 8192)
        ray.get(explorer.shutdown.remote())


class TestExplorerCoordinatorPath(unittest.IsolatedAsyncioTestCase):
    async def test_explore_step_submits_train_batch_to_rollout_coordinator(self):
        explorer, _ = _build_fake_coordinator_explorer()

        should_continue = await explorer.explore_step()

        self.assertTrue(should_continue)
        self.assertEqual(explorer.explore_step_num, 1)
        self.assertEqual(
            explorer.rollout_coordinator.submit_calls,
            [
                {
                    "batch_id": 1,
                    "tasks": [SimpleNamespace(is_eval=False), SimpleNamespace(is_eval=False)],
                    "batch_type": "train",
                    "min_wait_num": None,
                }
            ],
        )

    async def test_finish_current_steps_uses_rollout_coordinator_finalize(self):
        explorer, feedback_calls = _build_fake_coordinator_explorer()
        explorer.explore_step_num = 2

        await explorer.finish_current_steps()

        self.assertEqual(explorer.rollout_coordinator.finalize_train_calls, [1, 2])
        self.assertEqual(len(feedback_calls), 2)
        self.assertEqual([step for step, _ in explorer.monitor.logged], [1, 2])
        self.assertEqual(explorer.last_monitored_step, 2)

    async def test_finish_eval_step_uses_rollout_coordinator_finalize(self):
        explorer, _ = _build_fake_coordinator_explorer()
        explorer.pending_eval_tasks.append((3, "eval_set"))
        explorer.eval_start_time = time.time()

        await explorer._finish_eval_step(step=3)

        self.assertEqual(explorer.rollout_coordinator.finalize_eval_calls, ["3/eval_set"])
        self.assertEqual([step for step, _ in explorer.monitor.logged], [3])
        self.assertIn("eval/eval_set/accuracy", explorer.monitor.logged[0][1])


class TestExplorerCoordinatorPolicies(unittest.IsolatedAsyncioTestCase):
    async def test_over_rollout_submits_partial_finalize_policy_to_rollout_coordinator(self):
        explorer, _ = _build_fake_coordinator_explorer()
        explorer.min_wait_num = 1
        explorer.config.explorer.over_rollout.return_partial_tasks = True

        should_continue = await explorer.explore_step()

        self.assertTrue(should_continue)
        self.assertEqual(
            explorer.rollout_coordinator.submit_calls,
            [
                {
                    "batch_id": 1,
                    "tasks": [SimpleNamespace(is_eval=False), SimpleNamespace(is_eval=False)],
                    "batch_type": "train",
                    "min_wait_num": 1,
                }
            ],
        )


def run_serve(config):
    config.check_and_update()
    run_stage(config)


def run_agent(proxy_url, model_path: str, stream: bool):
    proxy_client = TrinityClient(proxy_url=proxy_url)
    openai_client = proxy_client.get_openai_client()
    contents = [
        "Hello, how are you?",
        "What is the capital of China?",
        "Tell me a joke.",
        "Explain the theory of relativity.",
        "What is the meaning of life?",
        "How does a computer work?",
        "What is the weather like today?",
        "Can you recommend a good book?",
        "What is the best way to learn programming?",
        "Describe the process of photosynthesis.",
    ]
    if stream:
        stream_response = openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": random.choice(contents)}],
            stream=True,
        )
        response_id = None
        text_parts = []
        for chunk in stream_response:
            if response_id is None and getattr(chunk, "id", None):
                response_id = chunk.id
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)

        if response_id is not None:
            proxy_client.feedback(reward=2.0, msg_ids=[response_id])
        return "".join(text_parts)
    else:
        response = openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": random.choice(contents)}],
            stream=False,
        )
        proxy_client.feedback(reward=2.0, msg_ids=[response.id])
        return response.choices[0].message.content


class ServeTest(RayUnittestBaseAsync):
    def setUp(self):
        self.config = get_template_config()
        self.config.name = f"explorer-test-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.mode = "serve"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.algorithm.repeat_times = 1
        self.config.monitor.monitor_type = "tensorboard"
        self.config.project = "Trinity-unittest"
        self.config.explorer.rollout_model.engine_num = 4
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.explorer.proxy_port = 8010
        self.config.explorer.service_status_check_interval = 30
        self.config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="experience_buffer",
            storage_type=StorageType.SQL.value,
        )
        self.config.check_and_update()
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

    async def test_serve(self):  # noqa: C901
        serve_process = multiprocessing.Process(target=run_serve, args=(self.config,))
        serve_process.start()
        await asyncio.sleep(10)

        state_manager = StateManager(
            path=self.config.checkpoint_job_dir,
            explorer_name=self.config.explorer.name,
        )

        # wait for explorer initialization
        for i in range(50):
            try:
                server_url = state_manager.load_explorer_server_url()
            except Exception:
                server_url = None
            if server_url:
                break
            await asyncio.sleep(3)
        if not server_url:
            raise RuntimeError("Explorer server URL not found.")
        # wait for server setup
        for i in range(10):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{server_url}/health")
                    if response.status_code == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(2)

        task_num = 10
        apps = []
        for i in range(task_num):
            app_process = multiprocessing.Process(
                target=run_agent, args=(server_url, self.config.model.model_path, i % 2 == 0)
            )
            apps.append(app_process)
            app_process.start()

        for app in apps:
            app.join(timeout=60)
            self.assertFalse(app.is_alive())

        finish_step = None
        proxy_client = TrinityClient(proxy_url=server_url)
        for i in range(20):
            metrics = await proxy_client.get_metrics_async()
            metrics_keys = list(metrics.keys())
            self.assertIn("explore_step_num", metrics_keys)
            self.assertIn("rollout/total_experience_count", metrics_keys)
            self.assertIn("rollout/model_0/total_request_count", metrics_keys)
            self.assertIn("rollout/model_3/model_version", metrics_keys)
            if not finish_step and metrics["rollout/total_experience_count"] == task_num:
                finish_step = metrics["explore_step_num"]
                await proxy_client.commit_async()
            if finish_step and metrics["explore_step_num"] >= finish_step + 1:
                # wait for one more step to ensure all data are written to buffer
                break
            await asyncio.sleep(3)

        # check buffer
        self.config.buffer.trainer_input.experience_buffer.max_read_timeout = 5
        buffer_reader = get_buffer_reader(
            self.config.buffer.trainer_input.experience_buffer,
        )
        exps = await buffer_reader.read(batch_size=10)
        for exp in exps:
            self.assertTrue(len(exp.tokens) > 0)
            self.assertTrue(len(exp.logprobs) > 0)
            self.assertTrue(exp.prompt_length > 0)
            self.assertTrue(exp.reward == 2.0)
        self.assertEqual(len(exps), task_num)
        serve_process.terminate()
        serve_process.join(timeout=10)

    def tearDown(self):
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)
