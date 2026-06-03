import asyncio
import shutil
from collections import deque

import torch
from parameterized import parameterized_class

from tests.tools import RayUnittestBaseAsync, get_template_config
from trinity.algorithm.sample_strategy import SAMPLE_STRATEGY
from trinity.algorithm.sample_strategy.sample_strategy import SampleStrategy
from trinity.buffer.buffer import get_buffer_writer
from trinity.common.config import ExperienceBufferConfig, ReplayBufferConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience


@parameterized_class(
    ("exp_write_batch_size",),
    [
        (3,),
        (6,),
    ],
)
class ExperienceStorageTest(RayUnittestBaseAsync):
    def setUp(self):
        self.config = get_template_config()
        self.num_steps = 20

    def _default_exp_list(self):
        return [
            [
                Experience(
                    tokens=torch.tensor([float(k) for k in range(j + 3)]),
                    reward=float(i),  # using reward to carry model_version for testing
                    prompt_length=2,
                    info={"model_version": i, "use_count": 0},
                )
                for j in range(self.exp_write_batch_size)
            ]
            for i in range(self.num_steps)
        ]

    def _default_steps(self):
        return [0, 5, 10, 15]

    def _init_buffer_writer_and_sample_strategy(self):
        # Initialize buffer writer and sample strategy
        self.buffer_writer = get_buffer_writer(
            self.config.buffer.trainer_input.experience_buffer,  # type: ignore [arg-type]
        )
        self.sample_strategy: SampleStrategy = SAMPLE_STRATEGY.get(
            self.config.algorithm.sample_strategy
        )(
            buffer_config=self.config.buffer,
            **self.config.algorithm.sample_strategy_args,
        )

    async def _verify_model_version(self, step, expected_versions):
        batch, metrics, _ = await self.sample_strategy.sample(step=step)
        self.assertEqual(
            [exp.reward for exp in batch],
            expected_versions,
            f"Model versions mismatch at step {step}",
        )
        self.assertEqual(
            metrics["sample/model_version/min"],
            min(expected_versions),
            f"Min model version mismatch at step {step}",
        )
        self.assertEqual(
            metrics["sample/model_version/max"],
            max(expected_versions),
            f"Max model version mismatch at step {step}",
        )
        self.assertEqual(
            metrics["sample/model_version/mean"],
            sum(expected_versions) / len(expected_versions),
            f"Mean model version mismatch at step {step}",
        )

    async def _verify_sampling_model_versions(self, exps_list, expected_model_versions_map):
        self._init_buffer_writer_and_sample_strategy()

        # Write experiences to buffer, while sample and validate model versions
        current_task = None
        for step, exps in enumerate(exps_list):
            await self.buffer_writer.write(exps)
            if step in expected_model_versions_map:
                if current_task:
                    await current_task
                current_task = asyncio.create_task(
                    self._verify_model_version(step, expected_model_versions_map[step])
                )
                await asyncio.sleep(0.1)

        if current_task:
            await current_task

    async def _flexible_verify_model_version(self, step, max_staleness):
        _, metrics, _ = await self.sample_strategy.sample(step=step)
        self.assertGreaterEqual(
            metrics["sample/model_version/min"],
            step - max_staleness,
            f"Min model version mismatch at step {step}",
        )

    async def _flexible_verify_sampling_model_versions(self, exps_list, check_steps, max_staleness):
        self._init_buffer_writer_and_sample_strategy()

        # Write experiences to buffer, while sample and validate model versions
        current_task = None
        for step, exps in enumerate(exps_list):
            await self.buffer_writer.write(exps)
            if step in check_steps:
                if current_task:
                    await current_task
                current_task = asyncio.create_task(
                    self._flexible_verify_model_version(step, max_staleness)
                )
                await asyncio.sleep(0.1)

        if current_task:
            await current_task

    async def test_default_queue_default_sample_strategy(self):
        self.config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="default_queue_default_strategy",
            storage_type=StorageType.QUEUE.value,
            replay_buffer=ReplayBufferConfig(enable=False),
        )
        self.config.check_and_update()

        # init testing data
        exps_list = self._default_exp_list()
        steps = self._default_steps()
        train_batch_size = self.config.buffer.train_batch_size
        expected_model_versions_map = {}
        for idx, step in enumerate(steps):
            start_idx = idx * train_batch_size
            batch_versions = [
                (start_idx + offset) // self.exp_write_batch_size
                for offset in range(train_batch_size)
            ]
            expected_model_versions_map[step] = batch_versions

        await self._verify_sampling_model_versions(exps_list, expected_model_versions_map)

    async def test_default_queue_staleness_control_sample_strategy(self):
        max_staleness = 3
        self.config.algorithm.sample_strategy = "staleness_control"
        self.config.algorithm.sample_strategy_args = {"max_staleness": max_staleness}
        self.config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="default_queue_staleness_control",
            storage_type=StorageType.QUEUE.value,
            replay_buffer=ReplayBufferConfig(enable=False),
        )
        self.config.check_and_update()

        # init testing data
        exps_list = self._default_exp_list()
        steps = self._default_steps()
        expected_model_versions_map = {}
        for step in steps:
            predict_version = max(step - max_staleness, 0)
            expected_model_versions_map[step] = [
                predict_version + i // self.exp_write_batch_size
                for i in range(self.config.buffer.train_batch_size)
            ]

        await self._verify_sampling_model_versions(exps_list, expected_model_versions_map)

    def _simulate_priority_queue(self, steps, max_staleness=float("inf")):
        expected_model_versions_map = {}
        buffer = deque()
        exp_pool = deque()
        step_idx = 0
        train_batch_size = self.config.buffer.train_batch_size
        for i in range(self.num_steps):
            buffer.append([i] * self.exp_write_batch_size)
            step = steps[step_idx]
            if i < step:
                continue
            batch_versions = expected_model_versions_map.get(step, [])
            if len(batch_versions) < train_batch_size:
                while len(buffer) > 0:
                    if len(exp_pool) == 0:
                        exp_pool.extend(buffer.pop())
                    while len(exp_pool) > 0 and len(batch_versions) < train_batch_size:
                        exp_version = exp_pool.popleft()
                        if exp_version < step - max_staleness:
                            continue
                        batch_versions.append(exp_version)
                    if len(batch_versions) >= train_batch_size:
                        step_idx += 1
                        break
                expected_model_versions_map[step] = batch_versions
            if step_idx >= len(steps):
                break
        return expected_model_versions_map

    async def test_priority_queue_default_sample_strategy(self):
        self.config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="priority_queue_default_strategy",
            storage_type=StorageType.QUEUE.value,
            replay_buffer=ReplayBufferConfig(enable=True),
        )
        self.config.check_and_update()

        # init testing data
        exps_list = self._default_exp_list()
        steps = self._default_steps()
        expected_model_versions_map = self._simulate_priority_queue(steps)

        await self._verify_sampling_model_versions(exps_list, expected_model_versions_map)

    async def test_priority_queue_staleness_control_sample_strategy(self):
        max_staleness = 2
        self.config.algorithm.sample_strategy = "staleness_control"
        self.config.algorithm.sample_strategy_args = {"max_staleness": max_staleness}
        self.config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="priority_queue_staleness_control",
            storage_type=StorageType.QUEUE.value,
            replay_buffer=ReplayBufferConfig(enable=True),
        )
        self.config.check_and_update()

        # init testing data
        exps_list = self._default_exp_list()
        steps = self._default_steps()
        expected_model_versions_map = self._simulate_priority_queue(steps, max_staleness)

        await self._verify_sampling_model_versions(exps_list, expected_model_versions_map)

    async def test_sql_staleness_control_sample_strategy(self):
        max_staleness = 2
        self.config.algorithm.sample_strategy = "staleness_control"
        self.config.algorithm.sample_strategy_args = {"max_staleness": max_staleness}
        self.config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="sql_staleness_control",
            storage_type=StorageType.SQL.value,
        )
        self.config.check_and_update()

        # init testing data
        exps_list = self._default_exp_list()
        steps = self._default_steps()

        await self._flexible_verify_sampling_model_versions(exps_list, steps, max_staleness)

    def tearDown(self):
        asyncio.run(self.buffer_writer.release())
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)
        return super().tearDown()
