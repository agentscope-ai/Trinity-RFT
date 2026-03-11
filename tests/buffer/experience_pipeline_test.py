import os
from typing import List

import ray
import torch

from tests.tools import RayUnittestBaseAsync, get_template_config
from trinity.buffer import get_buffer_reader
from trinity.buffer.pipelines.experience_pipeline import ExperiencePipeline
from trinity.common.config import (
    ExperienceBufferConfig,
    ExperiencePipelineConfig,
    OperatorConfig,
)
from trinity.common.constants import SELECTOR_METRIC
from trinity.common.experience import EID, Experience

BUFFER_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_pipeline_buffer.jsonl")


def get_experiences(task_num: int, repeat_times: int = 1, step_num: int = 1) -> List[Experience]:
    """Generate a list of experiences for testing."""
    return [
        Experience(
            eid=EID(task=i, run=j, step=k),
            tokens=torch.zeros((5,)),
            prompt_length=4,
            reward=j,
            logprobs=torch.tensor([0.1]),
        )
        for i in range(task_num)
        for j in range(repeat_times)
        for k in range(step_num)
    ]


class TestExperiencePipeline(RayUnittestBaseAsync):
    def setUp(self):
        if os.path.exists(BUFFER_FILE_PATH):
            os.remove(BUFFER_FILE_PATH)

    async def test_experience_pipeline(self):
        # test input cache
        config = get_template_config()
        config.data_processor.experience_pipeline = ExperiencePipelineConfig(
            save_input=True,
            input_save_path=BUFFER_FILE_PATH,
            operators=[
                OperatorConfig(
                    name="reward_filter",
                    args={"threshold": 0.5},
                )
            ],
        )
        config.algorithm.algorithm_type = "grpo"
        config.algorithm.advantage_fn = (
            "grpo"  # grpo will add an operator at the end of the pipeline
        )
        config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="pipeline_test_experience_buffer",
            max_read_timeout=3,
        )
        config.check_and_update()

        pipeline = (
            ray.remote(ExperiencePipeline)
            .options(name=f"{config.explorer.name}_pipeline")
            .remote(config)
        )
        await pipeline.prepare.remote()
        task_num = 8
        repeat_times = 4
        experiences = get_experiences(task_num=task_num, repeat_times=repeat_times)
        metrics = await pipeline.process.remote(Experience.serialize_many(experiences))
        self.assertEqual(
            metrics["experience_pipeline/experience_count"], task_num * (repeat_times - 1)
        )  # first experience of each task will be filtered out by the reward filter

        # tests
        reader = get_buffer_reader(config.buffer.trainer_input.experience_buffer)
        exps = await reader.read_async(batch_size=task_num * (repeat_times - 1))
        self.assertEqual(len(exps), task_num * (repeat_times - 1))
        with self.assertRaises(TimeoutError):
            await reader.read_async(batch_size=task_num)

        with open(config.data_processor.experience_pipeline.input_save_path, "r") as f:
            input_data = f.readlines()
        self.assertEqual(len(input_data), len(experiences))

    async def test_pass_rate_calculation(self) -> None:
        config = get_template_config()
        config.data_processor.experience_pipeline = ExperiencePipelineConfig(
            save_input=True,
            input_save_path=BUFFER_FILE_PATH,
            operators=[
                OperatorConfig(
                    name="pass_rate_calculator",
                )
            ],
        )
        config.check_and_update()
        config.buffer.trainer_input.experience_buffer.name = "pipeline_test_experience_buffer"
        config.buffer.trainer_input.experience_buffer.max_read_timeout = 3

        pipeline = (
            ray.remote(ExperiencePipeline)
            .options(name=f"{config.explorer.name}_pipeline")
            .remote(config)
        )
        await pipeline.prepare.remote()
        task_num = 8
        repeat_times = 4
        experiences = get_experiences(task_num=task_num, repeat_times=repeat_times)
        for exp in experiences:
            exp.info["task_index"] = {
                "taskset_id": 0,
                "index": exp.eid.task,
            }
        metrics = await pipeline.process.remote(Experience.serialize_many(experiences))
        self.assertIn(SELECTOR_METRIC, metrics)
        selector_metrics = metrics[SELECTOR_METRIC]
        self.assertEqual(len(selector_metrics), 1)
        self.assertEqual(set(selector_metrics[0]["indices"]), set(range(task_num)))
        self.assertEqual(selector_metrics[0]["values"], [(repeat_times - 1.0) / 2] * task_num)
