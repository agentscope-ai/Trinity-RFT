# -*- coding: utf-8 -*-
"""Test for the general step-wise workflow module"""
import asyncio
import unittest
from dataclasses import dataclass, field
from typing import Dict, Optional
from unittest.mock import MagicMock

from parameterized import parameterized
from torch import Tensor

from tests.tools import get_unittest_dataset_config
from trinity.common.experience import EID, Experience
from trinity.common.workflows.step_wise_workflow import (
    AsyncRewardPropagationWorkflow,
    AsyncStepWiseRewardWorkflow,
    RewardPropagationWorkflow,
    StepWiseRewardWorkflow,
)
from trinity.common.workflows.workflow import Task


@dataclass
class MockResponse:
    response_text: str
    reward: float = 0.0
    metrics: Optional[Dict[str, float]] = None
    info: Optional[Dict] = None
    unique_id: Optional[str] = "0"
    tokens: Optional[Tensor] = Tensor([0, 0])
    prompt_length: int = 1
    eid: EID = field(default_factory=EID)


class DummyStepWiseRewardWorkflow(StepWiseRewardWorkflow):
    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.repeat_times = task.repeat_times
        self.max_env_steps = task.workflow_args.get("max_env_steps", 1)
        self.actual_steps = task.workflow_args.get("actual_steps", 1)

    def step(self, step_num: int):
        return step_num < self.actual_steps - 1

    def reward(self, exps: list[Experience], step_num: int):
        return 0.1 * step_num

    @property
    def max_step_num(self):
        return self.max_env_steps


class DummyAsyncStepWiseRewardWorkflow(AsyncStepWiseRewardWorkflow):
    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.repeat_times = task.repeat_times
        self.max_env_steps = task.workflow_args.get("max_env_steps", 1)
        self.actual_steps = task.workflow_args.get("actual_steps", 1)

    async def step_async(self, step_num: int):
        await asyncio.sleep(0.1)
        return step_num < self.actual_steps - 1

    async def reward_async(self, exps: list[Experience], step_num: int):
        await asyncio.sleep(0.1)
        return 0.1 * step_num

    @property
    def max_step_num(self):
        return self.max_env_steps


class DummyRewardPropagationWorkflow(RewardPropagationWorkflow):
    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.repeat_times = task.repeat_times
        self.max_env_steps = task.workflow_args.get("max_env_steps", 1)
        self.actual_steps = task.workflow_args.get("actual_steps", 1)

    def step(self, step_num: int):
        return step_num < self.actual_steps - 1

    def reward(self, exps: list[Experience]):
        return 0.1 * len(exps)

    @property
    def max_step_num(self):
        return self.max_env_steps


class DummyAsyncRewardPropagationWorkflow(AsyncRewardPropagationWorkflow):
    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.repeat_times = task.repeat_times
        self.max_env_steps = task.workflow_args.get("max_env_steps", 1)
        self.actual_steps = task.workflow_args.get("actual_steps", 1)

    async def step_async(self, step_num: int):
        await asyncio.sleep(0.1)
        return step_num < self.actual_steps - 1

    async def reward_async(self, exps: list[Experience]):
        await asyncio.sleep(0.1)
        return 0.1 * len(exps)

    @property
    def max_step_num(self):
        return self.max_env_steps


_dummy_workflows = [
    DummyStepWiseRewardWorkflow,
    DummyAsyncStepWiseRewardWorkflow,
    DummyRewardPropagationWorkflow,
    DummyAsyncRewardPropagationWorkflow,
]


class WorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MagicMock()
        self.model.enable_history = True
        self.model.extract_experience_from_history.side_effect = lambda: [
            MockResponse(f"The answer is \\boxed{42}")
        ]
        self.taskset_config = get_unittest_dataset_config("countdown")

    @parameterized.expand([(DummyStepWiseRewardWorkflow,), (DummyAsyncStepWiseRewardWorkflow,)])
    def test_step_wise_reward_workflow(self, workflow_cls) -> None:
        task = Task(
            workflow=workflow_cls,
            repeat_times=self.taskset_config.repeat_times,
            workflow_args={"max_env_steps": 10, "actual_steps": 5},
        )
        workflow = task.to_workflow(model=self.model)
        if workflow.is_async:
            experiences = asyncio.run(workflow.run_async())
        else:
            experiences = workflow.run()

        self.assertEqual(len(experiences), 5)
        actual_steps = [exp.eid.step for exp in experiences]
        self.assertEqual(actual_steps, [0, 1, 2, 3, 4])
        actual_rewards = [exp.reward for exp in experiences]
        expected_rewards = [0.0, 0.1, 0.2, 0.3, 0.4]
        for actual, expected in zip(actual_rewards, expected_rewards):
            self.assertAlmostEqual(actual, expected)  # type: ignore

    @parameterized.expand(
        [(DummyRewardPropagationWorkflow,), (DummyAsyncRewardPropagationWorkflow,)]
    )
    def test_reward_propagation_workflow(self, workflow_cls) -> None:
        task = Task(
            workflow=workflow_cls,
            repeat_times=self.taskset_config.repeat_times,
            workflow_args={"max_env_steps": 10, "actual_steps": 5},
        )
        workflow = task.to_workflow(model=self.model)
        if workflow.is_async:
            experiences = asyncio.run(workflow.run_async())
        else:
            experiences = workflow.run()

        self.assertEqual(len(experiences), 5)
        actual_steps = [exp.eid.step for exp in experiences]
        self.assertEqual(actual_steps, [0, 1, 2, 3, 4])
        expected_reward = 0.5
        for exp in experiences:
            self.assertAlmostEqual(exp.reward, expected_reward)  # type: ignore

    def test_workflows_stop_at_max_env_steps(self) -> None:
        for workflow_cls in _dummy_workflows:
            task = Task(
                workflow=workflow_cls,
                repeat_times=self.taskset_config.repeat_times,
                workflow_args={"max_env_steps": 3, "actual_steps": 100},  # actual > max
            )
            workflow = task.to_workflow(model=self.model)
            if workflow.is_async:
                experiences = asyncio.run(workflow.run_async())  # type: ignore
            else:
                experiences = workflow.run()
            self.assertEqual(len(experiences), 3)

    def test_workflows_raise_error(self) -> None:
        self.model.enable_history = False
        for workflow_cls in _dummy_workflows:
            task = Task(
                workflow=workflow_cls,
                repeat_times=self.taskset_config.repeat_times,
                workflow_args={"max_env_steps": 10, "actual_steps": 5},
            )
            with self.assertRaises(AssertionError):
                task.to_workflow(model=self.model)
