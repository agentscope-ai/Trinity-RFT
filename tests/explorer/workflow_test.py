# -*- coding: utf-8 -*-
"""Test for the workflow module"""
import asyncio
import copy
import os
import shutil
import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import Dict, Optional
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import openai
import ray
from parameterized import parameterized, parameterized_class
from torch import Tensor

from tests.tools import (
    CHAT_TEMPLATE,
    RayUnittestBaseAsync,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.buffer.store import get_record_key
from trinity.common.constants import LOG_DIR_ENV_VAR, LOG_LEVEL_ENV_VAR
from trinity.common.experience import EID, Experience
from trinity.common.models.allocator import Allocator
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Workflow
from trinity.common.workflows.customized_math_workflows import MathBoxedWorkflow
from trinity.common.workflows.eval_workflow import MathEvalWorkflow
from trinity.common.workflows.workflow import (
    MathWorkflow,
    Metrics,
    MultiTurnWorkflow,
    Task,
    WorkflowWithRecording,
)
from trinity.explorer.workflow_runner import Status, WorkflowRunner


def patch_runner_models(*wrappers):
    return mock.patch(
        "trinity.explorer.workflow_runner.Allocator.get_model",
        side_effect=list(wrappers),
    )


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
    truncate_status: str = "not_truncated"
    action_mask: Optional[Tensor] = None


class DummyWorkflow(Workflow):
    can_reset: bool = True

    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]
        self.repeat_times = task.rollout_args.n
        # Check self.auxiliary_models (OpenAI clients derived from ModelWrapper)
        if self.auxiliary_models is not None:
            for m in self.auxiliary_models:
                assert isinstance(m, openai.OpenAI)

    def reset(self, task: Task):
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    def run(self):
        exps = []
        if self.output_format == "json":
            import json

            for i in range(self.repeat_times):
                exp = Experience(tokens=Tensor([0, 1, 2, 3]), prompt_length=1)
                exp.response_text = json.dumps(self.obj)
                exps.append(exp)
            return exps
        elif self.output_format == "yaml":
            import yaml

            for i in range(self.repeat_times):
                exp = Experience(tokens=Tensor([0, 1, 2, 3]), prompt_length=1)
                exp.response_text = yaml.safe_dump(self.obj)
                exps.append(exp)
            return exps
        else:
            raise ValueError("Invalid output format")


class DummyAsyncWorkflow(Workflow):
    can_reset: bool = True
    is_async: bool = True

    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]
        self.repeat_times = task.rollout_args.n
        # Check self.auxiliary_models (AsyncOpenAI clients derived from ModelWrapper)
        if self.auxiliary_models is not None:
            for m in self.auxiliary_models:
                assert isinstance(m, openai.AsyncOpenAI)

    def reset(self, task: Task):
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    async def run_async(self):
        await asyncio.sleep(0.1)
        exps = []
        if self.output_format == "json":
            import json

            for i in range(self.repeat_times):
                exp = Experience(tokens=Tensor([0, 1, 2, 3]), prompt_length=1)
                exp.response_text = json.dumps(self.obj)
                exps.append(exp)
            return exps
        elif self.output_format == "yaml":
            import yaml

            for i in range(self.repeat_times):
                exp = Experience(tokens=Tensor([0, 1, 2, 3]), prompt_length=1)
                exp.response_text = yaml.safe_dump(self.obj)
                exps.append(exp)
            return exps
        else:
            raise ValueError("Invalid output format")


class DummyMultiTurnWorkflow(MultiTurnWorkflow):
    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.contents = task.raw_task["contents"]  # type: ignore

    def run(self):
        memory = [{"role": "system", "content": "You are a helpful assistant."}]
        experience_list = []
        for content in self.contents:
            memory.append({"role": "user", "content": content})
            memory.append({"role": "assistant", "content": content.upper()})
            experience = self.process_messages_to_experience(memory, 0, {})
            experience_list.append(experience)
        return experience_list


class DummyAsyncMultiTurnWorkflow(MultiTurnWorkflow):
    is_async: bool = True

    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.contents = task.raw_task["contents"]  # type: ignore

    async def run_async(self):
        memory = [{"role": "system", "content": "You are a helpful assistant."}]
        experience_list = []
        for content in self.contents:
            await asyncio.sleep(0.1)
            memory.append({"role": "user", "content": content})
            memory.append({"role": "assistant", "content": content.upper()})
            experience = await self.process_messages_to_experience_async(memory, 0, {})
            experience_list.append(experience)
        return experience_list


class WorkflowTest(unittest.TestCase):
    def test_math_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse(r"\boxed{2}"),
            MockResponse(r"\boxted{3}"),
            MockResponse(r"2"),
            MockResponse("<think>\nThinking\n</think>\n<answer>\n3\n</answer>"),
            MockResponse("<think>\nThinking\n</think>\n<answer>\n\\boxed{2}\n</answer>"),
            MockResponse("<think>Missing closing</think><answer>\\boxed{2}"),
            MockResponse("<answer>\nOnly answer\n</answer>"),
            MockResponse("<think>\nOnly thinking\n</think>"),
            MockResponse("<think>Thinking</think><answer>Answer is not end</answer><answer>1"),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathWorkflow,
            repeat_times=taskset_config.repeat_times,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "1+1=",
                taskset_config.format.response_key: "2",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(len(experiences), 9)
        self.assertEqual(experiences[0].reward, 0.9)
        self.assertEqual(experiences[1].reward, -0.1)
        self.assertEqual(experiences[2].reward, 0.9)
        self.assertEqual(experiences[3].reward, 0.1)
        self.assertEqual(experiences[4].reward, 1.1)
        self.assertEqual(experiences[5].reward, 0.9)
        self.assertEqual(experiences[6].reward, -0.1)
        self.assertEqual(experiences[7].reward, -0.1)
        self.assertEqual(experiences[8].reward, -0.1)

    def test_math_fraction_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse(r"\boxed{\frac{40}{400}}"),
            MockResponse(r"\boxed{\frac{1}{10}}"),
            MockResponse(r"\boxed{0.1}"),
            MockResponse(r"\boxed{0.1000}"),
            MockResponse(r"\boxed{\frac{1} {10}}"),
            MockResponse(r"The answer is \boxed{\frac{40}{400}}"),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathWorkflow,
            repeat_times=taskset_config.repeat_times,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: r"\frac{40}{400}",
                taskset_config.format.response_key: r"\frac{40}{400}",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(len(experiences), 6)
        self.assertEqual(experiences[0].reward, 0.9)
        self.assertEqual(experiences[1].reward, 0.9)
        self.assertEqual(experiences[2].reward, 0.9)
        self.assertEqual(experiences[3].reward, 0.9)
        self.assertEqual(experiences[4].reward, 0.9)
        self.assertEqual(experiences[5].reward, 0.9)

    def test_math_complex_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse(
                r"$\boxed{\dfrac{108 + 31\sqrt{5}}{216}} \quad \text{and} \quad \boxed{\dfrac{108 - 31\sqrt{5}}{216}}$"
            ),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathWorkflow,
            repeat_times=taskset_config.repeat_times,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"$x_{1}=\frac{1}{2}+\frac{31\sqrt{5}}{216},\quadx_{2}=\frac{1}{2}-\frac{31\sqrt{5}}{216}$",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0].reward, 0.9)

    def test_math_boxed_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse("<think> balabalabala 99 </think>\n \\boxed{36}"),
            MockResponse("answer is \\boxed{36 }"),
            MockResponse("Kim's total points are 6 + 30 =\\boxed{36}"),
            MockResponse("<think> balalaba </think> \\boxed{35.00}"),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathBoxedWorkflow,
            repeat_times=taskset_config.repeat_times,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            workflow_args={
                "with_think": False,
                "format_score_coef": 0.2,
            },
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"36",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(experiences[0].reward, 1.0)
        self.assertEqual(experiences[1].reward, 1.0)
        self.assertEqual(experiences[2].reward, 1.0)
        self.assertEqual(experiences[3].reward, 0.0)
        task_new = Task(
            workflow=MathBoxedWorkflow,
            repeat_times=taskset_config.repeat_times,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            workflow_args={
                "with_think": True,
                "format_score_coef": 0.2,
            },
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"36",
            },
        )
        workflow.reset(task_new)
        workflow_new = task_new.to_workflow(model=model)
        experiences = workflow_new.run()
        self.assertEqual(experiences[0].reward, 1.0)
        self.assertEqual(experiences[1].reward, 0.8)
        self.assertEqual(experiences[2].reward, 0.8)
        self.assertEqual(experiences[3].reward, 0.0)

    def test_gsm8k_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse("<think> balabalabala 99 </think>\n<answer> 36 </answer>"),
            MockResponse("<answer> 36.0 </answer>"),
            MockResponse("<answer>Kim's total points are 6 + 30 = 36 </answer>"),
            MockResponse("<think> balalaba </think><answer> 35.00 </answer>"),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathWorkflow,
            repeat_times=taskset_config.repeat_times,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"36",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(experiences[0].reward, 1.1)
        self.assertEqual(experiences[1].reward, 0.9)
        self.assertEqual(experiences[2].reward, 0.9)
        self.assertEqual(experiences[3].reward, 0.1)
        task_new = Task(
            workflow=MathWorkflow,
            repeat_times=taskset_config.repeat_times,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"35",
            },
        )
        workflow.reset(task_new)
        workflow_new = task_new.to_workflow(model=model)
        experiences = workflow_new.run()
        self.assertEqual(experiences[0].reward, 0.1)
        self.assertEqual(experiences[1].reward, -0.1)
        self.assertEqual(experiences[2].reward, -0.1)
        self.assertEqual(experiences[3].reward, 1.1)

    def test_math_eval_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse("My step-by-step reasoning leads to the answer \boxed{36}"),
            MockResponse("Here is the answer of \boxed{36.0}"),
            MockResponse("I made a mistake, the answer is \boxed{42}"),
            MockResponse("The answer is 36, but I forgot the box."),
        ]

        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathEvalWorkflow,
            repeat_times=taskset_config.repeat_times,
            is_eval=True,
            format_args=taskset_config.format,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: "36",
            },
        )

        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(len(experiences), 4)
        expected_accuracies = [1.0, 1.0, 0.0, 0.0]
        for i, (exp, expected_acc) in enumerate(zip(experiences, expected_accuracies)):
            with self.subTest(f"Response {i}"):
                self.assertEqual(exp.reward, 0.0)
                assert exp.metrics is not None, f"Metrics for response {i} should not be None"
                self.assertEqual(exp.metrics["accuracy"], expected_acc)

    @parameterized.expand([(DummyWorkflow,), (DummyAsyncWorkflow,)])
    def test_workflow_resettable(self, workflow_cls) -> None:
        model = MagicMock()
        json_task = Task(
            workflow=workflow_cls,
            repeat_times=1,
            raw_task={"a": 1},
            workflow_args={"output_format": "json"},
        )
        yaml_task = Task(
            workflow=workflow_cls,
            repeat_times=1,
            raw_task={"a": 1},
            workflow_args={"output_format": "yaml"},
        )
        workflow = json_task.to_workflow(model)
        if workflow.is_async:
            answer = asyncio.run(workflow.run_async())
        else:
            answer = workflow.run()
        self.assertEqual(answer[0].response_text, '{"a": 1}')
        workflow.reset(yaml_task)
        if workflow.is_async:
            answer = asyncio.run(workflow.run_async())
        else:
            answer = workflow.run()
        self.assertEqual(answer[0].response_text, "a: 1\n")

    @parameterized.expand([(DummyWorkflow,), (DummyAsyncWorkflow,)])
    def test_workflow_repeatable(self, workflow_cls) -> None:
        model = MagicMock()
        task = Task(
            workflow=workflow_cls,
            repeat_times=3,
            raw_task={"a": 1},
            workflow_args={"output_format": "json"},
        )
        workflow = task.to_workflow(model)
        workflow.set_repeat_times(2, run_id_base=0)
        self.assertEqual(workflow.repeat_times, 2)
        if workflow.is_async:
            answer = asyncio.run(workflow.run_async())
        else:
            answer = workflow.run()
        self.assertEqual(len(answer), 2)


@parameterized_class(
    ("workflow_cls",),
    [
        (DummyMultiTurnWorkflow,),
        (DummyAsyncMultiTurnWorkflow,),
    ],
)
class MultiTurnWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # configure the model
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_num = 1  # self.engine_num
        self.config.explorer.rollout_model.tensor_parallel_size = 1  # self.tensor_parallel_size
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.algorithm.repeat_times = 2  # self.repeat_times
        self.config.explorer.rollout_model.enable_history = True  # self.enable_history
        self.config.check_and_update()
        allocator = Allocator(self.config.explorer)
        rollout_model, _ = await allocator.create_all_models()
        self.model_wrapper = rollout_model[0]
        await self.model_wrapper.prepare()

    async def test_multi_turn_workflow(self):
        task = Task(
            workflow=self.workflow_cls,
            repeat_times=3,
            raw_task={"contents": ["hello world!", "how are you?"]},
            workflow_args={"output_format": "json"},
        )
        workflow = task.to_workflow(self.model_wrapper)
        workflow.set_repeat_times(2, run_id_base=0)
        if workflow.is_async:
            answer = await workflow.run_async()
        else:
            answer = workflow.run()
        self.assertEqual(len(answer), 2)

    def tearDown(self):
        ray.shutdown(_exiting_interpreter=True)


class TestAgentScopeWorkflowAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_v1(self):
        try:
            from agentscope.model import ChatModelBase
            from agentscope.tuner import JudgeOutput, WorkflowOutput
        except ImportError:
            self.skipTest("agentscope >= 1.0.12 is not installed")

        async def as_workflow_func(task, model) -> WorkflowOutput:
            self.assertIsInstance(task, dict)
            self.assertIsInstance(model, ChatModelBase)
            return WorkflowOutput(
                reward=task["reward"],
                response=task["reward"],
                metrics={"workflow_metric_1": 0.0},
            )

        async def as_judge_func(task, response) -> JudgeOutput:
            self.assertIsInstance(task, dict)
            self.assertIsInstance(response, float)
            return JudgeOutput(
                reward=response,
                metrics={"judge_metric_1": 1.0},
            )

        model = MagicMock()
        openai_client = MagicMock()
        openai_client.model_path = "Qwen/Qwen3-8B"
        model.get_openai_async_client.return_value = openai_client
        model.extract_experience_from_history.return_value = [
            Experience(tokens=Tensor([0, 1, 2]), prompt_length=1, logprobs=Tensor([0.1, 0.2])),
        ]

        as_adapter_cls = WORKFLOWS.get("agentscope_workflow_adapter_v1")
        as_adapter = as_adapter_cls(
            task=Task(
                raw_task={"reward": 0.2},
                workflow_args={
                    "workflow_func": as_workflow_func,
                    "judge_func": as_judge_func,
                },
            ),
            model=model,
        )
        result = await as_adapter.run_async()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].reward, 0.2)
        self.assertEqual(result[0].prompt_length, 1)
        metrics = result[-1].metrics
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics["workflow_metric_1"], 0.0)
        self.assertEqual(metrics["judge_metric_1"], 1.0)


class DummyModelWrapper:
    def __init__(self, model, **kwargs):
        self._api_key = "EMPTY"

    async def prepare(self):
        return

    def set_api_key(self, api_key: str) -> None:
        """Mirror ModelWrapper.set_api_key for the refactored WorkflowBase."""
        self._api_key = api_key

    def clone_with_isolated_state(self) -> "DummyModelWrapper":
        """Mirror ModelWrapper.clone_with_isolated_state for the runner's
        isolated workflow instances used in async/multi-threading modes."""
        return copy.copy(self)

    async def overwrite_history_experiences_async(self, experiences, key: str) -> None:
        """Mirror ModelWrapper.overwrite_history_experiences_async; a no-op for
        tests since DummyWorkflow does not record history."""
        return

    def get_openai_client(self):
        return openai.OpenAI(api_key="EMPTY")

    def get_openai_async_client(self):
        return openai.AsyncOpenAI(api_key="EMPTY")

    @property
    async def model_version_async(self):
        return 0


class APIWorkflow(Workflow):
    is_async: bool = True

    def __init__(self, model: ModelWrapper, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.client = model.get_openai_async_client()
        self.raise_except = task.raw_task.get("raise_except", False)

    async def run_async(self):
        _ = await self.client.chat.completions.create(
            model=self.client.model_path,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        if self.raise_except:
            raise RuntimeError("Intentional Exception for testing.")
        exps = self.model.extract_experience_from_history()
        exps[0].reward = 0.5
        return exps


class PartialFailureWorkflow(Workflow):
    can_reset: bool = True

    _call_lock = threading.Lock()
    _call_count = 0

    def __init__(self, model: ModelWrapper, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.fail_call_ids = set(task.raw_task.get("fail_call_ids", []))

    def reset(self, task: Task):
        self.fail_call_ids = set(task.raw_task.get("fail_call_ids", []))

    @classmethod
    def reset_call_count(cls):
        with cls._call_lock:
            cls._call_count = 0

    @classmethod
    def next_call_id(cls) -> int:
        with cls._call_lock:
            call_id = cls._call_count
            cls._call_count += 1
            return call_id

    def run(self):
        call_id = self.next_call_id()
        if call_id in self.fail_call_ids:
            raise RuntimeError(f"Intentional failure for run call {call_id}")

        exp = Experience(
            tokens=Tensor([0, 1, 2]),
            prompt_length=1,
            metrics={"run_metrics": float(call_id)},
        )
        exp.response_text = str(call_id)
        return [exp]


class TestWorkflowRunner(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_runner(self):
        config = get_template_config()
        config.explorer.auxiliary_models = [
            config.explorer.rollout_model,
            config.explorer.rollout_model,
        ]

        with patch_runner_models(
            DummyModelWrapper(MagicMock()),
            DummyModelWrapper(MagicMock()),
            DummyModelWrapper(MagicMock()),
        ):
            runner = WorkflowRunner(
                config,
                rollout_model_id=0,
                auxiliary_model_ids=[0, 1],
                runner_id=0,
            )
            await runner.prepare()
            task = Task(
                workflow=DummyWorkflow,
                repeat_times=3,
                raw_task={"a": 1},
                workflow_args={"output_format": "json"},
            )

            status = await runner.run_task(task, repeat_times=3, run_id_base=0)

            self.assertTrue(status.ok)
            self.assertEqual(status.completed_runs, 3)
            self.assertEqual(status.total_runs, 3)
            self.assertEqual(len(status.metrics), 3)

            task = Task(
                workflow=DummyAsyncWorkflow,
                repeat_times=2,
                raw_task={"a": 1},
                workflow_args={"output_format": "yaml"},
            )

            status = await runner.run_task(task, repeat_times=2, run_id_base=0)
            self.assertTrue(status.ok)
            self.assertEqual(status.completed_runs, 2)
            self.assertEqual(status.total_runs, 2)
            self.assertEqual(len(status.metrics), 2)

    @parameterized.expand(
        [
            ("sequential", 2),
            ("asynchronous", 2),
            ("multi-threading", 2),
        ]
    )
    async def test_workflow_runner_partial_success_non_repeatable(
        self, concurrent_mode: str, expected_success_runs: int
    ):
        config = get_template_config()
        config.explorer.concurrent_mode = concurrent_mode

        with patch_runner_models(DummyModelWrapper(MagicMock())):
            runner = WorkflowRunner(
                config,
                rollout_model_id=0,
                runner_id=0,
            )
            await runner.prepare()
            PartialFailureWorkflow.reset_call_count()
            task = Task(
                workflow=PartialFailureWorkflow,
                repeat_times=3,
                raw_task={"fail_call_ids": [1]},
                batch_id="test",
                task_id=0,
            )

            status = await runner.run_task(task, repeat_times=3, run_id_base=0)

            self.assertFalse(status.ok)
            self.assertEqual(status.completed_runs, expected_success_runs)
            self.assertEqual(status.total_runs, 3)

            # One internal run fails with call_id=1, so runner-level metrics should
            # retain only the successful runs from this single subtask: call_id=0 and 2.
            self.assertEqual(len(status.metrics), expected_success_runs)
            self.assertEqual(
                sorted(metric["run_metrics"] for metric in status.metrics),
                [0.0, 2.0],
            )
            assert status.message is not None
            self.assertIn(
                f"{expected_success_runs}/3 runs completed successfully",
                status.message,
            )

    @parameterized.expand(
        [
            ("sequential",),
            ("asynchronous",),
            ("multi-threading",),
        ]
    )
    async def test_workflow_runner_fail_fast_without_partial_collection(self, concurrent_mode: str):
        config = get_template_config()
        config.explorer.concurrent_mode = concurrent_mode

        with patch_runner_models(DummyModelWrapper(MagicMock())):
            runner = WorkflowRunner(
                config,
                rollout_model_id=0,
                runner_id=0,
            )
            task = Task(
                workflow=PartialFailureWorkflow,
                repeat_times=3,
                raw_task={"fail_call_ids": []},
                batch_id="test",
            )
            await runner.prepare()

            async def mock_execute_single_run(
                workflow: Workflow,
            ):
                run_index = int(workflow.task.run_id)
                if run_index == 0:
                    await asyncio.sleep(0.01)
                    return Status(
                        completed_runs=1,
                        total_runs=1,
                        metrics=[{"run_metrics": 0.0}],
                        successful_ids=[workflow.task.api_key],
                    )
                if run_index == 1:
                    await asyncio.sleep(0.02)
                    return Status(
                        completed_runs=0,
                        total_runs=1,
                        metrics=[],
                        message="planned failure",
                    )
                await asyncio.sleep(0.5)
                return Status(
                    completed_runs=1,
                    total_runs=1,
                    metrics=[{"run_metrics": 2.0}],
                    successful_ids=[workflow.task.api_key],
                )

            runner._execute_single_run = AsyncMock(side_effect=mock_execute_single_run)

            status = await runner.run_task(
                task,
                repeat_times=3,
                run_id_base=0,
                collect_partial_runs=False,
            )

            self.assertFalse(status.ok)
            self.assertEqual(status.completed_runs, 1)
            self.assertEqual(status.total_runs, 3)
            assert status.message is not None
            self.assertIn(
                "1/3 runs completed successfully",
                status.message,
            )

    async def test_workflow_with_openai(self):
        config = get_template_config()
        config.mode = "explore"
        config.model.model_path = get_model_path()
        config.explorer.rollout_model.engine_num = 1
        config.explorer.rollout_model.enable_openai_api = True
        config.explorer.rollout_model.enable_history = True
        config.check_and_update()
        allocator = Allocator(config.explorer)
        rollout_model, auxiliary_models = await allocator.create_all_models()
        runner = WorkflowRunner(
            config,
            rollout_model_id=0,
            runner_id=0,
        )
        await runner.prepare()
        tasks = [
            Task(
                workflow=APIWorkflow,
                raw_task={"raise_except": True},
                repeat_times=2,
                batch_id="openai_test",
                task_id=0,
            ),
            Task(
                workflow=APIWorkflow,
                raw_task={},
                repeat_times=2,
                batch_id="openai_test",
                task_id=1,
            ),
        ]

        status = await runner.run_task(tasks[0], repeat_times=2, run_id_base=0)
        self.assertEqual(status.ok, False)
        # The run raised after the chat call, so the partial experience recorded
        # under the last run's key persists (execute/overwrite is never reached).
        exps = runner.model_wrapper.extract_experience_from_history(clear_history=False)
        self.assertEqual(len(exps), 1)
        status = await runner.run_task(tasks[1], repeat_times=2, run_id_base=0)
        self.assertEqual(status.ok, True)
        self.assertEqual(status.completed_runs, 2)
        # A successful run extracts the recorded history (clearing it) and then
        # `Workflow.execute` overwrites the final experiences back under the key,
        # so the last run's key still holds one experience (drained later by the
        # coordinator, not by run_task).
        exps = runner.model_wrapper.extract_experience_from_history(clear_history=False)
        self.assertEqual(len(exps), 1)
        self.assertEqual(len(rollout_model), 1)
        await rollout_model[0].shutdown()

    def tearDown(self):
        ray.shutdown(_exiting_interpreter=True)


class ConcurrentTestWorkflow(Workflow):
    is_async: bool = True

    def __init__(self, model: ModelWrapper, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.client = self.model.get_openai_async_client()

    async def run_async(self):
        assert self.task.raw_task is not None
        text = self.task.raw_task["text"]
        # Both calls opt into recording under the run's record key
        # (enable_recording=True is required for chat_async to stamp the key;
        # otherwise the engine recorder skips the turn entirely). Distinct prompts
        # guarantee the two recorded experiences never form a token-prefix chain,
        # so the prefix merger leaves them as two separate experiences.
        _ = await self.model.chat_async([{"role": "user", "content": text}], enable_recording=True)
        await asyncio.sleep(1.0)
        _ = await self.client.chat.completions.create(
            model=self.client.model_path,
            messages=[{"role": "user", "content": "What is the result of one plus one?"}],
        )
        history_exps = self.model.extract_experience_from_history()
        assert len(history_exps) == 2, "Expected 2 experiences from history, got {}".format(
            len(history_exps)
        )
        for exp in history_exps:
            assert exp.prompt_length > 0, "Expected a positive prompt length, got {}".format(
                exp.prompt_length
            )
        self.logger.debug("[DEBUG MESSAGE]")
        self.logger.info("[INFO MESSAGE]")
        self.logger.warning("[WARNING MESSAGE]")
        return history_exps


class ConcurrentRecordingWorkflow(WorkflowWithRecording):
    def __init__(self, model: ModelWrapper, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    def reset(self, task: Task):
        self.task = task
        self.model.set_api_key(task.api_key)
        self.client.api_key = task.api_key

    async def _chat(self, messages):
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=16,
        )

    async def run_async(self) -> Metrics:
        prefix_messages = [{"role": "user", "content": "Reply with the word alpha only."}]
        first = await self._chat(prefix_messages)
        first_text = first.choices[0].message.content or ""

        merged_messages = [
            *prefix_messages,
            {"role": "assistant", "content": first_text},
            {"role": "user", "content": "Now reply with the word beta only."},
        ]
        await self._chat(merged_messages)

        await self._chat([{"role": "user", "content": "This is an unrelated single-turn branch."}])
        await self.update_reward(0.75, info={"source": "workflow_with_recording"})
        return {"recording_workflow/updated_reward": 1.0}


class TestConcurrentWorkflowRunner(RayUnittestBaseAsync):
    def setUp(self) -> None:
        config = get_template_config()
        config.mode = "explore"
        config.model.model_path = get_model_path()
        config.explorer.rollout_model.engine_num = 1
        config.explorer.rollout_model.enable_history = True
        config.explorer.rollout_model.enable_openai_api = True
        config.check_and_update()
        self.config = config
        os.makedirs(self.config.log.save_dir, exist_ok=True)

    async def test_concurrent_workflow_runner(self):
        allocator = Allocator(self.config.explorer)
        rollout_model, _ = await allocator.create_all_models()

        self.config.explorer.concurrent_mode = "sequential"
        sequential_runner = (
            ray.remote(WorkflowRunner)
            .options(
                runtime_env={
                    "env_vars": {
                        LOG_DIR_ENV_VAR: os.path.join(self.config.log.save_dir),
                        LOG_LEVEL_ENV_VAR: "DEBUG",
                    },
                }
            )
            .remote(
                self.config,
                rollout_model_id=0,
                runner_id=0,
            )
        )
        self.config.explorer.concurrent_mode = "asynchronous"
        async_runner = (
            ray.remote(WorkflowRunner)
            .options(
                runtime_env={
                    "env_vars": {
                        LOG_DIR_ENV_VAR: os.path.join(self.config.log.save_dir),
                        LOG_LEVEL_ENV_VAR: "INFO",
                    },
                }
            )
            .remote(
                self.config,
                rollout_model_id=0,
                runner_id=1,
            )
        )
        thread_runner = (
            ray.remote(WorkflowRunner)
            .options(
                runtime_env={
                    "env_vars": {
                        LOG_DIR_ENV_VAR: os.path.join(self.config.log.save_dir),
                        LOG_LEVEL_ENV_VAR: "WARNING",
                    },
                }
            )
            .remote(
                self.config,
                rollout_model_id=0,
                runner_id=2,
            )
        )
        await asyncio.gather(
            sequential_runner.prepare.remote(),
            async_runner.prepare.remote(),
            thread_runner.prepare.remote(),
        )
        task = Task(
            workflow=ConcurrentTestWorkflow,
            repeat_times=4,
            raw_task={"text": "Hello, world!"},
            batch_id="concurrent",
            task_id=0,
        )

        # Each run_task call uses a distinct batch_id so the record keys
        # (<batch_id>/<task_id>/<run_id>) never collide across calls on the shared
        # rollout-model store. `Workflow.execute` overwrites the final experiences
        # back under each key, so reusing a key would let a later call observe the
        # previous call's leftovers and break the per-run `assert len==2`.
        # warmup
        task.batch_id = "concurrent_async_warmup"
        async_status = await async_runner.run_task.remote(task, repeat_times=2, run_id_base=0)

        st = time.time()
        task.batch_id = "concurrent_async"
        async_status = await async_runner.run_task.remote(task, repeat_times=4, run_id_base=0)
        async_runtime = time.time() - st

        # warmup
        task.batch_id = "concurrent_thread_warmup"
        thread_status = await thread_runner.run_task.remote(task, repeat_times=1, run_id_base=0)

        st = time.time()
        task.batch_id = "concurrent_thread"
        thread_status = await thread_runner.run_task.remote(task, repeat_times=4, run_id_base=0)
        thread_runtime = time.time() - st
        st = time.time()
        task.batch_id = "concurrent_sequential"
        sequential_status = await sequential_runner.run_task.remote(
            task, repeat_times=4, run_id_base=0
        )
        sequential_runtime = time.time() - st

        self.assertTrue(async_status.ok)
        self.assertTrue(thread_status.ok)
        self.assertTrue(sequential_status.ok)
        self.assertEqual(async_status.completed_runs, 4)
        self.assertEqual(thread_status.completed_runs, 4)
        self.assertEqual(sequential_status.completed_runs, 4)

        self.assertLessEqual(async_runtime * 2, sequential_runtime)
        self.assertLessEqual(thread_runtime * 2, sequential_runtime)

        recording_task = Task(
            workflow=ConcurrentRecordingWorkflow,
            repeat_times=1,
            raw_task={},
            batch_id="concurrent_recording",
            task_id=0,
        )
        recording_status = await sequential_runner.run_task.remote(
            recording_task, repeat_times=1, run_id_base=0
        )
        self.assertTrue(recording_status.ok)
        self.assertEqual(recording_status.completed_runs, 1)
        self.assertEqual(recording_status.successful_ids, ["concurrent_recording/0/0"])
        self.assertEqual(recording_status.metrics[0]["recording_workflow/updated_reward"], 1.0)

        recording_exps = rollout_model[0].extract_experience_from_history(
            key="concurrent_recording/0/0"
        )
        self.assertEqual(len(recording_exps), 2)
        for exp in recording_exps:
            self.assertEqual(get_record_key(exp), "concurrent_recording/0/0")
            self.assertEqual(exp.reward, 0.75)
            self.assertEqual(exp.info["source"], "workflow_with_recording")

        merged_exps = [exp for exp in recording_exps if "merged_turn_count" in (exp.info or {})]
        branch_exps = [exp for exp in recording_exps if "merged_turn_count" not in (exp.info or {})]
        self.assertEqual(len(merged_exps), 1)
        self.assertEqual(len(branch_exps), 1)
        self.assertEqual(merged_exps[0].info["merged_turn_count"], 2)
        self.assertEqual(len(merged_exps[0].info["merged_eid_suffixes"]), 2)

        # check log files
        sequential_log_path = os.path.join(self.config.log.save_dir, "explorer_runner_0.log")
        async_log_path = os.path.join(self.config.log.save_dir, "explorer_runner_1.log")
        thread_log_path = os.path.join(self.config.log.save_dir, "explorer_runner_2.log")
        with open(sequential_log_path, "r") as f:
            sequential_logs = f.read()
            assert "[DEBUG MESSAGE]" in sequential_logs
            assert "[INFO MESSAGE]" in sequential_logs
            assert "[WARNING MESSAGE]" in sequential_logs
            # count the occurrences of each log level
            debug_count = sequential_logs.count("[DEBUG MESSAGE]")
            info_count = sequential_logs.count("[INFO MESSAGE]")
            warning_count = sequential_logs.count("[WARNING MESSAGE]")
            assert debug_count == 4
            assert info_count == 4
            assert warning_count == 4
        with open(async_log_path, "r") as f:
            async_logs = f.read()
            assert "[DEBUG MESSAGE]" not in async_logs
            assert "[INFO MESSAGE]" in async_logs
            assert "[WARNING MESSAGE]" in async_logs
            info_count = async_logs.count("[INFO MESSAGE]")
            warning_count = async_logs.count("[WARNING MESSAGE]")
            assert info_count == 6
            assert warning_count == 6
        with open(thread_log_path, "r") as f:
            thread_logs = f.read()
            assert "[DEBUG MESSAGE]" not in thread_logs
            assert "[INFO MESSAGE]" not in thread_logs
            assert "[WARNING MESSAGE]" in thread_logs
            warning_count = thread_logs.count("[WARNING MESSAGE]")
            assert warning_count == 5
        await rollout_model[0].shutdown()

    def tearDown(self):
        shutil.rmtree(self.config.log.save_dir, ignore_errors=True)
