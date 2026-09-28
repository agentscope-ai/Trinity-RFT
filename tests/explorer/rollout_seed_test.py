# -*- coding: utf-8 -*-
"""Regression tests for sampling seeds in repeated rollouts."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from trinity.common.config import Config, InferenceModelConfig
from trinity.common.experience import Experience
from trinity.common.workflows.envs.alfworld.alfworld_workflow import (
    StepWiseAlfworldWorkflow,
)
from trinity.common.workflows.workflow import Task, Workflow
from trinity.explorer.workflow_runner import WorkflowRunner


class RepeatIndexWorkflow(Workflow):
    """Expose the run context through an experience without model inference."""

    can_reset = True

    def reset(self, task):
        self.task = task

    def run(self):
        return [
            Experience(
                tokens=torch.tensor([1, 2]),
                prompt_length=1,
                reward=float(self.repeat_index),
            )
        ]


class RolloutSeedTest(unittest.IsolatedAsyncioTestCase):
    async def test_repeat_indices_include_shard_offset_in_every_runner_mode(self):
        for mode in ("sequential", "asynchronous", "multi-threading"):
            with self.subTest(mode=mode):
                config = Config()
                config.explorer.concurrent_mode = mode
                config.explorer.rollout_model.enable_history = False
                model = MagicMock()
                model.clean_workflow_state = AsyncMock()
                with patch(
                    "trinity.explorer.workflow_runner.Allocator.get_model", return_value=model
                ):
                    runner = WorkflowRunner(config, rollout_model_id=0, runner_id=0)
                task = Task(workflow=RepeatIndexWorkflow)

                # Simulate a group split into two runner assignments.
                first = await runner._run_task(task, repeat_times=2, run_id_base=0)
                second = await runner._run_task(task, repeat_times=2, run_id_base=2)
                self.assertTrue(first.status.ok)
                self.assertTrue(second.status.ok)
                experiences = first.experiences + second.experiences
                self.assertEqual([exp.reward for exp in experiences], [0, 1, 2, 3])
                self.assertEqual([exp.eid.run for exp in experiences], [0, 1, 2, 3])

    def test_alfworld_repeats_have_distinct_reproducible_seeds(self):
        model = MagicMock()
        model.chat.return_value = [SimpleNamespace(response_text="<action>look</action>")]
        with patch.object(StepWiseAlfworldWorkflow, "_setup_environment"):
            workflow = StepWiseAlfworldWorkflow(model=model, task=Task(raw_task={}))
        workflow.env = MagicMock()
        workflow.env.step.return_value = ("room", 0, False, {})

        seeds = []
        for repeat_index in (0, 1, 7, 0):
            workflow.repeat_index = repeat_index
            workflow.observation = "room"
            workflow.memory = []
            workflow.step(0)
            seeds.append(model.chat.call_args.kwargs["seed"])
        self.assertEqual(seeds, [1, 2, 8, 1])

    @unittest.skipUnless(importlib.util.find_spec("tinker"), "Tinker SDK is not installed")
    async def test_tinker_only_uses_explicit_request_seeds(self):
        from trinity.common.models.tinker_model import TinkerModel

        # Avoid a Ray actor and network service; exercise request construction.
        model = TinkerModel.__new__(TinkerModel)
        model.config = InferenceModelConfig(seed=42, max_response_tokens=16)
        model.model = SimpleNamespace(sample_async=AsyncMock())
        for kwargs, expected_seed in (({}, None), ({"seed": 0}, 0), ({"seed": 7}, 7)):
            with self.subTest(kwargs=kwargs):
                await model._generate_internal({"prompt_token_ids": [1, 2]}, **kwargs)
                params = model.model.sample_async.call_args.kwargs["sampling_params"]
                self.assertEqual(params["seed"], expected_seed)
                self.assertEqual(params["max_tokens"], 16)


if __name__ == "__main__":
    unittest.main()
