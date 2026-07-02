# -*- coding: utf-8 -*-
"""Tests for HarborWorkflow without starting Docker or another sandbox."""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import openai
import ray

REPO_ROOT = Path(__file__).resolve().parents[2]
HARBOR_SRC = REPO_ROOT / "thirdparties" / "harbor" / "src"
if str(HARBOR_SRC) not in sys.path:
    sys.path.insert(0, str(HARBOR_SRC))

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.agent.context import AgentContext
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from tests.tools import get_model_path, get_template_config
from trinity.common.constants import MODEL_PATH_ENV_VAR
from trinity.common.models.allocator import Allocator
from trinity.common.workflows.harbor_workflow import HarborWorkflow
from trinity.common.workflows.workflow import Task


class InProcessHarborEnvironment(BaseEnvironment):
    """A minimal Harbor environment for unit tests.

    It satisfies Harbor's trial lifecycle but never starts Docker, a sandbox,
    or a subprocess. The arithmetic agent either calls a real OpenAI-compatible
    vLLM service or uses a deterministic fallback response, so the environment
    only needs to accept lifecycle calls.
    """

    @staticmethod
    def type() -> str:
        return "in-process"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self) -> None:
        pass

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool) -> None:
        pass

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        pass

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        pass

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        Path(target_dir).mkdir(parents=True, exist_ok=True)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        return ExecResult(stdout="", stderr="", return_code=0)


class OpenAIArithmeticAgent(BaseAgent):
    """A Harbor agent that solves the task via the injected OpenAI endpoint."""

    @staticmethod
    def name() -> str:
        return "openai-arithmetic-agent"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self.extra_env.get("TRINITY_FAKE_OPENAI_RESPONSE") is not None:
            answer = self.extra_env["TRINITY_FAKE_OPENAI_RESPONSE"]
        else:
            client = openai.AsyncOpenAI(
                base_url=self.extra_env["OPENAI_BASE_URL"],
                api_key=self.extra_env["OPENAI_API_KEY"],
            )
            response = await client.chat.completions.create(
                model=self.extra_env["OPENAI_MODEL"],
                messages=[
                    {
                        "role": "system",
                        "content": "Solve the problem. Return only the final answer.",
                    },
                    {"role": "user", "content": instruction},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            answer = response.choices[0].message.content or ""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "answer.txt").write_text(answer.strip())


class ExactAnswerHarborVerifier(BaseVerifier):
    """Rule-based verifier for answer-only arithmetic tasks."""

    def __init__(self, *args: Any, expected_answer: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.expected_answer = expected_answer

    async def verify(self) -> VerifierResult:
        answer_path = self.trial_paths.agent_dir / "answer.txt"
        answer = answer_path.read_text().strip() if answer_path.exists() else ""
        integers = re.findall(r"-?\d+", answer)
        final_answer = integers[-1] if integers else ""
        reward = 1.0 if final_answer == self.expected_answer else 0.0
        self.trial_paths.reward_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.trial_paths.reward_json_path.write_text(
            f'{{"reward": {reward}, "exact_match": {reward}}}'
        )
        return VerifierResult(rewards={"reward": reward, "exact_match": reward})


class DummyRecordingModelWrapper:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model_name: str = "dummy-rollout-model",
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = "EMPTY"
        self.reward_updates: list[dict[str, Any]] = []

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    async def update_experience_reward_async(
        self,
        *,
        key: str,
        reward: float,
        info: dict | None = None,
        sample_ids: list[str] | None = None,
    ) -> None:
        self.reward_updates.append(
            {
                "key": key,
                "reward": reward,
                "info": info,
                "sample_ids": sample_ids,
            }
        )


class HarborWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def _create_model_wrapper(self):
        if not os.environ.get(MODEL_PATH_ENV_VAR):
            return DummyRecordingModelWrapper(), False, []

        ray.init(ignore_reinit_error=True, namespace="trinity_unittest")
        config = get_template_config()
        config.mode = "explore"
        config.model.model_path = get_model_path()
        config.explorer.rollout_model.engine_num = 1
        config.explorer.rollout_model.enable_history = True
        config.explorer.rollout_model.enable_openai_api = True
        config.check_and_update()

        allocator = Allocator(config.explorer)
        rollout_models, _ = await allocator.create_all_models()
        return rollout_models[0], True, rollout_models

    async def test_harbor_workflow_runs_arithmetic_task(self) -> None:
        task_dir = REPO_ROOT / "tests" / "template" / "data" / "harbor" / "arithmetic-task"
        model, uses_real_vllm, rollout_models = await self._create_model_wrapper()
        expected_reward = 1.0 if uses_real_vllm else 0.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            task = Task(
                workflow=HarborWorkflow,
                raw_task={"task_dir": str(task_dir)},
                workflow_args={
                    "agent_import_path": (
                        "tests.explorer.harbor_workflow_test:"
                        "OpenAIArithmeticAgent"
                    ),
                    "environment": {
                        "import_path": (
                            "tests.explorer.harbor_workflow_test:"
                            "InProcessHarborEnvironment"
                        ),
                    },
                    "verifier": {
                        "import_path": (
                            "tests.explorer.harbor_workflow_test:"
                            "ExactAnswerHarborVerifier"
                        ),
                        "kwargs": {"expected_answer": "42"},
                    },
                    "provider_env": (
                        {}
                        if uses_real_vllm
                        else {"TRINITY_FAKE_OPENAI_RESPONSE": "not-a-number"}
                    ),
                    "trial_name": "harbor-workflow-vllm-arithmetic",
                    "trials_dir": tmp_dir,
                },
                batch_id="harbor",
                task_id=0,
                run_id=0,
            )

            try:
                metrics = await HarborWorkflow(task=task, model=model).run_async()
            finally:
                for wrapper in rollout_models:
                    await wrapper.shutdown()
                if uses_real_vllm:
                    ray.shutdown()

        self.assertEqual(metrics["harbor/reward"], expected_reward)
        self.assertEqual(metrics["harbor/has_exception"], 0.0)
        self.assertEqual(metrics["harbor/num_rewards"], 2.0)
        if isinstance(model, DummyRecordingModelWrapper):
            self.assertEqual(len(model.reward_updates), 1)
            self.assertEqual(model.reward_updates[0]["key"], "harbor/0/0")
            self.assertEqual(model.reward_updates[0]["reward"], expected_reward)
            self.assertEqual(
                model.reward_updates[0]["info"]["harbor_rewards"],
                {"reward": expected_reward, "exact_match": expected_reward},
            )
