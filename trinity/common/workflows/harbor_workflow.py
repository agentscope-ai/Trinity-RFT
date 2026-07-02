# -*- coding: utf-8 -*-
"""Base workflow for Harbor directory tasks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from trinity.common.workflows.workflow import Metrics, Task, WorkflowWithRecording

if TYPE_CHECKING:
    from harbor.models.task.config import TaskConfig
    from harbor.models.task.paths import TaskPaths
    from harbor.models.trial.config import TrialConfig
    from harbor.models.trial.result import TrialResult
    from harbor.viewer.task_scanner import TaskDefinitionScanner

    from trinity.common.models.model import ModelWrapper


class HarborWorkflow(WorkflowWithRecording):
    """Workflow that runs a Harbor task and writes the verifier reward back.

    ``task.raw_task["task_dir"]`` points at one Harbor task directory. The
    workflow builds one Harbor trial, injects Trinity's rollout OpenAI endpoint
    into the Harbor agent, runs the trial, and then calls ``update_reward`` so
    the rollout model's recorded experiences are labeled with Harbor's verifier
    reward.
    """

    DEFAULT_AGENT_NAME = "openclaw"
    DEFAULT_TRIALS_DIR = "harbor_trials"
    DEFAULT_REWARD_KEY = "reward"

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.harbor_task_dir = self._get_harbor_task_dir(task)
        self.harbor_task_name = self.harbor_task_dir.name
        (
            self.harbor_scanner,
            self.harbor_task_paths,
            self.harbor_task_config,
            self.harbor_instruction,
            self.harbor_task_paths_info,
        ) = self._load_harbor_task(self.harbor_task_dir)

    def _get_harbor_task_dir(self, task: Task) -> Path:
        if task.raw_task is None:
            raise ValueError("HarborWorkflow requires `task.raw_task` to be configured.")

        task_dir = task.raw_task.get("task_dir")
        if task_dir is None:
            raise ValueError("HarborWorkflow requires `task.raw_task['task_dir']`.")

        task_dir_path = Path(task_dir).expanduser().resolve()
        if not task_dir_path.exists():
            raise FileNotFoundError(f"Harbor task directory does not exist: {task_dir_path}")
        if not task_dir_path.is_dir():
            raise ValueError(f"Harbor task path must be a directory: {task_dir_path}")
        return task_dir_path

    def _load_harbor_task(
        self,
        task_dir: Path,
    ) -> tuple["TaskDefinitionScanner", "TaskPaths", "TaskConfig", str | None, dict[str, bool]]:
        try:
            from harbor.models.task.paths import TaskPaths
            from harbor.viewer.task_scanner import TaskDefinitionScanner
        except ImportError as exc:
            raise ImportError(
                "HarborWorkflow requires the `harbor` package to be installed."
            ) from exc

        scanner = TaskDefinitionScanner(task_dir.parent)
        task_name = task_dir.name
        config = scanner.get_task_config(task_name)
        if config is None:
            raise ValueError(
                f"Failed to load Harbor task config from: {task_dir / TaskPaths.CONFIG_FILENAME}"
            )

        paths = TaskPaths(task_dir)
        instruction = scanner.get_instruction(task_name)
        paths_info = scanner.get_task_paths_info(task_name)
        return scanner, paths, config, instruction, paths_info

    async def run_async(self) -> Metrics:
        trial_config = self._build_trial_config()
        trial_result = await self._run_harbor_trial(trial_config)
        reward, reward_info = self._extract_reward(trial_result)
        info = self._build_reward_info(trial_result, reward_info)
        await self.update_reward(reward, info=info)
        return self._build_metrics(trial_result, reward)

    def _build_trial_config(self) -> "TrialConfig":
        try:
            from harbor.models.trial.config import AgentConfig, EnvironmentConfig
            from harbor.models.trial.config import TaskConfig as TrialTaskConfig
            from harbor.models.trial.config import TrialConfig, VerifierConfig
        except ImportError as exc:
            raise ImportError(
                "HarborWorkflow requires the `harbor` package to be installed."
            ) from exc

        workflow_args = self.task.workflow_args or {}
        return TrialConfig(
            task=TrialTaskConfig(path=self.harbor_task_dir),
            trial_name=self._get_optional_str_arg("trial_name"),
            trials_dir=self._get_trials_dir(),
            agent=self._build_agent_config(AgentConfig),
            environment=self._build_model_config(
                EnvironmentConfig, workflow_args.get("environment")
            ),
            verifier=self._build_model_config(VerifierConfig, workflow_args.get("verifier")),
            artifacts=list(workflow_args.get("artifacts", [])),
            extra_instruction_paths=[
                Path(path).expanduser() for path in workflow_args.get("extra_instruction_paths", [])
            ],
            timeout_multiplier=workflow_args.get("timeout_multiplier", 1.0),
            agent_timeout_multiplier=workflow_args.get("agent_timeout_multiplier"),
            verifier_timeout_multiplier=workflow_args.get("verifier_timeout_multiplier"),
            agent_setup_timeout_multiplier=workflow_args.get("agent_setup_timeout_multiplier"),
            environment_build_timeout_multiplier=workflow_args.get(
                "environment_build_timeout_multiplier"
            ),
        )

    def _build_agent_config(self, agent_config_cls):
        workflow_args = self.task.workflow_args or {}
        agent_env = dict(workflow_args.get("agent_env", {}))
        agent_env.update(self._rollout_env())
        agent_name = workflow_args.get("agent_name")
        agent_import_path = workflow_args.get("agent_import_path")
        if agent_name is None and agent_import_path is None:
            agent_name = self.DEFAULT_AGENT_NAME

        return agent_config_cls(
            name=agent_name,
            import_path=agent_import_path,
            model_name=workflow_args.get("harbor_model_name", self.model_name),
            kwargs=dict(workflow_args.get("agent_kwargs", {})),
            env=agent_env,
        )

    def _build_model_config(self, config_cls, raw_config: Any):
        if raw_config is None:
            return config_cls()
        if isinstance(raw_config, config_cls):
            return raw_config
        if isinstance(raw_config, dict):
            return config_cls(**raw_config)
        raise TypeError(
            f"Expected {config_cls.__name__} config as a dict or {config_cls.__name__}, "
            f"got {type(raw_config).__name__}."
        )

    def _rollout_env(self) -> dict[str, str]:
        workflow_args = self.task.workflow_args or {}
        base_url = workflow_args.get("rollout_base_url", self.base_url)
        api_key = workflow_args.get("rollout_api_key", self.api_key)
        model_name = workflow_args.get("harbor_model_name", self.model_name)
        env = {
            "OPENAI_BASE_URL": str(base_url),
            "OPENAI_API_KEY": str(api_key),
            "OPENAI_MODEL": str(model_name),
        }

        provider_env = workflow_args.get("provider_env", {})
        if provider_env:
            env.update({str(key): str(value) for key, value in provider_env.items()})
        return env

    def _get_trials_dir(self) -> Path:
        workflow_args = self.task.workflow_args or {}
        trials_dir = workflow_args.get("trials_dir", self.DEFAULT_TRIALS_DIR)
        return Path(trials_dir).expanduser()

    def _get_optional_str_arg(self, key: str) -> str:
        value = (self.task.workflow_args or {}).get(key, "")
        return "" if value is None else str(value)

    async def _run_harbor_trial(self, trial_config: "TrialConfig") -> "TrialResult":
        try:
            from harbor import Trial
        except ImportError as exc:
            raise ImportError(
                "HarborWorkflow requires the `harbor` package to be installed."
            ) from exc

        trial = await Trial.create(trial_config)
        return await trial.run()

    def _extract_reward(self, trial_result: "TrialResult") -> tuple[float, dict[str, Any]]:
        rewards = self._get_harbor_rewards(trial_result)
        if not rewards:
            return 0.0, {
                "harbor_reward_missing": True,
                "harbor_reward_reason": "verifier_result.rewards is empty",
            }

        reward_key = self._get_reward_key(rewards)
        reward = rewards[reward_key]
        return float(reward), {"harbor_reward_key": reward_key}

    def _get_harbor_rewards(self, trial_result: "TrialResult") -> dict[str, float | int]:
        verifier_result = trial_result.verifier_result
        if verifier_result is None or verifier_result.rewards is None:
            return {}
        return dict(verifier_result.rewards)

    def _get_reward_key(self, rewards: dict[str, float | int]) -> str:
        workflow_args = self.task.workflow_args or {}
        configured_key = workflow_args.get("reward_key")
        if configured_key:
            reward_key = str(configured_key)
            if reward_key not in rewards:
                raise KeyError(
                    f"Harbor reward key {reward_key!r} not found. "
                    f"Available reward keys: {sorted(rewards)}"
                )
            return reward_key

        if self.DEFAULT_REWARD_KEY in rewards:
            return self.DEFAULT_REWARD_KEY
        if len(rewards) == 1:
            return next(iter(rewards))
        raise ValueError(
            "Harbor verifier returned multiple rewards. Set "
            "`workflow_args.reward_key` to choose one. "
            f"Available reward keys: {sorted(rewards)}"
        )

    def _build_reward_info(
        self,
        trial_result: "TrialResult",
        reward_info: dict[str, Any],
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "harbor_task_name": trial_result.task_name,
            "harbor_trial_name": trial_result.trial_name,
            "harbor_trial_uri": trial_result.trial_uri,
            "harbor_rewards": self._get_harbor_rewards(trial_result),
            **reward_info,
        }
        if trial_result.exception_info is not None:
            info.update(
                {
                    "harbor_exception_type": trial_result.exception_info.exception_type,
                    "harbor_exception_message": (trial_result.exception_info.exception_message),
                }
            )
        return info

    def _build_metrics(self, trial_result: "TrialResult", reward: float) -> Metrics:
        rewards = self._get_harbor_rewards(trial_result)
        metrics: Metrics = {
            "harbor/reward": float(reward),
            "harbor/has_exception": (1.0 if trial_result.exception_info is not None else 0.0),
            "harbor/num_rewards": float(len(rewards)),
        }

        (
            input_tokens,
            cache_tokens,
            output_tokens,
            cost_usd,
        ) = trial_result.compute_token_cost_totals()
        if input_tokens is not None:
            metrics["harbor/input_tokens"] = float(input_tokens)
        if cache_tokens is not None:
            metrics["harbor/cache_tokens"] = float(cache_tokens)
        if output_tokens is not None:
            metrics["harbor/output_tokens"] = float(output_tokens)
        if cost_usd is not None:
            metrics["harbor/cost_usd"] = float(cost_usd)
        return metrics
