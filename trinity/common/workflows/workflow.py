# -*- coding: utf-8 -*-
"""Base Workflow Class"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from trinity.common.config import FormatConfig, GenerationConfig
from trinity.common.experience import Experience
from trinity.common.rewards.reward_fn import RewardFn
from trinity.utils.log import get_logger

if TYPE_CHECKING:
    import openai

    from trinity.common.models.model import ModelWrapper


@dataclass(frozen=True)
class Status:
    """Status of workflow, task, and batch execution."""

    completed_runs: int
    total_runs: int
    metrics: List[Dict[str, float]]
    successful_ids: List[str] = field(default_factory=list)
    message: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.completed_runs == self.total_runs


Metrics = Dict[str, float]


@dataclass
class Task(dict):
    """A Task class that defines a task and its associated reward function / workflow."""

    workflow: Type[WorkflowBase] = None
    repeat_times: Optional[int] = None
    format_args: FormatConfig = field(default_factory=FormatConfig)
    rollout_args: GenerationConfig = field(default_factory=GenerationConfig)
    workflow_args: dict = field(default_factory=dict)
    reward_fn_args: dict = field(default_factory=dict)
    is_eval: bool = False
    reward_fn: Optional[Type[RewardFn]] = None
    raw_task: Optional[dict] = None  # The raw data sample

    # automatically assigned ids
    batch_id: Union[int, str] = ""
    task_id: Union[int, str] = ""
    run_id: int = 0

    index: dict = field(default_factory=dict)

    def to_workflow(
        self,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ) -> "WorkflowBase":
        """Convert the task to a workflow.

        Args:
            model (ModelWrapper): The rollout model for the workflow.
            auxiliary_models (List[ModelWrapper]): The auxiliary model wrappers.
                Workflows can access both the ModelWrapper and OpenAI client via
                self.auxiliary_model_wrappers and self.auxiliary_models respectively.

        Returns:
            Workflow: The generated workflow object.
        """
        return self.workflow(
            model=model,
            task=self,
            auxiliary_models=auxiliary_models,
        )

    # Deprecated property, will be removed in the future
    @property
    def task_desc(self) -> Union[str, None]:
        prompt_key = self.format_args.prompt_key
        return self.raw_task[prompt_key] if prompt_key in self.raw_task else None  # type: ignore

    # Deprecated property, will be removed in the future
    @property
    def truth(self) -> Union[str, None]:
        response_key = self.format_args.response_key
        return self.raw_task[response_key] if response_key in self.raw_task else None  # type: ignore

    @property
    def api_key(self) -> str:
        if self.batch_id is None or self.task_id is None or self.run_id is None:
            raise ValueError("batch_id, task_id, and run_id must be set before generating API_KEY.")
        return f"{self.batch_id}/{self.task_id}/{self.run_id}"

    def to_dict(self) -> dict:
        return self.raw_task  # type: ignore


class WorkflowBase:
    """The base workflow interface."""

    def __init__(self, task: Task, model: ModelWrapper, **kwargs) -> None:
        self.task = task
        self.model = model
        self.model.set_api_key(task.api_key)  # set the API key for the rollout model
        self.logger = get_logger(__name__)

    @abstractmethod
    async def execute(self) -> Status:
        """Execute the workflow and return a Status object."""

    def reset(self, task: Task):
        """Reset the workflow with a new task."""
        self.task = task
        self.model.set_api_key(task.api_key)  # set the API key for the rollout model


class Workflow(WorkflowBase):
    """The base workflow class.

    A workflow is a runnable object which generates a list of experiences.

    Attributes:
        auxiliary_model_wrappers: List of ModelWrapper instances for auxiliary models.
        auxiliary_models: List of OpenAI clients (sync or async based on is_async) for auxiliary models.
    """

    can_reset: bool = False  # whether the workflow can be reset with a new task. If true, `reset()` must be implemented.
    can_repeat: bool = False  # whether the workflow can be repeated multiple times. If true, `set_repeat_times()` must be implemented.
    is_async: bool = False  # whether the workflow runs in async mode. If true, `run_async()` must be implemented, else `run()` must be implemented.

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model)
        # Store ModelWrapper instances
        self.auxiliary_model_wrappers = auxiliary_models
        # Get OpenAI clients from ModelWrapper (async or sync based on workflow type)
        self.auxiliary_models: Optional[Union[List[openai.OpenAI], List[openai.AsyncOpenAI]]] = None
        if auxiliary_models:
            if self.is_async:
                self.auxiliary_models = [m.get_openai_async_client() for m in auxiliary_models]
            else:
                self.auxiliary_models = [m.get_openai_client() for m in auxiliary_models]
        self.run_id_base = 0
        self.repeat_times = 1

    def set_repeat_times(self, repeat_times: int, run_id_base: int) -> None:
        """
        Set the number of times to repeat the workflow.
        Args:
            repeat_times (int): number of times to repeat the workflow (if repeatable).
            run_id_base (int): base run_id for setting run_id in experiences.
        """
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    def set_single_run_context(self, run_id_base: int) -> None:
        """
        Set the workflow context for a single non-repeat run.

        This only updates runner bookkeeping fields and intentionally avoids
        repeat-workflow side effects such as changing rollout_args.n.
        """
        self.repeat_times = 1
        self.run_id_base = run_id_base

    def run(self) -> List[Experience]:
        """Run workflow and return a list of experiences."""
        raise NotImplementedError

    async def run_async(self) -> List[Experience]:
        """Run workflow in async and return a list of experiences."""
        raise NotImplementedError

    async def execute(self) -> Status:
        if self.is_async:
            exps = await self.run_async()
        else:
            exps = self.run()
        await self.model.overwrite_history_experiences_async(
            experiences=exps, key=self.task.api_key
        )
        return Status(
            completed_runs=self.can_repeat and self.repeat_times or 1,
            total_runs=self.can_repeat and self.repeat_times or 1,
            metrics=[exp.metrics for exp in exps if exp.metrics is not None],
            successful_ids=[self.task.api_key],
        )


class MultiTurnWorkflow(Workflow):
    """
    The base workflow class for concatenated multi-turn tasks.
    """

    def _build_experience_from_converted(
        self, converted_experience, reward, info={}, truncate_status=None
    ) -> Experience:
        """Private helper method to build Experience from converted_experience.

        Args:
            converted_experience: The converted experience from the model.
            reward: The reward value.
            info: Additional info dictionary.
            truncate_status: Optional truncate status to override.

        Returns:
            Experience: The constructed Experience object.
        """
        if converted_experience.truncate_status == "response_truncated":
            reward = 0.0

        tokens = converted_experience.tokens
        log_probs = converted_experience.logprobs
        assert converted_experience.action_mask is not None
        generation_mask = converted_experience.action_mask
        log_probs = log_probs * generation_mask

        metrics = {}
        for k, v in info.items():
            if isinstance(v, float) or isinstance(v, int):
                metrics[k] = float(v)

        experience = Experience(
            tokens=tokens,
            action_mask=generation_mask,
            prompt_length=converted_experience.prompt_length,
            prompt_text=converted_experience.prompt_text,
            response_text=converted_experience.response_text,
            truncate_status=converted_experience.truncate_status or truncate_status,
            reward=reward,
            logprobs=log_probs,
            info=info,
            metrics=metrics,
        )
        return experience

    def process_messages_to_experience(
        self, messages, reward, info={}, truncate_status=None
    ) -> Experience:
        converted_experience = self.model.convert_messages_to_experience(messages)
        return self._build_experience_from_converted(
            converted_experience,
            reward,
            info,
            converted_experience.truncate_status or truncate_status,
        )

    async def process_messages_to_experience_async(
        self, messages, reward, info={}, truncate_status=None
    ) -> Experience:
        converted_experience = await self.model.convert_messages_to_experience_async(messages)
        return self._build_experience_from_converted(
            converted_experience,
            reward,
            info,
            converted_experience.truncate_status or truncate_status,
        )


class BaseSimpleWorkflow(Workflow):
    """A simple workflow for single-round tasks, which use the batch generation
    API to generate multiple responses in one call."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        self.reset(task)
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

    def reset(self, task: Task):
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix
        self.reward_fn_args = task.reward_fn_args

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        reward_fn = task.reward_fn
        if isinstance(reward_fn, type) and issubclass(reward_fn, RewardFn):
            self.reward_fn: RewardFn = reward_fn(**self.reward_fn_args)
        else:
            raise ValueError("`reward_fn` must be a subclass of `RewardFn`")

    def set_repeat_times(self, repeat_times, run_id_base):
        super().set_repeat_times(repeat_times, run_id_base)
        self.task.rollout_args.n = repeat_times

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        """Format messages for the instruct model."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.task_desc})
        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})
        return messages


class SimpleWorkflow(BaseSimpleWorkflow):
    """A workflow for simple single-round task."""

    can_reset: bool = True
    can_repeat: bool = True
    is_async: bool = False

    def run(self) -> List[Experience]:
        # TODO: Optimize the generate function
        messages = self.format_messages()

        self.logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)
        for i, response in enumerate(responses):
            reward_dict = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
            )
            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(reward_dict)
            reward = sum(reward_dict.values())
            response.reward = reward
            response.eid.run = i + self.run_id_base

            self.logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
            )
        return responses


class AsyncSimpleWorkflow(BaseSimpleWorkflow):
    can_reset: bool = True
    can_repeat: bool = True
    is_async: bool = True

    async def run_async(self) -> List[Experience]:
        # TODO: Optimize the generate function
        messages = self.format_messages()

        self.logger.info("start chat")
        responses = await self.model.chat_async(messages, **self.rollout_args)
        for i, response in enumerate(responses):
            reward_dict = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
            )

            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(reward_dict)
            reward = sum(reward_dict.values())
            response.reward = reward
            response.eid.run = i + self.run_id_base

            self.logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
            )
        return responses


class MathWorkflow(SimpleWorkflow):
    """A workflow for math tasks as introduced in DeepSeek-R1."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        self.reset(task)
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

    def reset(self, task: Task):
        from trinity.common.rewards.math_reward import MathRewardFn

        if task.reward_fn is None:
            task.reward_fn = MathRewardFn
        if task.reward_fn == MathRewardFn and task.format_args.system_prompt is None:
            task.format_args.system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>.
"""
        # call the SimpleWorkflow.reset
        super().reset(task)


class AsyncMathWorkflow(AsyncSimpleWorkflow, MathWorkflow):
    pass


class WorkflowWithRecording(WorkflowBase):
    """A workflow that using the rollout model's built-in recording path to capture
    experience data.

    This interface is designed for complex agentic workflows (e.g., QwenPaw, Claude Code)
    which are hard to extract experience data from the agent itself.

    It provides `base_url` and `api_key` to the OpenAI API of the rollout model, and the
    workflow can use them to call the model and the model will record the experience data
    automatically.
    After the agentic workflow is completed, the workflow can call `update_reward` to update
    the recorded experience data with the reward and optional info.
    """

    can_reset: bool = False
    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model)
        # Store ModelWrapper instances
        self.auxiliary_models = auxiliary_models

    @property
    def base_url(self) -> str:
        """BASE_URL of the OpenAI API of the rollout model."""
        return self.model.base_url

    @property
    def api_key(self) -> str:
        """API_KEY of the OpenAI API of the rollout model."""
        return self.task.api_key

    @property
    def model_name(self) -> str:
        """Model name of the rollout model."""
        return self.model.model_name

    async def run_async(self) -> Metrics:
        """Run workflow asynchronously and return metrics for the completed run."""
        raise NotImplementedError

    async def execute(self) -> Status:
        """Execute the workflow and normalize the user return value to Status."""
        result = await self.run_async()
        return self._to_status(result)

    def _to_status(self, result: Metrics) -> Status:
        return Status(
            completed_runs=1,
            total_runs=1,
            metrics=[result],
            successful_ids=[self.task.api_key],
        )

    async def update_reward(
        self,
        reward: float,
        info: Optional[dict] = None,
        sample_ids: Optional[List[str]] = None,
    ) -> None:
        """Update recorded experiences for one run with reward and optional info."""
        await self.model.update_experience_reward_async(
            key=self.api_key,
            reward=reward,
            info=info,
            sample_ids=sample_ids,
        )

    def set_single_run_context(self, run_id_base: int) -> None:
        """Only a placeholder to align with the Workflow interface.
        This workflow does not support repeat runs."""
        pass
