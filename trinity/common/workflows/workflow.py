# -*- coding: utf-8 -*-
"""Base Workflow Class"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, List, Optional, Type, Union

from trinity.common.config import FormatConfig, GenerationConfig
from trinity.common.experience import Experience
from trinity.common.rewards.reward_fn import RewardFn
from trinity.utils.log import get_logger

if TYPE_CHECKING:
    import openai

    from trinity.common.models.model import ModelWrapper


@dataclass
class Task(dict):
    """A Task class that defines a task and its associated reward function / workflow."""

    workflow: Type[Workflow] = None
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

    index: dict = field(default_factory=dict)

    def to_workflow(
        self,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ) -> Workflow:
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

    def to_dict(self) -> dict:
        return self.raw_task  # type: ignore


class Workflow:
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
        self.task = task
        self.model = model
        # Store ModelWrapper instances
        self.auxiliary_model_wrappers = auxiliary_models
        # Get OpenAI clients from ModelWrapper (async or sync based on workflow type)
        self.auxiliary_models: Optional[Union[List[openai.OpenAI], List[openai.AsyncOpenAI]]] = None
        if auxiliary_models:
            if self.__class__.is_async:
                self.auxiliary_models = [m.get_openai_async_client() for m in auxiliary_models]
            else:
                self.auxiliary_models = [m.get_openai_client() for m in auxiliary_models]
        self.run_id_base = 0
        self.logger = get_logger(__name__)

    @property
    def resettable(self):
        """Deprecated, use cls.can_reset instead."""
        return self.__class__.can_reset

    @property
    def repeatable(self):
        """Deprecated, use cls.can_repeat instead.
        A workflow is repeatable if it can be run multiple times within the run() or run_async() method.
        """
        return self.__class__.can_repeat

    @property
    def asynchronous(self):
        """Deprecated, use cls.is_async instead.
        Whether the workflow runs in async mode."""
        return self.__class__.is_async

    def reset(self, task: Task):
        """Reset the workflow."""
        raise NotImplementedError

    def set_repeat_times(self, repeat_times: int, run_id_base: int) -> None:
        """
        Set the number of times to repeat the workflow.
        Args:
            repeat_times (int): number of times to repeat the workflow (if repeatable).
            run_id_base (int): base run_id for setting run_id in experiences.
        """
        raise NotImplementedError(
            "set_repeat_times() must be implemented for a repeatable workflow."
        )

    def run(self) -> List[Experience]:
        """Run workflow and return a list of experiences."""
        raise NotImplementedError

    async def run_async(self) -> List[Experience]:
        """Run workflow in async and return a list of experiences."""
        raise NotImplementedError


class MultiTurnWorkflow(Workflow):
    """
    The base workflow class for concatenated multi-turn tasks.
    """

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

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

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
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

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
    is_async: bool = True

    async def run_async(self) -> List[Experience]:
        # TODO: Optimize the generate function
        messages = self.format_messages()

        self.logger.debug("start chat")
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
