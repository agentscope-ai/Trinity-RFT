from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.mm_utils import build_mm_message
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.reward_fn import RewardFn
from trinity.common.workflows.workflow import SimpleWorkflow, Task


class SimpleMMWorkflow(SimpleWorkflow):
    """A workflow for simple single-round task."""

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

    def format_messages(self):
        """Format messages for the instruct model."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append(build_mm_message(self.task_desc, self.images, self.videos))

        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})
        return messages

    def reset(self, task: Task):
        self.format_args = task.format_args
        self.system_prompt = """You are a helpful assistant that solves MATH problems. You should first thinks about the reasoning process in mind and then provides the user with the answer. You should present your reasoning process using the format: <think>\n ...your reasoning process here... </think>\n first. You should always include your final answer in \\boxed{} as closed-form results."""  # TODO: check
        self.reply_prefix = task.format_args.reply_prefix
        self.reward_fn_args = task.reward_fn_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        assert task.raw_task is not None
        self.truth = task.raw_task[task.format_args.response_key] or task.truth

        reward_fn = task.reward_fn
        if isinstance(reward_fn, type) and issubclass(reward_fn, RewardFn):
            self.reward_fn: RewardFn = reward_fn(**self.reward_fn_args)
        else:
            raise ValueError("`reward_fn` must be a subclass of `RewardFn`")

        self.images = self.raw_task.get(task.format_args.image_key, [])
        self.videos = self.raw_task.get(task.format_args.video_key, [])
        self.messages = self.format_messages()

    def run(self) -> List[Experience]:
        # TODO: test generate_mm
        self.logger.debug("start chat")
        responses = self.model.chat(messages=self.messages, **self.rollout_args)
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

        self.logger.debug(f"Generated {len(responses)} responses")
        return responses


class AsyncSimpleMMWorkflow(SimpleMMWorkflow):
    is_async: bool = True

    async def run_async(self) -> List[Experience]:
        # TODO: test generate_mm
        self.logger.debug("start chat")
        responses = await self.model.chat_async(messages=self.messages, **self.rollout_args)
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

        self.logger.debug(f"Generated {len(responses)} responses")
        return responses
