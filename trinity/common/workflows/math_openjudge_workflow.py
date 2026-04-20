# -*- coding: utf-8 -*-
"""We include the math workflow with rm-gallery reward in this file."""

from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import SimpleWorkflow, Task


class MathOpenJudgeWorkflow(SimpleWorkflow):
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

    def run(self) -> List[Experience]:
        messages = self.format_messages()

        self.logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)
        for i, response in enumerate(responses):
            reward_dict = self.reward_fn(  # type: ignore
                response,
                messages,
                ground_truth=self.truth,
            )

            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(reward_dict)
            # Use the dedicated "reward" key when present (e.g. OpenJudge reward fns),
            # otherwise fall back to summing all dimension scores (legacy rm_gallery behaviour).
            reward = reward_dict.get("reward", sum(reward_dict.values()))
            response.reward = reward
            response.eid.run = i + self.run_id_base

            self.logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
            )
        return responses


class AsyncMathOpenJudgeWorkflow(MathOpenJudgeWorkflow):
    is_async: bool = True

    async def run_async(self) -> List[Experience]:
        messages = self.format_messages()

        self.logger.debug("start chat")
        responses = await self.model.chat_async(messages, **self.rollout_args)
        for i, response in enumerate(responses):
            reward_dict = self.reward_fn(  # type: ignore
                response,
                messages,
                ground_truth=self.truth,
            )

            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(reward_dict)
            # Use the dedicated "reward" key when present (e.g. OpenJudge reward fns),
            # otherwise fall back to summing all dimension scores (legacy rm_gallery behaviour).
            reward = reward_dict.get("reward", sum(reward_dict.values()))
            response.reward = reward
            response.eid.run = i + self.run_id_base

            self.logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
            )
        return responses
