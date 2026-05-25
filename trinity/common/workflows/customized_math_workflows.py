# -*- coding: utf-8 -*-
"""We include the customized math workflows in this file."""

from typing import List

import torch

from trinity.common.experience import Experience
from trinity.common.rewards.math_reward import MathBoxedRewardFn
from trinity.common.workflows.workflow import SimpleWorkflow, Task


class MathBoxedWorkflow(SimpleWorkflow):
    """A workflow for math tasks that give answers in boxed format."""

    def reset(self, task: Task):
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        self.workflow_args = task.workflow_args
        self.reward_fn_args = task.reward_fn_args

        self.use_base = self.workflow_args.get("use_base", False)
        self.with_think = self.workflow_args.get("with_think", False)
        self.format_score_coef = self.workflow_args.get("format_score_coef", 0.1)

        default_prompt = (
            """Please reason step by step, and put your final answer within \\boxed{}."""
        )

        default_prompt_with_think = """You are a helpful assistant that solves MATH problems. You should first thinks about the reasoning process in mind and then provides the user with the answer. You should present your reasoning process using the format: <think>\n ...your reasoning process here... </think>\n first. You should always include your final answer in \\boxed{} as closed-form results."""

        if self.system_prompt is None:
            if self.with_think:
                self.system_prompt = default_prompt_with_think
            else:
                self.system_prompt = default_prompt

        if task.reward_fn is None:
            self.reward_fn = MathBoxedRewardFn(**self.reward_fn_args)
        else:
            self.reward_fn = task.reward_fn(**self.reward_fn_args)

    def format_prompt(self):
        prompt_text = ""
        if self.system_prompt:
            prompt_text += "System:" + self.system_prompt
            prompt_text += "\nUser:\n" + self.task_desc + "\nAssistant:\n"
        else:
            prompt_text += "User:\n" + self.task_desc + "\nAssistant:\n"
        return prompt_text

    def run(self) -> List[Experience]:
        if not self.use_base:
            messages = self.format_messages()
        else:
            prompt_text = self.format_prompt()

        self.logger.debug("start chat")
        if not self.use_base:
            responses = self.model.chat(messages, **self.rollout_args)
        else:
            responses = self.model.generate([prompt_text], **self.rollout_args)

        for i, response in enumerate(responses):
            reward_dict = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
                with_think=self.with_think,
                format_score_coef=self.format_score_coef,
                response_token=response.tokens[response.prompt_length :],
            )

            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(reward_dict)
            reward = sum(reward_dict.values())
            response.reward = reward
            response.eid.run = i + self.run_id_base
            if response.truncate_status == "response_truncated" and response.action_mask is not None:
                response.action_mask = torch.zeros_like(response.action_mask, dtype=torch.bool)

            if not self.use_base:
                self.logger.debug(
                    f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
                )
            else:
                self.logger.debug(
                    f"self.task_desc: {self.task_desc}, prompt_text: {prompt_text}, response: {response.response_text}, reward: {reward}"
                )
        return responses


class AsyncMathBoxedWorkflow(MathBoxedWorkflow):
    is_async: bool = True

    async def run_async(self) -> List[Experience]:
        if not self.use_base:
            messages = self.format_messages()
        else:
            prompt_text = self.format_prompt()

        self.logger.debug("start chat")
        if not self.use_base:
            responses = await self.model.chat_async(messages, **self.rollout_args)
        else:
            responses = await self.model.generate_async([prompt_text], **self.rollout_args)

        for i, response in enumerate(responses):
            reward_dict = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
                with_think=self.with_think,
                format_score_coef=self.format_score_coef,
                response_token=response.tokens[response.prompt_length :],
            )

            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(reward_dict)
            reward = sum(reward_dict.values())
            response.reward = reward
            response.eid.run = i + self.run_id_base
            if response.truncate_status == "response_truncated" and response.action_mask is not None:
                response.action_mask = torch.zeros_like(response.action_mask, dtype=torch.bool)

            if not self.use_base:
                self.logger.debug(
                    f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
                )
            else:
                self.logger.debug(
                    f"self.task_desc: {self.task_desc}, prompt_text: {prompt_text}, response: {response.response_text}, reward: {reward}"
                )
        return responses
