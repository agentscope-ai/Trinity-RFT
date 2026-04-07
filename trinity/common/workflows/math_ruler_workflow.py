# -*- coding: utf-8 -*-
"""Math workflow with RULER."""
import ast
from typing import Any, List, Optional, Tuple

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.math_reward import MathRewardFn
from trinity.common.workflows.workflow import SimpleWorkflow, Task


class MathRULERWorkflow(SimpleWorkflow):
    """A workflow for math with RULER reward function.

    Modified from `MathWorkflow`.
    Adapted from https://github.com/OpenPipe/ART/blob/main/src/art/rewards/ruler.py
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

    def reset(self, task: Task):
        """
        Note that in this workflow, MathRewardFn is only used for calculating the 'golden reward',
        whereasa the rewards used by RL training are calculated by RULER.
        """

        if task.reward_fn is None:
            task.reward_fn = MathRewardFn
        if task.reward_fn == MathRewardFn and task.format_args.system_prompt is None:
            task.format_args.system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>.
"""
        # call the SimpleWorkflow.reset
        super().reset(task)

    def run(self) -> List[Experience]:
        """Modified from SimpleWorkflow.run"""

        messages = self.format_messages()

        self.logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)

        for i, response in enumerate(responses):
            gold_reward_dict = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
            )

            if response.metrics is None:
                response.metrics = {}

            response.metrics.update(gold_reward_dict)
            gold_reward = sum(gold_reward_dict.values())
            response.metrics.update({"gold_reward": gold_reward})
            response.eid.run = i + self.run_id_base

            self.logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, gold_reward: {gold_reward}"
            )

        # === RULER scores as rewards ===
        assert (
            self.auxiliary_models is not None
        ), "Current implementation of RULER requires that auxiliary_models is not None."
        judge_success, ruler_scores = self.get_ruler_scores(
            responses=responses, judge=self.auxiliary_models[0]
        )
        for i, response in enumerate(responses):
            response.reward = ruler_scores[i]
            response.metrics.update({"judge_success": float(judge_success)})

        return responses

    def get_ruler_scores(self, responses: List[Experience], judge: Any) -> Tuple[bool, List[float]]:
        """Get RULER scores"""

        num_responses = len(responses)

        # Step 1: format prompt for judge
        ruler_system_prompt = f"You are a fair judge. The user will provide a question and {num_responses} candidate solutions to it. Your task is to compare the solutions, see how well they resolve the question, and assign a score within the range [0, 1] for each solution."

        question_prompt = (
            f"Question: {self.task_desc}\n\n"
            f"""Solution format requirement: first think about the reasoning process in the mind and then provides the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>."""
        )

        solutions_prompt_parts = [
            f"Candidate solution {i + 1}: {response.response_text}"
            for i, response in enumerate(responses)
        ]
        solutions_prompt = "\n\n".join(solutions_prompt_parts)

        ruler_user_prompt = f"""
Below is a question and several candidate solutions.

{question_prompt}

{solutions_prompt}

Please assign a score within the range [0, 1] for each of them, reflecting how well they solve the question.
You may compare them against each other and think step by step before returning your final scores, but keep your reasoning process brief and concise when possible.

Conclude your response with a list of scores, in the following format: [score for solution 1, score for solution 2, ..., score for solution {num_responses}]
"""

        # Step 2: invoke judge LLM
        messages = [
            {"role": "system", "content": ruler_system_prompt},
            {"role": "user", "content": ruler_user_prompt},
        ]
        completion = judge.chat.completions.create(
            model=judge.model_path, messages=messages, stream=False
        )
        judge_response = completion.choices[0].message.content
        self.logger.info(f"LLM judge response: {judge_response}")

        # Step 3: extract scores from judge's response
        idx1, idx2 = judge_response.rfind("["), judge_response.rfind("]")
        if (idx1 == -1) or (idx2 == -1) or (idx1 > idx2):
            self.logger.warning(
                "Unable to extract a list from judge response, set scores to all zero."
            )
            return False, [0.0 for _ in range(num_responses)]
        lst_as_str = judge_response[idx1 : (idx2 + 1)]
        try:
            scores = ast.literal_eval(lst_as_str)
            scores = [max(0.0, min(1.0, score)) for score in scores]  # clip to range [0, 1]
            if len(scores) != num_responses:
                self.logger.warning(
                    "The length of list in judge response does not match num_responses."
                )
                return False, [0.0 for _ in range(num_responses)]
            return True, scores
        except Exception:
            self.logger.warning(
                "Unable to parse the list in judge response, set scores to all zero."
            )
            return False, [0.0 for _ in range(num_responses)]


class AsyncMathRULERWorkflow(MathRULERWorkflow):
    is_async: bool = True

    async def run_async(self) -> List[Experience]:
        """Modified from SimpleWorkflow.run"""

        messages = self.format_messages()

        self.logger.debug("start chat")
        responses = await self.model.chat_async(messages, **self.rollout_args)

        for i, response in enumerate(responses):
            gold_reward_dict = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
            )

            if response.metrics is None:
                response.metrics = {}

            response.metrics.update(gold_reward_dict)
            gold_reward = sum(gold_reward_dict.values())
            response.metrics.update({"gold_reward": gold_reward})
            response.eid.run = i + self.run_id_base

            self.logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, gold_reward: {gold_reward}"
            )

        # === RULER scores as rewards ===
        assert (
            self.auxiliary_models is not None
        ), "Current implementation of RULER requires that auxiliary_models is not None."
        judge_success, ruler_scores = self.get_ruler_scores(
            responses=responses, judge=self.auxiliary_models[0]
        )
        for i, response in enumerate(responses):
            response.reward = ruler_scores[i]
            response.metrics.update({"judge_success": float(judge_success)})

        return responses
