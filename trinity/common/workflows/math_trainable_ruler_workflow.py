# -*- coding: utf-8 -*-
"""Math workflow with trainable RULER."""
import ast
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import numpy as np

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.math_reward import MathRewardFn
from trinity.common.workflows.workflow import SimpleWorkflow, Task

# the probability that the ground truth is assumed to be available for RL
PROBABILITY_GROUND_TRUTH_AVAILABLE = 0.2


class MathTrainableRULERWorkflow(SimpleWorkflow):
    """A workflow for math, where the policy model itself serves as a RULER reward model.
    Modified from `MathRULERWorkflow`.
    RULER is adapted from https://github.com/OpenPipe/ART/blob/main/src/art/rewards/ruler.py
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
        """Modified from MathRULERWorkflow.run"""

        # Part 1: generate responses to the original task (as in usual workflows)
        messages = self.format_messages()
        self.logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)
        gold_rewards = []
        gold_scores_scaled = []
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

            # set task_id explicitly within workflow!
            response.eid.task = str(self.task.task_id)
            response.eid.run = i + self.run_id_base

            gold_rewards.append(gold_reward)
            gold_scores_scaled.append(
                (gold_reward + 0.1) / 1.2
            )  # scale from range [-0.1, 1.1] to [0, 1]

        # Part 2: get and use RULER scores
        ruler_rollout_args = deepcopy(self.rollout_args)
        ground_truth_is_available = np.random.rand() < PROBABILITY_GROUND_TRUTH_AVAILABLE

        if ground_truth_is_available:
            # Assuming that ground truth is accessible to RL:
            # - set exp's reward to gold reward
            # - generate RULER scores for repeat_times, construct ruler_responses
            # - return responses + ruler_responses

            judge_success_rate, ruler_responses, ruler_scores = self.get_ruler_responses(
                responses=responses,
                judge=self.model,  # use the policy model itself as judge!
                ruler_rollout_args=ruler_rollout_args,
                gold_scores=gold_scores_scaled,
            )

            for i, response in enumerate(responses):
                response.reward = gold_rewards[i]
                response.metrics.update({"judge_success": judge_success_rate})

            for i, ruler_response in enumerate(ruler_responses):
                # set task_id explicitly, to distinguish two types of experiences!
                ruler_response.eid.task = str(self.task.task_id) + "-ruler"
                ruler_response.eid.run = i + self.run_id_base

            return responses + ruler_responses

        else:
            # Assuming that ground truth is not accessible to RL:
            # - generate RULER scores only once
            # - set exp's reward to RULER score
            # - return responses

            ruler_rollout_args["n"] = 1
            judge_success_rate, ruler_responses, ruler_scores = self.get_ruler_responses(
                responses=responses,
                judge=self.model,  # use the policy model itself as judge!
                ruler_rollout_args=ruler_rollout_args,
                gold_scores=None,
            )

            for i, response in enumerate(responses):
                response.reward = ruler_scores[i]
                response.metrics.update({"judge_success": judge_success_rate})

            return responses

    def get_ruler_responses(
        self,
        responses: List[Experience],
        judge: Any,
        ruler_rollout_args: Any,
        gold_scores: Optional[List[float]] = None,
    ) -> Tuple[float, List[Experience], List[float]]:
        """Get RULER scores
        Returns:
            judge_success_rate: float
            ruler_responses: List[Experience]
            ruler_scores: List[float]
        """

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

        # Step 2: invoke judge LLM (actually self.model), get ruler_responses: List[Experience]
        messages = [
            {"role": "system", "content": ruler_system_prompt},
            {"role": "user", "content": ruler_user_prompt},
        ]
        ruler_responses = judge.chat(messages, **ruler_rollout_args)

        # Step 3: extract scores from each ruler_response, and update its reward if needed
        ruler_scores = [0.0 for _ in range(num_responses)]
        judge_success_count = 0
        for ruler_response in ruler_responses:
            # default reward is 0; update if gold_scores is provided & judge returns valid scores
            ruler_response.reward = 0.0
            ruler_response_text = ruler_response.response_text
            idx1, idx2 = ruler_response_text.rfind("["), ruler_response_text.rfind("]")

            if (idx1 == -1) or (idx2 == -1) or (idx1 > idx2):
                self.logger.warning("Unable to extract a list from judge response.")
                continue

            lst_as_str = ruler_response_text[idx1 : (idx2 + 1)]
            try:
                scores = ast.literal_eval(lst_as_str)
                scores = [max(0.0, min(1.0, score)) for score in scores]  # clip to range [0, 1]
                if len(scores) == num_responses:
                    judge_success_count += 1
                    ruler_scores = [ruler_scores[i] + scores[i] for i in range(len(ruler_scores))]
                    if gold_scores:
                        mae_error = np.abs(np.array(scores) - np.array(gold_scores)).mean()
                        ruler_response.reward = 1.0 - mae_error
                else:
                    self.logger.warning(
                        "The length of list in judge response does not match num_responses."
                    )
            except Exception:
                self.logger.warning("Unable to parse the list in judge response.")

        if judge_success_count > 0:
            ruler_scores = [score / judge_success_count for score in ruler_scores]
        if len(ruler_responses) > 0:
            judge_success_rate = 1.0 * judge_success_count / len(ruler_responses)
        else:
            judge_success_rate = 0.0

        for ruler_response in ruler_responses:
            if ruler_response.metrics is None:
                ruler_response.metrics = {}
            ruler_response.metrics.update({"judge_success": judge_success_rate})
            ruler_response.metrics.update({"reward_for_judge": ruler_response.reward})

        return judge_success_rate, ruler_responses, ruler_scores
