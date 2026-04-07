# -*- coding: utf-8 -*-
"""A workflow with LLM-as-a-judge."""
import json
import os
from typing import List, Optional, Tuple

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import SimpleWorkflow, Task


class RubricJudgeWorkflow(SimpleWorkflow):
    """A workflow using LLM-as-a-judge and rubrics to get the reward.

    Adapted from https://arxiv.org/pdf/2507.17746
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
        """Modified from SimpleWorkflow.reset"""
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix

        if self.system_prompt is None:
            self.system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    """

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth
        self.rubric = self.raw_task.get("rubric", [])

    def _build_judge(self) -> openai.OpenAI:
        """Return the judge client.  Subclasses may override this to supply a
        different client (e.g. an external API) without duplicating ``run``."""
        assert (
            self.auxiliary_models is not None
        ), "auxiliary_models must be set when using the default judge."
        return self.auxiliary_models[0]

    def run(self) -> List[Experience]:
        """Modified from SimpleWorkflow.run"""
        messages = self.format_messages()
        judge = self._build_judge()

        self.logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)

        judge_success_list = []
        for i, response in enumerate(responses):
            judge_success, reward = self.get_judge_reward(
                response=response.response_text, judge=judge
            )
            response.reward = reward
            response.eid.run = i + self.run_id_base
            judge_success_list.append(judge_success)

            if i == 0:
                self.logger.debug(
                    f"self.task_desc: {self.task_desc}, messages: {messages}, "
                    f"response: {response.response_text}, reward: {response.reward}"
                )

        judge_success_rate = (
            sum(judge_success_list) / len(judge_success_list) if judge_success_list else 0.0
        )
        for response in responses:
            if response.metrics is None:
                response.metrics = {}
            response.metrics.update({"judge_success": float(judge_success_rate)})

        return responses

    def get_judge_reward(self, response: str, judge: openai.OpenAI) -> Tuple[bool, float]:
        """Get rewards with LLM-as-a-judge
        The prompts are adapted from RAR-IMPLICIT method in https://arxiv.org/pdf/2507.17746
        """
        # Step 1: format prompts
        # system prompt
        ruler_system_prompt = """You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.
Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a lower score, while a response that satisfies all rubrics should receive a higher score.
Start your response with a valid JSON object that starts with "```json" and ends with "```". The JSON object should contain
a single key "rating" and the value should be an integer between 1 and 10.
Example response:
```json
{
"rating": 7
}```"""
        # user prompt
        if len(self.rubric) > 0:
            rubric_prompt_parts = [
                f"Rubric {i} (weight: {single_rubric['weight']}): {single_rubric['description']}"
                for i, single_rubric in enumerate(self.rubric)
            ]
            rubric_list_string = "\n".join(rubric_prompt_parts)
        else:
            self.logger.warning("No rubric is provided!")
            rubric_list_string = "Rubrics are not provided."

        ruler_user_prompt = f"""Given the following prompt, response, and rubrics, please rate the overall quality of the response on a scale of 1 to 10 based
on how well it satisfies the rubrics.
<prompt>
{self.task_desc}
</prompt>
<response>
{response}
</response>
<rubrics>
{rubric_list_string}
</rubrics>
Your JSON Evaluation:
""".strip()

        # Step 2: invoke judge LLM
        messages = [
            {"role": "system", "content": ruler_system_prompt},
            {"role": "user", "content": ruler_user_prompt},
        ]
        completion = judge.chat.completions.create(
            model=judge.model_path, messages=messages, stream=False, temperature=0.0
        )
        judge_response = completion.choices[0].message.content
        self.logger.debug(f"LLM judge response: {judge_response}")

        # Step 3: extract score from judge's response (expecting a JSON block with "rating")
        try:
            # Extract content between ```json and ```
            start_tag = "```json"
            start_index = judge_response.find(start_tag)
            if start_index == -1:
                start_tag = "```"
                start_index = judge_response.find(start_tag)

            if start_index == -1:
                self.logger.warning("No JSON code block found in judge response.")
                return False, 0.0

            end_index = judge_response.find("```", start_index + len(start_tag))
            if end_index == -1:
                self.logger.warning("Malformed JSON code block in judge response.")
                return False, 0.0

            json_str = judge_response[start_index + len(start_tag) : end_index].strip()
            parsed = json.loads(json_str)
            rating = parsed.get("rating")

            if not isinstance(rating, (int, float)) or not (1 <= rating <= 10):
                self.logger.warning(f"Invalid or out-of-range rating: {rating}")
                return False, 0.0

            normalized_score = rating * 0.1  # Normalize 1-10 to 0-1 scale
            return True, normalized_score

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from judge response: {e}")
            return False, 0.0
        except Exception as e:
            self.logger.warning(f"Unexpected error when processing judge response: {e}")
            return False, 0.0


class RubricJudgeWorkflowWithAPI(RubricJudgeWorkflow):
    """Rubric judge workflow using an external OpenAI-compatible API as the judge.

    Example of workflow_args:
        judge_model_name: "gpt-4o"
        judge_api_base_url_env: "OPENAI_BASE_URL"
        judge_api_key_env: "OPENAI_API_KEY"
    """

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

        workflow_args = task.workflow_args or {}
        judge_model_name = workflow_args.get("judge_model_name")
        base_url_env = workflow_args.get("judge_api_base_url_env", "OPENAI_BASE_URL")
        api_key_env = workflow_args.get("judge_api_key_env", "OPENAI_API_KEY")
        if not judge_model_name:
            raise ValueError(
                "Judge model name is missing. Set `judge_model_name` in workflow_args."
            )
        judge_base_url = os.getenv(base_url_env, "")
        if not judge_base_url:
            raise ValueError(f"Judge base URL is missing. Set env `{base_url_env}`.")

        self._judge_client = openai.OpenAI(
            base_url=judge_base_url, api_key=os.getenv(api_key_env, "")
        )
        self._judge_model_name = judge_model_name
        self.logger.info(
            "Initialized external judge model: base_url=%s model=%s",
            judge_base_url,
            judge_model_name,
        )

    def _build_judge(self) -> openai.OpenAI:
        """Return the pre-built external API judge client."""
        client = self._judge_client
        client.model_path = self._judge_model_name  # expected by get_judge_reward
        return client
