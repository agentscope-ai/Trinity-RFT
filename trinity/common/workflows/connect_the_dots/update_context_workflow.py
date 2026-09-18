# -*- coding: utf-8 -*-
"""Update-context workflows for CoD packs.

After a task in a pack is solved, the update-context episode reads the previous
context together with the solved trajectory and produces an updated context that
conditions the later tasks in the pack.

``AsyncCoDUpdateContextWorkflow`` is the default implementation: it builds the
prompt, parses the response and scores it with a length-shaped reward, generating
through ``chat_async``. ``AsyncCoDUpdateContextAgentWorkflow`` inherits all of
that and replaces only the generation path with a single-shot AgentScope agent.
Override ``build_messages`` / ``parse_context`` / ``compute_reward`` for a
different context-update strategy.
"""

from dataclasses import asdict
from typing import List, Optional, Tuple

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.connect_the_dots.agentscope_utils import (
    build_agentscope_single_turn_agent,
)
from trinity.common.workflows.connect_the_dots.cod_utils import CoDPrompts
from trinity.common.workflows.workflow import Task, Workflow


class AsyncCoDUpdateContextWorkflow(Workflow):
    """Async workflow for the CoD update-context episode.

    ``CoDWorkflow`` runs it once per task transition within a pack. The previous
    context, solved trajectory, reward and feedback are injected via
    ``set_context_inputs`` before each ``run_async``. The default strategy
    iteratively refines a ``Hints:`` block and scores it with a length-shaped
    reward; generation goes through ``chat_async``.
    """

    can_reset: bool = True
    can_repeat: bool = False
    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        # Inputs for the next context update, set via set_context_inputs.
        self.prev_context: str = ""
        self.trajectory: str = ""
        self.reward: float = 0.0
        self.feedback: str = ""
        self.reset(task)

    def reset(self, task: Task):
        """Bind task-derived config and clear the per-call inputs."""
        self.task = task
        self.prev_context = ""
        self.trajectory = ""
        self.reward = 0.0
        self.feedback = ""

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def set_context_inputs(
        self, *, prev_context: str, trajectory: str, reward: float, feedback: str
    ) -> None:
        """Set the inputs for the next context update."""
        self.prev_context = prev_context
        self.trajectory = trajectory
        self.reward = reward
        self.feedback = feedback

    def _append_token_limit(self, prompt: str) -> str:
        """Append a response-length instruction when configured."""
        max_tokens = self.task.workflow_args.get("max_response_tokens_restraint")
        if max_tokens:
            prompt += f"\n\nPlease limit your response to {max_tokens} tokens."
        return prompt

    def build_messages(self) -> List[dict]:
        """Build the chat messages that ask the model to refine the hints."""
        hint_example = self.task.workflow_args.get("hint_example", False)
        sys_prompt = CoDPrompts.sys_prompt_gen_hint_iteratively(hint_example)
        sys_prompt = self._append_token_limit(sys_prompt)
        user_prompt = CoDPrompts.user_prompt_gen_hint_iteratively(
            prev_hint=self.prev_context,
            trajectory=self.trajectory,
            reward=self.reward,
            feedback=self.feedback,
        )
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse_context(self, response: str) -> Tuple[str, bool]:
        """Extract the updated context and a parse-success flag from the response."""
        return CoDPrompts.extract_hint(response)

    def compute_reward(self, exp: Experience, parse_success: bool) -> float:
        """Length-shaped penalty for the generated context.

        Penalizes a parse failure, an over-short response, or an over-long one
        ramped from ``len_zero_penalty`` to ``len_max_penalty``, scaled by
        ``hint_penalty_coef``; returns 0 when these penalties are unset. Override
        for a different reward.
        """
        args = self.task.workflow_args
        hint_penalty_coef = args.get("hint_penalty_coef", 0.0)
        len_zero_penalty = args.get("len_zero_penalty", None)
        len_max_penalty = args.get("len_max_penalty", None)
        len_min_penalty = args.get("len_min_penalty", None)

        # Parse failure is the harshest signal: full penalty regardless of length.
        if not parse_success:
            return -1.0 * hint_penalty_coef
        if (len_zero_penalty is None) or (len_max_penalty is None):
            return 0.0
        assert (
            len_zero_penalty < len_max_penalty
        ), "len_zero_penalty must be smaller than len_max_penalty."

        resp_len = len(exp.tokens) - exp.prompt_length
        if len_min_penalty is not None and resp_len < len_min_penalty:
            return -1.0 * hint_penalty_coef
        ramp = min(
            1.0,
            max(
                0.0,
                (resp_len - len_zero_penalty) / (len_max_penalty - len_zero_penalty),
            ),
        )
        return -1.0 * hint_penalty_coef * ramp

    def _finalize(self, exp: Experience, messages: List[dict]) -> List[Experience]:
        """Parse the response, score it, and attach the context to ``exp.info``."""
        new_context, parse_success = self.parse_context(exp.response_text or "")
        exp.reward = self.compute_reward(exp, parse_success)

        if exp.metrics is None:
            exp.metrics = {}
        exp.metrics["hint_parse_success"] = 1.0 * parse_success

        exp.info["sys_prompt"] = messages[0]["content"]
        exp.info["user_prompt"] = messages[-1]["content"]
        exp.info["hint_parse_success"] = parse_success
        exp.info["hint"] = new_context
        exp.info["prev_hint"] = self.prev_context
        return [exp]

    async def run_async(self) -> List[Experience]:
        """Generate one updated context and return it as a single experience.

        The updated context is returned via ``exp.info["hint"]`` so callers do
        not depend on the parsing details.
        """
        messages = self.build_messages()
        rollout_args = self.rollout_args
        rollout_args["n"] = 1
        exps = await self.model.chat_async(messages, **rollout_args)
        return self._finalize(exps[0], messages)


class AsyncCoDUpdateContextAgentWorkflow(AsyncCoDUpdateContextWorkflow):
    """Update-context episode whose generation runs through an AgentScope agent.

    Prompt building, parsing and reward are inherited from the base workflow;
    only the generation path is replaced with a single-shot AgentScope agent on
    an isolated-history model clone.
    """

    requires_isolated_model_history: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self._agent = None
        self._agent_rollout_args = None

    async def _ensure_agent(self):
        rollout_args = asdict(self.task.rollout_args)
        if self._agent is None or self._agent_rollout_args != rollout_args:
            self._agent = await build_agentscope_single_turn_agent(
                name="cod_context_update",
                model=self.model,
                rollout_args=self.task.rollout_args,
            )
            self._agent_rollout_args = rollout_args
        return self._agent

    async def run_async(self) -> List[Experience]:
        """Generate one updated context through the AgentScope agent."""
        self.model.history.clear()
        messages = self.build_messages()
        agent = await self._ensure_agent()
        await agent(messages)
        exp = self.model.extract_experience_from_history()[-1]
        return self._finalize(exp, messages)
