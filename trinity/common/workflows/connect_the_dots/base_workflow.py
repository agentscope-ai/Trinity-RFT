# -*- coding: utf-8 -*-
"""
Base workflow class for CoD (Connect-the-Dots) multi-step workflows.

This workflow is designed for use with CoDWorkflow, which handles metrics
post-processing via _post_process_task_solving_exp. Therefore, this base class
does NOT broadcast metrics to every experience (avoiding metric explosion like
mean@101, mean@102, etc. in eval logs).
"""

import re
from dataclasses import asdict
from typing import List, Optional, Tuple

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.connect_the_dots.utils import extract_content_between_keys
from trinity.common.workflows.workflow import Task, Workflow


class AsyncCoDMultiStepWorkflow(Workflow):
    """
    Async base class for CoD multi-step workflows.

    This is the recommended base class for CoD multi-step workflows like
    CoDFrozenLakeWorkflow, CoDAlfworldWorkflow, CoDTextWorldWorkflow, etc.

    Unlike RewardPropagationWorkflow, this class does NOT set metrics on every
    experience. Metrics should be handled by CoDWorkflow._post_process_task_solving_exp.

    Subclasses should:
    - Initialize self.memory as a list of message dicts in __init__ or reset
    - The memory should follow the format: [system, user, assistant, user, assistant, ...]
    """

    is_async: bool = True

    def __init__(
        self, *, task: Task, model: ModelWrapper, auxiliary_models=None, use_openai_client=True
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.client: Optional[openai.OpenAI] = None
        if use_openai_client:
            self.client = model.get_openai_client()

        # Memory for conversation history (subclasses should populate this)
        self.memory: List[dict] = []

        # State variables (TODO: merge with those in children classes)
        self.final_reward: float = 0.0  # outcome reward
        self.early_termination_by_format_issue: bool = False  # terminate upon format error

    def reset(self, task: Task):
        """Set task-derived attrs. Subclasses override and call super()."""
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task or {}
        self.task_desc = task.task_desc
        self.reply_prefix = task.format_args.reply_prefix if task.format_args else None
        self.hint = None
        self.icl_examples = None
        self.max_response_tokens_restraint = None

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def _compress_assistant_response(self, response: str) -> str:
        """Various methods of compressing assistant response to reduce context size."""
        context_compression_mode = self.task.workflow_args.get("context_compression_mode", "keep_all")
        if context_compression_mode == "remove_think":
            # TODO: need update, maybe return None when no match of <think> tag
            compressed = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            return compressed.strip()
        elif context_compression_mode == "keep_answer":
            content, ok = extract_content_between_keys(response, "<answer>", "</answer>")
            if ok:
                # re-wrap with tags to allow repeated compression
                return f"<answer>{content.strip()}</answer>"
            return "(Failed to parse answer from response)"
        elif context_compression_mode == "keep_all":
            return response
        else:
            raise ValueError(f"Invalid context_compression_mode {context_compression_mode}")

    def _compress_memory(self) -> None:
        """Compress all assistant messages in memory by stripping <think> blocks.

        Called after each step_async to prevent context bloat in multi-turn
        conversations. The thinking/reasoning content is removed while
        preserving answers and other non-thinking output.

        !!! TODO: change to compressing the last assistant message? !!!
        """
        for msg in self.memory:
            if msg.get("role") == "assistant":
                msg["content"] = self._compress_assistant_response(msg["content"])

    async def run_async(self) -> list[Experience]:
        """Run the workflow asynchronously and return a list of experiences."""
        experiences = []
        step = 0
        for step in range(self.max_step_num):
            # Run a single step of the agent application and get experiences directly
            continue_run, exps = await self.step_async(step_num=step)
            # Compress prior assistant messages in memory to avoid context bloat
            self._compress_memory()
            # Set the step number in each experience
            for exp in exps:
                exp.eid.step = step
            # Store the step experiences
            experiences.extend(exps)
            if not continue_run:
                break

        # Calculate final reward and propagate to all experiences
        reward = await self.reward_async(experiences)
        for exp in experiences:
            exp.reward = reward
            if exp.metrics is None:
                exp.metrics = {}

        # Only set actual_env_steps and trajectory on the LAST experience
        # (CoDWorkflow._post_process_task_solving_exp will handle per-experience metrics)
        if experiences:
            experiences[-1].metrics["actual_env_steps"] = step + 1
            experiences[-1].info["trajectory"] = self._build_trajectory()

        return experiences

    def _augment_system_prompt(self, sys_prompt: str) -> str:
        """Augment system prompt, injecting hint and max_response_tokens_restraint."""
        if self.hint:
            sys_prompt = sys_prompt + (
                "\n\n## Hints that might help\n"
                "Below are some hints that might be helpful, but there is no guarantee "
                "that they must be correct or applicable to the current task. You might "
                "leverage them as prior knowledge, while incorporating new information "
                "from your own experience in interacting with the environment.\n\n"
                f"{self.hint}"
            )
        if self.max_response_tokens_restraint:
            sys_prompt = sys_prompt + (
                f"\n\n## Response length limit\n"
                f"Please limit your response (including your thinking process) "
                f"to {self.max_response_tokens_restraint} tokens."
            )
        return sys_prompt

    @staticmethod
    def _strip_system_prompt(sys_content: str) -> str:
        """Strip hint and token limit from system prompt, keeping only task rules.
        Markers correspond to those in the _augment_system_prompt method above.
        """
        for marker in [
            "\n\n## Hints",
            "\n\n## Response length limit",
        ]:
            sys_content = sys_content.split(marker)[0]
        return sys_content

    def _build_trajectory(self) -> str:
        """Build trajectory string from memory for hint generation.

        Includes the system prompt (with hint/token limit stripped) so that
        hint generation can see the task rules, followed by observation/action
        pairs from the conversation history.

        Subclasses can override this method for custom trajectory formatting.
        """
        if not self.memory:
            return ""

        trajectory_parts = []
        step_num = 0
        i = 0
        while i < len(self.memory):
            msg = self.memory[i]
            # Include system prompt with hint/token limit stripped
            if msg.get("role") == "system":
                clean_sys = self._strip_system_prompt(msg.get("content", ""))
                if clean_sys.strip():
                    trajectory_parts.append(f"System:\n{clean_sys}")
                i += 1
                continue
            # Process user message
            if msg.get("role") == "user":
                step_num += 1
                observation = msg.get("content", "")
                # Remove ICL examples if present
                if "\n\nHere are some reference examples:\n\n" in observation:
                    observation = observation.split("\n\nHere are some reference examples:\n\n")[0]
                # Find the last assistant message before next user (to skip reply_prefix)
                action = "(no action)"
                j = i + 1
                while j < len(self.memory) and self.memory[j].get("role") == "assistant":
                    action = self.memory[j].get("content", "")
                    j += 1
                trajectory_parts.append(f"Step {step_num}:\nObservation: {observation}\nAction: {action}")
                i = j
            else:
                i += 1

        return "\n\n".join(trajectory_parts)

    async def step_async(self, step_num: int) -> Tuple[bool, List[Experience]]:
        """Run a single step of your agent application asynchronously.

        Args:
            step_num (int): The current step number.

        Returns:
            Tuple[bool, List[Experience]]: A tuple of (continue_run, experiences).
                - continue_run: Whether to continue running the agent application.
                - experiences: List of experiences from this step.
        """
        raise NotImplementedError

    def _get_feedback(self) -> str:
        """Get environment feedback that will be added to exps[-1].info["feedback"]"""
        raise NotImplementedError

    async def reward_async(self, exps: List[Experience]) -> float:
        """Default reward function for CoD multi-step workflows."""
        if exps:
            exps[-1].info["feedback"] = self._get_feedback()
            if exps[-1].metrics is None:
                exps[-1].metrics = {}
            exps[-1].metrics["format_error_termination"] = 1.0 if self.early_termination_by_format_issue else 0.0
            exps[-1].info["early_termination_by_format_issue"] = self.early_termination_by_format_issue

        # Original outcome reward
        returned_reward = self.final_reward

        # Length penalties
        length_penalty_coef = self.task.workflow_args.get("length_penalty_coef", 0.0)
        if exps and (length_penalty_coef > 0.0) and (self.final_reward > 0.0):
            len_full_penalty = self.task.workflow_args.get("len_full_penalty", -1)
            len_zero_penalty = self.task.workflow_args.get("len_zero_penalty", -1)
            assert len_full_penalty > 0, "len_full_penalty should be set to a positive integer."
            assert len_zero_penalty > 0, "len_zero_penalty should be set to a positive integer."
            assert len_full_penalty > len_zero_penalty, "len_full_penalty should be larger than len_zero_penalty."

            # response-wise penalty
            resp_len_penalties = []
            for exp in exps:
                resp_len = len(exp.tokens) - exp.prompt_length
                penalty = min(1.0, max(0.0, (resp_len - len_zero_penalty) / (len_full_penalty - len_zero_penalty)))
                resp_len_penalties.append(penalty)
            response_wise_penalty = sum(resp_len_penalties) / len(resp_len_penalties)  # range [0, 1]
        
            # episode-wise penalty
            episode_wise_penalty = len(exps) / self.max_step_num  # range [0, 1]

            # combine and substract from original reward
            total_penalty = length_penalty_coef * (response_wise_penalty + episode_wise_penalty)
            returned_reward = max(0.0, returned_reward - total_penalty)

        return returned_reward

    @property
    def max_step_num(self) -> int:
        """Return the maximum number of steps in the task."""
        raise NotImplementedError
