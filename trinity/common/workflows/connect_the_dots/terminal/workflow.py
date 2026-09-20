# -*- coding: utf-8 -*-
"""
CoD Terminal workflow for file operation tasks.

Extends AsyncCoDMultiStepWorkflow following the same patterns as
CoDFrozenLakeWorkflow and CoDRandomAlchemyWorkflow.
"""

from typing import List, Optional, Tuple

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.connect_the_dots.base_workflow import (
    AsyncCoDMultiStepWorkflow,
)
from trinity.common.workflows.connect_the_dots.utils import extract_content_between_keys
from trinity.common.workflows.connect_the_dots.terminal.env import (
    OSType,
    TerminalEnv,
)
from trinity.common.workflows.connect_the_dots.terminal.goal_check import (
    check_goal,
)
from trinity.common.workflows.connect_the_dots.terminal.prompts import (
    load_system_prompt,
    load_user_prompt,
)
from trinity.common.workflows.connect_the_dots.terminal.task_gen import (
    TerminalTask,
    build_env_from_task,
    generate_task,
)
from trinity.common.workflows.workflow import Task


_OS_DISPLAY = {
    OSType.WINDOWS: "Windows",
    OSType.MAC: "macOS",
    OSType.LINUX: "Linux",
}


def _parse_action(response: str) -> Optional[str]:
    """Extract command from <answer>...</answer> tags."""
    content, success = extract_content_between_keys(response, "<answer>", "</answer>")
    if not success:
        return None
    cmd = content.strip()
    return cmd if cmd else None


class CoDTerminalWorkflow(AsyncCoDMultiStepWorkflow):
    """CoD workflow for simulated terminal file-operation tasks."""

    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
        use_openai_client: bool = False,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
            use_openai_client=use_openai_client,
        )

        workflow_args = task.workflow_args if hasattr(task, "workflow_args") else {}
        self.agent_max_steps: int = workflow_args.get("agent_max_steps", 15)
        self.max_response_tokens_restraint = workflow_args.get(
            "max_response_tokens_restraint", None
        )
        composite_ratio: float = workflow_args.get("composite_ratio", 0.5)

        # Reconstruct task from seed
        raw_task = task.raw_task if hasattr(task, "raw_task") else {}
        self.seed: int = raw_task.get("seed", 42)
        forced_task_type = raw_task.get("forced_task_type", None)

        self.terminal_task: TerminalTask = generate_task(
            seed=self.seed,
            composite_ratio=composite_ratio,
            forced_task_type=forced_task_type,
        )
        self.env: Optional[TerminalEnv] = None

        # State
        self.last_output: str = ""
        self.format_error: bool = False
        self.early_completion: bool = False

    async def run_async(self) -> List[Experience]:
        self.env = build_env_from_task(self.terminal_task, max_steps=self.agent_max_steps)
        self.last_output = ""
        self.format_error = False
        self.early_completion = False

        os_type = self.terminal_task.local_os
        sys_prompt = load_system_prompt(
            os_type=os_type.value,
            os_name=_OS_DISPLAY[os_type],
            remote_user=self.terminal_task.remote_user,
            remote_host=self.terminal_task.remote_host,
            task_description=self.terminal_task.description,
            max_steps=self.agent_max_steps,
        )
        sys_prompt = self._augment_system_prompt(sys_prompt)

        self.memory.clear()
        self.memory.append({"role": "system", "content": sys_prompt})

        return await super().run_async()

    async def step_async(self, step_num: int) -> Tuple[bool, List[Experience]]:
        # Build user prompt
        terminal_prompt = self.env.get_prompt_string()
        user_content = load_user_prompt(
            current_step=step_num + 1,
            max_steps=self.agent_max_steps,
            command_output=self.last_output if step_num > 0 else None,
            terminal_prompt=terminal_prompt,
        )

        if self.icl_examples and step_num == 0:
            user_content = (
                f"{user_content}\n\n"
                f"Here are some reference examples:\n\n{self.icl_examples}"
            )

        self.memory.append({"role": "user", "content": user_content})

        if self.reply_prefix:
            self.memory.append({"role": "assistant", "content": self.reply_prefix})

        # Get model response
        experiences = await self.model.chat_async(self.memory)
        response_text = experiences[0].response_text
        self.memory.append({"role": "assistant", "content": response_text})

        # Store prompt info on experiences
        sys_prompt = (
            self.memory[0]["content"]
            if self.memory and self.memory[0]["role"] == "system"
            else ""
        )
        for exp in experiences:
            exp.info["sys_prompt"] = sys_prompt
            exp.info["user_prompt"] = user_content

        # Parse action
        action = _parse_action(response_text)
        if action is None:
            self.format_error = True
            self.last_output = (
                "ERROR: Could not parse command from <answer>...</answer> tags. "
                "Episode terminated."
            )
            return False, experiences

        # Execute command via gym-style step
        def _goal_fn(env):
            return check_goal(self.terminal_task, env)

        observation, reward, done, info = self.env.step(
            action, goal_check_fn=_goal_fn
        )
        self.last_output = observation

        if info.get("early_completion"):
            self.early_completion = True
            self.last_output = "Task completed successfully."

        return not done, experiences

    async def reward_async(self, exps: List[Experience]) -> float:
        if self.format_error:
            reward = 0.0
        elif self.early_completion:
            reward = 1.0
        else:
            reward = check_goal(self.terminal_task, self.env)

        if exps:
            # Feedback
            if self.format_error:
                feedback = "Failed: Episode terminated due to action format error."
            elif reward > 0.5:
                feedback = "Success! Task completed correctly."
            else:
                feedback = "Failed: Goal state not achieved."

            exps[-1].info["feedback"] = feedback
            if exps[-1].metrics is None:
                exps[-1].metrics = {}
            exps[-1].metrics["format_error_termination"] = (
                1.0 if self.format_error else 0.0
            )
            exps[-1].metrics["early_completion"] = (
                1.0 if self.early_completion else 0.0
            )

        return reward

    @property
    def max_step_num(self) -> int:
        return self.agent_max_steps
