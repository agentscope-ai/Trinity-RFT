"""CoD multi-step workflow for the persistent-cost grid-navigation task."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from trinity.common.experience import Experience
from trinity.common.workflows.connect_the_dots.base_workflow import AsyncCoDMultiStepWorkflow
from trinity.common.workflows.connect_the_dots.grid_navigation.env import (
    GridNavigationEnv,
    generate_task,
    get_pack_state,
)
from trinity.common.workflows.connect_the_dots.grid_navigation.prompts import (
    load_system_prompt,
    load_user_prompt,
)
from trinity.common.workflows.connect_the_dots.utils import extract_content_between_keys
from trinity.common.workflows.workflow import Task

if TYPE_CHECKING:
    from trinity.common.models.model import ModelWrapper


def parse_action(response: str) -> Optional[Tuple[int, int]]:
    """Parse one row,column destination from the answer tags."""
    content, success = extract_content_between_keys(response, "<answer>", "</answer>")
    if not success:
        return None
    match = re.fullmatch(r"\s*(-?\d+)\s*,\s*(-?\d+)\s*", content)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


class CoDGridNavigationWorkflow(AsyncCoDMultiStepWorkflow):
    """Navigate a hidden persistent cost grid for an exact number of rounds."""

    is_async: bool = True
    can_reset: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
        use_openai_client: bool = False,
    ):
        """Initialize the workflow and bind it to its pack state."""
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
            use_openai_client=use_openai_client,
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset task configuration and acquire the shared pack environment."""
        super().reset(task)
        args = task.workflow_args
        self.grid_min_size = int(args.get("grid_min_size", 12))
        self.grid_max_size = int(args.get("grid_max_size", 16))
        self.landscape_num_components = int(args.get("landscape_num_components", 6))
        self.landscape_min_scale = float(args.get("landscape_min_scale", 0.10))
        self.landscape_max_scale = float(args.get("landscape_max_scale", 0.35))
        self.min_rounds = int(args.get("min_rounds", 6))
        self.max_rounds = int(args.get("max_rounds", 10))
        self.reveal_radius = int(args.get("reveal_radius", 1))

        self.seed = int(self.raw_task.get("seed", 42))
        self.pack_seed = int(self.raw_task.get("pack_seed", self.seed))
        self.task_idx = int(self.raw_task.get("task_idx", 0))

        pack_state = get_pack_state(
            pack_seed=self.pack_seed,
            task_idx=self.task_idx,
            grid_min_size=self.grid_min_size,
            grid_max_size=self.grid_max_size,
            landscape_num_components=self.landscape_num_components,
            landscape_min_scale=self.landscape_min_scale,
            landscape_max_scale=self.landscape_max_scale,
        )
        navigation_task = generate_task(
            task_seed=self.seed,
            grid_size=pack_state.size,
            min_rounds=self.min_rounds,
            max_rounds=self.max_rounds,
        )
        self.env = GridNavigationEnv(
            pack_state=pack_state,
            task=navigation_task,
            reveal_radius=self.reveal_radius,
        )

        self.done = False
        self.final_reward = 0.0
        self.current_step = 0
        self.action_feedback: Optional[str] = None
        self.early_termination_by_format_issue = False
        self.memory: List[dict] = []

    def _build_system_prompt(self) -> str:
        return self._augment_system_prompt(load_system_prompt(reveal_radius=self.reveal_radius))

    async def run_async(self) -> List[Experience]:
        """Reset task-local episode state and run all navigation rounds."""
        self.env.reset_task()
        self.done = False
        self.final_reward = 0.0
        self.current_step = 0
        self.action_feedback = None
        self.early_termination_by_format_issue = False

        self.memory.clear()
        self.memory.append({"role": "system", "content": self._build_system_prompt()})
        return await super().run_async()

    def _terminate_invalid_action(self, message: str) -> None:
        self.action_feedback = f"Invalid action: {message} Game over."
        self.done = True
        self.early_termination_by_format_issue = True

    async def step_async(self, step_num: int) -> Tuple[bool, List[Experience]]:
        """Prompt for, parse, and execute one grid move."""
        if self.done:
            return False, []

        user_content = load_user_prompt(
            current_round=step_num + 1,
            max_rounds=self.env.task.num_rounds,
            current_position=self.env.current_position,
            goal_position=self.env.task.goal,
            observation=self.env.render(),
            action_feedback=self.action_feedback,
        )
        if self.icl_examples and step_num == 0:
            user_content = (
                f"{user_content}\n\nHere are some reference examples:\n\n{self.icl_examples}"
            )

        self.memory.append({"role": "user", "content": user_content})
        if self.reply_prefix:
            self.memory.append({"role": "assistant", "content": self.reply_prefix})

        experiences = await self.model.chat_async(self.memory)
        response_text = experiences[0].response_text
        self.memory.append({"role": "assistant", "content": response_text})

        sys_prompt = self.memory[0]["content"]
        for exp in experiences:
            exp.info["sys_prompt"] = sys_prompt
            exp.info["user_prompt"] = user_content

        destination = parse_action(response_text)
        if destination is None:
            self._terminate_invalid_action(
                "expected exactly one <answer>row,col</answer> destination."
            )
            self.current_step = step_num + 1
            return False, experiences

        validation_error = self.env.validate_destination(destination)
        if validation_error is not None:
            self._terminate_invalid_action(validation_error)
            self.current_step = step_num + 1
            return False, experiences

        result = self.env.step(destination)
        path_text = " -> ".join(f"({row},{col})" for row, col in result["path"])
        self.action_feedback = (
            f"You moved to ({destination[0]},{destination[1]}) through {len(result['path'])} "
            f"entered cells: {path_text}. "
            f"Cumulative normalized loss: {result['normalized_loss']:.4f}."
        )
        self.current_step = step_num + 1
        self.done = bool(result["done"])
        if self.done:
            self.final_reward = float(result["reward"])

        for exp in experiences:
            exp.info["destination"] = destination
            exp.info["normalized_loss"] = result["normalized_loss"]

        return not self.done, experiences

    def _build_trajectory(self) -> str:
        trajectory = super()._build_trajectory()
        final_state = (
            "Final environment state:\n"
            f"Position: {self.env.current_position}\n"
            f"Target: {self.env.task.goal}\n"
            f"Normalized loss: {self.env.normalized_loss:.4f}\n"
            f"Observed cost grid:\n{self.env.render()}"
        )
        return f"{trajectory}\n\n{final_state}" if trajectory else final_state

    def _get_feedback(self) -> str:
        if self.early_termination_by_format_issue:
            outcome = self.action_feedback or "Invalid action."
        elif self.env.current_position == self.env.task.goal:
            outcome = (
                f"Success: finished round {self.env.task.num_rounds} at the target. "
                f"Normalized loss: {self.env.normalized_loss:.4f}; "
                f"reward: {self.final_reward:.4f}."
            )
        else:
            outcome = (
                f"Failed: after {self.env.task.num_rounds} rounds, position "
                f"{self.env.current_position} did not equal target {self.env.task.goal}. Reward: 0."
            )
        return f"{outcome}\n\nFinal observed cost grid:\n{self.env.render()}"

    @property
    def max_step_num(self) -> int:
        """Return the task's exact number of rounds."""
        return self.env.task.num_rounds
