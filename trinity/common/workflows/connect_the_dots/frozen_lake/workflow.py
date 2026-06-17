# -*- coding: utf-8 -*-
"""
CoD (Connect-the-Dots) workflow for FrozenLake environment using generic multi-turn.

Each step produces an independent Experience with step number. Prior assistant
responses are compressed to <answer> tags only by the base class to avoid
context bloat in multi-turn conversations.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from trinity.common.experience import Experience
from trinity.common.workflows.connect_the_dots.base_workflow import AsyncCoDMultiStepWorkflow
from trinity.common.workflows.connect_the_dots.utils import extract_content_between_keys
from trinity.common.workflows.workflow import Task
from trinity.common.workflows.envs.frozen_lake.utils import (
    GRID_LOOKUP,
    MAP_LOOKUP,
    generate_random_map,
    get_goal_position,
)
from trinity.common.workflows.connect_the_dots.frozen_lake.prompts import (
    load_system_prompt,
    load_user_prompt,
)

if TYPE_CHECKING:
    from trinity.common.models.model import ModelWrapper


def parse_action(response: str) -> Optional[str]:
    """Parse action from model response.

    Args:
        response: Model response text.

    Returns:
        Action string if found, None otherwise.
    """
    VALID_ACTIONS = {"up", "down", "left", "right"}

    key_start, key_end = "<answer>", "</answer>"
    content, success = extract_content_between_keys(response, key_start, key_end)
    if not success:
        return None
    
    action = content.strip().lower()
    if action in VALID_ACTIONS:
        return action.capitalize()
    return None


class CoDFrozenLakeWorkflow(AsyncCoDMultiStepWorkflow):
    """
    CoD FrozenLake workflow using generic multi-turn mechanism.

    This workflow:
    - Produces independent Experience per step (with exp.eid.step)
    - Prior assistant responses are compressed to <answer> only (via base class)
    """

    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
        use_openai_client: bool = False,
    ):
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
            use_openai_client=use_openai_client,
        )

        # Import gymnasium here to avoid import error if not installed
        try:
            import gymnasium as gym
            from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
        except ImportError as e:
            raise ImportError(
                f"Gymnasium is not installed. Please install gymnasium first. Error: {e}"
            )

        # Extract workflow-specific arguments
        workflow_args = task.workflow_args if hasattr(task, "workflow_args") else {}
        self.env_max_steps = workflow_args.get("env_max_steps", 8)
        self.agent_max_steps = workflow_args.get("agent_max_steps", 10)
        self.desc = workflow_args.get("desc", None)
        self.is_slippery = workflow_args.get("is_slippery", False)
        self.max_response_tokens_restraint = workflow_args.get("max_response_tokens_restraint", None)

        # Extract task-specific arguments
        self.raw_task = task.raw_task if hasattr(task, "raw_task") else {}
        self.size = self.raw_task.get("size", 4)
        self.p = self.raw_task.get("p", 0.8)
        self.seed = self.raw_task.get("seed", 42)

        # Generate or use provided map
        if self.desc is None:
            random_map, goal_position = generate_random_map(
                size=self.size, p=self.p, seed=self.seed, max_steps=self.env_max_steps
            )
        else:
            random_map = np.asarray(copy.deepcopy(self.desc), dtype="c")
            goal_position = get_goal_position(random_map)

        self.goal_position = goal_position

        # Create gym environment
        self.gym_env = GymFrozenLakeEnv(desc=random_map[:], is_slippery=self.is_slippery)
        self.action_space = gym.spaces.Discrete(4, start=1)

        # Action mapping: our action -> gym action
        self.action_map = {
            "Left": 0,
            "Down": 1,
            "Right": 2,
            "Up": 3,
        }

        # State variables
        self.observation: Optional[str] = None
        self.done: bool = False
        self.final_reward: float = 0.0
        self.early_termination_by_format_issue = False
        self.memory: List[dict] = []
        self.current_step: int = 0

    def _get_player_position(self):
        """Get current player position as (row, col)."""
        return (
            self.gym_env.s // self.gym_env.ncol,
            self.gym_env.s % self.gym_env.ncol,
        )

    def _is_success(self) -> bool:
        """Check if player reached the goal."""
        player_pos = self._get_player_position()
        return self.gym_env.desc[player_pos] == b"G"

    def render(self) -> str:
        """Render current state as text."""
        room_state = copy.deepcopy(self.gym_env.desc)

        # Replace start 'S' with frozen 'F'
        position_S = np.where(room_state == b"S")
        room_state[position_S] = b"F"

        # Mark player position
        position_P = self._get_player_position()
        room_state[position_P] = b"P"

        # Convert to state array
        state_array = np.vectorize(lambda x: MAP_LOOKUP[x])(room_state)

        # Handle player on hole or goal
        if self.gym_env.desc[position_P] == b"H":
            state_array[position_P] = 4  # player fell into hole
        elif self.gym_env.desc[position_P] == b"G":
            state_array[position_P] = 5  # player on goal

        # Render as text
        result = "\n".join(
            "".join(GRID_LOOKUP.get(cell, "?") for cell in row)
            for row in state_array.tolist()
        )
        return result

    def _build_action_feedback(
        self, action_str: Optional[str], prev_pos: Tuple[int, int],
        cur_pos: Tuple[int, int], action_effective: bool,
    ) -> str:
        """Build position change feedback string after an action.

        Args:
            action_str: The action taken (e.g. "Up" or None if invalid).
            prev_pos: (row, col) before action.
            cur_pos: (row, col) after action.
            action_effective: Whether the position changed.

        Returns:
            Feedback string describing position change.
        """
        if action_str is None:
            return "Your action was invalid. Position unchanged."
        if action_effective:
            return (
                f"You executed action {action_str}. "
                f"Your position changed from (row={prev_pos[0]}, col={prev_pos[1]}) "
                f"to (row={cur_pos[0]}, col={cur_pos[1]})."
            )
        else:
            return (
                f"You executed action {action_str}. "
                f"Your position did not change (stayed at row={cur_pos[0]}, col={cur_pos[1]}). "
                f"This may be because you tried to move beyond the edge of the map."
            )

    def env_step(self, action: Optional[str]):
        """Execute action in environment.

        Args:
            action: Action string (Up/Down/Left/Right) or None for invalid.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._is_success():
            return self.render(), 1.0, True, {"action_is_effective": False}

        if action is None or action not in self.action_map:
            return self.render(), 0.0, False, {"action_is_effective": False}

        prev_pos = int(self.gym_env.s)

        # Execute action
        _, reward, done, _, _ = self.gym_env.step(self.action_map[action])

        obs = self.render()
        action_effective = prev_pos != int(self.gym_env.s)

        return obs, reward, done, {"action_is_effective": action_effective}

    def _build_system_prompt(self) -> str:
        """Build system prompt with hint and token limit."""
        sys_prompt = load_system_prompt()
        sys_prompt = self._augment_system_prompt(sys_prompt)
        return sys_prompt

    async def run_async(self) -> List[Experience]:
        """Run the workflow."""
        # Reset environment
        self.gym_env.reset(seed=self.seed)
        self.observation = self.render()
        self.done = False
        self.final_reward = 0.0
        self.current_step = 0
        self.action_feedback = None  # Position change feedback for next step

        # Initialize memory with system prompt (includes hint and token limit)
        self.memory.clear()
        self.memory.append({"role": "system", "content": self._build_system_prompt()})

        return await super().run_async()

    async def step_async(self, step_num: int) -> Tuple[bool, List[Experience]]:
        """Execute one step of the workflow.

        Args:
            step_num: Current step number.

        Returns:
            Tuple of (continue_run, experiences):
                - continue_run: True to continue, False to stop.
                - experiences: List of experiences from this step.
        """
        if self.done:
            return False, []

        # Format observation as user message using jinja template
        user_content = load_user_prompt(
            current_step=step_num + 1,
            max_steps=self.agent_max_steps,
            observation=self.observation,
            goal_row=self.goal_position[0],
            goal_col=self.goal_position[1],
            is_success=self._is_success(),
            action_feedback=self.action_feedback,
        )

        # ========== CoD-specific: add icl_examples and reply_prefix ==========
        # icl_examples: only added in the first turn to avoid token waste
        if self.icl_examples and step_num == 0:
            user_content = f"{user_content}\n\nHere are some reference examples:\n\n{self.icl_examples}"

        self.memory.append({"role": "user", "content": user_content})

        # reply_prefix: prefix for model's reply, used for guided generation
        if self.reply_prefix:
            self.memory.append({"role": "assistant", "content": self.reply_prefix})
        # ====================================================================

        # Get model response - chat_async returns List[Experience] directly
        experiences = await self.model.chat_async(self.memory)
        response_text = experiences[0].response_text
        self.memory.append({"role": "assistant", "content": response_text})

        # Store prompt information in exp.info for logging
        sys_prompt = self.memory[0]["content"] if self.memory and self.memory[0]["role"] == "system" else ""
        for exp in experiences:
            exp.info["sys_prompt"] = sys_prompt
            exp.info["user_prompt"] = user_content

        # Parse action
        action = parse_action(response_text)

        # Track position before action
        prev_pos = self._get_player_position()

        # Execute action
        observation, reward, done, info = self.env_step(action)

        # Build action feedback for next step
        cur_pos = self._get_player_position()
        self.action_feedback = self._build_action_feedback(
            action_str=action, prev_pos=prev_pos, cur_pos=cur_pos,
            action_effective=info.get("action_is_effective", False),
        )

        # Update state
        self.observation = observation
        self.done = done
        self.current_step = step_num + 1

        if done and reward > 0:
            self.final_reward = reward

        return not self.done, experiences

    def _get_feedback(self) -> str:
        """Generate structured feedback about task completion.

        Returns feedback relevant to reward calculation:
        - Success: reached the goal
        - Failure: fell into hole or ran out of steps
        """
        if self._is_success():
            return "Success! Reached the goal (G)."

        player_pos = self._get_player_position()
        if self.gym_env.desc[player_pos] == b"H":
            return "Failed: Fell into a hole (O). Avoid holes by planning a safe path."

        if self.early_termination_by_format_issue:
            return self.action_feedback

        # Ran out of steps
        return f"Failed: Ran out of steps ({self.current_step}/{self.agent_max_steps}) without reaching the goal. Try to find a shorter path."

    @property
    def max_step_num(self) -> int:
        """Maximum number of steps."""
        return self.agent_max_steps

    def __del__(self):
        """Cleanup environment."""
        if hasattr(self, "gym_env"):
            self.gym_env.close()
