# -*- coding: utf-8 -*-
"""
This file defines a multi-step workflow for the FrozenLake environment.
Modified from https://github.com/rllm-org/rllm/blob/main/rllm/environments/frozenlake/frozenlake.py
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.frozen_lake.utils import (
    GRID_LOOKUP,
    MAP_LOOKUP,
    SYSTEM_PROMPT,
    generate_random_map,
    get_goal_position,
)
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task


class FrozenLakeWorkflow(MultiTurnWorkflow):
    """
    FrozenLake environment for multi-step workflows.

    ## Description
    The game starts with the player at random location of the frozen lake grid world with the
    goal located at another random location for the 4x4 environment.

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.
    NOTE the action space is different from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
    use action_map to map from custom action to action defined in FrozenLakeEnv in gymnasium
    - 0: Still
    - 1: Left
    - 2: Down
    - 3: Right
    - 4: Up

    ## Starting State
    The episode starts with the player at random location

    ## Rewards
    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

    ## Arguments
    `is_slippery`: if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3

    ## Example
    P   _   _   _
    _   _   _   O
    O   _   O   _
    O   _   _   G
    """

    can_reset: bool = False  # GymFrozenLakeEnv can only reset the player position, not the environment configuration.
    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        """Initialize the FrozenLake workflow.

        Args:
            model: The model wrapper to use for generating actions.
            task: The task configuration containing workflow-specific arguments.
            auxiliary_models: Optional list of auxiliary models.
        """
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
        )

        # Import gymnasium here to avoid import error if gymnasium is not installed
        # and this workflow is not used
        try:
            import gymnasium as gym
            from gymnasium.envs.toy_text.frozen_lake import (
                FrozenLakeEnv as GymFrozenLakeEnv,
            )
        except ImportError as e:
            error_message = (
                f"Gymnasium is not installed. Please install gymnasium first before "
                f"running the frozen_lake workflow. Error: {str(e)}"
            )
            self.logger.error(error_message)
            raise ImportError(error_message)

        # Extract workflow-specific arguments
        workflow_args = task.workflow_args if hasattr(task, "workflow_args") else {}
        self.env_max_steps = workflow_args.get("env_max_steps", 8)
        self.agent_max_steps = workflow_args.get("agent_max_steps", 10)
        self.desc = workflow_args.get("desc", None)
        self.is_slippery = workflow_args.get("is_slippery", False)
        self.max_response_tokens = self.rollout_args.get("max_response_tokens", 10240)

        # Extract task-specific arguments
        self.raw_task = task.raw_task if hasattr(task, "raw_task") else {}
        self.size = self.raw_task.get("size", 1)
        self.p = self.raw_task.get("p", 0.8)
        self.seed = self.raw_task.get("seed", 42)

        if self.desc is None:
            random_map, goal_position = generate_random_map(
                size=self.size, p=self.p, seed=self.seed, max_steps=self.env_max_steps
            )
        else:
            random_map = np.asarray(copy.deepcopy(self.desc), dtype="c")
            goal_position = get_goal_position(random_map)

        self.goal_position = goal_position

        # Create the gym environment
        self.gym_env = GymFrozenLakeEnv(desc=random_map[:], is_slippery=self.is_slippery)
        self.action_space = gym.spaces.Discrete(4, start=1)

        # Define action map and invalid action
        self.action_map = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        }  # map from custom Env action to action defined in FrozenLakeEnv in gymnasium
        self.invalid_action = 0

        # Agent-related state
        self.step_count: int = 0
        self.step_rewards: List[float] = []
        self.current_observation: Optional[str] = None
        self.done: bool = False
        self.last_observation: Optional[str] = None

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def _get_player_position(self) -> Tuple[int, int]:
        """Get the current player position.

        Returns:
            Tuple of (row, col) representing the player position.
        """
        return (
            self.gym_env.s // self.gym_env.ncol,
            self.gym_env.s % self.gym_env.ncol,
        )  # (row, col)

    def finished(self) -> bool:
        """Check if the episode is finished.

        Returns:
            True if the player is on goal (G) or hole (H), False otherwise.
        """
        player_pos = self._get_player_position()
        return self.gym_env.desc[player_pos] in b"GH"

    def success(self) -> bool:
        """Check if the agent has reached the goal (G).

        Returns:
            True if the player is on goal (G), False otherwise.
        """
        player_pos = self._get_player_position()
        return self.gym_env.desc[player_pos] in b"G"

    def env_step(self, action: int):
        """Execute a step in the environment.

        Maps custom action to gymnasium FrozenLakeEnv action and takes the step.
        Checks if the action is effective (whether player moves in the env).

        Args:
            action: The action to take.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self.success():
            return self.render(), 1, True, {"action_is_effective": False}

        if not action:
            action = self.invalid_action

        action = int(action)
        if action == self.invalid_action or action not in self.action_map:
            return self.render(), 0, False, {"action_is_effective": False}

        prev_player_position = int(self.gym_env.s)

        player_pos, reward, done, _, prob = self.gym_env.step(self.action_map[action])

        obs = self.render()
        return obs, reward, done, {"action_is_effective": prev_player_position != int(player_pos)}

    def render(self, mode="tiny_rgb_array"):
        """Render the environment.

        Args:
            mode: Rendering mode. Options: "tiny_rgb_array", "list", "state", "rgb_array", "ansi".

        Returns:
            Rendered observation based on the mode.
        """
        assert mode in ["tiny_rgb_array", "list", "state", "rgb_array", "ansi"]
        if mode in ["rgb_array", "ansi"]:
            prev_render_mode = self.gym_env.render_mode
            self.gym_env.render_mode = mode
            obs = self.gym_env.render()
            self.gym_env.render_mode = prev_render_mode
            return obs
        room_state = copy.deepcopy(self.gym_env.desc)

        # replace the position of start 'S' with 'F'
        position_S = np.where(room_state == b"S")
        room_state[position_S] = b"F"

        # replace the position of the player with 'P'
        position_P = self._get_player_position()
        room_state[position_P] = b"P"

        if mode == "state":
            # transform 'S', 'F', 'H', 'G' to numpy integer array
            room_state = np.vectorize(lambda x: MAP_LOOKUP[x])(room_state)
            # add player in hole or player on goal
            if self.gym_env.desc[position_P] == b"H":
                room_state[position_P] = 4
            elif self.gym_env.desc[position_P] == b"G":
                room_state[position_P] = 5
            return room_state

        room_state = self.render(mode="state").tolist()

        if mode == "list":

            def lookup(cell):
                return GRID_LOOKUP.get(cell, "?").strip("\t").strip()

            return [" ".join(lookup(cell) for cell in row) for row in room_state]

        if mode == "tiny_rgb_array":

            def lookup(cell):
                return GRID_LOOKUP.get(cell, "?")

            result = "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
            return result

    async def run_async(self) -> List[Experience]:
        """Run the workflow and return a list of experiences.

        Returns:
            List of Experience objects, one for each rollout.
        """
        # Reset environment and state for a new episode
        # But this only resets the player position, not the environment configuration.
        self.gym_env.reset(seed=self.seed)
        observation = self.render()
        self.current_observation = str(observation)
        self.done = False
        self.step_rewards = []
        self.step_count = 0
        self.action = None
        terminate_reason = None
        truncate_status = None

        # Initialize messages
        messages = []
        system_prompt = SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_prompt})

        # Run episode until done or max_steps reached
        for step in range(self.agent_max_steps):
            # Format observation for the model
            user_prompt_content = (
                f"Current Observation ({self.step_count}): \n"
                + self.current_observation
                + "\n"
                + "You have not achieved the goal, P has not reached G yet. Please give the next action."
            )

            if self.step_count > 0 and self.action is not None:
                if self.last_observation == self.current_observation:
                    user_prompt_content += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response. Remember, you should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```."

            if self.agent_max_steps is not None and self.agent_max_steps - self.step_count > 0:
                user_prompt_content += f"\nThe maximum number of steps remaining is {self.agent_max_steps - self.step_count}."

            messages.append({"role": "user", "content": user_prompt_content})

            messages_token_len = await self.model.get_message_token_len(messages)
            if step == 0:
                max_tokens = self.max_response_tokens
                init_prompt_token_len = messages_token_len
            else:
                response_token_len = messages_token_len - init_prompt_token_len
                max_tokens = self.max_response_tokens - response_token_len

            if max_tokens <= 0:
                messages = messages[:-1]  # Remove the last user message
                self.done = False
                self.step_rewards.append(0)
                terminate_reason = "max_tokens_reached"
                truncate_status = "response_truncated"
                break

            # Get action from the model
            rollout_args = self.rollout_args.copy()
            rollout_args["n"] = 1
            rollout_args["max_tokens"] = max_tokens
            responses = await self.model.chat_async(messages, **rollout_args)
            response_text = responses[0].response_text
            messages.append({"role": "assistant", "content": response_text})

            # Parse action from response
            _, action_str = self._parse_model_response(response_text)
            action = int(action_str) if action_str.isdigit() else self.invalid_action
            self.action = action

            # Execute action in the environment
            observation, reward, done, info = self.env_step(action)

            # Update internal state
            self.last_observation = self.current_observation
            self.current_observation = str(observation)
            self.done = done
            self.step_rewards.append(reward)
            self.step_count += 1

            if self.done:
                terminate_reason = "success"
                break
        if terminate_reason is None:
            terminate_reason = "max_steps_reached"

        # Create experience from messages
        final_reward = sum(self.step_rewards)
        # print(f"final_reward: {final_reward}, terminate_reason: {terminate_reason}")
        experience = await self.process_messages_to_experience_async(
            messages=messages,
            reward=float(final_reward),
            info={
                "env_steps": self.step_count,
                "env_done": 1 if self.done else 0,
                "test_score": final_reward,
            },
            truncate_status=truncate_status,
        )
        return [experience]

    def _parse_model_response(self, response: str) -> tuple[str, str]:
        """Parse the model response to extract thought and action.

        Args:
            response: The model's response text.

        Returns:
            Tuple of (thought, action_str).
        """
        DIRECTION_MAP = {"left": 1, "down": 2, "right": 3, "up": 4}

        thought = response
        action_str = str(self.invalid_action)

        matches = re.findall(r"```(.*?)```", response, re.DOTALL)

        if matches:
            last_match_content = matches[-1].strip()
            last_match_index = response.rfind(f"```{last_match_content}```")
            if last_match_index != -1:
                thought = response[:last_match_index].strip()

            extracted_text = last_match_content.lower()

            if extracted_text in DIRECTION_MAP:
                action_str = str(DIRECTION_MAP[extracted_text])
            elif extracted_text.isdigit() and int(extracted_text) in DIRECTION_MAP.values():
                action_str = str(int(extracted_text))

        return thought, action_str
