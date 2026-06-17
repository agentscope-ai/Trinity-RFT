# -*- coding: utf-8 -*-
"""
CoD FrozenLake Obscure Workflow - uses numeric actions (1,2,3,4) instead of directions.

Two mapping modes:
- "global": All episodes use the same fixed mapping
- "per_task": Each task uses its own mapping (via raw_task)
"""

from __future__ import annotations

import itertools
import random
import re
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from trinity.common.experience import Experience
from trinity.common.workflows.connect_the_dots.frozen_lake.workflow import (
    CoDFrozenLakeWorkflow,
)
from trinity.common.workflows.connect_the_dots.utils import extract_content_between_keys

if TYPE_CHECKING:
    from trinity.common.models.model import ModelWrapper
    from trinity.common.workflows.workflow import Task


# All 24 permutations of directions
ALL_PERMUTATIONS = list(itertools.permutations(["Down", "Right", "Up", "Left"]))

# Default mapping: 1=Down, 2=Right, 3=Up, 4=Left
DEFAULT_ACTION_MAPPING = {1: "Down", 2: "Right", 3: "Up", 4: "Left"}


def get_mapping_by_index(index: int) -> Dict[int, str]:
    """Get action mapping by permutation index (0-23)."""
    perm = ALL_PERMUTATIONS[index % len(ALL_PERMUTATIONS)]
    return {i + 1: perm[i] for i in range(4)}


def get_random_mapping(seed: Optional[int] = None) -> Dict[int, str]:
    """Get a random action mapping."""
    rng = random.Random(seed)
    directions = ["Down", "Right", "Up", "Left"]
    rng.shuffle(directions)
    return {i + 1: directions[i] for i in range(4)}


def parse_numeric_action_number(response: str) -> Optional[int]:
    """Parse 'Direction X' action from response, return the integer or None."""
    key_start, key_end = "<answer>", "</answer>"
    content, success = extract_content_between_keys(response, key_start, key_end)
    if not success:
        return None
    action_str = content.strip()
    dir_match = re.match(r"[Dd]irection\s+(\d+)", action_str)
    if dir_match:
        return int(dir_match.group(1))
    return None


class CoDFrozenLakeObscureWorkflow(CoDFrozenLakeWorkflow):
    """
    FrozenLake with numeric actions (1,2,3,4) instead of Up/Down/Left/Right.

    Configuration (workflow_args):
        - mapping_mode: "global", "per_pack", or "per_task"
            - "global": All tasks use the same fixed mapping
            - "per_pack": All tasks in one CoD pack share one mapping,
                          different packs get different mappings
            - "per_task": Each task uses its own mapping
        - global_mapping: {1: "Down", ...} for global mode
        - mapping_index: 0-23 for predefined permutations (global mode)

    Per-task (raw_task, for per_task mode):
        - action_mapping: explicit mapping
        - mapping_index: permutation index
        - mapping_seed: seed for random mapping
    """

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
        use_openai_client: bool = False,
    ):
        # Get mapping config before parent init
        workflow_args = task.workflow_args if hasattr(task, "workflow_args") else {}
        self.mapping_mode = workflow_args.get("mapping_mode", "global")
        self.global_mapping = workflow_args.get("global_mapping", None)
        self.global_mapping_index = workflow_args.get("mapping_index", None)

        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
            use_openai_client=use_openai_client,
        )

        # Determine action mapping
        self.action_mapping = self._determine_action_mapping()

    def _determine_action_mapping(self) -> Dict[int, str]:
        """Determine action mapping based on mode.

        Modes:
            - "global": All tasks use the same mapping (from workflow_args)
            - "per_pack": All tasks in the same CoD pack share one mapping,
                          different packs use different mappings
            - "per_task": Each task uses its own mapping (from raw_task)
        """
        if self.mapping_mode == "per_pack":
            # Use pack-level seed injected by CoDWorkflow.reset()
            if "pack_seed" in self.raw_task:
                return get_random_mapping(self.raw_task["pack_seed"])
            else:
                # Fallback: not running under CoDWorkflow, use task seed
                return get_random_mapping(self.raw_task.get("seed", 42))
        elif self.mapping_mode == "per_task":
            if "action_mapping" in self.raw_task:
                return self.raw_task["action_mapping"]
            elif "mapping_index" in self.raw_task:
                return get_mapping_by_index(self.raw_task["mapping_index"])
            elif "mapping_seed" in self.raw_task:
                return get_random_mapping(self.raw_task["mapping_seed"])
            else:
                return get_random_mapping(self.raw_task.get("seed", 42))
        else:
            # "global" mode
            if self.global_mapping is not None:
                return self.global_mapping
            elif self.global_mapping_index is not None:
                return get_mapping_by_index(self.global_mapping_index)
            else:
                return DEFAULT_ACTION_MAPPING

    def _build_system_prompt(self) -> str:
        """Build system prompt with numeric actions using jinja template."""
        from trinity.common.workflows.connect_the_dots.frozen_lake.prompts import load_system_prompt_obscure

        sys_prompt = load_system_prompt_obscure()
        sys_prompt = self._augment_system_prompt(sys_prompt)
        return sys_prompt

    async def step_async(self, step_num: int) -> Tuple[bool, List[Experience]]:
        """Execute one step - override to use numeric action parsing."""
        if self.done:
            return False, []

        from trinity.common.workflows.connect_the_dots.frozen_lake.prompts import load_user_prompt

        user_content = load_user_prompt(
            current_step=step_num + 1,
            max_steps=self.agent_max_steps,
            observation=self.observation,
            goal_row=self.goal_position[0],
            goal_col=self.goal_position[1],
            is_success=self._is_success(),
            action_feedback=self.action_feedback,
        )

        if self.icl_examples and step_num == 0:
            user_content = f"{user_content}\n\nHere are some reference examples:\n\n{self.icl_examples}"

        self.memory.append({"role": "user", "content": user_content})

        if self.reply_prefix:
            self.memory.append({"role": "assistant", "content": self.reply_prefix})

        experiences = await self.model.chat_async(self.memory)
        response_text = experiences[0].response_text
        self.memory.append({"role": "assistant", "content": response_text})

        sys_prompt = self.memory[0]["content"] if self.memory and self.memory[0]["role"] == "system" else ""
        for exp in experiences:
            exp.info["sys_prompt"] = sys_prompt
            exp.info["user_prompt"] = user_content
            exp.info["action_mapping"] = self.action_mapping

        # Check for malformed response: missing <answer> tag
        numeric_action = parse_numeric_action_number(response_text)
        if numeric_action is None:
            # Early termination: reward 0, give feedback about format
            self.action_feedback = "Invalid format: could not parse action. Expected format: {your reasoning process here}<answer>Direction X</answer>, where X is 1, 2, 3, or 4. Game over."
            self.done = True
            self.current_step = step_num + 1
            self.early_termination_by_format_issue = True
            return False, experiences

        direction = self.action_mapping.get(numeric_action)
        if direction is None:
            # Valid format but invalid action number (e.g. Direction 5)
            self.action_feedback = f"Invalid format: Direction {numeric_action} is not a valid action. Expected: Direction 1, 2, 3, or 4. Game over."
            self.done = True
            self.current_step = step_num + 1
            self.early_termination_by_format_issue = True
            return False, experiences

        # Track position before action
        prev_pos = self._get_player_position()

        observation, reward, done, info = self.env_step(direction)

        # Build action feedback with "Direction X" format
        cur_pos = self._get_player_position()
        self.action_feedback = self._build_action_feedback(
            action_str=f"Direction {numeric_action}",
            prev_pos=prev_pos, cur_pos=cur_pos,
            action_effective=info.get("action_is_effective", False),
        )

        self.observation = observation
        self.done = done
        self.current_step = step_num + 1

        if done and reward > 0:
            self.final_reward = reward

        return not self.done, experiences
