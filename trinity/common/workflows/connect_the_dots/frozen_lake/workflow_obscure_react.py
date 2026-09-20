# -*- coding: utf-8 -*-
"""ReAct-agent version of the CoD FrozenLake-Obscure solve workflow.

An AgentScope ReActAgent emits one numeric action per environment turn; the
workflow steps the environment and feeds the next observation back. Each model
call inside a turn is captured via an isolated-history model clone.
"""

from __future__ import annotations

from typing import List, Tuple

from trinity.common.experience import Experience
from trinity.common.workflows.connect_the_dots.agentscope_utils import (
    build_agentscope_react_agent,
    run_agentscope_agent_step,
)
from trinity.common.workflows.connect_the_dots.frozen_lake.prompts import load_user_prompt
from trinity.common.workflows.connect_the_dots.frozen_lake.workflow_obscure import (
    CoDFrozenLakeObscureWorkflow,
    parse_numeric_action_number,
)


class CoDFrozenLakeObscureReActWorkflow(CoDFrozenLakeObscureWorkflow):
    """FrozenLake-Obscure solved by a ReActAgent that acts once per turn.

    Env, mapping, prompts, feedback and reward are inherited from the parent;
    this class replaces only the generation path.
    """

    is_async: bool = True
    can_reset: bool = True
    requires_isolated_model_history: bool = True

    async def run_async(self) -> List[Experience]:
        # Reset env and per-episode state.
        self.gym_env.reset(seed=self.seed)
        self.observation = self.render()
        self.done = False
        self.final_reward = 0.0
        self.current_step = 0
        self.action_feedback = None
        self.early_termination_by_format_issue = False

        # reply_prefix cannot be injected through the agent loop; fail loudly
        # rather than diverge silently.
        if self.reply_prefix:
            raise NotImplementedError(
                "reply_prefix is not supported by the ReAct solve workflow."
            )

        # Rebuild the agent each run since the hint in the system prompt changes.
        self.model.history.clear()
        sys_prompt = self._build_system_prompt()
        agent = await build_agentscope_react_agent(
            name="cod_frozenlake_obscure",
            model=self.model,
            system_prompt=sys_prompt,
            compress_assistant_fn=self._compress_assistant_response,
            max_iters=self.task.workflow_args.get("react_max_iters", 1),
        )

        history_spans: List[Tuple[int, int, str]] = []
        for step_num in range(self.agent_max_steps):
            if self.done:
                break
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
                user_content = (
                    f"{user_content}\n\nHere are some reference examples:\n\n"
                    f"{self.icl_examples}"
                )

            history_start = len(self.model.history)
            response_text = await run_agentscope_agent_step(agent, user_content)
            history_spans.append(
                (history_start, len(self.model.history), user_content)
            )

            numeric_action = parse_numeric_action_number(response_text)
            if numeric_action is None:
                self.action_feedback = (
                    "Invalid format: could not parse action. Expected format: "
                    "{your reasoning process here}<answer>Direction X</answer>, "
                    "where X is 1, 2, 3, or 4. Game over."
                )
                self.done = True
                self.early_termination_by_format_issue = True
                self.current_step = step_num + 1
                break

            direction = self.action_mapping.get(numeric_action)
            if direction is None:
                self.action_feedback = (
                    f"Invalid format: Direction {numeric_action} is not a valid "
                    "action. Expected: Direction 1, 2, 3, or 4. Game over."
                )
                self.done = True
                self.early_termination_by_format_issue = True
                self.current_step = step_num + 1
                break

            prev_pos = self._get_player_position()
            observation, reward, done, info = self.env_step(direction)
            cur_pos = self._get_player_position()
            self.action_feedback = self._build_action_feedback(
                action_str=f"Direction {numeric_action}",
                prev_pos=prev_pos,
                cur_pos=cur_pos,
                action_effective=info.get("action_is_effective", False),
            )
            self.observation = observation
            self.done = done
            self.current_step = step_num + 1
            if done and reward > 0:
                self.final_reward = reward

        experiences = self.model.extract_experience_from_history()
        exp_step = 0
        for history_start, history_end, user_content in history_spans:
            for exp in experiences[history_start:history_end]:
                exp.eid.step = exp_step
                exp.info["sys_prompt"] = sys_prompt
                exp.info["user_prompt"] = user_content
                exp.info["action_mapping"] = self.action_mapping
                exp_step += 1

        trajectory = self._build_agentscope_trajectory(
            sys_prompt,
            await agent.memory.get_memory(prepend_summary=False),
        )

        reward = await self.reward_async(experiences)
        for exp in experiences:
            exp.reward = reward
            if exp.metrics is None:
                exp.metrics = {}
        if experiences:
            experiences[-1].metrics["actual_env_steps"] = self.current_step
            experiences[-1].info["trajectory"] = trajectory
        return experiences
