from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow


class GomokuWorkflow(Workflow):
    """Multi-step AgentScope workflow for Gomoku."""

    can_reset: bool = True
    is_async: bool = True
    can_repeat: bool = False

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
        )

        workflow_args = (
            task.workflow_args
            if hasattr(task, "workflow_args")
            else {}
        )

        self.agent_max_steps = workflow_args.get(
            "agent_max_steps",
            20,
        )

        self.board_size = workflow_args.get(
            "board_size",
            5,
        )

        self.win_length = workflow_args.get(
            "win_length",
            4,
        )

        from agentscope.model import OpenAIChatModel

        from examples.agentscope_gomoku.agent import GomokuAgent
        from examples.agentscope_gomoku.env import GomokuEnv

        self.agentscope_model = OpenAIChatModel(
            api_key="EMPTY",
            model_name=model.model_path,
            generate_kwargs=self.rollout_args,
            stream=False,
        )

        self.agentscope_model.client = (
            self.model.get_openai_async_client()
        )

        self.agent = GomokuAgent(
            model=self.agentscope_model,
            max_steps=self.agent_max_steps,
        )

        self.env = GomokuEnv(
            size=self.board_size,
            win_length=self.win_length,
        )

    @property
    def rollout_args(self):
        return {
            "temperature": self.task.rollout_args.temperature,
            "max_tokens": self.task.rollout_args.max_tokens,
        }

    async def run_async(self) -> List[Experience]:
        """Run one Gomoku episode."""

        observation_str = self.env.reset()

        terminate_reason = None
        rewards = []
        step_count = 0
        done = False

        for _ in range(self.agent_max_steps):
            step_count += 1

            try:
                action = await self.agent.step(
                    current_observation=observation_str,
                )
            except Exception as e:
                self.logger.error(
                    f"Agent failed to produce action due to error: {e}"
                )
                terminate_reason = "agent_error"
                break

            observation, reward, done, info = self.env.step(action)

            observation_str = str(observation)
            rewards.append(reward)

            if done:
                terminate_reason = info.get(
                    "reason",
                    "game_finished",
                )
                break

        if terminate_reason is None:
            terminate_reason = "max_steps_reached"

        final_reward = sum(rewards)

        exps = self.model.extract_experience_from_history()

        for exp in exps:
            exp.reward = final_reward
            exp.info["terminate_reason"] = terminate_reason

        if len(exps) > 0:
            exps[-1].metrics = {
                "env_steps": step_count,
                "env_done": int(done),
                "final_reward": final_reward,
            }

        return exps
