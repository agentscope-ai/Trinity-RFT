import re

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from examples.agentscope_gomoku.utils import SYSTEM_PROMPT


INVALID_ACTION = (-1, -1)


class GomokuAgent:
    """AgentScope-based agent that selects the next Gomoku move."""

    def __init__(
        self,
        model: OpenAIChatModel,
        max_steps: int = 20,
    ):
        self.model = model

        self.agent = ReActAgent(
            name="gomoku_agent",
            sys_prompt=SYSTEM_PROMPT,
            model=model,
            formatter=OpenAIChatFormatter(),
            max_iters=2,
        )

        self.current_step = 0
        self.last_action = None
        self.last_observation = None
        self.max_steps = max_steps

    def get_prompt(self, observation: str) -> str:
        """Build the prompt containing the current board state."""

        prompt = (
            f"Current Board (step {self.current_step}):\n"
            f"{observation}\n\n"
            "You are playing as X. "
            "Choose the next empty cell.\n"
            "Output ONLY the row and column inside triple backticks.\n"
            "Example: ```2,3```"
        )

        if self.current_step > 0 and self.last_observation == observation:
            prompt += (
                "\nYour previous move was invalid or did not change the board. "
                "Choose a different empty cell."
            )

        if self.max_steps is not None:
            remaining = self.max_steps - self.current_step
            if remaining > 0:
                prompt += (
                    f"\nYou have {remaining} step(s) remaining."
                )

        return prompt

    def get_action(self, msg: Msg) -> tuple[int, int]:
        """Extract a row, column action from the model response."""

        response = (
            msg.content
            if isinstance(msg.content, str)
            else msg.content[0].get("text")
        )

        matches = re.findall(
            r"```(.*?)```",
            response,
            re.DOTALL,
        )

        if not matches:
            return INVALID_ACTION

        action_text = matches[-1].strip()

        match = re.fullmatch(
            r"\s*(\d+)\s*,\s*(\d+)\s*",
            action_text,
        )

        if not match:
            return INVALID_ACTION

        return int(match.group(1)), int(match.group(2))

    async def step(self, current_observation: str) -> tuple[int, int]:
        """Ask the LLM for the next Gomoku move."""

        prompt = self.get_prompt(current_observation)

        response = await self.agent.reply(
            Msg("user", prompt, role="user")
        )

        action = self.get_action(response)

        self.last_observation = current_observation
        self.last_action = action
        self.current_step += 1

        return action