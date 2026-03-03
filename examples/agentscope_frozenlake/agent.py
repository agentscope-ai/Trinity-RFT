import re

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from examples.agentscope_frozenlake.utils import SYSTEM_PROMPT, FrozenLakeAction

INVALID_ACTION = "still"
VALID_ACTIONS = {
    "left": 1,
    "down": 2,
    "right": 3,
    "up": 4,
}


class FrozenLakeAgent:
    def __init__(self, model: OpenAIChatModel, max_steps: int = 20):
        self.model = model
        self.agent = ReActAgent(
            name="frozenlake_agent",
            sys_prompt=SYSTEM_PROMPT,
            model=model,
            formatter=OpenAIChatFormatter(),
            max_iters=2,
        )
        self.agent.set_console_output_enabled(False)
        self.response_structure = FrozenLakeAction
        self.current_step = 0
        self.last_action = None
        self.last_observation = None
        self.max_steps = max_steps

    def get_prompt(self, observation: str) -> str:
        prompt = (
            f"Current Observation ({self.current_step}): \n"
            + observation
            + "\n"
            + "You have not achieved the goal, P has not reached G yet. Please give the next action."
        )
        if self.current_step > 0 and self.last_action is not None:
            if self.last_observation == observation:
                prompt += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response. Remember, you should only output the NEXT ACTION at each iteration in the ``` ```. For example, if you want to move up, you should output ```Up```."

        if self.max_steps is not None and self.max_steps - self.current_step > 0:
            prompt += (
                f"\nThe maximum number of steps remaining is {self.max_steps - self.current_step}."
            )

        return prompt

    def get_action(self, msg: Msg) -> str:
        response: str = msg.content if isinstance(msg.content, str) else msg.content[0].get("text")
        action = INVALID_ACTION

        matches = re.findall(r"```(.*?)```", response, re.DOTALL)

        if matches:
            last_match_content = matches[-1].strip()
            action = last_match_content.lower()
            if action not in VALID_ACTIONS:
                action = INVALID_ACTION

        return action

    async def step(self, current_observation: str) -> str:
        prompt = self.get_prompt(current_observation)
        response = await self.agent.reply(Msg("user", prompt, role="user"))
        action = self.get_action(response)
        self.last_observation = current_observation
        self.last_action = action
        self.current_step += 1
        return action
