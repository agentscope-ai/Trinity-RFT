# -*- coding: utf-8 -*-
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.step_wise_workflow import RewardPropagationWorkflow
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task

EXAMPLE_PROMPT = """
Observation:
-= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.

Your task is to: put some spraybottle on toilet.

Response:
<think>To solve the task, I need to find and take a spraybottle, then put it on toilet. First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.</think><action>go to cabinet 1</action>

Observation:
On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.

Response:
<think>The spraybottle is not here, I should go to cabinet2 to check one by one</think><action>go to cabinet 2</action>

Observation:
The cabinet 2 is closed.

Response:
<think>I should open the cabinet 2 to see what's in it</think><action>open cabinet 2</action>

Observation:
You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.

Response:
<think>Now I find a spraybottle (2). Next, I need to take it.</think><action>take spraybottle 2 from cabinet 2</action>

Observation:
You pick up the spraybottle 2 from the cabinet 2.

Response:
<think>Now I take a spraybottle (2). Next, I need to put it in/on toilet 1.</think><action>go to toilet 1</action>

Observation:
On the toilet 1, you see a soapbottle 2.

Response:
<think> Now I should put the taken spraybottle2 on toilet 1 to finish my task. </think><action>move spraybottle 2 to toilet 1</action>
"""

AlfWORLD_SYSTEM_PROMPT = """
You are an agent interacting with a virtual text-based environment.

## Notes:
At each step, you should first think then perform action to fulfill the instruction. You should ALWAYS wrap your thinking with the <think> </think> tag and wrap your action with the <action> </action> tag.
You should ALWAYS take one action each step.
DO NOT try to interact with the user at anytime. Finish the task by yourself.

## Action Format:
Below are the available commands you can use:
  look:                             look around your current location
  inventory:                        check your current inventory(you can only have 1 item in your inventory)
  go to (receptacle):               move to a receptacle
  open (receptacle):                open a receptacle
  close (receptacle):               close a receptacle
  take (object) from (receptacle):  take an object from a receptacle
  move (object) to (receptacle):  place an object in or on a receptacle
  examine (something):              examine a receptacle or an object
  use (object):                     use an object
  heat (object) with (receptacle):  heat an object using a receptacle
  clean (object) with (receptacle): clean an object using a receptacle
  cool (object) with (receptacle):  cool an object using a receptacle
  slice (object) with (object):     slice an object using a sharp object

For example your output should be like this:
<think> To solve the task, I need first to ... </think><action>go to cabinet 1</action>
"""


def format_observation(observation: str):
    if "Nothing happens." in observation:
        observation += "Please check if the action you take is valid or you have carefully followed the action format."
    return "Observation: " + observation


def parse_action(response):
    try:
        # parse the action within the <action> </action> tag
        action = response.split("<action>")[1].split("</action>")[0].strip()
        return action
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""


class AlfworldWorkflow(MultiTurnWorkflow):
    """A workflow for alfworld task."""

    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        super().__init__(
            model=model,
            task=task,
        )
        self.task_desc = task.task_desc or "0"
        self.repeat_times = task.repeat_times
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)

    async def get_model_response(self, messages):
        responses = await self.model.chat_async(messages, n=1)
        return responses

    async def get_model_response_text(self, messages):
        return (await self.get_model_response(messages))[0].response_text

    async def generate_env_inference_samples(self, env) -> List[Experience]:
        observation, info = env.reset()
        final_reward = -0.1
        memory = []
        memory.append({"role": "system", "content": AlfWORLD_SYSTEM_PROMPT})
        for r in range(self.max_env_steps):
            format_obs = format_observation(observation)
            memory = memory + [{"role": "user", "content": format_obs}]
            response_text = await self.get_model_response_text(memory)
            memory.append({"role": "assistant", "content": response_text})
            action = parse_action(response_text)
            observation, reward, done, info = env.step(action)
            if done:
                final_reward = reward
                break
        experience = await self.process_messages_to_experience_async(
            memory, final_reward, {"env_rounds": r, "env_done": 1 if done else 0}
        )
        # Close the env to save cpu memory
        env.close()
        return [experience]

    async def run_async(self) -> List[Experience]:
        # assume the task_description is the game_file_path generated.
        # see Trinity-RFT/examples/grpo_alfworld/get_alfworld_data.py
        game_file_path = self.task_desc
        try:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import (
                AlfredDemangler,
                AlfredExpert,
                AlfredExpertType,
            )

            def create_environment(game_file):
                expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
                request_infos = textworld.EnvInfos(
                    description=True, inventory=True, admissible_commands=True
                )

                env_id = textworld.gym.register_game(
                    game_file, request_infos, wrappers=[AlfredDemangler(), expert]
                )
                env = textworld.gym.make(env_id)
                return env

        except Exception as e:
            print("Please make sure you have installed the alfworld package.")
            error_message = f"Error importing AlfworldTWEnv {str(e)}. Please make sure you have installed the alfworld package successfully, following the instructions in https://github.com/alfworld/alfworld"
            raise ImportError(error_message)
        env = create_environment(game_file_path)
        return await self.generate_env_inference_samples(env)


class StepWiseAlfworldWorkflow(RewardPropagationWorkflow):
    """
    An Alfworld workflow refactored to use the RewardPropagationWorkflow base class.

    This workflow manages an Alfworld environment, interacts with it step-by-step
    using a model, and calculates a final reward based on the episode's outcome.
    """

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
        self.game_file_path = task.task_desc or "0"
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)

        self._setup_environment()

        self.observation: Optional[str] = None
        self.done: bool = False
        self.final_reward: float = 0.0
        self.memory: List[dict] = []

    def _setup_environment(self):
        """Initializes the Alfworld text-based environment."""
        try:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import (
                AlfredDemangler,
                AlfredExpert,
                AlfredExpertType,
            )

            def create_environment(game_file):
                expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
                request_infos = textworld.EnvInfos(
                    description=True, inventory=True, admissible_commands=True
                )
                env_id = textworld.gym.register_game(
                    game_file, request_infos, wrappers=[AlfredDemangler(), expert]
                )
                env = textworld.gym.make(env_id)
                return env

            self.env = create_environment(self.game_file_path)

        except ImportError as e:
            error_message = (
                f"Error importing Alfworld dependencies: {e}. Please ensure "
                "Alfworld is installed correctly by following the instructions at "
                "https://github.com/alfworld/alfworld"
            )
            raise ImportError(error_message)

    def run(self) -> List[Experience]:
        # Reset environment and state for a new episode
        self.observation, info = self.env.reset()
        self.done = False
        self.final_reward = -0.1

        self.memory.clear()
        self.memory.append({"role": "system", "content": AlfWORLD_SYSTEM_PROMPT})

        return super().run()

    def step(self, step_num: int) -> bool:
        if self.done:
            return False

        # Format observation for the model
        format_obs = format_observation(self.observation)  # type: ignore
        self.memory.append({"role": "user", "content": format_obs})

        # Get action from the model
        responses = self.model.chat(self.memory)
        response_text = responses[0].response_text
        self.memory.append({"role": "assistant", "content": response_text})
        action = parse_action(response_text)

        # Execute action in the environment
        observation, reward, done, info = self.env.step(action)

        # Update internal state
        self.observation = observation
        self.done = done
        if self.done:
            self.final_reward = reward

        # Return False to stop the run if the episode is done
        return not self.done

    def reward(self, exps: list[Experience]) -> float:
        return self.final_reward

    @property
    def max_step_num(self) -> int:
        """Return the maximum number of steps allowed in an episode."""
        return self.max_env_steps

    def __del__(self):
        """Ensures the environment is closed when the workflow object is destroyed."""
        if hasattr(self, "env"):
            self.env.close()
