# -*- coding: utf-8 -*-
import json
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task

SCIWORLD_SYSTEM_PROMPT = """
You are an agent, your job is to do some scientific experiment in a virtual text-based environment.

## Notes:
At each step, you should first think then perform action to fulfill the instruction. You should ALWAYS wrap your thinking with the <think> </think> tag and wrap your action with the <action> </action> tag.
You should ALWAYS take one action each step.
DO NOT try to interact with the user at anytime. Finish the task by yourself.

## Action Format:
Below are the available commands you can use:
    open OBJ: open a container
    close OBJ: close a container
    activate OBJ: activate a device
    deactivate OBJ: deactivate a device
    connect OBJ to OBJ: connect electrical components
    disconnect OBJ: disconnect electrical components
    use OBJ [on OBJ]: use a device/item
    look around: describe the current room
    examine OBJ: describe an object in detail
    look at OBJ: describe a container's contents
    read OBJ: read a note or book
    move OBJ to OBJ: move an object to a container
    pick up OBJ: move an object to the inventory
    pour OBJ into OBJ: pour a liquid into a container
    mix OBJ: chemically mix a container
    teleport to LOC: teleport to a specific room
    focus on OBJ: signal intent on a task object
    wait: task no action for 10 steps
    wait1: task no action for a step

For example your output should be like this:
<think> Now I will check the bedroom ... </think><action>teleport to bedroom</action>
"""


def format_observation(observation: str):
    return "Observation: \n" + observation


def parse_action(response):
    try:
        # parse the action within the <action> </action> tag
        action = response.split("<action>")[1].split("</action>")[0].strip()
        return action
    except Exception as e:
        print("Error parsing action:", e)
        return ""


class SciWorldWorkflow(MultiTurnWorkflow):
    """A workflow for sciworld task."""

    is_async: bool = True
    can_repeat: bool = True

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
        self.max_env_steps = 30  # should be less than 100

    async def get_model_response(self, messages):
        responses = await self.model.chat_async(messages, n=1)
        return responses

    async def get_model_response_text(self, messages):
        return (await self.get_model_response(messages))[0].response_text

    async def generate_env_inference_samples(self, env, rollout_num) -> List[Experience]:
        # TODO: Make this parallel
        print("Generating env inference samples...")
        golden_rounds = len(env.get_gold_action_sequence())
        experience_list = []
        for i in range(rollout_num):
            observation, info = env.reset()
            observation = (
                "Task Description: " + str(env.get_task_description()) + "\n" + observation
            )
            final_reward = 0.0
            current_reward = 0.0
            memory = []
            memory.append({"role": "system", "content": SCIWORLD_SYSTEM_PROMPT})
            for r in range(self.max_env_steps):
                format_obs = format_observation(observation)
                memory = memory + [{"role": "user", "content": format_obs}]
                response_text = await self.get_model_response_text(memory)
                memory.append({"role": "assistant", "content": response_text})
                action = parse_action(response_text)
                observation, reward, done, info = env.step(action)
                current_reward += reward
                final_reward = max(current_reward, final_reward)
                if done:
                    break
            final_reward = final_reward / 100.0
            experience = await self.process_messages_to_experience_async(
                memory,
                final_reward,
                {"env_rounds": r, "env_done": 1 if done else 0, "golden_rounds": golden_rounds},
            )
            experience_list.append(experience)
        # Close the env to save cpu memory
        env.close()
        return experience_list

    async def run_async(self) -> List[Experience]:
        # assume the task_description is the json object containing task index and the var_num
        # see Trinity-RFT/script/data_prepare/get_scriworld_data.py
        task_desc = self.task_desc
        task_config = json.loads(task_desc)

        rollout_n = self.repeat_times
        # TODO: Make parallel envs
        try:
            from scienceworld import ScienceWorldEnv

            def create_environment(task_config):
                var_num = task_config["var_num"]
                task_name = task_config["task_name"]
                jar_path = task_config["jar_path"]
                simplificationStr = "easy"
                env = ScienceWorldEnv("", jar_path, envStepLimit=100)
                env.load(task_name, var_num, simplificationStr, generateGoldPath=True)
                return env

        except Exception as e:
            print("Please make sure you have installed the sciworld package.")
            error_message = f"Error importing SciWorldTWEnv {str(e)}. Please make sure you have installed the sciworld package successfully, following the instructions in https://github.com/allenai/ScienceWorld"
            raise ImportError(error_message)
        env = create_environment(task_config)
        return await self.generate_env_inference_samples(env, rollout_n)
