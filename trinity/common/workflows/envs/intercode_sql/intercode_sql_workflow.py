# -*- coding: utf-8 -*-
"""InterCode-SQL workflow for Trinity multi-turn GRPO."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task

INTERCODE_SQL_SYSTEM_PROMPT = """
You are an agent interacting with the InterCode-SQL environment.

You need to answer the user's natural-language database question by issuing SQL commands.
At each step, first reason briefly, then provide exactly one environment action.

## Action Format
Wrap your reasoning in <think></think> and your action in <action></action>.
The action can be either:
- a MySQL SQL statement to execute in the current database
- submit, when the latest SQL query result is your final answer

Examples:
<think>I should inspect the relevant rows.</think><action>SELECT * FROM table_name LIMIT 5;</action>
<think>The latest result answers the question.</think><action>submit</action>

Do not ask the user for help. Finish the task by yourself.
"""


def record_db(record: Dict) -> str:
    extra = record.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    return record.get("db") or extra.get("db") or ""


def preprocess_sql(record: Dict) -> str:
    return f"use {record_db(record)}"


def parse_action(response: str) -> str:
    match = re.search(r"<action>(.*?)</action>", response, flags=re.DOTALL)
    if match is None:
        return ""
    return match.group(1).strip()


def format_observation(observation: Any, query: str = "", db: str = "") -> str:
    if query or db:
        return f"Question: {query}\nDatabase: {db}\nObservation: {observation}"
    return f"Observation: {observation}"


class InterCodeSQLWorkflow(MultiTurnWorkflow):
    """A workflow for InterCode-SQL tasks."""

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
            auxiliary_models=auxiliary_models,
        )
        self.raw_task = task.raw_task or {}
        self.query_idx = int(task.task_desc or self.raw_task.get("query_idx", 0))
        self.max_env_steps = int(task.workflow_args.get("max_env_steps", 10))
        self.image_name = task.workflow_args.get("image_name", "docker-env-sql")
        self.verbose = bool(task.workflow_args.get("verbose", False))
        self.auto_submit = bool(task.workflow_args.get("auto_submit", True))
        self.data_path = self._resolve_data_path(task.workflow_args)

    def _resolve_data_path(self, workflow_args: dict) -> str:
        data_path = self.raw_task.get("data_path") or workflow_args.get("data_path")
        if data_path is None:
            raise ValueError(
                "InterCodeSQLWorkflow requires `workflow_args.data_path` or task `data_path`."
            )
        return str(Path(data_path).expanduser())

    async def get_model_response_text(self, messages: List[Dict[str, str]]) -> str:
        responses = await self.model.chat_async(messages, n=1)
        return responses[0].response_text

    def _close_env_connection(self, env: Any) -> None:
        for attr in ("cur", "cnx"):
            handle = getattr(env, attr, None)
            if handle is None:
                continue
            try:
                handle.close()
            except Exception:
                pass

    async def run_async(self) -> List[Experience]:
        try:
            from intercode.envs import SqlEnv
        except Exception as e:
            error_message = (
                f"Error importing InterCode SqlEnv: {e}. Please install InterCode with "
                "`pip install intercode-bench` and make sure Docker is running."
            )
            raise ImportError(error_message)

        env = SqlEnv(
            self.image_name,
            data_path=self.data_path,
            preprocess=preprocess_sql,
            verbose=self.verbose,
        )
        try:
            observation, info = env.reset(self.query_idx)
            query = getattr(env, "query", self.raw_task.get("query", ""))
            db = self.raw_task.get("db") or record_db(getattr(env, "record", {}))
            memory: List[Dict[str, str]] = [
                {"role": "system", "content": INTERCODE_SQL_SYSTEM_PROMPT}
            ]
            final_reward = 0.0
            done = False
            env_info: dict[str, Any] = info or {}

            for step in range(self.max_env_steps):
                if step == 0:
                    user_content = format_observation(observation, query, db)
                else:
                    user_content = format_observation(observation)
                memory.append({"role": "user", "content": user_content})
                response_text = await self.get_model_response_text(memory)
                memory.append({"role": "assistant", "content": response_text})
                action = parse_action(response_text)
                if not action:
                    observation = (
                        "Invalid action. Wrap one SQL statement or submit with "
                        "<action></action>."
                    )
                    env_info = {"action_executed": False}
                    continue

                observation, reward, done, env_info = env.step(action)
                final_reward = float(reward)
                if done:
                    break
            else:
                step = self.max_env_steps
                if self.auto_submit:
                    observation, reward, done, env_info = env.step("submit")
                    final_reward = float(reward)

            experience = await self.process_messages_to_experience_async(
                memory,
                final_reward,
                {
                    "query_idx": self.query_idx,
                    "env_rounds": step,
                    "env_done": 1 if done else 0,
                    "action_executed": 1 if env_info.get("action_executed") else 0,
                    "final_reward": final_reward,
                },
            )
            return [experience]
        finally:
            self._close_env_connection(env)
