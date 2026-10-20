# -*- coding: utf-8 -*-
"""Multi-turn text-to-SQL workflow on local SQLite databases (BIRD / Spider), no Docker needed."""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.sqlite_sql.sqlite_sql_env import (
    SQLiteSQLEnv,
    db_file,
    parse_response,
    schema_text,
)
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task

SQLITE_SQL_SYSTEM_PROMPT = """
You are a data analyst answering questions about a SQLite database.

At each turn, think briefly and then reply with exactly one of:
- <action>SQL</action> to run a read-only SQLite query and see up to {max_obs_rows} rows of its result
- <answer>SQL</answer> to submit the final query that answers the question

You have at most {max_env_steps} turns. Only the submitted query is graded: it is correct when it
returns the same rows as the reference answer.
"""

USER_PROMPT = """Database schema:
{schema}

Question: {question}
{evidence}"""


class SQLiteSQLWorkflow(MultiTurnWorkflow):
    """Agent explores a SQLite database with read-only queries, then submits one query.

    Task fields (BIRD format): ``db_id``, ``question``, ``SQL`` (reference query), optional
    ``evidence``. ``workflow_args``: ``db_root`` (required, ``<db_root>/<db_id>/<db_id>.sqlite``),
    ``max_env_steps`` (default 5), ``max_obs_rows`` (default 10), ``timeout`` (seconds, default 30),
    ``auto_submit`` (default False: grade the last successful query if the agent never submits).
    """

    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        super().__init__(model=model, task=task, auxiliary_models=auxiliary_models)
        args = task.workflow_args
        self.raw_task = task.raw_task or {}
        if not args.get("db_root"):
            raise ValueError("SQLiteSQLWorkflow requires `workflow_args.db_root`.")
        self.db_root = args["db_root"]
        self.max_env_steps = int(args.get("max_env_steps", 5))
        self.max_obs_rows = int(args.get("max_obs_rows", 10))
        self.timeout = float(args.get("timeout", 30))
        self.auto_submit = bool(args.get("auto_submit", False))

    def build_messages(self, db_path) -> List[dict]:
        evidence = self.raw_task.get("evidence")
        system = SQLITE_SQL_SYSTEM_PROMPT.format(
            max_obs_rows=self.max_obs_rows, max_env_steps=self.max_env_steps
        ).strip()
        user = USER_PROMPT.format(
            schema=schema_text(db_path),
            question=self.raw_task["question"],
            evidence=f"External knowledge: {evidence}" if evidence else "",
        ).strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def run_async(self) -> List[Experience]:
        db_path = db_file(self.db_root, self.raw_task["db_id"])
        env = SQLiteSQLEnv(
            db_path, self.raw_task["SQL"], timeout=self.timeout, max_obs_rows=self.max_obs_rows
        )
        if env.gold_error:
            self.logger.warning(f"Reference SQL fails on {db_path}: {env.gold_error}")
        memory = self.build_messages(db_path)
        reward, rounds, last_ok_sql = 0.0, 0, None
        # Honour the task's sampling settings (e.g. eval temperature); one response per turn.
        rollout_args = asdict(self.task.rollout_args)
        rollout_args["n"] = 1

        for rounds in range(1, self.max_env_steps + 1):
            responses = await self.model.chat_async(memory, **rollout_args)
            text = responses[0].response_text
            memory.append({"role": "assistant", "content": text})
            result = env.step(text)
            if result.done:
                reward = result.reward
                break
            if result.error is None:
                last_ok_sql = parse_response(text)[1]
            memory.append({"role": "user", "content": result.observation})
        else:
            if self.auto_submit and last_ok_sql:
                reward, _ = env.score(last_ok_sql)

        info = {
            "env_rounds": rounds,
            "submitted": int(env.submitted),
            "num_queries": env.num_queries,
            "num_query_errors": env.num_errors,
            "gold_error": int(env.gold_error is not None),
            "gold_empty": int(env.gold_empty),
            "final_reward": reward,
        }
        experience = await self.process_messages_to_experience_async(memory, reward, info)
        return [experience]
