# -*- coding: utf-8 -*-
"""CoD workflow for the one-dimensional optimal-control deployment demo.

Each task is a single-turn controller-design problem: the model submits one
scalar action expression, which is evaluated as closed-loop feedback under
hidden linear dynamics for a fixed horizon.

The workflow inherits from ``AsyncCoDMultiStepWorkflow`` to reuse standard CoD
utilities (system-prompt augmentation/trajectory stripping, ICL-example
stripping, and trajectory formatting).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import openai
import torch

from trinity.common.experience import Experience
from trinity.common.workflows.connect_the_dots.base_workflow import (
    AsyncCoDMultiStepWorkflow,
)
from trinity.common.workflows.connect_the_dots.optimalcontrol.env import (
    OptimalControlEnv,
    format_reward,
    serialize_action_expression_xml,
    validate_action_expression,
)
from trinity.common.workflows.connect_the_dots.optimalcontrol.prompts import (
    load_system_prompt,
    load_user_prompt,
)
from trinity.common.workflows.connect_the_dots.utils import parse_xml_answer
from trinity.common.workflows.workflow import Task

if TYPE_CHECKING:
    from trinity.common.models.model import ModelWrapper


_PACK_CONTROL_POLICIES: dict[tuple[str, int], dict[str, object]] = {}  #!!!
PARSE_FAILURE_REWARD = -0.1


def parse_action_expression_submission(response: str) -> tuple[Optional[str], str]:
    """Parse one terminal XML action expression."""
    if "```" in response:
        return None, "python_or_markdown_code_is_not_allowed"
    payload, parse_error = parse_xml_answer(response)
    if payload is None:
        return None, parse_error
    answer_end = response.rfind("</answer>") + len("</answer>")
    if response[answer_end:].strip():
        return None, "answer_must_end_the_response"
    if payload.get("action") != "action_expression":
        return None, "expected_action_expression"
    try:
        return validate_action_expression(payload.get("args")), ""
    except ValueError as error:
        return None, str(error)


class CoDOptimalControlWorkflow(AsyncCoDMultiStepWorkflow):
    """Single-turn action-expression workflow for optimal control.

    The model emits one scalar feedback expression; the workflow validates it,
    evaluates it under hidden dynamics, and assigns the rollout reward.
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        # The sync OpenAI client from the base class is not needed; in
        # external mode an async client is created lazily in `_chat`.
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
            use_openai_client=False,
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset task-specific configuration and environment state."""
        super().reset(task)

        self.workflow_args = task.workflow_args if hasattr(task, "workflow_args") else {}
        self.trajectory_dump_dir = self.workflow_args.get("trajectory_dump_dir", None)
        self.task_idx = int(self.raw_task["task_idx"])
        self.include_previous_control_policy = bool(
            self.workflow_args.get("include_previous_control_policy", False)
        )
        self.previous_control_policy: Optional[str] = None
        self.submitted_control_policy: Optional[str] = None
        self.previous_control_policy_evaluation: Optional[dict[str, object]] = None  #!!!
        self.submitted_control_policy_evaluation: Optional[dict[str, object]] = None  #!!!

        task_config = OptimalControlEnv.resolve_task(self.raw_task, self.workflow_args)
        self.a_env = task_config["a_env"]
        self.b_env = task_config["b_env"]
        self.min_abs_b_env = task_config["min_abs_b_env"]  #!!!
        self.x0 = task_config["x0"]
        self.v0 = task_config["v0"]
        self.x_target = task_config["x_target"]
        self.v_target = task_config["v_target"]
        self.horizon = task_config["horizon"]
        self.control_penalty_coef = task_config["control_penalty_coef"]
        self.enable_process_noise = task_config["enable_process_noise"]  #!!!
        self.process_noise_std = task_config["process_noise_std"]  #!!!
        self.task_seed = task_config["task_seed"]  #!!!

        self.env = OptimalControlEnv(
            a_env=self.a_env,
            b_env=self.b_env,
            x0=self.x0,
            v0=self.v0,
            x_target=self.x_target,
            v_target=self.v_target,
            horizon=self.horizon,
            control_penalty_coef=self.control_penalty_coef,
            enable_process_noise=self.enable_process_noise,  #!!!
            process_noise_std=self.process_noise_std,  #!!!
            task_seed=self.task_seed,  #!!!
        )

        # State for the current task
        self.memory: List[dict] = []

    @property
    def max_step_num(self) -> int:
        """This is a single-turn workflow: one environment step per task."""
        return 1

    def _pack_policy_key(self) -> tuple[str, int]:
        pack_seed = int(self.raw_task.get("pack_seed", self.raw_task.get("seed", 42)))
        return str(self.task.batch_id), pack_seed

    def _build_system_prompt(self) -> str:
        """Build system prompt, appending CoD hint and token-limit notes."""
        b_lower, b_upper = map(float, self.workflow_args["b_env_range"])
        known_b_sign = "positive" if b_lower > 0.0 else "negative" if b_upper < 0.0 else None
        sys_prompt = load_system_prompt(
            control_penalty_coef=self.control_penalty_coef,
            enable_process_noise=self.enable_process_noise,  #!!!
            process_noise_std=self.process_noise_std,  #!!!
            known_b_sign=known_b_sign,
        )
        return self._augment_system_prompt(sys_prompt)

    def _build_user_prompt(self) -> str:
        """Build the task-specific user prompt."""
        user_prompt = load_user_prompt(
            x0=self.x0,
            v0=self.v0,
            x_target=self.x_target,
            v_target=self.v_target,
            horizon=self.horizon,
        )
        if self.icl_examples:
            user_prompt = (
                f"{user_prompt}\n\nHere are some reference examples:\n\n{self.icl_examples}"
            )
        if self.previous_control_policy:  #!!!
            evaluation = self.previous_control_policy_evaluation or {}
            executed = bool(evaluation.get("action_execution_success", False))
            reward = float(evaluation.get("reward", 0.0))
            reward_text = format_reward(reward)
            position_error = evaluation.get("signed_position_error")
            velocity_error = evaluation.get("signed_velocity_error")
            behavior = (
                "The expression completed the earlier rollout."
                if executed
                else "The expression did not complete the earlier rollout."
            )
            position_text = "unknown" if position_error is None else f"{float(position_error):.3f}"
            velocity_text = "unknown" if velocity_error is None else f"{float(velocity_error):.3f}"
            user_prompt += (  #!!!
                "\n\n## Controller evidence from an earlier task\n\n"
                "The expression and measurements below come from an earlier, different "
                "task in the same hidden environment. They are not rollout results for "
                "the current task. Use them only as evidence about the shared hidden "
                "dynamics and controller behavior.\n\n"
                f"- Earlier initial state: x0={evaluation.get('x0')}, "
                f"v0={evaluation.get('v0')}\n"
                f"- Earlier target: x_target={evaluation.get('x_target')}, "
                f"v_target={evaluation.get('v_target')}\n"
                f"- Earlier horizon: {evaluation.get('horizon')}\n"
                f"- Completed that earlier rollout: {'yes' if executed else 'no'}\n"
                f"- Reward on that earlier task: {reward_text}\n"
                f"- Signed terminal position residual x_T - x_target: {position_text}\n"
                f"- Signed terminal velocity residual v_T - v_target: {velocity_text}\n"
                f"- Main behavior: {behavior}\n\n"
                "Action expression used on that earlier task:\n\n"
                f"{serialize_action_expression_xml(self.previous_control_policy)}\n\n"  #!!!
                "Adapt the expression to the current initial state, target, and horizon, "
                "or submit a new one if it performed poorly."
            )  #!!!
        return user_prompt

    def _build_feedback(
        self,
        result: Optional[dict],
        evaluation_status: str,
        error_detail: Optional[str] = None,
        response_truncated: bool = False,
    ) -> str:
        """Build environment feedback string for this rollout."""
        if evaluation_status == "format_error" and response_truncated:
            return (
                "Invalid response: generation reached the maximum response length before a "
                "valid final action-expression XML submission was completed. The environment "
                f"was not executed and reward was set to {PARSE_FAILURE_REWARD}."
            )
        if evaluation_status == "format_error":
            detail = error_detail or "invalid action-expression XML"
            return (
                "Invalid controller submission: expected exactly one terminal "
                "<answer><action_expression>...</action_expression></answer> block "
                f"({detail}). The environment was not executed and reward was set to "
                f"{PARSE_FAILURE_REWARD}."
            )
        if evaluation_status == "action_error":
            detail = error_detail or "expression evaluation failed."
            return (
                f"Invalid policy: {detail}. Controller evaluation stopped when the expression "
                "failed; no complete trajectory or loss was returned, and reward was forced "
                "to 0."
            )
        if result is None:
            return "Invalid policy: controller evaluation failed. Reward was forced to 0."

        return "\n\n".join(
            [
                "The environment executed the submitted policy under the fixed hidden dynamics.",
                self.env.render_trajectory(result),
                self.env.render_summary(result),
            ]
        )

    def _build_rollout_trajectory(
        self,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        action_expression: Optional[str],
    ) -> str:
        """Build controller evidence for CoD hint generation."""
        if action_expression is not None:
            policy_label = "Submitted action expression"
            policy = serialize_action_expression_xml(action_expression)
        else:
            policy_label = "Raw agent response; no valid action expression was extracted"
            policy = response_text
        return "\n\n".join(
            [
                f"System:\n{self._strip_system_prompt(system_prompt)}",
                f"Task:\n{self._strip_icl_examples(user_prompt)}",
                f"{policy_label}:\n{policy}",
            ]
        )

    def _dump_trajectory(
        self,
        run_id: int,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        action_expression: Optional[str],
        result: Optional[dict],
        format_error: bool,
        action_error: bool,
        response_token_count: int,
        max_response_tokens: Optional[int],
        response_hit_max_tokens: bool,
        response_truncated: bool,
        error_detail: Optional[str],
        feedback: str,
    ) -> None:
        """Persist one task trajectory to a JSONL file for offline analysis."""
        if not self.trajectory_dump_dir:
            return

        os.makedirs(self.trajectory_dump_dir, exist_ok=True)
        dump_path = os.path.join(
            self.trajectory_dump_dir,
            "trajectories.jsonl",
        )

        # This workflow only handles the actual solve_task step; the CoD
        # hint-generation step is executed by AsyncCoDUpdateContextWorkflow and
        # already captured in the CoD per-part JSON logs.
        exp_type = "solve_task"
        rollout = None
        metrics = {
            "reward": (
                PARSE_FAILURE_REWARD
                if format_error
                else 0.0 if result is None else result["reward"]
            ),
            "format_error": format_error,
            "expression_schema_error": format_error,
            "action_error": action_error,
            "response_token_count": response_token_count,
            "max_response_tokens": max_response_tokens,
            "response_hit_max_tokens": response_hit_max_tokens,
            "response_truncated": response_truncated,
        }
        if result is not None:
            rollout = {
                "xs": result["xs"],
                "vs": result["vs"],
                "us": result["us"],
                "process_noises": result["process_noises"],
            }
            metrics.update(
                {
                    "loss": result["loss"],
                    "x_final": result["x_final"],
                    "v_final": result["v_final"],
                }
            )

        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": run_id,
            "batch_id": self.task.batch_id,
            "task_id": self.task.task_id,
            "task_idx": self.task_idx,
            "seed": self.raw_task.get("seed", None),
            "exp_type": exp_type,
            "a_env": self.a_env,
            "b_env": self.b_env,
            "min_abs_b_env": self.min_abs_b_env,  #!!!
            "x0": self.x0,
            "v0": self.v0,
            "x_target": self.x_target,
            "v_target": self.v_target,
            "horizon": self.horizon,
            "control_penalty_coef": self.control_penalty_coef,
            "enable_process_noise": self.enable_process_noise,  #!!!
            "process_noise_std": self.process_noise_std,  #!!!
            "task_seed": self.task_seed,  #!!!
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_text": response_text,
            "action_expression": action_expression,
            "controller_xml": (
                None
                if action_expression is None
                else serialize_action_expression_xml(action_expression)
            ),
            "controller_error_detail": error_detail,
            "hint": self.hint,
            "rollout": rollout,
            "metrics": metrics,
            "feedback": feedback,
        }

        try:
            with open(dump_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # Never crash the main workflow because of a logging failure.
            print(f"[OptimalControlWorkflow] Failed to dump trajectory: {e}")

    async def _chat(self, messages: List[dict]) -> List[Experience]:
        """Generate responses via the local engine or an external API.

        For local engines (vLLM/SGLang) this delegates to ``chat_async``. In
        Trinity's native external mode (``model.external_model.enable: true``,
        bench only), ``ModelWrapper.model`` is ``None`` and only the OpenAI
        client path is available, so the external API is called directly and
        each choice is wrapped into a minimal ``Experience``.
        """
        if self.model.model is not None:
            return await self.model.chat_async(messages, **self.rollout_args)

        # Trinity appends `/v1` when building the OpenAI client, so tolerate
        # base URLs that already include the suffix (e.g. DashScope's
        # `.../compatible-mode/v1`).
        if (
            self.model.openai_async_client is None
            and self.model.api_address
            and self.model.api_address.rstrip("/").endswith("/v1")
        ):
            self.model.api_address = self.model.api_address.rstrip("/")[: -len("/v1")]
        client = self.model.get_openai_async_client()
        rollout_args = self.rollout_args
        request_kwargs = {
            "model": self.model.config.external_model_config.model_name,
            "messages": messages,
            "max_completion_tokens": rollout_args.get("max_tokens")
            or self.model.config.max_response_tokens,
            "n": rollout_args.get("n", 1),
        }
        if rollout_args.get("temperature") is not None:
            request_kwargs["temperature"] = rollout_args["temperature"]
        # DashScope-compatible endpoints expect a top-level `enable_thinking`
        # in extra_body (not vLLM-style chat_template_kwargs).
        if self.model.config.enable_thinking is not None:
            request_kwargs["extra_body"] = {"enable_thinking": self.model.config.enable_thinking}

        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(**request_kwargs)
                break
            except (openai.RateLimitError, openai.InternalServerError):
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1.0 * (2**attempt))

        usage_metrics = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            for usage_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_val = getattr(usage, usage_key, None)
                if isinstance(usage_val, (int, float)):
                    usage_metrics[f"usage/{usage_key}"] = float(usage_val)

        experiences: List[Experience] = []
        for choice in response.choices:
            finish_reason = str(choice.finish_reason or "").lower()
            experiences.append(
                Experience(
                    # Minimal valid token tensor; external APIs return no token ids.
                    tokens=torch.tensor([0, 0], dtype=torch.int32),
                    logprobs=torch.tensor([0.0], dtype=torch.float32),
                    prompt_length=1,
                    response_text=choice.message.content or "",
                    truncate_status=("response_truncated" if finish_reason == "length" else None),
                    metrics=dict(usage_metrics),
                    info={"finish_reason": finish_reason},
                )
            )
        return experiences

    async def step_async(self, step_num: int) -> Tuple[bool, List[Experience]]:
        """Execute one single-turn optimal-control task."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt()

        self.memory = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.reply_prefix:
            self.memory.append({"role": "assistant", "content": self.reply_prefix})

        responses = await self._chat(self.memory)

        experiences: List[Experience] = []
        for i, response in enumerate(responses):
            response.eid.run = i
            response.eid.step = 0
            response_text = response.response_text or ""
            response.response_text = response_text
            action_expression, expression_error = parse_action_expression_submission(response_text)
            controller_xml = (
                None
                if action_expression is None
                else serialize_action_expression_xml(action_expression)
            )

            max_response_tokens = self.rollout_args.get("max_tokens")
            if max_response_tokens is None:
                max_response_tokens = self.model.config.max_response_tokens
            if self.model.model is None:
                response_token_count = (
                    int(response.metrics.get("usage/completion_tokens", -1))
                    if len(responses) == 1
                    else -1
                )
                response_hit_max_tokens = response.truncate_status == "response_truncated"
            else:
                response_token_count = len(response.tokens) - response.prompt_length
                response_hit_max_tokens = (
                    max_response_tokens is not None and response_token_count >= max_response_tokens
                )
            response_truncated = response_hit_max_tokens and action_expression is None

            if action_expression is None:
                result = None
                evaluation_status = "format_error"
                error_detail = expression_error
            else:
                result, evaluation_status, error_detail = self.env.rollout_action_expression(
                    action_expression
                )
            format_error = evaluation_status == "format_error"
            action_error = evaluation_status == "action_error"
            reward = (
                PARSE_FAILURE_REWARD
                if format_error
                else 0.0 if result is None else float(result["reward"])
            )
            feedback = self._build_feedback(
                result,
                evaluation_status=evaluation_status,
                error_detail=error_detail,
                response_truncated=response_truncated,
            )
            if action_expression is not None:  #!!!
                self.submitted_control_policy = action_expression  #!!!
                self.submitted_control_policy_evaluation = {  #!!!
                    "format_parse_success": True,  #!!!
                    "action_execution_success": not action_error,  #!!!
                    "reward": reward,  #!!!
                    "max_score": 1.0,  #!!!
                    "x_final": None if result is None else float(result["x_final"]),  #!!!
                    "v_final": None if result is None else float(result["v_final"]),  #!!!
                    "position_error": (
                        None if result is None else abs(float(result["x_final"]) - self.x_target)
                    ),
                    "velocity_error": (
                        None if result is None else abs(float(result["v_final"]) - self.v_target)
                    ),
                    "signed_position_error": (
                        None if result is None else float(result["x_final"]) - self.x_target
                    ),
                    "signed_velocity_error": (
                        None if result is None else float(result["v_final"]) - self.v_target
                    ),
                    "x0": self.x0,
                    "v0": self.v0,
                    "x_target": self.x_target,
                    "v_target": self.v_target,
                    "horizon": self.horizon,
                    "feedback": feedback,  #!!!
                }  #!!!
            trajectory = self._build_rollout_trajectory(
                system_prompt,
                user_prompt,
                response_text,
                action_expression,
            )

            response.reward = reward
            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(
                {
                    "reward": reward,
                    "format_error": float(format_error),
                    "format_error_termination": float(format_error),
                    "expression_schema_error": float(format_error),
                    "action_error": float(action_error),
                    "response_token_count": float(response_token_count),
                    "response_hit_max_tokens": float(response_hit_max_tokens),
                    "response_truncated": float(response_truncated),
                    "previous_control_policy_used": float(bool(self.previous_control_policy)),
                    "process_noise_rms": (  #!!!
                        0.0 if result is None else float(result["process_noise_rms"])  #!!!
                    ),  #!!!
                }
            )
            if result is not None:
                response.metrics.update(
                    {
                        "loss": float(result["loss"]),
                        "x_final": float(result["x_final"]),
                        "v_final": float(result["v_final"]),
                    }
                )

            if response.info is None:
                response.info = {}
            response.info.update(
                {
                    "sys_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "feedback": feedback,
                    "trajectory": trajectory,
                    "a_env": self.a_env,
                    "b_env": self.b_env,
                    "min_abs_b_env": self.min_abs_b_env,  #!!!
                    "x0": self.x0,
                    "v0": self.v0,
                    "x_target": self.x_target,
                    "v_target": self.v_target,
                    "horizon": self.horizon,
                    "previous_control_policy": self.previous_control_policy or "",
                    "action_expression": action_expression or "",
                    "controller_xml": controller_xml or "",
                    "previous_control_policy_evaluation": (  #!!!
                        self.previous_control_policy_evaluation or {}  #!!!
                    ),  #!!!
                    "enable_process_noise": self.enable_process_noise,  #!!!
                    "process_noise_std": self.process_noise_std,  #!!!
                    "task_seed": self.task_seed,  #!!!
                    "early_termination_by_format_issue": format_error,
                    "expression_schema_error": format_error,
                    "action_error": action_error,
                    "response_token_count": response_token_count,
                    "max_response_tokens": max_response_tokens,
                    "response_hit_max_tokens": response_hit_max_tokens,
                    "response_truncated": response_truncated,
                    "controller_error_detail": error_detail or "",
                }
            )

            # Persist and update conversation memory.
            self._dump_trajectory(
                run_id=response.eid.run,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=response_text,
                action_expression=action_expression,
                result=result,
                format_error=format_error,
                action_error=action_error,
                response_token_count=response_token_count,
                max_response_tokens=max_response_tokens,
                response_hit_max_tokens=response_hit_max_tokens,
                response_truncated=response_truncated,
                error_detail=error_detail,
                feedback=feedback,
            )

            experiences.append(response)

        # Single-step workflow: do not continue after this turn.
        return False, experiences

    async def run_async(self) -> List[Experience]:
        """Run one optimal-control task with optional pack-level policy reuse."""
        policy_key = None
        if self.include_previous_control_policy:
            policy_key = self._pack_policy_key()
            if self.task_idx == 0:
                _PACK_CONTROL_POLICIES.pop(policy_key, None)
            previous_record = _PACK_CONTROL_POLICIES.get(policy_key)  #!!!
            if previous_record:  #!!!
                self.previous_control_policy = str(previous_record["policy"])  #!!!
                self.previous_control_policy_evaluation = dict(  #!!!
                    previous_record.get("evaluation", {})  #!!!
                )  #!!!

        _, experiences = await self.step_async(step_num=0)

        if policy_key is not None:
            submitted_evaluation = self.submitted_control_policy_evaluation or {}
            if self.submitted_control_policy and bool(
                submitted_evaluation.get("action_execution_success", False)
            ):
                _PACK_CONTROL_POLICIES[policy_key] = {  #!!!
                    "policy": self.submitted_control_policy,  #!!!
                    "evaluation": submitted_evaluation,  #!!!
                }  #!!!
            pack_size = int(self.raw_task.get("pack_size", self.task_idx + 1))
            if self.task_idx + 1 >= pack_size:
                _PACK_CONTROL_POLICIES.pop(policy_key, None)

        if experiences:
            # Set step and env-step count only on the last experience; CoD
            # post-processing will propagate metrics as needed. The trajectory
            # string was already built per-response during step_async.
            experiences[-1].eid.step = 0
            experiences[-1].metrics["actual_env_steps"] = 1

        return experiences
