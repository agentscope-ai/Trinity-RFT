# -*- coding: utf-8 -*-
"""
CoD (Connect-the-Dots) workflow for the Learn2Ask medical inquiry task.

Single-turn workflow adapted from examples/learn_to_ask/workflow/
workflow_learn2ask.py. Uses an auxiliary model as LLM judge (Qwen judge
from the Learn2Ask paper) for content/format scoring; action score is a
hard rule based on <stop /> detection. On top of the original workflow
this module surfaces trajectory / feedback into exp.info so the CoD
meta-workflow can consume them for cross-task hint learning.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from trinity.common.experience import Experience
from trinity.common.workflows.connect_the_dots.base_workflow import (
    AsyncCoDMultiStepWorkflow,
)
from trinity.common.workflows.connect_the_dots.learn2ask.prompts import (
    load_reward_judge_prompt,
    load_system_prompt,
)
from trinity.common.workflows.workflow import (
    Task,
    log_sys_user_prompts_in_exp,
)

if TYPE_CHECKING:
    import openai

    from trinity.common.models.model import ModelWrapper


# --- Tag parsing for judge output ---

_TAG_PATTERN = re.compile(r"\[(\w+)\](.*?)\[/\1\]", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_tag_string(text: str) -> Dict[str, str]:
    """Parse [tag]...[/tag] pairs, ignoring <think> blocks."""
    return {
        tag: value.strip()
        for tag, value in _TAG_PATTERN.findall(_THINK_PATTERN.sub("", text))
    }


# --- Dialogue formatting helpers ---

def _merge_dialogue(msg_list: List[dict]) -> str:
    """Render a list of {role, content} messages as 'patient:' / 'doctor:' lines."""
    lines = []
    for msg in msg_list:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"patient: {content}")
        elif role == "assistant":
            lines.append(f"doctor: {content}")
    return "\n".join(lines)


def _merge_consecutive_same_role(messages: List[dict]) -> List[dict]:
    """Collapse consecutive same-role messages by joining contents with newlines."""
    if not messages:
        return messages
    merged = [dict(messages[0])]
    for msg in messages[1:]:
        if msg.get("role") == merged[-1].get("role"):
            merged[-1]["content"] = f"{merged[-1]['content']}\n{msg['content']}"
        else:
            merged.append(dict(msg))
    return merged


class CoDLearn2AskWorkflow(AsyncCoDMultiStepWorkflow):
    """Learn2Ask single-turn workflow wired into the CoD meta-workflow.

    Reward stays aligned with the original Learn2Ask paper (LLM judge for
    content/format, hard-rule action gating, fusion formula). The main
    CoD-side additions are:

    * `set_hint / set_icl_examples / set_max_response_tokens_restraint`
      to receive the CoD context between tasks.
    * `exp.info["trajectory"]` and `exp.info["feedback"]` populated for
      the iterative hint generator to consume.
    """

    @property
    def max_step_num(self) -> int:
        return 1

    def __init__(
        self,
        *,
        task: Task,
        model: "ModelWrapper",
        auxiliary_models: Optional[List["ModelWrapper"]] = None,
        use_openai_client: bool = False,
    ):
        assert (
            auxiliary_models is not None and len(auxiliary_models) == 1
        ), "CoDLearn2AskWorkflow expects exactly one auxiliary model (judge)."
        self.reset(task)
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
            use_openai_client=use_openai_client,
        )

    # ------------------------------------------------------------------
    # Lifecycle: reset / CoD setters
    # ------------------------------------------------------------------

    def reset(self, task: Task):
        """Reset per-task state. Does not call super().reset() on purpose —
        Learn2Ask computes reward in run_async() rather than via RewardFn,
        so we skip the RewardFn instantiation that BaseSimpleWorkflow.reset
        would do."""
        self.task = task
        workflow_args = task.workflow_args or {}
        self.train_mode: str = workflow_args.get("train_mode", "Ra+Rs")
        self.fusion_mode: str = workflow_args.get("fusion_mode", "default")

        self.format_args = task.format_args
        self.reply_prefix = (
            task.format_args.reply_prefix if task.format_args else None
        )
        self.raw_task: Dict = task.raw_task or {}
        self.task_desc = task.task_desc
        if isinstance(self.task_desc, list):
            self.task_desc = _merge_consecutive_same_role(self.task_desc)

        # Ground truth (from predata)
        self.action_truth: str = self.raw_task.get("decision_truth", "continue")
        self.info_truth: str = self.raw_task.get("info_truth", "")
        self.session_id = str(self.raw_task.get("session_id", ""))
        self.diagn: str = self.raw_task.get("diagn", "")

        self.system_prompt: str = load_system_prompt(train_mode=self.train_mode)

        # CoD-controlled fields, reset every task
        self.hint: Optional[str] = None
        self.icl_examples: Optional[str] = None
        self.max_response_tokens_restraint: Optional[int] = None

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def format_messages(self) -> List[dict]:
        sys_prompt = self._augment_system_prompt(self.system_prompt)
        if isinstance(self.task_desc, list):
            messages = [{"role": "system", "content": sys_prompt}] + self.task_desc
        elif isinstance(self.task_desc, str):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": self.task_desc},
            ]
        else:
            raise ValueError(
                f"task.task_desc must be a list of messages or a str, got {type(self.task_desc)}"
            )
        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})
        return messages

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def run_async(self) -> List[Experience]:
        messages = self.format_messages()
        responses = await self.model.chat_async(messages, **self.rollout_args)

        # Score every response in parallel so concurrent judge calls overlap.
        score_results = await asyncio.gather(
            *(self._compute_reward(r.response_text or "") for r in responses)
        )

        history_text = self._history_as_dialogue()
        task_desc_str = history_text  # single-turn: the prompt dialogue is the task

        for response, (reward, component_scores, judge_feedback, judge_format_error) in zip(
            responses, score_results
        ):
            response.reward = reward
            if response.metrics is None:
                response.metrics = {}
            # Skip per-component scores when judge failed — they're 0/0/0 placeholders, not real signal.
            if not judge_format_error:
                response.metrics.update(component_scores)

            resp_text = response.response_text or ""
            response.info["task_desc"] = task_desc_str
            response.info["trajectory"] = (
                f"# Task instructions given to the assistant\n{self.system_prompt}\n\n"
                f"# Resulting conversation\n{history_text}\ndoctor: {resp_text}"
                if history_text else
                f"# Task instructions given to the assistant\n{self.system_prompt}\n\n"
                f"# Resulting conversation\ndoctor: {resp_text}"
            )
            response.info["feedback"] = self._build_feedback(
                component_scores, resp_text, judge_feedback
            )
            if judge_format_error:
                response.info["judge_format_error"] = True

        log_sys_user_prompts_in_exp(messages, responses)
        return responses

    # ------------------------------------------------------------------
    # Reward computation (mirrors examples/learn_to_ask reward_fn)
    # ------------------------------------------------------------------

    async def _compute_reward(
        self, response: str
    ) -> Tuple[float, Dict[str, float], str, bool]:
        action_response = "stop" if "<stop />" in response else "continue"
        judge_feedback = ""
        judge_format_error = False

        if self.action_truth != action_response:
            action_score = format_score = content_score = 0.0
        else:
            action_score = 1.0
            if self.action_truth == "continue":
                score_dict = await self._llm_judge(response)
                try:
                    format_score = float(score_dict.get("format_score"))
                    content_score = float(score_dict.get("content_score"))
                except (TypeError, ValueError):
                    judge_format_error = True
                    format_score, content_score = 0.0, 0.0
                judge_feedback = (score_dict.get("feedback") or "").strip()
            else:
                content_score = 1.0
                format_score = 1.0 if response.strip() == "<stop />" else 0.0

        final_reward = self._fuse_scores(action_score, content_score, format_score)
        metrics = {
            "action_score": action_score,
            "content_score": content_score,
            "format_score": format_score,
        }
        return final_reward, metrics, judge_feedback, judge_format_error

    def _fuse_scores(
        self, action_score: float, content_score: float, format_score: float
    ) -> float:
        if self.train_mode == "Ra+Rs":
            if self.fusion_mode == "sum":
                return action_score + content_score + format_score
            return action_score * (1 + 2 * content_score) + format_score
        if self.train_mode == "Ra":
            return 2 * content_score + format_score
        # "Rs"
        return action_score * 3 + format_score

    async def _llm_judge(
        self, response: str, max_retries: int = 5
    ) -> Dict[str, str]:
        """Call the auxiliary judge model with the dialogue + candidate reply.

        Matches examples/learn_to_ask/workflow/llm_reward's retry loop:
        one initial attempt plus `max_retries` retries (total of
        max_retries + 1 = 6 tries), sleeping attempt-count seconds
        between tries.
        """
        client: "openai.AsyncOpenAI" = self.auxiliary_models[0]
        history = self._history_as_dialogue()
        judge_user = (
            f"{history}\ndoctor: {response}\n" if history else f"doctor: {response}\n"
        )
        judge_messages = [
            {"role": "system", "content": load_reward_judge_prompt(info_truth=self.info_truth)},
            {"role": "user", "content": judge_user},
        ]

        total_tries = max_retries + 1
        for attempt in range(total_tries):
            try:
                completion = await client.chat.completions.create(
                    model=client.model_path,
                    messages=judge_messages,
                    stream=False,
                    temperature=1.0,
                    top_p=0.95,
                    presence_penalty=0.0,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True},
                        "top_k": 20,
                        "min_p": 0.0,
                        "repetition_penalty": 1.0,
                    },
                )
                content = completion.choices[0].message.content or ""
                return _parse_tag_string(content)
            except Exception:
                if attempt >= total_tries - 1:
                    return {}
                await asyncio.sleep(1.0 * (attempt + 1))
        return {}

    # ------------------------------------------------------------------
    # Auxiliary artefacts for CoD hint generation
    # ------------------------------------------------------------------

    def _history_as_dialogue(self) -> str:
        """Render task_desc (observed context) as patient/doctor dialogue."""
        if isinstance(self.task_desc, list):
            return _merge_dialogue(self.task_desc)
        if isinstance(self.task_desc, str):
            return f"patient: {self.task_desc}"
        return ""

    def _build_feedback(
        self, component_scores: Dict[str, float], response: str, judge_feedback: str = ""
    ) -> str:
        """Human-readable feedback summarising the judge + action outcome."""
        action_s = component_scores["action_score"]
        content_s = component_scores["content_score"]
        format_s = component_scores["format_score"]
        got = "stop" if "<stop />" in response else "continue"
        total = action_s * (1 + 2 * content_s) + format_s

        parts = [
            f"Reward {total:.1f}/4.0. Subscores: action={action_s:.0f}, "
            f"content={content_s:.1f}, format={format_s:.1f}."
        ]

        if action_s == 0:
            if self.action_truth == "continue":
                parts.append(
                    "Action: chose 'stop' but expected 'continue'. This is premature "
                    "termination. The patient still has unfilled symptom dimensions "
                    "worth probing; a focused follow-up question targeting one of them "
                    "would have scored up to 4.0. action=0 zeros all subscores. The big "
                    "penalty for stopping early signals that information sufficiency is "
                    "the primary stop criterion."
                )
            else:
                parts.append(
                    "Action: chose 'continue' and asked a question but expected 'stop'. "
                    "This is a redundant question. Critical symptom info has already been "
                    "gathered, so the right action was to emit '<stop />' for a 4.0 reward. "
                    "action=0 zeros all subscores. The penalty signals that over-asking "
                    "after enough info is collected is as costly as a wrong question."
                )
            return " ".join(parts)

        parts.append(f"Action: '{got}' matched expected. action=1.0.")

        if self.action_truth == "stop":
            if format_s > 0:
                parts.append("Format=1.0: response was clean '<stop />'. Content auto-1.0.")
            else:
                parts.append(
                    "Format=0.0: response had extra characters around '<stop />', "
                    "losing 1.0. Content auto-1.0."
                )
        else:
            parts.append(
                f"Format={format_s:.1f}. Judge penalizes multi-question turns: "
                f"1 question scores 1.0, 2 scores 0.5, 3 or more scores 0.0."
            )
            parts.append(
                f"Content={content_s:.1f}. Judge rates how directly the question targets "
                f"an unfilled symptom dimension: direct=1.0, relevant=0.5, irrelevant=0.0."
            )
            if judge_feedback:
                parts.append(f"Judge note: {judge_feedback}")

        return " ".join(parts)
