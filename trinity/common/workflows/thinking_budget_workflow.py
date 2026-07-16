"""OpenAI-compatible workflow with vLLM thinking-budget support."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.workflows.workflow import Task, Workflow

if TYPE_CHECKING:
    from trinity.common.models.model import ModelWrapper


class ThinkingBudgetWorkflow(Workflow):
    """Run raw OpenAI messages with a per-request reasoning-token budget.

    vLLM enforces the end-of-reasoning sequence by forcing its logits. Those
    tokens therefore have an exact log probability of zero. The workflow masks
    the first such contiguous forced run at/after the budget boundary out of the
    action mask so it does not contribute to the policy loss.
    """

    can_repeat = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        raw_task = task.raw_task or {}
        self.messages = raw_task.get("messages")
        if not isinstance(self.messages, list):
            raise ValueError("ThinkingBudgetWorkflow requires raw_task['messages'] to be a list.")

        self.thinking_token_budget = int(task.workflow_args.get("thinking_token_budget"))  # type: ignore [arg-type]
        if not isinstance(self.thinking_token_budget, int) or self.thinking_token_budget < 0:
            raise ValueError(
                "workflow_args['thinking_token_budget'] must be a non-negative integer."
            )
        self.reasoning_end_str = task.workflow_args.get("reasoning_end_str")
        if self.reasoning_end_str is not None and not isinstance(self.reasoning_end_str, str):
            raise ValueError("workflow_args['reasoning_end_str'] must be a string.")
        if not model.enable_history:
            raise ValueError(
                "ThinkingBudgetWorkflow requires explorer.rollout_model.enable_history=true."
            )
        self.repeat_times = task.rollout_args.n
        self.run_id_base = 0
        self.client = model.get_openai_client()

    def set_repeat_times(self, repeat_times: int, run_id_base: int) -> None:
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    @staticmethod
    def _mask_forced_reasoning_end(
        exp: Experience,
        budget: int,
        reasoning_end_token_ids: Optional[List[int]] = None,
    ) -> None:
        """Set the vLLM-forced reasoning-end token action entries to zero."""
        response_len = len(exp.tokens) - exp.prompt_length  # type: ignore [arg-type]
        if exp.action_mask is None or len(exp.action_mask) != response_len:
            exp.action_mask = torch.ones(response_len, dtype=torch.bool)
        response_token_ids = exp.tokens[exp.prompt_length :].tolist()  # type: ignore [index]
        if reasoning_end_token_ids:
            for start in range(response_len - len(reasoning_end_token_ids) + 1):
                end = start + len(reasoning_end_token_ids)
                if response_token_ids[start:end] == reasoning_end_token_ids:
                    exp.action_mask[start:end] = False
                    return
        if exp.logprobs is None or len(exp.logprobs) != response_len:
            return

        # Forced tokens receive probability 1 after vLLM's logits override, so
        # their returned log probability is exactly 0. Search from the earliest
        # possible budget boundary and mask only the first contiguous run.
        forced = torch.eq(exp.logprobs, 0)
        start = min(budget, response_len)
        while start < response_len and not bool(forced[start]):
            start += 1
        end = start
        while end < response_len and bool(forced[end]):
            end += 1
        if start < end:
            exp.action_mask[start:end] = False

    def run(self) -> List[Experience]:
        rollout_args = {
            key: value
            for key, value in asdict(self.task.rollout_args).items()
            if value is not None and key not in {"top_k", "logprobs", "n"}
        }
        rollout_args["n"] = self.repeat_times
        extra_body = {"thinking_token_budget": self.thinking_token_budget}
        if self.task.rollout_args.top_k >= 0:
            extra_body["top_k"] = self.task.rollout_args.top_k

        self.client.chat.completions.create(
            model=self.client.model_path,
            messages=self.messages,
            extra_body=extra_body,
            **rollout_args,
        )
        experiences = self.model.extract_experience_from_history()
        reasoning_end_token_ids = None
        if self.reasoning_end_str:
            reasoning_end_token_ids = self.model.tokenizer.encode(
                self.reasoning_end_str, add_special_tokens=False
            )
        for index, exp in enumerate(experiences):
            self._mask_forced_reasoning_end(
                exp,
                self.thinking_token_budget,
                reasoning_end_token_ids=reasoning_end_token_ids,
            )
            exp.eid.run = self.run_id_base + index
        return experiences
