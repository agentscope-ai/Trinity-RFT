# -*- coding: utf-8 -*-
"""On-Policy Distillation Workflow.

Reference: Tinker library's on-policy distillation implementation.

Algorithm:
1. Student samples trajectories (with logprobs)
2. Teacher computes logprobs on same trajectories
3. Store teacher_logprobs in experience.info["teacher_logprobs"]
4. Trainer's advantage_fn computes: advantages = teacher_logprobs - student_logprobs
5. Train with importance_sampling loss
"""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.qwen25_eval import verify_math_answer
from trinity.common.workflows.workflow import Task, Workflow


class OnPolicyDistillWorkflow(Workflow):
    """On-policy distillation workflow.

    Computes and stores teacher_logprobs in experience.info.
    The advantage_fn in trainer will compute:
        advantages = teacher_logprobs - student_logprobs

    Note: This workflow does NOT use reward_fn because:
    - Advantage is computed from teacher-student logprobs difference
    - No external reward signal is needed
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.reset(task)

        assert (
            self.auxiliary_model_wrappers is not None and len(self.auxiliary_model_wrappers) >= 1
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)

    def reset(self, task: Task):
        """Reset the workflow with a new task.

        Unlike BaseSimpleWorkflow, this does NOT require reward_fn.
        """
        self.task = task
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        """Format messages for the instruct model.

        Default format: system_prompt (optional) + task_desc + reply_prefix (optional)
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.task_desc})
        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})
        return messages

    def compute_reward(self, response: Experience) -> float:
        """Compute reward for a response.

        In base class, returns 0.0 as advantage is computed from teacher-student logprobs.
        Subclasses can override this to compute actual rewards.
        """
        return 0.0

    async def run_async(self) -> List[Experience]:
        messages = self.format_messages()

        # Step 1: Student samples trajectories
        responses = await self.model.chat_async(messages, **self.rollout_args)

        for i, response in enumerate(responses):
            # Step 2: Teacher computes logprobs
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )

            # Extract response portion
            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs
            assert student_resp_logprobs is not None, "Student logprobs should not be None."

            # Verify lengths match (they should be equal for the same token sequence)
            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
            )

            # Step 3: Store teacher_logprobs for advantage_fn
            response.teacher_logprobs = teacher_resp_logprobs

            # Initialize metrics
            if response.metrics is None:
                response.metrics = {}

            # Compute reward (subclasses can override compute_reward)
            response.reward = self.compute_reward(response)

            response.eid.run = i + self.run_id_base

            # KL divergence for monitoring
            kl = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            response.metrics["kl_divergence"] = kl

        return responses


class OnPolicyDistillMathWorkflow(OnPolicyDistillWorkflow):
    """On-policy distillation workflow with Qwen2.5-Math style format.

    This workflow:
    - Uses Qwen2.5-Math style prompt format (same as math_eval_workflow)
    - Computes accuracy using verify_math_answer as reward
    - Suitable for math reasoning tasks like GSM8K, MATH, etc.
    """

    def format_messages(self):
        """Format messages using Qwen2.5-Math style.

        System prompt: "You are a helpful assistant."
        User prompt: "{question}\nPlease reason step by step, and put your final answer within \\boxed{}."
        """
        system_prompt = "You are a helpful assistant."
        user_prompt = f"{self.task_desc}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def compute_reward(self, response: Experience) -> float:
        """Compute accuracy as reward using Qwen2.5-Math evaluation.

        Returns 1.0 if answer is correct, 0.0 otherwise.
        """
        if response.response_text and self.truth:
            accuracy, _ = verify_math_answer(
                response_text=response.response_text, ground_truth=self.truth
            )
            # Store accuracy in metrics
            if response.metrics is None:
                response.metrics = {}
            response.metrics["accuracy"] = accuracy
            return float(accuracy)
        return 0.0
