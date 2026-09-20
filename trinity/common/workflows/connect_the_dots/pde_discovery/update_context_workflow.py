# -*- coding: utf-8 -*-
"""PDE-specific context update without exposing the training reward."""

from typing import List

from trinity.common.workflows.connect_the_dots.update_context_workflow import (
    AsyncCoDUpdateContextWorkflow,
)


class PDEUpdateContextWorkflow(AsyncCoDUpdateContextWorkflow):
    """Build the standard context-update prompt without its reward block."""

    def build_messages(self) -> List[dict]:
        messages = super().build_messages()
        user_prompt = messages[-1]["content"]
        feedback_block = (
            f"\n\nEnvironment feedback: {self.feedback}\n\n--- Your job ---"
        )
        before_feedback, feedback_marker, after_feedback = user_prompt.rpartition(
            feedback_block
        )
        before_reward, reward_marker, reward_text = before_feedback.rpartition(
            "\nReward: "
        )
        if (
            not feedback_marker
            or not reward_marker
            or not reward_text.strip()
            or "\n" in reward_text
        ):
            raise ValueError(
                "Expected reward block was not found in context-update prompt"
            )
        messages[-1]["content"] = (
            before_reward + feedback_marker + after_feedback
        )
        messages[-1]["content"] += """

Protocol reminder: after your brief reasoning, end with exactly this block:
--- Start of updated hints ---
- Your concise, transferable hints go here.
--- End of updated hints ---
Do not shorten the opening delimiter to "--- Updated hints ---", omit either delimiter, or write anything after the end delimiter.
"""
        return messages
