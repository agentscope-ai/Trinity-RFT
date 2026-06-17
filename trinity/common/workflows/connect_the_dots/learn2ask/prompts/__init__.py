# -*- coding: utf-8 -*-
"""Prompt management for CoD Learn2Ask workflow using Jinja2 templates.

Two templates live here:

* system.jinja2 — rollout system prompt. A single template covers
  both training modes: the `<stop />` guideline is emitted when
  train_mode != "Ra", matching the original rollout_prompt_med /
  rollout_prompt_med_Ra split in
  examples/learn_to_ask/workflow/prompt_learn2ask.py byte-for-byte.

* reward_judge.jinja2 — judge system prompt fed to the auxiliary
  model. Holds a single placeholder, info_truth, that Trinity fills
  per-cid. Renders byte-identically to
  reward_prompt_med.format(info_truth).
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


PROMPTS_DIR = Path(__file__).parent


def get_jinja_env() -> Environment:
    """Get Jinja2 environment with template loader."""
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_system_prompt(train_mode: str = "Ra+Rs", **kwargs) -> str:
    """Render the rollout system prompt.

    Args:
        train_mode: "Ra+Rs" / "Rs" / "Ra" — "Ra" drops the <stop />
            instruction because the decision signal is not part of Ra's
            reward. All other values emit it.
        **kwargs: extra variables forwarded to the template.
    """
    env = get_jinja_env()
    template = env.get_template("system.jinja2")
    return template.render(train_mode=train_mode, **kwargs)


def load_reward_judge_prompt(info_truth: str, **kwargs) -> str:
    """Render the judge system prompt for the current cid.

    Args:
        info_truth: comma-separated structured symptom points extracted
            by data prep step 1/2 (same string the upstream Learn2Ask
            reward_fn passes to reward_prompt_med.format).
        **kwargs: extra variables forwarded to the template.
    """
    env = get_jinja_env()
    template = env.get_template("reward_judge.jinja2")
    return template.render(info_truth=info_truth, **kwargs)


__all__ = [
    "load_system_prompt",
    "load_reward_judge_prompt",
    "PROMPTS_DIR",
]
