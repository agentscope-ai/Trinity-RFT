# -*- coding: utf-8 -*-
"""Prompt management for CoD FrozenLake workflow using Jinja2 templates."""

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

# Get the directory where prompt templates are stored
PROMPTS_DIR = Path(__file__).parent


def get_jinja_env() -> Environment:
    """Get Jinja2 environment with template loader."""
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_system_prompt(**kwargs) -> str:
    """Load and render system prompt template.

    Args:
        **kwargs: Variables to pass to the template.

    Returns:
        Rendered system prompt string.
    """
    env = get_jinja_env()
    template = env.get_template("system.jinja2")
    return template.render(**kwargs)


def load_system_prompt_obscure(**kwargs) -> str:
    """Load and render obscure system prompt template (numeric actions).

    Args:
        **kwargs: Variables to pass to the template.

    Returns:
        Rendered system prompt string.
    """
    env = get_jinja_env()
    template = env.get_template("system_obscure.jinja2")
    return template.render(**kwargs)


def load_user_prompt(
    current_step: int,
    max_steps: int,
    observation: str,
    goal_row: int,
    goal_col: int,
    is_success: bool = False,
    **kwargs,
) -> str:
    """Load and render user prompt template.

    Args:
        current_step: Current step number (1-indexed for display).
        max_steps: Maximum number of steps.
        observation: Current map observation.
        is_success: Whether player has reached the goal.
        **kwargs: Additional variables to pass to the template.

    Returns:
        Rendered user prompt string.
    """
    env = get_jinja_env()
    template = env.get_template("user.jinja2")
    return template.render(
        current_step=current_step,
        max_steps=max_steps,
        observation=observation,
        goal_row=goal_row,
        goal_col=goal_col,
        is_success=is_success,
        **kwargs,
    )


__all__ = ["load_system_prompt", "load_system_prompt_obscure", "load_user_prompt", "PROMPTS_DIR"]
