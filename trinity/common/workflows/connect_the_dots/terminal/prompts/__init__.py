# -*- coding: utf-8 -*-
"""Prompt management for Terminal workflow using Jinja2 templates."""

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent


def get_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_system_prompt(**kwargs) -> str:
    env = get_jinja_env()
    template = env.get_template("system.jinja2")
    return template.render(**kwargs)


def load_user_prompt(
    current_step: int,
    max_steps: int,
    command_output: Optional[str] = None,
    terminal_prompt: Optional[str] = None,
    **kwargs,
) -> str:
    env = get_jinja_env()
    template = env.get_template("user.jinja2")
    return template.render(
        current_step=current_step,
        max_steps=max_steps,
        command_output=command_output,
        terminal_prompt=terminal_prompt,
        **kwargs,
    )
