# -*- coding: utf-8 -*-
"""Prompt management for CoD Alchemy workflow using Jinja2 templates."""

from pathlib import Path
from typing import Dict, Optional

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
    current_round: int,
    max_rounds: int,
    inventory: Dict[str, int],
    target: str,
    action_feedback: Optional[str] = None,
    **kwargs,
) -> str:
    env = get_jinja_env()
    template = env.get_template("user.jinja2")
    return template.render(
        current_round=current_round,
        max_rounds=max_rounds,
        inventory=inventory,
        target=target,
        action_feedback=action_feedback,
        **kwargs,
    )
