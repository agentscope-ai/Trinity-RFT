# -*- coding: utf-8 -*-
"""Prompt management for CoD PDE discovery."""

from pathlib import Path

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


def load_user_prompt(**kwargs) -> str:
    env = get_jinja_env()
    template = env.get_template("user.jinja2")
    return template.render(**kwargs)
