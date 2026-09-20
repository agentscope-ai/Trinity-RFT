# -*- coding: utf-8 -*-
"""Prompt management for CoD Optimal Control workflow using Jinja2 templates."""

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


def load_system_prompt(
    control_penalty_coef: float = 0.03,
    enable_process_noise: bool = False,  #!!!
    process_noise_std: float = 0.0,  #!!!
    known_b_sign=None,
    **kwargs,
) -> str:
    """Load and render the system prompt template.

    Args:
        control_penalty_coef: Coefficient used in the control-effort term of
            the loss function shown to the agent.
        known_b_sign: Disclosed control direction, or ``None`` when hidden.
        **kwargs: Additional template variables.

    Returns:
        Rendered system prompt string.
    """
    env = get_jinja_env()
    template = env.get_template("system.jinja2")
    return template.render(
        control_penalty_coef=control_penalty_coef,
        enable_process_noise=enable_process_noise,  #!!!
        process_noise_std=process_noise_std,  #!!!
        known_b_sign=known_b_sign,
        **kwargs,
    )


def load_user_prompt(
    x0: float,
    v0: float,
    x_target: float,
    v_target: float,
    horizon: int,
    **kwargs,
) -> str:
    """Load and render the user prompt template for one optimal-control task.

    Args:
        x0: Initial position.
        v0: Initial velocity.
        x_target: Target position.
        v_target: Target velocity.
        horizon: Rollout horizon T.
        **kwargs: Additional template variables.

    Returns:
        Rendered user prompt string.
    """
    env = get_jinja_env()
    template = env.get_template("user.jinja2")
    return template.render(
        x0=x0,
        v0=v0,
        x_target=x_target,
        v_target=v_target,
        horizon=horizon,
        **kwargs,
    )


__all__ = ["load_system_prompt", "load_user_prompt", "PROMPTS_DIR"]
