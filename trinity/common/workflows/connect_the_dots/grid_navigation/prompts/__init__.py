"""Prompt loading helpers for the CoD grid-navigation workflow."""

from pathlib import Path
from typing import Optional, Tuple

from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent


def get_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_system_prompt(*, reveal_radius: int) -> str:
    """Render the system prompt with the configured observation radius."""
    return (
        get_jinja_env()
        .get_template("system.jinja2")
        .render(
            reveal_radius=reveal_radius,
        )
    )


def load_user_prompt(
    *,
    current_round: int,
    max_rounds: int,
    current_position: Tuple[int, int],
    goal_position: Tuple[int, int],
    observation: str,
    action_feedback: Optional[str],
) -> str:
    """Render one round's grid observation and task state."""
    return (
        get_jinja_env()
        .get_template("user.jinja2")
        .render(
            current_round=current_round,
            max_rounds=max_rounds,
            current_position=current_position,
            goal_position=goal_position,
            observation=observation,
            action_feedback=action_feedback,
        )
    )


__all__ = ["load_system_prompt", "load_user_prompt", "PROMPTS_DIR"]
