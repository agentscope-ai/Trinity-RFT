# -*- coding: utf-8 -*-
"""
Goal checking for terminal tasks.

Uses a unified check-list approach: every task (single or composite) defines
a list of ``goal_checks``, each a dict like::

    {"type": "file_exists", "machine": "remote", "path": "...", "content": "..."}

All checks must pass for reward = 1.0; any failure gives 0.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from trinity.common.workflows.connect_the_dots.terminal.env import TerminalEnv
    from trinity.common.workflows.connect_the_dots.terminal.task_gen import TerminalTask


def check_goal(task: TerminalTask, env: TerminalEnv) -> float:
    """Return 1.0 if all goal checks pass, 0.0 otherwise."""
    for check in task.goal_checks:
        if not _evaluate_check(check, env):
            return 0.0
    return 1.0


def _evaluate_check(check: Dict[str, Any], env: TerminalEnv) -> bool:
    """Evaluate a single goal check."""
    machine = env.remote if check["machine"] == "remote" else env.local
    fs = machine.fs
    path = check["path"]
    check_type = check["type"]

    if check_type == "file_exists":
        if not fs.isfile(path):
            return False
        if "content" in check:
            try:
                actual = fs.readtext(path)
                if actual != check["content"]:
                    return False
            except Exception:
                return False
        return True

    elif check_type == "file_not_exists":
        return not fs.exists(path)

    elif check_type == "dir_exists":
        return fs.isdir(path)

    elif check_type == "permission_equals":
        if not fs.exists(path):
            return False
        meta = fs.get_meta(path)
        return meta.permissions == check["value"]

    elif check_type == "archive_contains":
        if not fs.isfile(path):
            return False
        meta = fs.get_meta(path)
        if meta.archive_type is None:
            return False
        expected_entries = check.get("entries", {})
        actual_entries = meta.archive_entries or {}
        for rel_path, expected_content in expected_entries.items():
            if rel_path not in actual_entries:
                return False
            if actual_entries[rel_path] != expected_content:
                return False
        return True

    else:
        raise ValueError(f"Unknown check type: {check_type}")
