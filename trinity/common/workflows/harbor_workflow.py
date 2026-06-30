# -*- coding: utf-8 -*-
"""Base workflow for Harbor directory tasks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from trinity.common.workflows.workflow import Task, Workflow

if TYPE_CHECKING:
    from harbor.models.task.config import TaskConfig
    from harbor.models.task.paths import TaskPaths
    from harbor.viewer.task_scanner import TaskDefinitionScanner

    from trinity.common.models.model import ModelWrapper


class HarborWorkflow(Workflow):
    """Base workflow that loads a Harbor task directory during initialization.

    This class does not implement Harbor execution. It only bridges Trinity
    folder-style tasks to Harbor's own task parser, so concrete subclasses can
    focus on rollout, verification, and experience construction.
    """

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
        self.harbor_task_dir = self._get_harbor_task_dir(task)
        self.harbor_task_name = self.harbor_task_dir.name
        (
            self.harbor_scanner,
            self.harbor_task_paths,
            self.harbor_task_config,
            self.harbor_instruction,
            self.harbor_task_paths_info,
        ) = self._load_harbor_task(self.harbor_task_dir)

    def _get_harbor_task_dir(self, task: Task) -> Path:
        if task.raw_task is None:
            raise ValueError("HarborWorkflow requires `task.raw_task` to be configured.")

        task_dir = task.raw_task.get("task_dir")
        if task_dir is None:
            raise ValueError("HarborWorkflow requires `task.raw_task['task_dir']`.")

        task_dir_path = Path(task_dir).expanduser().resolve()
        if not task_dir_path.exists():
            raise FileNotFoundError(f"Harbor task directory does not exist: {task_dir_path}")
        if not task_dir_path.is_dir():
            raise ValueError(f"Harbor task path must be a directory: {task_dir_path}")
        return task_dir_path

    def _load_harbor_task(
        self,
        task_dir: Path,
    ) -> tuple["TaskDefinitionScanner", "TaskPaths", "TaskConfig", str | None, dict[str, bool]]:
        try:
            from harbor.models.task.paths import TaskPaths
            from harbor.viewer.task_scanner import TaskDefinitionScanner
        except ImportError as exc:
            raise ImportError(
                "HarborWorkflow requires the `harbor` package to be installed."
            ) from exc

        scanner = TaskDefinitionScanner(task_dir.parent)
        task_name = task_dir.name
        config = scanner.get_task_config(task_name)
        if config is None:
            raise ValueError(
                f"Failed to load Harbor task config from: {task_dir / TaskPaths.CONFIG_FILENAME}"
            )

        paths = TaskPaths(task_dir)
        instruction = scanner.get_instruction(task_name)
        paths_info = scanner.get_task_paths_info(task_name)
        return scanner, paths, config, instruction, paths_info
