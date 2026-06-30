"""Directory-backed taskset reader."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.schema.formatter import TaskFormatter
from trinity.common.config import StorageConfig


class _TaskDirBatchReader:
    def __init__(
        self,
        task_dirs: list[Path],
        name: str,
        default_batch_size: int,
        total_epochs: int = 1,
        offset: int = 0,
        drop_last: bool = True,
        total_steps: Optional[int] = None,
    ):
        self.task_dirs = task_dirs
        self.dataset_size = len(task_dirs)
        if self.dataset_size == 0:
            raise ValueError(f"Task directory dataset [{name}] is empty and cannot be read.")
        self.name = name
        self.default_batch_size = default_batch_size
        self.drop_last = drop_last
        self.current_offset = offset

        if total_steps is not None:
            self.total_samples = default_batch_size * total_steps
        else:
            self.total_samples = self.dataset_size * total_epochs

    def _sample_for_index(self, index: int) -> dict:
        task_dir = self.task_dirs[index]
        task_name = task_dir.name
        return {
            "task_id": f"{self.name}:{index}:{task_name}",
            "task_name": task_name,
            "task_dir": str(task_dir),
            "taskset_name": self.name,
            "source_type": "task_dir",
        }

    def read_batch(self, batch_size: int) -> Tuple[List[dict], List[int]]:
        batch, indices = [], []
        while len(batch) < batch_size:
            if self.current_offset >= self.total_samples:
                if not self.drop_last and len(batch) > 0:
                    break
                raise StopIteration

            index = self.current_offset % self.dataset_size
            batch.append(self._sample_for_index(index))
            indices.append(index)
            self.current_offset += 1

        return batch, indices

    def select_batch(self, indices: List[int]) -> List[dict]:
        batch = []
        for i in indices:
            if not 0 <= i < self.dataset_size:
                raise IndexError(f"Task directory index {i} out of range.")
            if self.current_offset >= self.total_samples:
                if not self.drop_last and len(batch) > 0:
                    break
                raise StopIteration
            batch.append(self._sample_for_index(int(i)))
            self.current_offset += 1
        return batch


class TaskDirReader(BufferReader):
    """Read folder-style tasksets as Trinity workflow tasks.

    This reader is intentionally format-agnostic. It is useful for datasets where
    every task is represented by a directory, such as Harbor-style benchmark
    tasks. The workflow owns task parsing; the reader only provides task paths.
    """

    INDEX_FILENAME = "index.txt"

    def __init__(self, config: StorageConfig):
        self.config = config
        self.name = config.name
        self.read_batch_size = config.batch_size
        self.formatter = TaskFormatter(config)
        self.dataset = _TaskDirBatchReader(
            self._discover_task_dirs(config),
            name=config.name,
            default_batch_size=self.read_batch_size,
            total_epochs=config.total_epochs if not config.is_eval else 1,
            offset=config.index,
            drop_last=not config.is_eval,
            total_steps=config.total_steps if not config.is_eval else None,
        )
        self._init_selector(config)

    def _init_selector(self, config: StorageConfig) -> None:
        if config.data_selector is not None:
            from trinity.buffer.selector import SELECTORS
            from trinity.buffer.selector.selector import BaseSelector

            selector_cls = SELECTORS.get(config.data_selector.selector_type)
            self.selector: BaseSelector = selector_cls(self.dataset, config.data_selector)
        else:
            self.selector = None

    def _discover_task_dirs(self, config: StorageConfig) -> list[Path]:
        if config.path is None:
            raise ValueError("TaskDirReader requires `path` to be configured.")

        root = Path(config.path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Task directory dataset path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Task directory dataset path must be a directory: {root}")

        index_file = root / self.INDEX_FILENAME
        if index_file.exists():
            return self._discover_indexed_task_dirs(root, index_file)

        return sorted(
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )

    def _discover_indexed_task_dirs(self, root: Path, index_file: Path) -> list[Path]:
        task_dirs = []
        for line_number, line in enumerate(index_file.read_text().splitlines(), start=1):
            task_path = line.strip()
            if not task_path or task_path.startswith("#"):
                continue

            relative_path = Path(task_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Task directory index entry must stay under dataset root: "
                    f"{index_file}:{line_number}"
                )

            resolved_path = (root / relative_path).resolve()
            if not resolved_path.is_dir():
                raise FileNotFoundError(
                    f"Task directory index entry is not a directory: "
                    f"{index_file}:{line_number} -> {resolved_path}"
                )
            task_dirs.append(resolved_path)
        return task_dirs

    async def read(self, batch_size: Optional[int] = None, **kwargs):
        try:
            return self._read_sync(batch_size, **kwargs)
        except StopIteration as e:
            raise StopAsyncIteration from e

    def _read_sync(self, batch_size: Optional[int] = None, **kwargs):
        batch_size = batch_size or self.read_batch_size
        if self.selector is not None:
            indices = self.selector.get_indices(batch_size)
            samples = self.dataset.select_batch(indices)
        else:
            samples, indices = self.dataset.read_batch(batch_size)

        tasks = []
        for sample, index in zip(samples, indices):
            task = self.formatter.format(sample)
            task.index["index"] = int(index)
            task.index["task_name"] = sample["task_name"]
            task.index["task_dir"] = sample["task_dir"]
            tasks.append(task)
        return tasks

    def state_dict(self):
        if self.selector is not None:
            return self.selector.state_dict()
        return {"current_index": self.dataset.current_offset}

    def load_state_dict(self, state_dict):
        if self.selector is not None:
            self.selector.load_state_dict(state_dict)
        else:
            self.dataset.current_offset = state_dict["current_index"]

    def feedback(self, **pipeline_metrics):
        if self.selector is not None:
            self.selector.feedback(**pipeline_metrics)

    def __len__(self):
        return self.dataset.dataset_size
