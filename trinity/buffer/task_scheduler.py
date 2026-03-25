# -*- coding: utf-8 -*-
"""The taskset scheduler."""

from collections import Counter
from copy import deepcopy
from typing import Dict, List

import numpy as np

from trinity.buffer.buffer import get_buffer_reader
from trinity.common.config import Config
from trinity.common.constants import SELECTOR_METRIC
from trinity.utils.annotations import Experimental


def get_taskset_scheduler(explorer_state: Dict, config: Config) -> "TasksetSchedulerBase":
    """Get a taskset scheduler according to the config.

    Args:
        explorer_state (Dict): Restoration state from checkpoint (may include progress info)
        config (Config): Full system configuration containing buffer and taskset settings

    Returns:
        TasksetSchedulerBase: The taskset scheduler instance
    """
    taskset_configs = config.buffer.explorer_input.tasksets
    if len(taskset_configs) == 1 and taskset_configs[0].data_selector.selector_type == "sequential":
        return SimpleTasksetScheduler(explorer_state, config)
    else:
        return TasksetScheduler(explorer_state, config)


class TasksetSchedulerBase:
    def __init__(self, explorer_state: Dict, config: Config):
        self.config = config
        self.explorer_state = explorer_state

    async def read_async(self) -> List:
        """Asynchronously reads a batch of tasks according to the current schedule."""
        raise NotImplementedError

    def state_dict(self) -> List[Dict]:
        """return persistent state for checkpointing.

        Returns:
            List[Dict]: State dicts for all selectors (one per taskset)
        """
        raise NotImplementedError

    def feedback(self, pipeline_metrics: Dict) -> None:
        """Update selectors using feedback from the training pipeline."""
        raise NotImplementedError


class SimpleTasksetScheduler(TasksetSchedulerBase):
    """
    A simple taskset scheduler that only reads from one taskset without task selection strategies.
    """

    def __init__(self, explorer_state: Dict, config: Config):
        super().__init__(explorer_state, config)
        if "latest_task_index" in self.explorer_state:
            self.explorer_state["taskset_states"] = [
                {
                    "current_index": explorer_state["latest_task_index"],
                }
            ]
        index = self.explorer_state.get("taskset_states", [{"current_index": 0}])[0].get(
            "current_index", 0
        )
        taskset_config = deepcopy(self.config.buffer.explorer_input.tasksets[0])
        taskset_config.index = index
        taskset_config.data_selector = None  # disable selection
        self.reader = get_buffer_reader(taskset_config)

    async def read_async(self) -> List:
        return await self.reader.read_async()

    def state_dict(self) -> List[Dict]:
        return [self.reader.state_dict()]

    def feedback(self, pipeline_metrics: Dict) -> None:
        # do nothing here
        return


@Experimental
class TasksetScheduler(TasksetSchedulerBase):
    """
    Coordinates multiple datasets (tasksets) with customizable task selection strategies per taskset.

    The scheduler:
      - Manages multiple data sources (tasksets)
      - Uses a selector per taskset to determine which samples to read
      - Shuffles the order of taskset access across epochs
      - Supports adaptive selectors via feedback (e.g., difficulty-based sampling)
      - Enables curriculum-like or interleaved multi-task training

    It assumes that each call to `read_async()` corresponds to one training step,
    and batches are built by aggregating samples from different tasksets based on
    a shuffled global schedule.
    """

    def __init__(self, explorer_state: Dict, config: Config):
        """
        Initialize the scheduler from configuration and previous state (for resume support).

        Args:
            explorer_state (Dict): Restoration state from checkpoint (may include progress info)
            config (Config): Full system configuration containing buffer and taskset settings
        """
        super().__init__(explorer_state, config)

        # Backward compatibility: old format stored 'latest_task_index' directly
        if "latest_task_index" in explorer_state:
            assert len(config.buffer.explorer_input.tasksets) == 1  # old format
            explorer_state["taskset_states"] = [
                {
                    "current_index": explorer_state["latest_task_index"],
                }
            ]

        self.read_batch_size = config.buffer.batch_size
        taskset_configs = config.buffer.explorer_input.tasksets

        from trinity.buffer.reader.file_reader import FileReader

        taskset_states = explorer_state.get(
            "taskset_states", [{"current_index": 0}] * len(taskset_configs)
        )
        self.tasksets = []
        for taskset_config, taskset_state in zip(taskset_configs, taskset_states):
            assert not taskset_config.is_eval  # assume drop last
            taskset = get_buffer_reader(taskset_config)
            if not isinstance(taskset, FileReader):
                raise TypeError(
                    f"Taskset '{taskset_config.name}' has an unsupported type '{type(taskset).__name__}'."
                    f"Currently, only 'FileReader' is supported by TasksetScheduler."
                )
            taskset.load_state_dict(taskset_state)  # Restore any prior state
            self.tasksets.append(taskset)

        # Each explorer step calls read_async once → track step globally
        self.step = explorer_state.get("latest_iteration", 0)

        # Build flat list indicating how often each taskset should appear per epoch
        self.base_taskset_ids = []
        for i, taskset in enumerate(self.tasksets):
            self.base_taskset_ids.extend([i] * len(taskset))
        if len(self.base_taskset_ids) == 0:
            raise ValueError("Empty tasksets provided!")

        self.epoch = self.step * self.read_batch_size // len(self.base_taskset_ids)
        self.orders = self.build_orders(self.epoch)

        if self.config.buffer.total_steps:
            self.max_steps = self.config.buffer.total_steps
        else:
            self.max_steps = (
                self.config.buffer.total_epochs * len(self.base_taskset_ids) // self.read_batch_size
            )

    def build_orders(self, epoch: int):
        """
        Creates a shuffled sequence of taskset IDs to control sampling priority per step.

        At the start of each epoch, all tasksets are shuffled proportionally to their size,
        ensuring balanced exposure while introducing randomness in selection order.

        Args:
            epoch (int): Epoch ID used as seed for deterministic shuffling

        Returns:
            List[int]: Sequence of taskset IDs, length = steps_per_epoch * batch_size
        """
        taskset_ids = self.base_taskset_ids.copy()
        rng = np.random.default_rng(epoch)
        rng.shuffle(taskset_ids)
        return taskset_ids

    def _should_stop(self) -> bool:
        return self.step >= self.max_steps

    async def read_async(self) -> List:
        """
        Asynchronously reads a batch of tasks according to the current schedule.

        For each step:
          - Checks if a new epoch has started; rebuilds order if so
          - Determines which tasksets contribute to this batch
          - Uses each taskset's selector to pick specific samples
          - Annotates each task with its source taskset_id
          - Returns combined list of tasks

        Raises:
            StopAsyncIteration: When total_epochs is reached

        Returns:
            List[Task]: A batch of tasks from potentially multiple tasksets
        """
        if self._should_stop():
            raise StopAsyncIteration

        batch_size = self.read_batch_size
        start = self.step * batch_size % len(self.base_taskset_ids)
        end = start + batch_size
        if end <= len(self.base_taskset_ids):
            taskset_ids = self.orders[start:end]
            if end == len(self.base_taskset_ids):
                self.epoch += 1
                self.orders = self.build_orders(self.epoch)
        else:
            taskset_ids = self.orders[start:]
            self.epoch += 1
            self.orders = self.build_orders(self.epoch)
            taskset_ids += self.orders[: (end - len(self.base_taskset_ids))]

        counter = Counter(taskset_ids)
        batch = []
        for taskset_id, count in counter.items():
            tasks = await self.tasksets[taskset_id].read_async(batch_size=count)
            # Annotate each task with its origin
            for task in tasks:
                task.index["taskset_id"] = taskset_id
            batch.extend(tasks)

        self.step += 1
        return batch

    def state_dict(self) -> List[Dict]:
        """
        Save persistent state for checkpointing.

        Returns:
            List[Dict]: State dicts for all tasksets
        """
        return [taskset.state_dict() for taskset in self.tasksets]

    def feedback(self, pipeline_metrics: Dict) -> None:
        """
        Update selectors in tasksets using feedback from the training pipeline.

        Expected format:
            pipeline_metrics = {
                SELECTOR_METRIC: {
                    0: {"indices": [...], "values": [...]},
                    1: {"indices": [...], "values": [...]}
                },
                ...  # other metrics
            }

        This allows adaptive selectors (like `DifficultyBasedSelector`) to refine difficulty estimates.

        Args:
            pipeline_metrics (Dict): Metrics dictionary passed from explorer.
        """
        if SELECTOR_METRIC not in pipeline_metrics:
            return
        selector_metric = pipeline_metrics.pop(SELECTOR_METRIC, {})
        for taskset_id, taskset_kwargs in selector_metric.items():
            taskset = self.tasksets[taskset_id]
            taskset.feedback(**taskset_kwargs)
