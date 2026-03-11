"""Ray Queue storage"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
from sortedcontainers import SortedDict

from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.utils.log import get_logger


def is_database_url(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ["sqlite:///", "postgresql://", "mysql://"])


def is_json_file(path: str) -> bool:
    return path.endswith(".json") or path.endswith(".jsonl")


class PriorityFunction(ABC):
    """
    Each priority_fn,
        Args:
            item: List[Experience], assume that all experiences in it have the same model_version and use_count
            priority_fn_args: Dict, the arguments for priority_fn

        Returns:
            priority: float
            put_into_queue: bool, decide whether to put item into queue

    Note that put_into_queue takes effect both for new item from the explorer and for item sampled from the buffer.
    """

    @abstractmethod
    def __call__(self, item: List[Experience]) -> Tuple[float, bool]:
        """Calculate the priority of item."""

    @classmethod
    @abstractmethod
    def default_config(cls) -> Dict:
        """Return the default config."""


class LinearDecayPriority(PriorityFunction):
    """Calculate priority by linear decay.

    Priority is calculated as `model_version - decay * use_count. The item is always put back into the queue for reuse (as long as `reuse_cooldown_time` is not None).
    """

    def __init__(self, decay: float = 2.0):
        self.decay = decay

    def __call__(self, item: List[Experience]) -> Tuple[float, bool]:
        priority = float(item[0].info["model_version"] - self.decay * item[0].info["use_count"])
        put_into_queue = True
        return priority, put_into_queue

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "decay": 2.0,
        }


class LinearDecayUseCountControlPriority(PriorityFunction):
    """Calculate priority by linear decay, use count control, and randomization.

    Priority is calculated as `model_version - decay * use_count`; if `sigma` is non-zero, priority is further perturbed by random Gaussian noise with standard deviation `sigma`.  The item will be put back into the queue only if use count does not exceed `use_count_limit`.
    """

    def __init__(self, decay: float = 2.0, use_count_limit: int = 3, sigma: float = 0.0):
        self.decay = decay
        self.use_count_limit = use_count_limit
        self.sigma = sigma

    def __call__(self, item: List[Experience]) -> Tuple[float, bool]:
        priority = float(item[0].info["model_version"] - self.decay * item[0].info["use_count"])
        if self.sigma > 0.0:
            priority += float(np.random.randn() * self.sigma)
        put_into_queue = (
            item[0].info["use_count"] < self.use_count_limit if self.use_count_limit > 0 else True
        )
        return priority, put_into_queue

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "decay": 2.0,
            "use_count_limit": 3,
            "sigma": 0.0,
        }


class QueueBuffer(ABC):
    async def set_min_model_version(self, min_model_version: int):
        self.min_model_version = max(min_model_version, 0)

    @abstractmethod
    async def put(self, exps: List[Experience]) -> None:
        """Put a list of experiences into the queue."""

    @abstractmethod
    async def get(self) -> List[Experience]:
        """Get a list of experience from the queue."""

    @abstractmethod
    def qsize(self) -> int:
        """Get the current size of the queue."""

    @abstractmethod
    async def close(self) -> None:
        """Close the queue."""

    @abstractmethod
    def stopped(self) -> bool:
        """Check if there is no more data to read."""

    @classmethod
    def get_queue(cls, config: StorageConfig) -> "QueueBuffer":
        """Get a queue instance based on the storage configuration."""
        logger = get_logger(__name__)
        if config.replay_buffer.enable:
            capacity = config.capacity
            logger.info(
                f"Using AsyncPriorityQueue with capacity {capacity}, reuse_cooldown_time {config.replay_buffer.reuse_cooldown_time}."
            )
            return AsyncPriorityQueue(
                capacity=capacity,
                reuse_cooldown_time=config.replay_buffer.reuse_cooldown_time,
                priority_fn=config.replay_buffer.priority_fn,
                priority_fn_args=config.replay_buffer.priority_fn_args,
            )
        else:
            return AsyncQueue(capacity=config.capacity)


class AsyncQueue(asyncio.Queue, QueueBuffer):
    def __init__(self, capacity: int):
        """
        Initialize the async queue with a specified capacity.

        Args:
            capacity (`int`): The maximum number of items the queue can hold.
        """
        super().__init__(maxsize=capacity)
        self._closed = False
        self.min_model_version = 0

    async def put(self, item: List[Experience]):
        if len(item) == 0:
            return
        await super().put(item)

    async def get(self):
        while True:
            item = await super().get()
            if (
                self.min_model_version <= 0
                or item[0].info["model_version"] >= self.min_model_version
            ):
                return item

    async def close(self) -> None:
        """Close the queue."""
        self._closed = True
        for getter in self._getters:
            if not getter.done():
                getter.set_exception(StopAsyncIteration())
        self._getters.clear()

    def stopped(self) -> bool:
        """Check if there is no more data to read."""
        return self._closed and self.empty()


class AsyncPriorityQueue(QueueBuffer):
    """
    An asynchronous priority queue that manages a fixed-size buffer of experience items.
    Items are prioritized using a user-defined function and reinserted after a cooldown period.

    Attributes:
        capacity (int): Maximum number of items the queue can hold. This value is automatically
            adjusted to be at most twice the read batch size.
        reuse_cooldown_time (float): Delay before reusing an item (set to infinity to disable).
        priority_fn (callable): Function used to determine the priority of an item.
        priority_groups (SortedDict): Maps priorities to deques of items with the same priority.
    """

    def __init__(
        self,
        capacity: int,
        reuse_cooldown_time: Optional[float] = None,
        priority_fn: str = "linear_decay",
        priority_fn_args: Optional[dict] = None,
    ):
        """
        Initialize the async priority queue.

        Args:
            capacity (`int`): The maximum number of items the queue can store.
            reuse_cooldown_time (`float`): Time to wait before reusing an item. Set to None to disable reuse.
            priority_fn (`str`): Name of the function to use for determining item priority.
            kwargs: Additional keyword arguments for the priority function.
        """
        from trinity.buffer.storage import PRIORITY_FUNC

        self.capacity = capacity
        self.item_count = 0
        self.priority_groups = SortedDict()  # Maps priority -> deque of items
        priority_fn_cls = PRIORITY_FUNC.get(priority_fn)
        kwargs = priority_fn_cls.default_config()
        kwargs.update(priority_fn_args or {})
        self.priority_fn = priority_fn_cls(**kwargs)
        self.reuse_cooldown_time = reuse_cooldown_time
        self._condition = asyncio.Condition()  # For thread-safe operations
        self._closed = False
        self.min_model_version = 0

    async def _put(self, item: List[Experience], delay: float = 0) -> None:
        """
        Insert an item into the queue, replacing the lowest-priority item if full.

        Args:
            item (`List[Experience]`): A list of experiences to add.
            delay (`float`): Optional delay before insertion (for simulating timing behavior).
        """
        if delay > 0:
            await asyncio.sleep(delay)
        if len(item) == 0:
            return

        priority, put_into_queue = self.priority_fn(item=item)
        if not put_into_queue:
            return

        async with self._condition:
            if self.item_count == self.capacity:
                # If full, only insert if new item has higher or equal priority than the lowest
                lowest_priority, item_queue = self.priority_groups.peekitem(index=0)
                if lowest_priority > priority:
                    return  # Skip insertion if lower priority
                # Remove the lowest priority item
                item_queue.popleft()
                self.item_count -= 1
                if not item_queue:
                    self.priority_groups.popitem(index=0)

            # Add the new item
            if priority not in self.priority_groups:
                self.priority_groups[priority] = deque()
            self.priority_groups[priority].append(item)
            self.item_count += 1
            self._condition.notify()

    async def put(self, item: List[Experience]) -> None:
        await self._put(item, delay=0)

    async def get(self) -> List[Experience]:
        """
        Retrieve the highest-priority item from the queue.

        Returns:
            List[Experience]: The highest-priority item (list of experiences).

        Notes:
            - After retrieval, the item is optionally reinserted after a cooldown period.
        """
        async with self._condition:
            while True:
                while len(self.priority_groups) == 0:
                    if self._closed:
                        raise StopAsyncIteration()
                    await self._condition.wait()

                _, item_queue = self.priority_groups.peekitem(index=-1)
                item = item_queue.popleft()
                self.item_count -= 1
                if not item_queue:
                    self.priority_groups.popitem(index=-1)

                if (
                    self.min_model_version <= 0
                    or item[0].info["model_version"] >= self.min_model_version
                ):
                    break

        for exp in item:
            exp.info["use_count"] += 1
        # Optionally resubmit the item after a cooldown
        if self.reuse_cooldown_time is not None:
            asyncio.create_task(self._put(item, delay=self.reuse_cooldown_time))

        return item

    def qsize(self):
        return self.item_count

    async def close(self) -> None:
        """
        Close the queue.
        """
        async with self._condition:
            self._closed = True
            # No more items will be added, but existing items can still be processed.
            self.reuse_cooldown_time = None
            self._condition.notify_all()

    def stopped(self) -> bool:
        return self._closed and len(self.priority_groups) == 0


class QueueStorage:
    """An wrapper of a async queue."""

    def __init__(self, config: StorageConfig) -> None:
        self.logger = get_logger(f"queue_{config.name}", in_ray_actor=True)
        self.config = config
        self.capacity = config.capacity
        self.queue = QueueBuffer.get_queue(config)
        st_config = deepcopy(config)
        st_config.wrap_in_ray = False
        if st_config.path:
            if is_database_url(st_config.path):
                from trinity.buffer.writer.sql_writer import SQLWriter

                st_config.storage_type = StorageType.SQL.value
                self.writer = SQLWriter(st_config)
            elif is_json_file(st_config.path):
                from trinity.buffer.writer.file_writer import JSONWriter

                st_config.storage_type = StorageType.FILE.value
                self.writer = JSONWriter(st_config)
            else:
                self.logger.warning("Unknown supported storage path: %s", st_config.path)
                self.writer = None
        else:
            from trinity.buffer.writer.file_writer import JSONWriter

            st_config.storage_type = StorageType.FILE.value
            self.writer = JSONWriter(st_config)
        self.logger.warning(f"Save experiences in {st_config.path}.")
        self.ref_count = 0
        self.exp_pool = deque()  # A pool to store experiences
        self.closed = False

    async def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    async def release(self) -> int:
        """Release the queue."""
        self.ref_count -= 1
        if self.ref_count <= 0:
            await self.queue.close()
            if self.writer is not None:
                await self.writer.release()
        return self.ref_count

    def length(self) -> int:
        """The length of the queue."""
        return self.queue.qsize()

    async def put_batch(self, exp_bytes: bytes) -> None:
        """Put batch of experience."""
        exp_list = Experience.deserialize_many(exp_bytes)
        await self.queue.put(exp_list)
        if self.writer is not None:
            self.writer.write(exp_list)

    async def get_batch(self, batch_size: int, timeout: float, min_model_version: int = 0) -> bytes:
        """Get batch of experience."""
        await self.queue.set_min_model_version(min_model_version)
        start_time = time.time()
        result = []
        while len(result) < batch_size:
            while len(self.exp_pool) > 0 and len(result) < batch_size:
                exp = self.exp_pool.popleft()
                if min_model_version > 0 and exp.info["model_version"] < min_model_version:
                    continue
                result.append(exp)
            if len(result) >= batch_size:
                break

            if self.queue.stopped():
                # If the queue is stopped, ignore the rest of the experiences in the pool
                raise StopAsyncIteration("Queue is closed and no more items to get.")
            try:
                exp_list = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.exp_pool.extend(exp_list)
            except asyncio.TimeoutError:
                if time.time() - start_time > timeout:
                    self.logger.error(
                        f"Timeout when waiting for experience, only get {len(self.exp_pool)} experiences.\n"
                        "This phenomenon is usually caused by the workflow not returning enough "
                        "experiences or running timeout. Please check your workflow implementation."
                    )
                    batch = list(self.exp_pool)
                    self.exp_pool.clear()
                    return Experience.serialize_many(batch)
        return Experience.serialize_many(result)

    @classmethod
    def get_wrapper(cls, config: StorageConfig):
        """Get the queue actor."""
        return (
            ray.remote(cls)
            .options(
                name=f"queue-{config.name}",
                namespace=config.ray_namespace or ray.get_runtime_context().namespace,
                get_if_exists=True,
            )
            .remote(config)
        )
