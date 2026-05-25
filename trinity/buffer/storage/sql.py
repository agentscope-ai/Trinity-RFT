"""SQL database storage"""

import os
import time
from abc import abstractmethod
from typing import Dict, List, Optional

import ray
from datasets import Dataset
from sqlalchemy import asc, desc
from sqlalchemy.orm import sessionmaker

from trinity.buffer.schema import FORMATTER, init_engine
from trinity.buffer.schema.formatter import TaskFormatter
from trinity.buffer.utils import retry_session
from trinity.common.config import StorageConfig
from trinity.common.experience import Experience
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows import WORKFLOWS, Task
from trinity.utils.log import get_logger


class SQLStorage:
    """
    An Storage based on SQL Database.

    If `wrap_in_ray` in `StorageConfig` is `True`, this class will be run as a Ray Actor,
    and provide a remote interface to the local database.

    For databases that do not support multi-processing read/write (e.g. sqlite, duckdb), please
    set `wrap_in_ray` to `True`.
    """

    def __init__(self, config: StorageConfig) -> None:
        self.logger = get_logger(f"sql_{config.name}", in_ray_actor=True)
        if not config.path:
            raise ValueError("`path` is required for SQL storage type.")
        self.logger.info(
            f"Init engine {config.path} with table {config.name} with schema {config.schema_type}"
        )
        self.engine, self.table_model_cls = init_engine(
            db_url=config.path,
            table_name=config.name,
            schema_type=config.schema_type,
        )
        self.logger.info(f"Init SQL storage at {config.path}")
        self.session = sessionmaker(bind=self.engine)
        self.max_retry_times = config.max_retry_times
        self.max_retry_interval = config.max_retry_interval
        self.ref_count = 0
        self.stopped = False
        # Assume that the auto-increment ID starts counting from 1, so the default offset should be 0.
        self.offset = config.index

    @classmethod
    def get_wrapper(cls, config: StorageConfig):
        if config.schema_type is None:
            storage_cls = SQLTaskStorage
        else:
            storage_cls = SQLExperienceStorage
        if config.wrap_in_ray:
            return (
                ray.remote(storage_cls)
                .options(
                    name=f"sql-{config.name}",
                    namespace=config.ray_namespace or ray.get_runtime_context().namespace,
                    get_if_exists=True,
                    max_concurrency=5,
                )
                .remote(config)
            )
        else:
            return storage_cls(config)

    @abstractmethod
    def write(self, data: List) -> None:
        """Write a batch of data."""

    @abstractmethod
    def read(self, batch_size: Optional[int] = None) -> List:
        """Read a batch of data."""

    def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    def release(self) -> int:
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.stopped = True
        return self.ref_count


class SQLExperienceStorage(SQLStorage):
    """Used as trainer input."""

    def __init__(self, config: StorageConfig) -> None:
        super().__init__(config)
        self.max_timeout = config.max_read_timeout
        self.batch_size = config.batch_size
        self.enable_replay = config.replay_buffer is not None and config.replay_buffer.enable
        self.max_experience_bytes = int(
            os.getenv("TRINITY_SQL_MAX_EXPERIENCE_BYTES", str(32 * 1024 * 1024))
        )
        # TODO: optimize the following logic
        if config.schema_type == "experience":
            # NOTE: consistent with the old version of experience buffer
            self._read_method = self._read_priority
        else:
            # SFT / DPO uses FIFO style
            self._read_method = self._read_fifo

    def write(self, data: List[Experience]) -> None:
        experience_models = []
        skipped = 0
        for exp in data:
            model = self.table_model_cls.from_experience(exp)
            payload = model.experience_bytes
            if (
                self.max_experience_bytes > 0
                and payload is not None
                and len(payload) > self.max_experience_bytes
            ):
                skipped += 1
                self.logger.warning(
                    "Skip oversized experience: size=%s bytes > limit=%s bytes (task_id=%s run_id=%s msg_id=%s)",
                    len(payload),
                    self.max_experience_bytes,
                    str(exp.eid.task),
                    exp.eid.run,
                    str(exp.eid.suffix),
                )
                continue
            experience_models.append(model)

        if not experience_models:
            self.logger.warning(
                "Skip SQL write: no valid experiences in batch. total=%s skipped=%s limit=%s",
                len(data),
                skipped,
                self.max_experience_bytes,
            )
            return

        with retry_session(self.session, self.max_retry_times, self.max_retry_interval) as session:
            session.add_all(experience_models)
        self.logger.info(
            "Write %s experiences to SQL storage (input=%s skipped=%s limit=%s).",
            len(experience_models),
            len(data),
            skipped,
            self.max_experience_bytes,
        )

    def _read_fifo(self, batch_size: int) -> List[Experience]:
        """Read experiences in FIFO order."""
        exp_list = []
        start_time = time.time()
        while len(exp_list) < batch_size:
            if self.stopped:
                raise StopIteration()
            if time.time() - start_time > self.max_timeout:
                self.logger.warning(
                    f"Max read timeout reached ({self.max_timeout} s), only get {len(exp_list)} experiences, stopping..."
                )
                raise StopIteration()
            with retry_session(
                self.session, self.max_retry_times, self.max_retry_interval
            ) as session:
                # get a batch of experiences from the database
                experiences = (
                    session.query(self.table_model_cls)
                    .filter(self.table_model_cls.id > self.offset)
                    .order_by(asc(self.table_model_cls.id))
                    .limit(batch_size - len(exp_list))
                    .all()
                )
                if experiences:
                    self.offset = experiences[-1].id
                    start_time = time.time()
                exp_list.extend([self.table_model_cls.to_experience(exp) for exp in experiences])
            if len(exp_list) < batch_size:
                self.logger.info(f"Waiting for {batch_size - len(exp_list)} more experiences...")
                time.sleep(1)
        return exp_list

    def _read_priority(self, batch_size: int, min_model_version: int = 0) -> List[Experience]:
        exp_list = []
        start_time = time.time()
        latest_size = 0
        while latest_size < batch_size:
            if self.stopped:
                raise StopIteration()
            if time.time() - start_time > self.max_timeout:
                self.logger.warning(
                    f"Max read timeout reached ({self.max_timeout} s), only get {latest_size} experiences, stopping..."
                )
                raise StopIteration()
            with retry_session(
                self.session, self.max_retry_times, self.max_retry_interval
            ) as session:
                query = session.query(self.table_model_cls)
                if min_model_version > 0:
                    query = query.filter(self.table_model_cls.model_version >= min_model_version)
                if not self.enable_replay:
                    query = query.filter(self.table_model_cls.consumed == 0)
                experiences = (
                    query.order_by(
                        asc(self.table_model_cls.consumed), desc(self.table_model_cls.id)
                    )
                    .limit(batch_size)
                    .with_for_update()
                    .all()
                )
                if len(experiences) != batch_size:
                    if latest_size != len(experiences):
                        latest_size = len(experiences)
                        start_time = time.time()
                else:
                    ids = [exp.id for exp in experiences]
                    session.query(self.table_model_cls).filter(
                        self.table_model_cls.id.in_(ids)
                    ).update(
                        {self.table_model_cls.consumed: self.table_model_cls.consumed + 1},
                        synchronize_session=False,
                    )
                    exp_list.extend(
                        [self.table_model_cls.to_experience(exp) for exp in experiences]
                    )
                    break

            self.logger.info(f"Waiting for {batch_size - len(exp_list)} more experiences...")
            time.sleep(1)
        return exp_list

    def read(self, batch_size: Optional[int] = None, **kwargs) -> List[Experience]:
        if self.stopped:
            raise StopIteration()

        batch_size = self.batch_size if batch_size is None else batch_size
        return self._read_method(batch_size, **kwargs)

    @classmethod
    def load_from_dataset(cls, dataset: Dataset, config: StorageConfig) -> "SQLExperienceStorage":
        storage = cls(config)
        formatter = FORMATTER.get(config.schema_type)(
            tokenizer_path=config.tokenizer_path, format_config=config.format
        )
        batch_size = storage.batch_size
        batch = []
        for item in dataset:
            batch.append(formatter.format(item))
            if len(batch) >= batch_size:
                storage.write(batch)
                batch.clear()
        if batch:
            storage.write(batch)
        return storage


class SQLTaskStorage(SQLStorage):
    """Used as explorer input."""

    def __init__(self, config: StorageConfig) -> None:
        super().__init__(config)
        self.batch_size = config.batch_size
        self.is_eval = config.is_eval
        self.default_workflow_cls = WORKFLOWS.get(config.default_workflow_type)  # type: ignore
        self.default_reward_fn_cls = REWARD_FUNCTIONS.get(config.default_reward_fn_type)  # type: ignore
        self.formatter = TaskFormatter(config)
        self.offset = config.index
        if config.total_steps:
            self.total_samples = self.batch_size * config.total_steps
        else:
            if config.total_epochs > 1:
                self.logger.warning(
                    f"SQL Storage do not support total_epochs, the value {config.total_epochs} will be ignored"
                )
            self.total_samples = float("inf")

    def write(self, data: List[Dict]) -> None:
        with retry_session(self.session, self.max_retry_times, self.max_retry_interval) as session:
            tasks = [self.table_model_cls.from_dict(item) for item in data]
            session.add_all(tasks)

    def read(self, batch_size: Optional[int] = None) -> List[Task]:
        if self.stopped:
            raise StopIteration()
        if self.offset > self.total_samples:
            raise StopIteration()
        batch_size = self.batch_size if batch_size is None else batch_size
        with retry_session(self.session, self.max_retry_times, self.max_retry_interval) as session:
            query = (
                session.query(self.table_model_cls)
                .filter(self.table_model_cls.id > self.offset)
                .order_by(asc(self.table_model_cls.id))
                .limit(batch_size)
            )
            results = query.all()
            if len(results) == 0:
                raise StopIteration()
            if not self.is_eval and len(results) < batch_size:
                raise StopIteration()
            self.offset = results[-1].id
            return [self.formatter.format(item.raw_task) for item in results]

    @classmethod
    def load_from_dataset(cls, dataset: Dataset, config: StorageConfig) -> "SQLTaskStorage":
        storage = cls(config)
        batch_size = config.batch_size
        batch = []
        for item in dataset:
            batch.append(item)
            if len(batch) >= batch_size:
                storage.write(batch)
                batch.clear()
        if batch:
            storage.write(batch)
        return storage
