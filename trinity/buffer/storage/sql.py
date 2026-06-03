"""SQL database storage for experience and task buffers.

Primary classes (async, used by Ray actors and production hot paths):
    SQLExperienceStorage, SQLTaskStorage

Factory:
    SQLStorage.get_wrapper(config) — returns Ray actor handle.
"""

import asyncio
import os
import time
from typing import Dict, List, Optional

import ray
from datasets import Dataset
from sqlalchemy import and_, asc, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from trinity.buffer.schema import FORMATTER
from trinity.buffer.schema.sql_schema import init_async_engine
from trinity.buffer.utils import async_run_with_retry_session
from trinity.common.config import StorageConfig
from trinity.common.constants import MAX_EXP_BYTES_ENV_VAR
from trinity.common.experience import Experience
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows import WORKFLOWS, Task
from trinity.utils.log import get_logger

# ---------------------------------------------------------------------------
# Async primary implementations
# ---------------------------------------------------------------------------


class SQLExperienceStorage:
    """Primary async SQL storage for experiences. Used directly as Ray actor."""

    def __init__(self, config: StorageConfig) -> None:
        self.logger = get_logger(f"sql_{config.name}")
        self.config = config
        self.max_timeout = config.max_read_timeout
        self.batch_size = config.batch_size
        self.enable_replay = config.replay_buffer is not None and config.replay_buffer.enable
        self.max_experience_bytes = int(os.getenv(MAX_EXP_BYTES_ENV_VAR, 1024 * 1024 * 32))
        self.max_retry_times = config.max_retry_times
        self.max_retry_interval = config.max_retry_interval
        self.ref_count = 0
        self.stopped = False
        self.offset = config.index
        self._initialized = False

        if config.schema_type == "experience":
            self._read_method = self._read_priority
        else:
            self._read_method = self._read_fifo

    async def prepare(self) -> None:
        """Initialize async engine and create tables."""
        if self._initialized:
            return
        result = await init_async_engine(
            self.config.path, self.config.name, self.config.schema_type  # type: ignore
        )
        self.engine, self.table_model_cls, self.blob_model_cls = result
        self.session = async_sessionmaker(self.engine, expire_on_commit=False)
        self._initialized = True
        self.logger.info(f"SQL storage initialized at {self.config.path}")

    async def write(self, data: List[Experience]) -> None:
        await self.prepare()

        async def operation(session: AsyncSession):
            for exp in data:
                exp_bytes = exp.serialize()
                if (
                    self.max_experience_bytes > 0
                    and exp_bytes is not None
                    and len(exp_bytes) > self.max_experience_bytes
                ):
                    self.logger.warning(
                        f"Experience size {len(exp_bytes)} bytes exceeds "
                        f"max_experience_bytes {self.max_experience_bytes}, skipping."
                    )
                    continue
                meta_row = self.table_model_cls.from_experience(exp)
                session.add(meta_row)
                await session.flush()
                blob_row = self.blob_model_cls(id=meta_row.id, experience_bytes=exp_bytes)
                session.add(blob_row)

        await async_run_with_retry_session(
            self.session, operation, self.max_retry_times, self.max_retry_interval
        )
        self.logger.info(f"Write {len(data)} experiences to SQL storage.")

    async def _fetch_blobs(self, session: AsyncSession, ids: List[int]) -> Dict[int, bytes]:
        stmt = select(self.blob_model_cls).where(self.blob_model_cls.id.in_(ids))
        result = await session.execute(stmt)
        blobs = result.scalars().all()
        return {b.id: b.experience_bytes for b in blobs}

    def _assemble_experiences(self, meta_rows, blob_map: Dict[int, bytes]) -> List[Experience]:
        experiences = []
        for row in meta_rows:
            blob_bytes = blob_map.get(row.id)
            if blob_bytes is None:
                self.logger.warning(f"Missing blob for experience id={row.id}, skipping.")
                continue
            experiences.append(row.to_experience(blob_bytes))
        return experiences

    async def _read_fifo(self, batch_size: int) -> List[Experience]:
        exp_list = []
        start_time = time.time()
        while len(exp_list) < batch_size:
            if self.stopped:
                raise StopAsyncIteration()
            if time.time() - start_time > self.max_timeout:
                self.logger.warning(
                    f"Max read timeout reached ({self.max_timeout} s), "
                    f"only got {len(exp_list)} experiences, stopping..."
                )
                raise StopAsyncIteration()

            current_offset = self.offset
            remaining = batch_size - len(exp_list)

            async def operation(session: AsyncSession):
                stmt = (
                    select(self.table_model_cls)
                    .where(self.table_model_cls.id > current_offset)
                    .order_by(asc(self.table_model_cls.id))
                    .limit(remaining)
                )
                result = await session.execute(stmt)
                meta_rows = result.scalars().all()
                if not meta_rows:
                    return [], None
                ids = [row.id for row in meta_rows]
                blob_map = await self._fetch_blobs(session, ids)
                return (
                    self._assemble_experiences(meta_rows, blob_map),
                    meta_rows[-1].id,
                )

            experiences, next_offset = await async_run_with_retry_session(
                self.session, operation, self.max_retry_times, self.max_retry_interval
            )
            if next_offset is not None:
                self.offset = next_offset
                start_time = time.time()
            exp_list.extend(experiences)
            if len(exp_list) < batch_size:
                self.logger.info(f"Waiting for {batch_size - len(exp_list)} more experiences...")
                await asyncio.sleep(1)
        return exp_list

    async def _read_priority(self, batch_size: int, min_model_version: int = 0) -> List[Experience]:
        exp_list = []
        start_time = time.time()
        latest_size = 0
        while latest_size < batch_size:
            if self.stopped:
                raise StopAsyncIteration()
            if time.time() - start_time > self.max_timeout:
                self.logger.warning(
                    f"Max read timeout reached ({self.max_timeout} s), "
                    f"only got {latest_size} experiences, stopping..."
                )
                raise StopAsyncIteration()

            enable_replay = self.enable_replay
            table_cls = self.table_model_cls
            is_sqlite = self.engine.dialect.name == "sqlite"

            async def operation(session: AsyncSession):
                stmt = select(table_cls)
                if min_model_version > 0:
                    stmt = stmt.where(table_cls.model_version >= min_model_version)
                if not enable_replay:
                    stmt = stmt.where(table_cls.consumed == 0)
                stmt = stmt.order_by(asc(table_cls.consumed), desc(table_cls.id)).limit(batch_size)

                if not is_sqlite:
                    stmt = stmt.with_for_update()

                result = await session.execute(stmt)
                meta_rows = result.scalars().all()

                if len(meta_rows) != batch_size:
                    return len(meta_rows), False, []

                ids = [row.id for row in meta_rows]
                update_stmt = (
                    update(table_cls)
                    .where(table_cls.id.in_(ids))
                    .values(consumed=table_cls.consumed + 1)
                )
                await session.execute(update_stmt)

                blob_map = await self._fetch_blobs(session, ids)
                return (
                    len(meta_rows),
                    True,
                    self._assemble_experiences(meta_rows, blob_map),
                )

            latest_batch_size, has_full_batch, experiences = await async_run_with_retry_session(
                self.session, operation, self.max_retry_times, self.max_retry_interval
            )
            if not has_full_batch:
                if latest_size != latest_batch_size:
                    latest_size = latest_batch_size
                    start_time = time.time()
            else:
                exp_list.extend(experiences)
                break

            self.logger.info(f"Waiting for {batch_size - len(exp_list)} more experiences...")
            await asyncio.sleep(1)
        return exp_list

    async def read(self, batch_size: Optional[int] = None, **kwargs) -> List[Experience]:
        await self.prepare()
        if self.stopped:
            raise StopAsyncIteration()
        batch_size = self.batch_size if batch_size is None else batch_size
        return await self._read_method(batch_size, **kwargs)

    def _build_filter_conditions(self, filters: Optional[Dict] = None):
        conditions = []
        if not filters:
            return conditions
        if filters.get("reward_min") is not None:
            conditions.append(self.table_model_cls.reward >= filters["reward_min"])
        if filters.get("reward_max") is not None:
            conditions.append(self.table_model_cls.reward <= filters["reward_max"])
        if filters.get("model_version_min") is not None:
            conditions.append(self.table_model_cls.model_version >= filters["model_version_min"])
        if filters.get("model_version_max") is not None:
            conditions.append(self.table_model_cls.model_version <= filters["model_version_max"])
        if filters.get("task_id"):
            conditions.append(self.table_model_cls.task_id == filters["task_id"])
        return conditions

    async def count(self, filters: Optional[Dict] = None) -> int:
        await self.prepare()

        async def operation(session: AsyncSession):
            stmt = select(func.count()).select_from(self.table_model_cls)
            conditions = self._build_filter_conditions(filters)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            result = await session.execute(stmt)
            return result.scalar()

        return await async_run_with_retry_session(
            self.session, operation, self.max_retry_times, self.max_retry_interval
        )

    async def query(
        self, offset: int = 0, limit: int = 10, filters: Optional[Dict] = None
    ) -> List[Experience]:
        await self.prepare()

        async def operation(session: AsyncSession):
            stmt = select(self.table_model_cls)
            conditions = self._build_filter_conditions(filters)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.offset(offset).limit(limit)
            result = await session.execute(stmt)
            meta_rows = result.scalars().all()
            if not meta_rows:
                return []
            ids = [row.id for row in meta_rows]
            blob_map = await self._fetch_blobs(session, ids)
            return self._assemble_experiences(meta_rows, blob_map)

        return await async_run_with_retry_session(
            self.session, operation, self.max_retry_times, self.max_retry_interval
        )

    @classmethod
    async def load_from_dataset(
        cls, dataset: Dataset, config: StorageConfig
    ) -> "SQLExperienceStorage":
        storage = cls(config)
        await storage.prepare()
        formatter = FORMATTER.get(config.schema_type)(
            tokenizer_path=config.tokenizer_path, format_config=config.format
        )
        batch_size = storage.batch_size
        batch = []
        for item in dataset:
            batch.append(formatter.format(item))
            if len(batch) >= batch_size:
                await storage.write(batch)
                batch.clear()
        if batch:
            await storage.write(batch)
        return storage

    def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    def release(self) -> int:
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.stopped = True
        return self.ref_count


class SQLTaskStorage:
    """Primary async SQL storage for tasks. Used directly as Ray actor."""

    def __init__(self, config: StorageConfig) -> None:
        self.logger = get_logger(f"sql_{config.name}")
        self.config = config
        self.batch_size = config.batch_size
        self.is_eval = config.is_eval
        self.max_retry_times = config.max_retry_times
        self.max_retry_interval = config.max_retry_interval
        self.ref_count = 0
        self.stopped = False
        self.offset = config.index
        self._initialized = False

        if config.total_steps:
            self.total_samples = self.batch_size * config.total_steps
        else:
            self.total_samples = float("inf")

    async def prepare(self) -> None:
        """Initialize async engine and create tables."""
        if self._initialized:
            return
        from trinity.buffer.schema.formatter import TaskFormatter

        result = await init_async_engine(
            self.config.path, self.config.name, self.config.schema_type  # type: ignore
        )
        self.engine, self.table_model_cls = result
        self.session = async_sessionmaker(self.engine, expire_on_commit=False)
        self.default_workflow_cls = WORKFLOWS.get(self.config.default_workflow_type)
        self.default_reward_fn_cls = REWARD_FUNCTIONS.get(self.config.default_reward_fn_type)
        self.formatter = TaskFormatter(self.config)
        self._initialized = True
        self.logger.info(f"SQL task storage initialized at {self.config.path}")

    async def write(self, data: List[Dict]) -> None:
        await self.prepare()

        async def operation(session: AsyncSession):
            tasks = [self.table_model_cls.from_dict(item) for item in data]
            session.add_all(tasks)

        await async_run_with_retry_session(
            self.session, operation, self.max_retry_times, self.max_retry_interval
        )

    async def read(self, batch_size: Optional[int] = None) -> List[Task]:
        await self.prepare()
        if self.stopped:
            raise StopAsyncIteration()
        if self.offset > self.total_samples:
            raise StopAsyncIteration()
        batch_size = self.batch_size if batch_size is None else batch_size

        table_cls = self.table_model_cls

        async def operation(session: AsyncSession):
            stmt = (
                select(table_cls)
                .where(table_cls.id > self.offset)
                .order_by(asc(table_cls.id))
                .limit(batch_size)
            )
            result = await session.execute(stmt)
            results = result.scalars().all()
            if len(results) == 0:
                raise StopAsyncIteration()
            if not self.is_eval and len(results) < batch_size:
                raise StopAsyncIteration()
            return results[-1].id, [self.formatter.format(item.raw_task) for item in results]

        self.offset, tasks = await async_run_with_retry_session(
            self.session, operation, self.max_retry_times, self.max_retry_interval
        )
        return tasks

    @classmethod
    async def load_from_dataset(cls, dataset: Dataset, config: StorageConfig) -> "SQLTaskStorage":
        storage = cls(config)
        await storage.prepare()
        batch_size = config.batch_size
        batch = []
        for item in dataset:
            batch.append(item)
            if len(batch) >= batch_size:
                await storage.write(batch)
                batch.clear()
        if batch:
            await storage.write(batch)
        return storage

    def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    def release(self) -> int:
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.stopped = True
        return self.ref_count


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class SQLStorage:
    """Factory for creating SQL storage Ray actors."""

    @classmethod
    def get_wrapper(cls, config: StorageConfig):
        if config.schema_type is None:
            async_cls = SQLTaskStorage
        else:
            async_cls = SQLExperienceStorage
        return (
            ray.remote(async_cls)
            .options(
                name=f"sql-{config.name}",
                namespace=config.ray_namespace or ray.get_runtime_context().namespace,
                get_if_exists=True,
                max_concurrency=5,
            )
            .remote(config)
        )
