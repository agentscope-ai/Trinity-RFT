"""Tests for SQL buffer storage (async primary + Ray actors)."""

import os
import unittest
from copy import deepcopy

import torch
from parameterized import parameterized

from tests.tools import RayUnittestBaseAsync
from trinity.buffer.reader.sql_reader import SQLReader
from trinity.buffer.storage.sql import SQLExperienceStorage, SQLTaskStorage
from trinity.buffer.utils import to_async_url
from trinity.buffer.writer.sql_writer import SQLWriter
from trinity.common.config import (
    ExperienceBufferConfig,
    ReplayBufferConfig,
    StorageConfig,
    TasksetConfig,
)
from trinity.common.constants import StorageType
from trinity.common.experience import Experience

db_path = os.path.join(os.path.dirname(__file__), "test.db")


# ---------------------------------------------------------------------------
# Unit tests: URL conversion
# ---------------------------------------------------------------------------


class TestToAsyncUrl(unittest.TestCase):
    """Tests for URL dialect conversion."""

    def test_sqlite(self):
        self.assertEqual(
            to_async_url("sqlite:///path/to/db.sqlite"),
            "sqlite+aiosqlite:///path/to/db.sqlite",
        )

    def test_postgresql(self):
        self.assertEqual(
            to_async_url("postgresql://user:pass@host/db"),
            "postgresql+asyncpg://user:pass@host/db",
        )

    def test_mysql(self):
        self.assertEqual(
            to_async_url("mysql://user:pass@host/db"),
            "mysql+aiomysql://user:pass@host/db",
        )

    def test_already_async(self):
        url = "sqlite+aiosqlite:///path/to/db.sqlite"
        self.assertEqual(to_async_url(url), url)


# ---------------------------------------------------------------------------
# Unit tests: async storage directly
# ---------------------------------------------------------------------------


class TestSQLExperienceStorageAsync(unittest.IsolatedAsyncioTestCase):
    """Tests for SQLExperienceStorage (async primary class)."""

    def _make_config(self, schema_type="experience") -> StorageConfig:
        config = StorageConfig()
        config.name = f"test_{schema_type}"
        config.path = f"sqlite:///{db_path}"
        config.schema_type = schema_type
        config.storage_type = "sql"
        config.wrap_in_ray = False
        config.batch_size = 4
        config.max_read_timeout = 3
        return config

    def _make_experiences(self, num: int, token_length: int = 32) -> list:
        return [
            Experience(
                tokens=torch.randint(0, 1000, (token_length,), dtype=torch.int32),
                prompt_length=token_length // 4,
                reward=float(i) / num,
                logprobs=torch.randn(token_length - token_length // 4),
                info={"model_version": i % 3},
            )
            for i in range(num)
        ]

    async def test_write_and_read_fifo(self):
        """Write experiences and read them back in FIFO order."""
        config = self._make_config(schema_type="sft")
        storage = SQLExperienceStorage(config)
        await storage.prepare()

        exps = self._make_experiences(8)
        await storage.write(exps)

        result = await storage.read(batch_size=4)
        self.assertEqual(len(result), 4)

        result2 = await storage.read(batch_size=4)
        self.assertEqual(len(result2), 4)

    async def test_write_and_read_priority(self):
        """Write experiences and read them back in priority order."""
        config = self._make_config(schema_type="experience")
        storage = SQLExperienceStorage(config)
        await storage.prepare()

        exps = self._make_experiences(8)
        await storage.write(exps)

        result = await storage.read(batch_size=8)
        self.assertEqual(len(result), 8)

    async def test_stop_iteration_on_release(self):
        """After release, read should raise StopAsyncIteration."""
        config = self._make_config(schema_type="sft")
        storage = SQLExperienceStorage(config)
        await storage.prepare()

        exps = self._make_experiences(4)
        await storage.write(exps)

        storage.acquire()
        storage.release()

        with self.assertRaises(StopAsyncIteration):
            await storage.read(batch_size=4)

    async def test_oversized_experience_skipped(self):
        """Experiences exceeding max_experience_bytes should be skipped."""
        config = self._make_config(schema_type="sft")
        storage = SQLExperienceStorage(config)
        await storage.prepare()

        small_exps = self._make_experiences(2, token_length=16)
        large_exps = self._make_experiences(3, token_length=2048)

        small_size = len(small_exps[0].serialize())
        large_size = len(large_exps[0].serialize())
        storage.max_experience_bytes = (small_size + large_size) // 2

        await storage.write(large_exps + small_exps)

        result = await storage.read(batch_size=2)
        self.assertEqual(len(result), 2)

    async def test_read_timeout(self):
        """Read should raise StopAsyncIteration when timeout is reached."""
        config = self._make_config(schema_type="sft")
        config.max_read_timeout = 1
        storage = SQLExperienceStorage(config)
        await storage.prepare()

        exps = self._make_experiences(2)
        await storage.write(exps)

        with self.assertRaises(StopAsyncIteration):
            await storage.read(batch_size=10)

    async def test_meta_blob_consistency(self):
        """Meta and blob tables must have identical row counts and matching ids."""
        config = self._make_config(schema_type="experience")
        storage = SQLExperienceStorage(config)
        await storage.prepare()

        exps = self._make_experiences(10)
        await storage.write(exps)

        from sqlalchemy import select

        async with storage.session() as session:
            meta_result = await session.execute(select(storage.table_model_cls))
            meta_ids = sorted(r.id for r in meta_result.scalars().all())
            blob_result = await session.execute(select(storage.blob_model_cls))
            blob_ids = sorted(r.id for r in blob_result.scalars().all())

        self.assertEqual(len(meta_ids), 10)
        self.assertEqual(meta_ids, blob_ids)

    async def test_count_and_query_with_filters(self):
        """count() and query() correctly apply reward/model_version/task_id filters."""
        config = self._make_config(schema_type="experience")
        storage = SQLExperienceStorage(config)
        await storage.prepare()

        exps = self._make_experiences(30)
        await storage.write(exps)

        total = await storage.count()
        self.assertEqual(total, 30)

        count_reward = await storage.count(filters={"reward_min": 0.5, "reward_max": 1.0})
        self.assertEqual(count_reward, 15)

        count_mv = await storage.count(filters={"model_version_min": 1, "model_version_max": 1})
        self.assertEqual(count_mv, 10)

        results = await storage.query(offset=0, limit=5, filters={"reward_min": 0.5})
        self.assertEqual(len(results), 5)
        for exp in results:
            self.assertGreaterEqual(exp.reward, 0.5)

    async def test_oversized_blob_skipped_consistently(self):
        """Experiences exceeding max_experience_bytes must not leave orphaned meta rows."""
        config = self._make_config(schema_type="experience")
        storage = SQLExperienceStorage(config)
        await storage.prepare()

        small_exps = self._make_experiences(2, token_length=16)
        large_exps = self._make_experiences(3, token_length=2048)

        small_size = len(small_exps[0].serialize())
        large_size = len(large_exps[0].serialize())
        storage.max_experience_bytes = (small_size + large_size) // 2

        await storage.write(large_exps + small_exps)

        from sqlalchemy import func, select

        async with storage.session() as session:
            meta_count = (
                await session.execute(select(func.count()).select_from(storage.table_model_cls))
            ).scalar()
            blob_count = (
                await session.execute(select(func.count()).select_from(storage.blob_model_cls))
            ).scalar()

        self.assertEqual(meta_count, blob_count)
        self.assertEqual(meta_count, 2)

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)


class TestSQLTaskStorageAsync(unittest.IsolatedAsyncioTestCase):
    """Tests for SQLTaskStorage (async primary class)."""

    def _make_config(self) -> StorageConfig:
        config = StorageConfig()
        config.name = "test_task"
        config.path = f"sqlite:///{db_path}"
        config.schema_type = None
        config.storage_type = "sql"
        config.wrap_in_ray = False
        config.batch_size = 4
        config.max_read_timeout = 5
        config.default_workflow_type = "math_workflow"
        return config

    async def test_write_and_read_tasks(self):
        """Write tasks and read them back."""
        config = self._make_config()
        storage = SQLTaskStorage(config)
        await storage.prepare()

        tasks = [{"question": f"q_{i}", "answer": f"a_{i}"} for i in range(8)]
        await storage.write(tasks)

        result = await storage.read(batch_size=4)
        self.assertEqual(len(result), 4)

        result2 = await storage.read(batch_size=4)
        self.assertEqual(len(result2), 4)

    async def test_stop_iteration_on_empty(self):
        """Read should raise StopAsyncIteration when no data available."""
        config = self._make_config()
        storage = SQLTaskStorage(config)
        await storage.prepare()

        with self.assertRaises(StopAsyncIteration):
            await storage.read(batch_size=4)

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)


# ---------------------------------------------------------------------------
# Unit tests: reader/writer async roundtrip (no Ray)
# ---------------------------------------------------------------------------


class TestReaderWriterAsync(unittest.IsolatedAsyncioTestCase):
    """Tests for SQLReader/SQLWriter async methods without Ray."""

    def _make_config(self) -> StorageConfig:
        config = StorageConfig()
        config.name = "test_rw"
        config.path = f"sqlite:///{db_path}"
        config.schema_type = "sft"
        config.storage_type = "sql"
        config.wrap_in_ray = False
        config.batch_size = 4
        config.max_read_timeout = 3
        return config

    async def test_writer_reader_roundtrip(self):
        """Writer.write and Reader.read should work end-to-end."""
        config = self._make_config()
        writer = SQLWriter(config)
        reader = SQLReader(config)

        await writer.acquire()

        exps = [
            Experience(
                tokens=torch.randint(0, 1000, (32,), dtype=torch.int32),
                prompt_length=8,
                reward=float(i),
                logprobs=torch.randn(24),
                info={"model_version": i},
            )
            for i in range(8)
        ]

        await writer.write(exps)

        result = await reader.read(batch_size=4)
        self.assertEqual(len(result), 4)

        result2 = await reader.read(batch_size=4)
        self.assertEqual(len(result2), 4)

        await writer.release()

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)


# ---------------------------------------------------------------------------
# Integration tests: Ray actors
# ---------------------------------------------------------------------------


class TestSQLBufferRay(RayUnittestBaseAsync):
    """Tests for SQL buffer via Ray actors (SQLWriter/SQLReader with wrap_in_ray=True)."""

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    async def test_sql_exp_buffer_read_write(self, enable_replay: bool) -> None:
        total_num = 8
        put_batch_size = 2
        read_batch_size = 4
        config = ExperienceBufferConfig(
            name="test_buffer",
            schema_type="experience",
            path=f"sqlite:///{db_path}",
            storage_type=StorageType.SQL.value,
            batch_size=read_batch_size,
            max_read_timeout=3,
        )
        if enable_replay:
            config.replay_buffer = ReplayBufferConfig(enable=True)
        writer_config = deepcopy(config)
        writer_config.batch_size = put_batch_size
        sql_writer = SQLWriter(writer_config.to_storage_config())
        sql_reader = SQLReader(config.to_storage_config())
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                prompt_length=i,
                reward=float(i),
                logprobs=torch.tensor([0.1]),
                info={"model_version": i},
            )
            for i in range(1, put_batch_size + 1)
        ]
        self.assertEqual(await sql_writer.acquire(), 1)
        for _ in range(total_num // put_batch_size):
            await sql_writer.write(exps)
        for _ in range(total_num // read_batch_size):
            exps = await sql_reader.read()
            self.assertEqual(len(exps), read_batch_size)

        # dynamic read/write
        await sql_writer.write(
            [
                Experience(
                    tokens=torch.tensor([float(j) for j in range(i + 1)]),
                    reward=float(i),
                    logprobs=torch.tensor([0.1]),
                    action_mask=torch.tensor([j % 2 for j in range(i + 1)]),
                    info={"model_version": i + put_batch_size},
                )
                for i in range(1, put_batch_size * 2 + 1)
            ]
        )
        exps = await sql_reader.read(batch_size=put_batch_size * 2)
        self.assertEqual(len(exps), put_batch_size * 2)
        for exp in exps:
            self.assertTrue(exp.info["model_version"] > put_batch_size)
        if enable_replay:
            exps = await sql_reader.read(batch_size=(put_batch_size * 2 + total_num))
            self.assertEqual(len(exps), (put_batch_size * 2 + total_num))
            with self.assertRaises(StopAsyncIteration):
                await sql_reader.read(batch_size=(put_batch_size * 3 + total_num))
        import ray

        db_wrapper = ray.get_actor("sql-test_buffer")
        self.assertIsNotNone(db_wrapper)
        self.assertEqual(await sql_writer.release(), 0)
        with self.assertRaises(StopAsyncIteration):
            await sql_reader.read()

    async def test_sql_task_buffer_read_write(self) -> None:
        total_samples = 8
        batch_size = 4
        config = TasksetConfig(
            name="test_task_buffer",
            path=f"sqlite:///{db_path}",
            storage_type=StorageType.SQL.value,
            batch_size=batch_size,
            default_workflow_type="math_workflow",
        )
        sql_writer = SQLWriter(config.to_storage_config())
        sql_reader = SQLReader(config.to_storage_config())
        self.assertEqual(await sql_writer.acquire(), 1)
        await sql_writer.write(
            [{"question": f"question_{i}", "answer": f"answer_{i}"} for i in range(total_samples)]
        )
        read_tasks = []
        while True:
            try:
                cur_tasks = await sql_reader.read()
                read_tasks.extend(cur_tasks)
            except StopAsyncIteration:
                break
        self.assertEqual(len(read_tasks), total_samples)
        self.assertIn("question", read_tasks[0].raw_task)
        self.assertIn("answer", read_tasks[0].raw_task)
        import ray

        db_wrapper = ray.get_actor("sql-test_task_buffer")
        self.assertIsNotNone(db_wrapper)
        self.assertEqual(await sql_writer.release(), 0)

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    unittest.main()
