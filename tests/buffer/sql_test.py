import os
import unittest
from copy import deepcopy

import ray
import torch
from parameterized import parameterized
from sqlalchemy.orm import sessionmaker

from tests.tools import RayUnittestBaseAsync
from trinity.buffer import get_buffer_reader
from trinity.buffer.reader.sql_reader import SQLReader
from trinity.buffer.storage.sql import SQLExperienceStorage
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


class TestSQLBuffer(RayUnittestBaseAsync):
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
        # Create buffer by writer, so buffer.batch_size will be set to put_batch_size
        # This will check whether read_batch_size tasks effect
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
            await sql_writer.write_async(exps)
        for _ in range(total_num // read_batch_size):
            exps = sql_reader.read()
            self.assertEqual(len(exps), read_batch_size)

        # dynamic read/write
        sql_writer.write(
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
        exps = sql_reader.read(batch_size=put_batch_size * 2)
        self.assertEqual(len(exps), put_batch_size * 2)
        for exp in exps:
            self.assertTrue(exp.info["model_version"] > put_batch_size)
        if enable_replay:
            # support replay, so we can read all again
            exps = sql_reader.read(batch_size=(put_batch_size * 2 + total_num))
            self.assertEqual(len(exps), (put_batch_size * 2 + total_num))
            # if read more than available, will wait until timeout
            with self.assertRaises(StopIteration):
                exps = sql_reader.read(batch_size=(put_batch_size * 3 + total_num))
        db_wrapper = ray.get_actor("sql-test_buffer")
        self.assertIsNotNone(db_wrapper)
        self.assertEqual(await sql_writer.release(), 0)
        self.assertRaises(StopIteration, sql_reader.read)

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
        tasks = [
            {"question": f"question_{i}", "answer": f"answer_{i}"} for i in range(total_samples)
        ]
        self.assertEqual(await sql_writer.acquire(), 1)
        sql_writer.write(tasks)
        sql_reader = get_buffer_reader(config.to_storage_config())
        read_tasks = []
        try:
            while True:
                cur_tasks = sql_reader.read()
                read_tasks.extend(cur_tasks)
        except StopIteration:
            pass
        self.assertEqual(len(read_tasks), total_samples)
        self.assertIn("question", read_tasks[0].raw_task)
        self.assertIn("answer", read_tasks[0].raw_task)
        db_wrapper = ray.get_actor("sql-test_task_buffer")
        self.assertIsNotNone(db_wrapper)
        self.assertEqual(await sql_writer.release(), 0)

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)


class TestSQLSplitTable(unittest.TestCase):
    """Tests for the meta/blob split table architecture."""

    def _make_storage(self) -> SQLExperienceStorage:
        config = StorageConfig()
        config.name = "test_split"
        config.path = f"sqlite:///{db_path}"
        config.schema_type = "experience"
        config.storage_type = "sql"
        config.wrap_in_ray = False
        config.batch_size = 10
        config.max_read_timeout = 5
        return SQLExperienceStorage(config)

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

    def test_meta_blob_consistency(self):
        """Meta and blob tables must have identical row counts and matching ids."""
        storage = self._make_storage()
        exps = self._make_experiences(10)
        storage.write(exps)

        session = sessionmaker(bind=storage.engine)
        with session() as s:
            meta_ids = sorted(r.id for r in s.query(storage.table_model_cls).all())
            blob_ids = sorted(r.id for r in s.query(storage.blob_model_cls).all())

        self.assertEqual(len(meta_ids), 10)
        self.assertEqual(meta_ids, blob_ids)

    def test_count_and_query_with_filters(self):
        """count() and query() correctly apply reward/model_version/task_id filters."""
        storage = self._make_storage()
        exps = self._make_experiences(30)
        storage.write(exps)

        # reward ranges from 0.0 to 29/30 ≈ 0.967
        # model_version cycles 0, 1, 2, 0, 1, 2, ...
        total = storage.count()
        self.assertEqual(total, 30)

        # Reward filter: [0.5, 1.0] should match i/30 >= 0.5 → i >= 15
        count_reward = storage.count(filters={"reward_min": 0.5, "reward_max": 1.0})
        self.assertEqual(count_reward, 15)

        # Model version filter: only version 1 → i%3==1, that's indices 1,4,7,...,28 = 10 items
        count_mv = storage.count(filters={"model_version_min": 1, "model_version_max": 1})
        self.assertEqual(count_mv, 10)

        # Query with limit
        results = storage.query(offset=0, limit=5, filters={"reward_min": 0.5})
        self.assertEqual(len(results), 5)
        for exp in results:
            self.assertGreaterEqual(exp.reward, 0.5)

    def test_oversized_blob_skipped_consistently(self):
        """Experiences exceeding max_experience_bytes must not leave orphaned meta rows."""
        storage = self._make_storage()

        small_exps = self._make_experiences(2, token_length=16)
        large_exps = self._make_experiences(3, token_length=2048)

        # Set threshold between small and large serialized sizes
        small_size = len(small_exps[0].serialize())
        large_size = len(large_exps[0].serialize())
        storage.max_experience_bytes = (small_size + large_size) // 2

        storage.write(large_exps + small_exps)

        session = sessionmaker(bind=storage.engine)
        with session() as s:
            meta_count = s.query(storage.table_model_cls).count()
            blob_count = s.query(storage.blob_model_cls).count()

        # Only small exps should be written; meta and blob counts must match
        self.assertEqual(meta_count, blob_count)
        self.assertEqual(meta_count, 2)

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)
