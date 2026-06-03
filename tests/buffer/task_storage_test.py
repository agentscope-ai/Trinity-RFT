import os

import datasets
from parameterized import parameterized

from tests.tools import (
    RayUnittestBaseAsync,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.buffer import get_buffer_reader
from trinity.buffer.storage.sql import SQLTaskStorage
from trinity.common.constants import StorageType

db_path = os.path.join(os.path.dirname(__file__), "test.db")


class TaskStorageTest(RayUnittestBaseAsync):
    @parameterized.expand(
        [
            (StorageType.FILE.value, True, 2),
            (StorageType.SQL.value, True, 2),
            (StorageType.FILE.value, False, 0),
            (StorageType.SQL.value, False, 0),
            (StorageType.FILE.value, False, 2),
            (StorageType.SQL.value, False, 2),
        ]
    )
    async def test_read_task(self, storage_type, is_eval, offset):
        config = get_template_config()
        total_samples = 17
        batch_size = 4
        config.buffer.explorer_input.taskset = get_unittest_dataset_config(
            "countdown"
        )  # 17 samples
        config.buffer.explorer_input.taskset.storage_type = storage_type
        config.buffer.explorer_input.taskset.is_eval = is_eval
        config.buffer.explorer_input.taskset.index = offset
        config.buffer.explorer_input.taskset.batch_size = batch_size
        if storage_type == StorageType.SQL.value:
            dataset = datasets.load_dataset(
                config.buffer.explorer_input.taskset.path, split="train"
            )
            config.buffer.explorer_input.taskset.path = f"sqlite:///{db_path}"
            await SQLTaskStorage.load_from_dataset(
                dataset, config.buffer.explorer_input.taskset.to_storage_config()
            )
        reader = get_buffer_reader(config.buffer.explorer_input.taskset)
        tasks = []
        while True:
            try:
                cur_tasks = await reader.read()
                tasks.extend(cur_tasks)
            except StopAsyncIteration:
                break
        if is_eval:
            self.assertEqual(len(tasks), total_samples - offset)
        else:
            self.assertEqual(len(tasks), (total_samples - offset) // batch_size * batch_size)

    def setUp(self):
        if os.path.exists(db_path):
            os.remove(db_path)
