import os
import unittest

import ray

from tests.tools import (
    get_checkpoint_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.buffer.buffer import get_buffer_reader, get_buffer_writer
from trinity.common.config import ExperienceBufferConfig
from trinity.common.constants import StorageType


class TestFileBuffer(unittest.IsolatedAsyncioTestCase):
    async def test_file_reader(self):  # noqa: C901
        """Test file reader."""
        reader = get_buffer_reader(self.config.buffer.explorer_input.tasksets[0])

        tasks = []
        while True:
            try:
                tasks.extend(await reader.read())
            except StopAsyncIteration:
                break
        self.assertEqual(len(tasks), 16)

        # test epoch and offset
        self.config.buffer.explorer_input.tasksets[0].total_epochs = 2
        self.config.buffer.explorer_input.tasksets[0].index = 4
        reader = get_buffer_reader(
            self.config.buffer.explorer_input.tasksets[0],
        )
        tasks = []
        while True:
            try:
                tasks.extend(await reader.read())
            except StopAsyncIteration:
                break
        self.assertEqual(len(tasks), 16 * 2 - 4)

        # test total steps and offset
        self.config.buffer.explorer_input.tasksets[0].total_steps = 5
        self.config.buffer.explorer_input.tasksets[0].index = 8
        reader = get_buffer_reader(self.config.buffer.explorer_input.tasksets[0])
        tasks = []
        while True:
            try:
                tasks.extend(await reader.read())
            except StopAsyncIteration:
                break
        self.assertEqual(len(tasks), 20 - 8)

        # test offset > dataset_len with total_epoch
        self.config.buffer.explorer_input.tasksets[0].total_steps = None
        self.config.buffer.explorer_input.tasksets[0].total_epochs = 3
        self.config.buffer.explorer_input.tasksets[0].index = 20
        reader = get_buffer_reader(self.config.buffer.explorer_input.tasksets[0])
        tasks = []
        while True:
            try:
                tasks.extend(await reader.read())
            except StopAsyncIteration:
                break
        self.assertEqual(len(tasks), 16 * 3 - 20)

        # test offset > dataset_len with total_steps
        self.config.buffer.explorer_input.tasksets[0].total_steps = 10
        self.config.buffer.explorer_input.tasksets[0].index = 24
        reader = get_buffer_reader(self.config.buffer.explorer_input.tasksets[0])
        tasks = []
        while True:
            try:
                tasks.extend(await reader.read())
            except StopAsyncIteration:
                break
        self.assertEqual(len(tasks), 40 - 24)

    async def test_file_writer(self):
        writer = get_buffer_writer(self.config.buffer.trainer_input.experience_buffer)
        await writer.acquire()
        await writer.write(
            [
                {"prompt": "hello world"},
                {"prompt": "hi"},
            ]
        )
        await writer.write(
            [
                {"prompt": "My name is"},
                {"prompt": "What is your name?"},
            ]
        )
        await writer.release()
        file_wrapper = ray.get_actor("json-test_buffer")
        self.assertIsNotNone(file_wrapper)
        file_path = self.config.buffer.trainer_input.experience_buffer.path
        with open(file_path, "r") as f:
            self.assertEqual(len(f.readlines()), 4)

    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        dataset_config = get_unittest_dataset_config("countdown", "train")
        self.config.buffer.explorer_input.taskset = dataset_config
        self.config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="test_buffer", storage_type=StorageType.FILE.value
        )
        self.config.check_and_update()
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": self.config.get_envs()},
            namespace="trinity_unittest",
        )
        os.makedirs(self.config.buffer.cache_dir, exist_ok=True)
        file_path = self.config.buffer.trainer_input.experience_buffer.path
        if os.path.exists(file_path):
            os.remove(file_path)

    def tearDown(self):
        ray.shutdown()
