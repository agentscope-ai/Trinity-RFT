import os
import queue
import threading
import time

import ray
import torch
from parameterized import parameterized

from tests.tools import RayUnittestBaseAsync
from trinity.buffer.reader.queue_reader import QueueReader
from trinity.buffer.writer.queue_writer import QueueWriter
from trinity.common.config import ExperienceBufferConfig, ReplayBufferConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience

BUFFER_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_queue_buffer.jsonl")


class TestQueueBuffer(RayUnittestBaseAsync):
    @parameterized.expand(
        [
            (
                "queue",
                False,
            ),
            (
                "priority_queue",
                True,
            ),
        ]
    )
    async def test_queue_buffer(self, name, use_priority_queue):
        config = ExperienceBufferConfig(
            name=name,
            schema_type="experience",
            storage_type=StorageType.QUEUE.value,
            max_read_timeout=3,
            path=BUFFER_FILE_PATH,
            batch_size=self.train_batch_size,
        )
        config.replay_buffer.enable = use_priority_queue
        config = config.to_storage_config()
        writer = QueueWriter(config)
        reader = QueueReader(config)
        self.assertEqual(await writer.acquire(), 1)
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                prompt_length=i,
                reward=float(i),
                logprobs=torch.tensor([0.1]),
            )
            for i in range(1, self.put_batch_size + 1)
        ]
        for exp in exps:
            exp.info = {"model_version": 0, "use_count": 0}
        for _ in range(self.total_num // self.put_batch_size):
            await writer.write_async(exps)
        for _ in range(self.total_num // self.train_batch_size):
            exps = reader.read()
            self.assertEqual(len(exps), self.train_batch_size)
            print(f"finish read {self.train_batch_size} experience")
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                reward=float(i),
                logprobs=torch.tensor([0.1]),
                action_mask=torch.tensor([j % 2 for j in range(i + 1)]),
            )
            for i in range(1, self.put_batch_size * 2 + 1)
        ]
        for exp in exps:
            exp.info = {"model_version": 1, "use_count": 0}
        writer.write(exps)
        exps = reader.read(batch_size=self.put_batch_size * 2)
        self.assertEqual(len(exps), self.put_batch_size * 2)

        def thread_read(reader, result_queue):
            try:
                batch = reader.read()
                result_queue.put(batch)
            except StopIteration as e:
                result_queue.put(e)

        result_queue = queue.Queue()
        t = threading.Thread(target=thread_read, args=(reader, result_queue))
        t.start()
        time.sleep(2)  # make sure the thread is waiting for data
        self.assertEqual(await writer.release(), 0)
        t.join(timeout=1)
        self.assertIsInstance(result_queue.get(), StopIteration)
        with open(BUFFER_FILE_PATH, "r") as f:
            self.assertEqual(len(f.readlines()), self.total_num + self.put_batch_size * 2)
        self.assertRaises(StopIteration, reader.read, batch_size=1)

    async def test_priority_queue_capacity(self):
        # test priority queue capacity
        self.train_batch_size = 4
        config = ExperienceBufferConfig(
            name="test_buffer_small",
            schema_type="experience",
            storage_type=StorageType.QUEUE.value,
            max_read_timeout=1,
            capacity=8,
            path=BUFFER_FILE_PATH,
            replay_buffer=ReplayBufferConfig(
                enable=True,
                priority_fn="linear_decay",
                reuse_cooldown_time=None,
                priority_fn_args={"decay": 0.6},
            ),
            batch_size=self.train_batch_size,
        )
        config = config.to_storage_config()
        writer = QueueWriter(config)
        reader = QueueReader(config)

        for i in range(12):
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": i, "use_count": 0},
                    ),
                ]
            )

        self.assertEqual(ray.get(reader.queue.length.remote()), 8)

        exps = reader.read(batch_size=8)
        self.assertEqual(exps[0].info["model_version"], 11)
        self.assertEqual(exps[0].info["use_count"], 1)
        self.assertEqual(exps[1].info["model_version"], 10)
        self.assertEqual(exps[1].info["use_count"], 1)
        self.assertEqual(exps[7].info["model_version"], 4)

        with self.assertRaises(TimeoutError):
            reader.read(batch_size=1)

        for i in range(12):
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": i, "use_count": 0},
                    ),
                ]
            )
        await writer.release()
        exps = reader.read(batch_size=8)

        with self.assertRaises(StopIteration):
            reader.read(batch_size=1)

    async def test_queue_buffer_capacity(self):
        # test queue capacity
        config = ExperienceBufferConfig(
            name="test_buffer_small",
            schema_type="experience",
            storage_type=StorageType.QUEUE.value,
            max_read_timeout=3,
            capacity=4,
            path=BUFFER_FILE_PATH,
            batch_size=self.train_batch_size,
        )
        config = config.to_storage_config()
        writer = QueueWriter(config)
        reader = QueueReader(config)
        writer.write(
            [
                Experience(
                    tokens=torch.tensor([1, 2, 3]),
                    prompt_length=2,
                    info={"model_version": 0, "use_count": 0},
                )
            ]
        )
        writer.write(
            [
                Experience(
                    tokens=torch.tensor([1, 2, 3]),
                    prompt_length=2,
                    info={"model_version": 1, "use_count": 0},
                )
            ]
        )
        writer.write(
            [
                Experience(
                    tokens=torch.tensor([1, 2, 3]),
                    prompt_length=2,
                    info={"model_version": 2, "use_count": 0},
                )
            ]
        )
        writer.write(
            [
                Experience(
                    tokens=torch.tensor([1, 2, 3]),
                    prompt_length=2,
                    info={"model_version": 3, "use_count": 0},
                )
            ]
        )

        # should be blocked
        def write_blocking_call():
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": 4, "use_count": 0},
                    )
                ]
            )

        thread = threading.Thread(target=write_blocking_call)
        thread.start()
        thread.join(timeout=2)
        self.assertTrue(thread.is_alive(), "write() did not block as expected")
        reader.read()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

    async def test_priority_queue_buffer_reuse(self):
        # test experience replay
        config = ExperienceBufferConfig(
            name="test_buffer_small",
            schema_type="experience",
            storage_type=StorageType.QUEUE.value,
            max_read_timeout=3,
            capacity=4,  # max total number of items; each item is List[Experience]
            path=BUFFER_FILE_PATH,
            replay_buffer=ReplayBufferConfig(
                enable=True,
                priority_fn="linear_decay",
                reuse_cooldown_time=0.5,
                priority_fn_args={"decay": 0.6},
            ),
            batch_size=self.train_batch_size,
        )
        config = config.to_storage_config()
        writer = QueueWriter(config)
        reader = QueueReader(config)
        for i in range(4):
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": i, "use_count": 0},
                    ),
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": i, "use_count": 0},
                    ),
                ]
            )

        # should not be blocked
        def replace_call():
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": 4, "use_count": 0},
                    ),
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": 4, "use_count": 0},
                    ),
                ]
            )

        thread = threading.Thread(target=replace_call)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 1)
        self.assertEqual(exps[2].info["model_version"], 3)
        self.assertEqual(exps[2].info["use_count"], 1)

        # model_version  4,   3,   2,   1
        # use_count      1,   1,   0,   0
        # priority      3.4, 2.4, 2.0, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 2)
        self.assertEqual(exps[2].info["model_version"], 3)
        self.assertEqual(exps[2].info["use_count"], 2)

        # model_version  4,   3,   2,   1
        # use_count      2,   2,   0,   0
        # priority      2.8, 1.8, 2.0, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 3)
        self.assertEqual(exps[2].info["model_version"], 2)
        self.assertEqual(exps[2].info["use_count"], 1)

        # model_version  4,   3,   2,   1
        # use_count      3,   2,   1,   0
        # priority      2.2, 1.8, 1.4, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 4)
        self.assertEqual(exps[2].info["model_version"], 3)
        self.assertEqual(exps[2].info["use_count"], 3)

        # model_version  4,   3,   2,   1
        # use_count      4,   3,   1,   0
        # priority      1.6, 1.2, 1.4, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 5)
        self.assertEqual(exps[2].info["model_version"], 2)
        self.assertEqual(exps[2].info["use_count"], 2)

        # model_version  4,   3,   2,   1
        # use_count      5,   3,   2,   0
        # priority      1.0, 1.2, 0.8, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 3)
        self.assertEqual(exps[0].info["use_count"], 4)
        self.assertEqual(exps[2].info["model_version"], 1)
        self.assertEqual(exps[2].info["use_count"], 1)

        # model_version  4,   3,   2,   1
        # use_count      5,   4,   2,   1
        # priority      1.0, 0.6, 0.8, 0.4

    async def test_priority_queue_reuse_count_control(self):
        # test experience replay with linear decay and use count control
        config = ExperienceBufferConfig(
            name="test_buffer_small",
            schema_type="experience",
            storage_type=StorageType.QUEUE.value,
            max_read_timeout=3,
            capacity=4,  # max total number of items; each item is List[Experience]
            path=BUFFER_FILE_PATH,
            replay_buffer=ReplayBufferConfig(
                enable=True,
                priority_fn="decay_limit_randomization",
                reuse_cooldown_time=0.5,
                priority_fn_args={"decay": 1.2, "use_count_limit": 2, "sigma": 0.0},
            ),
            batch_size=self.train_batch_size,
        )
        config = config.to_storage_config()
        writer = QueueWriter(config)
        reader = QueueReader(config)
        for i in range(4):
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": i, "use_count": 0},
                    ),
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": i, "use_count": 0},
                    ),
                ]
            )

        # should not be blocked
        def replace_call():
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": 4, "use_count": 0},
                    ),
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": 4, "use_count": 0},
                    ),
                ]
            )

        thread = threading.Thread(target=replace_call)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 1)
        self.assertEqual(exps[2].info["model_version"], 3)
        self.assertEqual(exps[2].info["use_count"], 1)

        # model_version  4,   3,   2,   1
        # use_count      1,   1,   0,   0
        # priority      2.8, 1.8, 2.0, 1.0
        # in queue       Y,   Y,   Y,   Y

        time.sleep(1)
        self.assertEqual(ray.get(reader.queue.length.remote()), 4)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 2)
        self.assertEqual(exps[2].info["model_version"], 2)
        self.assertEqual(exps[2].info["use_count"], 1)

        # model_version  4,   3,   2,   1
        # use_count      2,   1,   1,   0
        # priority      1.6, 1.8, 0.8, 1.0
        # in queue       N,   Y,   Y,   Y
        # model_version = 4 item is discarded for reaching use_count_limit

        time.sleep(1)
        self.assertEqual(ray.get(reader.queue.length.remote()), 3)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 3)
        self.assertEqual(exps[0].info["use_count"], 2)
        self.assertEqual(exps[2].info["model_version"], 1)
        self.assertEqual(exps[2].info["use_count"], 1)

        # model_version  3,    2,    1
        # use_count      2,    1,    1
        # priority      0.6,  0.8, -0.2
        # in queue       N,    Y,    Y
        # model_version = 3 item is discarded for reaching use_count_limit

        time.sleep(1)
        self.assertEqual(ray.get(reader.queue.length.remote()), 2)

    def setUp(self):
        self.total_num = 8
        self.put_batch_size = 2
        self.train_batch_size = 4

        if os.path.exists(BUFFER_FILE_PATH):
            os.remove(BUFFER_FILE_PATH)
