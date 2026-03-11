import os
import shutil
import time
import unittest
from multiprocessing import Process

try:
    import data_juicer  # noqa: [F401]
except ImportError:
    raise ImportError(
        "data_juicer module is not installed. Please install it with `pip install py-data-juicer` to run the tests."
    )
import ray
import torch
from jsonargparse import Namespace

from tests.tools import RayUnittestBase, get_template_config
from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.pipelines import ExperiencePipeline, check_and_run_task_pipeline
from trinity.common.config import (
    DataJuicerServiceConfig,
    OperatorConfig,
    TaskPipelineConfig,
    TasksetConfig,
)
from trinity.common.experience import Experience
from trinity.service.data_juicer.client import DataJuicerClient
from trinity.service.data_juicer.server.server import main
from trinity.service.data_juicer.server.utils import DJConfig, parse_config
from trinity.utils.distributed import get_available_port

TASKSET_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "taskset_output")


class TestDataJuicer(unittest.TestCase):
    def test_config(self):
        trinity_config = {
            "operators": [
                {
                    "llm_quality_score_filter": {
                        "api_or_hf_model": "qwen2.5-7b-instruct",
                        "min_score": 0.0,
                        "input_keys": ["prompt_text"],
                        "field_names": ["prompt", "response"],
                    }
                },
                {
                    "llm_difficulty_score_filter": {
                        "api_or_hf_model": "qwen2.5-7b-instruct",
                        "min_score": 0.0,
                        "enable_vllm": False,
                    }
                },
            ],
            "np": 8,
        }
        config = DJConfig.model_validate(trinity_config)
        dj_config = parse_config(config)
        self.assertIsInstance(dj_config, Namespace)

    def test_server_start(self):
        config = DataJuicerServiceConfig(
            server_url="http://localhost:5005",
            auto_start=False,
        )
        with self.assertRaises(ConnectionError):
            # server is not running, and auto_start is disabled
            # this should raise a ConnectionError
            DataJuicerClient(config)

        # Start the server in a separate process
        def start_server(port):
            server_process = Process(
                target=main, kwargs={"host": "localhost", "port": port, "debug": False}
            )
            server_process.start()
            return server_process

        port = get_available_port()
        config.port = port
        server_process = start_server(port)
        time.sleep(15)  # Wait for the server to start
        config.server_url = f"http://localhost:{port}"
        client = DataJuicerClient(config)
        client.initialize(
            {
                "operators": [
                    {
                        "llm_quality_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "field_names": ["prompt", "response"],
                        }
                    },
                    {
                        "llm_difficulty_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "enable_vllm": False,
                        }
                    },
                ],
                "np": 6,
            }
        )
        self.assertIsNotNone(client.session_id)
        server_process.terminate()
        server_process.join()

        # Test auto start
        config.auto_start = True
        client = DataJuicerClient(config)
        client.initialize(
            {
                "operators": [
                    {
                        "llm_quality_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "input_keys": ["prompt_text"],
                            "field_names": ["prompt", "response"],
                        }
                    },
                    {
                        "llm_difficulty_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "enable_vllm": False,
                        }
                    },
                ]
            }
        )
        self.assertIsNotNone(client.session_id)
        self.assertIsNotNone(client.server)
        client.close()
        self.assertIsNone(client.session_id)
        self.assertIsNone(client.server)


class TestDataJuicerExperiencePipeline(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        ray.shutdown()

    async def test_data_juicer_operators(self):
        config = get_template_config()
        config.service.data_juicer = DataJuicerServiceConfig(
            auto_start=True,
        )
        config.data_processor.experience_pipeline.operators = [
            OperatorConfig(
                name="data_juicer",
                args={
                    "operators": [
                        {
                            "text_length_filter": {
                                "min_len": 10,
                                "max_len": 50,
                                "text_key": "response_text",
                            }
                        },
                        {
                            "word_repetition_filter": {
                                "rep_len": 3,
                                "min_ratio": 0.0,
                                "max_ratio": 0.2,
                                "text_key": "response_text",
                            }
                        },
                    ],
                    "np": 2,
                },
            )
        ]
        config.check_and_update()
        config.buffer.trainer_input.experience_buffer.max_read_timeout = 5
        pipeline = ray.remote(ExperiencePipeline).options(num_cpus=2).remote(config)
        await pipeline.prepare.remote()
        exps = [
            Experience(
                tokens=torch.tensor([1, 2, 3, 4, 5]),
                prompt_length=3,
                prompt_text="Hello, how are you?",
                response_text="Hi, I am fine.",
                info={"model_version": 0},
            ),
            Experience(  # too short response
                tokens=torch.tensor([1, 2, 3, 4, 5]),
                prompt_length=3,
                prompt_text="What is your name?",
                response_text="Trinity.",
                info={"model_version": 0},
            ),
            Experience(  # repeated words
                tokens=torch.tensor([1, 2, 3, 4, 5]),
                prompt_length=3,
                prompt_text="What day is it today?",
                response_text="Today is Sunday Sunday Sunday Sunday Sunday and it's a happy day!",
                info={"model_version": 0},
            ),
            Experience(
                tokens=torch.tensor([1, 2, 3, 4, 5]),
                prompt_length=3,
                prompt_text="What is your favorite color?",
                response_text="My favorite color is blue.",
                info={"model_version": 0},
            ),
        ]
        metrics = await pipeline.process.remote(Experience.serialize_many(exps))
        self.assertIsInstance(metrics, dict)
        reader = get_buffer_reader(config.buffer.trainer_input.experience_buffer)
        filtered_exps = reader.read(batch_size=2)
        self.assertEqual(len(filtered_exps), 2)
        with self.assertRaises(TimeoutError):
            reader.read(batch_size=1)
        await pipeline.close.remote()


class TestDataJuicerTaskPipeline(RayUnittestBase):
    def setUp(self):
        if os.path.exists(TASKSET_OUTPUT_DIR):
            shutil.rmtree(TASKSET_OUTPUT_DIR, ignore_errors=True)

    def test_data_juicer_task_pipeline(self):
        config = get_template_config()
        config.service.data_juicer = DataJuicerServiceConfig(
            auto_start=True,
        )
        config.data_processor.task_pipeline = TaskPipelineConfig(
            operators=[
                OperatorConfig(
                    name="text_length_filter",
                    args={
                        "min_len": 10,
                        "max_len": 500,
                        "text_key": "question",
                    },
                ),
                OperatorConfig(
                    name="word_repetition_filter",
                    args={
                        "rep_len": 3,
                        "min_ratio": 0.0,
                        "max_ratio": 0.2,
                        "text_key": "question",
                    },
                ),
            ],
            inputs=[
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "template",
                    "data",
                    "gsm8k",
                    "train.jsonl",
                ),
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "template",
                    "data",
                    "countdown",
                    "train.jsonl",
                ),
            ],
            target_fields=["question", "answer"],
        )
        config.buffer.explorer_input.taskset = TasksetConfig(
            name="taskset",
            path=TASKSET_OUTPUT_DIR,
        )
        config.check_and_update()
        metrics = check_and_run_task_pipeline(config)
        self.assertTrue("sample_num" in metrics)
        self.assertEqual(metrics["sample_num"], 16)
        from datasets import load_dataset

        ds = load_dataset(
            TASKSET_OUTPUT_DIR,
            split="train",
        )
        self.assertEqual(ds.num_rows, 16)
