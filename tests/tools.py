import gc
import os
import unittest
from collections import defaultdict
from typing import Dict, List

import ray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trinity.common.config import (
    Config,
    ExperienceBufferConfig,
    FormatConfig,
    LoRAConfig,
    TasksetConfig,
    load_config,
)
from trinity.common.constants import (
    CHECKPOINT_ROOT_DIR_ENV_VAR,
    MODEL_PATH_ENV_VAR,
    PromptType,
    StorageType,
)

API_MODEL_PATH_ENV_VAR = "TRINITY_API_MODEL_PATH"
VLM_MODEL_PATH_ENV_VAR = "TRINITY_VLM_MODEL_PATH"
SFT_DATASET_PATH_ENV_VAR = "TRINITY_SFT_DATASET_PATH"


def get_template_config() -> Config:
    config_path = os.path.join(os.path.dirname(__file__), "template", "config.yaml")
    config = load_config(config_path)
    if ray.is_initialized():
        config.ray_namespace = ray.get_runtime_context().namespace
    else:
        config.ray_namespace = "trinity_unittest"
    return config


def get_model_path() -> str:
    path = os.environ.get(MODEL_PATH_ENV_VAR)
    if not path:
        raise EnvironmentError(
            f"Please set `export {MODEL_PATH_ENV_VAR}=<your_model_dir>` before running this test."
        )
    return path


def get_api_model_path() -> str:
    path = os.environ.get(API_MODEL_PATH_ENV_VAR)
    if not path:
        raise EnvironmentError(
            f"Please set `export {API_MODEL_PATH_ENV_VAR}=<your_api_model_checkpoint_dir>` before running this test."
        )
    return path


def get_checkpoint_path() -> str:
    path = os.environ.get(CHECKPOINT_ROOT_DIR_ENV_VAR)
    if not path:
        raise EnvironmentError(
            f"Please set `export {CHECKPOINT_ROOT_DIR_ENV_VAR}=<your_checkpoint_dir>` before running this test."
        )
    return path


def get_vision_language_model_path() -> str:
    path = os.environ.get(VLM_MODEL_PATH_ENV_VAR)
    if not path:
        raise EnvironmentError(
            f"Please set `export {VLM_MODEL_PATH_ENV_VAR}=<your_model_dir>` before running this test."
        )
    return path


def get_lora_config() -> LoRAConfig:
    return LoRAConfig(name="lora", lora_rank=16, lora_alpha=16)


def get_unittest_dataset_config(dataset_name: str = "countdown", split: str = "train"):
    if dataset_name == "countdown" or dataset_name == "copy_countdown":
        # Countdown dataset with 17 samples
        return TasksetConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "countdown"),
            split=split,
            enable_progress_bar=False,
            format=FormatConfig(
                prompt_key="question",
                response_key="answer",
            ),
            default_workflow_type="math_workflow",
            default_reward_fn_type="countdown_reward",
        )
    elif dataset_name in {"eval_short", "eval_long"}:
        # Eval_short dataset with 2 samples, eval_long dataset with 8 samples
        return TasksetConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", dataset_name),
            split="test",
            format=FormatConfig(
                prompt_key="question",
                response_key="answer",
            ),
            default_workflow_type="math_workflow",
            default_reward_fn_type="math_reward",
        )
    elif dataset_name == "gsm8k":
        # GSM8K dataset with 16 samples
        return TasksetConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "gsm8k"),
            split="train",
            format=FormatConfig(
                prompt_key="question",
                response_key="answer",
            ),
            default_workflow_type="math_workflow",
            default_reward_fn_type="math_reward",
        )
    elif dataset_name == "gsm8k_ruler":
        return TasksetConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "gsm8k"),
            split="train",
            format=FormatConfig(
                prompt_key="question",
                response_key="answer",
            ),
            default_workflow_type="math_ruler_workflow",
        )
    elif dataset_name == "sft_for_gsm8k":
        # SFT dataset with 8 samples
        return ExperienceBufferConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "sft_for_gsm8k"),
            split="train",
            storage_type=StorageType.FILE.value,
            schema_type="sft",
            format=FormatConfig(
                prompt_type=PromptType.PLAINTEXT,
                prompt_key="prompt",
                response_key="response",
            ),
        )
    elif dataset_name == "sft_with_tools":
        # SFT_with_tools dataset with 4 samples
        return ExperienceBufferConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "sft_with_tools"),
            split="train",
            storage_type=StorageType.FILE.value,
            format=FormatConfig(
                prompt_type=PromptType.MESSAGES,
                messages_key="messages",
                tools_key="tools",
                enable_concatenated_multi_turn=True,
            ),
        )
    elif dataset_name == "dpo":
        # HumanLike DPO dataset with 17 samples
        return ExperienceBufferConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "human_like"),
            split="train",
            storage_type=StorageType.FILE.value,
            schema_type="dpo",
            format=FormatConfig(
                prompt_type=PromptType.PLAINTEXT,
                prompt_key="prompt",
                chosen_key="chosen",
                rejected_key="rejected",
            ),
        )
    elif dataset_name == "geometry":
        # Multi-modal geometry dataset with 8 samples
        return TasksetConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "geometry"),
            split="train",
            format=FormatConfig(
                prompt_type=PromptType.PLAINTEXT,
                prompt_key="problem",
                response_key="answer",
                image_key="images",
            ),
            default_workflow_type="simple_mm_workflow",
            default_reward_fn_type="math_boxed_reward",
        )
    elif dataset_name == "geometry_sft":
        # Multi-modal geometry dataset for sft with 8 samples
        return ExperienceBufferConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "geometry"),
            split="train",
            storage_type=StorageType.FILE.value,
            format=FormatConfig(
                prompt_type=PromptType.PLAINTEXT,
                prompt_key="problem",
                response_key="answer",
                image_key="images",
            ),
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


class TensorBoardParser:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._event_files = self._find_event_files(log_dir)
        self._metrics = self._load_metrics()

    def _find_event_files(self, log_dir: str) -> List[str]:
        event_files = []
        for root, _, files in os.walk(log_dir):
            for f in files:
                if f.startswith("events.out.tfevents."):
                    event_files.append(os.path.join(root, f))
        return event_files

    def _load_metrics(self) -> Dict[str, Dict[int, float]]:
        metrics = defaultdict(dict)

        for event_file in self._event_files:
            ea = EventAccumulator(event_file)
            ea.Reload()
            tags = ea.Tags()["scalars"]
            for tag in tags:
                scalars = ea.Scalars(tag)
                for scalar in scalars:
                    step = scalar.step
                    value = scalar.value
                    if step not in metrics[tag] or value > metrics[tag][step]:
                        metrics[tag][step] = value
        return dict(metrics)

    def metric_exist(self, metric_name: str) -> bool:
        return metric_name in self._metrics

    def metric_min_step(self, metric_name: str) -> int:
        return min(self.metric_steps(metric_name))

    def metric_max_step(self, metric_name: str) -> int:
        return max(self.metric_steps(metric_name))

    def metric_steps(self, metric_name: str) -> List[int]:
        if not self.metric_exist(metric_name):
            raise ValueError(f"Metric '{metric_name}' does not exist.")
        return list(self._metrics[metric_name].keys())

    def metric_values(self, metric_name: str) -> List:
        if not self.metric_exist(metric_name):
            raise ValueError(f"Metric '{metric_name}' does not exist.")
        return list(self._metrics[metric_name].values())

    def metric_list(self, metric_prefix: str) -> List[str]:
        return [name for name in self._metrics if name.startswith(metric_prefix)]


class RayCleanupPlugin:
    @classmethod
    def _cleanup_ray_data_state(cls):
        """clean up the global states of Ray Data"""
        try:
            # reset execution context
            if hasattr(ray.data._internal.execution.streaming_executor, "_execution_context"):
                ray.data._internal.execution.streaming_executor._execution_context = None

            # trigger gc.collect() on all workers in the cluster
            ray._private.internal_api.global_gc()

            # clean up stats manager
            from ray.data._internal.stats import StatsManager

            if hasattr(StatsManager, "_instance"):
                StatsManager._instance = None

        except Exception:
            pass


class RayUnittestBase(unittest.TestCase, RayCleanupPlugin):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True, namespace="trinity_unittest")

        # erase existing resources
        cls._cleanup_ray_data_state()
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        ray.shutdown(_exiting_interpreter=True)


class RayUnittestBaseAsync(unittest.IsolatedAsyncioTestCase, RayCleanupPlugin):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True, namespace="trinity_unittest")

        # erase existing resources
        cls._cleanup_ray_data_state()
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        ray.shutdown(_exiting_interpreter=True)
