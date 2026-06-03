from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from trinity.algorithm.sample_strategy.utils import representative_sample
from trinity.buffer import get_buffer_reader
from trinity.common.config import BufferConfig
from trinity.common.experience import Experience
from trinity.utils.annotations import Deprecated
from trinity.utils.metrics import aggregate_metrics
from trinity.utils.timer import Timer


class SampleStrategy(ABC):
    def __init__(self, buffer_config: BufferConfig, **kwargs) -> None:
        pass

    def set_model_version_metric(self, exp_list: List[Experience], metrics: Dict):
        metric_list = [
            {"model_version": exp.info["model_version"]}
            for exp in exp_list
            if "model_version" in exp.info
        ]
        metrics.update(aggregate_metrics(metric_list, prefix="sample"))

    @abstractmethod
    async def sample(self, step: int) -> Tuple[List[Experience], Dict, List]:
        """Sample data from buffer.

        Args:
            step (`int`): The step number of current step.

        Returns:
            `List[Experience]`: The sampled List[Experience] data.
            `Dict`: Metrics for logging.
            `List`: Representative data for logging.
        """

    @classmethod
    @abstractmethod
    def default_args(cls) -> dict:
        """Get the default arguments of the sample strategy."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Get the state dict of the sample strategy."""

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dict of the sample strategy."""


class DefaultSampleStrategy(SampleStrategy):
    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
        self.exp_buffer = get_buffer_reader(buffer_config.trainer_input.experience_buffer)  # type: ignore[arg-type]

    async def sample(self, step: int, **kwargs) -> Tuple[List[Experience], Dict, List]:
        metrics = {}
        with Timer(metrics, "time/read_experience"):
            exp_list = await self.exp_buffer.read()
            repr_samples = representative_sample(exp_list)
        self.set_model_version_metric(exp_list, metrics)
        return exp_list, metrics, repr_samples

    @classmethod
    def default_args(cls) -> dict:
        return {}

    def state_dict(self) -> dict:
        return self.exp_buffer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict:
            self.exp_buffer.load_state_dict(state_dict)


class StalenessControlSampleStrategy(DefaultSampleStrategy):
    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
        self.max_staleness = kwargs.get("max_staleness", float("inf"))

    async def sample(self, step: int, **kwargs) -> Tuple[List[Experience], Dict, List]:
        min_model_version = max(step - self.max_staleness, 0)
        metrics = {}
        with Timer(metrics, "time/read_experience"):
            exp_list = await self.exp_buffer.read(min_model_version=min_model_version)
            repr_samples = representative_sample(exp_list)
        self.set_model_version_metric(exp_list, metrics)
        return exp_list, metrics, repr_samples


@Deprecated
class WarmupSampleStrategy(DefaultSampleStrategy):
    """The warmup sample strategy.
    Deprecated, keep this class for backward compatibility only.
    Please use DefaultSampleStrategy instead."""

    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
