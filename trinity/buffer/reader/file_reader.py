"""Filed based buffer reader."""

from typing import List, Optional, Tuple

import datasets
from datasets import Dataset, load_dataset

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.schema import FORMATTER
from trinity.common.config import StorageConfig


class DummyProgressBar:
    def __init__(self):
        pass

    def update(self, num: int):
        pass

    def close(self):
        pass


class _HFBatchReader:
    def __init__(
        self,
        dataset: Dataset,
        name: str,
        default_batch_size: int,
        total_epochs: int = 1,
        offset: int = 0,
        drop_last: bool = True,
        total_steps: Optional[int] = None,
        enable_progress_bar: Optional[bool] = True,
    ):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        if self.dataset_size == 0:
            raise ValueError(f"Dataset [{name}] is empty and cannot be read in batches.")
        self.name = name
        self.current_batch_size = None
        self.drop_last = drop_last

        self.current_offset = offset

        # convert epochs/steps to sample number
        if total_steps is not None:
            self.total_samples = default_batch_size * total_steps
        else:
            self.total_samples = self.dataset_size * total_epochs

        if enable_progress_bar:
            from ray.experimental.tqdm_ray import tqdm

            self.progress_bar = tqdm(
                total=self.total_samples,
                desc=f"Dataset [{self.name}] Progressing",
            )
        else:
            self.progress_bar = DummyProgressBar()

        self.progress_bar.update(self.current_offset)

    def read_batch(self, batch_size: int) -> Tuple[List, List]:
        batch, indices = [], []
        while len(batch) < batch_size:
            if self.current_offset >= self.total_samples:
                if not self.drop_last and len(batch) > 0:
                    break
                self.progress_bar.close()
                raise StopIteration
            index = self.current_offset % self.dataset_size
            batch.append(self.dataset[index])
            indices.append(index)
            self.current_offset += 1

        self.progress_bar.update(len(batch))
        return batch, indices

    def select_batch(self, indices: List[int]) -> List:
        batch = []
        for i in indices:
            assert 0 <= i < self.dataset_size
            if self.current_offset >= self.total_samples:
                if not self.drop_last and len(batch) > 0:
                    break
                self.progress_bar.close()
                raise StopIteration
            batch.append(self.dataset[int(i)])
            self.current_offset += 1

        self.progress_bar.update(len(batch))  # update progress bar
        return batch


class BaseFileReader(BufferReader):
    async def read_async(self, batch_size: Optional[int] = None, **kwargs):
        try:
            return self.read(batch_size, **kwargs)
        except StopIteration as e:
            raise StopAsyncIteration from e


class _DatasetFileReader(BaseFileReader):
    def __init__(self, config: StorageConfig):
        self.config = config
        self.name = config.name
        self.read_batch_size = config.batch_size
        self.formatter, self.dataset = self._init_formatter_and_dataset(config)
        self._init_selector(config)

    def _init_formatter_and_dataset(self, config: StorageConfig):
        raise NotImplementedError

    def _init_selector(self, config: StorageConfig):
        if config.data_selector is not None:
            from trinity.buffer.selector import SELECTORS
            from trinity.buffer.selector.selector import BaseSelector

            self.selector: BaseSelector = SELECTORS.get(config.data_selector.selector_type)(
                self.dataset, config.data_selector
            )
        else:
            self.selector = None

    def _read_samples(self, batch_size: int) -> Tuple[List, List]:
        if self.selector is not None:
            indices = self.selector.get_indices(batch_size)
            samples = self.dataset.select_batch(indices)
            return samples, indices
        return self.dataset.read_batch(batch_size)

    def state_dict(self):
        if self.selector is not None:
            return self.selector.state_dict()
        return {"current_index": self.dataset.current_offset}

    def load_state_dict(self, state_dict):
        if self.selector is not None:
            self.selector.load_state_dict(state_dict)
        else:
            self.dataset.current_offset = state_dict["current_index"]

    def __len__(self):
        return self.dataset.dataset_size

    def _convert_batch(self, samples: List, indices: List) -> List:
        raise NotImplementedError

    def read(self, batch_size: Optional[int] = None, **kwargs) -> List:
        batch_size = batch_size or self.read_batch_size
        samples, indices = self._read_samples(batch_size)
        return self._convert_batch(samples, indices)


class FileReader(BaseFileReader):
    """Provide a unified interface for Experience and Task file readers."""

    def __init__(self, config: StorageConfig):
        if config.schema_type and config.schema_type != "task":
            self.reader = ExperienceFileReader(config)
        else:
            self.reader = TaskFileReader(config)

    def read(self, batch_size: Optional[int] = None, **kwargs) -> List:
        return self.reader.read(batch_size, **kwargs)

    def state_dict(self):
        return self.reader.state_dict()

    def load_state_dict(self, state_dict):
        return self.reader.load_state_dict(state_dict)

    def feedback(self, **pipeline_metrics):
        if self.reader.selector is not None:
            self.reader.selector.feedback(**pipeline_metrics)

    def __len__(self):
        return self.reader.__len__()


class ExperienceFileReader(_DatasetFileReader):
    """Reader for SFT / DPO file data."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)

    def _init_formatter_and_dataset(self, config: StorageConfig):
        formatter = FORMATTER.get(config.schema_type)(
            tokenizer_path=config.tokenizer_path, format_config=config.format
        )
        dataset = _HFBatchReader(
            load_dataset(config.path, name=config.subset_name, split=config.split),
            name=config.name,
            default_batch_size=self.read_batch_size,
            total_epochs=config.total_epochs,
            drop_last=True,
            total_steps=config.total_steps,
            enable_progress_bar=config.enable_progress_bar,
        )
        return formatter, dataset

    def _convert_batch(self, samples: List, indices: List) -> List:
        exp_list = []
        for sample in samples:
            experience = self.formatter.format(sample)
            exp_list.append(experience)
        return exp_list


class TaskFileReader(_DatasetFileReader):
    """A Reader for task file data."""

    def __init__(self, config: StorageConfig):
        datasets.disable_caching()
        super().__init__(config)

    def _init_formatter_and_dataset(self, config):
        formatter = FORMATTER.get("task")(config)
        dataset = _HFBatchReader(
            load_dataset(config.path, name=config.subset_name, split=config.split),
            name=config.name,
            default_batch_size=self.read_batch_size,
            total_epochs=config.total_epochs if not config.is_eval else 1,
            offset=config.index,
            drop_last=not config.is_eval,
            total_steps=config.total_steps if not config.is_eval else None,
            enable_progress_bar=config.enable_progress_bar,
        )
        return formatter, dataset

    def _convert_batch(self, samples: List, indices: List) -> List:
        tasks = []
        for sample, index in zip(samples, indices):
            task = self.formatter.format(sample)
            task.index["index"] = int(index)
            tasks.append(task)
        return tasks
