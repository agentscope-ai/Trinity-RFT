import os
from numbers import Number
from typing import Any, Dict, List, Literal, Optional

from data_juicer.config import get_init_configs, prepare_side_configs
from data_juicer.utils.constant import Fields
from datasets import Dataset
from jsonargparse import Namespace
from pydantic import BaseModel, model_validator


class DJConfig(BaseModel):
    pipeline_type: Literal["task", "experience"] = "experience"

    # For both `task` and `experience`
    operators: Optional[List[Dict[str, Dict[str, Any]]]] = None
    config_path: Optional[str] = None
    np: int = 4

    # For `task` only
    executor_type: Literal["ray", "default"] = "default"
    inputs: List[str] = []  # List of input files
    output_dir: Optional[str] = None
    target_fields: List[str] = []  # fields in the output dataset
    priority_weights: Dict[str, float] = {}  # weights for priority computing
    top_k: int = -1  # number of samples to select after task pipeline. -1 means all
    order_method: Literal["keep", "shuffle", "sort", "folding"] = "sort"
    order_args: Dict = {}

    @model_validator(mode="after")
    def check_dj_config(self):
        if not (self.config_path or self.operators):
            raise ValueError("Must provide at least one of config_path or operators.")
        if self.np <= 0:
            raise ValueError("np must be a positive integer.")
        return self


def parse_config(config: DJConfig) -> Namespace:
    """Convert Trinity config to DJ config"""
    if config.config_path is not None:
        task_config = prepare_side_configs(config.config_path)
        task_config = get_init_configs(task_config)
        return task_config

    if config.pipeline_type == "experience":
        return _parse_experience_pipeline_config(config)
    elif config.pipeline_type == "task":
        return _parse_task_pipeline_config(config)
    else:
        raise ValueError(f"Unknown pipeline type: {config.pipeline_type}")


def _parse_experience_pipeline_config(config: DJConfig) -> Namespace:
    """Parse the experience pipeline configuration."""
    if config.operators is not None:
        exp_config = Namespace(process=[op for op in config.operators], np=config.np)
        exp_config = get_init_configs(exp_config)
    else:
        raise ValueError("At least one of operators or config_path should be provided.")
    return exp_config


def _parse_task_pipeline_config(config: DJConfig) -> Namespace:
    """Parse the task pipeline configuration."""
    if config.operators is not None:
        for input in config.inputs:
            if not os.path.exists(input):
                raise FileNotFoundError(f"{input} does not exist.")
            if not os.path.isfile(input):
                raise ValueError(
                    f"{input} is not a file. Currently, the task pipeline only supports processing files."
                )
        if config.output_dir is None:
            raise ValueError("`output_dir` must be set for task pipeline.")
        os.makedirs(config.output_dir, exist_ok=True)
        task_config = Namespace(
            process=[op for op in config.operators],
            np=config.np,
            dataset={
                "configs": [
                    {
                        "type": "local",
                        "weight": 1.0,
                        "path": path,
                    }
                    for path in config.inputs
                ]
            },
            text_keys=config.target_fields,
            export_shard_size=128 * 1024 * 1024,  # 128 MB
        )
        task_config = get_init_configs(task_config)
    else:
        raise ValueError("At least one of operators or config_path should be provided.")
    return task_config


# For task pipeline
DIMENSION_STATS_KEYS = {
    "quality_score": {
        "alnum_ratio": {"better": "higher", "range": [0.0, 1.0]},
        "char_rep_ratio": {"better": "lower", "range": [0.0, 1.0]},
        "flagged_words_ratio": {"better": "lower", "range": [0.0, 1.0]},
        "special_char_ratio": {"better": "lower", "range": [0.0, 1.0]},
        "stopwords_ratio": {"better": "higher", "range": [0.0, 1.0]},
        "word_rep_ratio": {"better": "lower", "range": [0.0, 1.0]},
        "llm_quality_score": {"better": "higher", "range": [0.0, 1.0]},
    },
    "difficulty_score": {
        "perplexity": {"better": "higher", "range": [0.0, None]},
        "lang_score": {"better": "lower", "range": [0.0, 1.0]},
        "llm_difficulty_score": {"better": "higher", "range": [0.0, 1.0]},
    },
}


def group_scores(dataset: Dataset) -> Dataset:
    if Fields.stats not in dataset.features or len(dataset) == 0:
        return dataset
    # for perplexity, normalize them with the max value.
    stats_min_max = {}
    for stats in dataset[Fields.stats][0]:
        all_stats = [
            sample[Fields.stats][stats] for sample in dataset.data if Fields.stats in sample
        ]
        if len(all_stats) > 0 and isinstance(all_stats[0], Number):
            stats_min_max[stats] = [min(all_stats), max(all_stats)]

    def _group_single(sample):
        stats = sample[Fields.stats]
        for group_score, related_stats in DIMENSION_STATS_KEYS.items():
            total_score = 0.0
            hit_cnt = 0
            details = {}
            for stats_key in related_stats:
                stats_meta = related_stats[stats_key]
                if stats_key in stats:
                    # min-max normalization
                    min_val, max_val = stats_meta["range"]
                    if min_val is None or max_val is None:
                        min_val, max_val = stats_min_max[stats_key]
                    current_score = (stats[stats_key] - min_val) / (max_val - min_val)
                    if stats_meta["better"] == "lower":
                        current_score = 1.0 - current_score
                    total_score += current_score
                    hit_cnt += 1
                    # record original stats
                    details[stats_key] = stats[stats_key]
                    # record normalized score
                    details[f"normalized_{stats_key}"] = current_score
            final_score = total_score / hit_cnt if hit_cnt > 0 else 0.0
            sample[Fields.stats][group_score] = final_score
            sample[group_score] = final_score
            sample[f"{group_score}_detail"] = details
        return sample

    dataset = dataset.map(_group_single)
    return dataset


def compute_priority_scores(
    sample,
    priority_weights,
) -> float:
    """Combine different factors into final priority score"""
    if "priority" in sample:
        return sample

    from data_juicer.utils.constant import Fields

    if Fields.stats not in sample:
        return sample
    stats = sample[Fields.stats]
    if isinstance(stats, list):
        stats = stats[0]
    score = 0.0

    # Usage frequency penalty
    if "usage_frequency" in priority_weights:
        freq = stats.get("consumed_cnt", 0)
        # normalized_freq = min(freq / 10.0, 1.0)  # Normalize to [0,1]
        score += priority_weights["usage_frequency"] * freq

    # Data quality score
    if "quality" in priority_weights:
        quality = stats.get("quality_score", 0.5)
        score += priority_weights["quality"] * quality

    # Data difficulty score
    if "difficulty" in priority_weights:
        difficulty = stats.get("difficulty_score", 0.5)
        score += priority_weights["difficulty"] * difficulty

    sample["priority"] = [score] if isinstance(sample[Fields.stats], list) else score
    return sample
