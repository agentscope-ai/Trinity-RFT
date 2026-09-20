"""Generate train/test datasets for the CoD Optimal Control workflow.

The script supports reproducible random task generation with configurable
difficulty. A YAML config (e.g. the benchmark YAML) can be passed via
``--config``; any CLI arguments override the config values.
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from trinity.common.constants import TASKSET_PATH_ENV_VAR

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "optimal_control"
)

# Difficulty presets control the horizon and sampled state space. The hard
# preset keeps state scales bounded while moving targets farther away.
DIFFICULTY_PRESETS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "max_horizon": 8,
        "x0_min": -1.0,
        "x0_max": 1.0,
        "v0_min": -0.5,
        "v0_max": 0.5,
        "x_target_min": -3.0,
        "x_target_max": 3.0,
    },
    "medium": {
        "max_horizon": 12,
        "x0_min": -2.0,
        "x0_max": 2.0,
        "v0_min": -1.0,
        "v0_max": 1.0,
        "x_target_min": -5.0,
        "x_target_max": 5.0,
    },
    "hard": {
        "max_horizon": 12,
        "x0_min": -2.0,
        "x0_max": 2.0,
        "v0_min": -1.5,
        "v0_max": 1.5,
        "x_target_min": -8.0,
        "x_target_max": 8.0,
    },
}


def save_dataset_to_local(data_path: str, data: list[dict], split: str = "default") -> str:
    """Save dataset directly to local data_path."""
    os.makedirs(data_path, exist_ok=True)
    data_df = pd.DataFrame(data)
    dataset_path = os.path.join(data_path, f"{split}.parquet")
    data_df.to_parquet(dataset_path)
    print(
        f"Saved dataset optimal_control split '{split}' with {len(data)} examples at "
        f"{dataset_path}. Make sure to set the environment variable {TASKSET_PATH_ENV_VAR} "
        f"to {data_path}."
    )
    return dataset_path


def _sample_float(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample a single float in [low, high)."""
    return float(rng.uniform(low, high))


def _difficulty_params(
    difficulty: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve difficulty preset and apply per-field overrides."""
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. Choose from {list(DIFFICULTY_PRESETS.keys())}."
        )
    params = dict(DIFFICULTY_PRESETS[difficulty])
    # The control penalty is independent of difficulty by default.
    params.setdefault("control_penalty_coef", 0.03)
    params.setdefault("v_target", 0.0)
    # The workflow uses max_horizon directly as the rollout horizon.
    params.setdefault("max_horizon", 30)
    if overrides:
        for key, value in overrides.items():
            if value is not None and key in params:
                params[key] = value
    return params


def _generate_random_tasks(
    rng: np.random.Generator,
    size: int,
    params: Dict[str, Any],
    seed_offset: int,
) -> List[dict]:
    """Generate independent tasks whose hidden dynamics are resolved at runtime."""
    return [
        {
            "x0": _sample_float(rng, params["x0_min"], params["x0_max"]),
            "v0": _sample_float(rng, params["v0_min"], params["v0_max"]),
            "x_target": _sample_float(
                rng,
                params["x_target_min"],
                params["x_target_max"],
            ),
            "v_target": float(params["v_target"]),
            "max_horizon": int(params["max_horizon"]),
            "control_penalty_coef": float(params["control_penalty_coef"]),
            "seed": seed_offset + task_idx,
        }
        for task_idx in range(size)
    ]


def prepare_optimal_control_data(
    data_path: str,
    train_size: int = 128,
    test_size: int = 16,
    difficulty: str = "medium",
    train_seed: int = 42,
    test_seed: int = 2024,
    **range_overrides: Any,
) -> Tuple[List[dict], List[dict]]:
    """Generate pack-independent train and test splits."""
    params = _difficulty_params(difficulty, range_overrides)
    train_data = _generate_random_tasks(
        np.random.default_rng(train_seed),
        train_size,
        params,
        seed_offset=0,
    )
    test_data = _generate_random_tasks(
        np.random.default_rng(test_seed),
        test_size,
        params,
        seed_offset=train_size,
    )
    save_dataset_to_local(data_path, train_data, "train")
    save_dataset_to_local(data_path, test_data, "test")
    return train_data, test_data


def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load an optional YAML config and extract ``task_generation`` settings."""
    if not config_path:
        return {}
    cfg = OmegaConf.load(config_path)
    task_gen = cfg.get("task_generation", {})
    if not task_gen:
        return {}
    return OmegaConf.to_container(task_gen, resolve=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--config", default=None, help="Path to a YAML config with a 'task_generation' section."
    )
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument("--test_seed", type=int, default=None)
    parser.add_argument("--max_horizon", type=int, default=None)
    parser.add_argument("--control_penalty_coef", type=float, default=None)
    parser.add_argument("--x0_min", type=float, default=None)
    parser.add_argument("--x0_max", type=float, default=None)
    parser.add_argument("--v0_min", type=float, default=None)
    parser.add_argument("--v0_max", type=float, default=None)
    parser.add_argument("--x_target_min", type=float, default=None)
    parser.add_argument("--x_target_max", type=float, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Config values are the base; CLI args override them.
    config = _load_config(args.config)

    def get(name: str, default: Any) -> Any:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            return cli_value
        return config.get(name, default)

    train_size = int(get("train_size", 128))
    test_size = int(get("test_size", 16))
    difficulty = str(get("difficulty", "medium"))
    train_seed = int(get("train_seed", 42))
    test_seed = int(get("test_seed", 2024))

    range_overrides = {
        "max_horizon": get("max_horizon", None),
        "control_penalty_coef": get("control_penalty_coef", None),
        "x0_min": get("x0_min", None),
        "x0_max": get("x0_max", None),
        "v0_min": get("v0_min", None),
        "v0_max": get("v0_max", None),
        "x_target_min": get("x_target_min", None),
        "x_target_max": get("x_target_max", None),
    }
    # Filter out unset overrides so the difficulty preset is not overwritten.
    range_overrides = {k: v for k, v in range_overrides.items() if v is not None}

    train_data, test_data = prepare_optimal_control_data(
        data_path=args.local_dir,
        train_size=train_size,
        test_size=test_size,
        difficulty=difficulty,
        train_seed=train_seed,
        test_seed=test_seed,
        **range_overrides,
    )

    print(f"Train dataset: {len(train_data)} examples")
    print(f"Test dataset: {len(test_data)} examples")
    print("Sample train example:", train_data[0])
    print("Sample test example:", test_data[0])


if __name__ == "__main__":
    main()
