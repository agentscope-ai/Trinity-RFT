"""Generate seed-only datasets for the CoD grid-navigation environment."""

import argparse
import os

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "grid_navigation"
)


def save_dataset_to_local(data_path: str, data: list[dict], split: str) -> str:
    """Write one dataset split as a parquet file."""
    os.makedirs(data_path, exist_ok=True)
    dataset_path = os.path.join(data_path, f"{split}.parquet")
    pd.DataFrame(data).to_parquet(dataset_path)
    print(f"Saved split '{split}' with {len(data)} examples at {dataset_path}")
    return dataset_path


def prepare_grid_navigation_data(
    data_path: str,
    train_size: int,
    test_size: int,
    seed: int,
):
    """Generate disjoint train and test task seeds and save both splits."""
    rng = np.random.default_rng(seed)
    all_seeds = rng.choice(10_000_000, size=train_size + test_size, replace=False)
    train_seeds = all_seeds[:train_size]
    test_seeds = all_seeds[train_size:]

    def process_fn(task_seed: int, index: int) -> dict:
        task_seed = int(task_seed)
        return {
            "seed": task_seed,
            "index": index,
            "uid": f"grid_navigation_{task_seed}",
        }

    train_data = [process_fn(task_seed, i) for i, task_seed in enumerate(train_seeds)]
    test_data = [process_fn(task_seed, i) for i, task_seed in enumerate(test_seeds)]

    save_dataset_to_local(data_path, train_data, "train")
    save_dataset_to_local(data_path, test_data, "test")
    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CoD grid-navigation data")
    parser.add_argument("--local_dir", default=DEFAULT_DATA_PATH)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--test_size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_data, test_data = prepare_grid_navigation_data(
        data_path=args.local_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"\nTrain: {len(train_data)} examples")
    print(f"Test: {len(test_data)} examples")
    print(f"Sample: {train_data[0]}")
