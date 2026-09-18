"""
Generate FrozenLake datasets with configurable map size range.
Modified from examples/grpo_frozen_lake/get_frozen_lake_data.py
"""
import argparse
import os

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frozen_lake")


def save_dataset_to_local(data_path: str, data: list[dict], split: str = "default") -> str:
    os.makedirs(data_path, exist_ok=True)
    data_df = pd.DataFrame(data)
    dataset_path = os.path.join(data_path, f"{split}.parquet")
    data_df.to_parquet(dataset_path)
    print(f"Saved split '{split}' with {len(data)} examples at {dataset_path}")
    return dataset_path


def prepare_frozenlake_data(
        data_path,
        train_size=10000,
        test_size=100,
        map_min_size=4,
        map_max_size=8,
        tile_min_prob=0.6,
        tile_max_prob=0.85,
    ):
    np.random.seed(42)

    train_seeds = np.random.randint(0, 100000, size=train_size)
    test_seeds = np.random.randint(0, 100000, size=test_size)
    # randint is [low, high), so use max_size+1 to include max_size
    train_sizes = np.random.randint(map_min_size, map_max_size + 1, size=train_size)
    test_sizes = np.random.randint(map_min_size, map_max_size + 1, size=test_size)
    # p is the probability of frozen tile, i.e., 1 - p is the probability of hole
    train_ps = np.random.uniform(tile_min_prob, tile_max_prob, size=train_size)
    test_ps = np.random.uniform(tile_min_prob, tile_max_prob, size=test_size)

    def frozenlake_process_fn(seed, size, p, idx):
        return {"seed": seed, "size": size, "p": p, "index": idx, "uid": f"{seed}_{size}_{p}"}

    train_data = [
        frozenlake_process_fn(seed, train_sizes[idx], train_ps[idx], idx)
        for idx, seed in enumerate(train_seeds)
    ]
    test_data = [
        frozenlake_process_fn(seed, test_sizes[idx], test_ps[idx], idx)
        for idx, seed in enumerate(test_seeds)
    ]

    save_dataset_to_local(data_path, train_data, "train")
    save_dataset_to_local(data_path, test_data, "test")

    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=DEFAULT_DATA_PATH)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--map_min_size", type=int, default=4)
    parser.add_argument("--map_max_size", type=int, default=5)
    parser.add_argument("--tile_min_prob", type=float, default=0.6)  # tile prob: larger is easier
    parser.add_argument("--tile_max_prob", type=float, default=0.7)
    args = parser.parse_args()

    train_data, test_data = prepare_frozenlake_data(
        data_path=args.local_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        map_min_size=args.map_min_size,
        map_max_size=args.map_max_size,
        tile_min_prob=args.tile_min_prob,
        tile_max_prob=args.tile_max_prob,
    )

    print(f"\nTrain: {len(train_data)} examples")
    print(f"Test: {len(test_data)} examples")
    print(f"Size range: [{args.map_min_size}, {args.map_max_size}]")
    print(f"Sample: {train_data[0]}")
