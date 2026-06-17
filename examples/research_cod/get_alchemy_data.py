"""Generate Alchemy datasets with configurable size."""

import argparse
import os

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "alchemy"
)


def save_dataset_to_local(data_path: str, data: list, split: str = "default") -> str:
    os.makedirs(data_path, exist_ok=True)
    data_df = pd.DataFrame(data)
    dataset_path = os.path.join(data_path, f"{split}.parquet")
    data_df.to_parquet(dataset_path)
    print(f"Saved split '{split}' with {len(data)} examples at {dataset_path}")
    return dataset_path


def prepare_alchemy_data(data_path: str, train_size: int, test_size: int, seed: int):
    np.random.seed(seed)

    all_seeds = np.random.choice(10_000_000, size=train_size + test_size, replace=False)
    train_seeds = all_seeds[:train_size]
    test_seeds = all_seeds[train_size:]

    def process_fn(seed_val, idx):
        return {"seed": int(seed_val), "index": idx, "uid": f"alchemy_{seed_val}"}

    train_data = [process_fn(s, i) for i, s in enumerate(train_seeds)]
    test_data = [process_fn(s, i) for i, s in enumerate(test_seeds)]

    save_dataset_to_local(data_path, train_data, "train")
    save_dataset_to_local(data_path, test_data, "test")

    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Alchemy dataset")
    parser.add_argument("--local_dir", default=DEFAULT_DATA_PATH)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_data, test_data = prepare_alchemy_data(
        data_path=args.local_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"\nTrain: {len(train_data)} examples")
    print(f"Test: {len(test_data)} examples")
    print(f"Sample: {train_data[0]}")
