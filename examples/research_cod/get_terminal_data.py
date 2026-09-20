# -*- coding: utf-8 -*-
"""Generate training and evaluation datasets for the terminal file-ops task.

Usage::

    python examples/research_cod/get_terminal_data.py \
        --local_dir examples/research_cod/data/terminal \
        --train_size 50000 \
        --test_size 200
"""

import argparse
import os
import random

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = "examples/research_cod/data/terminal"

# Task types that can appear in the dataset
SINGLE_TYPES = [
    "upload", "download", "rename", "move", "chmod",
    "delete", "copy", "pack", "mkdir",
]
COMPOSITE_TYPES = [
    "pack_upload", "download_extract", "mkdir_upload",
    "upload_chmod", "upload_delete_source", "pack_upload_extract",
    "download_rename", "backup_replace",
]
ALL_TYPES = SINGLE_TYPES + COMPOSITE_TYPES


def save_dataset_to_local(data_path: str, data: list, split: str) -> str:
    os.makedirs(data_path, exist_ok=True)
    df = pd.DataFrame(data)
    path = os.path.join(data_path, f"{split}.parquet")
    df.to_parquet(path)
    print(f"Saved {len(data)} samples to {path}")
    return path


def prepare_data(
    data_path: str,
    train_size: int = 50000,
    test_size: int = 200,
    seed: int = 42,
    composite_ratio: float = 0.5,
):
    np.random.seed(seed)

    # Generate unique seeds
    all_seeds = set()
    while len(all_seeds) < train_size + test_size:
        all_seeds.add(np.random.randint(0, 10_000_000))
    all_seeds = list(all_seeds)
    np.random.shuffle(all_seeds)

    train_seeds = all_seeds[:train_size]
    test_seeds = all_seeds[train_size:train_size + test_size]

    def make_record(task_seed, idx):
        # The task type is determined by the seed at runtime;
        # but we precompute it here for metadata.source (pack grouping).
        task_rng = random.Random(task_seed)
        task_rng.choice(["windows", "mac", "linux"])  # skip OS pick
        task_rng.randint(10, 99)  # skip remote host
        task_rng.choice(["admin", "deploy", "user", "webmaster", "devops", "ops", "ubuntu"])  # skip user
        is_composite = task_rng.random() < composite_ratio
        if is_composite:
            task_type = task_rng.choice(COMPOSITE_TYPES)
        else:
            task_type = task_rng.choice(SINGLE_TYPES)

        return {
            "seed": int(task_seed),
            "prompt": str(task_seed),
            "index": idx,
            "uid": f"terminal_{task_seed}",
            "metadata": {
                "source": task_type,
                "source_dataset": "terminal",
            },
        }

    train_data = [make_record(s, i) for i, s in enumerate(train_seeds)]
    test_data = [make_record(s, i) for i, s in enumerate(test_seeds)]

    save_dataset_to_local(data_path, train_data, "train")
    save_dataset_to_local(data_path, test_data, "test")

    # Print distribution
    from collections import Counter
    train_types = Counter(r["metadata"]["source"] for r in train_data)
    print(f"\nTask type distribution (train):")
    for t, c in sorted(train_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100 * c / train_size:.1f}%)")

    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate terminal task dataset")
    parser.add_argument("--local_dir", default=DEFAULT_DATA_PATH)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--composite_ratio", type=float, default=0.5)
    args = parser.parse_args()

    prepare_data(
        data_path=args.local_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        composite_ratio=args.composite_ratio,
    )
