"""
We use this script to create the huggingface format dataset files for the InterCode-SQL dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/sql/spider/ic_spider_dev.json"
)
RAW_DATA_FILENAME = "ic_spider_dev.json"
DEFAULT_TEST_SIZE = 200
DEFAULT_SEED = 42


def download_file(url: str, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        print(f"Using existing raw data: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading InterCode-SQL data from {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved raw data to {output_path}")


def load_records(data_path: Path) -> list[dict[str, Any]]:
    with data_path.open() as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {data_path}")
    return records


def build_examples(
    records: list[dict[str, Any]],
    raw_data_path: Path,
) -> list[dict[str, Any]]:
    examples = []
    for idx, record in enumerate(records):
        examples.append(
            {
                "query_idx": idx,
                "data_path": str(raw_data_path),
                "db": record.get("db", ""),
                "query": record.get("query", ""),
                "difficulty": record.get("difficulty", record.get("hardness", "")),
                "target": "",
            }
        )
    return examples


def split_examples(
    examples: list[dict[str, Any]],
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if test_size < 0:
        raise ValueError("test_size must be non-negative")
    if test_size > len(examples):
        raise ValueError(f"test_size {test_size} exceeds available examples {len(examples)}")

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    test_indices = set(indices[:test_size])
    train_indices = [idx for idx in indices if idx not in test_indices]
    test_indices_ordered = indices[:test_size]
    return (
        [examples[idx] for idx in train_indices],
        [examples[idx] for idx in test_indices_ordered],
    )


def create_dataset_files(
    output_dir: Path,
    data_url: str,
    test_size: int,
    seed: int,
    overwrite: bool,
) -> None:
    raw_data_path = output_dir / "raw" / RAW_DATA_FILENAME
    download_file(data_url, raw_data_path, overwrite=overwrite)

    records = load_records(raw_data_path)
    examples = build_examples(records, raw_data_path)
    train_data, test_data = split_examples(examples, test_size, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict = {"train": train_data, "test": test_data}

    for split, data in dataset_dict.items():
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    dataset_info = {
        "citation": "",
        "description": "Custom dataset",
        "splits": {
            "train": {"name": "train", "num_examples": len(train_data)},
            "test": {"name": "test", "num_examples": len(test_data)},
        },
    }
    with (output_dir / "dataset_dict.json").open("w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Created {len(train_data)} train and {len(test_data)} test examples in {output_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=Path, default=None)
    parser.add_argument("--data_url", type=str, default=DEFAULT_DATA_URL)
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    local_dir = args.local_dir
    if local_dir is None:
        local_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "intercode_sql_data"

    create_dataset_files(
        output_dir=local_dir,
        data_url=args.data_url,
        test_size=args.test_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
