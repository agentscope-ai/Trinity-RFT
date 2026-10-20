"""
Create the Trinity taskset for `sqlite_sql_workflow` from BIRD.

Download the BIRD train set (https://bird-bench.github.io/) and BIRD Mini-Dev
(https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip), then run:

    python examples/grpo_sqlite_sql/get_bird_data.py \
        --train_json train/train.json \
        --test_json minidev/MINIDEV/mini_dev_sqlite.json

This writes `examples/grpo_sqlite_sql/bird_data/train.jsonl` and `test.jsonl` with the fields
the workflow reads: `db_id`, `question`, `evidence` and `SQL`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("db_id", "question", "evidence", "SQL")


def load(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def write(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            row = {key: record.get(key) or "" for key in FIELDS}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} tasks to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("examples/grpo_sqlite_sql/bird_data")
    )
    args = parser.parse_args()
    write(load(args.train_json), args.output_dir / "train.jsonl")
    write(load(args.test_json), args.output_dir / "test.jsonl")


if __name__ == "__main__":
    main()
