# -*- coding: utf-8 -*-
"""Re-split learn2ask train+test into an OOD split by disease (diagn).

Merges the existing train.jsonl + test.jsonl produced by
2_build_dataset.py, then routes whole diagn groups into the new
test split so train/test diagns are disjoint. Default picks diagns
greedily (random seeded order) until the cumulative sample count
reaches `--test_ratio * total`; pass `--test_diagns` to override
with an explicit list.

Prints the diagn distribution at the start of every run so the
ratio target / explicit list can be sanity-checked against actual
data.
"""
import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, samples: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def pick_test_diagns_by_ratio(
    diagn_counts: Dict[str, int],
    target_samples: int,
    seed: int,
) -> List[str]:
    """Greedy: shuffle diagns under `seed`, append until cumulative
    sample count reaches the target. Overshoot up to one diagn is
    accepted — use --test_diagns for finer control."""
    diagns = list(diagn_counts.keys())
    random.Random(seed).shuffle(diagns)
    chosen, acc = [], 0
    for d in diagns:
        if acc >= target_samples:
            break
        chosen.append(d)
        acc += diagn_counts[d]
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input_train",
        default="examples/research_cod/data/learn2ask/train.jsonl",
    )
    parser.add_argument(
        "--input_test",
        default="examples/research_cod/data/learn2ask/test.jsonl",
    )
    parser.add_argument(
        "--output_train",
        default="examples/research_cod/data/learn2ask_ood/train.jsonl",
    )
    parser.add_argument(
        "--output_test",
        default="examples/research_cod/data/learn2ask_ood/test.jsonl",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Target fraction of samples routed to test, picked by diagn group.",
    )
    parser.add_argument(
        "--test_diagns",
        type=str,
        default=None,
        help="Comma-separated explicit diagn list. Overrides --test_ratio.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show_dist_only",
        action="store_true",
        help="Print the diagn distribution and exit without writing splits.",
    )
    args = parser.parse_args()

    train_in = load_jsonl(args.input_train)
    test_in = load_jsonl(args.input_test)
    samples = train_in + test_in
    print(
        f"Loaded {len(samples)} samples "
        f"({len(train_in)} train + {len(test_in)} test) from {args.input_train} / {args.input_test}"
    )

    counts = Counter(s.get("diagn", "(missing)") for s in samples)
    total = sum(counts.values())

    print(f"\nDiagn distribution ({len(counts)} unique diagns):")
    for d, n in counts.most_common():
        print(f"  {n:6d}  ({100 * n / total:5.2f}%)  {d}")

    if args.show_dist_only:
        return

    if args.test_diagns:
        test_diagns = {d.strip() for d in args.test_diagns.split(",") if d.strip()}
        unknown = test_diagns - set(counts)
        if unknown:
            raise ValueError(f"--test_diagns contains unknown diagns: {sorted(unknown)}")
    else:
        target = int(total * args.test_ratio)
        test_diagns = set(pick_test_diagns_by_ratio(counts, target, args.seed))

    train_samples = [s for s in samples if s.get("diagn") not in test_diagns]
    test_samples = [s for s in samples if s.get("diagn") in test_diagns]

    write_jsonl(args.output_train, train_samples)
    write_jsonl(args.output_test, test_samples)

    train_diagns = set(counts) - test_diagns
    print("\nSplit summary:")
    print(
        f"  train: {len(train_samples)} samples / {len(train_diagns)} diagns "
        f"-> {args.output_train}"
    )
    print(
        f"  test:  {len(test_samples)} samples / {len(test_diagns)} diagns "
        f"({100 * len(test_samples) / total:.2f}% of total) -> {args.output_test}"
    )
    print(f"\nTest diagns ({len(test_diagns)}): {sorted(test_diagns)}")


if __name__ == "__main__":
    main()
