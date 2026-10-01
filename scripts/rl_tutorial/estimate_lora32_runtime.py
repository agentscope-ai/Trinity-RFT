#!/usr/bin/env python3
"""Recompute the published LoRA32 timing estimate without server access.

Only the Python standard library and the adjacent timing JSON are required.
Source hashes identify the archived inputs; this checks the published details
and arithmetic, not the original log bytes. No training or evaluation is run.
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


DEFAULT_DATA = Path(__file__).parent / "sample_data" / "lora32_runner72_timing.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_number(actual, expected, label):
    require(
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8),
        f"{label}: stored {actual!r}, recomputed {expected!r}",
    )


def recompute(data):
    """Validate all intervals and recompute summaries and budget arithmetic."""
    require(data["schema_version"] == 1, "Unsupported schema_version")
    window = data["window"]
    require(window["runners"] == 72, "Expected the 72-runner stage")
    require(
        (window["first_step"], window["last_step"]) == (103, 150),
        "Expected the complete 103-150 timing window",
    )
    rows = data["observations"]
    require([r["step"] for r in rows] == list(range(103, 151)), "Missing or repeated steps")
    endpoint = window["previous_endpoint"]
    require(endpoint["step"] == 102, "The first interval needs the step-102 endpoint")

    def timestamp(raw):
        return datetime.strptime(
            f"{window['calendar_year']}-{raw}",
            "%Y-" + window["timestamp_raw_format"],
        )

    previous = timestamp(endpoint["timestamp_raw"])
    previous_line = endpoint["source_line"]
    intervals = []
    for row in rows:
        current = timestamp(row["timestamp_raw"])
        elapsed = (current - previous).total_seconds()
        require(elapsed > 0, f"Step {row['step']}: timestamp does not advance")
        require(row["source_line"] > previous_line, "Source lines must advance")
        check_number(row["wall_seconds"], elapsed, f"Step {row['step']} wall_seconds")
        for field in ("read_experience_seconds", "train_step_seconds", "total_tokens"):
            value = row[field]
            require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0,
                f"Step {row['step']}: invalid {field}",
            )
        intervals.append(elapsed)
        previous, previous_line = current, row["source_line"]

    summary = {
        "observed_steps": len(rows),
        "total_wall_seconds": sum(intervals),
        "mean_wall_seconds": statistics.mean(intervals),
        "mean_wall_minutes": statistics.mean(intervals) / 60,
        "median_wall_seconds": statistics.median(intervals),
        "min_wall_seconds": min(intervals),
        "max_wall_seconds": max(intervals),
        "total_read_experience_seconds": sum(r["read_experience_seconds"] for r in rows),
        "total_train_step_seconds": sum(r["train_step_seconds"] for r in rows),
        "mean_other_wall_seconds": statistics.mean(
            seconds - row["read_experience_seconds"] - row["train_step_seconds"]
            for seconds, row in zip(intervals, rows)
        ),
        "total_tokens": sum(r["total_tokens"] for r in rows),
    }
    for name, value in summary.items():
        check_number(data["summary"][name], value, f"summary.{name}")

    require([b["steps"] for b in data["budgets"]] == [100, 130, 150], "Unexpected budgets")
    budgets = []
    for stored in data["budgets"]:
        seconds = sum(intervals) * stored["steps"] / len(rows)
        hours = seconds / 3600
        check_number(stored["estimated_seconds"], seconds, "budget.estimated_seconds")
        check_number(stored["estimated_hours"], hours, "budget.estimated_hours")
        budgets.append({"steps": stored["steps"], "estimated_seconds": seconds, "estimated_hours": hours})
    return {"summary": summary, "budgets": budgets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Timing JSON to validate")
    parser.add_argument("--json", action="store_true", help="Print recomputed numbers as JSON")
    args = parser.parse_args()
    try:
        result = recompute(json.loads(args.data.read_text(encoding="utf-8")))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"Timing validation failed: {exc}\n")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    summary = result["summary"]
    print(f"Verified 48 consecutive steps (103-150): {summary['total_wall_seconds']:g} seconds.")
    print(f"Observed mean: {summary['mean_wall_minutes']:.6f} minutes/step.")
    for budget in result["budgets"]:
        print(f"Estimated {budget['steps']} steps: {budget['estimated_hours']:.1f} hours.")
    print("Estimates exclude setup, initial filling and independent evaluation; no fresh fixed-72 run was measured.")


if __name__ == "__main__":
    main()
