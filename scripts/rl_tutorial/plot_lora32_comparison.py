#!/usr/bin/env python3
"""Plot measured learning traces and a separate estimated LoRA runtime budget."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "scripts/rl_tutorial/sample_data"


def main():
    fig, (score_ax, time_ax) = plt.subplots(
        1, 2, figsize=(11, 4.5), width_ratios=[1.9, 1], layout="constrained"
    )
    for filename, label, color in [
        ("verl_baseline_metrics.csv", "Full parameter / verl", "#6554b5"),
        ("lora32_speed_metrics.csv", "LoRA32 / TuFT", "#007f86"),
    ]:
        with (DATA / filename).open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if int(row["step"]) <= 150]
        # Do not smooth across LoRA's documented restores.
        segments = [(1, 150)] if "verl" in filename else [(1, 10), (11, 90), (91, 100), (101, 150)]
        for index, (start, end) in enumerate(segments):
            selected = [row for row in rows if start <= int(row["step"]) <= end]
            x = [int(row["step"]) for row in selected]
            y = [float(row["score"]) for row in selected]
            mean = [sum(y[max(0, i-9):i+1]) / len(y[max(0, i-9):i+1]) for i in range(len(y))]
            score_ax.plot(x, y, color=color, alpha=.2, lw=.7)
            score_ax.plot(x, mean, color=color, lw=2, label=label if index == 0 else None)
    score_ax.set(xlabel="Trainer log step", ylabel="Training score (turn-weighted)",
                 title="Learning traces: first 150 steps", ylim=(-.15, 1.02))
    score_ax.legend(loc="upper left", frameon=False, fontsize=9)
    score_ax.text(.48, .02, "Bold: trailing mean, up to 10 points\nNot held-out episode success rate",
                  transform=score_ax.transAxes, fontsize=8, color="#555")
    timing = json.loads((DATA / "lora32_runner72_timing.json").read_text())
    budgets = timing["budgets"]
    hours = [budget["estimated_hours"] for budget in budgets]
    bars = time_ax.bar([f'{budget["steps"]} steps' for budget in budgets], hours,
                       color="#007f86", alpha=.65, hatch="//", width=.55)
    time_ax.bar_label(bars, labels=[f"~{value:.1f} h" for value in hours], padding=5)
    time_ax.set(ylabel="Estimated training hours", title="LoRA32 budget (ESTIMATED)", ylim=(0, 47))
    time_ax.text(.5, .97, "72 runners / 8 x A100 80GB\n14.1 min/step from steps 103-150\nExcludes setup and initial warmup",
                 transform=time_ax.transAxes, ha="center", va="top", fontsize=8, color="#555")
    for ax in (score_ax, time_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    output = ROOT / "docs/RL_tutorial/ch6_lora32_comparison.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
