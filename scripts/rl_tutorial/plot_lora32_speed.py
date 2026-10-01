#!/usr/bin/env python3
"""Plot the saved LoRA32 trace, with distinct segments for planned restarts."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def plot(rows, limit, output):
    rows = [r for r in rows if int(r["step"]) <= limit]
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    color = "#007f86"
    for index, runners in enumerate((16, 32, 48, 72)):
        segment = [r for r in rows if int(r["runners"]) == runners]
        x = [int(r["step"]) for r in segment]
        y = [float(r["score"]) for r in segment]
        mean = [sum(y[max(0, i - 9):i + 1]) / len(y[max(0, i - 9):i + 1])
                for i in range(len(y))]
        ax.plot(x, y, color=color, alpha=.3, linewidth=.8,
                label="Step reward" if index == 0 else None)
        ax.plot(x, mean, color=color, linewidth=2.1,
                label="Trailing mean (up to 10 steps)" if index == 0 else None)
    for step in (10, 90, 100):
        ax.axvline(step + .5, color="#9ca3af", linestyle=":", linewidth=.8)
    ax.set(title=f"Learning with LoRA32 + TuFT | first {limit} steps",
           ylabel="Mean reward (turn-weighted)", xlabel="Training step",
           ylim=(-.15, .85))
    ax.text(.02, .96, "Dotted lines mark resumes in the reference run", transform=ax.transAxes,
            va="top", fontsize=9, color="#555")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(output)


def main():
    source = ROOT / "scripts/rl_tutorial/sample_data/lora32_speed_metrics.csv"
    with source.open(newline="") as f:
        rows = list(csv.DictReader(f))
    plot(rows, 150, ROOT / "docs/RL_tutorial/ch6_lora32_speed_first150.png")
    plot(rows, 250, ROOT / "docs/RL_tutorial/ch6_lora32_speed_curve.png")


if __name__ == "__main__":
    main()
