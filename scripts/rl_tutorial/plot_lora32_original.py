#!/usr/bin/env python3
"""Plot saved original LoRA32 training metrics; never interpolate the resume gap."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/rl_tutorial/sample_data/lora32_original_metrics.csv"
OUTPUT = ROOT / "docs/RL_tutorial/ch6_lora32_original_curve.png"


def main():
    with SOURCE.open(newline="") as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    segments = [[r for r in rows if r["step"] <= 70],
                [r for r in rows if r["step"] >= 72]]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    for j, segment in enumerate(segments):
        x = [r["step"] for r in segment]
        y = [r["score"] for r in segment]
        mean = [sum(y[max(0, i - 4):i + 1]) / len(y[max(0, i - 4):i + 1])
                for i in range(len(y))]
        axes[0].plot(x, y, color="#93b4dd", linewidth=.8,
                     label="Trainer score" if j == 0 else None)
        axes[0].plot(x, mean, color="#2455a4", linewidth=2,
                     label="Trailing mean (up to 5 points)" if j == 0 else None)
        axes[1].plot(x, [r["entropy"] for r in segment], color="#b45b2a", linewidth=1.2)
    for ax in axes:
        ax.axvspan(70, 72, color="#d3d7dd", alpha=.7)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].annotate("Interruption / resume\nstep 71 missing", (71, .62),
                     xytext=(12, .73), arrowprops={"arrowstyle": "->", "color": "#555"}, fontsize=9)
    axes[0].set(ylabel="Training score (not test success rate)", ylim=(-.15, .85),
                title="Original LoRA32 RL run | Qwen3-1.7B / ALFWorld / lr 5e-5")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set(ylabel="Entropy", xlabel="Trainer log step (resume labels differ from optimizer updates)")
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
