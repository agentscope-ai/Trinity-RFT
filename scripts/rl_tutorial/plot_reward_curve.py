#!/usr/bin/env python3
"""第 1 章配套脚本：从 trainer.log 画 ALFWorld multi-step GRPO 的真实训练曲线。

用法：
  python scripts/rl_tutorial/plot_reward_curve.py \
      --log checkpoints/ALFWORLD/Step_Wise_Alfworld/log/trainer.log \
      --out docs/RL_tutorial/ch1_reward_curve.png

画两张子图：
  上：critic/score/mean（左轴）+ turn 加权成功占比（右轴），标出峰值与崩溃区
  下：actor/entropy_loss 与 actor/ppo_kl，展示"训练过头→策略退化"的信号

reward → turn 加权成功占比：成功 reward=1.0、失败 reward=-0.1。
  critic/score/mean 按 experience 求平均，p = (reward + 0.1) / 1.1
  是训练 batch 中来自成功轨迹的 turn 占比，不是每局等权成功率。
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


def parse_trainer_log(path: str) -> dict[int, dict]:
    """解析 trainer.log 里的 `Step N: {...}` 指标行，按 step 去重（保留最后一条）。"""
    by_step: dict[int, dict] = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = re.search(r"Step (\d+): (\{.*\})", line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            continue
        by_step[int(m.group(1))] = d
    return by_step


def reward_to_success(reward: float) -> float:
    """训练 reward 换成成功轨迹的 turn 占比，保持 experience 加权口径。"""
    return (reward + 0.1) / 1.1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="trainer.log 路径")
    ap.add_argument("--out", required=True, help="输出 PNG 路径")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_step = parse_trainer_log(args.log)
    steps = sorted(by_step)
    reward = [by_step[s].get("critic/score/mean") for s in steps]
    entropy = [by_step[s].get("actor/entropy_loss") for s in steps]
    ppo_kl = [by_step[s].get("actor/ppo_kl") for s in steps]
    success = [reward_to_success(r) * 100 if r is not None else None for r in reward]

    peak_i = max(range(len(reward)), key=lambda i: (reward[i] if reward[i] is not None else -9))
    peak_step, peak_reward = steps[peak_i], reward[peak_i]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # --- 上：reward + turn 加权成功占比 ---
    ax1.plot(steps, reward, color="#d62728", lw=2, label="critic/score/mean (reward)")
    ax1.axhline(0, color="#999", lw=0.8, ls=":")
    ax1.set_ylabel("reward  (1.0=success, -0.1=fail)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(-0.2, 1.05)
    ax1.scatter([peak_step], [peak_reward], color="#d62728", zorder=5, s=40)
    ax1.annotate(
        f"peak {peak_reward:.3f} @ step {peak_step}\n(turn-weighted {reward_to_success(peak_reward)*100:.0f}%)",
        xy=(peak_step, peak_reward),
        xytext=(peak_step - 46, peak_reward - 0.24),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        fontsize=9,
        color="#d62728",
    )
    # 崩溃区：最后 reward 明显回落 + ppo_kl 飙升
    ax1.axvspan(118, steps[-1], color="#ff7f0e", alpha=0.12)
    ax1.text(
        118, 0.84, "over-training\ncollapse", color="#ff7f0e", fontsize=9, ha="right"
    )

    ax1b = ax1.twinx()
    ax1b.plot(steps, success, color="#1f77b4", lw=1.4, ls="--", label="success share (turn-weighted, %)")
    ax1b.set_ylabel("success share (turn-weighted, %)", color="#1f77b4")
    ax1b.tick_params(axis="y", labelcolor="#1f77b4")
    ax1b.set_ylim(-10, 105)
    ax1.set_title(
        "ALFWorld x Trinity-RFT multi-step GRPO (Qwen3-1.7B, 8xA100) - real training curve",
        fontsize=12,
    )
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)

    # --- 下：entropy + ppo_kl ---
    ax2.plot(steps, entropy, color="#2ca02c", lw=1.8, label="actor/entropy_loss")
    ax2.set_ylabel("entropy_loss", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2b = ax2.twinx()
    ax2b.plot(steps, ppo_kl, color="#9467bd", lw=1.4, ls="--", label="actor/ppo_kl")
    ax2b.axhline(0.03, color="#9467bd", lw=0.8, ls=":")
    ax2b.text(steps[1], 0.032, "ppo_kl warning line 0.03", color="#9467bd", fontsize=8)
    ax2b.set_ylabel("ppo_kl", color="#9467bd")
    ax2b.tick_params(axis="y", labelcolor="#9467bd")
    ax2.set_xlabel("trainer step")
    ax2.set_title(
        "Policy-degeneration signal: entropy climbs 0.03 -> 0.67, ppo_kl breaks the warning line late",
        fontsize=10.5,
    )
    h3, l3 = ax2.get_legend_handles_labels()
    h4, l4 = ax2b.get_legend_handles_labels()
    ax2.legend(h3 + h4, l3 + l4, loc="upper left", fontsize=9)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved figure → {args.out}")
    print(f"steps parsed: {len(steps)} (1..{steps[-1]})")
    print(f"peak reward {peak_reward:.3f} @ step {peak_step} "
          f"(成功 turn 占比 {reward_to_success(peak_reward)*100:.1f}%)")
    print(f"last-step reward {reward[-1]:.3f} @ step {steps[-1]} "
          f"(成功 turn 占比 {reward_to_success(reward[-1])*100:.1f}%)")


if __name__ == "__main__":
    main()
