#!/usr/bin/env python3
"""第 3 章配套脚本：reward 换算，区分 turn 加权占比与每局成功率。

ALFWorld reward 只有两档：成功=1.0，失败=-0.1。换算保持输入的统计单位：
critic/score/mean 按 experience 求平均，换算后是成功轨迹的 turn 占比；
只有每局仅计一次终局 reward 时，换算才得到每局等权成功率。
    mean_reward = 1.1 * p - 0.1      →      p = (mean_reward + 0.1) / 1.1

用法：
  python scripts/rl_tutorial/ch3_reward_to_success.py --sample        # 真实 step 换算表
  python scripts/rl_tutorial/ch3_reward_to_success.py --reward 0.62   # 单个 reward 换算
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data"

# 本实验真实 trainer.log 采样（step, critic/score/mean）
REAL_STEPS = [
    (1, -0.094), (20, 0.012), (40, 0.252), (60, 0.524), (80, 0.905),
    (100, 0.768), (115, 0.936), (133, 0.308),
]


def reward_to_success(r: float) -> float:
    """换算当前统计单位下的成功占比；不能自动从 turn 换成 episode。"""
    return (r + 0.1) / 1.1


def success_to_reward(p: float) -> float:
    return 1.1 * p - 0.1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--reward", type=float)
    args = ap.parse_args()

    if args.reward is not None:
        p = reward_to_success(args.reward)
        print(f"\n  reward = {args.reward:+.3f}  →  成功轨迹的 turn 占比 = {p*100:.1f}%")
        print(f"  (输入按 critic/score/mean 解读；p = (reward + 0.1) / 1.1，不是每局等权成功率)\n")
        return

    print("\n" + "=" * 60)
    print("  ALFWorld reward ↔ turn 加权成功占比（真实 trainer.log 采样）")
    print("=" * 60)
    print(f"  {'step':>5} | {'critic/score/mean':>18} | {'turn 占比':>8} | 阶段")
    print("  " + "-" * 56)
    labels = {
        1: "base（成功 turn 很少）", 20: "穿过 0", 40: "稳定上升", 60: "成功 turn 过半",
        80: "高位", 100: "高位波动", 115: "⭐ 峰值", 133: "💥 过度训练崩溃(末)",
    }
    for step, r in REAL_STEPS:
        p = reward_to_success(r) * 100
        print(f"  {step:>5} | {r:>+18.3f} | {p:>7.1f}% | {labels.get(step,'')}")
    print("=" * 60)
    print("  以上按训练 batch 的 turn/experience 加权，不是每局等权成功率。")
    print("  例：成功 1 局有 10 turn，失败 1 局有 30 turn：")
    print("    每局成功率 = 1/2 = 50%；成功 turn 占比 = 10/40 = 25%。")
    print("    对应平均 reward 分别为 0.45（每局）和 0.175（每 turn）。")

    # 真实 reward 分布（一个 task 组的 16 条 run）
    grp = json.loads((SAMPLE_DIR / "group_rewards.json").read_text())
    rewards = grp["rewards"]
    n_succ = sum(1 for r in rewards if r == 1.0)
    n_fail = sum(1 for r in rewards if r == -0.1)
    mean = sum(rewards) / len(rewards)
    print(f"\n  真实 task 组 (batch{grp['batch']} task{grp['task']}, G={grp['n_runs']}):")
    print(f"    成功(reward=1.0): {n_succ} 条   失败(reward=-0.1): {n_fail} 条")
    print(f"    组内每局成功率 = {n_succ/len(rewards)*100:.1f}%")
    print(f"    组内每局 mean reward = {mean:+.4f}  → 换算每局成功率 {reward_to_success(mean)*100:.1f}%")
    print(f"    ⚠️ 只有两档取值 {{-0.1, 1.0}}，没有中间分 —— 这就是「稀疏 reward」")
    print(f"    ✓ 但组内成功/失败混合({n_succ}+{n_fail}) → 有差异 → 能算 advantage（第 4 章）\n")

    print("  反向换算速查（统计单位须一致）：")
    for p in (0.1, 0.5, 0.9, 0.94):
        print(f"    成功占比 {p*100:>5.1f}%  →  reward {success_to_reward(p):+.3f}")
    print()


if __name__ == "__main__":
    main()
