#!/usr/bin/env python3
"""第 4 章配套脚本：用真实 16-run 组算 step-wise GRPO advantage。

完整复刻 StepWiseGRPOAdvantageFn 的计算：
    advantage_i = (reward_i - group_mean) / (group_std + epsilon)
其中 group_std 用无偏估计(n-1)，与 torch.std 默认一致；epsilon=1e-6。

用法：
  python scripts/rl_tutorial/ch4_compute_advantage.py --sample              # 真实组(13失败+3成功)
  python scripts/rl_tutorial/ch4_compute_advantage.py --sample --shift 0.1  # 验证平移不变(第3章§3.6)
  python scripts/rl_tutorial/ch4_compute_advantage.py --rewards 1.0,1.0,...  # 自定义
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data"
EPSILON = 1e-6


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def std_unbiased(xs: list[float]) -> float:
    """无偏标准差(n-1)，与 torch.std 默认一致。"""
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def compute_advantage(rewards: list[float], verbose: bool = True) -> list[float]:
    mu, sigma = mean(rewards), std_unbiased(rewards)
    advs = [(r - mu) / (sigma + EPSILON) for r in rewards]

    if verbose:
        n_succ = sum(1 for r in rewards if r == max(rewards))
        print("\n  ┌────────────────────────────────────────────────────────┐")
        print("  │ Step 1: 组内统计 (G = %d 条轨迹)                       │" % len(rewards))
        print("  └────────────────────────────────────────────────────────┘")
        print(f"    rewards      = {rewards}")
        print(f"    group_mean   = {mu:.5f}")
        print(f"    group_std    = {sigma:.5f}   (无偏 n-1)")
        print(f"    epsilon      = {EPSILON}")
        if sigma < 1e-9:
            print("\n    ⚠️ group_std = 0 → 组内 reward 全相同 → advantage 全 ≈0 → 这一组白跑！")
            print("       (对应第3章§3.5 的 reward_std/min=0：全成功或全失败的死组)")
        print("\n  ┌────────────────────────────────────────────────────────┐")
        print("  │ Step 2: advantage_i = (reward_i - mean) / (std + eps)  │")
        print("  └────────────────────────────────────────────────────────┘")
        for i, (r, a) in enumerate(zip(rewards, advs)):
            if a > 0.5:
                lab = "← 强力鼓励 (成功且稀有)"
            elif a > 0:
                lab = "← 温和鼓励"
            elif a > -0.5:
                lab = "← 温和抑制 (失败但常见)"
            else:
                lab = "← 强力抑制"
            print(f"    run {i+1:>2}: reward={r:+.3f} → adv={a:+.4f}  {lab}")

        # 核心洞察：力度比 = 数量反比，总推力平衡
        uniq = sorted(set(rewards))
        if len(uniq) == 2 and sigma > 1e-9:
            lo, hi = uniq
            n_hi = sum(1 for r in rewards if r == hi)
            n_lo = len(rewards) - n_hi
            a_hi = (hi - mu) / (sigma + EPSILON)
            a_lo = (lo - mu) / (sigma + EPSILON)
            print("\n  ┌────────────────────────────────────────────────────────┐")
            print("  │ 🔑 核心洞察：力度比 = 数量反比，总推力自平衡            │")
            print("  └────────────────────────────────────────────────────────┘")
            print(f"    高 reward({hi:+.1f}) {n_hi} 条, 每条 adv={a_hi:+.4f}")
            print(f"    低 reward({lo:+.1f}) {n_lo} 条, 每条 adv={a_lo:+.4f}")
            print(f"    |adv_hi / adv_lo| = {abs(a_hi/a_lo):.2f}   vs   n_lo/n_hi = {n_lo/n_hi:.2f}  ← 相等！")
            print(f"    总上升推力 = {n_hi}×{a_hi:+.3f} = {n_hi*a_hi:+.3f}")
            print(f"    总下降推力 = {n_lo}×{a_lo:+.3f} = {n_lo*a_lo:+.3f}")
            print(f"    所有 advantage 之和 = {sum(advs):+.2e}  ≈ 0 (减去均值的数学必然)")
            print(f"    → 稀有的成功被用力推高、常见的失败被轻轻压低，梯度自平衡。")
    return advs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="用真实 16-run 组")
    ap.add_argument("--rewards", help="逗号分隔的自定义 rewards")
    ap.add_argument("--shift", type=float, default=0.0, help="给所有 reward 加常数，验证平移不变")
    args = ap.parse_args()

    if args.sample:
        grp = json.loads((SAMPLE_DIR / "group_rewards.json").read_text())
        rewards = list(grp["rewards"])
        print(f"\n  真实 task 组: batch{grp['batch']} task{grp['task']}, "
              f"success_rate={grp['success_rate']}")
    elif args.rewards:
        rewards = [float(x) for x in args.rewards.split(",")]
    else:
        rewards = [-0.1] * 13 + [1.0] * 3  # 默认演示组

    base = compute_advantage(rewards)

    if args.shift:
        shifted = [r + args.shift for r in rewards]
        print("\n" + "#" * 60)
        print(f"# 平移不变验证（第3章§3.6）：所有 reward + {args.shift}")
        print("#" * 60)
        adv2 = compute_advantage(shifted, verbose=False)
        mu2, sig2 = mean(shifted), std_unbiased(shifted)
        print(f"    平移后 group_mean = {mu2:.5f} (变了), group_std = {sig2:.5f} (不变)")
        print(f"    平移后 rewards = {shifted}")
        maxdiff = max(abs(a - b) for a, b in zip(base, adv2))
        print(f"    advantage 最大变化 = {maxdiff:.2e}")
        print(f"    → {'✓ advantage 完全不变！' if maxdiff < 1e-4 else '✗ 变了'}")
        print(f"    结论: group-relative GRPO 对 reward 整体平移不变，")
        print(f"          所以失败用 -0.1 还是 0，advantage 一模一样（负号只影响曲线可读性）。")
    print()


if __name__ == "__main__":
    main()
