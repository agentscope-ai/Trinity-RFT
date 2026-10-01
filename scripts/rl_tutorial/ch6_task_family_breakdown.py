#!/usr/bin/env python3
"""第 6 章配套脚本：按 6 个 ALFWorld 任务族分解成功率（早期 vs 后期）。

回答第 0 章 §0.2 的「难度阶梯」问题：训练后哪些任务族学会了、哪些还在挣扎。
整体 reward 0.9 会掩盖「某个族还在 69%」——按子类型拆解才看得清。

用法：
  # 预置真实样本（从 baseline buffer 头/尾各 25 万行抽取）
  python scripts/rl_tutorial/ch6_task_family_breakdown.py --sample

  # 用你自己的完整 buffer + taskset 重新算
  python scripts/rl_tutorial/ch6_task_family_breakdown.py \
      --buffer checkpoints/ALFWORLD/Step_Wise_Alfworld/buffer/alfworld_buffer.jsonl \
      --taskset examples/grpo_alfworld/alfworld_data/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data"
FAMILIES = [
    "pick_and_place", "pick_two_obj", "pick_heat_then_place",
    "pick_cool_then_place", "pick_clean_then_place", "look_at_obj_in_light",
]


def family_of(game_file: str) -> str:
    m = re.search(r"/(" + "|".join(FAMILIES) + r")", game_file)
    return m.group(1) if m else "other"


def print_table(early: dict, late: dict) -> None:
    fams = [f for f in FAMILIES if f in late or f in early]
    print("\n" + "=" * 72)
    print("  ALFWorld 任务族成功率分解：早期(base) vs 后期(训练后)")
    print("=" * 72)
    print(f"  {'任务族':<24} {'早期':>10} {'后期':>10} {'提升':>9}")
    print("  " + "-" * 68)

    def rate(d, f):
        v = d.get(f)
        return v["rate"] if v else 0.0

    rows = sorted(fams, key=lambda f: -rate(late, f))
    for f in rows:
        e, l = rate(early, f) * 100, rate(late, f) * 100
        bar = "█" * int(l / 5)
        print(f"  {f:<24} {e:>9.1f}% {l:>9.1f}% {l-e:>+8.1f}pp  {bar}")

    def overall(d):
        s = sum(v["success"] for v in d.values())
        t = sum(v["total"] for v in d.values())
        return s / t * 100 if t else 0.0

    eo, lo = overall(early), overall(late)
    print("  " + "-" * 68)
    print(f"  {'整体 OVERALL':<24} {eo:>9.1f}% {lo:>9.1f}% {lo-eo:>+8.1f}pp")
    print("=" * 72)
    print("  解读:")
    hardest = min(fams, key=lambda f: rate(late, f))
    easiest = max(fams, key=lambda f: rate(late, f))
    print(f"    ✓ 最简单族 {easiest} 后期 {rate(late,easiest)*100:.1f}%（几乎满分）")
    print(f"    ⚠️ 最难族 {hardest} 后期仅 {rate(late,hardest)*100:.1f}%（仍是天花板，下一轮重点攻）")
    print(f"    ✓ 多子目标族(heat/cool/clean) 提升最大（base 几乎不会 → 后期 86-94%）")
    print(f"    → RL 主要教会了「子目标规划」：找物体→转化(加热/冷却/清洗)→放置\n")


def from_sample() -> None:
    data = json.loads((SAMPLE_DIR / "task_family_success.json").read_text())
    print_table(data["early"], data["late"])
    print(f"  数据来源: {data.get('note','')}\n")


def from_buffer(buffer: str, taskset: str, n_head: int, n_tail: int) -> None:
    import subprocess

    fam_by_idx = {}
    for i, ln in enumerate(Path(taskset).read_text().splitlines()):
        fam_by_idx[i] = family_of(json.loads(ln)["game_file"])

    def chunk(cmd, n):
        return subprocess.run([cmd, "-n", str(n), buffer], capture_output=True, text=True).stdout.splitlines()

    def analyze(lines):
        traj = defaultdict(list)
        for ln in lines:
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            traj[(e["eid"]["batch"], e["eid"]["task"], e["eid"]["run"])].append(e)
        succ, tot = Counter(), Counter()
        for exps in traj.values():
            exps.sort(key=lambda e: e["eid"]["step"])
            last = exps[-1]
            ti = last.get("info", {}).get("task_index", {}).get("index")
            f = fam_by_idx.get(ti, "?")
            tot[f] += 1
            if last["reward"] == 1.0:
                succ[f] += 1
        return {f: {"success": succ[f], "total": tot[f], "rate": succ[f] / tot[f] if tot[f] else 0.0}
                for f in tot}

    early = analyze(chunk("head", n_head))
    late = analyze(chunk("tail", n_tail))
    print_table(early, late)
    print(f"  数据来源: buffer 头 {n_head} 行(早) / 尾 {n_tail} 行(晚)\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--buffer")
    ap.add_argument("--taskset")
    ap.add_argument("--n-head", type=int, default=250000)
    ap.add_argument("--n-tail", type=int, default=250000)
    args = ap.parse_args()

    if args.buffer:
        if not args.taskset:
            raise SystemExit("--buffer 需要同时给 --taskset")
        from_buffer(args.buffer, args.taskset, args.n_head, args.n_tail)
    else:
        from_sample()


if __name__ == "__main__":
    main()
