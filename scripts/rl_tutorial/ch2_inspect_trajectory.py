#!/usr/bin/env python3
"""第 2 章配套脚本：查看一条真实 ALFWorld trajectory 的 obs→action 序列。

用法：
  # 预置真实样本（无需 6GB buffer）
  python scripts/rl_tutorial/ch2_inspect_trajectory.py --sample fail      # base policy 失败案例
  python scripts/rl_tutorial/ch2_inspect_trajectory.py --sample success   # 训练后成功案例
  python scripts/rl_tutorial/ch2_inspect_trajectory.py --sample compare   # 并排对比（推荐）

  # 用你自己跑出的完整 buffer，按 (batch, task, run) 定位一条轨迹
  python scripts/rl_tutorial/ch2_inspect_trajectory.py \
      --buffer checkpoints/ALFWORLD/Step_Wise_Alfworld/buffer/alfworld_buffer.jsonl \
      --batch 1 --task 1 --run 14

展示一条轨迹的：任务指令、任务族、reward、每个 env step 的 observation / think / action。
对照第 2 章 §2.2（失败：拿错物体 + look 死循环）和 §2.3（成功：多子目标 10 步）。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data"

# chat-template 特殊标记（程序化构造，避免源码里出现字面量）
_SPECIAL_TOKENS = [f"<|{name}|>" for name in ("im_end", "im_start", "endoftext")]


def clean(text: str, limit: int = 78) -> str:
    """去掉 chat-template 特殊标记与残留的 turn 头，压成单行，截断。"""
    if not text:
        return ""
    for tok in _SPECIAL_TOKENS:
        text = text.replace(tok, "")
    # obs 抽取时会带上下一个 assistant turn 的模板头，清掉
    text = re.sub(r"\bassistant\b\s*(<think>\s*</think>)?", " ", text)
    text = text.replace("<think>", " ").replace("</tool_response>", " ")
    text = " ".join(text.split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def print_traj(rec: dict, max_steps: int | None = None) -> None:
    verdict = "✅ SUCCESS" if rec["success"] else "❌ FAIL"
    print("\n" + "=" * 74)
    print(f"  trajectory {rec['eid']}   {verdict}")
    print("=" * 74)
    print(f"  任务指令 : {rec['instruction']}")
    print(f"  任务族   : {rec['task_family']}")
    print(f"  reward   : {rec['reward']}   (成功=1.0 / 失败=-0.1)")
    print(f"  env steps: {rec['actual_env_steps']}   (上限 30)")
    print("-" * 74)
    steps = rec["steps"][:max_steps] if max_steps else rec["steps"]
    prev_action = None
    repeat = 0
    for s in steps:
        act = s["action"]
        # 检测死循环：连续相同 action
        if act == prev_action:
            repeat += 1
        else:
            if repeat >= 2:
                print(f"           ⚠️ 上一步 action 重复了 {repeat} 次（死循环！）")
            repeat = 0
        prev_action = act
        print(f"  [step {s['step']:>2}]")
        if s.get("observation"):
            print(f"     obs   : {clean(s['observation'])}")
        if s.get("think"):
            # enable_thinking=false → 输出不含 think 标签；此处是裸的 pre-action 推理文本
            print(f"     reason: {clean(s['think'])}   (裸文本, 无 think 标签)")
        print(f"     action: {act}   (resp {s.get('response_length','?')} tok)")
    if repeat >= 2:
        print(f"           ⚠️ 结尾 action 重复了 {repeat} 次（死循环耗尽步数！）")
    print("=" * 74)


def load_sample(which: str) -> dict:
    fname = {"fail": "traj_fail_early.json", "success": "traj_success_late.json"}[which]
    return json.loads((SAMPLE_DIR / fname).read_text())


def load_from_buffer(buffer: str, batch: int, task: int, run: int) -> dict:
    """从完整 buffer.jsonl 里捞出 (batch,task,run) 这条轨迹并重建。"""
    target = (batch, task, run)
    exps = []
    with open(buffer, errors="ignore") as f:
        for ln in f:
            if f'"batch": {batch}' not in ln and f'"batch":{batch}' not in ln:
                continue  # 粗筛，减少 json 解析量
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            eid = e["eid"]
            if (eid["batch"], eid["task"], eid["run"]) == target:
                exps.append(e)
    if not exps:
        raise SystemExit(f"没在 buffer 里找到轨迹 {target}")
    exps.sort(key=lambda e: e["eid"]["step"])

    def parse_action(rt: str) -> str:
        return rt.split("<action>")[-1].split("</action>")[0].strip() if "<action>" in rt else rt.strip()

    def parse_think(rt: str) -> str:
        if "<think>" in rt:
            return rt.split("<think>")[-1].split("</tool_response>")[0].strip()
        return rt.split("<action>")[0].strip()

    def last_obs(pt: str) -> str:
        parts = pt.split("Observation:")
        return parts[-1].strip() if len(parts) >= 2 else ""

    return {
        "eid": {"batch": batch, "task": task, "run": run},
        "task_family": "(需 --taskset 映射，此处略)",
        "instruction": _instruction(exps[0]["prompt_text"]),
        "reward": exps[-1]["reward"],
        "success": exps[-1]["reward"] == 1.0,
        "actual_env_steps": exps[-1].get("metrics", {}).get("actual_env_steps"),
        "steps": [
            {
                "step": e["eid"]["step"],
                "observation": last_obs(e["prompt_text"]),
                "think": parse_think(e["response_text"]),
                "action": parse_action(e["response_text"]),
                "response_length": e["response_length"],
            }
            for e in exps
        ],
    }


def _instruction(prompt_text: str) -> str:
    import re

    m = re.search(r"Your task is to:([^\n]+)", prompt_text)
    if not m:
        return ""
    instr = m.group(1)
    for tok in _SPECIAL_TOKENS:
        instr = instr.replace(tok, "")
    return instr.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["fail", "success", "compare"])
    ap.add_argument("--buffer")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--task", type=int)
    ap.add_argument("--run", type=int)
    ap.add_argument("--max-steps", type=int, default=None, help="只打印前 N 步")
    args = ap.parse_args()

    if args.sample:
        if args.sample == "compare":
            f, s = load_sample("fail"), load_sample("success")
            print("\n########## 对比：base policy 失败 vs 训练后成功 ##########")
            print_traj(f, args.max_steps or 12)
            print_traj(s, args.max_steps)
            print("\n########## 关键差异 ##########")
            print(f"  失败: {f['instruction']} → {f['actual_env_steps']} 步耗尽, reward {f['reward']}")
            print(f"        缺陷: 拿错物体 + 无效动作 + look around 死循环")
            print(f"  成功: {s['instruction']} → {s['actual_env_steps']} 步完成, reward {s['reward']}")
            print(f"        学到: 正确识别物体 + 多子目标规划(找→转化→放) + 零浪费")
        else:
            print_traj(load_sample(args.sample), args.max_steps)
    elif args.buffer:
        if None in (args.batch, args.task, args.run):
            raise SystemExit("--buffer 需要同时给 --batch --task --run")
        print_traj(load_from_buffer(args.buffer, args.batch, args.task, args.run), args.max_steps)
    else:
        raise SystemExit("给 --sample {fail,success,compare} 或 --buffer + --batch/--task/--run")


if __name__ == "__main__":
    main()
