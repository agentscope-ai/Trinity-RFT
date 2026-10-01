#!/usr/bin/env python3
"""配套脚本的样本数据抽取器：从真实训练产物 buffer.jsonl 导出小样本 JSON。

生成的 sample_data/*.json 会被 ch2/ch3/ch4/ch6 的演示脚本直接读取，
这样读者即使没有 6GB 的完整 buffer，也能离线跑通所有演示。

用法（在跑过训练的机器上，指向真实 buffer）：
  python scripts/rl_tutorial/extract_samples.py \
      --buffer checkpoints/ALFWORLD/Step_Wise_Alfworld/buffer/alfworld_buffer.jsonl \
      --taskset examples/grpo_alfworld/alfworld_data/train.jsonl \
      --outdir scripts/rl_tutorial/sample_data

抽取内容：
  traj_fail_early.json    一条早期失败轨迹（拿错物体 + look 死循环，30 步耗尽）
  traj_success_late.json  一条后期成功轨迹（heat 类多子目标，11 步高效完成）
  group_rewards.json      一个真实 task 组的 16 条 run 终止 reward（ch4 算 advantage 用）
  task_family_success.json 早/后期按任务族的成功率（ch6 分析用）
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = [
    "pick_and_place",
    "pick_two_obj",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "pick_clean_then_place",
    "look_at_obj_in_light",
]


def family_of(game_file: str) -> str:
    m = re.search(r"/(" + "|".join(FAMILIES) + r")", game_file)
    return m.group(1) if m else "other"


def last_observation(prompt_text: str) -> str:
    """prompt_text 里最后一个 `Observation: ...` 段落 = 该步模型看到的 obs。"""
    parts = prompt_text.split("Observation:")
    if len(parts) < 2:
        return ""
    return parts[-1].strip()


def parse_action(response_text: str) -> str:
    if "<action>" in response_text:
        return response_text.split("<action>")[-1].split("</action>")[0].strip()
    return response_text.strip()


def parse_think(response_text: str) -> str:
    if "<think>" in response_text:
        return response_text.split("<think>")[-1].split("</think>")[0].strip()
    # 早期 base policy 常不包 <think> 标签，直接输出裸文本
    head = response_text.split("<action>")[0].strip()
    return head


def task_instruction(prompt_text: str) -> str:
    m = re.search(r"Your task is to:([^\n]+)", prompt_text)
    return m.group(1).replace("<|im_end|>", "").strip() if m else ""


def load_chunks(buffer: str, n_head: int, n_tail: int):
    """读 buffer 的头/尾若干行（避免全量加载 6GB）。返回 (head_lines, tail_lines)。"""
    import subprocess

    head = subprocess.run(
        ["head", "-n", str(n_head), buffer], capture_output=True, text=True
    ).stdout.splitlines()
    tail = subprocess.run(
        ["tail", "-n", str(n_tail), buffer], capture_output=True, text=True
    ).stdout.splitlines()
    return head, tail


def group_traj(lines):
    traj = defaultdict(list)
    for ln in lines:
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        k = (e["eid"]["batch"], e["eid"]["task"], e["eid"]["run"])
        traj[k].append(e)
    for k in traj:
        traj[k].sort(key=lambda e: e["eid"]["step"])
    return traj


def traj_to_record(exps, family="?"):
    return {
        "eid": {k: exps[0]["eid"][k] for k in ("batch", "task", "run")},
        "task_family": family,
        "instruction": task_instruction(exps[0]["prompt_text"]),
        "reward": exps[-1]["reward"],
        "success": exps[-1]["reward"] == 1.0,
        "actual_env_steps": exps[-1].get("metrics", {}).get("actual_env_steps"),
        "steps": [
            {
                "step": e["eid"]["step"],
                "observation": last_observation(e["prompt_text"]),
                "think": parse_think(e["response_text"]),
                "action": parse_action(e["response_text"]),
                "response_length": e["response_length"],
                "step_reward": e.get("info", {}).get("step_reward"),
                "env_state_hash": e.get("info", {}).get("env_state_hash"),
            }
            for e in exps
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--taskset", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-head", type=int, default=250000)
    ap.add_argument("--n-tail", type=int, default=250000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # global task index -> family
    fam_by_idx = {}
    for i, ln in enumerate(Path(args.taskset).read_text().splitlines()):
        fam_by_idx[i] = family_of(json.loads(ln)["game_file"])

    head, tail = load_chunks(args.buffer, args.n_head, args.n_tail)
    traj_head, traj_tail = group_traj(head), group_traj(tail)

    def fam_of_exps(exps):
        ti = exps[-1].get("info", {}).get("task_index", {}).get("index")
        return fam_by_idx.get(ti, "?")

    # 1) 早期失败轨迹：挑一条 30 步耗尽、reward=-0.1 的
    fail = None
    for k, exps in traj_head.items():
        if exps[-1]["reward"] == -0.1 and len(exps) >= 25 and exps[0]["eid"]["step"] == 0:
            fail = traj_to_record(exps, fam_of_exps(exps))
            break
    if fail:
        (outdir / "traj_fail_early.json").write_text(
            json.dumps(fail, ensure_ascii=False, indent=2)
        )
        print(f"traj_fail_early.json  ← {fail['eid']} '{fail['instruction']}' "
              f"{len(fail['steps'])} steps reward={fail['reward']}")

    # 2) 后期成功轨迹：挑一条中等长度、多子目标(heat/cool/clean)的成功轨迹
    succ_cands = [
        (len(exps), k, exps)
        for k, exps in traj_tail.items()
        if exps[-1]["reward"] == 1.0
        and exps[0]["eid"]["step"] == 0
        and fam_of_exps(exps) in ("pick_heat_then_place", "pick_cool_then_place", "pick_clean_then_place")
        and 8 <= len(exps) <= 16
    ]
    succ_cands.sort()
    if succ_cands:
        _, k, exps = succ_cands[len(succ_cands) // 2]
        succ = traj_to_record(exps, fam_of_exps(exps))
        (outdir / "traj_success_late.json").write_text(
            json.dumps(succ, ensure_ascii=False, indent=2)
        )
        print(f"traj_success_late.json ← {succ['eid']} '{succ['instruction']}' "
              f"{len(succ['steps'])} steps reward={succ['reward']}")

    # 3) 一个真实 task 组的 16 条 run 终止 reward（优先挑有分化、非全同的组）
    def find_group(traj, want_spread=True):
        bytask = defaultdict(dict)
        for (b, t, r), exps in traj.items():
            bytask[(b, t)][r] = exps[-1]["reward"]
        best = None
        for (b, t), runs in bytask.items():
            if len(runs) < 8:
                continue
            vals = list(runs.values())
            spread = len(set(vals)) > 1
            if want_spread and not spread:
                continue
            cand = {
                "batch": b, "task": t, "n_runs": len(runs),
                "rewards": [round(v, 3) for _, v in sorted(runs.items())],
                "success_rate": round(sum(1 for v in vals if v == 1.0) / len(vals), 3),
            }
            if best is None or cand["n_runs"] > best["n_runs"]:
                best = cand
        return best

    grp = find_group(traj_tail) or find_group(traj_head) or find_group(traj_head, False)
    if grp:
        (outdir / "group_rewards.json").write_text(json.dumps(grp, ensure_ascii=False, indent=2))
        print(f"group_rewards.json    ← batch{grp['batch']} task{grp['task']} "
              f"{grp['n_runs']} runs success_rate={grp['success_rate']}")

    # 4) 早/后期按任务族成功率
    def family_success(traj):
        succ, tot = Counter(), Counter()
        for k, exps in traj.items():
            f = fam_of_exps(exps)
            tot[f] += 1
            if exps[-1]["reward"] == 1.0:
                succ[f] += 1
        return {
            f: {"success": succ[f], "total": tot[f],
                "rate": round(succ[f] / tot[f], 3) if tot[f] else 0.0}
            for f in sorted(tot, key=lambda x: -tot[x])
        }

    fam_data = {
        "early": family_success(traj_head),
        "late": family_success(traj_tail),
        "note": "early=buffer 头 25 万行(≈训练前段), late=尾 25 万行(≈训练后段); "
                "reward=1.0 记为成功",
    }
    (outdir / "task_family_success.json").write_text(
        json.dumps(fam_data, ensure_ascii=False, indent=2)
    )
    eo = sum(v["success"] for v in fam_data["early"].values())
    et = sum(v["total"] for v in fam_data["early"].values())
    lo = sum(v["success"] for v in fam_data["late"].values())
    lt = sum(v["total"] for v in fam_data["late"].values())
    print(f"task_family_success.json ← early {eo}/{et}={eo/et*100:.1f}%  late {lo}/{lt}={lo/lt*100:.1f}%")


if __name__ == "__main__":
    main()
