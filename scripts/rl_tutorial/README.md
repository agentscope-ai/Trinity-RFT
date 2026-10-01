# ALFWorld 教程配套脚本

配合 [`docs/RL_tutorial/`](../../docs/RL_tutorial/README.md) 使用。所有脚本**默认读 `sample_data/` 里预置的真实样本**（从本实验完整 buffer 抽取），无需 6GB 完整产物即可离线跑通；也可 `--buffer` 指向你自己的训练产物。

## 脚本一览

| 脚本 | 对应章 | 作用 |
|---|---|---|
| [`plot_reward_curve.py`](./plot_reward_curve.py) | ch1 / ch5 | 从 `trainer.log` 画 reward + entropy + ppo_kl 三联曲线（含崩溃区）|
| [`ch2_inspect_trajectory.py`](./ch2_inspect_trajectory.py) | ch2 | 打印一条真实轨迹的 obs→think→action 序列，可对比失败/成功 |
| [`ch3_reward_to_success.py`](./ch3_reward_to_success.py) | ch3 | reward ↔ 成功率换算（`p=(r+0.1)/1.1`）+ 真实 reward 分布 |
| [`ch4_compute_advantage.py`](./ch4_compute_advantage.py) | ch4 | 用真实 16-run 组算 step-wise GRPO advantage，验证「平移不变」「力度比=数量反比」|
| [`ch6_task_family_breakdown.py`](./ch6_task_family_breakdown.py) | ch6 | 按 6 个任务族分解成功率（早期 vs 后期）|
| [`extract_samples.py`](./extract_samples.py) | 全部 | 从完整 buffer 抽取上述脚本用的 `sample_data/*.json` |

## 快速开始（离线，用预置样本）

```bash
python ch2_inspect_trajectory.py --sample compare      # 失败 vs 成功轨迹对比
python ch3_reward_to_success.py --sample               # reward↔成功率换算表
python ch4_compute_advantage.py --sample --shift 0.1   # GRPO advantage + 平移不变验证
python ch6_task_family_breakdown.py --sample           # 任务族成功率分解
```

## 用你自己的训练产物

```bash
CK=checkpoints/ALFWORLD/Step_Wise_Alfworld

# 重画曲线
python plot_reward_curve.py --log $CK/log/trainer.log --out /tmp/curve.png

# 看某条轨迹
python ch2_inspect_trajectory.py --buffer $CK/buffer/alfworld_buffer.jsonl --batch 1 --task 1 --run 14

# 重新抽样本（覆盖 sample_data/）
python extract_samples.py \
  --buffer $CK/buffer/alfworld_buffer.jsonl \
  --taskset examples/grpo_alfworld/alfworld_data/train.jsonl \
  --outdir sample_data
```

## sample_data/ 说明

预置样本均来自本实验真实产物（`extract_samples.py` 抽取）：

| 文件 | 内容 |
|---|---|
| `traj_fail_early.json` | 早期失败轨迹（batch1 task1 run14，`put a candle in countertop`，拿错物体 + look 死循环，30 步耗尽，reward -0.1）|
| `traj_success_late.json` | 后期成功轨迹（batch229 task11 run3，`put a cool cup in diningtable`，多子目标 10 步完成，reward 1.0）|
| `group_rewards.json` | 一个真实 task 组的 16 条 run 终止 reward（batch163 task13，3 成功 + 13 失败）|
| `task_family_success.json` | 早/后期按 6 个任务族的成功率（整体 10.0% → 90.3%）|

> 依赖：仅标准库（`plot_reward_curve.py` 额外需要 `matplotlib`）。
