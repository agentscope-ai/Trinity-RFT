[**English**](README.md) | **中文**

# Connect the Dots (CoD)

**Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning**

当 LLM agent 部署到一个环境中，它会连续解决一长串任务，同时不断探索环境、从自身经验中学习、迭代地自我更新关于环境的 context，后续任务在更新后的 context 条件下越解越好。

CoD 把相关任务打成一个 pack（任务包），作为一条交替进行**解决任务**（solve-task）与**更新 context**（update-context）回合的长序列展开：每解完一个任务，模型就根据刚发生的事更新 context（一段简短的 `Hints:` 块），靠后的任务在累积的 context 条件下求解。
整个 pack 用 RL 端到端训练，细粒度的信用分配让“使后续任务更好解的 context 更新”获得奖励。
训练好的模型 reward 随 pack 位置递增，这正是 CoD 元能力被激发出来的标志。

<p align="center">
  <img src="assets/cod_overview.png" alt="CoD 方法总览" width="760">
  <br><sub><em>图 1：CoD-Deploy 与 CoD-Train 的示意（与标准的逐任务 RL 对比）。环境 A、B 用于训练，M 是部署或评测的新环境。每个 block 是一个 rollout 回合：求解任务 x_i（其本身可能是长程多轮任务），或更新 agent 对当前环境的 context z_i。</em></sub>
</p>

---

## 工作原理

```
pack = [ task0, task1, task2, task3 ]      # 同一 taskset 的一包相关任务(大小 = task_pack_size,图中以 4 示意)

 task0  --求解(无 hint)-->     reward0, feedback0  --生成 hint-->  hint1
 task1  --求解(用 hint1)-->    reward1, feedback1  --生成 hint-->  hint2
 task2  --求解(用 hint2)-->    reward2, feedback2  --生成 hint-->  hint3
 task3  --求解(用 hint3)-->    reward3
```

- **Pack**：来自同一 taskset 的相关任务被分到一包，大小由 `task_pack_size`（训练）/ `eval_task_pack_size`（评测）决定，分组在 `pack_tasks()` 里完成。
- **位置（position）**：任务在 pack 中的序号；位置 k 带着前 k 个任务蒸馏出的 hint 来求解，所以位置越靠后，代表模型在上下文里已经学到越多。
- **任务奖励**：图里每个 task_i 解完得到的 reward，来自环境（答对 / 答错，或按质量打分），并扣掉过长解答、过长 hint 的长度惩罚（鼓励简洁）。
- **迭代 hint**：每解完一个任务，模型就基于 `(上一段 hint, 轨迹, reward, feedback)` 更新作为 context 的 `Hints:` 块，下一题求解时把它拼到 prompt 最前面。
- **细粒度信用分配（reward-to-go）**：遵循经典动态规划原理，每个回合（解题、更新 context）的回报由当前与未来任务的奖励构成。一次 context 更新在帮到后续任务时会得到更高得分，模型于是学着写出有用的 hint。

核心评估指标：`reward_iterative_hint_e2e_taskset_{ts}_pos_{pos}`，即每个 pack 位置的平均 reward；位置越靠后、reward 越高，就是 CoD 效应。

<p align="center">
  <img src="assets/cod_effect.png" alt="reward 随 pack 位置递增" width="820">
  <br><sub><em>图 2：Qwen3-8B 的 CoD 实验结果。reward 随 pack 位置上升，训练时如此，OOD 评测时也如此，包括 in-domain（更难的 FrozenLake）与 cross-domain（Alchemy、Terminal）。</em></sub>
</p>

<p align="center">
  <img src="assets/cod_qwen27b.png" alt="CoD 训练后的 Qwen3.6-27B 与 Qwen3.8 系列在三个环境上的对比" width="1000">
  <br><sub><em>CoD 训练后的 Qwen3.6-27B 与基础模型、Qwen3.8-27B、Qwen3.8-Max 在 PDE Discovery、Optimal Control 和 Grid Navigation 上的对比。</em></sub>
</p>

---

## 环境

每个环境里，一个 pack 的任务之间都有可复用的东西：有时是隐藏的规律（动作映射、合成配方），有时是解题策略或踩过的坑。模型在与环境的交互和反馈中把它摸清，写进 hint，再带给 pack 内后面的任务。

| 环境 | `default_workflow_type` | pack 内可迁移的知识 |
|---|---|---|
| FrozenLake-Obscure | `cod_frozenlake_obscure_workflow` | 隐藏的动作映射，代号 1–4 各代表哪个移动方向 |
| Alchemy-Random | `cod_random_alchemy_workflow` | 隐藏的合成配方，哪些元素能组合出哪个新元素 |
| Terminal | `cod_terminal_workflow` | 命令、路径的用法与易踩的坑，以及文件大致在哪 |
| Learn2Ask | `cod_learn2ask_workflow` | 何时继续追问、何时停下来给诊断 |
| Optimal Control | `cod_optimalcontrol_workflow` | 从交互反馈中了解系统的运动规律，更好地控制它到达不同目标 |
| PDE Discovery | `CoDPDEDiscoveryWorkflow` | 从采样数据中发现未知的反应项，在后续任务中不断完善判断 |
| Grid Navigation | `cod_grid_navigation_workflow` | 逐步探索并记住地图中的移动代价，为后续任务选择低代价路线 |

实现位于 [`trinity/common/workflows/connect_the_dots/`](../../trinity/common/workflows/connect_the_dots/)。

---

## 目录结构

```
examples/research_cod/
├── get_*_data.py        # 各环境的数据生成器
├── exp_plan_final/      # 主实验
│   ├── train/           # 训练配置
│   └── bench/           # 评测配置
└── exp_plan_learn2ask/   # learn2ask的数据准备

trinity/common/workflows/connect_the_dots/   # CoD workflow 实现
├── cod_workflow.py     # 打包 / 迭代 hint / 任务奖励
├── base_workflow.py    # AsyncCoDMultiStepWorkflow 基类
└── <env>/              # 每个环境各自有独立子目录
    ├── workflow.py     # 任务渲染 / 评分
    └── prompts/        # system / user prompt

trinity/algorithm/advantage_fn/cod_advantage.py   # reward-to-go 信用分配（CoDAdvantageFn）
```

---

## 快速开始

```bash
git clone <REPO_URL>
cd <REPO_DIR>
conda create -n trinity python=3.12 && conda activate trinity
pip install -e ".[vllm,flash_attn]"
pip install gymnasium jinja2 pandas
```

**1. 生成数据。**
```bash
# FrozenLake-Obscure
python examples/research_cod/get_frozen_lake_data.py --local_dir examples/research_cod/data/frozen_lake_4567 \
    --train_size 50000 --test_size 4000 --map_min_size 4 --map_max_size 5 --tile_min_prob 0.6 --tile_max_prob 0.7
python examples/research_cod/get_frozen_lake_data.py --local_dir examples/research_cod/data/frozen_lake_6767 \
    --train_size 50000 --test_size 4000 --map_min_size 6 --map_max_size 7 --tile_min_prob 0.6 --tile_max_prob 0.7
# Alchemy-Random
python examples/research_cod/get_alchemy_data.py  --local_dir examples/research_cod/data/alchemy_random --train_size 50000 --test_size 4000 --seed 42
# Terminal
python examples/research_cod/get_terminal_data.py --local_dir examples/research_cod/data/terminal --train_size 50000 --test_size 4000 --seed 42 --composite_ratio 0.5
# PDE Discovery
python examples/research_cod/get_pde_discovery_data.py \
    --local_dir examples/research_cod/data/pde_discovery_runtime_seed \
    --train_size 50000 --test_size 32 --seed 42
# Optimal Control
python examples/research_cod/get_optimal_control_data.py \
    --local_dir examples/research_cod/data/optimal_control \
    --train_size 50000 --test_size 4000 --difficulty hard --train_seed 42 --test_seed 2024
# Grid Navigation
python examples/research_cod/get_grid_navigation_data.py \
    --local_dir examples/research_cod/data/grid_navigation \
    --train_size 50000 --test_size 4000 --seed 42
```

**2. CoD 模型训练。**
将 `TRINITY_MODEL_PATH` 设为本地模型目录，并按实际硬件调整 YAML 中的 `cluster`（节点数 / 每节点 GPU 数）。在配置指定的 W&B 项目中，查看 `rollout/reward_iterative_hint_e2e_taskset_0_pos_{pos}/mean`，比较不同 pack 位置的平均奖励。
```bash
# FrozenLake-Obscure
trinity run --config examples/research_cod/exp_plan_final/train/frozen_lake_obscure.yaml
# Mixed（FrozenLake-Obscure + Alchemy-Random 联合训练）
trinity run --config examples/research_cod/exp_plan_final/train/mixed_flobs_alchran.yaml

# Qwen3.6-27B
pip install -e ".[qwen3_5]"
export TRINITY_MODEL_PATH=/path/to/Qwen3.6-27B

# PDE Discovery
trinity run --config examples/research_cod/exp_plan_final/train/pde_discovery_cod_600steps_hard.yaml
# Optimal Control
trinity run --config examples/research_cod/exp_plan_final/train/optimal_control_improve.yaml
# Grid Navigation
trinity run --config examples/research_cod/exp_plan_final/train/grid_navigation_27b.yaml
```

Mixed OPD 每个环境使用一个教师模型，tokenizer 须与学生兼容：

```bash
# 将教师 checkpoint 转换为 Hugging Face 格式
PDE_CKPT=/path/to/pde-run/global_step_75
CONTROL_CKPT=/path/to/control-run/global_step_200
GRID_CKPT=/path/to/grid-run/global_step_100
for ckpt in "$PDE_CKPT" "$CONTROL_CKPT" "$GRID_CKPT"; do
    trinity convert --checkpoint-dir "$ckpt" --base-model-dir "$TRINITY_MODEL_PATH"
done

# Mixed OPD：PDE + Optimal Control + Grid Navigation
export TRINITY_PDE_TEACHER_MODEL_PATH="$PDE_CKPT/actor/huggingface"
export TRINITY_OPTIMAL_CONTROL_TEACHER_MODEL_PATH="$CONTROL_CKPT/actor/huggingface"
export TRINITY_GRID_NAVIGATION_TEACHER_MODEL_PATH="$GRID_CKPT/actor/huggingface"
trinity run --config examples/research_cod/exp_plan_final/train/pde_control_grid_opd.yaml
```

**3. 评测。**
把训好的 checkpoint 逐个评测，衡量分布外（OOD）泛化：既包括 in-domain（训练环境的更难版本），也包括 cross-domain（训练中未见过的环境）。
```bash
# FrozenLake-Obscure ckpt → 同域 FrozenLake-hard，跨域 Alchemy-easy / Terminal
bash examples/research_cod/exp_plan_final/bench/run_eval.sh --train-tasks frozen_lake_obscure
# Mixed ckpt → 同域 FrozenLake-hard，跨域 Alchemy-hard / Terminal
bash examples/research_cod/exp_plan_final/bench/run_eval.sh --train-tasks mixed_flobs_alchran
```

PDE Discovery、Optimal Control 和 Grid Navigation 共用 checkpoint 评测配置。生成 PDE 评测集，Control 和 Grid 复用前面的 test split：

```bash
python examples/research_cod/get_pde_discovery_data.py \
    --local_dir examples/research_cod/data/pde_discovery_eval_hard_stratified_4000_disjoint_testonly \
    --train_size 1 --test_size 4000 --seed 20260902 \
    --eval_pack_size 8 --eval_template_count 25 \
    --eval_ground_truth_family physical_full_eval_4000.json --test_only
```

`EVAL_PROJECT`、`EVAL_GROUP`、`EVAL_NAME` 对应实际训练目录。`TRINITY_MODEL_PATH` 指向基础模型，各 `global_step_*/actor/` 中须有 `model.safetensors`。

```bash
export TRINITY_CHECKPOINT_ROOT_DIR=/path/to/checkpoints

# PDE RL checkpoint → 三个环境
EVAL_PROJECT=trinity-cod EVAL_GROUP=pde_discovery EVAL_NAME="your-pde-run" EVAL_TRAIN_DOMAIN=pde \
    trinity run --config examples/research_cod/exp_plan_final/bench/eval_source_checkpoints_all_domains.yaml
# Optimal Control RL checkpoint → 三个环境
EVAL_PROJECT=trinity-cod EVAL_GROUP=optimal_control EVAL_NAME="your-control-run" EVAL_TRAIN_DOMAIN=control \
    trinity run --config examples/research_cod/exp_plan_final/bench/eval_source_checkpoints_all_domains.yaml
# Grid Navigation RL checkpoint → 三个环境
EVAL_PROJECT=trinity-cod-final EVAL_GROUP=grid_navigation EVAL_NAME="your-grid-run" EVAL_TRAIN_DOMAIN=grid \
    trinity run --config examples/research_cod/exp_plan_final/bench/eval_source_checkpoints_all_domains.yaml
```

Mixed OPD checkpoint 评测（目录中保留第 5、10、…、60 步，第 0 步评测基础模型）：

```bash
EVAL_PROJECT=trinity-cod EVAL_GROUP=mixed_multi_teacher_opd EVAL_NAME="your-mixed-opd-eval-run" \
    trinity run --config examples/research_cod/exp_plan_final/bench/eval_mixed_opd_steps_0to60_all_domains.yaml
```

---

## 关键配置项（`cod.cod_workflow_args`）

| 字段 | 含义 |
|---|---|
| `activated_cod_methods` | 启用的 CoD 方法；主实验用 `["iterative_hint_e2e"]` |
| `hint_penalty_coef` / `length_penalty_coef` | 对 hint / 对正确解答的长度惩罚 |
| `task_pack_size` / `eval_task_pack_size` | 训练 / 评测时的 pack 大小 |

---

## 新增 CoD 环境

1. 在 `trinity/common/workflows/connect_the_dots/<env>/workflow.py` 继承 CoD 基类 workflow（可参考 `frozen_lake/workflow_obscure.py` 这个精简示例）。
2. 在 [`trinity/common/workflows/__init__.py`](../../trinity/common/workflows/__init__.py) 的 `default_mapping` 里登记 `"cod_<env>_workflow": "...workflow.CoD<Env>Workflow"`。
3. 在 `examples/research_cod/` 下添加生成器 + 配置，并设 `default_workflow_type: 'cod_<env>_workflow'`。

继承 `AsyncCoDMultiStepWorkflow`（`base_workflow.py`）后，实现函数：

- `step_async(step_num)`：为任务准备 prompt、调用模型推理、执行其动作 / 评判答案，将 reward 写入 `self.final_reward`，返回 `(是否继续, experiences)`。
- `_get_feedback()`：返回环境对这步的反馈文本（写进 `exp.info["feedback"]`，用于生成 hint）。
- `max_step_num`：单个任务的最大步数。

---

## 引用
```bibtex
@article{chen2026connect,
  title={Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning},
  author={Chen, Yanxi and Shi, Weijie and Xie, Yuexiang and Hu, Boyi and Li, Yaliang and Ding, Bolin and Zhou, Jingren},
  journal={arXiv preprint},
  year={2026}
}

@article{pan2025trinity,
  title={Trinity-rft: A general-purpose and unified framework for reinforcement fine-tuning of large language models},
  author={Pan, Xuchen and Chen, Yanxi and Chen, Yushuo and Sun, Yuchang and Chen, Daoyuan and Zhang, Wenhao and Xie, Yuexiang and Huang, Yilun and Zhang, Yilei and Gao, Dawei and others},
  journal={arXiv preprint arXiv:2505.17826},
  year={2025}
}
```
