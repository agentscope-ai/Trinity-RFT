# RL Tutorial（配套可运行代码）：跑通并看懂强化学习

> 本仓库是《RL Tutorial：跑通并看懂强化学习》教程的**配套可运行代码**（教程正文在 [`docs/RL_tutorial/`](./docs/RL_tutorial/)，以 ALFWorld 为实验环境；网页版 [agentscope-ai.github.io/agentic-rl](https://agentscope-ai.github.io/agentic-rl)）。框架本体 [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) 作为**依赖包**安装，本仓库提供教程、示例和运行脚本。

**获取教程代码**：使用 [Trinity-RFT 的 `tutorial` 分支](https://github.com/agentscope-ai/Trinity-RFT/tree/tutorial)；阅读地址仍为 [Agentic RL 教程](https://agentscope-ai.github.io/agentic-rl/RL_tutorial/index.html)。

```bash
git clone --branch tutorial --single-branch https://github.com/agentscope-ai/Trinity-RFT.git agentic-rl
cd agentic-rl
```

## 先看结果：一条曲线证明 RL 真的有用

<div align="center">
  <img src="./docs/RL_tutorial/ch1_reward_curve.png" alt="ALFWorld 真实训练 reward 曲线：-0.09 → +0.94 → 过度训练崩回 +0.31" width="720">
  <p><b>一次真实训练（133 step、8×A100、GRPO）</b>：训练 reward <b>-0.09 → +0.94</b>；训练过头后崩回 +0.31</p>
</div>

我们把 **Qwen3-1.7B** 放进 [ALFWorld](https://github.com/alfworld/alfworld) 文字具身环境（「找到物体 → 加热/冷却/清洗 → 放到目标位置」这类多步家务任务），用 **GRPO**（每个任务采 16 条轨迹做组内相对比较）训练。这条「**涨上去又掉下来**」的完整曲线，就是本教程最好的教材：它既证明 RL 有用，也教会你 RL 的边界（为什么最终采用 step-100 checkpoint、过度训练为什么会崩）。

前五章用这次全参数实验，把 `rollout → reward → advantage → loss → 权重更新` 一层层拆开；第六章推荐动手做 LoRA32 实践，提供参考结果，并引出开放实验。配套脚本提供真实样本，**无需 GPU** 也能查看轨迹、重算统计并理解计算过程。👉 从 [`docs/RL_tutorial/`](./docs/RL_tutorial/) 开始读。

## 目录

| 路径 | 内容 |
|---|---|
| [`docs/RL_tutorial/`](./docs/RL_tutorial/) | 7 章中文教程（Markdown + 离线 HTML，入口 [`index.html`](./docs/RL_tutorial/index.html)） |
| [`run.sh`](./run.sh) | 一次跑通脚本：安装锁定环境（含 FlashAttention2）+ 生成 taskset + 启动训练 |
| [`examples/alfworld_lora32/`](./examples/alfworld_lora32/) | 第六章 LoRA32 一键运行示例：环境准备、服务启动和训练 |
| [`examples/grpo_alfworld_general_multi_step/`](./examples/grpo_alfworld_general_multi_step/) | baseline 训练配置（multi-step GRPO, G=16） |
| [`examples/gigpo_alfworld/`](./examples/gigpo_alfworld/) | GiGPO 对比配置（同一个 workflow，换算法） |
| [`examples/grpo_alfworld/`](./examples/grpo_alfworld/) | taskset 生成脚本 + 数据 |
| [`scripts/rl_tutorial/`](./scripts/rl_tutorial/) | 第 2–6 章配套拆解脚本（轨迹 / reward / advantage / 任务族分解） |

## 先跑全参数基线

前置：**Linux x86_64、Python 3.12**、[`uv`](https://github.com/astral-sh/uv)、支持 CUDA 13 的 NVIDIA 驱动、**8×GPU 单机**（4 训练 + 4 推理）、本地 [ALFWorld](https://github.com/alfworld/alfworld) 数据、一个本地 base 模型（教程用 Qwen3-1.7B）。参考环境为 8×A100 80GB；脚本会选择 Python 3.12 并安装锁定的 CUDA 运行库。

```bash
# 1) 填自己的路径（.env 已被 gitignore，切勿提交）
cp .env.example .env
#   编辑 .env：ALFWORLD_DATA=<包含 json_2.1.1/ 和 logic/ 的数据根目录>、TRINITY_MODEL_PATH=<Qwen3-1.7B 目录>

# 2) 一次跑通（安装锁定环境 → 生成 taskset → 启动 Ray → 训练）
bash run.sh
```

默认用 **TensorBoard** 记录指标，不需要 W&B 账号或 API key。脚本使用 `.venv` 中的 Ray 和训练命令；默认启动本机 Ray，固定服务端口为 **16376–16379**（worker 另用动态端口），不连接其他实验的“最新”集群，也不停止其他服务。端口被占用时会保留错误并退出，可在 `.env` 中调整 `TRINITY_RAY_PORT`。复用已有的同版本集群时，显式设置 `RAY_ADDRESS=127.0.0.1:16379` 再运行；训练结束后 Ray 服务保留。

这次历史运行从日志 step 0 到 133 约 **15.2 小时**，可按约 15 小时预留；实际耗时随环境和运行状态变化。`critic/score/mean` 从 **-0.09** 涨到 **+0.94**（step 115 峰值），随后过度训练崩回 +0.31，最终在 step 133 手动停止——**本次采用 step-100 checkpoint**。完整背景与逐章讲解见 [`docs/RL_tutorial/README.md`](./docs/RL_tutorial/README.md)。

> 只想看曲线/拆原理、不想真跑？教程第 2–6 章的配套脚本用预置真实样本即可离线运行，无需 GPU：
> ```bash
> python scripts/rl_tutorial/ch2_inspect_trajectory.py --sample compare
> python scripts/rl_tutorial/ch4_compute_advantage.py --sample --shift 0.1
> ```

## 推荐实践：LoRA32

第六章推荐你动手运行 [LoRA32 + TuFT 实践](docs/RL_tutorial/ch6_动手实验.md)：只更新约 2% 的参数，观察模型能否学会完成任务。章节提供参考训练曲线和评测结果，帮助你解读自己的训练。[LoRA32 示例](examples/alfworld_lora32/)自动复用第一章的模型和数据路径，没有配置时使用相同的默认目录。在教程根目录直接运行：

```bash
bash examples/alfworld_lora32/run.sh
```

脚本默认使用 72 个 runner，训练 150 步。准备要求和结果查看方法见 [LoRA32 运行参考](docs/RL_tutorial/lora32_reproduction.md)。

## 关于依赖与版本

- **两个实验使用同一份 Trinity 源码**：[`6513971`](https://github.com/agentscope-ai/Trinity-RFT/commit/65139711219da4338c954e546c4b8434e4f75ec2)。全参数入口由 `pyproject.toml` + `uv.lock` 安装 `trinity-rft[vllm]`；LoRA 入口由 `requirements-lora32.txt` 安装同一提交的 `[tinker]`。两者各有依赖环境和配置。
- 全参数环境固定 `verl==0.9.0`、vLLM 0.23.0、PyTorch 2.11.0、ALFWorld 0.4.2 等依赖版本，由脚本自动安装。
- FlashAttention2 使用与 TuFT 配套的 **2.8.3 社区预编译 wheel**，支持 Python 3.12 / Linux x86_64 / PyTorch 2.11 / CUDA 13 / CXX11 ABI TRUE；下载地址及 SHA-256 固定在 `uv.lock`，`uv sync` 不会再移除它。其他 Python/CUDA/架构组合不在此一键入口的支持范围内。
- 全参数入口按公开 PyPI 锁文件安装，并显式选择公开索引；终端中临时设置的 `UV_DEFAULT_INDEX` 不会替换锁文件中的下载地址，也不会改写锁文件。这与 LoRA 入口的 `uv pip` 安装流程不同。
- 当前固定的上游 Trinity 已包含 verl 0.9.0 兼容修复，无需额外修改已安装的框架文件。
- ALFWorld taskset（`examples/grpo_alfworld/alfworld_data/*.jsonl`）里的 `game_file` 是**绝对路径**，对外部机器无效；`run.sh` 会在缺失时用 `get_alfworld_data.py` 按你的 `$ALFWORLD_DATA` 重新生成。
- 密钥/路径仅通过环境变量 / `.env` 注入，仓库内不含任何真实路径或 key。

## 反馈

发现教程里的错误 / 不清楚的地方，欢迎在 [Trinity-RFT issue 页面](https://github.com/agentscope-ai/Trinity-RFT/issues)反馈（请注明 RL Tutorial），或向 `tutorial` 分支提交 PR。
