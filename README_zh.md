[**English Homepage**](https://github.com/agentscope-ai/Trinity-RFT/blob/main/README.md) | [**中文文档**](https://agentscope-ai.github.io/Trinity-RFT/zh/) | [**常见问题**](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/faq.html)

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01lvLpfw25Pl4ohGZnU_!!6000000007519-2-tps-1628-490.png" alt="Trinity-RFT" style="height: 120px;">
</div>



<h2 align="center">Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models</h2>


<div align="center">

[![paper](http://img.shields.io/badge/cs.LG-2505.17826-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.17826)
[![doc](https://img.shields.io/badge/Docs-blue?logo=markdown)](https://agentscope-ai.github.io/Trinity-RFT/)
[![pypi](https://img.shields.io/pypi/v/trinity-rft?logo=pypi&color=026cad)](https://pypi.org/project/trinity-rft/)
![license](https://img.shields.io/badge/license-Apache--2.0-000000.svg)

</div>

## 💡 什么是 Trinity-RFT ?

Trinity-RFT 是一个通用、灵活、用户友好的大语言模型（LLM）强化微调（RFT）框架。 其将 RFT 流程解耦为三个协同运行的关键模块：

* **Explorer** 负责执行智能体-环境交互，并生成经验数据；

* **Trainer** 在经验数据上最小化损失函数，以此更新模型参数；

* **Buffer** 负责协调整个 RFT 生命周期中的数据处理流水线。


Trinity-RFT 面向不同背景和目标的用户提供相应功能：

* 🤖 **智能体应用开发者:** 训练智能体应用，以增强其在特定领域中完成任务的能力 [[教程]](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/develop_workflow.html)

* 🧠 **强化学习算法研究者:** 通过定制化简洁、可插拔的模块，设计、实现与验证新的强化学习算法 [[教程]](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/develop_algorithm.html)

* 📊 **数据工程师:** 设计针对任务定制的数据集，构建处理流水线以支持数据清洗、增强以及人类参与场景 [[教程]](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/develop_operator.html)



## 🚀 新闻

* [2026-02] [[发布说明]](https://github.com/agentscope-ai/Trinity-RFT/releases/tag/v0.5.0) Trinity-RFT v0.5.0 发布：单 GPU 场景下的 colocate 模式，trainer 驱动的权重同步，自动并行设置建议等新功能。
* [2026-01] 🎉 三篇论文被 ICLR 2026 接收：[CHORD](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/mix_chord)、[BOTS](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/bots) 和 [Group-relative REINFORCE 系列变种](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k)。在 Trinity-RFT 中尝试这些新算法吧！
* [2026-01] [[发布说明]](https://github.com/agentscope-ai/Trinity-RFT/releases/tag/v0.4.1) Trinity-RFT v0.4.1 发布：升级 verl 至 v0.7.0，Tinker 后端支持 OpenAI API，修复若干 Bug。
* [2026-01] 推出 [R3L](https://github.com/shiweijiezero/R3L)：基于反思-重试的强化学习机制，由自然语言反馈引导高效探索，并达成稳定的 off-policy 学习（[论文](https://arxiv.org/abs/2601.03715)）。
* [2025-12] [[发布说明]](https://github.com/agentscope-ai/Trinity-RFT/releases/tag/v0.4.0) Trinity-RFT v0.4.0 发布：新增[Tinker](https://thinkingmachines.ai/tinker/) 后端以支持在 **无 GPU** 的设备上训练，增加更多基准测试，增强在线 RL 等功能。
* [2025-12] Trinity-RFT 已支持 [tinker](https://thinkingmachines.ai/tinker/) 训练后端，可在**无 GPU 的设备**上进行模型训练。
* [2025-12] Trinity-RFT 助力淘宝闪购医药健康业务，让 AI 智能体能够理解模糊症状、主动询问后续问题，并提供精准推荐（[新闻](https://tech.china.com.cn/sx/20251201/411376.shtml)）。
* [2025-11] [[发布说明](https://github.com/agentscope-ai/Trinity-RFT/releases/tag/v0.3.3)] Trinity-RFT v0.3.3 发布：修复若干 Bug。
* [2025-11] 推出 [Learn-to-Ask](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/learn_to_ask)：利用离线专家数据，训练具备主动问询能力的对话智能体（[论文](https://arxiv.org/pdf/2510.25441)）.
* [2025-11] 推出 [BOTS](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/bots)：在线 RL 任务选择，实现高效 LLM 微调（[论文](https://arxiv.org/pdf/2510.26374)）。
* [2025-09] 我们的 [论文](https://arxiv.org/pdf/2509.24203) 揭示了 group-relative REINFORCE 及其变种（如 GRPO 和 AsymRE）的 off-policy 解释（[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k)）。
* [2025-08] 推出 [CHORD](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/mix_chord)：动态 SFT + RL 集成，实现进阶 LLM 微调（[论文](https://arxiv.org/pdf/2508.11408)）。

<details><summary> More... </summary>
<ul>
  <li> [2025-11] Trinity-RFT v0.3.2 发布：修复若干 Bug 并支持进阶的任务选择和调度。</li>
  <li> [2025-10] Trinity-RFT v0.3.1 发布：多阶段训练支持、改进的智能体 RL 示例、LoRA 支持、调试模式和全新 RL 算法。</li>
  <li> [2025-09] Trinity-RFT v0.3.0 发布：增强的 Buffer、FSDP2 & Megatron 支持，多模态模型，以及全新 RL 算法/示例。</li>
  <li> [2025-08] Trinity-RFT v0.2.1 发布。</li>
  <li> [2025-07] Trinity-RFT v0.2.0 发布。</li>
  <li> [2025-07] 技术报告（arXiv v2）更新，包含新功能、示例和实验：[链接](https://arxiv.org/abs/2505.17826)。</li>
  <li> [2025-06] Trinity-RFT v0.1.1 发布。</li>
  <li> [2025-05] Trinity-RFT v0.1.0 发布，同时发布 [技术报告](https://arxiv.org/abs/2505.17826)。</li>
  <li> [2025-04] Trinity-RFT 开源。</li>
</ul>
</details>



## 🔨 教程与指南


| 类别 | 教程 / 指南  |
| --- | ----|
| *运行各种 RFT 模式* | + [快速开始：在 GSM8k 上运行 GRPO](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_basic.html)<br>+ [Off-policy RFT](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_advanced.html)<br>+ [全异步 RFT](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_async_mode.html)<br>+ [通过 DPO 或 SFT 进行离线学习](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_dpo.html)<br>+ [在无GPU环境下运行RFT训练（Tinker 后端）](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_tinker_backend.html)   |
| *多轮智能体强化学习* | + [拼接多轮任务](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_multi_turn.html)<br>+ [通用多轮任务](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_step_wise.html)<br>+ [调用智能体框架中的 ReAct 工作流](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_react.html)  <br>+ [例子：训练一个网络搜索智能体](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/agentscope_websearch) |
| *全生命周期的数据流水线* | + [Rollout 任务混合与选取](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/develop_selector.html)<br>+ [在线任务选择](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/bots) (📝 [论文](https://arxiv.org/pdf/2510.26374))<br>+ [研究项目：learn-to-ask](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/learn_to_ask) (📝 [论文](https://arxiv.org/pdf/2510.25441)) <br>+ [经验回放机制](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/ppo_countdown_exp_replay)<br>+ [高级数据处理能力 &  Human-in-the-loop](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_data_functionalities.html)  |
| *强化学习算法开发* | + [使用 Trinity-RFT 进行 RL 算法开发](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_mix_algo.html) (📝 [论文](https://arxiv.org/pdf/2508.11408))<br>+ [研究项目: R3L (基于反思-重试的强化学习)](https://github.com/shiweijiezero/R3L) (📝 [论文](https://arxiv.org/abs/2601.03715))<br>+ [研究项目: group-relative REINFORCE](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k) (📝 [论文](https://arxiv.org/abs/2509.24203)) <br>+ 不可验证的领域: [RULER](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k_ruler), [可训练 RULER](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k_trainable_ruler), [rubric-as-reward](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_rubric_as_reward) |
| *基准测试* | + [基准测试工具 (快速验证与实验)](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/README.md)<br>+ [Guru-Math 测试 & 对比 veRL](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/guru_math.md)<br>+ [FrozenLake 测试 & 对比 rLLM](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/frozenlake.md)<br>+ [Alfworld 测试 & 对比 rLLM](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/alfworld.md) |
| *深入认识 Trinity-RFT* | + [完整配置指南](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/trinity_configs.html)<br>+ [GPU 资源与训练配置对应指南](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/trinity_gpu_configs.html)<br>+ [理解 explorer-trainer 同步逻辑](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/synchronizer.html)<br>+ [如何与 verl 对齐配置](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/align_with_verl.html)   |


> [!NOTE]
> 更多教程请参考 [Trinity-RFT 文档](https://agentscope-ai.github.io/Trinity-RFT/)。



## 🌟 核心特性

* **灵活的 RFT 模式：**
  - 支持同步/异步、on-policy/off-policy 以及在线/离线强化学习
  - 采样与训练可分离运行，并可在多设备上独立扩展
  - 支持经验回放，进一步提升样本与时间效率

  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="Trinity-RFT 支持的 RFT 模式" width="600" />

* **Agentic RL 支持：**
  - 支持拼接式多轮和通用多轮交互
  - 能够直接训练使用 [AgentScope](https://github.com/agentscope-ai/agentscope) 等智能体框架开发的 Agent 应用

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01z1i7kk1jlMEVa8ZHV_!!6000000004588-2-tps-1262-695.png" alt="智能体工作流" width="600" />

* **全生命周期的数据流水线：**
  - 支持 rollout 任务和经验数据的流水线处理
  - 贯穿 RFT 生命周期的主动数据管理（优先级排序、清洗、增强等）
  - 原生支持多任务联合训练与课程学习

  <img src="https://img.alicdn.com/imgextra/i2/O1CN01Gk9CRw28NsL09nbOj_!!6000000007921-2-tps-2530-660.png" alt="数据流水线设计" width="600" />

* **用户友好的框架设计：**
  - 即插即用模块与解耦式架构，便于快速上手和二次开发
  - 丰富的图形界面，支持低代码使用

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01Ti0o4320RywoAuyhN_!!6000000006847-2-tps-3840-2134.png" alt="系统架构" width="600" />


## 🔨 算法支持

下表列出了 Trinity-RFT 支持的算法，更多算法请参考 [算法模块](https://github.com/agentscope-ai/Trinity-RFT/blob/main/trinity/algorithm/algorithm.py)。您也可以通过自定义不同的模块来构建新算法，参见 [教程](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/develop_algorithm.html)。

| 算法 | 文档/示例 | 核心代码 | 关键配置 |
|:-----------|:-----------|:---------------|:-----------|
| PPO [[论文](https://arxiv.org/pdf/1707.06347)] | [[文档](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_basic.html)] [[Countdown 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/ppo_countdown)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/ppo_policy_loss.py)] | `algorithm_type: ppo` |
| GRPO [[论文](https://arxiv.org/pdf/2402.03300)] | [[文档](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_basic.html)] [[GSM8K 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k)]| [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/grpo_advantage.py)] | `algorithm_type: grpo` |
| SFT            | [[Mixture-of-Thoughts 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/sft_mot)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sft_loss.py)]  | `algorithm_type: sft` |
| DPO [[论文](https://arxiv.org/pdf/2305.18290)]  | [[HumanLike 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/dpo_humanlike)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/dpo_loss.py)] | `algorithm_type: dpo` |
| CHORD 💡 [[论文](https://arxiv.org/pdf/2508.11408)] | [[文档](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_mix_algo.html)] [[ToolACE 示例](https://github.com/agentscope-ai/Trinity-RFT/blob/main/examples/mix_chord/mix_chord_toolace.yaml)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/chord_policy_loss.py)] | `algorithm_type: mix_chord` |
| REC Series 💡 [[论文](https://arxiv.org/pdf/2509.24203)] | [[GSM8K 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/rec_policy_loss.py)] | `algorithm_type: rec` |
| RLOO [[论文](https://arxiv.org/pdf/2402.14740)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/rloo_advantage.py)] | `algorithm_type: rloo` |
| REINFORCE++ [[论文](https://arxiv.org/pdf/2501.03262)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/reinforce_advantage.py)] | `algorithm_type: reinforceplusplus` |
| GSPO [[论文](https://arxiv.org/pdf/2507.18071)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/gspo_policy_loss.py)] | `algorithm_type: gspo` |
| TOPR [[论文](https://arxiv.org/pdf/2503.14286)] | [[GSM8K 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/topr_gsm8k)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/topr_policy_loss.py)] | `algorithm_type: topr` |
| sPPO [[论文](https://arxiv.org/pdf/2108.05828)] | [[GSM8K 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/sppo_gsm8k)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sppo_loss_fn.py)] | `algorithm_type: sppo` |
| AsymRE [[论文](https://arxiv.org/pdf/2506.20520)] | [[GSM8K 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/asymre_gsm8k)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/asymre_advantage.py)] | `algorithm_type: asymre` |
| CISPO [[论文](https://arxiv.org/pdf/2506.13585)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/cispo_policy_loss.py)] | `algorithm_type: cispo` |
| SAPO [[论文](https://arxiv.org/pdf/2511.20347)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sapo_policy_loss.py)] | `algorithm_type: sapo` |
| On-Policy Distillation [[博客](https://thinkingmachines.ai/blog/on-policy-distillation/)] [[论文](https://arxiv.org/pdf/2306.13649)] | [[GSM8K 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/opd_gsm8k)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/common/workflows/on_policy_distill_workflow.py)] | `algorithm_type: on_policy_distill` |
| JSD（Jensen-Shannon 散度） | [[GSM8K 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/opd_gsm8k/opd_gsm8k_jsd.yaml)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/jsd_advantage.py)] | `algorithm_type: jsd` |



---

## 目录


- [快速上手](#快速上手)
  - [第一步：安装](#第一步安装)
  - [第二步：准备数据集和模型](#第二步准备数据集和模型)
  - [第三步：准备配置文件](#第三步准备配置文件)
  - [第四步：运行 RFT 流程](#第四步运行-rft-流程)
- [贡献指南](#贡献指南)
- [致谢](#致谢)
- [引用](#引用)



## 快速上手


> [!NOTE]
> 本项目正处于活跃开发阶段。欢迎提出意见和建议！
>
> **没有 GPU？没问题！** 您仍然可以尝试使用：
> 1. 按照安装步骤进行操作（可跳过 `flash-attn` 等 GPU 专用的软件包）
> 2. 运行 **[Tinker 训练示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/tinker)**，该示例专为仅使用 CPU 的系统设计。


### 第一步：安装

在安装之前，请确保您的系统满足以下要求：

- **Python**：版本 3.10 至 3.12（含）
- **CUDA**：版本 >= 12.8
- **GPU**： 至少一块 [compute capability](https://developer.nvidia.com/cuda/gpus) 为 8.0 或更高的 NVIDIA GPU（例如 RTX 30 系列、A100、H100）

## 源码安装（推荐）

如需修改、扩展 Trinity-RFT，推荐使用此方法。

### 1. 克隆仓库

```bash
git clone https://github.com/agentscope-ai/Trinity-RFT
cd Trinity-RFT
```

### 2. 构建环境

可选择以下任一方式：

#### 使用预构建 Docker 镜像（推荐初学者使用该方法）


```bash
docker pull ghcr.io/agentscope-ai/trinity-rft:latest

# 将 <path_to_your_data_and_checkpoints> 替换为实际需要挂载的路径
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v <path_to_your_data_and_checkpoints>:/data \
  ghcr.io/agentscope-ai/trinity-rft:latest
```

> 该镜像已经通过 `uv` 安装了 Trinity-RFT 以及所有 GPU 相关依赖，且会自动激活虚拟环境（也可通过 `source /opt/venv/bin/activate` 手动激活）。必要时可使用 `uv pip install` 添加额外的包。

#### 使用 Conda

```bash
conda create -n trinity python=3.12
conda activate trinity

pip install -e ".[vllm,flash_attn]"

# 如果没有GPU，可以注释上一行的命令，改为使用Tinker：
# pip install -e ".[tinker]"

# 如果安装 flash-attn 时遇到问题，可尝试：
# pip install flash-attn==2.8.1 --no-build-isolation

pip install -e ".[dev]"  # 用于调试和开发
```

#### 使用 venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -e ".[vllm,flash_attn]"

# 如果没有GPU，可以注释上一行的命令，改为使用Tinker：
# pip install -e ".[tinker]"

# 如果安装 flash-attn 时遇到问题，可尝试：
# pip install flash-attn==2.8.1 --no-build-isolation

pip install -e ".[dev]"  # 用于调试和开发
```

#### 使用 `uv`

[`uv`](https://github.com/astral-sh/uv) 是现代的 Python 包管理工具。

```bash
uv sync --extra vllm --extra dev --extra flash_attn

# 如果没有GPU，可以改为使用Tinker：
# uv sync --extra tinker --extra dev
```

## 通过 PyPI 安装

如果您只需使用 Trinity-RFT 而不打算修改代码：

```bash
pip install trinity-rft
pip install flash-attn==2.8.1
```

或使用 `uv`：

```bash
uv pip install trinity-rft
uv pip install flash-attn==2.8.1
```

> 如需使用 **Megatron-LM** 进行训练，请参考 [Megatron-LM 支持](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_megatron.html)


### 第二步：准备数据集和模型


Trinity-RFT 支持来自 Huggingface 和 ModelScope 的大多数数据集和模型。


**准备模型**，保存到本地目录 `$MODEL_PATH/{model_name}`：

```bash
# 使用 Huggingface
huggingface-cli download {model_name} --local-dir $MODEL_PATH/{model_name}

# 使用 ModelScope
modelscope download {model_name} --local_dir $MODEL_PATH/{model_name}
```

更多关于模型下载的细节，请参考 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 或  [ModelScope](https://modelscope.cn/docs/models/download)。



**准备数据集**，保存到本地目录 `$DATASET_PATH/{dataset_name}`：

```bash
# 使用 Huggingface
huggingface-cli download {dataset_name} --repo-type dataset --local-dir $DATASET_PATH/{dataset_name}

# 使用 ModelScope
modelscope download --dataset {dataset_name} --local_dir $DATASET_PATH/{dataset_name}
```

更多关于数据集下载的细节，请参考 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space) 或 [ModelScope](https://modelscope.cn/docs/datasets/download)。



### 第三步：准备配置文件


Trinity-RFT 提供了一个 Web 界面来配置您的 RFT 流程。

> [!NOTE]
> 这是一个实验性功能，我们将持续改进。


要启动 Web 界面进行配置，您可以运行：

```bash
trinity studio --port 8080
```

然后您可以在网页上配置您的 RFT 流程并生成一个配置文件。您可以保存该配置文件以备后用，或按照下一节的描述直接运行。

高阶用户也可以直接编辑配置文件。
我们在 [`examples`](examples/) 目录中提供了一些示例配置文件。

若需完整的 GUI 功能，请参考 [Trinity-Studio](https://github.com/modelscope/Trinity-Studio) 仓库。


<details>

<summary> 示例：配置管理器 GUI </summary>

![config-manager](https://img.alicdn.com/imgextra/i1/O1CN01yhYrV01lGKchtywSH_!!6000000004791-2-tps-1480-844.png)


</details>


### 第四步：运行 RFT 流程


启动一个 Ray 集群：

```shell
# 在主节点上
ray start --head

# 在工作节点上
ray start --address=<master_address>
```

（可选）您可以使用 [Wandb](https://docs.wandb.ai/quickstart/) / [TensorBoard](https://www.tensorflow.org/tensorboard) / [MLFlow](https://mlflow.org) 等工具，更方便地监控训练流程。
相应的配置方法请参考 [这个文档](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html#monitor-configuration)。
比如使用 Wandb 时，您需要先登录：

```shell
export WANDB_API_KEY=<your_api_key>
wandb login
```

对于命令行用户，运行 RFT 流程：

```shell
trinity run --config <config_path>
```

例如，以下是在 GSM8k 数据集上使用 GRPO 微调 Qwen2.5-1.5B-Instruct 的命令：

```shell
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```

对于 Studio 用户，在 Web 界面中点击“运行”。



## 贡献指南


本项目正处于活跃开发阶段，我们欢迎来自社区的贡献！


请参阅 [贡献指南](./CONTRIBUTING.md) 了解详情。


## 致谢


本项目基于许多优秀的开源项目构建，包括：

+ [verl](https://github.com/volcengine/verl)，[FSDP](https://pytorch.org/docs/stable/fsdp.html) 和 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 用于大模型训练；
+ [vLLM](https://github.com/vllm-project/vllm) 用于大模型推理；
+ [Data-Juicer](https://github.com/datajuicer/data-juicer) 用于数据处理流水线；
+ [AgentScope](https://github.com/agentscope-ai/agentscope) 用于智能体工作流；
+ [Ray](https://github.com/ray-project/ray) 用于分布式系统；
+ 我们也从 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[TRL](https://github.com/huggingface/trl) 和 [ChatLearn](https://github.com/alibaba/ChatLearn) 等框架中汲取了灵感；
+ ......

## 引用


```bibtex
@misc{trinity-rft,
      title={Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models},
      author={Xuchen Pan and Yanxi Chen and Yushuo Chen and Yuchang Sun and Daoyuan Chen and Wenhao Zhang and Yuexiang Xie and Yilun Huang and Yilei Zhang and Dawei Gao and Yaliang Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2505.17826},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17826},
}
```
