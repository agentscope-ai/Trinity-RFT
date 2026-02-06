## 💡 什么是 Trinity-RFT？


Trinity-RFT 是一个通用、灵活、用户友好的大语言模型（LLM）强化微调（RFT）框架。 其将 RFT 流程解耦为三个协同运行的关键模块：

* **Explorer** 负责执行智能体-环境交互，并生成经验数据；

* **Trainer** 在经验数据上最小化损失函数，以此更新模型参数；

* **Buffer** 负责协调整个 RFT 生命周期中的数据处理流水线。


Trinity-RFT 面向不同背景和目标的用户提供相应功能：

* 🤖 **智能体应用开发者:** 训练智能体应用，以增强其在特定领域中完成任务的能力 [[教程]](/tutorial/develop_workflow.md)

* 🧠 **强化学习算法研究者:** 通过定制化简洁、可插拔的模块，设计、实现与验证新的强化学习算法 [[教程]](/tutorial/develop_algorithm.md)

* 📊 **数据工程师:** 设计针对任务定制的数据集，构建处理流水线以支持数据清洗、增强以及人类参与场景 [[教程]](/tutorial/develop_operator.md)




## 🔨 教程与指南


| 类别 | 教程 / 指南  |
| --- | ----|
| *运行各种 RFT 模式* | + [快速开始：在 GSM8k 上运行 GRPO](/tutorial/example_reasoning_basic.md)<br>+ [Off-policy RFT](/tutorial/example_reasoning_advanced.md)<br>+ [全异步 RFT](/tutorial/example_async_mode.md)<br>+ [通过 DPO 或 SFT 进行离线学习](/tutorial/example_dpo.md)     |
| *多轮智能体强化学习* | + [拼接多轮任务](/tutorial/example_multi_turn.md)<br>+ [通用多轮任务](/tutorial/example_step_wise.md)<br>+ [调用智能体框架中的 ReAct 工作流](/tutorial/example_react.md)  <br>+ [例子：训练一个网络搜索智能体](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/agentscope_websearch) |
| *全生命周期的数据流水线* | + [Rollout 任务混合与选取](/tutorial/develop_selector.md)<br>+ [在线任务选择](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/bots) (📝 [论文](https://arxiv.org/pdf/2510.26374))<br>+ [研究项目：learn-to-ask](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/learn_to_ask) (📝 [论文](https://arxiv.org/pdf/2510.25441)) <br>+ [经验回放机制](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/ppo_countdown_exp_replay)<br>+ [高级数据处理能力 &  Human-in-the-loop](/tutorial/example_data_functionalities.md)  |
| *强化学习算法开发* | + [使用 Trinity-RFT 进行 RL 算法开发](/tutorial/example_mix_algo.md) (📝 [论文](https://arxiv.org/pdf/2508.11408))<br>+ [研究项目: R3L (基于反思-重试的强化学习)](https://github.com/shiweijiezero/R3L) (📝 [论文](https://arxiv.org/abs/2601.03715))<br>+ [研究项目: group-relative REINFORCE](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k) (📝 [论文](https://arxiv.org/abs/2509.24203)) <br>+ 不可验证的领域: [RULER](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k_ruler), [可训练 RULER](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k_trainable_ruler), [rubric-as-reward](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_rubric_as_reward) |
| *基准测试* | + [基准测试工具 (快速验证与实验)](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/README.md)<br>+ [Guru-Math 测试 & 对比 veRL](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/guru_math.md)<br>+ [FrozenLake 测试 & 对比 rLLM](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/frozenlake.md)<br>+ [Alfworld 测试 & 对比 rLLM](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/alfworld.md) |
| *深入认识 Trinity-RFT* | + [完整配置指南](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/trinity_configs.html)<br>+ [GPU 资源与训练配置对应指南](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/trinity_gpu_configs.html)<br>+ [理解 explorer-trainer 同步逻辑](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/synchronizer.html)<br>+ [如何与 verl 对齐配置](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/align_with_verl.html)   |


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

下表列出了 Trinity-RFT 支持的算法，更多算法请参考 [算法模块](https://github.com/agentscope-ai/Trinity-RFT/blob/main/trinity/algorithm/algorithm.py)。您也可以通过自定义不同的模块来构建新算法，参见 [教程](/tutorial/develop_algorithm.md)。

| 算法 | 文档/示例 | 核心代码 | 关键配置 |
|:-----------|:-----------|:---------------|:-----------|
| PPO [[论文](https://arxiv.org/pdf/1707.06347)] | [[文档](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_basic.html)] [[Countdown 例子](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/ppo_countdown)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/ppo_policy_loss.py)] | `algorithm_type: ppo` |
| GRPO [[论文](https://arxiv.org/pdf/2402.03300)] | [[文档](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_basic.html)] [[GSM8K 例子](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k)]| [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/grpo_advantage.py)] | `algorithm_type: grpo` |
| CHORD 💡 [[论文](https://arxiv.org/pdf/2508.11408)] | [[文档](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_mix_algo.html)] [[ToolACE 例子](https://github.com/agentscope-ai/Trinity-RFT/blob/main/examples/mix_chord/mix_chord_toolace.yaml)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/chord_policy_loss.py)] | `algorithm_type: mix_chord` |
| REC Series 💡 [[论文](https://arxiv.org/pdf/2509.24203)] | [[GSM8K 例子](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/rec_policy_loss.py)] | `algorithm_type: rec` |
| RLOO [[论文](https://arxiv.org/pdf/2402.14740)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/rloo_advantage.py)] | `algorithm_type: rloo` |
| REINFORCE++ [[论文](https://arxiv.org/pdf/2501.03262)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/reinforce_advantage.py)] | `algorithm_type: reinforceplusplus` |
| GSPO [[论文](https://arxiv.org/pdf/2507.18071)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/gspo_policy_loss.py)] | `algorithm_type: gspo` |
| TOPR [[论文](https://arxiv.org/pdf/2503.14286)] | [[GSM8K 例子](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/topr_gsm8k)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/topr_policy_loss.py)] | `algorithm_type: topr` |
| sPPO [[论文](https://arxiv.org/pdf/2108.05828)] | [[GSM8K 例子](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/sppo_gsm8k)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sppo_loss_fn.py)] | `algorithm_type: sppo` |
| AsymRE [[论文](https://arxiv.org/pdf/2506.20520)] | [[GSM8K 例子](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/asymre_gsm8k)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/asymre_advantage.py)] | `algorithm_type: asymre` |
| CISPO [[论文](https://arxiv.org/pdf/2506.13585)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/cispo_policy_loss.py)] | `algorithm_type: cispo` |
| SAPO [[论文](https://arxiv.org/pdf/2511.20347)] | - | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sapo_policy_loss.py)] | `algorithm_type: sapo` |
| On-Policy Distillation [[博客](https://thinkingmachines.ai/blog/on-policy-distillation/)] [[论文](https://arxiv.org/pdf/2306.13649)] | [[GSM8K 示例](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/on_policy_distill)] | [[代码](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/common/workflows/on_policy_distill_workflow.py)] | `algorithm_type: on_policy_distill` |



## 致谢


本项目基于许多优秀的开源项目构建，包括：

+ [verl](https://github.com/volcengine/verl)，[FSDP](https://pytorch.org/docs/stable/fsdp.html) 和 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 用于大模型训练；
+ [vLLM](https://github.com/vllm-project/vllm) 用于大模型推理；
+ [Data-Juicer](https://github.com/datajuicer/data-juicer?tab=readme-ov-file) 用于数据处理流水线；
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
