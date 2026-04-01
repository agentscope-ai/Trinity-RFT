# 🤖🤖🤖 BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning

<p align="center">
  <a href="https://arxiv.org/abs/2510.26374">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2510.26374-b31b1b?style=flat&logo=arxiv">
  </a>
</p>

### 概览

BOTS是一个统一的LLM强化微调的**贝叶斯在线任务选择**框架。

<img src="https://gw.alicdn.com/imgextra/i2/O1CN01MO34b71y4VQnD3WRp_!!6000000006525-2-tps-1247-567.png" alt="Agentic workflows" width="700" />

BOTS 以任务选择、模型训练和后验概率更新的连续循环运行。
(1) **任务选择**：从后验概率信念中采用汤普森采样选择一批估计成功概率接近目标难度（例如，$p^*=0.5$）的任务。
(2) **模型训练和证据收集**：对 LLM 模型进行微调，从而获得所选任务批次的直接成功/失败计数（显式证据）。
对于未选择的任务，预测计数（隐式证据）由插件生成；我们引入了一种基于插值的超轻量级变体，其开销可忽略不计。
(3) **后验概率更新**：使用我们提出的广义贝叶斯更新规则融合显式和隐式证据。
### 使用

##### 第一步：环境准备

确保Trinity-RFT安装好了（[安装指南](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html)）。不需要额外的依赖。

##### 第二步：模型和数据准备

下载你想要训练的模型（例如：[Qwen2.5-1.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct)）。
下载[GURU](https://huggingface.co/datasets/LLM360/guru-RL-92k)数据集，
请参考LLM360提供的[数据准备指南](https://github.com/LLM360/Reasoning360?tab=readme-ov-file#data-preparation)和[技术报告](https://www.arxiv.org/pdf/2506.14965)。
请修改`bots.yaml`和`random.yaml`中相应的模型/数据路径。


##### （可选）客制参考评估结果

修改 `ref_eval_collect.yaml` 以设置你想要评估的参考模型，例如Qwen2.5-1.5B-Instruct。

执行以下命令启动评估：
```bash
BOTS_REF_EVAL_LOG_FILE="path/to/save/eval/logs" trinity run --config examples/bots/ref_eval_collect.yaml --plugin-dir examples/bots/workflow
```

评估日志会保存在指定的路径下。接下来将评估结果作为新列聚合到原数据集：

```bash
python examples/bots/ref_eval_collect.py \
--data-path <your/path/to/original/dataset> \
--ref-eval-path <your/path/to/bots_ref_eval_log.jsonl> \
--ref-eval-key <column name, e.g., qwen2.5_1.5b_pass_rate>
```
记得修改`bots.yaml`中的`data_selector.feature_keys`字段。

##### 第三步：训练
执行以下命令启动训练：
```bash
trinity run --config examples/bots/bots.yaml
```
相比随机选择基线的提升可以被稳定地观察到🤖🤖🤖.

<img src="https://gw.alicdn.com/imgextra/i2/O1CN0127XIYA1FHBgkXCKQ5_!!6000000000461-2-tps-947-533.png" alt="Agentic workflows" width="700" />

### 完整复现

想要完整复现我们论文中的结果，请从[这里](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/public/BOTS_verl_version.zip)下载verl版本的框架。

### 引用
如果你觉得这个代码仓库有帮助，请引用：
```
@misc{TrinityRFT,
      title={Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models},
      author={Xuchen Pan and Yanxi Chen and Yushuo Chen and Yuchang Sun and Daoyuan Chen and Wenhao Zhang and Yuexiang Xie and Yilun Huang and Yilei Zhang and Dawei Gao and Weijie Shi and Yaliang Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2505.17826},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17826},
}

@misc{BOTS,
      title={BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning},
      author={Qianli Shen and Daoyuan Chen and Yilun Huang and Zhenqing Ling and Yaliang Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2510.26374},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.26374},
}
```
