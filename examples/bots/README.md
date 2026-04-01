# 🤖🤖🤖 BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning

<p align="center">
  <a href="https://arxiv.org/abs/2510.26374">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2510.26374-b31b1b?style=flat&logo=arxiv">
  </a>
</p>

### Overview

BOTS is a unified framework for **B**ayesian **O**nline **T**ask **S**election in LLM reinforcement finetuning.

<img src="https://gw.alicdn.com/imgextra/i2/O1CN01MO34b71y4VQnD3WRp_!!6000000006525-2-tps-1247-567.png" alt="Agentic workflows" width="700" />

BOTS operates in a continuous loop of task selection, model training, and posterior updating.
(1) **Selection**: Thompson sampling from the posterior beliefs selects a batch of tasks whose estimated success probabilities are near a target difficulty (e.g., $p^*=0.5$).
(2) **Training \& Evidence Collection**: The LLM is finetuned, yielding direct success/failure counts (_explicit evidence_) for the selected batch.
For unselected tasks, predicted counts (_implicit evidence_) are produced by a plug-in; We introduce an ultra-lightweight interpolation-based variant with negligible overhead.
(3) **Posterior Updating**: Explicit and implicit evidence are fused using our generalized Bayesian update rule.

### Usage

##### Step 1: Environment Preparation

Ensure Trinity-RFT is well installed ([Installation Guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html)). No extra dependence is required.

##### Step 2: Model & Dataset Preparation

Download the model your want to train (e.g., [Qwen2.5-1.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct)).

Download the [GURU](https://huggingface.co/datasets/LLM360/guru-RL-92k) dataset.
Also refer to the [Data Preparation Guide](https://github.com/LLM360/Reasoning360?tab=readme-ov-file#data-preparation) and the [Tech Report](https://www.arxiv.org/pdf/2506.14965) provided by the LLM360 team.

Remember to modify the model/data path in `bots.yaml` and `random.yaml` accordingly.

##### (Optional) Customize Reference Evaluation Results

Modify `ref_eval_collect.yaml` to set the reference model you want to evaluate, e.g., Qwen2.5-1.5B-Instruct.

Launch evaluation by executing:
```bash
BOTS_REF_EVAL_LOG_FILE="path/to/save/eval/logs" trinity run --config examples/bots/ref_eval_collect.yaml --plugin-dir examples/bots/workflow
```

The evaluation logs will be saved at the specified location. Then integrate the evaluation results as a new column into the original dataset:

```bash
python examples/bots/ref_eval_collect.py \
--data-path <your/path/to/original/dataset> \
--ref-eval-path <your/path/to/bots_ref_eval_log.jsonl> \
--ref-eval-key <column name, e.g., qwen2.5_1.5b_pass_rate>
```
Remember to update `data_selector.feature_keys` in `bots.yaml`.

##### Step 3: Training
Launch training by executing:
```bash
trinity run --config examples/bots/bots.yaml
```
The improvement over random selection baseline can be stably obtained 🤖🤖🤖.

<img src="https://gw.alicdn.com/imgextra/i2/O1CN0127XIYA1FHBgkXCKQ5_!!6000000000461-2-tps-947-533.png" alt="Agentic workflows" width="700" />

### Complete Reproduction

For complete reproduction of the results in our paper, please use the verl version implementation available [here](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/public/BOTS_verl_version.zip).

### Citation
If you find the repo helpful, please cite:
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
