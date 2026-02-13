[**中文主页**](https://github.com/agentscope-ai/Trinity-RFT/blob/main/README_zh.md) | [**Tutorial**](https://agentscope-ai.github.io/Trinity-RFT/) | [**FAQ**](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/faq.html)

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01lvLpfw25Pl4ohGZnU_!!6000000007519-2-tps-1628-490.png" alt="Trinity-RFT" style="height: 120px;">
</div>

<h2 align="center">Trinity-RFT: A General-Purpose and Unified Framework for<br>Reinforcement Fine-Tuning of Large Language Models</h2>

<div align="center">

[![paper](http://img.shields.io/badge/cs.LG-2505.17826-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.17826)
[![doc](https://img.shields.io/badge/Docs-blue?logo=markdown)](https://agentscope-ai.github.io/Trinity-RFT/)
[![pypi](https://img.shields.io/pypi/v/trinity-rft?logo=pypi&color=026cad)](https://pypi.org/project/trinity-rft/)
![license](https://img.shields.io/badge/license-Apache--2.0-000000.svg)

</div>

## 💡 What is Trinity-RFT?

Trinity-RFT is a general-purpose, flexible and user-friendly framework for LLM reinforcement fine-tuning (RFT).
It decouples RFT into three components that work in coordination:

* **Explorer** generates experience data via agent-environment interaction;
* **Trainer** updates model weights by minimizing losses on the data;
* **Buffer** pipelines data processing throughout the RFT lifecycle.

Trinity-RFT provides functionalities for users with different backgrounds and objectives:

* 🤖 **Agent application developers:** Train LLM-powered agents and improve their capabilities in specific domains [[tutorial]](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_workflow.html)
* 🧠 **Reinforcement learning researchers:** Design, implement and validate new RL algorithms using compact, plug-and-play modules that allow non-invasive customization [[tutorial]](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_algorithm.html)
* 📊 **Data engineers:** Create RFT datasets and build data pipelines for cleaning, augmentation, and human-in-the-loop scenarios [[tutorial]](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_operator.html)

## 🚀 News

* [2026-02] [[Release Notes]](https://github.com/agentscope-ai/Trinity-RFT/releases/tag/v0.5.1) Trinity-RFT v0.5.1 released: Enhanced VLM support, logging improvements, bug fixes.
* [2026-02] [[Release Notes]](https://github.com/agentscope-ai/Trinity-RFT/releases/tag/v0.5.0) Trinity-RFT v0.5.0 released: colocate mode for single-GPU scenarios, trainer driven weight synchronization, automatic parallelism setting suggestion, and more.
* [2026-01] 🎉 Three papers accepted by ICLR 2026: [CHORD](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/mix_chord), [BOTS](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/bots), and [Group-relative REINFORCE variants](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k). Try out these new algorithms in Trinity-RFT!
* [2026-01] [[Release Notes]](https://github.com/agentscope-ai/Trinity-RFT/releases/tag/v0.4.1) Trinity-RFT v0.4.1 released: upgraded verl to v0.7.0, Tinker backend supports OpenAI API, bug fixes.
* [2026-01] Introducing [R3L](https://github.com/shiweijiezero/R3L): a systematic reflect-then-retry RL mechanism with efficient language-guided exploration and stable off-policy learning ([paper](https://arxiv.org/abs/2601.03715)).
* [2025-12] [[Release Notes]](https://github.com/agentscope-ai/Trinity-RFT/releases/tag/v0.4.0) Trinity-RFT v0.4.0 released: added [Tinker](https://thinkingmachines.ai/tinker/) backend for users **without GPUs**, add more benchmarks, enhance online RL and more.
* [2025-12] Trinity-RFT powers the medical and health business of "Taobao Shangou", enabling the AI agent to understand vague symptoms, proactively ask follow-up questions, and provide precise recommendations ([News](https://tech.china.com.cn/sx/20251201/411376.shtml)).
* [2025-11] Introducing [Learn-to-Ask](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/learn_to_ask): a framework for training proactive dialogue agents from offline expert data ([paper](https://arxiv.org/pdf/2510.25441)).
* [2025-11] Introducing [BOTS](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/bots): online RL task selection for efficient LLM fine-tuning ([paper](https://arxiv.org/pdf/2510.26374)).
* [2025-09] [Our paper](https://arxiv.org/pdf/2509.24203) reveals a novel off-policy interpretation for group-relative REINFORCE and its variants like GRPO and AsymRE ([implementation](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k)).
* [2025-08] Introducing [CHORD](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/mix_chord): dynamic SFT + RL integration for advanced LLM fine-tuning ([paper](https://arxiv.org/pdf/2508.11408)).

<details><summary> More... </summary>
<ul>
  <li> [2025-11] Trinity-RFT v0.3.3 released: bug fixes.</li>
  <li> [2025-11] Trinity-RFT v0.3.2 released: bug fixes and advanced task selection & scheduling.</li>
  <li> [2025-10] Trinity-RFT v0.3.1 released: multi-stage training support, improved agentic RL examples, LoRA support, debug mode and new RL algorithms.</li>
  <li> [2025-09] Trinity-RFT v0.3.0 released: enhanced Buffer, FSDP2 & Megatron support, multi-modal models, and new RL algorithms/examples.</li>
  <li> [2025-08] Trinity-RFT v0.2.1 released.</li>
  <li> [2025-07] Trinity-RFT v0.2.0 released.</li>
  <li> [2025-07] Technical report (arXiv v2) updated with new features, examples, and experiments: [link](https://arxiv.org/abs/2505.17826).</li>
  <li> [2025-06] Trinity-RFT v0.1.1 released.</li>
  <li> [2025-05] Trinity-RFT v0.1.0 released, plus [technical report](https://arxiv.org/abs/2505.17826).</li>
  <li> [2025-04] Trinity-RFT open sourced.</li>
</ul>
</details>

## 🔨 Tutorials and Guidelines

| Category                          | Tutorial / Guideline                                                                                             |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------|
| *Run diverse RFT modes*           | • [Quick start: GRPO on GSM8k](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_reasoning_basic.html)<br>• [Off-policy RFT](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_reasoning_advanced.html)<br>• [Fully asynchronous RFT](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_async_mode.html)<br>• [Offline learning by DPO or SFT](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_dpo.html)<br>• [RFT without local GPU (Tinker Backend)](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_tinker_backend.html) |
| *Multi-step agentic RL*           | • [Concatenated multi-turn workflow](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_multi_turn.html)<br>• [General multi-step workflow](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_step_wise.html)<br>• [ReAct workflow with an agent framework](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_react.html)<br>• [Example: train a web-search agent](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/agentscope_websearch) |
| *Full-lifecycle data pipelines*   | • [Rollout task mixing and selection](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_selector.html)<br>• [Online task curriculum](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/bots) (📝 [paper](https://arxiv.org/pdf/2510.26374))<br>• [Research project: learn-to-ask](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/learn_to_ask) (📝 [paper](https://arxiv.org/pdf/2510.25441))<br>• [Experience replay with prioritization](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/ppo_countdown_exp_replay)<br>• [Advanced data processing & human-in-the-loop](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_data_functionalities.html) |
| *Algorithm development*           | • [RL algorithm development with Trinity-RFT](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_mix_algo.html) (📝 [paper](https://arxiv.org/pdf/2508.11408))<br>• [Research project: R3L (reflect-then-retry RL)](https://github.com/shiweijiezero/R3L) (📝 [paper](https://arxiv.org/abs/2601.03715))<br>• [Research project: group-relative REINFORCE](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k) (📝 [paper](https://arxiv.org/abs/2509.24203))<br>• Non-verifiable domains: [RULER](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k_ruler), [trainable RULER](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k_trainable_ruler), [rubric-as-reward](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_rubric_as_reward) |
| *Benchmarks*                      | • [Benchmark toolkit (quick verification & experimentation)](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/README.md)<br>• [Guru-Math benchmark & comparison with veRL](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/guru_math.md)<br>• [FrozenLake benchmark & comparison with rLLM](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/frozenlake.md)<br>• [Alfworld benchmark & comparison with rLLM](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/alfworld.md) |
| *Going deeper into Trinity-RFT*   | • [Full configurations](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html)<br>• [GPU resource and training configuration guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_gpu_configs.html)<br>• [Training VLM](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_vlm)<br>• [Understand the coordination between explorer and trainer](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/synchronizer.html)<br>• [How to align configuration with veRL](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/align_with_verl.html) |

> [!TIP]
> **Recommended Learning Paths**
>
> 🆕 **New users:** [Installation](https://github.com/agentscope-ai/Trinity-RFT/blob/main/docs/sphinx_doc/source/tutorial/trinity_installation.html) → [Quick Start (GSM8K)](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_reasoning_basic.html) → [Configuration Guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html) → [GPU Resource Guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_gpu_configs.html)
>
> 🔬 **Algorithm researchers:** [Developer Guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_overview.html) → [Algorithm Development Guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_algorithm.html) → [CHORD Algorithm Example](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_mix_algo.html)
>
> 🤖 **Agent developers:** [Developer Guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_overview.html) → [Workflow Development](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_workflow.html) → [General Multi-step Workflow Example](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_multi_turn.html)

> [!NOTE]
> For more tutorials, please refer to the [Trinity-RFT documentation](https://agentscope-ai.github.io/Trinity-RFT/).

## 🌟 Key Features

* **Flexible RFT Modes:**
  - Supports synchronous/asynchronous, on-policy/off-policy, and online/offline RL.
  - Rollout and training can run separately and scale independently across devices.
  - Boost sample and time efficiency by experience replay.
  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="RFT modes supported by Trinity-RFT" width="600" />

* **Agentic RL Support:**
  - Supports both concatenated and general multi-step agentic workflows.
  - Able to directly train agent applications developed using agent frameworks like [AgentScope](https://github.com/agentscope-ai/agentscope).
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01z1i7kk1jlMEVa8ZHV_!!6000000004588-2-tps-1262-695.png" alt="Agentic workflows" width="600" />

* **Full-Lifecycle Data Pipelines:**
  - Enables pipeline processing of rollout tasks and experience samples.
  - Active data management (prioritization, cleaning, augmentation, etc.) throughout the RFT lifecycle.
  - Native support for multi-task joint learning and online task curriculum construction.
  <img src="https://img.alicdn.com/imgextra/i2/O1CN01Gk9CRw28NsL09nbOj_!!6000000007921-2-tps-2530-660.png" alt="Data pipeline design" width="720" />

* **User-Friendly Design:**
  - Plug-and-play modules and decoupled architecture, facilitating easy adoption and development.
  - Rich graphical user interfaces enable low-code usage.
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01Ti0o4320RywoAuyhN_!!6000000006847-2-tps-3840-2134.png" alt="System architecture" width="600" />

## 🔧 Supported Algorithms

| Algorithm              | Doc / Example                                                                                   | Source Code                                                                                     | Key Configurations             |
|------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| PPO [[Paper](https://arxiv.org/pdf/1707.06347)] | [[Doc](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_reasoning_basic.html)] [[Countdown Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/ppo_countdown)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/ppo_policy_loss.py)] | `algorithm_type: ppo`          |
| GRPO [[Paper](https://arxiv.org/pdf/2402.03300)] | [[Doc](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_reasoning_basic.html)] [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/grpo_advantage.py)] | `algorithm_type: grpo`         |
| SFT            | [[Mixture-of-Thoughts Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/sft_mot)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sft_loss.py)]  | `algorithm_type: sft` |
| DPO [[Paper](https://arxiv.org/pdf/2305.18290)]  | [[HumanLike Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/dpo_humanlike)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/dpo_loss.py)] | `algorithm_type: dpo` |
| CHORD 💡 [[Paper](https://arxiv.org/pdf/2508.11408)] | [[Doc](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_mix_algo.html)] [[ToolACE Example](https://github.com/agentscope-ai/Trinity-RFT/blob/main/examples/mix_chord/mix_chord_toolace.yaml)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/chord_policy_loss.py)] | `algorithm_type: mix_chord`    |
| REC Series 💡 [[Paper](https://arxiv.org/pdf/2509.24203)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k)]       | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/rec_policy_loss.py)] | `algorithm_type: rec`          |
| RLOO [[Paper](https://arxiv.org/pdf/2402.14740)] | -                                                                                               | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/rloo_advantage.py)] | `algorithm_type: rloo`         |
| REINFORCE++ [[Paper](https://arxiv.org/pdf/2501.03262)] | -                                                                                               | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/reinforce_advantage.py)] | `algorithm_type: reinforceplusplus` |
| GSPO [[Paper](https://arxiv.org/pdf/2507.18071)] | -                                                                                               | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/gspo_policy_loss.py)] | `algorithm_type: gspo`         |
| TOPR [[Paper](https://arxiv.org/pdf/2503.14286)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/topr_gsm8k)]      | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/topr_policy_loss.py)] | `algorithm_type: topr`         |
| sPPO [[Paper](https://arxiv.org/pdf/2108.05828)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/sppo_gsm8k)]      | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sppo_loss_fn.py)] | `algorithm_type: sppo`         |
| AsymRE [[Paper](https://arxiv.org/pdf/2506.20520)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/asymre_gsm8k)]    | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/asymre_advantage.py)] | `algorithm_type: asymre`       |
| CISPO [[Paper](https://arxiv.org/pdf/2506.13585)] | -                                                                                               | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/cispo_policy_loss.py)] | `algorithm_type: cispo`        |
| SAPO [[Paper](https://arxiv.org/pdf/2511.20347)] | -                                                                                               | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sapo_policy_loss.py)] | `algorithm_type: sapo`         |
| On-Policy Distillation [[Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)] [[Paper](https://arxiv.org/pdf/2306.13649)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/opd_gsm8k)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/common/workflows/on_policy_distill_workflow.py)] | `algorithm_type: on_policy_distill` |
| JSD (Jensen-Shannon Divergence) | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/opd_gsm8k/opd_gsm8k_jsd.yaml)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/jsd_advantage.py)] | `algorithm_type: jsd` |


---

## Table of Contents

- [Quick Start](#quick-start)
  - [Minimal CPU-Only Quick Start](#minimal-cpu-only-quick-start)
  - [Step 1: installation](#step-1-installation)
  - [Step 2: prepare dataset and model](#step-2-prepare-dataset-and-model)
  - [Step 3: configurations](#step-3-configurations)
  - [Step 4: run the RFT process](#step-4-run-the-rft-process)
- [Contribution Guide](#contribution-guide)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Quick Start

> [!NOTE]
> This project is currently under active development. Comments and suggestions are welcome!

### Minimal CPU-Only Quick Start

If you do not have access to a GPU, you can still try Trinity-RFT using the Tinker backend.

```bash
# Create and activate environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install Trinity-RFT with CPU-only backend
pip install -e ".[tinker]"
```

Run a simple example:

```bash
trinity run --config examples/tinker/tinker.yaml
```

This example is designed to run on CPU-only machines. See the complete [Tinker training example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/tinker) for more details.

To run Trinity-RFT on GPU machines instead, please follow the steps below.

### Step 1: Installation

Before installing, make sure your system meets the following requirements:

#### GPU Requirements

- Python: version 3.10 to 3.12 (inclusive)
- CUDA: version >= 12.8
- GPUs: At least one NVIDIA GPU with [compute capability](https://developer.nvidia.com/cuda/gpus) 8.0 or higher (e.g., RTX 30 series, A100, H100)

**Recommended for first-time users:**

* If you have no GPU → Use Tinker backend
* If you want simple setup → Use Docker
* If you want development & contribution → Use Conda / venv

#### From Source (Recommended)

If you plan to customize or contribute to Trinity-RFT, this is the best option.

First, clone the repository:

```bash
git clone https://github.com/agentscope-ai/Trinity-RFT
cd Trinity-RFT
```

Then, set up environment via one of the following options:

**Using Pre-built Docker Image (Recommended for Beginners)**

```bash
docker pull ghcr.io/agentscope-ai/trinity-rft:latest

# Run the container, replacing <path_to_your_data_and_checkpoints> with your actual path
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v <path_to_your_data_and_checkpoints>:/data \
  ghcr.io/agentscope-ai/trinity-rft:latest
```
> This image has used `uv` to install all GPU-related dependencies of Trinity-RFT. The virtual environment will be automatically activated upon entering the container (you can also manually activate it via `source /opt/venv/bin/activate` if needed). You can use `uv pip install` to add extra packages as necessary.

**Using Conda**

```bash
conda create -n trinity python=3.12
conda activate trinity

pip install -e ".[vllm,flash_attn]"

# If you have no GPU, comment out the line above and uncomment this instead:
# pip install -e ".[tinker]"

# If you encounter issues when installing flash-attn, try:
# pip install flash-attn==2.8.1 --no-build-isolation
pip install -e ".[dev]"  # for development like linting and debugging
```

**Using venv**

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -e ".[vllm,flash_attn]"

# If you have no GPU, comment out the line above and uncomment this instead:
# pip install -e ".[tinker]"

# If you encounter issues when installing flash-attn, try:
# pip install flash-attn==2.8.1 --no-build-isolation

pip install -e ".[dev]"  # for development like linting and debugging
```

**Using uv**

```bash
uv sync --extra vllm --extra dev --extra flash_attn

# If you have no GPU, try to use Tinker instead:
# uv sync --extra tinker --extra dev
```

#### Via PyPI

If you just want to use the package without modifying the code:

```bash
pip install trinity-rft
pip install flash-attn==2.8.1
```

Or with `uv`:

```bash
uv pip install trinity-rft
uv pip install flash-attn==2.8.1
```

> For training with **Megatron-LM**, please refer to [Megatron-LM Backend](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_megatron.html).
### Step 2: prepare dataset and model

Trinity-RFT supports most datasets and models from Huggingface and ModelScope.

**Prepare the model** in the local directory `$MODEL_PATH/{model_name}`:

```bash
# Using Huggingface
huggingface-cli download {model_name} --local-dir $MODEL_PATH/{model_name}
# Using Modelscope
modelscope download {model_name} --local_dir $MODEL_PATH/{model_name}
```

For more details about model downloading, see [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) or [ModelScope](https://modelscope.cn/docs/models/download).



**Prepare the dataset** in the local directory `$DATASET_PATH/{dataset_name}`:

```bash
# Using Huggingface
huggingface-cli download {dataset_name} --repo-type dataset --local-dir $DATASET_PATH/{dataset_name}
# Using Modelscope
modelscope download --dataset {dataset_name} --local_dir $DATASET_PATH/{dataset_name}
```
For more details about dataset downloading, see [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space) or [ModelScope](https://modelscope.cn/docs/datasets/download).



### Step 3: configurations

Trinity-RFT provides a web interface for configuring your RFT process.

> [!NOTE]
> This is an experimental feature, and we will continue to improve it.

To launch the web interface for minimal configurations, you can run

```bash
trinity studio --port 8080
```

Then you can configure your RFT process in the web page and generate a config file. You can save the config file for later use or run it directly as described in the following section.

Advanced users can also edit the config file directly.
We provide example config files in [`examples`](examples/).

For complete GUI features, please refer to the monorepo for [Trinity-Studio](https://github.com/modelscope/Trinity-Studio).


<details>

<summary> Example: config manager GUI </summary>

![config-manager](https://img.alicdn.com/imgextra/i1/O1CN01yhYrV01lGKchtywSH_!!6000000004791-2-tps-1480-844.png)


</details>




### Step 4: run the RFT process

Start a ray cluster:

```shell
# On master node
ray start --head
# On worker nodes
ray start --address=<master_address>
```

(Optional) You may use [Wandb](https://docs.wandb.ai/quickstart/) / [TensorBoard](https://www.tensorflow.org/tensorboard) / [MLFlow](https://mlflow.org) for better monitoring. Please refer to [this documentation](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html#monitor-configuration) for the corresponding configurations.
For example, to log in to Wandb:

```shell
export WANDB_API_KEY=<your_api_key>
wandb login
```

For command-line users, run the RFT process:

```shell
trinity run --config <config_path>
```

Example — fine-tuning Qwen2.5-1.5B-Instruct on GSM8k with GRPO:

```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```

For studio users, click "Run" in the web interface.

---

## Contribution Guide

This project is currently under active development--star the repo to watch releases for the latest updates!

We welcome all kinds of contributions from the community, including:

* Documentation improvements
* Example workflows, algorithms, and data pipelines
* Bug fixes and performance optimizations

If you're new to the project, documentation and example updates are a great place to start.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed contribution guidelines, as well as our [good-first-issue list](https://github.com/agentscope-ai/Trinity-RFT/issues/470).

## Acknowledgements

This project is built upon many excellent open-source projects, including:

+ [verl](https://github.com/volcengine/verl), [FSDP](https://pytorch.org/docs/stable/fsdp.html) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for LLM training;
+ [vLLM](https://github.com/vllm-project/vllm) for LLM inference;
+ [Data-Juicer](https://github.com/datajuicer/data-juicer?tab=readme-ov-file) for data processing pipelines;
+ [AgentScope](https://github.com/agentscope-ai/agentscope) for agentic workflow;
+ [Ray](https://github.com/ray-project/ray) for distributed systems;
+ we have also drawn inspirations from RL frameworks like [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [TRL](https://github.com/huggingface/trl), [ChatLearn](https://github.com/alibaba/ChatLearn) and [rLLM](https://github.com/rllm-org/rllm);
+ ......


## Citation

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
