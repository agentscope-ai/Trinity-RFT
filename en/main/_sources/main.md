## 💡 What is Trinity-RFT?


Trinity-RFT is a general-purpose, flexible and user-friendly framework for LLM reinforcement fine-tuning (RFT).
It decouples RFT into three components that work in coordination:

* **Explorer** generates experience data via agent-environment interaction;

* **Trainer** updates model weights by minimizing losses on the data;

* **Buffer** pipelines data processing throughout the RFT lifecycle.


Trinity-RFT provides functionalities for users with different backgrounds and objectives:

* 🤖 **Agent application developers:** Train LLM-powered agents and improve their capabilities in specific domains [[tutorial]](/tutorial/develop_workflow.md)

* 🧠 **Reinforcement learning researchers:** Design, implement and validate new RL algorithms using compact, plug-and-play modules that allow non-invasive customization [[tutorial]](/tutorial/develop_algorithm.md)

* 📊 **Data engineers:** Create RFT datasets and build data pipelines for cleaning, augmentation, and human-in-the-loop scenarios [[tutorial]](/tutorial/develop_operator.md)




## 🔨 Tutorials and Guidelines


| Category | Tutorial / Guideline      |
| --- | ----|
| *Run diverse RFT modes* | + [Quick start: GRPO on GSM8k](/tutorial/example_reasoning_basic.md)<br>+ [Off-policy RFT](/tutorial/example_reasoning_advanced.md)<br>+ [Fully asynchronous RFT](/tutorial/example_async_mode.md)<br>+ [Offline learning by DPO or SFT](/tutorial/example_dpo.md)     |
| *Multi-step agentic RL* | + [Concatenated multi-turn workflow](/tutorial/example_multi_turn.md)<br>+ [General multi-step workflow](/tutorial/example_step_wise.md)<br>+ [ReAct workflow with an agent framework](/tutorial/example_react.md)  <br>+ [Example: train a web-search agent](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/agentscope_websearch) |
| *Full-lifecycle data pipelines* | + [Rollout task mixing and selection](/tutorial/develop_selector.md)<br>+ [Online task curriculum](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/bots) (📝 [paper](https://arxiv.org/pdf/2510.26374))<br>+ [Research project: learn-to-ask](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/learn_to_ask) (📝 [paper](https://arxiv.org/pdf/2510.25441)) <br>+ [Experience replay with prioritization](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/ppo_countdown_exp_replay)<br>+ [Advanced data processing & human-in-the-loop](/tutorial/example_data_functionalities.md)  |
| *Algorithm development* | + [RL algorithm development with Trinity-RFT](/tutorial/example_mix_algo.md) (📝 [paper](https://arxiv.org/pdf/2508.11408))<br>+ [Research project: R3L (reflect-then-retry RL)](https://github.com/shiweijiezero/R3L) (📝 [paper](https://arxiv.org/abs/2601.03715))<br>+ [Research project: group-relative REINFORCE](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k) (📝 [paper](https://arxiv.org/abs/2509.24203)) <br>+ Non-verifiable domains: [RULER](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k_ruler), [trainable RULER](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k_trainable_ruler), [rubric-as-reward](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_rubric_as_reward) |
| *Benchmarks* | + [Benchmark toolkit (quick verification & experimentation)](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/README.md)<br>+ [Guru-Math benchmark & comparison with veRL](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/guru_math.md)<br>+ [FrozenLake benchmark & comparison with rLLM](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/frozenlake.md)<br>+ [Alfworld benchmark & comparison with rLLM](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark/reports/alfworld.md) |
| *Going deeper into Trinity-RFT* | + [Full configurations](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html)<br>+ [GPU resource and training configuration guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_gpu_configs.html)<br>+ [Understand the coordination between explorer and trainer](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/synchronizer.html)<br>+ [How to align configuration with veRL](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/align_with_verl.html)    |





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

We list some algorithms supported by Trinity-RFT in the following table. For more details, the concrete configurations are shown in the [Algorithm module](https://github.com/agentscope-ai/Trinity-RFT/blob/main/trinity/algorithm/algorithm.py). You can also set up new algorithms by customizing different components, see [tutorial](/tutorial/develop_algorithm.md).

| Algorithm | Doc / Example | Source Code | Key Configurations |
|:-----------|:-----------|:---------------|:-----------|
| PPO [[Paper](https://arxiv.org/pdf/1707.06347)] | [[Doc](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_reasoning_basic.html)] [[Countdown Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/ppo_countdown)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/ppo_policy_loss.py)] | `algorithm_type: ppo` |
| GRPO [[Paper](https://arxiv.org/pdf/2402.03300)] | [[Doc](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_reasoning_basic.html)] [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k)]| [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/grpo_advantage.py)] | `algorithm_type: grpo` |
| SFT            | [[Mixture-of-Thoughts Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/sft_mot)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sft_loss.py)]  | `algorithm_type: sft` |
| DPO [[Paper](https://arxiv.org/pdf/2305.18290)]  | [[HumanLike Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/dpo_humanlike)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/dpo_loss.py)] | `algorithm_type: dpo` |
| CHORD 💡 [[Paper](https://arxiv.org/pdf/2508.11408)] | [[Doc](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_mix_algo.html)] [[ToolACE Example](https://github.com/agentscope-ai/Trinity-RFT/blob/main/examples/mix_chord/mix_chord_toolace.yaml)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/chord_policy_loss.py)] | `algorithm_type: mix_chord` |
| REC Series 💡 [[Paper](https://arxiv.org/pdf/2509.24203)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/rec_policy_loss.py)] | `algorithm_type: rec` |
| RLOO [[Paper](https://arxiv.org/pdf/2402.14740)] | - | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/rloo_advantage.py)] | `algorithm_type: rloo` |
| REINFORCE++ [[Paper](https://arxiv.org/pdf/2501.03262)] | - | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/reinforce_advantage.py)] | `algorithm_type: reinforceplusplus` |
| GSPO [[Paper](https://arxiv.org/pdf/2507.18071)] | - | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/gspo_policy_loss.py)] | `algorithm_type: gspo` |
| TOPR [[Paper](https://arxiv.org/pdf/2503.14286)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/topr_gsm8k)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/topr_policy_loss.py)] | `algorithm_type: topr` |
| sPPO [[Paper](https://arxiv.org/pdf/2108.05828)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/sppo_gsm8k)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sppo_loss_fn.py)] | `algorithm_type: sppo` |
| AsymRE [[Paper](https://arxiv.org/pdf/2506.20520)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/asymre_gsm8k)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/advantage_fn/asymre_advantage.py)] | `algorithm_type: asymre` |
| CISPO [[Paper](https://arxiv.org/pdf/2506.13585)] | - | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/cispo_policy_loss.py)] | `algorithm_type: cispo` |
| SAPO [[Paper](https://arxiv.org/pdf/2511.20347)] | - | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/policy_loss_fn/sapo_policy_loss.py)] | `algorithm_type: sapo` |
| On-Policy Distillation [[Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)] [[Paper](https://arxiv.org/pdf/2306.13649)] | [[GSM8K Example](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/opd_gsm8k)] | [[Code](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/common/workflows/on_policy_distill_workflow.py)] | `algorithm_type: on_policy_distill` |




## Acknowledgements

This project is built upon many excellent open-source projects, including:

+ [verl](https://github.com/volcengine/verl) and [PyTorch's FSDP](https://pytorch.org/docs/stable/fsdp.html) for LLM training;
+ [vLLM](https://github.com/vllm-project/vllm) for LLM inference;
+ [Data-Juicer](https://github.com/datajuicer/data-juicer) for data processing pipelines;
+ [AgentScope](https://github.com/agentscope-ai/agentscope) for agentic workflow;
+ [Ray](https://github.com/ray-project/ray) for distributed systems;
+ we have also drawn inspirations from RL frameworks like [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [TRL](https://github.com/huggingface/trl) and [ChatLearn](https://github.com/alibaba/ChatLearn);
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
