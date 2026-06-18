**English** | [**中文**](README_zh.md)

# Connect the Dots (CoD)

**Training LLMs to "Connect the Dots" in long-lifecycle agentic deployment, with end-to-end RL.**

As an LLM-based AI agent gets deployed in an environment, it solves a long sequence of tasks while continuously exploring the environment, learning from its own experiences, and iteratively self-updating its context about the environment, thereby achieving progressively better performance on future tasks conditioned on the updated context.

CoD groups related tasks into a pack and generates a long rollout trajectory interleaving **solve-task** and **update-context** episodes: after each task, the model updates its context (a short `Hints:` block), and later tasks are solved conditioned on the updated context.
The whole pack is trained end-to-end with RL, with fine-grained credit assignment rewarding context updates that make future tasks easier.
A trained model's reward rises across pack positions, which is the signature of the elicited CoD meta-capability.

> **The full implementation lives on the [`research/cod`](https://github.com/agentscope-ai/Trinity-RFT/tree/research/cod/examples/research_cod) branch.** Check out that branch to view or run the full code.

<p align="center">
  <img src="assets/cod_overview.png" alt="CoD overview" width="760">
  <br><sub><em>Figure 1: CoD overview. Packs of related tasks roll out as interleaved solve-task and update-context episodes, trained end-to-end (CoD-Train) and applied on a new environment at deployment (CoD-Deploy).</em></sub>
</p>

---

## How it works

```
pack = [ task0, task1, task2, task3 ]      # related tasks from one taskset (size = task_pack_size; 4 shown here)

 task0  --solve(no hint)-->      reward0, feedback0  --gen hint-->  hint1
 task1  --solve(hint1)-->        reward1, feedback1  --gen hint-->  hint2
 task2  --solve(hint2)-->        reward2, feedback2  --gen hint-->  hint3
 task3  --solve(hint3)-->        reward3
```

- **Pack**: related tasks from one taskset grouped together; size set by `task_pack_size` (train) / `eval_task_pack_size` (eval), grouped in `pack_tasks()`.
- **Position**: a task's index in the pack; position k is solved with the hint distilled from the prior k tasks, so a later position means the model has learned more in context.
- **Task reward**: each task_i's reward from the environment (correct / wrong, or graded), minus length penalties on over-long solutions and hints (to encourage brevity).
- **Iterative hint**: after each solve the model updates its context (the `Hints:` block) from `(prev hint, trajectory, reward, feedback)`, and the next solve prepends it to the prompt.
- **Fine-grained credit assignment (reward-to-go)**: following the classical dynamic-programming principle, each episode (solve-task or update-context) is credited with the rewards of the current and future tasks. A context update that helps later tasks scores higher, so the model learns to write useful hints.

Key metric: `reward_iterative_hint_e2e_taskset_{ts}_pos_{pos}`, the mean reward per pack position; reward rising with position is the CoD effect.

<p align="center">
  <img src="assets/cod_effect.png" alt="reward rising across pack positions" width="820">
  <br><sub><em>Figure 2: the CoD effect. Reward rises across pack positions during training and at OOD evaluation, both in-domain (harder FrozenLake) and cross-domain (Alchemy, Terminal).</em></sub>
</p>

---

## Environments

In each environment, the tasks in a pack share something reusable: sometimes a hidden rule (action mapping, crafting recipe), sometimes a solving strategy or a pitfall. The model figures it out from interaction and feedback, writes it into a hint, and carries it to the later tasks in the pack.

| Environment | `default_workflow_type` | Transferable knowledge within a pack |
|---|---|---|
| FrozenLake-Obscure | `cod_frozenlake_obscure_workflow` | A hidden action mapping: which move each of codes 1–4 stands for |
| Alchemy-Random | `cod_random_alchemy_workflow` | A hidden crafting recipe: which elements combine into which new element |
| Terminal | `cod_terminal_workflow` | How commands and paths work and their pitfalls, and roughly where files live |
| Learn2Ask | `cod_learn2ask_workflow` | When to keep asking vs. when to stop and give a diagnosis |

The workflow implementations live under [`trinity/common/workflows/connect_the_dots/`](https://github.com/agentscope-ai/Trinity-RFT/tree/research/cod/trinity/common/workflows/connect_the_dots) on the `research/cod` branch.

---

## Run it

```bash
git clone -b research/cod https://github.com/agentscope-ai/Trinity-RFT.git
cd Trinity-RFT
conda create -n trinity python=3.12 && conda activate trinity
pip install -e ".[vllm,flash_attn]"
```

**1. Generate data.**
```bash
# FrozenLake-Obscure
python examples/research_cod/get_frozen_lake_data.py --local_dir examples/research_cod/data/frozen_lake_4567 \
    --train_size 50000 --test_size 4000 --map_min_size 4 --map_max_size 5 --tile_min_prob 0.6 --tile_max_prob 0.7
python examples/research_cod/get_frozen_lake_data.py --local_dir examples/research_cod/data/frozen_lake_6767 \
    --train_size 50000 --test_size 4000 --map_min_size 6 --map_max_size 7 --tile_min_prob 0.6 --tile_max_prob 0.7
# Alchemy-Random
python examples/research_cod/get_alchemy_data.py --local_dir examples/research_cod/data/alchemy_random --train_size 50000 --test_size 4000 --seed 42
# Terminal
python examples/research_cod/get_terminal_data.py --local_dir examples/research_cod/data/terminal --train_size 50000 --test_size 4000 --seed 42 --composite_ratio 0.5
```

**2. Train.**
```bash
# FrozenLake-Obscure
trinity run --config examples/research_cod/exp_plan_final/train/frozen_lake_obscure.yaml
# Mixed (joint training on FrozenLake-Obscure + Alchemy-Random)
trinity run --config examples/research_cod/exp_plan_final/train/mixed_flobs_alchran.yaml
```

**3. Evaluate.** (per-checkpoint OOD generalization: harder in-domain + unseen cross-domain)
```bash
# FrozenLake-Obscure ckpt → FrozenLake-hard (in-domain) + Alchemy-easy / Terminal (cross-domain)
bash examples/research_cod/exp_plan_final/bench/run_eval.sh --train-tasks frozen_lake_obscure
# Mixed ckpt → FrozenLake-hard + Alchemy-hard / Terminal
bash examples/research_cod/exp_plan_final/bench/run_eval.sh --train-tasks mixed_flobs_alchran
```

---

## Citation

```bibtex
@article{chen2026connect,
  title={Connect the Dots: Training Large Language Models for Long-Lifecycle Agentic Deployment Via Reinforcement Learning},
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
