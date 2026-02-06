# 与 veRL 对齐训练配置

本指南为熟悉 [veRL](https://github.com/volcengine/verl) 的用户提供了将 Trinity-RFT 与 veRL 的参数和指标对齐的方法。

Trinity-RFT 使用 [veRL](https://github.com/volcengine/verl) 作为训练后端（`trainer`），包括 actor、reference 和 critic 模型。Trinity-RFT 中的 `explorer` 模块基于 [vllm](https://github.com/vllm-project/vllm) 实现，取代了 veRL 原生的 rollout 引擎。此外，Trinity-RFT 引入了新模块 `buffer` 来增强 RFT 的全生命周期数据管理，可以理解为对 veRL 的 RL dataset 和 DataProto 的进一步强化。

## 参数映射

veRL 中的核心参数分为以下几类：`algorithm`、`data`、`actor_rollout_ref`、`critic`、`reward_model` 和 `trainer`。
Trinity-RFT 根据功能将强化微调的大量参数分为几个部分，例如 `algorithm`、`model`、`buffer`、`explorer`、`trainer`、`monitor`、`synchronizer` 和 `cluster`。

大致来说，veRL 中的参数可以按照下面的方式映射到 Trinity-RFT 中：

| 配置 | veRL | Trinity-RFT |
|:----------|:-----|:-----|
| 算法，例如 Advantage 函数 | `algorithm` | `algorithm` |
| 训练和评估任务集 | `data` | `buffer.explorer_input` |
| 批次大小（💡 稍后说明） | `data.train_batch_size` 和 `actor_rollout_ref.actor.ppo_mini_batch_size` | `buffer.batch_size` 和 `buffer.train_batch_size` |
| Actor | `actor_rollout_ref.actor` | `model` 和 `trainer` |
| Rollout | `actor_rollout_ref.rollout` | `explorer.rollout_model` |
| Critic | `critic` | `trainer.trainer_config.critic` |
| 奖励模型 | `reward_model` | `explorer.auxiliary_models` |
| 一些全局配置 | `trainer` | `monitor`、`synchronizer`、`cluster` 等 |


在以下内容中，我们将展示如何将 veRL 中的参数映射到 Trinity-RFT 中的参数。有关 Trinity-RFT 的详细参数配置，请参考[文档](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/trinity_configs.html)。


```{note}
为了匹配 veRL 的默认训练设置，我们在 Trinity-RFT 中设置 `synchronizer.sync_style=fixed` 和 `synchronizer.sync_offset=0`。
```

### Algorithm

| veRL | Trinity-RFT | 说明 |
|:-----|:-----|:-----|
| `algorithm.adv_estimator` | `algorithm.advantage_fn` | 通过 `algorithm.advantage_fn_args` 传递参数 |
| `algorithm.gamma` | `algorithm.advantage_fn_args.gamma` | 与 `algorithm.advantage_fn: ppo/reinforceplusplus` 一起使用 |
| `algorithm.lam` | `algorithm.advantage_fn_args.lam` | 与 `algorithm.advantage_fn: ppo` 一起使用 |
| `algorithm.use_kl_in_reward` | `algorithm.kl_penalty_fn` | 通过设置 `algorithm.kl_penalty_fn=none` 禁用奖励中的 KL |
| `algorithm.kl_penalty` | `algorithm.kl_penalty_fn` | 从 `k2`、`low_var_kl` 等中选择 |
| `algorithm.kl_ctrl.kl_coef` | `algorithm.kl_penalty_fn_args.kl_coef` | - |

💡 详细说明：

* 在使用优势函数或策略损失函数的参数（例如 `algorithm.advantage_fn_args`）之前，建议检查源代码以确保这些参数能够被相应函数正确处理。


### Data

| veRL | Trinity-RFT | 说明 |
|:-----|:-----|:-----|
| `data.train_files` | `buffer.explorer_input.taskset.path` 或 `buffer.explorer_input.tasksets[i].path` | - |
| `data.val_files` | `buffer.explorer_input.eval_tasksets[i].path` | - |
| `data.prompt_key` | `buffer.explorer_input.taskset.format.prompt_key`| Taskset-specific |
| `data.response_key` | `buffer.explorer_input.taskset.format.response_key`| Taskset-specific |
| `data.train_batch_size` | `buffer.batch_size` * `synchronizer.sync_interval` | 要探索的任务数量 |
| `data.val_batch_size` | `buffer.batch_size` | 在 veRL 中已弃用 |
| `data.max_prompt_length` | `model.max_prompt_tokens` | - |
| `data.max_response_length` | `model.max_response_tokens` | - |
| `data.filter_overlong_prompts` | `model.enable_prompt_truncation` | 稍后说明 |
| `data.truncation` | - | 等同于 `right` |
| `data.shuffle` | `buffer.explorer_input.taskset.task_selector.selector_type:shuffle` | Taskset-specific |

💡 详细说明：

* 注释 `taskset-specific` 意味着您可以在 `buffer.explorer_input.tasksets[i]` 或 `buffer.explorer_input.eval_tasksets[i]` 中为每个训练或评估任务设置不同的参数。

* 对于与 `batch size` 相关的参数，Trinity-RFT 使用 `buffer.batch_size` 来控制每个探索步骤中要探索的任务数量，使用 `buffer.train_batch_size` 来控制每个梯度下降步骤中使用的任务数量。在大多数情况下，控制以下参数可以确保与 veRL 相同的效果：
    - Trinity-RFT 中的 `buffer.batch_size` = veRL 中的 `actor_rollout_ref.actor.ppo_mini_batch_size`
    - Trinity-RFT 中的 `buffer.train_batch_size`（自动）= veRL 中的 `actor_rollout_ref.rollout.n` * `actor_rollout_ref.actor.ppo_mini_batch_size`
    - Trinity-RFT 中的 `synchronizer.sync_interval` = veRL 中的 `data.train_batch_size` / `actor_rollout_ref.actor.ppo_mini_batch_size`
    - 不要设置 `ppo_mini_batch_size`，它会自动设置以匹配 veRL 的效果，尽管值可能不同。

* 如果您想过滤过长的提示，可以在 Trinity-RFT 中设置 `model.enable_prompt_truncation=True`。在这种情况下，相应的经验将不计入损失计算，因此 `truncation` 的方向不再重要。


### Actor、Rollout 和 Critic

本节包括 actor 和 rollout 的参数。为了便于理解，您可以将 veRL 中的 actor（`actor_rollout_ref.actor`）视为 Trinity-RFT 中的 trainer（`trainer`），将 rollout（`actor_rollout_ref.rollout`）视为 explorer（`explorer.rollout_model`）。

```{note}
Trinity-RFT 中 `actor_rollout_ref.rollout` 的任何参数都无效；请在其他字段中正确设置它们。
```

对于 veRL 的高级训练配置，您可以在 `trainer.trainer_config` 字段中设置这些参数。例如，veRL 中的 `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` 等同于 Trinity-RFT 中的 `trainer.trainer_config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`。如果您想在 `trainer.trainer_config` 字典中设置参数，请仔细阅读 `trinity/common/verl_config.py` 中的源代码！


| veRL | Trinity-RFT | 说明 |
|:-----|:-----|:-----|
| `actor_rollout_ref.model.path` | `model.model_path` | - |
| `actor_rollout_ref.actor.optim` | `algorithm.optimizer` | 例如 `lr` 和 `weight_decay` |
| `actor_rollout_ref.rollout.n` | `algorithm.repeat_times` | Eval taskset-specific：`eval_tasksets[i].repeat_times` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `buffer.batch_size` | 每个探索步骤中要探索的任务数量 |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `trainer.use_dynamic_bsz` | - |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `trainer.max_token_len_per_gpu` | - |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | `trainer.ulysses_sequence_parallel_size` | actor 的序列并行大小 |
| `actor_rollout_ref.actor.grad_clip` | `trainer.grad_clip` | actor 的梯度裁剪值 |
| `actor_rollout_ref.actor.use_kl_loss` | `algorithm.kl_loss_fn` | 如果设置为 `none`，将不计算 KL 散度损失 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `explorer.rollout_model.gpu_memory_utilization` | - |
| `actor_rollout_ref.rollout.temperature` | `model.temperature` | 可以是taskset-specific，例如 `buffer.explorer_input.taskset.rollout_args.temperature` |
| `actor_rollout_ref.rollout.top_p` | `model.top_p` | 可以是taskset-specific |
| `actor_rollout_ref.rollout.top_k` | `model.top_k` | 可以是taskset-specific |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `explorer.rollout_model.tensor_parallel_size` | - |
| `actor_rollout_ref.rollout.val_kwargs` | `buffer.explorer_input.eval_tasksets[i]` | Taskset-specific |
| `critic.model.path` | `model.critic_model_path` | 默认为 `model.model_path` |

💡 详细说明：

* 注释 `可以是taskset-specific`（以 `temperature` 为例）意味着您可以为所有任务集设置 `model.temperature`，或者在 `buffer.explorer_input.taskset.rollout_args.temperature` 或 `buffer.explorer_input.eval_tasksets[i].rollout_args.temperature` 中为每个任务设置不同的值。具体示例如下：
```yaml
buffer:
  explorer_input:
    eval_tasksets:
      - name: AIME2024
        storage_type: file
        path: HuggingFaceH4/aime_2024
        split: 'train'
        repeat_times: 32
        format:
          prompt_key: 'question'
          response_key: 'answer'
        rollout_args:
          temperature: 1.0
          top_p: 0.7
```

### Reward Model

Trinity-RFT 支持针对任务集定制的奖励函数以及奖励模型。对于自定义奖励函数，你可以通过设置 `buffer.explorer_input.default_reward_fn_type` 来选择对应的奖励函数；另外您可以设置 `explorer.auxiliary_models` 作为 reward model 并在工作流中使用它们。例如，
```yaml
buffer:
  explorer_input:
    default_reward_fn_type: 'custom_reward'
explorer:
  auxiliary_models:
    - model_path: Qwen/Qwen3-30B-A3B-Instruct-2507
      engine_num: 1
      tensor_parallel_size: 2
      enable_thinking: false
      max_prompt_tokens: 19456
      max_response_tokens: 1024
      max_model_len: 20480
```
请参考使用 LLM-as-a-judge 的[配置](https://github.com/agentscope-ai/Trinity-RFT/blob/main/examples/grpo_rubric_as_reward/rubric.yaml)和[工作流](https://github.com/agentscope-ai/Trinity-RFT/blob/main/trinity/common/workflows/rubric_judge_workflow.py)了解更多详情。


### Trainer

| veRL | Trinity-RFT | 说明 |
|:-----|:-----|:-----|
| `trainer.logger` | `monitor.monitor_type` | 支持选择的类型和（无需设置）`console` |
| `trainer.project_name` | `project` | - |
| `trainer.experiment_name` | `name` | - |
| `trainer.default_local_dir` | `checkpoint_root_dir` | 检查点保存在 `<checkpoint_root_dir>/<project>/<name>/` |
| `trainer.n_gpus_per_node` | `cluster.gpu_per_node` | - |
| `trainer.nnodes` | `cluster.node_num` | - |
| `trainer.save_freq` | `trainer.save_interval` | - |
| `trainer.test_freq` | `explorer.eval_interval` | - |
| `trainer.total_epochs` | `buffer.total_epochs` | - |
| `trainer.total_training_steps` | `buffer.total_steps` 和 `trainer.total_steps` | 如果不为 None，将忽略 `buffer.total_epochs` |
| `trainer.critic_warmup` | `trainer.trainer_config.trainer.critic_warmup` | - |
| `trainer.val_before_train` | `explorer.eval_on_startup` | - |
| `trainer.resume_mode` | `continue_from_checkpoint` | 稍后说明 |
| `trainer.resume_from_path` | - | 稍后说明 |

💡 详细说明：

* 如果您想从检查点恢复训练，可以将 `continue_from_checkpoint` 设置为 `True`，训练将从检查点路径 `<checkpoint_root_dir>/<project>/<name>/` 中的最新检查点开始（如果有的话）。


## GPU 资源分配

在 Trinity-RFT 中，GPU 资源需要手动分配给 `explorer`、`auxiliary models`（如果有）和 `trainer`。

* 总共有 `cluster.node_num` 个节点，每个节点有 `cluster.gpu_per_node` 个 GPU。
* `explorer` 使用的 GPU 数量为 `explorer.rollout_model.engine_num` * `explorer.rollout_model.tensor_parallel_size`。
* 辅助模型的 GPU 数量为 `explorer.auxiliary_models[i].engine_num` * `explorer.auxiliary_models[i].tensor_parallel_size`。
* 剩余的 GPU 用于 `trainer`。


## 指标映射

### 为什么每个实验会看到两个运行记录？

在 Trinity-RFT 中，explorer 负责 rollout 过程，而 trainer 负责训练过程。这两个过程的指标是独立计算的，并作为单独的运行上传到 monitor。这就是为什么您会看到每个实验会对应两个“run”，通过 "_explorer" 或 "_trainer" 后缀来区分。


### 为什么某些指标与 veRL 不同？

Trinity-RFT 使用 [vllm](https://github.com/vllm-project/vllm) 作为 rollout 引擎，使用 veRL 作为训练后端。由于这些框架之间的精度差异，在给定 token 上计算的对数概率可能不同。因此，某些指标（例如 `actor/ppo_kl` 和 `actor/pg_clipfrac`）可能与 veRL 中观察到的不同。但是，当使用与 veRL 相同的参数时，这些差异预计会很小。


## 示例：PPO 训练

我们将一个 PPO 训练示例 `run_qwen2-7b_rm.sh` 从 veRL 的配置转换为 Trinity-RFT 的配置。

veRL 的配置如下：
```bash
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# prepare model ckpt
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir $HOME/models/Qwen2-7B-Instruct &
huggingface-cli download sfairXC/FsfairX-LLaMA3-RM-v0.1 --local-dir $HOME/models/FsfairX-LLaMA3-RM-v0.1 &
wait

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$HOME/models/Qwen2-7B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="$HOME/models/Qwen2-7B-Instruct" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=True \
    reward_model.model.path="$HOME/models/FsfairX-LLaMA3-RM-v0.1" \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_example' \
    trainer.val_before_train=False \
    trainer.experiment_name='Qwen2-7B-Instruct_hybrid_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
```

Trinity-RFT 的相应配置（ppo_example.yaml）如下：
```yaml
project: verl_example
name: Qwen2-7B-Instruct_hybrid_rm
checkpoint_root_dir: ./checkpoints
algorithm:
  algorithm_type: ppo
  repeat_times: 1
  optimizer:
    lr: 1e-6
    lr_warmup_steps_ratio: 0.1  # actor_rollout_ref.actor.optim.lr_warmup_steps_ratio
  advantage_fn: ppo  # algorithm.adv_estimator=gae
  kl_penalty_fn: none  # algorithm.use_kl_in_reward=False
  kl_loss_fn: none  # actor_rollout_ref.actor.use_kl_loss=False

model:
  model_path: ${oc.env:HOME}/models/Qwen2-7B-Instruct
  critic_model_path: ${oc.env:HOME}/models/Qwen2-7B-Instruct  # critic.model.path
  max_prompt_tokens: 1024  # data.max_prompt_length
  max_response_tokens: 512  # data.max_response_length
  enable_prompt_truncation: true  # data.filter_overlong_prompts=True

cluster:
  node_num: 1  # trainer.nnodes
  gpu_per_node: 8  # trainer.n_gpus_per_node

buffer:
  total_epochs: 15  # trainer.total_epochs
  batch_size: 256  # actor_rollout_ref.actor.ppo_mini_batch_size
  train_batch_size: 256  # actor_rollout_ref.actor.ppo_mini_batch_size * actor_rollout_ref.rollout.n=256*1=256
  explorer_input:
    tasksets:
      - name: gsm8k
        storage_type: file
        path: ${oc.env:HOME}/data/gsm8k
        split: train
        format:
          prompt_key: prompt  # 检查数据集格式
          response_key: answer # 检查数据集格式
      - name: math
        storage_type: file
        path: ${oc.env:HOME}/data/math
        split: train
        format:
          prompt_key: prompt  # 检查数据集格式
          response_key: answer # 检查数据集格式
        rollout_args:
          temperature: 1.0
    eval_tasksets:
      - name: gsm8k_eval
        storage_type: file
        path: ${oc.env:HOME}/data/gsm8k
        split: test
        format:
          prompt_key: prompt  # 检查数据集格式
          response_key: answer # 检查数据集格式
      - name: math_eval
        storage_type: file
        path: ${oc.env:HOME}/data/math
        split: test
        format:
          prompt_key: prompt  # 检查数据集格式
          response_key: answer # 检查数据集格式

explorer:
  eval_interval: 5  # trainer.test_freq
  eval_on_startup: false  # trainer.val_before_train=False
  rollout_model:
    engine_num: 2 # rollout 模型的 GPU 数量
    tensor_parallel_size: 1  # actor_rollout_ref.rollout.tensor_model_parallel_size
    gpu_memory_utilization: 0.6  # actor_rollout_ref.rollout.gpu_memory_utilization
  auxiliary_models:  # reward_model 配置
    - model_path: ${oc.env:HOME}/models/FsfairX-LLaMA3-RM-v0.1
      engine_num: 2 # 奖励模型的 GPU 数量
      tensor_parallel_size: 1

synchronizer:
  sync_style: fixed
  sync_offset: 1
  sync_interval: 4  # sync_interval = data.train_batch_size / actor_rollout_ref.actor.ppo_mini_batch_size
  sync_timeout: 1200

trainer:
  save_interval: 20  # trainer.save_freq
  trainer_config:
    actor_rollout_ref:
      model:
        use_remove_padding: true  # actor_rollout_ref.model.use_remove_padding
        enable_gradient_checkpointing: true  # actor_rollout_ref.model.enable_gradient_checkpointing
      actor:
        ppo_micro_batch_size_per_gpu: 16  # actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        fsdp_config:
          param_offload: false  # actor_rollout_ref.actor.fsdp_config.param_offload
          optimizer_offload: false  # actor_rollout_ref.actor.fsdp_config.optimizer_offload
      rollout:
        log_prob_micro_batch_size_per_gpu: 16  # actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
    critic:
      model:
        use_remove_padding: true  # critic.model.use_remove_padding
        enable_gradient_checkpointing: true  # critic.model.enable_gradient_checkpointing
        fsdp_config:
          param_offload: false  # critic.model.fsdp_config.param_offload
          optimizer_offload: false  # critic.model.fsdp_config.optimizer_offload
      optim:
        lr: 1e-5  # critic.optim.lr
        lr_warmup_steps_ratio: 0.05  # critic.optim.lr_warmup_steps_ratio
      ppo_micro_batch_size_per_gpu: 32  # critic.ppo_micro_batch_size_per_gpu
    trainer:
      critic_warmup: 0  # trainer.critic_warmup

monitor:
  monitor_type: wandb  # trainer.logger='["console","wandb"]' - wandb 是设定值，console 是默认值
```

运行命令为：
```bash
trinity run --config ppo_example.yaml
```

## 示例：GRPO 训练

我们将一个 GRPO 训练示例 `run_deepseek7b_llm_seq_balance.sh` 从 veRL 的配置转换为 Trinity-RFT 的配置。

veRL 的配置如下：
```bash
set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='deepseek_llm_7b_function_rm_seq_packing' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
```

Trinity-RFT 的相应配置（grpo_example.yaml）如下：
```yaml
project: verl_grpo_example_gsm8k
name: deepseek_llm_7b_function_rm_seq_packing
checkpoint_root_dir: ./checkpoints
algorithm:
  algorithm_type: grpo
  repeat_times: 8  # actor_rollout_ref.rollout.n=8
  optimizer:
    lr: 1e-6  # actor_rollout_ref.actor.optim.lr
  advantage_fn: grpo  # algorithm.adv_estimator=grpo
  kl_penalty_fn: none  # algorithm.use_kl_in_reward=False
  kl_loss_fn: low_var_kl  # actor_rollout_ref.actor.kl_loss_type=low_var_kl
  kl_loss_fn_args:
    kl_coef: 0.001  # actor_rollout_ref.actor.kl_loss_coef
  entropy_loss_fn_args:
    entropy_coef: 0  # actor_rollout_ref.actor.entropy_coeff=0

model:
  model_path: deepseek-ai/deepseek-llm-7b-chat  # actor_rollout_ref.model.path
  max_prompt_tokens: 512  # data.max_prompt_length
  max_response_tokens: 512  # data.max_response_length
  enable_prompt_truncation: true  # data.filter_overlong_prompts=True

cluster:
  node_num: 1  # trainer.nnodes
  gpu_per_node: 8  # trainer.n_gpus_per_node

buffer:
  total_epochs: 15  # trainer.total_epochs
  batch_size: 256  # actor_rollout_ref.actor.ppo_mini_batch_size
  train_batch_size: 2048  # actor_rollout_ref.actor.ppo_mini_batch_size * actor_rollout_ref.rollout.n=256*8=2048
  explorer_input:
    tasksets:
      - name: gsm8k
        storage_type: file
        path: ${oc.env:HOME}/data/gsm8k
        split: train
        format:
          prompt_key: prompt  # 检查数据集格式
          response_key: answer  # 检查数据集格式
    eval_tasksets:
      - name: gsm8k_eval
        storage_type: file
        path: ${oc.env:HOME}/data/gsm8k
        split: test
        format:
          prompt_key: prompt  # 检查数据集格式
          response_key: answer  # 检查数据集格式

explorer:
  eval_interval: 5  # trainer.test_freq
  rollout_model:
    engine_num: 1
    tensor_parallel_size: 2  # actor_rollout_ref.rollout.tensor_model_parallel_size
    gpu_memory_utilization: 0.6  # actor_rollout_ref.rollout.gpu_memory_utilization

synchronizer:
  sync_style: fixed
  sync_offset: 1
  sync_interval: 4  # veRL 中的 data.train_batch_size / actor_rollout_ref.actor.ppo_mini_batch_size
  sync_timeout: 1200

trainer:
  save_interval: 20  # trainer.save_freq
  use_dynamic_bsz: true  # actor_rollout_ref.actor.use_dynamic_bsz=True
  max_token_len_per_gpu: 24000  # actor_rollout_ref.actor.ppo_max_token_len_per_gpu
  trainer_config:
    actor_rollout_ref:
      model:
        use_remove_padding: true  # actor_rollout_ref.model.use_remove_padding=True
        enable_gradient_checkpointing: true  # actor_rollout_ref.model.enable_gradient_checkpointing=True
      actor:
        fsdp_config:
          param_offload: false  # actor_rollout_ref.actor.fsdp_config.param_offload=False
          optimizer_offload: false  # actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
      ref:
        fsdp_config:
          param_offload: true  # actor_rollout_ref.ref.fsdp_config.param_offload=True
    trainer:
      critic_warmup: 0  # trainer.critic_warmup=0

monitor:
  monitor_type: wandb  # trainer.logger='["console","wandb"]' - wandb 是设定值，console 是默认值
```

运行命令为：
```bash
trinity run --config grpo_example.yaml
```
