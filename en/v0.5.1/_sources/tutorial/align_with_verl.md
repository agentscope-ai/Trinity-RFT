# Align configuration with veRL

This guide provides guidance for users familiar with [veRL](https://github.com/volcengine/verl) to align the parameters and metrics in Trinity-RFT with the ones in veRL.

Trinity-RFT uses [veRL](https://github.com/volcengine/verl) as the training backend (`trainer`), including the actor, reference, and critic models. The `explorer` module in Trinity-RFT is implemented based on [vllm](https://github.com/vllm-project/vllm), replacing veRL's native rollout engine. Besides, Trinity-RFT introduces a new module `buffer` to enhance RFT's full-lifecycle data pipeline, which can be understood as a further enhancement of veRL's RL dataset and DataProto.


## Parameter Mapping

The core parameters in veRL are divided into these categories: `algorithm`, `data`, `actor_rollout_ref`, `critic`, `reward_model`, and `trainer`.
Trinity-RFT divides massive parameters of reinforcement fine-tuning into several parts according to their functions, e.g., `algorithm`, `model`, `buffer`, `explorer`, `trainer`, `monitor`, `synchronizer`, and `cluster`.

Roughly speaking, the parameters in veRL are mapped to the following modules in Trinity-RFT:

| Configuration | veRL | Trinity-RFT |
|:----------|:-----|:-----|
| Algorithm, e.g., advantage function| `algorithm` | `algorithm` |
| Training and evaluation tasksets | `data` | `buffer.explorer_input` |
| Batch size (💡 explained later) | `data.train_batch_size` and `actor_rollout_ref.actor.ppo_mini_batch_size` | `buffer.batch_size` and `buffer.train_batch_size` |
| Actor | `actor_rollout_ref.actor` | `model` and `trainer` |
| Rollout | `actor_rollout_ref.rollout` | `explorer.rollout_model` |
| Critic | `critic` | `trainer.trainer_config.critic` |
| Reward model | `reward_model` | `explorer.auxiliary_models` |
| Some global configurations | `trainer` | `monitor`, `synchronizer`, `cluster`, etc |


In the following, we show how to map the parameters in veRL to the ones in Trinity-RFT. Please refer to the [documentation](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html) for the detailed parameter configuration of Trinity-RFT.

```{note}
To match the default training setup of veRL, we set `synchronizer.sync_style=fixed` and `synchronizer.sync_offset=0` in Trinity-RFT.
```

### Algorithm

| veRL | Trinity-RFT | Note |
|:-----|:-----|:-----|
| `algorithm.adv_estimator` | `algorithm.advantage_fn` | Pass parameters with `algorithm.advantage_fn_args` |
| `algorithm.gamma` | `algorithm.advantage_fn_args.gamma` | Along with `algorithm.advantage_fn: ppo/reinforceplusplus` |
| `algorithm.lam` | `algorithm.advantage_fn_args.lam` | Along with `algorithm.advantage_fn: ppo` |
| `algorithm.use_kl_in_reward` | `algorithm.kl_penalty_fn` | Disable KL in reward by setting `algorithm.kl_penalty_fn=none` |
| `algorithm.kl_penalty` | `algorithm.kl_penalty_fn` | Choose from `k2`, `low_var_kl`, etc |
| `algorithm.kl_ctrl.kl_coef` | `algorithm.kl_penalty_fn_args.kl_coef` | - |

💡 Detailed explanation:

* Before using args of advantage function or policy loss function (e.g., `algorithm.kl_penalty_fn_args`), a good practice is to check the source code to ensure these parameters can be processed by the corresponding function properly.


### Data

| veRL | Trinity-RFT | Note |
|:-----|:-----|:-----|
| `data.train_files` | `buffer.explorer_input.taskset.path` or `buffer.explorer_input.tasksets[i].path` | - |
| `data.val_files` | `buffer.explorer_input.eval_tasksets[i].path` | - |
| `data.prompt_key` | `buffer.explorer_input.taskset.format.prompt_key`| Taskset-specific |
| `data.response_key` | `buffer.explorer_input.taskset.format.response_key`| Taskset-specific |
| `data.train_batch_size` | `buffer.batch_size` * `synchronizer.sync_interval` | The number of tasks to be explored |
| `data.val_batch_size` | `buffer.batch_size` | Deprecated in veRL |
| `data.max_prompt_length` | `model.max_prompt_tokens` | - |
| `data.max_response_length` | `model.max_response_tokens` | - |
| `data.filter_overlong_prompts` | `model.enable_prompt_truncation` | Explained later |
| `data.truncation` | - | Equivalent to `right` |
| `data.shuffle` | `buffer.explorer_input.taskset.task_selector.selector_type:shuffle` | Taskset-specific |

💡 Detailed explanation:

* The note `taskset-specific` means you can set different parameters for each training or evaluation task in `buffer.explorer_input.tasksets[i]` or `buffer.explorer_input.eval_tasksets[i]`.

* For the parameters related to `batch size`, Trinity-RFT uses `buffer.batch_size` to control the number of tasks to be explored in each exploration step, and `buffer.train_batch_size` to control the number of tasks used in each gradient descent step. In most cases, controlling the following parameters can ensure the same effect as veRL:
    - `buffer.batch_size` in Trinity-RFT = `actor_rollout_ref.actor.ppo_mini_batch_size` in veRL
    - `buffer.train_batch_size` in Trinity-RFT (automatically) = `actor_rollout_ref.rollout.n` * `actor_rollout_ref.actor.ppo_mini_batch_size` in veRL
    - `synchronizer.sync_interval` in Trinity-RFT =  `data.train_batch_size` / `actor_rollout_ref.actor.ppo_mini_batch_size` in veRL
    - Do not set `ppo_mini_batch_size`, which is automatically set to match the effect of veRL, although the values may not be the same.

* If you want to filter the overlong prompts, you can set `model.enable_prompt_truncation=True` in Trinity-RFT. In this case, the corresponding experiences will not be counted in loss computation, and thus `truncation` side does not matter anymore.


### Actor, Rollout, and Critic

This section includes the parameters for the actor and the rollout. For easy understanding, you may think the actor in veRL (`actor_rollout_ref.actor`) as the trainer in Trinity-RFT (`trainer`), and the rollout (`actor_rollout_ref.rollout`) as the explorer (`explorer.rollout_model`).

```{note}
Any parameter in `actor_rollout_ref.rollout` in Trinity-RFT is not effective; please set them in other fields properly.
```

For advanced training configuration of veRL you can set these up in the field of `trainer.trainer_config`. For example,`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` in veRL is equivalent to `trainer.trainer_config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` in Trinity-RFT. If you want to setup the parameters in the `trainer.trainer_config` dictionary, please read the source code in `trinity/common/verl_config.py` carefully!


| veRL | Trinity-RFT | Note |
|:-----|:-----|:-----|
| `actor_rollout_ref.model.path` | `model.model_path` | - |
| `actor_rollout_ref.actor.optim` | `algorithm.optimizer` | Such as `lr` and `weight_decay` |
| `actor_rollout_ref.rollout.n` | `algorithm.repeat_times` | Eval taskset-specific: `eval_tasksets[i].repeat_times` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `buffer.batch_size` | The number of tasks to be explored in each exploration step |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `trainer.use_dynamic_bsz` | - |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `trainer.max_token_len_per_gpu` | - |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | `trainer.ulysses_sequence_parallel_size` | The sequence parallel size for the actor |
| `actor_rollout_ref.actor.grad_clip` | `trainer.grad_clip` | The gradient clip value for the actor |
| `actor_rollout_ref.actor.use_kl_loss` | `algorithm.kl_loss_fn` | If set to `none`, the KL divergence loss will not be computed |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `explorer.rollout_model.gpu_memory_utilization` | - |
| `actor_rollout_ref.rollout.temperature` | `model.temperature` | Can be taskset-specific, like `buffer.explorer_input.taskset.rollout_args.temperature` |
| `actor_rollout_ref.rollout.top_p` | `model.top_p` | Can be taskset-specific |
| `actor_rollout_ref.rollout.top_k` | `model.top_k` | Can be taskset-specific |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `explorer.rollout_model.tensor_parallel_size` | - |
| `actor_rollout_ref.rollout.val_kwargs` | `buffer.explorer_input.eval_tasksets[i]` | Taskset-specific |
| `critic.model.path` | `model.critic_model_path` | Defaults to `model.model_path` |

💡 Detailed explanation:

* The note `can be taskset-specific` (take `temperature` as an example) means you can set `model.temperature` for all the tasksets, or set different values for each task in `buffer.explorer_input.taskset.rollout_args.temperature` or `buffer.explorer_input.eval_tasksets[i].rollout_args.temperature`. A concrete example is as follows:
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

Trinity-RFT supports the taskset-specific reward functions as well as the reward models. For custom reward functions, you can set `buffer.explorer_input.default_reward_fn_type` to select the corresponding reward function; you can also set `explorer.auxiliary_models` as reward model and use them within your workflow. For example,
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
Please refer to the [configuration](https://github.com/agentscope-ai/Trinity-RFT/blob/main/examples/grpo_rubric_as_reward/rubric.yaml) and [workflow](https://github.com/agentscope-ai/Trinity-RFT/blob/main/trinity/common/workflows/rubric_judge_workflow.py) with LLM-as-a-judge for more details.


### Trainer

| veRL | Trinity-RFT | Note |
|:-----|:-----|:-----|
| `trainer.logger` | `monitor.monitor_type` | Support a chosen type and (no need to set) `console` |
| `trainer.project_name` | `project` | - |
| `trainer.experiment_name` | `name` | - |
| `trainer.default_local_dir` | `checkpoint_root_dir` | Checkpoint is saved in `<checkpoint_root_dir>/<project>/<name>/` |
| `trainer.n_gpus_per_node` | `cluster.gpu_per_node` | - |
| `trainer.nnodes` | `cluster.node_num` | - |
| `trainer.save_freq` | `trainer.save_interval` | - |
| `trainer.test_freq` | `explorer.eval_interval` | - |
| `trainer.total_epochs` | `buffer.total_epochs` | - |
| `trainer.total_training_steps` | `buffer.total_steps` and `trainer.total_steps` | If not None, `buffer.total_epochs` will be ignored |
| `trainer.critic_warmup` | `trainer.trainer_config.trainer.critic_warmup` | - |
| `trainer.val_before_train` | `explorer.eval_on_startup` | - |
| `trainer.resume_mode` | `continue_from_checkpoint` | Explained later |
| `trainer.resume_from_path` | - | Explained later |

💡 Detailed explanation:

* If you want to resume training from a checkpoint, you can set `continue_from_checkpoint` to `True` and the training will start from the latest checkpoint in the checkpoint path `<checkpoint_root_dir>/<project>/<name>/` (if any).


## GPU Resource Allocation

In Trinity-RFT, the GPU resource is allocated to the `explorer`, `auxiliary models` (if any), and `trainer` manually.

* There are total `cluster.node_num` nodes, and each node has `cluster.gpu_per_node` GPUs.
* The number of GPUs for the `explorer` is `explorer.rollout_model.engine_num` * `explorer.rollout_model.tensor_parallel_size`.
* The number of GPUs for auxiliary models is the sum of `explorer.auxiliary_models[i].engine_num` * `explorer.auxiliary_models[i].tensor_parallel_size`.
* The remaining GPUs are for the `trainer`.


## Metrics Mapping

### Why do we see two runs for each experiment?

In Trinity-RFT, the explorer is responsible for the rollout process, while the trainer is responsible for the training process. Metrics from these two processes are calculated independently and uploaded to the monitor as separate runs. This is why you will see two runs for each experiment, distinguished by the "_explorer" or "_trainer" suffix.


### Why are some metrics different from veRL?

Trinity-RFT uses [vllm](https://github.com/vllm-project/vllm) as the rollout engine and veRL as the training backend. Due to precision differences between these frameworks, the log probabilities computed on the given tokens may differ. As a result, some metrics (e.g., `actor/ppo_kl` and `actor/pg_clipfrac`) may differ from those observed in veRL. However, when using the same parameters with veRL, these differences are expected to be small.


## Example: PPO Training

We transfer a PPO training example `run_qwen2-7b_rm.sh` from veRL to Trinity-RFT.

The configuration file of veRL is as follows:
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

The corresponding configuration of Trinity-RFT (ppo_example.yaml) is as follows:
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
          prompt_key: prompt  # Check the dataset format
          response_key: answer # Check the dataset format
      - name: math
        storage_type: file
        path: ${oc.env:HOME}/data/math
        split: train
        format:
          prompt_key: prompt  # Check the dataset format
          response_key: answer # Check the dataset format
        rollout_args:
          temperature: 1.0
    eval_tasksets:
      - name: gsm8k_eval
        storage_type: file
        path: ${oc.env:HOME}/data/gsm8k
        split: test
        format:
          prompt_key: prompt  # Check the dataset format
          response_key: answer # Check the dataset format
      - name: math_eval
        storage_type: file
        path: ${oc.env:HOME}/data/math
        split: test
        format:
          prompt_key: prompt  # Check the dataset format
          response_key: answer # Check the dataset format

explorer:
  eval_interval: 5  # trainer.test_freq
  eval_on_startup: false  # trainer.val_before_train=False
  rollout_model:
    engine_num: 2 # The number of GPUs for the rollout model
    tensor_parallel_size: 1  # actor_rollout_ref.rollout.tensor_model_parallel_size
    gpu_memory_utilization: 0.6  # actor_rollout_ref.rollout.gpu_memory_utilization
  auxiliary_models:  # reward_model configuration
    - model_path: ${oc.env:HOME}/models/FsfairX-LLaMA3-RM-v0.1
      engine_num: 2 # The number of GPUs for the reward model
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
  monitor_type: wandb  # trainer.logger='["console","wandb"]' - wandb is the set value, console is default
```

The command to run this example is:
```bash
trinity run --config ppo_example.yaml
```


## Example: GRPO Training

We transfer a GRPO training example `run_deepseek7b_llm_seq_balance.sh` from veRL to Trinity-RFT.

The configuration file of veRL is as follows:
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

The corresponding configuration of Trinity-RFT (grpo_example.yaml) is as follows:
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
          prompt_key: prompt  # Check the dataset format
          response_key: answer  # Check the dataset format
    eval_tasksets:
      - name: gsm8k_eval
        storage_type: file
        path: ${oc.env:HOME}/data/gsm8k
        split: test
        format:
          prompt_key: prompt  # Check the dataset format
          response_key: answer  # Check the dataset format

explorer:
  eval_interval: 5  # trainer.test_freq
  rollout_model:
    engine_num: 1
    tensor_parallel_size: 2  # actor_rollout_ref.rollout.tensor_model_parallel_size
    gpu_memory_utilization: 0.6  # actor_rollout_ref.rollout.gpu_memory_utilization

synchronizer:
  sync_style: fixed
  sync_offset: 1
  sync_interval: 4  # data.train_batch_size / actor_rollout_ref.actor.ppo_mini_batch_size in veRL
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
  monitor_type: wandb  # trainer.logger='["console","wandb"]' - wandb is extracted, console is default
```

The command to run this example is:
```bash
trinity run --config grpo_example.yaml
```
