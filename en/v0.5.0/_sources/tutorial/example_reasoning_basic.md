# Quick Start

This tutorial shows a quick start guide for running RFT with Trinity-RFT.

## Step 0: Environment Preparation

Please follow the instructions in [Installation](./trinity_installation.md) to set up the environment.

## Step 1: Model and Data Preparation


**Model Preparation.**

Download the Qwen2.5-1.5B-Instruct model to the local directory `$MODEL_PATH/Qwen2.5-1.5B-Instruct`:

```bash
# Using Modelscope
modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir $MODEL_PATH/Qwen2.5-1.5B-Instruct

# Using Huggingface
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $MODEL_PATH/Qwen2.5-1.5B-Instruct
```

More details on model downloading are referred to [ModelScope](https://modelscope.cn/docs/models/download) or [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli).

**Data Preparation.**

Download the GSM8K dataset to the local directory `$DATASET_PATH/gsm8k`:

```bash
# Using Modelscope
modelscope download --dataset AI-ModelScope/gsm8k --local_dir $DATASET_PATH/gsm8k

# Using Huggingface
huggingface-cli download openai/gsm8k --repo-type dataset --local-dir $DATASET_PATH/gsm8k
```

More details on dataset downloading are referred to [ModelScope](https://modelscope.cn/docs/datasets/download) or [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space).
The dataset downloaded from ModelScope may lack the `dtype` field and cause error when loading the dataset. To solve this issue, please delete the `dataset_infos.json` file and run the experiment again.

## Step 2: Set up Configuration and Run Experiment

### Synchronous Mode of Trinity-RFT

We run the experiment in a synchronous mode where the Explorer and Trainer operate in turn. To enable this mode, we config `mode` to `both` (default) and set `sync_interval` properly. A smaller value of `sync_interval` makes the training closer to an on-policy setup. For example, we set `sync_interval` to 1 to simulate an on-policy setup.

### Use GRPO Algorithm

We use the configurations in [`gsm8k.yaml`](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k/gsm8k.yaml) for this experiment. Some important setups of `gsm8k.yaml` are listed in the following:


```yaml
project: <project_name>
name: <experiment_name>
checkpoint_root_dir: ${oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}
algorithm:
  algorithm_type: grpo
  repeat_times: 8
  optimizer:
    lr: 1e-5
model:
  model_path: ${oc.env:TRINITY_MODEL_PATH,Qwen/Qwen2.5-1.5B-Instruct}
  max_response_tokens: 1024
  max_model_len: 2048
cluster:
  node_num: 1
  gpu_per_node: 2
buffer:
  total_epochs: 1
  batch_size: 128
  explorer_input:
    taskset:
      name: gsm8k
      storage_type: file
      path: ${oc.env:TRINITY_TASKSET_PATH,openai/gsm8k}
      subset_name: 'main'
      split: 'train'
      format:
        prompt_key: 'question'
        response_key: 'answer'
      rollout_args:
        temperature: 1.0
      default_workflow_type: 'math_workflow'
    eval_tasksets:
    - name: gsm8k-eval
      storage_type: file
      path: ${oc.env:TRINITY_TASKSET_PATH,openai/gsm8k}
      subset_name: 'main'
      split: 'test'
      format:
        prompt_key: 'question'
        response_key: 'answer'
      default_workflow_type: 'math_workflow'
  trainer_input:
    experience_buffer:
      name: gsm8k_buffer
      storage_type: queue
      path: 'sqlite:///gsm8k.db'
explorer:
  eval_interval: 50
  runner_per_model: 16
  rollout_model:
    engine_num: 1
synchronizer:
  sync_method: 'nccl'
  sync_interval: 1
trainer:
  save_interval: 100
```


### Run the Experiment

Run the RFT process with the following command:

```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```

## Optional: Convert Checkpoints to Hugging Face Format

After running Trinity-RFT experiments, the system automatically saves training checkpoints to the following path:

```
${checkpoint_root_dir}/${project}/${name}
```

The directory structure is as follows:

```
${checkpoint_root_dir}/${project}/${name}
├── buffer
│   ├── experience_buffer.jsonl          # Stores experience data generated during training
│   └── explorer_output.db               # Database file output by the Explorer module
├── log                                  # Contains logs from multiple Ray Actors
│   ├── checkpoint_monitor.log
│   ├── explorer.log
│   ├── explorer_experience_pipeline.log
│   ├── explorer_runner_0.log  ...  explorer_runner_31.log
│   ├── queue_experience_buffer.log
│   └── synchronizer.log
├── monitor                              # Monitoring-related files (may be empty)
├── global_step_58                       # Example: Full checkpoint at step 58
│   └── actor
│       ├── huggingface                  # (Optional) Hugging Face formatted model files
│       │   ├── added_tokens.json
│       │   ├── chat_template.jinja
│       │   ├── config.json
│       │   ├── generation_config.json
│       │   ├── merges.txt
│       │   ├── model.safetensors        # ← Key model weights file
│       │   ├── special_tokens_map.json
│       │   ├── tokenizer.json
│       │   ├── tokenizer_config.json
│       │   └── vocab.json
│       ├── extra_state_world_size_4_rank_0.pt  # Additional state (e.g., random seeds)
│       ├── ...
│       ├── fsdp_config.json             # FSDP configuration file
│       ├── model_world_size_4_rank_0.pt ... model_world_size_4_rank_3.pt  # Sharded model parameters
│       ├── optim_world_size_4_rank_0.pt ... optim_world_size_4_rank_3.pt  # Sharded optimizer states
│       └── ...
├── explorer_meta.json                   # Metadata for the Explorer module
├── trainer_meta.json                    # Metadata for the Trainer module
├── latest_checkpointed_iteration.txt    # Training step of the most recent full checkpoint
└── latest_state_dict_iteration.txt      # Training step of the most recent model parameter save (used for checkpoint synchronization)
```

### When Is Conversion Needed?

If you wish to use the model in **Hugging Face format** (e.g., for inference or deployment), but find that the `model.safetensors` file is **missing** from the `global_step_*/actor/huggingface/` directory, you need to manually perform the conversion.

### Conversion Tool: `trinity convert`

The `trinity convert` command provides flexible model conversion capabilities and supports the following usage patterns:

#### ✅ Batch Conversion (Recommended)
Point `--checkpoint-dir` to your project root directory (i.e., the path containing multiple `global_step_*` subdirectories). The tool will **automatically recursively scan for all `global_step_*` directories** and convert each checkpoint accordingly.

```bash
trinity convert --checkpoint-dir ${checkpoint_root_dir}/${project}/${name}
```

This command will:
- Automatically detect all subdirectories matching the pattern `global_step_<number>`;
- Convert the `actor` model within each subdirectory;
- Save the resulting Hugging Face–formatted files (including `model.safetensors`, etc.) into the corresponding `actor/huggingface/` subdirectory.

#### ✅ Single-step Conversion
If you only want to convert a model from a specific training step, directly point `--checkpoint-dir` to the corresponding `global_step_XXX` folder:

```bash
trinity convert --checkpoint-dir ${checkpoint_root_dir}/${project}/${name}/global_step_120
```

#### ✅ Path Tolerance
Even if you specify a subpath inside a `global_step_XXX` directory (e.g., `.../global_step_120/actor`), the tool can intelligently recognize the correct context and complete the conversion successfully—no need to strictly align the path to the `global_step_XXX` level.

### Special Case: Missing Base Model Configuration

If a `config.json` file is **missing** from any `global_step_*/actor/huggingface/` directory (typically because the configuration wasn't fully saved during training), the conversion process requires the original base model's configuration. In this case, use `--base-model-dir` to specify the path to your base model:

```bash
trinity convert \
  --checkpoint-dir ${checkpoint_root_dir}/${project}/${name} \
  --base-model-dir /path/to/your/base/model
```

> 💡 This parameter applies to **all scanned checkpoints**. If any checkpoint lacks `config.json`, you must provide this argument.

### Notes

- **Actor Model Only**: The current `trinity convert` command only processes model parameters in the `actor` folder and **does not handle `critic` models** (even if they exist). Converting Critic models requires separate operations.
- **Automatic Training Format Detection**: `trinity convert` natively supports checkpoints from both **FSDP** and **Megatron** distributed training formats. **No additional parameters are required**—the tool automatically detects the format and correctly merges the sharded weights.
- **Idempotency**: If a `global_step_*` checkpoint already contains a complete set of Hugging Face files (especially `model.safetensors`) in its `huggingface/` directory, the conversion will be skipped to avoid redundant processing.
- **Performance Tip**: The conversion process can be time-consuming, especially when dealing with many checkpoints or large models. It's recommended to run this during off-peak hours.


## Optional: RFT with SFT Warmup

Before RFT, we may use SFT as a warmup step. Trinity-RFT supports adding SFT warmup stage before RFT by setting `stages` in the config file. The `experience_buffer` specifies the dataset used for SFT warmup, and `total_steps` specifies the number of training steps for SFT warmup.

```yaml
# Properly add the following configs in gsm8k.yaml
stages:
  - stage_name: sft_warmup
    mode: train
    algorithm:
      algorithm_type: sft
    buffer:
      train_batch_size: 128
      total_steps: 10
      trainer_input:
        experience_buffer:
          name: sft_warmup_dataset
          path: /PATH/TO/YOUR/SFT/DATASET
  - stage_name: rft  # leave empty to use the original configs for RFT
```

The following command runs SFT and RFT in sequence:

```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```
