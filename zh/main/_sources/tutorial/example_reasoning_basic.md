# 快速上手

本教程介绍了使用 Trinity-RFT 运行 RFT 的快速入门指南。

## 第 0 步：环境准备

请按照[安装指南](./trinity_installation.md)中的说明进行环境设置。


## 第 1 步：模型和数据准备


**模型准备**

将 Qwen2.5-1.5B-Instruct 模型下载到本地目录 `$MODEL_PATH/Qwen2.5-1.5B-Instruct`：

```bash
# 使用 Modelscope
modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir $MODEL_PATH/Qwen2.5-1.5B-Instruct

# 使用 Huggingface
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $MODEL_PATH/Qwen2.5-1.5B-Instruct
```

更多关于模型下载的细节请参考 [ModelScope](https://modelscope.cn/docs/models/download) 或 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)。

**数据准备**

将 GSM8K 数据集下载到本地目录 `$DATASET_PATH/gsm8k`：

```bash
# 使用 Modelscope
modelscope download --dataset AI-ModelScope/gsm8k --local_dir $DATASET_PATH/gsm8k

# 使用 Huggingface
huggingface-cli download openai/gsm8k --repo-type dataset --local-dir $DATASET_PATH/gsm8k
```

更多关于数据集下载的细节请参考 [ModelScope](https://modelscope.cn/docs/datasets/download) 或 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space)。
从 ModelScope 下载的数据集可能缺少 `dtype` 字段，导致加载数据集时出错。要解决这个问题，请删除 `dataset_infos.json` 文件并重新运行实验。

## 第 2 步：配置实验并运行

### Trinity-RFT 的同步模式

我们在同步模式下运行实验，其中 Explorer 和 Trainer 轮流执行。要启用此模式，需将 `mode` 设置为 `both`（默认）并合理设置 `sync_interval`。较小的 `sync_interval` 值使训练更接近 on-policy 设置。例如，我们将 `sync_interval` 设为 1 来模拟 on-policy 场景。

### 使用 GRPO 算法

本实验使用 [`gsm8k.yaml`](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/grpo_gsm8k/gsm8k.yaml) 中的配置。以下是 `gsm8k.yaml` 中一些重要配置项：

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


### 运行实验

使用以下命令启动 RFT 流程：

```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```


## 进阶选项：将检查点转换为 Hugging Face 格式

在运行 Trinity-RFT 进行实验后，系统会自动将训练过程中的检查点（checkpoint）保存到以下路径（其中 `${group}` 段为可选项，当 `group` 为空时省略，默认即 `${project}/${name}`）：

```
${checkpoint_root_dir}/${project}/${group}/${name}
```

该目录的结构如下：

```
${checkpoint_root_dir}/${project}/${group}/${name}
├── buffer
│   ├── experience_buffer.jsonl          # 存储训练过程中生成的经验数据
│   └── explorer_output.db               # Explorer 模块输出的数据库文件
├── log                                  # 包含多个 Ray Actor 的日志
│   ├── checkpoint_monitor.log
│   ├── explorer.log
│   ├── explorer_experience_pipeline.log
│   ├── explorer_runner_0.log  ...  explorer_runner_31.log
│   ├── queue_experience_buffer.log
│   └── synchronizer.log
├── monitor                              # 监控相关文件（可能为空）
├── global_step_58                       # 示例：第 58 步的完整检查点
│   └── actor
│       ├── huggingface                  # （可选）Hugging Face 格式的模型文件
│       │   ├── added_tokens.json
│       │   ├── chat_template.jinja
│       │   ├── config.json
│       │   ├── generation_config.json
│       │   ├── merges.txt
│       │   ├── model.safetensors        # ← 关键模型权重文件
│       │   ├── special_tokens_map.json
│       │   ├── tokenizer.json
│       │   ├── tokenizer_config.json
│       │   └── vocab.json
│       ├── extra_state_world_size_4_rank_0.pt  # 额外状态（如随机数种子等）
│       ├── ...
│       ├── fsdp_config.json             # FSDP 配置文件
│       ├── model_world_size_4_rank_0.pt ... model_world_size_4_rank_3.pt  # 分片模型参数
│       ├── optim_world_size_4_rank_0.pt ... optim_world_size_4_rank_3.pt  # 分片优化器状态
│       └── ...
├── explorer_meta.json                   # Explorer 模块的元数据
├── trainer_meta.json                    # Trainer 模块的元数据
├── latest_checkpointed_iteration.txt    # 最近一次完整检查点的训练步数
└── latest_state_dict_iteration.txt      # 最近一次保存模型参数的训练步数（用于 checkpoint 同步）
```

### 何时需要转换？

如果你希望使用 **Hugging Face 格式** 的模型（例如用于推理或部署），但发现 `global_step_*/actor/huggingface/` 目录中 **缺少 `model.safetensors` 文件**，就需要手动执行转换。

### 转换工具：`trinity convert`

`trinity convert` 命令提供了灵活的模型转换功能，支持以下几种使用方式：

#### ✅ 批量转换（推荐）
将 `--checkpoint-dir` 指向项目根目录（即包含多个 `global_step_*` 子目录的路径），工具会**自动递归查找所有 `global_step_*` 目录**，并对每个检查点执行转换。

```bash
trinity convert --checkpoint-dir ${checkpoint_root_dir}/${project}/${group}/${name}
```

该命令会：
- 自动识别所有形如 `global_step_数字` 的子目录；
- 对每个子目录中的 `actor` 模型进行转换；
- 将生成的 Hugging Face 格式文件（包括 `model.safetensors` 等）保存到对应的 `actor/huggingface/` 目录中。

#### ✅ 单步转换
如果只想转换某一个特定训练步的模型，可直接将 `--checkpoint-dir` 指向对应的 `global_step_XXX` 文件夹：

```bash
trinity convert --checkpoint-dir ${checkpoint_root_dir}/${project}/${group}/${name}/global_step_120
```

#### ✅ 路径容错
即使你指定了 `global_step_XXX` 内部的子路径（例如 `.../global_step_120/actor`），工具也能智能识别并正确完成转换，无需严格对齐到 `global_step_XXX` 层级。

### 特殊情况：缺少基础模型配置

如果某个 `global_step_*/actor/huggingface/` 目录中 **缺少 `config.json`**（通常是因为训练时未完整保存配置），转换过程需要原始基础模型的配置文件。此时，请通过 `--base-model-dir` 指定基础模型路径：

```bash
trinity convert \
  --checkpoint-dir ${checkpoint_root_dir}/${project}/${group}/${name} \
  --base-model-dir /path/to/your/base/model
```

> 💡 此参数适用于**所有被扫描到的检查点**。只要任意一个检查点缺少 `config.json`，就需要提供该参数。

### 注意事项

- **仅转换 Actor 模型**：当前 `trinity convert` 仅处理 `actor` 文件夹中的模型参数，**不会处理 `critic`**（即使存在）。若需转换 Critic 模型，需另行操作。
- **自动识别训练格式**：`trinity convert` 原生支持 **FSDP** 和 **Megatron** 两种分布式训练格式的检查点，**无需额外指定参数**，工具会自动检测并正确合并分片权重。
- **幂等性**：如果某个 `global_step_*` 的 `huggingface/` 目录已包含完整的 Hugging Face 文件（特别是 `model.safetensors`），该检查点将被跳过，避免重复转换。
- **性能提示**：转换过程可能较耗时，尤其是当检查点数量多或模型较大时。建议在空闲时段运行。


## 进阶选项：带 SFT warmup 的 RFT

在进行 RFT 之前，我们可以先使用 SFT 作为预热步骤。Trinity-RFT 支持通过在配置文件中设置 `stages` 来添加 SFT 预热阶段。`experience_buffer` 指定用于 SFT warmup 的数据集，`total_steps` 指定 SFT warmup 的训练步数。

```yaml
# 在 gsm8k.yaml 中正确添加以下配置
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
  - stage_name: rft  # 留空则使用原有的 RFT 配置
```

以下命令将按顺序运行 SFT 和 RFT：

```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```
