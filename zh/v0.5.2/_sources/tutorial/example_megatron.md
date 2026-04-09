(Megatron-LM)=
# Megatron-LM 支持

本指南将清晰地引导你如何使用 **Megatron-LM** 来训练模型。

```{note}
本指南假设你已经按照 {ref}`安装指南 <Installation>` 中的源码安装方式配置好了环境。如果还没有，请先参考该指南。
```

---

## 步骤 1：安装 Megatron-LM 支持

安装 Megatron-LM 相关依赖：

```bash
pip install -e ".[megatron]"

# for uv
# uv sync -extra megatron
```

另外还需要从源码安装 mbridge 和 NVIDIA 的 Apex 库以支持混合精度训练：

```bash
pip install git+https://github.com/ISEEKYAN/mbridge.git@20e9ffbbe72ae7b1df83bfe1bc3c11f7382f2612
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    --resume-retries 10 git+https://github.com/NVIDIA/apex.git

# for uv
# uv pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
#    --config-settings "--build-option=--cpp_ext" \
#    --config-settings "--build-option=--cuda_ext" \
#    --resume-retries 10 git+https://github.com/NVIDIA/apex.git
```

---

### 替代方案：使用 Docker

我们提供了 Docker 配置以简化环境管理。

#### 构建 Docker 镜像

Trinity-RFT 提供了专门用于 Megatron-LM 的 Dockerfile，位于 `scripts/docker/Dockerfile.megatron`。
可以使用以下命令构建镜像：

```bash
docker build -f scripts/docker/Dockerfile.megatron -t trinity-rft-megatron:latest .
```

> 💡 你可以在构建前自定义 Dockerfile —— 例如添加 pip 镜像源或设置 API 密钥。

#### 运行容器

```bash
docker run -it \
    --gpus all \
    --shm-size="64g" \
    --rm \
    -v $PWD:/workspace \
    -v <your_data_and_checkpoints_path>:/data \
    trinity-rft-megatron:latest
```

请将 `<your_data_and_checkpoints_path>` 替换为你机器上存储数据集和模型检查点的实际路径。

---

## 步骤 2：配置并运行训练

大多数配置设置已在 [快速入门指南](./example_reasoning_basic.md) 中涵盖。此处我们仅关注 **Megatron-LM 特有** 的配置。

### Megatron 配置示例

以下是将 actor、reference model 和 critic 配置为使用 Megatron-LM 的示例：

```yaml
actor_rollout_ref:
  ...
  actor:
    strategy: megatron  # 为保持向后兼容性保留
    megatron:
      # 模型并行设置
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1

      # offload 设置（除非内存受限，否则设为 false）
      param_offload: false
      grad_offload: false
      optimizer_offload: false

      # 使用 mBridge 进行参数导入/导出（可选）
      use_mbridge: false

      # 使用 Megatron checkpointing
      use_dist_checkpointing: false
      dist_checkpointing_path: null

      # 重计算设置（有助于训练期间节省内存）
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
  ...
  ref:
    megatron:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      param_offload: false
      grad_offload: false
      optimizer_offload: false
      use_mbridge: false
      use_dist_checkpointing: false
      dist_checkpointing_path: null
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
  ...

critic:
  strategy: megatron
  megatron:
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 1
    expert_model_parallel_size: 1
    param_offload: false
    grad_offload: false
    optimizer_offload: false
    use_mbridge: false
    use_dist_checkpointing: false
    dist_checkpointing_path: null
    override_transformer_config:
      recompute_granularity: full
      recompute_method: uniform
      recompute_num_layers: 1
  ...
```

---

### 训练 Mixture-of-Experts (MoE) 模型

如果你正在训练像 **Qwen/Qwen3-30B-A3B** 这样的 MoE 模型，则需要采用以下两种方法之一，以确保其正常工作：

1. **使用 MBridge（推荐）**：
   只需在配置文件中设置 `use_mbridge: true`。这将直接启用对 MoE 模型所需的支持。

2. **手动转换模型**：
   如果你不希望使用 MBridge，请设置 `use_mbridge: false`。在训练前，你必须先使用 **verl** 仓库中的 [Hugging Face 到 MCore 转换器](https://github.com/volcengine/verl/blob/main/scripts/converter_hf_to_mcore.py) 将 Hugging Face 模型转换为 MCore 格式。转换完成后，在配置中更新：
   - `use_dist_checkpointing: true`
   - `dist_checkpointing_path: /PATH/TO/CONVERTED/MODEL/`

> ⚠️ 重要提示：如果跳过上述任一方法，MoE 模型可能无法正确加载或训练。请务必选择以上两种方式之一。
