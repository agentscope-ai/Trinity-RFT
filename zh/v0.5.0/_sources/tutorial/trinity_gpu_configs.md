# GPU 资源相关配置指南

本文档为在 **NVIDIA A100 80GB** 和 **H20 96GB** 显卡上训练 Qwen3 系列模型提供推荐的训练配置建议。
根据模型大小（0.6B ~ 14B）与上下文长度（`model.max_model_len`），我们给出了Trainer模块在不同 GPU 数量下的可行方案。

> ⚠️ **注意**
> 由于在Trinity内，采样与训练是分离的。以下关于GPU数量的描述指的是`Trainer`部分可使用的数量，而非Trinity总共使用的GPU数量。

> 💡 **术语说明**
>
> - **vanilla**：无需特殊配置，使用默认设置即可。
> - **Env**：需在启动训练前（启动ray之前）设置环境变量：
>   ```bash
>   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
>   ```
> - **Offload**：需启用 **FSDP v2 + CPU Offload** 技术以节省显存。
> - **SP=N**：表示使用 **Sequence Parallelism（序列并行）**，并行度为 N（通常 N ≤ GPU 数量）。
> - **组合项（如 `Env + SP=2`）**：需同时满足所有列出的条件。
> - **“-”**：当前硬件与配置组合下，无法支持该模型在此序列长度下进行训练。

---

## 关于长上下文支持

Qwen3 系列模型原生支持的最大上下文长度为 **40,960 tokens**。
对于超过此长度的训练（如 51,200、81,920 等），我们通过 **YaRN RoPE 扩展** 实现。相关配置如下：

```yaml
model:
  model_path: ${oc.env:MODEL_PATH,Qwen/Qwen3-0.6B}
  max_prompt_tokens: 2048
  max_model_len: ${oc.env:MAX_MODEL_LEN,4096}
  rope_scaling:
    rope_type: yarn
    factor: ${oc.decode:${oc.env:FACTOR}}  # 推荐值 = MAX_MODEL_LEN / 40960
    original_max_position_embeddings: 40960
```

> ✅ 使用 YaRN 时，请确保 `factor` 设置合理，避免数值不稳定。

---

## 💡 显存使用与 `max_token_len_per_gpu` 的关系

Trinity Trainer 默认启用了动态批大小（`trainer.use_dynamic_bsz=True`），在固定模型的情况下，实际显存消耗主要由以下两个参数决定：

- `trainer.trainer_config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu`
- `trainer.trainer_config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`

如果未手动设置，Trinity会自动用该默认值：
```python
trainer.max_token_len_per_gpu = ceil(2 * model.max_model_len / trainer.ulysses_sequence_parallel_size)
```

📌 **这意味着**：
- 上下文越长，每张 GPU 要处理的 token 越多，显存压力越大。
- 如果想支持**更长上下文**，可以手动设置上述参数（但可能影响训练效率）。

> 本指南中的所有实验结果都是基于上述默认设置得出的。如需极限优化，请根据实际情况调整这些参数。

---

## A100 80GB 显卡配置建议

> ⚠️ **单卡限制**：在 1 张 A100 上训练 ≥4B 模型或 >20K 上下文时，显存压力极大，**强烈建议使用多卡方案**。

### 1 张 GPU

<details><summary>点击查看详细配置</summary>

|   `max_model_len` | Qwen3-0.6B    | Qwen3-1.7B    | Qwen3-4B      | Qwen3-8B      | Qwen3-14B     |
|------------------:|:--------------|:--------------|:--------------|:--------------|:--------------|
|              4096 | vanilla       | vanilla       | Env + Offload | Env + Offload | Env + Offload |
|              8192 | vanilla       | vanilla       | Env + Offload | Env + Offload | Env + Offload |
|             12288 | vanilla       | vanilla       | Env + Offload | Env + Offload | Env + Offload |
|             16384 | vanilla       | vanilla       | Env + Offload | Env + Offload | Env + Offload |
|             20480 | vanilla       | Env + Offload | Env + Offload | Env + Offload | Env + Offload |
|             24576 | Env           | Env + Offload | Env + Offload | Env + Offload | Env + Offload |
|             28672 | Env + Offload | Env + Offload | Env + Offload | -             | -             |
|             32768 | -             | -             | -             | -             | -             |

</details>

---

### 2 张 GPU

<details><summary>✅ 推荐：2 卡显著提升 4B~14B 模型的长上下文训练能力，上下文较长时建议启用 SP=2</summary>

|   `max_model_len` | Qwen3-0.6B           | Qwen3-1.7B           | Qwen3-4B             | Qwen3-8B             | Qwen3-14B            |
|------------------:|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
|              4096 | vanilla              | vanilla              | vanilla              | Env                  | Env + Offload        |
|              8192 | vanilla              | vanilla              | vanilla              | Env + Offload        | Env + Offload        |
|             12288 | vanilla              | vanilla              | vanilla              | Env + Offload        | Env + Offload        |
|             16384 | vanilla              | vanilla              | Env                  | Env + Offload        | Env + Offload        |
|             20480 | vanilla              | vanilla              | SP=2                 | Env + Offload        | Env + Offload        |
|             24576 | vanilla              | Env                  | SP=2                 | Env + Offload        | Env + Offload        |
|             28672 | Env                  | SP=2                 | Env + SP=2           | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             32768 | SP=2                 | SP=2                 | Env + SP=2           | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             36864 | SP=2                 | SP=2                 | Env + Offload + SP=2 | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             40960 | SP=2                 | SP=2                 | Env + Offload + SP=2 | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             51200 | Env + SP=2           | Env + Offload + SP=2 | Env + Offload + SP=2 | Env + Offload + SP=2 | -                    |
|             61440 | Env + Offload + SP=2 | -                    | -                    | -                    | -                    |
|             71680 | -                    | -                    | -                    | -                    | -                    |

</details>

---

### 4 张 GPU

<details><summary>✅ 推荐：训练 8B/14B 模型 + 超长上下文（>60K）的理想配置</summary>

|   `max_model_len` | Qwen3-0.6B           | Qwen3-1.7B           | Qwen3-4B             | Qwen3-8B             | Qwen3-14B            |
|------------------:|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
|              4096 | vanilla              | vanilla              | vanilla              | vanilla              | Env                  |
|              8192 | vanilla              | vanilla              | vanilla              | vanilla              | Env + SP=2           |
|             12288 | vanilla              | vanilla              | vanilla              | Env                  | Env + SP=4           |
|             16384 | vanilla              | vanilla              | vanilla              | SP=2                 | Env + SP=4           |
|             20480 | vanilla              | vanilla              | vanilla              | SP=2                 | Env + SP=4           |
|             24576 | vanilla              | Env                  | SP=2                 | Env + SP=2           | Env + Offload        |
|             28672 | Env                  | SP=2                 | SP=2                 | Env + SP=2           | Env + Offload + SP=2 |
|             32768 | SP=2                 | SP=2                 | SP=2                 | SP=4                 | Env + Offload + SP=2 |
|             36864 | SP=2                 | SP=2                 | SP=2                 | SP=4                 | Env + Offload + SP=2 |
|             40960 | SP=2                 | SP=2                 | Env + SP=2           | SP=4                 | Env + Offload + SP=2 |
|             51200 | Env + SP=2           | Env + SP=2           | SP=4                 | Env + SP=4           | Env + Offload + SP=4 |
|             61440 | SP=4                 | SP=4                 | SP=4                 | Env + Offload + SP=4 | Env + Offload + SP=4 |
|             71680 | SP=4                 | SP=4                 | SP=4                 | Env + Offload + SP=4 | Env + Offload + SP=4 |
|             81920 | SP=4                 | SP=4                 | Env + SP=4           | Env + Offload + SP=4 | Env + Offload + SP=4 |
|             92160 | SP=4                 | Env + SP=4           | Env + Offload + SP=4 | Env + Offload + SP=4 | Env + Offload + SP=4 |
|            102400 | Env + SP=4           | Env + SP=4           | Env + Offload + SP=4 | Env + Offload + SP=4 | -                    |
|            112640 | Env + SP=4           | Env + Offload + SP=4 | -                    | -                    | -                    |
|            122880 | Env + Offload + SP=4 | -                    | -                    | -                    | -                    |
|            133120 | -                    | -                    | -                    | -                    | -                    |

</details>

---

### 6 张 GPU

<details><summary>✅ 对中小模型（≤4B）支持较好，但对 14B 模型在超长上下文下仍存在限制</summary>

|   `max_model_len` | Qwen3-0.6B   | Qwen3-1.7B   | Qwen3-4B             | Qwen3-8B             | Qwen3-14B            |
|------------------:|:-------------|:-------------|:---------------------|:---------------------|:---------------------|
|              4096 | vanilla      | vanilla      | vanilla              | vanilla              | vanilla              |
|              8192 | vanilla      | vanilla      | vanilla              | vanilla              | vanilla              |
|             12288 | vanilla      | vanilla      | vanilla              | vanilla              | SP=2                 |
|             16384 | vanilla      | vanilla      | vanilla              | Env                  | SP=2                 |
|             20480 | vanilla      | vanilla      | vanilla              | SP=2                 | Env + SP=2           |
|             24576 | vanilla      | Env          | Env                  | SP=2                 | Env + Offload        |
|             28672 | Env          | Env          | SP=2                 | SP=2                 | Env + Offload + SP=2 |
|             32768 | SP=2         | SP=2         | SP=2                 | Env + SP=2           | Env + Offload + SP=2 |
|             36864 | SP=2         | SP=2         | SP=2                 | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             40960 | SP=2         | SP=2         | SP=2                 | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             51200 | Env + SP=2   | Env + SP=2   | Env + Offload + SP=2 | Env + Offload + SP=2 | -                    |
|             61440 | Env + SP=2   | -            | -                    | -                    | -                    |
|             71680 | -            | -            | -                    | -                    | -                    |

</details>

---

## H20 96GB 显卡配置建议

H20 显存更大（96GB），但计算能力弱于 A100。

### 1 张 GPU

<details><summary>单卡可支持 4B 模型至 ~32K 上下文</summary>

|   `max_model_len` | Qwen3-0.6B    | Qwen3-1.7B    | Qwen3-4B      | Qwen3-8B      | Qwen3-14B     |
|------------------:|:--------------|:--------------|:--------------|:--------------|:--------------|
|              4096 | vanilla       | vanilla       | vanilla       | Env + Offload | Env + Offload |
|              8192 | vanilla       | vanilla       | vanilla       | Env + Offload | Env + Offload |
|             12288 | vanilla       | vanilla       | Env + Offload | Env + Offload | Env + Offload |
|             16384 | vanilla       | vanilla       | Env + Offload | Env + Offload | Env + Offload |
|             20480 | vanilla       | vanilla       | Env + Offload | Env + Offload | Env + Offload |
|             24576 | vanilla       | Env           | Env + Offload | Env + Offload | Env + Offload |
|             28672 | vanilla       | Env + Offload | Env + Offload | Env + Offload | Env + Offload |
|             32768 | Env           | Env + Offload | Env + Offload | -             | -             |
|             36864 | Env + Offload | Env + Offload | -             | -             | -             |
|             40960 | -             | -             | -             | -             | -             |

</details>

---

### 2 张 GPU

<details><summary>支持 14B 模型至 50K 上下文</summary>

|   `max_model_len` | Qwen3-0.6B   | Qwen3-1.7B   | Qwen3-4B             | Qwen3-8B             | Qwen3-14B            |
|------------------:|:-------------|:-------------|:---------------------|:---------------------|:---------------------|
|              4096 | vanilla      | vanilla      | vanilla              | vanilla              | Env + Offload        |
|              8192 | vanilla      | vanilla      | vanilla              | vanilla              | Env + Offload        |
|             12288 | vanilla      | vanilla      | vanilla              | SP=2                 | Env + Offload        |
|             16384 | vanilla      | vanilla      | vanilla              | SP=2                 | Env + Offload        |
|             20480 | vanilla      | vanilla      | Env                  | Env + Offload        | Env + Offload        |
|             24576 | vanilla      | vanilla      | SP=2                 | Env + Offload        | Env + Offload        |
|             28672 | vanilla      | Env          | SP=2                 | Env + Offload        | Env + Offload        |
|             32768 | Env          | SP=2         | SP=2                 | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             36864 | Env          | SP=2         | SP=2                 | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             40960 | SP=2         | SP=2         | Env + SP=2           | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             51200 | SP=2         | SP=2         | Env + Offload + SP=2 | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             61440 | Env + SP=2   | Env + SP=2   | Env + Offload + SP=2 | Env + Offload + SP=2 | -                    |
|             71680 | Env + SP=2   | -            | -                    | -                    | -                    |
|             81920 | -            | -            | -                    | -                    | -                    |

</details>

---

### 4 张 GPU

<details><summary>✅ 可支持 14B 模型训练至 100K 上下文</summary>

|   `max_model_len` | Qwen3-0.6B   | Qwen3-1.7B           | Qwen3-4B             | Qwen3-8B             | Qwen3-14B            |
|------------------:|:-------------|:---------------------|:---------------------|:---------------------|:---------------------|
|              4096 | vanilla      | vanilla              | vanilla              | vanilla              | vanilla              |
|              8192 | vanilla      | vanilla              | vanilla              | vanilla              | vanilla              |
|             12288 | vanilla      | vanilla              | vanilla              | vanilla              | SP=2                 |
|             16384 | vanilla      | vanilla              | vanilla              | vanilla              | SP=2                 |
|             20480 | vanilla      | vanilla              | vanilla              | Env                  | Env + SP=2           |
|             24576 | vanilla      | vanilla              | vanilla              | SP=2                 | SP=4                 |
|             28672 | vanilla      | vanilla              | Env                  | SP=2                 | SP=4                 |
|             32768 | Env          | Env                  | SP=2                 | Env + SP=2           | SP=4                 |
|             36864 | Env          | SP=2                 | SP=2                 | Env + SP=2           | Env + SP=4           |
|             40960 | SP=2         | SP=2                 | SP=2                 | SP=4                 | Env + Offload + SP=2 |
|             51200 | SP=2         | SP=2                 | Env + SP=2           | SP=4                 | Env + Offload + SP=2 |
|             61440 | SP=2         | Env + SP=2           | SP=4                 | Env + SP=4           | Env + Offload + SP=4 |
|             71680 | Env + SP=2   | SP=4                 | SP=4                 | Env + SP=4           | Env + Offload + SP=4 |
|             81920 | SP=4         | SP=4                 | SP=4                 | Env + Offload + SP=4 | Env + Offload + SP=4 |
|             92160 | SP=4         | SP=4                 | SP=4                 | Env + Offload + SP=4 | Env + Offload + SP=4 |
|            102400 | SP=4         | SP=4                 | Env + SP=4           | Env + Offload + SP=4 | Env + Offload + SP=4 |
|            112640 | SP=4         | Env + SP=4           | Env + Offload + SP=4 | Env + Offload + SP=4 | Env + Offload + SP=4 |
|            122880 | Env + SP=4   | Env + SP=4           | Env + Offload + SP=4 | Env + Offload + SP=4 | -                    |
|            133120 | Env + SP=4   | Env + Offload + SP=4 | Env + Offload + SP=4 | -                    | -                    |
|            143360 | Env + SP=4   | -                    | -                    | -                    | -                    |
|            153600 | -            | -                    | -                    | -                    | -                    |

</details>

---

### 6 张 GPU

<details><summary>对中小模型（≤4B）支持较好，但对 14B 模型在超长上下文下仍存在限制</summary>

|   `max_model_len` | Qwen3-0.6B   | Qwen3-1.7B           | Qwen3-4B             | Qwen3-8B             | Qwen3-14B            |
|------------------:|:-------------|:---------------------|:---------------------|:---------------------|:---------------------|
|              4096 | vanilla      | vanilla              | vanilla              | vanilla              | vanilla              |
|              8192 | vanilla      | vanilla              | vanilla              | vanilla              | vanilla              |
|             12288 | vanilla      | vanilla              | vanilla              | vanilla              | vanilla              |
|             16384 | vanilla      | vanilla              | vanilla              | vanilla              | Env                  |
|             20480 | vanilla      | vanilla              | vanilla              | vanilla              | SP=2                 |
|             24576 | vanilla      | vanilla              | vanilla              | SP=2                 | SP=2                 |
|             28672 | vanilla      | vanilla              | Env                  | SP=2                 | Env + SP=2           |
|             32768 | Env          | Env                  | SP=2                 | SP=2                 | Env + SP=2           |
|             36864 | Env          | SP=2                 | SP=2                 | SP=2                 | Env + Offload + SP=2 |
|             40960 | SP=2         | SP=2                 | SP=2                 | Env + SP=2           | Env + Offload + SP=2 |
|             51200 | SP=2         | SP=2                 | SP=2                 | Env + Offload + SP=2 | Env + Offload + SP=2 |
|             61440 | SP=2         | Env + SP=2           | Env + Offload + SP=2 | Env + Offload + SP=2 | -                    |
|             71680 | Env + SP=2   | Env + Offload + SP=2 | -                    | -                    | -                    |
|             81920 | -            | -                    | -                    | -                    | -                    |

</details>

---

## ✅ 最佳实践建议

1. **从最简配置开始**：优先尝试 `vanilla`，仅在遇到 OOM 时逐步启用高级功能。
2. **长上下文必用 YaRN**：超过 40,960 tokens 时，务必配置 `rope_scaling` 并合理设置 `factor`。
3. **OOM 处理顺序**：
   - 第一步：设置 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - 第二步：增加 **Sequence Parallelism（SP）**
   - 第三步：启用 **FSDP v2 + CPU Offload**
4. **SP 并行度选择**：建议设为 **GPU 数量与注意力头数的公因数**（如 2、4）。
5. **多卡优于单卡**：即使显存足够，多卡也能通过并行提升训练效率与稳定性。
