# GPU Configuration Guide

This document provides recommended training configurations for Qwen3 series models on **NVIDIA A100 80GB** and **H20 96GB** GPUs.
Based on model size (0.6B ~ 14B) and context length (`model.max_model_len`), we present feasible Trainer module setups across varying numbers of GPUs.

> ⚠️ **Note**:
> Due to the sparation design of rollout and training with Trinity. The following description of the number of GPUs refers to the number available for `Trainer`, not the total number of GPUs used by Trinity.

> 💡 **Terminology**
>
> - **vanilla**: No special configuration required; default settings suffice.
> - **Env**: Set the following environment variable **before launching training (before starting Ray)**:
>   ```bash
>   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
>   ```
> - **Offload**: Enable **FSDP v2 + CPU Offload** to reduce GPU memory usage.
> - **SP=N**: Use **Sequence Parallelism** with parallelism degree N (typically N ≤ number of GPUs).
> - **Combined entries (e.g., `Env + SP=2`)**: All listed conditions must be satisfied simultaneously.
> - **“-”**: The combination of current hardware and configuration **cannot support training** for this model under the given sequence length.

---

## Long Context Support

Qwen3 series models natively support a maximum context length of **40,960 tokens**.
For training beyond this length (e.g., 51,200, 81,920 tokens), we use **YaRN RoPE extension**. The relevant configuration is as follows:

```yaml
model:
  model_path: ${oc.env:MODEL_PATH,Qwen/Qwen3-0.6B}
  max_prompt_tokens: 2048
  max_model_len: ${oc.env:MAX_MODEL_LEN,4096}
  rope_scaling:
    rope_type: yarn
    factor: ${oc.decode:${oc.env:FACTOR}}  # Recommended value = MAX_MODEL_LEN / 40960
    original_max_position_embeddings: 40960
```

> ✅ When using YaRN, ensure `factor` is set reasonably to avoid numerical instability.

---

## 💡 Relationship Between GPU Memory Usage and `max_token_len_per_gpu`

Trinity Trainer enables dynamic batch sizing by default (`trainer.use_dynamic_bsz=True`). With a fixed model, actual GPU memory consumption is primarily determined by the following two parameters:

- `trainer.trainer_config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu`
- `trainer.trainer_config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`

If these parameters are not manually configured, Trinity automatically uses the following default value:
```python
trainer.max_token_len_per_gpu = ceil(2 * model.max_model_len / trainer.ulysses_sequence_parallel_size)
```

📌 **This implies that**:
- The longer the context length, the more tokens each GPU must process, resulting in higher memory pressure.
- To support **longer context lengths**, you can manually adjust the parameters above (though this may impact training efficiency).

> All experimental results presented in this guide are based on the aforementioned default settings. For extreme optimization, please fine-tune these parameters according to your specific requirements.

---

## A100 80GB GPU Configuration Recommendations

> ⚠️ **Single-GPU Limitation**: Training models ≥4B or with context lengths >20K on a single A100 GPU places extreme pressure on VRAM. **Multi-GPU setups are strongly recommended**.

### 1 GPU

<details><summary>Click to view detailed configurations</summary>

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

### 2 GPUs

<details><summary>✅ Recommended: 2 GPUs significantly improve long-context training capability for 4B~14B models. Enable SP=2 when using longer contexts.</summary>

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

### 4 GPUs

<details><summary>✅ Recommended: Ideal setup for training 8B/14B models with ultra-long contexts (>60K)</summary>

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

### 6 GPUs

<details><summary>✅ Good support for small-to-medium models (≤4B), but still limited for 14B models with ultra-long contexts</summary>

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

## H20 96GB GPU Configuration Recommendations

The H20 has larger VRAM (96GB) but lower compute performance compared to the A100.

### 1 GPU

<details><summary>Single GPU supports 4B models up to ~32K context length</summary>

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

### 2 GPUs

<details><summary>Supports 14B models up to 50K context length</summary>

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

### 4 GPUs

<details><summary>✅ Supports training 14B models up to 100K context length</summary>

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

### 6 GPUs

<details><summary>Good support for small-to-medium models (≤4B), but still limited for 14B models with ultra-long contexts</summary>

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

## ✅ Best Practices

1. **Start with the simplest configuration**: Try `vanilla` first, and incrementally enable advanced features only when encountering OOM errors.
2. **Always use YaRN for long contexts**: For contexts exceeding 40,960 tokens, configure `rope_scaling` and set `factor` appropriately.
3. **OOM troubleshooting sequence**:
   - Step 1: Set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - Step 2: Increase **Sequence Parallelism (SP)**
   - Step 3: Enable **FSDP v2 + CPU Offload**
4. **Choosing SP parallelism degree**: Prefer values that are **common divisors of both GPU count and attention head count** (e.g., 2, 4).
5. **Prefer multi-GPU over single-GPU**: Even when VRAM appears sufficient, multi-GPU setups improve training efficiency and stability through parallelization.
