# GRPO with VLM

This example shows the usage of GRPO with Qwen2.5-VL-3B-Instruct on the [geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) dataset.

> [!NOTE]
> This feature is experimental and will be subject to change in future releases.

The specific requirements are:

```yaml
vllm>=0.10.2  # Qwen3 VL requires vllm>=0.11.0; it is recommended to use version >= 0.13.0
transformers>=4.54.0
qwen_vl_utils
```

For other detailed information, please refer to the [documentation](../../docs/sphinx_doc/source/tutorial/example_reasoning_basic.md).

The config file is located in [`vlm.yaml`](vlm.yaml), and the curve is shown below.

![vlm](../../docs/sphinx_doc/assets/geometry3k_qwen25_vl_3b_reward.png)

## Supported Model Architectures

The following vision-language model series are currently supported:

1. Qwen2.5-VL series
2. Qwen3-VL series
3. Kimi-VL-A3B-Thinking series
