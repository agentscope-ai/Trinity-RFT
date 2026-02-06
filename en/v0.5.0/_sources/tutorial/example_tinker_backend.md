# Tinker Backend

```{note}
This example demonstrates how to use Trinity-RFT with the [Tinker](https://thinkingmachines.ai/tinker/) backend, which enables model training on devices **without GPUs**.
```

## Setup Instructions

### 1. API Key Configuration

Before starting Ray, you must set the `TRINITY_API_KEY` environment variable to your Tinker API key to enable proper access to Tinker's API:

```bash
export TRINITY_API_KEY=your_tinker_api_key
ray start --head
```

### 2. Configuration File

Configure the Tinker backend in your YAML configuration file by setting the `model.tinker` parameters as shown below:

```yaml
model:
  tinker:
    enable: true
    rank: 32
    seed: null
    train_mlp: true
    train_attn: true
    train_unembed: true
```

#### Explanation of Configuration Parameters

- **`tinker`**: Tinker-specific configuration section. **Important**: When Tinker is enabled, any LoRA configuration settings (`model.lora_configs`) will be ignored.
  - **`enable`**: Whether to activate the Tinker backend. Default: `false`
  - **`rank`**: The LoRA rank that controls the size of the adaptation matrices. Default: `32`
  - **`seed`**: Random seed for reproducible Tinker operations. If not specified (`null`), no specific seed is set
  - **`train_mlp`**: Whether to train the MLP (feed-forward) layers. Default: `true`
  - **`train_attn`**: Whether to train the attention layers. Default: `true`
  - **`train_unembed`**: Whether to train the unembedding (output) layer. Default: `true`


## Usage

Once configured, Trinity-RFT works with the Tinker backend just like it does with the standard veRL backend. Start training with:

```bash
trinity run --config tinker.yaml  # Replace with your actual config file path
```

### Important Limitations of the Tinker Backend

1. **Entropy loss** is not consistent compared to veRL backends.
2. **Algorithms requiring `compute_advantage_in_trainer=true` are NOT supported currently**, including:
    - PPO (`algorithm.algorithm_type=ppo`)
    - Reinforce++ (`algorithm.algorithm_type=reinforceplusplus`)
    - RLOO (`algorithm.algorithm_type=rloo`)
    - On-policy distillation (`algorithm.algorithm_type=on_policy_distill`)

    Algorithms like `grpo`, `opmd`, `sft` are supported and we will support more algorithms in the future.

3. **Multiple stages training** is not supported currently, we will add support for this in the future.

> 💡 A complete example configuration file is available at [`tinker.yaml`](https://github.com/agentscope-ai/Trinity-RFT/blob/main/examples/tinker/tinker.yaml).


## Results on the Llama-3.2-3B Model

We trained the **Llama-3.2-3B** model on the **GSM8K** dataset using both the **Tinker** and **veRL** backends. Below are the full configuration files used in our experiments.


<details><summary>Click to expand: Tinker Backend Configuration</summary>

```yaml
mode: both
project: Trinity-RFT-gsm8k
group: alignment-tinker
name: tinker-llama3.2-3B-off1
checkpoint_root_dir: ${oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}
algorithm:
  algorithm_type: grpo
  repeat_times: 8
  kl_loss_fn_args:
    kl_coef: 0.0
  optimizer:
    lr: 1.0e-05
model:
  model_path: meta-llama/Llama-3.2-3B
  max_prompt_tokens: 1024
  max_response_tokens: 2048
  custom_chat_template: "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
  tinker:
    enable: true
cluster:
  node_num: 1
  gpu_per_node: 8
buffer:
  batch_size: 96
  total_epochs: 1
  explorer_input:
    taskset:
      name: taskset
      storage_type: file
      path: openai/gsm8k
      split: train
      subset_name: main
      format:
        prompt_key: question
        response_key: answer
    default_workflow_type: math_workflow
  trainer_input:
    experience_buffer:
      name: experience_buffer
      storage_type: queue
explorer:
  runner_per_model: 16
  rollout_model:
    engine_num: 4
    seed: 42
trainer:
  save_interval: 100
  grad_clip: 1.0
monitor:
  monitor_type: wandb
synchronizer:
  sync_method: checkpoint
  sync_style: fixed
  sync_interval: 1
  sync_offset: 1
  sync_timeout: 1200
```

</details>


<details><summary>Click to expand: veRL Backend Configuration (LoRA)</summary>

```yaml
mode: both
project: Trinity-RFT-gsm8k
group: alignment-tinker
name: verl-llama3.2-3B-lora-off1
checkpoint_root_dir: ${oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}
algorithm:
  algorithm_type: grpo
  repeat_times: 8
  kl_loss_fn_args:
    kl_coef: 0.0
  optimizer:
    lr: 1.0e-05
data_processor: {}
model:
  model_path: meta-llama/Llama-3.2-3B
  max_prompt_tokens: 1024
  max_response_tokens: 2048
  custom_chat_template: "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
  lora_configs:
  - name: lora
    lora_rank: 32
    lora_alpha: 32
cluster:
  node_num: 1
  gpu_per_node: 8
buffer:
  batch_size: 96
  total_epochs: 1
  explorer_input:
    taskset:
      name: taskset
      storage_type: file
      path: openai/gsm8k
      split: train
      subset_name: main
      format:
        prompt_key: question
        response_key: answer
    default_workflow_type: math_workflow
  trainer_input:
    experience_buffer:
      name: experience_buffer
      storage_type: queue
explorer:
  runner_per_model: 16
  rollout_model:
    engine_num: 4
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    dtype: bfloat16
    seed: 42
trainer:
  trainer_type: verl
  save_interval: 100
  grad_clip: 1.0
monitor:
  monitor_type: wandb
synchronizer:
  sync_method: checkpoint
  sync_style: fixed
  sync_interval: 1
  sync_offset: 1
  sync_timeout: 1200
```

</details>

### Observations

Since Llama-3.2-3B is a base (non-instruct-tuned) model, it has limited ability to follow formatting instructions. Additionally, we trained for only **one epoch**. As a result, both backends achieved final rewards just slightly above 0.1. Nonetheless, the training curves show a clear upward trend in reward, indicating successful learning. The results are visualized below:

![Training Rewards on GSM8K](../../assets/tinker-gsm8k.png)
