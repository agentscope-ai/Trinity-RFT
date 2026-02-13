# Offline DPO and SFT

This example describes DPO and SFT based on the Qwen2.5-1.5B-Instruct model.

## Step 1: Model and Data Preparation

### Model Preparation

Download the Qwen2.5-1.5B-Instruct model to the local directory `$MODEL_PATH/Qwen2.5-1.5B-Instruct`:

```shell
# Using Modelscope
modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir $MODEL_PATH/Qwen2.5-1.5B-Instruct

# Using Huggingface
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $MODEL_PATH/Qwen2.5-1.5B-Instruct
```

More details of model downloading are referred to [ModelScope](https://modelscope.cn/docs/models/download) or [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli).

### Data Preparation

For DPO, we download the [Human-like-DPO-dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) to the local directory `$DATASET_PATH/human_like_dpo_dataset`:

```shell
# Using Modelscope
modelscope download --dataset HumanLLMs/Human-Like-DPO-Dataset --local_dir $DATASET_PATH/human_like_dpo_dataset

# Using Huggingface
huggingface-cli download HumanLLMs/Human-Like-DPO-Dataset --repo-type dataset --local-dir $DATASET_PATH/human_like_dpo_dataset
```

Below are some data samples in JSONL format:
```json
{"prompt":"Oh, I just saw the best meme - have you seen it?","chosen":"\ud83d\ude02 Ah, no I haven't! I'm dying to know, what's the meme about? Is it a funny cat or a ridiculous situation? Spill the beans! \ud83e\udd23","rejected":"I'm an artificial intelligence language model, I don't have personal experiences or opinions. However, I can provide you with information on highly-rated and critically acclaimed films, as well as recommendations based on specific genres or themes. Would you like me to suggest some notable movies or discuss a particular genre of interest?"}
{"prompt":"Have you tried any new hobbies or activities recently?","chosen":"You know, I've been meaning to try my hand at gardening, but I haven't gotten around to it yet. I've heard it's super relaxing and a great way to get some fresh air. Maybe I'll finally get around to buying some seeds and pots this weekend. What about you? Have you taken up anything new and exciting lately? \ud83c\udf31\ud83d\udc40","rejected":"I'm an artificial intelligence language model, and as such, I don't have personal experiences or engage in physical activities such as dining or cooking. My purpose is to provide information, answer questions, and assist with tasks to the best of my abilities, while maintaining a professional and impartial demeanor. If you have any specific questions or topics related to restaurants or recipes, I'd be happy to provide information or guidance."}
```

More details of dataset downloading are referred to [ModelScope](https://modelscope.cn/docs/datasets/download) or [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space).

Note that the dataset has the keys `prompt`, `chosen` and `rejected`. If you use different datasets, pass the proper keys to the config.

For SFT, we download the `open-r1/Mixture-of-Thoughts` dataset to the local directory `$DATASET_PATH/Mixture-of-Thoughts`, which contains message-based data, we list a simplified sample here.

```json
{"messages": [{"content": "You will be given a competitive programming problem...","role": "user"},{"content": "<think>\n...</think>\n...This approach efficiently combines hashing and dynamic programming to solve the problem within the given constraints.","role": "assistant"}], "num_tokens": 22185, "source": "open-r1/codeforces-cots"}
```

## Step 2: Setup Configuration

### Configuration for DPO

We use the configurations in [`dpo.yaml`](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/dpo_humanlike/dpo.yaml) for this experiment. Some important setups are listed in the following:

We run the experiment in a train mode, as there is no Explorer. To enable this mode, we config `mode` to `train` and pass the data path to the trainer.

```yaml
project: <project_name>
name: <experiment_name>
mode: train
algorithm:
  algorithm_type: dpo
  kl_loss_fn: k1
  kl_loss_fn_args:
    kl_coef: 0.1  # value of beta in DPO
checkpoint_root_dir: ${oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}
model:
  model_path: ${oc.env:TRINITY_MODEL_PATH,Qwen/Qwen2.5-1.5B-Instruct}
  max_response_tokens: 1024
  max_model_len: 1536
cluster:
  node_num: 1
  gpu_per_node: 8
buffer:
  total_epochs: 2
  train_batch_size: 64
  trainer_input:
    experience_buffer:
      name: human_like_dpo
      storage_type: file
      path: $DATASET_PATH/human_like_dpo_dataset
      format:
        prompt_type: plaintext
        prompt_key: prompt
        chosen_key: chosen
        rejected_key: rejected
trainer:
  save_interval: 30
  trainer_config:
    ... # omitted here for simplicity
```

`buffer.trainer_input.experience_buffer` specifies the dataset to be used for training, including its name, storage type, path, and format.

- The name `human_like_dpo` is a unique identifier for this dataset configuration, you can use other names as long as they are unique within the project.
- The storage type `file` means the dataset is stored in a file on the local filesystem and the `path` is pointed to the local directory where the dataset is stored. Note that the `file` storage type also supports using huggingface datasets path like `HumanLLMs/Human-Like-DPO-Dataset`.
- The format specifies how the data is structured within the dataset. In this case, it is defined as follows:

  - `prompt_type: plaintext` indicates that the prompts are in plain text format.
  - `prompt_key: prompt` specifies the key in the dataset that contains the user prompts.
  - `chosen_key: chosen` specifies the key in the dataset that contains the chosen responses.
  - `rejected_key: rejected` specifies the key in the dataset that contains the rejected responses.

For more configuration options, please refer to the {ref}`Configuration Guide <Configuration Guide>`.

### Configuration for SFT

We set the `algorithm_type` as `sft` to run SFT process and then modify the config file [`examples/sft_mot/sft.yaml`](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/sft_mot/sft.yaml) with the following changes:

```yaml
project: <project_name>
name: <experiment_name>
mode: train
algorithm:
  algorithm_type: sft
checkpoint_root_dir: ${oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}
model:
  model_path: ${oc.env:TRINITY_MODEL_PATH,Qwen/Qwen2.5-1.5B-Instruct}
  max_response_tokens: 10240
  max_model_len: 10752
cluster:
  node_num: 1
  gpu_per_node: 2
buffer:
  total_epochs: 5
  train_batch_size: 64
  trainer_input:
    experience_buffer:
      name: <sft_dataset_name>
      storage_type: file
      path: $DATASET_PATH/Mixture-of-Thoughts
      split: train
      format:
        prompt_type: messages
        messages_key: messages
trainer:
  save_interval: 50
  trainer_config:
    ... # omitted here for simplicity
```

Here we set `buffer.trainer_input.experience_buffer.format.prompt_type` to `messages` because the source data is in message format. We also set `buffer.trainer_input.experience_buffer.format.messages_key` to `messages` to specify the key in the dataset that contains the messages.

## Step 3: Run the Experiment

Run DPO process with the following command:

```shell
trinity run --config examples/dpo_humanlike/dpo.yaml
```

or, for SFT:

```shell
trinity run --config examples/sft_mot/sft.yaml
```
