# Entropy dynamics of RL training

This example shows the two algorithms **Clip_B** and **Clip_V** from the work [On the Entropy Dynamics in Reinforcement Fine-Tuning of Large Language Models](https://arxiv.org/pdf/2602.03392).

## Data Preparation

We utilize the [DAPO-Math-17k](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed) dataset as our training set. We exclude 500 questions from the training set to form the validation set (denoted by dapo-validation-500).
The training set is filtered out samples from the training set with excessively high (≥ 15/16) or low (≤ 1/16) pass rates, as evaluated by Qwen2.5-7B-Instruct.

## Clip_B Experiment

1. Apply the patch to keep entropy information in the trainer batch:

```bash
cd /path/to/Trinity-RFT
git apply examples/entropy/clipb_trainer.patch
```

2. Update the dataset paths in the config file [`clipb.yaml`](clipb.yaml) to point to your local data.

3. Run the experiment:

```bash
trinity run examples/entropy/clipb.yaml
```

## Clip_V Implementation

Coming soon.
