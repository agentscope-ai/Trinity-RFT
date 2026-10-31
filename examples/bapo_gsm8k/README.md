# BAPO on GSM8K dataset

This example shows the usage of [BAPO](https://arxiv.org/abs/2510.18927) (Balanced Policy Optimization with Adaptive Clipping) on the GSM8K dataset.

BAPO keeps the PPO-clip surrogate but chooses the clipping bounds for every update. Starting from `[lower_bound_start, upper_bound_start]`, it first raises the upper bound by `upper_bound_step` (up to `upper_bound_max`) and then the lower bound by `lower_bound_step` (up to `lower_bound_max`), until tokens with positive advantages contribute at least `target_positive_ratio` of the policy gradient. The bounds are explicit importance-ratio values, not the `1 ± clip_range` epsilons used by PPO. The default values follow Sec. 5.1 of the paper.

The selected bounds and the resulting positive contribution ratio are logged as `actor/bapo_clip_low`, `actor/bapo_clip_high` and `actor/bapo_positive_ratio`.

The config file is located in [`gsm8k.yaml`](gsm8k.yaml).
