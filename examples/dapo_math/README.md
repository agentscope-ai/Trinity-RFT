# DAPO on DAPO-Math-17k

End-to-end example for [DAPO](https://arxiv.org/abs/2503.14476) (Decoupled Clip and Dynamic sAmpling Policy Optimization) on [DAPO-Math-17k-Processed](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed).

Config: [`dapo.yaml`](dapo.yaml). Implementation details: [docs/dapo_trinity_implementation_spec.md](../../docs/dapo_trinity_implementation_spec.md).

## DAPO techniques in this example

| Paper technique | Trinity wiring in `dapo.yaml` |
|-----------------|-------------------------------|
| Group-relative advantage (GRPO), no KL | `algorithm_type: dapo` → `advantage_fn: grpo`, `kl_penalty_fn: none` |
| Clip-Higher | `policy_loss_fn_args.clip_range_low/high: 0.2 / 0.28` |
| Token-level policy loss | `policy_loss_fn_args.loss_agg_mode: token-mean` |
| Dynamic sampling (accuracy-only) | `dapo_dynamic_sampling` in experience pipeline |
| Overlong filter (no loss on truncated) | `mask_response_truncated` + workflow mask |
| Soft overlong reward | `math_dapo_reward` + `reward_fn_args` (`max_response_length`, `cache_length`) |

Dynamic sampling keeps groups only when `0 < #correct < G`, using `metrics["accuracy"]` (±1 rule reward), not length-shaped total reward. Total reward for GRPO is still `accuracy + format_score`. Do not set `advantage_fn_args.std_threshold` when using the pipeline filter.

## Run

Set model and checkpoint paths, then:

```bash
trinity run --config examples/dapo_math/dapo.yaml
```

Environment variables used by the config:

- `TRINITY_MODEL_PATH` — base model (paper uses Qwen2.5-32B scale)
- `TRINITY_CHECKPOINT_ROOT_DIR` — optional checkpoint root (default `./checkpoints`)

## Metrics to watch

- `experience_pipeline/filtered_count` — dynamic sampling drops
- `group_advantages/reward_std/mean` — group reward spread
- `pg_clipfrac`, `ppo_kl` — policy update health
- Policy entropy (trainer logs) — should not collapse early with Clip-Higher

## Eval

AIME 2024 eval is configured with `repeat_times: 32`, `temperature: 1.0`, `top_p: 0.7` (paper-style avg@32).

## Not in this example (see spec divergence register)

- Resample-until-full prompt batch (paper verl loop); Trinity filters after one explorer pass
