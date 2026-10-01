# GiGPO on ALFWorld

End-to-end example for [GiGPO: Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978) on ALFWorld with step-wise rollouts.

Config: [`gigpo.yaml`](gigpo.yaml).

## GiGPO in this example

| Paper concept | Trinity wiring in `gigpo.yaml` |
|---------------|--------------------------------|
| N trajectories per task | `algorithm.repeat_times: 8` |
| Episode advantage $A^E(\tau)$ | `advantage_fn: gigpo` (GRPO-style over $R(\tau)=\sum_t r_t$) |
| Step advantage $A^S(a \mid \tilde s)$ | Anchor groups via `experience.info["env_state_hash"]` (set by `step_wise_alfworld_workflow`) |
| Combined $A = A^E + \omega A^S$ | `advantage_fn_args.omega: 1.0` |
| Discounted step return | `advantage_fn_args.gamma: 1.0` |
| $F_{\text{norm}}=1$ (agent default) | `advantage_fn_args.fnorm: none` |
| Clipped token-level policy loss | `policy_loss_fn: ppo` + `loss_agg_mode: token-mean` |

`StepWiseAlfworldWorkflow` emits per-step metadata:

- `info["env_state_hash"]` — SHA256 of the pre-action formatted observation (anchor state $\tilde s$)
- `info["step_reward"]` — immediate environment reward at that step

Terminal episode reward is still copied to `experience.reward` on all steps (compatible with GRPO-style workflows).

## Custom workflows

For non-ALFWorld environments, each step experience must set:

1. `eid.run` and `eid.step` (see [step-wise tutorial](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/example_step_wise.html))
2. `info["env_state_hash"]` — deterministic key for equivalent environment states
3. `info["step_reward"]` — immediate scalar $r_t$ (recommended; avoids sparse-reward fallback)

## Run

From the tutorial repository root, set the model and ALFWorld paths in `.env`
(copy `.env.example` first), then use the same locked environment and launcher as
the full-parameter baseline:

```bash
CONFIG=examples/gigpo_alfworld/gigpo.yaml bash run.sh
```

The entrypoint supports Linux x86_64, Python 3.12 and CUDA 13-compatible NVIDIA
drivers; the reference machine has 8 A100 80GB GPUs. It installs the locked
FlashAttention2 wheel and uses the virtual environment's Ray and Trinity commands.
The default monitor is TensorBoard (no W&B key required), Qwen3 thinking is
disabled, and the dynamic-batch token budget covers the full model sequence limit.
This configuration has local consistency checks; the independent two-step GPU
verification used the multi-step GRPO baseline, not GiGPO.

Environment variables:

- `TRINITY_MODEL_PATH` — base model
- `ALFWORLD_DATA` — local ALFWorld game data
- `TRINITY_CHECKPOINT_ROOT_DIR` — optional checkpoint root (default `./checkpoints`)
- `TRINITY_RAY_PORT` — head port for a new local cluster (default 16379;
  also reserves the preceding three ports; workers use additional dynamic ports)
- `RAY_ADDRESS` — optional explicit `host:port` of an existing compatible cluster;
  when set, the launcher checks and reuses it instead of starting another one

The launcher does not stop unrelated services. A port conflict or Ray version
mismatch fails with its original error. The Ray service remains available after
training. View this example's TensorBoard events with:

```bash
.venv/bin/tensorboard --logdir checkpoints/ALFWORLD/GiGPO_Alfworld/monitor/tensorboard --port 6006
```

## Metrics to watch

- `gigpo/anchor_group_hit_ratio` — fraction of anchor states with multiple visits (step-level signal active)
- `gigpo/mean_abs_A_E` vs `gigpo/mean_abs_A_S` — balance of episode vs step credit
- `gigpo/episode_reward_mean` — task-group return baseline
- `pg_clipfrac`, `ppo_kl` — policy update health

## Ablations

- `advantage_fn_args.omega: 0` — episode-only (GRPO-like at trajectory level)
- `advantage_fn_args.fnorm: std` — GRPO-style normalization within groups
- Compare against [`examples/grpo_alfworld_general_multi_step/alfworld.yaml`](../grpo_alfworld_general_multi_step/alfworld.yaml) (`algorithm_type: multi_step_grpo`)
