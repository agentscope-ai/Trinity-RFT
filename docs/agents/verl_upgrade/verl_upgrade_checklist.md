# Pre-Upgrade Checklist for veRL

This checklist is for quick verification before the next veRL upgrade in Trinity.

## 1. Confirm Upgrade Scope

1. Confirm the target veRL version.
2. Confirm the current Trinity baseline version.
3. Confirm the upstream snapshots for comparison have been generated under `trinity/trainer/verl/build/<version>/`.
4. Confirm this upgrade still focuses on the same 7 core migration files:
   - `fsdp_workers.py`
   - `dp_actor.py`
   - `fsdp_checkpoint_manager.py`
   - `megatron_workers.py`
   - `megatron_actor.py`
   - `megatron_checkpoint_manager.py`
   - `verl_trainer.py` (corresponds to upstream `ray_trainer.py`)

## 2. Prepare Three-Way Comparison

1. For each file, compare all three sources together:
   - Current Trinity file
   - `build/<old_version>/...`
   - `build/<new_version>/...`
2. Do not do whole-file overwrite.
3. Prioritize recording two categories of diffs:
   - What Trinity added on top of the old-version baseline
   - What upstream changed from old version to new version

## 3. Verify Repository Responsibility Boundaries

Before the next upgrade, verify these boundaries are still valid:

1. Reward computation is not executed in Trinity `verl_trainer.py`.
2. Rollout is not executed in Trinity veRL trainer main loop.
3. Trainer-side validation is currently not implemented.
4. Trinity does not run upstream `RayPPOTrainer.fit()` directly. It follows the path defined in `trinity/trainer/trainer.py`: `prepare()`, `train_step()`, `save_checkpoint()`, `save_state_dict()`, and `upload_state_dict()`.

If any boundary above changes, re-evaluate all following steps in this checklist.

## 4. Upstream Logic That Should Not Be Accidentally Reintroduced

Unless Trinity training responsibilities change, do not migrate these back by default:

1. Full reward pipeline inside `fit()`.
2. Validation main flow.
3. Reward loop / async rollout manager.
4. `CheckpointEngineManager` orchestration logic that is only used by the upstream trainer main loop.

## 5. Must-Check Configuration Wiring

Before upgrade, verify whether these config items still need end-to-end wiring into implementation:

1. `trust_remote_code`
2. `use_prefix_grouper`
3. `calculate_sum_pi_squared`
4. `sum_pi_squared_checkpointing`
5. Compatibility reads for `lora.rank` and `lora_rank`
6. `rollout_correction`
7. Compatibility structure for `reward.reward_model` and `reward_model`

## 6. File-Level Priority Order

Recommended processing order:

1. `dp_actor.py`
2. `fsdp_workers.py`
3. `megatron_actor.py`
4. `megatron_workers.py`
5. `fsdp_checkpoint_manager.py`
6. `megatron_checkpoint_manager.py`
7. `verl_trainer.py`
8. `verl_config.py`

Reason: the first four files define data fields and config wiring; the next three depend on these contracts being stable.

## 7. Convergence Checks Required for Every File

For every migration file, ask:

1. Is this subclass implementation only a copy of parent-class code?
2. If it is fully identical to upstream parent implementation, can we delete the override directly?
3. If this is only a historical workaround, has upstream already absorbed it?
4. If this is a true Trinity-specific responsibility, has the reason to keep it been documented?

## 8. Trinity Customizations Confirmed as Non-Removable

1. Algorithm integration and loss composition logic in `dp_actor.py` and `megatron_actor.py`.
2. `CheckpointMonitor` / `Synchronizer` collaboration logic in `fsdp_checkpoint_manager.py` and `megatron_checkpoint_manager.py`.
3. `CheckpointMonitor`, Trinity custom `train_step()`, and state sync path in `verl_trainer.py`.
4. Trinity's independent experience pipeline and trainer scheduling relationship.

## 9. Known Migration Sensitive Points

1. `use_prefix_grouper` is an end-to-end chain from config to monkey patch to actor/ref worker.
2. `sum_pi_squared` must be passed from actor output all the way to the advantage consumer.
3. Megatron LoRA reference logprob follows the actor/no-adapter path, not the regular ref worker path.
4. To collect MFU in multimodal training, `images_seqlens` must be added to `batch.meta_info` in trainer.
5. Checkpoint manager cannot be replaced by whole-file upstream overwrite, otherwise Trinity async threads and monitoring logic are lost.

## 10. Local Checks After Upgrade

1. Run Problems check for all migrated files.
2. Run `python -m py_compile` uniformly for all migrated files.
3. Verify new config items are closed-loop across dataclass, defaults, and loading path.
4. Verify actor output fields match worker/trainer consumer fields.
5. Verify function signatures for checkpoint save and restore are consistent.

## 11. Minimal Remote GPU Regression

After local checks pass, run at least:

1. FSDP single-step training.
2. Megatron single-step training.
3. Recompute path for old logprob / ref logprob.
4. Megatron reference logprob under LoRA.
5. Checkpoint save and restore.
6. Minimal regression for `use_prefix_grouper`.
7. Minimal regression for `calculate_sum_pi_squared`.

## 12. Final Confirmation

Before submitting the upgrade, reconfirm:

1. Reward, rollout, and validation logic were not accidentally moved back into Trinity trainer.
2. Duplicate subclass implementations that are already identical to upstream were not kept.
3. Features previously trimmed by Trinity were not restored only for version alignment.
4. Documentation has been updated with newly added repository constraints and reasons for retained customizations.
