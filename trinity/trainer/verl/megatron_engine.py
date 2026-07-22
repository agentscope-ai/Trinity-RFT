# -*- coding: utf-8 -*-
"""Megatron-specific checkpoint and weight sync helpers for Trinity.

These helper functions are called by `TrinityActorRolloutRefWorker` to
perform Megatron-specific operations that veRL 0.8's engine does not
provide natively:

- megatron_save_state_dict:    Save state dict for checkpoint sync
- megatron_upload_state_dict:  Upload state dict to Synchronizer (memory sync)
- megatron_sync_weight_nccl:   Broadcast params via NCCL

All functions receive the `engine` object (a McoreEngine instance from
`verl.workers.engine.megatron`) which exposes:
  - engine.module: the Megatron model
  - engine.get_per_tensor_param(): generator of (name, tensor) pairs
  - engine.save_checkpoint() / engine.load_checkpoint()
"""
import ray
import torch
from verl.utils.memory_utils import aggressive_empty_cache

from trinity.trainer.verl.checkpoint import CheckpointCoordinator
from trinity.utils.device import get_device_name


def megatron_save_state_dict(
    engine,
    local_path: str,
    global_step: int,
    coordinator: CheckpointCoordinator,
    logger,
):
    """Save Megatron actor model state dict for checkpoint-based weight sync.

    Delegates to the engine's built-in ``save_checkpoint`` for proper
    distributed checkpoint handling, but temporarily restricts the
    checkpoint manager's ``checkpoint_save_contents`` to ``["model"]`` so
    that only the actor model parameters are written to disk -- optimizer
    and extra (rng / scheduler) states are skipped, since this path is
    used only for weight-sync to the rollout side, not for training
    resume. The save is wrapped with CheckpointMonitor notifications via
    the coordinator.

    Note: Megatron's save_checkpoint involves distributed barriers internally,
    so it runs synchronously. The coordinator's ``save_sync`` is used to add
    Monitor notifications without background threading.

    Args:
        engine: The McoreEngine instance (engine.actor.engine).
        local_path: Local directory path to save the state dict.
        global_step: Current training step.
        coordinator: CheckpointCoordinator for Monitor integration.
        logger: Logger instance from the calling worker.
    """
    if local_path is None:
        return

    # Temporarily restrict checkpoint contents to model-only. The veRL
    # Megatron engine attribute is ``checkpoint_mananager`` (sic, upstream typo).
    ckpt_mgr = getattr(engine, "checkpoint_mananager", None) or getattr(
        engine, "checkpoint_manager", None
    )
    original_save_contents = None
    if ckpt_mgr is not None and hasattr(ckpt_mgr, "checkpoint_save_contents"):
        original_save_contents = list(ckpt_mgr.checkpoint_save_contents)
        # Preserve hf_model export if it was originally enabled, but drop
        # optimizer / extra so we don't pay their cost on every weight sync.
        new_contents = ["model"]
        if "hf_model" in original_save_contents:
            new_contents.append("hf_model")
        ckpt_mgr.checkpoint_save_contents = new_contents

    def _do_save():
        engine.save_checkpoint(local_path=local_path, global_step=global_step)

    try:
        if torch.distributed.get_rank() == 0:
            coordinator.save_sync(_do_save, global_step, is_state_dict=True)
        else:
            _do_save()
    finally:
        if original_save_contents is not None:
            ckpt_mgr.checkpoint_save_contents = original_save_contents

    torch.distributed.barrier()
    logger.info(
        f"[Megatron] actor state_dict saved: path={local_path}, step={global_step} "
        f"(optimizer/extra skipped)"
    )


def megatron_upload_state_dict(engine, synchronizer, global_step: int, logger):
    """Upload Megatron model state dict to Synchronizer for memory-based weight sync.

    Iterates over per-tensor parameters and collects them on rank 0,
    then sends the full state dict to the Synchronizer actor.

    Args:
        engine: The McoreEngine instance (engine.actor.engine).
        synchronizer: The Synchronizer Ray actor handle.
        global_step: Current training step (used as version key).
    """
    if global_step == 0:
        return

    aggressive_empty_cache(force_sync=True)

    state_dict = {}
    per_tensor_param, _ = engine.get_per_tensor_param()
    for name, weight in per_tensor_param:
        if torch.distributed.get_rank() == 0:
            state_dict[name] = weight.cpu().detach()
        del weight

    if torch.distributed.get_rank() == 0:
        ray.get(synchronizer.set_model_state_dict.remote(state_dict, global_step))

    torch.distributed.barrier()
    getattr(torch, get_device_name()).empty_cache()
    logger.info(f"[Megatron] state_dict uploaded to Synchronizer: step={global_step}")


def megatron_sync_weight_nccl(engine, model_update_group):
    """Broadcast Megatron model parameters via NCCL.

    Uses the engine's get_per_tensor_param() to iterate over parameters
    and broadcasts each from rank 0.

    Args:
        engine: The McoreEngine instance (engine.actor.engine).
        model_update_group: The NCCL process group for weight broadcast.
    """
    per_tensor_param, _ = engine.get_per_tensor_param()
    for _, param in per_tensor_param:
        torch.distributed.broadcast(param, src=0, group=model_update_group)

    if torch.distributed.get_rank() == 0:
        getattr(torch, get_device_name()).synchronize()
