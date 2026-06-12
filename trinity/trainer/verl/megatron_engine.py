# -*- coding: utf-8 -*-
"""Megatron-specific weight sync helpers for Trinity.

These helper functions are called by `TrinityActorRolloutRefWorker` to
perform Megatron-specific operations:

- megatron_upload_state_dict:  Upload state dict to Synchronizer (memory sync)

Note: ``save_state_dict`` (checkpoint sync) is now handled uniformly by
the worker via ``get_per_tensor_param()`` + safetensors — the old
``megatron_save_state_dict`` has been removed.
"""
import ray
import torch
from verl.utils.memory_utils import aggressive_empty_cache


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
    torch.cuda.empty_cache()
    logger.info(f"[Megatron] state_dict uploaded to Synchronizer: step={global_step}")
