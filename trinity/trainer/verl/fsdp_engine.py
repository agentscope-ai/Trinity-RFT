# -*- coding: utf-8 -*-
"""FSDP-specific weight sync helpers for Trinity.

These helper functions are called by `TrinityActorRolloutRefWorker` to
perform FSDP/FSDP2-specific operations:

- fsdp_upload_state_dict:  Upload full state dict to Synchronizer (memory sync)

Note: ``save_state_dict`` (checkpoint sync) is now handled uniformly by
the worker via ``get_per_tensor_param()`` + safetensors — the old
``fsdp_save_state_dict`` has been removed.
"""

import ray
import torch
from verl.utils.fsdp_utils import get_fsdp_full_state_dict


def fsdp_upload_state_dict(engine, synchronizer, global_step: int, logger):
    """Upload full FSDP model state dict to Synchronizer for memory-based weight sync.

    Gathers the full state dict on rank 0 and sends it to the Synchronizer
    actor, which makes it available for the Explorer to load.

    Args:
        engine: The FSDPEngine instance (engine.actor.engine).
        synchronizer: The Synchronizer Ray actor handle.
        global_step: Current training step (used as version key).
    """
    if global_step == 0:
        return

    model = engine.module
    state_dict = get_fsdp_full_state_dict(model, offload_to_cpu=True, rank0_only=True)

    if torch.distributed.get_rank() == 0:
        ray.get(synchronizer.set_model_state_dict.remote(state_dict, global_step))

    torch.distributed.barrier()
    logger.info(f"[FSDP] state_dict uploaded to Synchronizer: step={global_step}")
