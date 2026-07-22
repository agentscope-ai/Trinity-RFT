# -*- coding: utf-8 -*-
"""FSDP-specific checkpoint and weight sync helpers for Trinity.

These helper functions are called by `TrinityActorRolloutRefWorker` to
perform FSDP/FSDP2-specific operations that veRL 0.8's engine does not
provide natively:

- fsdp_save_state_dict:    Save sharded state dict for checkpoint sync
- fsdp_upload_state_dict:  Upload full state dict to Synchronizer (memory sync)
- fsdp_sync_weight_nccl:   Broadcast full params via NCCL

All functions receive the `engine` object (an FSDPEngine instance from
`verl.workers.engine.fsdp`) which exposes:
  - engine.module: the FSDP-wrapped model
  - engine.get_per_tensor_param(): generator of (name, tensor) pairs
  - engine.save_checkpoint() / engine.load_checkpoint()
"""
import os
import warnings

import ray
import torch
from torch.distributed.fsdp import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from verl.utils.device import is_cuda_available
from verl.utils.fs import local_mkdir_safe
from verl.utils.fsdp_utils import get_fsdp_full_state_dict, get_fsdp_state_ctx

from trinity.trainer.verl.checkpoint import CheckpointCoordinator
from trinity.utils.device import get_device_name


def _save_checkpoint_metadata(engine, local_path: str, logger):
    """Save HF model config and tokenizer to ``{local_path}/huggingface/``.

    ``FSDPModelMerger`` (used by Synchronizer to load sharded checkpoints)
    requires ``config.json`` to understand the model structure.  This is
    cheap — only a few small JSON files, no model weights.

    ``fsdp_config.json`` is NOT saved here — the loading side infers
    ``world_size`` from shard filenames instead.
    """
    from verl.utils.fsdp_utils import fsdp_version

    model = engine.module
    if fsdp_version(model) == 1:
        unwrapped = getattr(model, "_fsdp_wrapped_module", model)
    else:
        unwrapped = model

    hf_path = os.path.join(local_path, "huggingface")
    local_mkdir_safe(hf_path)

    model_config = unwrapped.config
    if hasattr(model_config, "auto_map") and None in model_config.auto_map:
        model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}
    model_config.save_pretrained(hf_path)

    # Save tokenizer / processor if the engine's checkpoint_manager has one.
    ckpt_mgr = getattr(engine, "checkpoint_manager", None)
    if ckpt_mgr is not None and getattr(ckpt_mgr, "processing_class", None) is not None:
        ckpt_mgr.processing_class.save_pretrained(hf_path)

    logger.info(f"Saved HF config/tokenizer to {hf_path}")


def fsdp_save_state_dict(
    engine,
    local_path: str,
    global_step: int,
    coordinator: CheckpointCoordinator,
    logger,
):
    """Save FSDP model state dict (sharded) for checkpoint-based weight sync.

    Collects the sharded state dict on the main thread (requires FSDP context),
    then offloads ``torch.save`` to a background thread via ``coordinator`` so
    the training loop can continue without waiting for I/O.

    On rank 0, also saves HF model config and tokenizer to
    ``{local_path}/huggingface/`` so that ``FSDPModelMerger`` (used by
    Synchronizer) can merge the shards back into a full state dict.

    The ``coordinator`` notifies CheckpointMonitor before and after saving, which
    gates the iteration-file update and prevents the Synchronizer from reading
    an incomplete checkpoint.

    Args:
        engine: The FSDPEngine instance (engine.actor.engine).
        local_path: Local directory path to save the state dict.
        global_step: Current training step.
        coordinator: CheckpointCoordinator for background save + Monitor integration.
        logger: Logger instance from the calling worker.
    """
    if local_path is None:
        return

    local_path = local_mkdir_safe(local_path)

    model = engine.module
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
    optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with get_fsdp_state_ctx(model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            state_dict = model.state_dict()

    # Save metadata (HF config/tokenizer + fsdp_config.json) on rank 0
    # so FSDPModelMerger can merge the shards.
    if rank == 0:
        _save_checkpoint_metadata(engine, local_path, logger)

    path = os.path.join(local_path, f"model_world_size_{world_size}_rank_{rank}.pt")

    if rank == 0:
        coordinator.save_async(
            "model_state_dict",
            lambda: torch.save(state_dict, path),
            global_step,
            is_state_dict=True,
        )
    else:
        torch.save(state_dict, path)

    logger.info(f"[FSDP] actor state_dict save initiated: path={local_path}, step={global_step}")


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


def fsdp_sync_weight_nccl(engine, model_update_group):
    """Broadcast full model parameters via NCCL for FSDP/FSDP2.

    For FSDP1: Uses FSDP.summon_full_params to gather full parameters before broadcast.
    For FSDP2: Uses param.full_tensor() to get the full parameter.

    Args:
        engine: The FSDPEngine instance (engine.actor.engine).
        model_update_group: The NCCL process group for weight broadcast.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    strategy = getattr(engine, "_strategy", "fsdp")

    if strategy == "fsdp":
        # FSDP1 path: summon full params for each FSDP module
        for name_prefix, module in engine.module.named_modules():
            if isinstance(module, FSDP):
                with FSDP.summon_full_params(module, recurse=False):
                    if torch.distributed.get_rank() == 0:
                        for name, param in module.named_parameters():
                            torch.distributed.broadcast(param, 0, group=model_update_group)
    else:
        # FSDP2 path: use full_tensor()
        per_tensor_param, _ = engine.get_per_tensor_param()
        for name, param in per_tensor_param:
            if hasattr(param, "full_tensor"):
                full_param = param.full_tensor().detach()
            else:
                full_param = param
            if torch.distributed.get_rank() == 0:
                torch.distributed.broadcast(full_param, 0, group=model_update_group)

    if torch.distributed.get_rank() == 0:
        getattr(torch, get_device_name()).synchronize()
