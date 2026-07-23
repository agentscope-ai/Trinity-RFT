# -*- coding: utf-8 -*-
"""Trinity-specific worker extending veRL's engine-based ActorRolloutRefWorker.

This replaces the old fsdp_workers.py and megatron_workers.py with a single
thin extension that adds Trinity-specific hooks on top of veRL's unified
engine-based training worker.

Key additions over the base class:
- set_trinity_config(): injects Trinity's pluggable loss function, rollout engine type
  and Ray namespace
- save_state_dict / upload_state_dict / sync_weight_nccl: Trinity-specific
  checkpoint and weight sync methods that delegate to strategy-specific helpers
- get_weight_sync_info / setup_weight_sync_group / teardown_weight_sync_group:
  NCCL weight sync group lifecycle for Explorer↔Trainer
"""
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

import torch
from omegaconf import DictConfig
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.engine_workers import (
    ActorRolloutRefWorker,
    TrainingWorker,
    TrainingWorkerConfig,
)

from trinity.common.config import AlgorithmConfig
from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.verl.checkpoint import CheckpointCoordinator
from trinity.trainer.verl.losses import build_trinity_loss
from trinity.utils.device import get_device_name
from trinity.utils.distributed import WeightTransferEngine
from trinity.utils.log import get_logger
from trinity.utils.stream_saver import save_safetensors_streaming


class TrinityActorRolloutRefWorker(ActorRolloutRefWorker):
    """Extends veRL's ActorRolloutRefWorker with Trinity-specific hooks.

    Additions over the base class:
    - set_trinity_config(): injects Trinity's pluggable loss function, rollout engine
      type and Ray namespace
    - save_state_dict / upload_state_dict: checkpoint and memory-based weight sync
    - sync_weight_nccl: NCCL-based weight broadcast for Explorer↔Trainer
    - get_weight_sync_info / setup_weight_sync_group / teardown_weight_sync_group:
      NCCL weight sync group lifecycle (setup is driven by Synchronizer)
    - Applies Trinity-specific monkey patches after model init
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config=config, role=role, **kwargs)
        self.logger = get_logger(f"{role}_{self.rank}", in_ray_actor=True)
        self._is_rollout = False  # Disable rollout in Trainer
        self._algo_config: Optional[AlgorithmConfig] = None
        self._ray_namespace: Optional[str] = None
        self._state_dict_meta_list = None
        self._coordinator: Optional[CheckpointCoordinator] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model, then apply Trinity-specific monkey patches.

        veRL main already applies its own monkey patches (ulysses SP, prefix
        grouper, tiled MLP, VLM patching) inside the engine. Trinity's
        monkey_patch module adds patches that veRL main does NOT have:
        - Qwen3.5 GatedDeltaNet / hybrid-attention layer support
        - GLM4V text model forward
        - Fused kernels VLM SP bugfix (patch_fused_kernels)
        - Flops counter registration for qwen3_5
        """
        from trinity.trainer.verl.monkey_patch import patch_verl_engine

        patch_verl_engine()

        # Patch veRL engine for LoRA + FSDP2 dtype alignment.
        # veRL's _build_lora_module does not align trainable param dtypes
        # to the FSDP2 mixed-precision param_dtype, causing a gradient dtype
        # mismatch during backward. We monkey-patch _build_lora_module to add
        # the alignment step that the legacy fsdp_workers.py had.
        self._maybe_patch_lora_fsdp2()

        # Strip "rollout" from role so the base class skips rollout engine init.
        # veRL checks `if "rollout" in self.role:` to decide whether to build
        # the rollout engine — Trinity handles rollout in Explorer, not Trainer.
        original_role = self.role
        self.role = self.role.replace("_rollout", "")
        super().init_model()
        self.role = original_role

        self._cache_state_dict_meta()

    def _maybe_patch_lora_fsdp2(self):
        """Monkey-patch veRL engine to align LoRA param dtypes for FSDP2.

        When LoRA is used with FSDP2, PEFT may create adapter parameters in
        fp32 while FSDP2 produces bf16 gradients (from MixedPrecisionPolicy
        param_dtype). This causes a RuntimeError during backward because the
        gradient dtype doesn't match the parameter's grad_dtype.

        We wrap the engine's _build_lora_module to cast all trainable params
        to the FSDP2 param_dtype (bf16) after LoRA creation, before FSDP2
        wrapping — matching the legacy fsdp_workers.py behavior.
        """
        is_lora = (
            self.config.model.get("lora_rank", 0) > 0
            or self.config.model.get("lora_adapter_path") is not None
        )
        strategy = self.config.actor.get("strategy", "fsdp2")
        if not is_lora or strategy != "fsdp2":
            return

        from verl.utils.torch_dtypes import PrecisionType
        from verl.workers.engine.fsdp import transformer_impl as fsdp_impl

        mixed_precision_config = self.config.actor.get("fsdp_config", {}).get("mixed_precision")
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
        else:
            param_dtype = torch.bfloat16

        # Find the base FSDP engine class that defines _build_lora_module.
        # Use attribute lookup to be robust across veRL versions.
        for name in dir(fsdp_impl):
            cls = getattr(fsdp_impl, name)
            if (
                isinstance(cls, type)
                and hasattr(cls, "_build_lora_module")
                and "_build_lora_module" in cls.__dict__
            ):
                original_build_lora = cls._build_lora_module

                def _patched_build_lora_module(engine_self, module, _orig=original_build_lora):
                    module = _orig(engine_self, module)
                    for param in module.parameters():
                        if param.requires_grad and param.dtype != param_dtype:
                            param.data = param.data.to(dtype=param_dtype)
                    return module

                cls._build_lora_module = _patched_build_lora_module
                break

    def _save_lora(self, local_path: str):
        """Save LoRA adapter weights alongside the checkpoint.

        The Explorer needs the LoRA adapter at ``{local_path}/lora_adapter``
        to reload weights into vLLM after checkpoint sync. This mirrors the
        logic in the legacy ``fsdp_workers.py`` ``_save_lora`` method.
        """
        if self.actor is None or not hasattr(self.actor, "engine"):
            return
        engine = self.actor.engine
        model = engine.module
        # Check if model is a PeftModel
        peft_model = getattr(model, "_fsdp_wrapped_module", model)
        if not hasattr(peft_model, "peft_config"):
            return

        import json
        import os
        from dataclasses import asdict

        from safetensors.torch import save_file
        from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params

        rank = torch.distributed.get_rank()
        lora_save_path = os.path.join(local_path, "lora_adapter")

        peft_config = {}
        if rank == 0:
            os.makedirs(lora_save_path, exist_ok=True)
            peft_config = asdict(peft_model.peft_config.get("default", {}))
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])

        try:
            if fsdp_version(model) > 0:
                model = model.to(getattr(torch, get_device_name()).current_device())
                lora_params = layered_summon_lora_params(model)
                if rank == 0:
                    save_file(
                        lora_params, os.path.join(lora_save_path, "adapter_model.safetensors")
                    )
                    with open(
                        os.path.join(lora_save_path, "adapter_config.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(peft_config, f, ensure_ascii=False, indent=4)
            else:
                # FSDP2: parameters are DTensors, use full_tensor() to collect
                from peft.utils.save_and_load import get_peft_model_state_dict

                state_dict = {}
                for name, param in model.named_parameters():
                    if hasattr(param, "full_tensor"):
                        state_dict[name] = param.full_tensor().detach().cpu()
                    else:
                        state_dict[name] = param.detach().cpu()
                lora_params = get_peft_model_state_dict(peft_model, state_dict=state_dict)
                if rank == 0:
                    save_file(
                        dict(lora_params),
                        os.path.join(lora_save_path, "adapter_model.safetensors"),
                    )
                    with open(
                        os.path.join(lora_save_path, "adapter_config.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(peft_config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.error(f"Save LoRA adapter error: {e}")

        torch.distributed.barrier()
        self.logger.info(f"Saved LoRA adapter to: {lora_save_path}")

    @contextmanager
    def _save_hf_checkpoint_if_requested(self, save_as_hf: bool):
        """Temporarily enable HF weight export for a single checkpoint save."""
        if not save_as_hf:
            yield
            return

        engine = getattr(self.actor, "engine", None)
        checkpoint_manager = getattr(engine, "checkpoint_manager", None)
        if checkpoint_manager is None:
            checkpoint_manager = getattr(engine, "checkpoint_mananager", None)
        if checkpoint_manager is None:
            yield
            return

        original_contents = list(checkpoint_manager.checkpoint_save_contents)
        if "hf_model" not in checkpoint_manager.checkpoint_save_contents:
            checkpoint_manager.checkpoint_save_contents = [
                *checkpoint_manager.checkpoint_save_contents,
                "hf_model",
            ]
        try:
            yield
        finally:
            checkpoint_manager.checkpoint_save_contents = original_contents

    def _cache_state_dict_meta(self):
        """Cache state_dict meta (names, dtypes, shapes) from get_per_tensor_param.

        Uses the same parameter source as sync_weight_nccl to ensure dtype
        and shape are consistent with what is actually broadcast.
        """
        if self.actor is None:
            return
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        self._state_dict_meta_list = [
            (name, str(param.dtype).split(".")[-1], tuple(param.shape))
            for name, param in per_tensor_param
        ]
        self.logger.info(
            f"Cached state_dict meta: {len(self._state_dict_meta_list or [])} parameters"
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_trinity_config(
        self, algo_config: AlgorithmConfig, rollout_engine_type: str, ray_namespace: str
    ):
        """Set Trinity-specific runtime config on the worker.

        This is called by VERLTrainer after worker initialization to inject:
        - The pluggable policy loss, KL loss, and entropy loss from Trinity's algorithm registry
        - The Ray namespace used to locate Synchronizer and CheckpointMonitor actors
        """
        self._algo_config = algo_config
        self._ray_namespace = ray_namespace
        self._rollout_engine_type = rollout_engine_type
        if self.actor is not None:
            loss_fn = build_trinity_loss(algo_config)
            self.actor.set_loss_fn(loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_weight_sync_info(self):
        """Return (addr, port, state_dict_meta) from rank 0 for NCCL group setup.

        Uses cached meta from init_model() — no parameter materialization.
        Other ranks return None.
        """
        if torch.distributed.get_rank() == 0:
            aggressive_empty_cache(force_sync=True)
            addr, port = self.get_available_master_addr_port()
            self.logger.info(f"Weight sync info: {addr}:{port}")
            return addr, int(port), self._state_dict_meta_list
        return None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def setup_weight_sync_group(
        self,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        timeout: int,
    ):
        """Join the NCCL process group for weight sync.

        Called concurrently with Explorer's setup_weight_sync_group.
        Only rank 0 creates the process group.
        """
        if torch.distributed.get_rank() == 0:
            self.logger.info(
                f"Trainer init_process_group {master_address}:{master_port} ({world_size})."
            )
            self.weight_transfer_engine = WeightTransferEngine.create(
                engine_type=self._rollout_engine_type,
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                group_name=group_name,
            )
            self.logger.info("Trainer init_process_group done.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def teardown_weight_sync_group(self):
        """Destroy the NCCL process group for weight sync."""
        if torch.distributed.get_rank() == 0 and self.weight_transfer_engine is not None:
            self.logger.info("Tearing down weight sync group.")
            self.weight_transfer_engine.teardown()
            self.weight_transfer_engine = None

    def _rank0_iterator_wrapper(
        self, per_tensor_param: Iterator[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Advance distributed parameter gathering on every rank, yielding only on rank 0."""
        is_rank0 = torch.distributed.get_rank() == 0
        for name, param in per_tensor_param:
            if is_rank0:
                yield name, param

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_weight_nccl(self):
        """Sync model weights across workers using NCCL.

        Broadcasts full model parameters from rank 0 (trainer) to all
        Explorer ranks via the NCCL process group.
        """
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        weight_iterator = self._rank0_iterator_wrapper(per_tensor_param)
        if torch.distributed.get_rank() == 0:
            self.logger.info("Starting NCCL weight sync broadcast.")
            self.weight_transfer_engine.sync_weight(
                iterator=weight_iterator,
            )
            getattr(torch, get_device_name()).synchronize()
            self.logger.info("Finished NCCL weight sync broadcast.")
        else:
            for _ in weight_iterator:
                pass

    def _get_coordinator(self) -> CheckpointCoordinator:
        if self._coordinator is None:
            from trinity.trainer.verl.trainer import CheckpointMonitor

            monitor = CheckpointMonitor.get_actor(
                namespace=self._ray_namespace,  # type: ignore
            )
            self._coordinator = CheckpointCoordinator(monitor)
        return self._coordinator

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_state_dict(self, local_path, global_step=0):
        """Save model state dict for checkpoint-based weight sync.

        On rank 0, saving is offloaded to a background thread via the
        CheckpointCoordinator, which also notifies CheckpointMonitor so the
        iteration file is only updated after the save completes.
        """
        """Save model state dict for checkpoint-based weight sync.

        Uses ``get_per_tensor_param()`` to gather full tensors from all
        ranks (FSDP all-gather / DTensor full_tensor), then **streams**
        them to a safetensors file one tensor at a time on rank 0.
        Non-rank-0 workers consume the generator to participate in
        collective operations but do not save.

        The streaming write goes through the OS page cache (fast memcpy).
        After the main-thread write completes, ``fsync`` + atomic rename
        are offloaded to a background thread via
        :class:`CheckpointCoordinator` so that the training loop is not
        blocked by disk I/O.
        """
        coordinator = self._get_coordinator()
        rank = torch.distributed.get_rank()

        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()

        if rank == 0:
            os.makedirs(local_path, exist_ok=True)
            filepath = os.path.join(local_path, "model.safetensors")

            def _rank0_iter():
                for name, weight in per_tensor_param:
                    yield name, weight.cpu().detach()

            tmp_path = save_safetensors_streaming(
                _rank0_iter(),
                filepath,
                state_dict_meta=self._state_dict_meta_list,
                rename=False,
            )

            def _finalize():
                fd = os.open(tmp_path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(tmp_path, filepath)

            coordinator.save_async("model_state_dict", _finalize, global_step, is_state_dict=True)
            self.logger.info(
                f"Actor state_dict save initiated: path={local_path}, step={global_step}"
            )
        else:
            # Consume generator to participate in FSDP all-gather.
            for _ in per_tensor_param:
                pass

        self._save_lora(local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, global_step=0, **kwargs):
        """Save full checkpoint with CheckpointMonitor coordination.

        Delegates to the engine's save_checkpoint (which is synchronous due to
        internal distributed barriers), but wraps it with CheckpointMonitor
        notifications on rank 0 to prevent the Synchronizer from reading an
        incomplete checkpoint.

        Note: max_ckpt_to_keep is NOT passed to the engine — checkpoint
        retention is managed by VERLTrainer to avoid veRL's double-counting
        bug when the same path is registered multiple times.
        """
        coordinator = self._get_coordinator()
        rank = torch.distributed.get_rank()
        save_as_hf = kwargs.pop("save_as_hf", False)
        with self._save_hf_checkpoint_if_requested(save_as_hf):
            if rank == 0:
                coordinator.save_sync(
                    lambda: self.actor.save_checkpoint(local_path, global_step=global_step),
                    global_step,
                    is_state_dict=True,
                )
            else:
                self.actor.save_checkpoint(local_path, global_step=global_step)
        self._save_lora(local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def wait_on_save_thread(self):
        """Block until all background save threads complete."""
        if self._coordinator is not None:
            self._coordinator.wait_all()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def upload_state_dict(self, global_step=0):
        """Upload model state dict to Synchronizer for memory-based weight sync.

        Delegates to strategy-specific helpers in fsdp_engine.py or
        megatron_engine.py based on the actor's strategy.
        """
        strategy = self.config.actor.strategy
        synchronizer = Synchronizer.get_actor(namespace=self._ray_namespace)

        if strategy.startswith("fsdp"):
            from trinity.trainer.verl.fsdp_engine import fsdp_upload_state_dict

            fsdp_upload_state_dict(self.actor.engine, synchronizer, global_step, logger=self.logger)
        elif strategy.startswith("megatron"):
            from trinity.trainer.verl.megatron_engine import megatron_upload_state_dict

            megatron_upload_state_dict(
                self.actor.engine, synchronizer, global_step, logger=self.logger
            )
        else:
            raise ValueError(f"Unsupported strategy for upload_state_dict: {strategy}")


class TrinityCriticWorker(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig):
        super().__init__(config)
        from trinity.trainer.verl.monkey_patch import patch_verl_engine

        patch_verl_engine()
