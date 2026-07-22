# -*- coding: utf-8 -*-
"""Custom vLLM worker classes for Trinity.

Provides device-aware worker subclasses for both GPU (vLLM) and NPU
(vllm-ascend). The appropriate class is selected at runtime via
:func:`get_trinity_worker_cls_name` based on the detected device type.
"""
import logging
from typing import Any

from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker

from trinity.utils.device import is_npu


def _suppress_layerwise_reload_warnings() -> None:
    """Silence benign vLLM layerwise reload warnings during weight sync."""
    try:
        logger = logging.getLogger("vllm.model_executor.model_loader.reload.layerwise")
        if logger is not None:
            logger.setLevel(logging.ERROR)
    except Exception:  # pragma: no cover - best-effort suppression
        pass


def _apply_trinity_patches(model_runner: Any) -> None:
    """Apply Trinity-specific patches shared by GPU and NPU workers.

    - Patches the vLLM fused-MoE weight loader (workaround for missing
      ``weight_loader`` on MoE params, also handles ACLGraphWrapper on NPU).
    - Patches prompt logprobs computation to apply temperature scaling.
    - Suppresses benign layerwise reload warnings during weight sync.
    """
    from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

    from trinity.common.models.vllm_patch.worker_patch import (
        patch_vllm_prompt_logprobs,
    )

    patch_vllm_moe_model_weight_loader(model_runner.model)
    patch_vllm_prompt_logprobs(model_runner)
    _suppress_layerwise_reload_warnings()


def _register_trinity_weight_transfer_engine() -> None:
    """Register Trinity's checkpoint weight transfer backend with vLLM."""
    from trinity.common.models.vllm_extension import (
        register_checkpoint_weight_transfer_engine,
    )

    register_checkpoint_weight_transfer_engine()


class TrinityGPUWorker(VLLMGPUWorker):
    def apply_patches(self) -> None:
        """Apply necessary patches to vLLM."""
        _apply_trinity_patches(self.model_runner)

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """Register Trinity weight-transfer engines before vLLM loads them."""
        _register_trinity_weight_transfer_engine()
        return super().load_model(*args, **kwargs)


if is_npu():
    from vllm_ascend.worker.worker import NPUWorker

    class TrinityNPUWorker(NPUWorker):
        """Trinity worker for Ascend NPU, based on vllm-ascend's NPUWorker.

        vllm-ascend's own patches are applied via ``adapt_patch()`` in
        ``NPUWorker.__init__``; this subclass only adds Trinity-specific
        patches (MoE weight loader + prompt logprobs) and registers
        Trinity's checkpoint weight transfer engine before model loading.
        """

        def apply_patches(self) -> None:
            """Apply Trinity-specific patches after vllm-ascend patches.

            ``adapt_patch()`` is already invoked in ``NPUWorker.__init__``,
            so here we only apply the Trinity-specific MoE weight loader and
            prompt logprobs patches.
            """
            _apply_trinity_patches(self.model_runner)

        def load_model(self, *args: Any, **kwargs: Any) -> None:
            """Register Trinity weight-transfer engines before vLLM loads them."""
            _register_trinity_weight_transfer_engine()
            return super().load_model(*args, **kwargs)


def get_trinity_worker_cls_name() -> str:
    """Return the Trinity worker class qualified name for the current device.

    Returns ``"trinity.common.models.vllm_worker.TrinityNPUWorker"`` on NPU
    and ``"trinity.common.models.vllm_worker.TrinityGPUWorker"`` otherwise.
    """
    if is_npu():
        return "trinity.common.models.vllm_worker.TrinityNPUWorker"
    return "trinity.common.models.vllm_worker.TrinityGPUWorker"
