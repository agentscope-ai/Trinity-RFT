# -*- coding: utf-8 -*-
"""Custom vLLM worker classes for Trinity."""
import logging
from typing import Any

from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker


def _suppress_layerwise_reload_warnings() -> None:
    """Silence benign vLLM layerwise reload warnings during weight sync."""
    try:
        logger = logging.getLogger("vllm.model_executor.model_loader.reload.layerwise")
        if logger is not None:
            logger.setLevel(logging.ERROR)
    except Exception:  # pragma: no cover - best-effort suppression
        pass


class TrinityGPUWorker(VLLMGPUWorker):
    def apply_patches(self) -> None:
        """Apply necessary patches to vLLM."""
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        from trinity.common.models.vllm_patch.worker_patch import (
            patch_vllm_prompt_logprobs,
        )

        patch_vllm_moe_model_weight_loader(self.model_runner.model)
        patch_vllm_prompt_logprobs(self.model_runner)
        _suppress_layerwise_reload_warnings()

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """Register Trinity weight-transfer engines before vLLM loads them."""
        from trinity.common.models.vllm_extension import (
            register_checkpoint_weight_transfer_engine,
        )

        register_checkpoint_weight_transfer_engine()
        return super().load_model(*args, **kwargs)
