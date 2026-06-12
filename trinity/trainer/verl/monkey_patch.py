from types import MethodType
from typing import Optional

import torch
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngine,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)


# from https://github.com/verl-project/verl/pull/6604
# Remove this patch once the fix is released in veRL
def save_checkpoint(
    self,
    local_path: str,
    hdfs_path: Optional[str] = None,
    global_step: int = 0,
    max_ckpt_to_keep: Optional[int] = None,
    **kwargs,
) -> None:
    """
    Save FSDP checkpoint, handling parameter offload as needed.
    """
    origin_module_device = next(self.module.parameters()).device.type
    if (self._is_offload_param or origin_module_device == "cpu") and not getattr(
        self, "_uses_fsdp2_cpu_offload_policy", False
    ):
        load_fsdp_model_to_gpu(self.module)

    self.checkpoint_manager.save_checkpoint(
        local_path=local_path,
        hdfs_path=hdfs_path,
        global_step=global_step,
        max_ckpt_to_keep=max_ckpt_to_keep,
    )

    torch.distributed.barrier()
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.module)


def patch_verl_engine(engine):
    if engine is None:
        return
    if isinstance(engine, FSDPEngine):
        engine.save_checkpoint = MethodType(save_checkpoint, engine)
