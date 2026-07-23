# -*- coding: utf-8 -*-
"""Device detection and abstraction layer.

Unifies the differences among NPU / GPU / CPU devices for trinity modules.
Low-level device operations (e.g. get_torch_device / get_device_id) prefer
reusing verl.utils.device; this module only handles
"detection + initialization + metadata".

Detection priority:
    1. Environment variable TRINITY_DEVICE forces override (for debugging/special cases)
    2. torch_npu installed and torch.npu.is_available() -> NPU
    3. torch.cuda.is_available() -> CUDA
    4. CPU fallback
"""
import functools
import importlib
import os
from enum import Enum


class DeviceType(str, Enum):
    """Device type enum. Inherits str so it can be passed directly to APIs that
    require "npu"/"cuda" strings."""

    NPU = "npu"
    CUDA = "cuda"
    CPU = "cpu"


# ---------- Core detection API ----------

@functools.lru_cache(maxsize=1)
def get_device_type() -> DeviceType:
    """Detect the currently available device type, with process-level caching.

    Returns:
        DeviceType.NPU / DeviceType.CUDA / DeviceType.CPU
    """
    env_override = os.environ.get("TRINITY_DEVICE", "").lower()
    if env_override in ("npu", "cuda", "cpu"):
        return DeviceType(env_override)

    if _is_torch_npu_available():
        return DeviceType.NPU

    import torch
    if torch.cuda.is_available():
        return DeviceType.CUDA

    return DeviceType.CPU


def is_npu() -> bool:
    """Whether the current process is running in an NPU environment."""
    return get_device_type() is DeviceType.NPU


def is_cuda() -> bool:
    """Whether the current process is running in a CUDA environment."""
    return get_device_type() is DeviceType.CUDA


def is_cpu() -> bool:
    """Whether the current process is running in a CPU environment."""
    return get_device_type() is DeviceType.CPU


# ---------- Ray / distributed related ----------

def get_ray_resource_key() -> str:
    """Accelerator key name in the Ray cluster Resources dict.

    NPU nodes report as "NPU", GPU nodes report as "GPU".
    Used by config_validator to auto-detect gpu_per_node.
    """
    return "NPU" if is_npu() else "GPU"


def get_collective_backend() -> str:
    """Collective communication backend name. NPU uses hccl, GPU uses nccl."""
    return "hccl" if is_npu() else "nccl"


def get_device_capability() -> int:
    """Get major device capability version (device-agnostic).

    Used to decide whether to enable meta tensor initialization for FSDP2.
    - NPU: returns 10 (supports meta tensor init, equivalent to sm90+)
    - CUDA: returns the actual major compute capability from torch.cuda
    - CPU: returns 0 (meta tensor not beneficial)
    """
    if is_npu():
        return 10
    if is_cuda():
        import torch
        major, _ = torch.cuda.get_device_capability(0)
        return major
    return 0



# ---------- Private helpers ----------

def _is_torch_npu_available() -> bool:
    """Detect whether torch_npu is available, without hard-depending on the package.

    Note: in some environments torch_npu is installed but the CANN environment
    variables are not sourced, causing is_available() to raise an exception;
    we need to fall back to returning False here.
    """
    try:
        torch_npu = importlib.import_module("torch_npu")
        return bool(torch_npu.npu.is_available())
    except ImportError:
        return False
    except Exception:
        return False