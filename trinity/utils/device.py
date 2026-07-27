# -*- coding: utf-8 -*-
"""Device detection and abstraction layer.

Unifies the differences among NPU / GPU / CPU devices for trinity modules.

"""
import functools
import os
from enum import Enum

import torch


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

    if hasattr(torch, "npu") and torch.npu.is_available():
        return DeviceType.NPU
    elif torch.cuda.is_available():
        return DeviceType.CUDA
    else:
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
        major, _ = torch.cuda.get_device_capability(0)
        return major
    return 0
