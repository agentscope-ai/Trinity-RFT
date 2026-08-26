# -*- coding: utf-8 -*-
"""Device detection and abstraction layer.

Unifies the differences among NPU / GPU / CPU devices for trinity modules.

"""
import functools
import os
from enum import Enum

import torch

from trinity.utils.log import get_logger

logger = get_logger(__name__)


class DeviceType(str, Enum):
    """Device type enum. Inherits str so it can be passed directly to APIs that
    require "npu"/"cuda" strings."""

    NPU = "npu"
    CUDA = "cuda"
    CPU = "cpu"


# ---------- Private helpers ----------


def _normalize_ascend_visible_devices() -> None:
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not raw or not raw.strip():
        return

    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if len(parts) <= 1:
        return

    try:
        ids = [int(p) for p in parts]
    except ValueError:
        # Non-integer values (e.g. UUIDs in some CANN versions); skip normalization
        return

    if ids == sorted(ids):
        return

    normalized = ",".join(str(i) for i in sorted(ids))
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = normalized
    logger.warning(
        "ASCEND_RT_VISIBLE_DEVICES was non-ascending (%r); normalized to %r. "
        "Non-ascending values cause torch.npu.is_available() to return False, "
        "leading to incorrect device detection. Please fix the upstream "
        "configuration (e.g. Ray actor env vars or launch script).",
        raw,
        normalized,
    )


# ---------- Core detection API ----------


@functools.lru_cache(maxsize=1)
def get_device_type() -> DeviceType:
    """Detect the currently available device type, with process-level caching.

    Returns:
        DeviceType.NPU / DeviceType.CUDA / DeviceType.CPU
    """
    _normalize_ascend_visible_devices()
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
