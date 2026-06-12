# -*- coding: utf-8 -*-
"""Streaming safetensors writer — saves one tensor at a time to avoid OOM.

Standard ``safetensors.torch.save_file`` requires the entire state dict in
memory.  For 70B+ models this can exceed available CPU RAM.  This module
provides :func:`save_safetensors_streaming` which iterates a tensor generator
and writes each tensor directly to disk, keeping only one tensor in memory at
any point.

The implementation uses a **seek-back** approach:

1. Reserve space for the safetensors header (exactly sized from pre-collected
   ``state_dict_meta``).
2. Write tensor data sequentially, collecting metadata along the way.
3. Build the JSON header and seek back to the beginning to write it.

The resulting file is a fully valid safetensors file that can be loaded by
``safetensors.torch.load_file`` (which uses mmap internally).

Writes go through the OS page cache, so the function returns quickly after
the last tensor is written.  Callers can hand off ``fsync`` + ``rename`` to
a background thread for minimal main-thread blocking.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Iterable, NamedTuple

import torch

# ---------------------------------------------------------------------------
# StateDictMeta — lightweight tensor metadata for exact header sizing
# ---------------------------------------------------------------------------


class TensorMeta(NamedTuple):
    """Metadata for a single tensor, sufficient to compute its safetensors header entry."""

    name: str
    dtype: str  # safetensors dtype string, e.g. "BF16"
    shape: list[int]


# A full state-dict's worth of tensor metadata.
StateDictMeta = list[TensorMeta]

# ---------------------------------------------------------------------------
# Dtype mapping (PyTorch → safetensors string / element size)
# ---------------------------------------------------------------------------

DTYPE_TO_SAFETENSORS: dict[torch.dtype, str] = {
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}

DTYPE_ELEMENT_SIZE: dict[str, int] = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a CPU tensor to its raw bytes in C-contiguous order."""
    t = tensor.detach().contiguous()
    # numpy() handles the dtype↔raw mapping correctly for all types,
    # including bfloat16 (via numpy ≥ 1.24 or torch's internal conversion).
    try:
        return t.numpy().tobytes()
    except (RuntimeError, TypeError):
        # Fallback for dtypes not natively supported by numpy (e.g. bfloat16
        # on older numpy).  View as uint8 to get the raw byte representation.
        return t.view(torch.uint8).numpy().tobytes()


def _compute_exact_header_size(meta: StateDictMeta) -> int:
    """Compute the exact safetensors header JSON size from pre-collected metadata.

    Builds a full header dict (with placeholder data_offsets) just to measure
    the serialised JSON length, then aligns to 8 bytes as required by the
    safetensors format.
    """
    header: dict[str, dict] = {}
    data_offset = 0
    for name, dtype_str, shape in meta:
        element_size = DTYPE_ELEMENT_SIZE.get(dtype_str, 0)
        nbytes = 1
        for s in shape:
            nbytes *= s
        nbytes *= element_size
        header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [data_offset, data_offset + nbytes],
        }
        data_offset += nbytes
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return ((len(header_json) + 7) // 8) * 8


def collect_state_dict_meta(
    tensor_iter: Iterable[tuple[str, torch.Tensor]],
) -> StateDictMeta:
    """Collect ``(name, dtype, shape)`` metadata by iterating a tensor generator.

    Iterates *tensor_iter* once, recording each tensor's name, safetensors
    dtype string, and shape.  The returned :data:`StateDictMeta` can be
    passed directly to :func:`save_safetensors_streaming` for exact header
    sizing.

    This is the shared helper used by all worker ``_cache_state_dict_meta``
    implementations — callers only need to provide the correct iterator
    (and manage any context / offload setup around the call).

    Args:
        tensor_iter: Yields ``(name, tensor)`` pairs (same format expected
            by :func:`save_safetensors_streaming`).

    Returns:
        A :data:`StateDictMeta` list.
    """
    meta: StateDictMeta = []
    for name, tensor in tensor_iter:
        dtype_str = DTYPE_TO_SAFETENSORS.get(tensor.dtype)
        if dtype_str is not None:
            meta.append(TensorMeta(name, dtype_str, list(tensor.shape)))
        del tensor
    return meta


def save_safetensors_streaming(
    tensor_iter: Iterable[tuple[str, torch.Tensor]],
    filepath: str | os.PathLike,
    state_dict_meta: StateDictMeta,
    *,
    rename: bool = True,
) -> str:
    """Stream-write a safetensors file, holding only one tensor in memory.

    Args:
        tensor_iter: Yields ``(name, tensor)`` pairs.  Each tensor must
            already reside on CPU.  After the tensor's bytes are written it
            is no longer referenced — the caller should ``del`` the tensor
            from their side as well.
        filepath: Destination path.  Data is first written to
            ``filepath + ".tmp"``; if *rename* is ``True`` the temp file is
            atomically renamed to *filepath* before returning.
        state_dict_meta: Pre-collected ``(name, dtype, shape)`` metadata for
            every tensor that *tensor_iter* will yield.  The header space is
            computed **exactly** from this metadata, so the header will
            always fit the reserved space without any rewrite.
            Callers should collect this once during initialization via
            ``_cache_state_dict_meta()`` and reuse it for every save.
        rename: If ``True`` (default), atomically replace *filepath* with
            the completed temp file.  If ``False``, leave the temp file on
            disk — the caller is responsible for ``fsync`` + ``rename``
            (useful for hybrid async mode).

    Returns:
        The path actually written: *filepath* when ``rename=True``,
        otherwise ``str(filepath) + ".tmp"``.
    """
    filepath = str(filepath)
    tmp_path = filepath + ".tmp"

    # --- Exact header sizing from pre-collected metadata ---
    max_header_size = _compute_exact_header_size(state_dict_meta)
    # Align to 8 bytes (safetensors convention).
    max_header_size = ((max_header_size + 7) // 8) * 8

    header: dict[str, dict] = {}
    data_offset = 0

    with open(tmp_path, "w+b") as f:
        # Phase 1: Reserve space for header_size (8B) + header JSON.
        f.write(b"\x00" * (8 + max_header_size))

        # Phase 2: Stream tensor data.
        for name, tensor in tensor_iter:
            dtype_str = DTYPE_TO_SAFETENSORS.get(tensor.dtype)
            if dtype_str is None:
                raise ValueError(
                    f"Unsupported dtype {tensor.dtype} for tensor '{name}'. "
                    f"Supported: {list(DTYPE_TO_SAFETENSORS.keys())}"
                )

            raw = _tensor_to_bytes(tensor)
            nbytes = len(raw)

            header[name] = {
                "dtype": dtype_str,
                "shape": list(tensor.shape),
                "data_offsets": [data_offset, data_offset + nbytes],
            }

            f.write(raw)
            data_offset += nbytes
            # Explicitly delete to help GC release memory promptly.
            del raw, tensor

        # Phase 3: Build header JSON and write it back at the start.
        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

        # Header always fits — it was sized exactly from state_dict_meta.
        # Pad with spaces to fill exactly max_header_size bytes.
        padded = header_json + b" " * (max_header_size - len(header_json))
        f.seek(0)
        f.write(struct.pack("<Q", max_header_size))
        f.write(padded)

    if rename:
        os.replace(tmp_path, filepath)
        return filepath
    return tmp_path
