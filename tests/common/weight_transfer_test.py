# -*- coding: utf-8 -*-
"""Unit tests for trinity.common.weight_transfer utilities.

These tests run on CPU only — no GPU or NCCL required.  They validate the
chunk splitting/merging round-trip, TensorMeta serialization, and the
per-tensor mode ZMQ protocol.
"""
import pickle
from typing import List, Tuple

import torch

from trinity.common.weight_transfer.base import (
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_weights(shapes_dtypes: List[Tuple[Tuple[int, ...], torch.dtype]]):
    """Create a list of (name, tensor) pairs."""
    return [
        (f"layer.{i}.weight", torch.randn(shape, dtype=dtype))
        for i, (shape, dtype) in enumerate(shapes_dtypes)
    ]


# ---------------------------------------------------------------------------
# TensorMeta tests
# ---------------------------------------------------------------------------


class TestTensorMeta:
    def test_create(self):
        meta = TensorMeta(
            name="fc.weight",
            shape=torch.Size([128, 64]),
            dtype=torch.float32,
            chunk_offset=0,
            chunk_size=32768,
            offset=0,
        )
        assert meta.name == "fc.weight"
        assert meta.shape == torch.Size([128, 64])
        assert meta.dtype == torch.float32
        assert meta.chunk_size == 32768

    def test_pickle_roundtrip(self):
        """TensorMeta must survive pickle (used by ZMQ send_pyobj)."""
        meta = TensorMeta(
            name="attn.qkv",
            shape=torch.Size([4096, 4096]),
            dtype=torch.bfloat16,
            chunk_offset=1024,
            chunk_size=2048,
            offset=512,
        )
        restored = pickle.loads(pickle.dumps(meta))
        assert restored.name == meta.name
        assert restored.shape == meta.shape
        assert restored.dtype == meta.dtype
        assert restored.chunk_offset == meta.chunk_offset
        assert restored.chunk_size == meta.chunk_size
        assert restored.offset == meta.offset


# ---------------------------------------------------------------------------
# split_weight_chunks tests
# ---------------------------------------------------------------------------


class TestSplitWeightChunks:
    def test_small_tensor_single_chunk(self):
        """A tensor smaller than bucket_size should produce exactly one chunk."""
        weights = _make_weights([((32, 16), torch.float32)])
        bucket_size = 1024 * 1024  # 1MB — much larger than 32*16*4=2048 bytes

        chunks = list(split_weight_chunks(weights, bucket_size))
        assert len(chunks) == 1
        meta, chunk = chunks[0]
        assert meta.name == "layer.0.weight"
        assert meta.chunk_offset == 0
        assert meta.chunk_size == 32 * 16 * 4  # float32 = 4 bytes
        assert chunk.dtype == torch.uint8

    def test_large_tensor_multiple_chunks(self):
        """A tensor larger than bucket_size should be split into multiple chunks."""
        # 256 * 256 * 4 bytes = 262144 bytes
        weights = _make_weights([((256, 256), torch.float32)])
        bucket_size = 100_000  # < 262144, so should get 3 chunks

        chunks = list(split_weight_chunks(weights, bucket_size))
        assert len(chunks) == 3  # ceil(262144 / 100000) = 3

        total_bytes = sum(meta.chunk_size for meta, _ in chunks)
        assert total_bytes == 256 * 256 * 4

        # Verify chunk_offsets are sequential.
        offset = 0
        for meta, _ in chunks:
            assert meta.chunk_offset == offset
            offset += meta.chunk_size

    def test_multiple_tensors(self):
        """Multiple tensors should each produce their own chunks."""
        weights = _make_weights(
            [
                ((8, 8), torch.float32),  # 256 bytes
                ((16, 16), torch.float32),  # 1024 bytes
            ]
        )
        bucket_size = 1024 * 1024

        chunks = list(split_weight_chunks(weights, bucket_size))
        assert len(chunks) == 2  # one chunk per tensor
        assert chunks[0][0].name == "layer.0.weight"
        assert chunks[1][0].name == "layer.1.weight"

    def test_bf16_dtype(self):
        """bfloat16 tensors should be chunked by their actual byte size."""
        weights = _make_weights([((64, 64), torch.bfloat16)])
        bucket_size = 1024 * 1024

        chunks = list(split_weight_chunks(weights, bucket_size))
        assert len(chunks) == 1
        meta, chunk = chunks[0]
        assert meta.chunk_size == 64 * 64 * 2  # bf16 = 2 bytes


# ---------------------------------------------------------------------------
# merge_weight_chunks tests (round-trip with split)
# ---------------------------------------------------------------------------


class TestMergeWeightChunks:
    def test_roundtrip_small(self):
        """Small tensors should survive a split→merge round-trip exactly."""
        original = _make_weights(
            [
                ((32, 16), torch.float32),
                ((64, 32), torch.float32),
            ]
        )
        bucket_size = 1024 * 1024

        merged = list(merge_weight_chunks(split_weight_chunks(original, bucket_size), bucket_size))
        assert len(merged) == len(original)

        for (orig_name, orig_tensor), (merged_name, merged_tensor) in zip(original, merged):
            assert merged_name == orig_name
            assert torch.equal(merged_tensor, orig_tensor)

    def test_roundtrip_large_tensor(self):
        """A tensor split across multiple buckets should merge back exactly."""
        original = _make_weights([((256, 256), torch.float32)])
        bucket_size = 100_000  # Forces 3 chunks

        merged = list(merge_weight_chunks(split_weight_chunks(original, bucket_size), bucket_size))
        assert len(merged) == 1
        assert merged[0][0] == "layer.0.weight"
        assert torch.equal(merged[0][1], original[0][1])

    def test_roundtrip_mixed_sizes(self):
        """A mix of small and large tensors should all survive round-trip."""
        original = _make_weights(
            [
                ((4, 4), torch.float32),  # 64 bytes — fits in one bucket
                ((256, 256), torch.float32),  # 262144 bytes — split into chunks
                ((8, 8), torch.bfloat16),  # 128 bytes — fits in one bucket
            ]
        )
        bucket_size = 100_000

        merged = list(merge_weight_chunks(split_weight_chunks(original, bucket_size), bucket_size))
        assert len(merged) == len(original)

        for (orig_name, orig_tensor), (merged_name, merged_tensor) in zip(original, merged):
            assert merged_name == orig_name
            assert torch.equal(merged_tensor, orig_tensor)

    def test_roundtrip_exact_bucket_size(self):
        """A tensor whose size exactly equals bucket_size should work."""
        # 100 * 250 * 4 = 100000 bytes = bucket_size
        original = _make_weights([((100, 250), torch.float32)])
        bucket_size = 100_000

        merged = list(merge_weight_chunks(split_weight_chunks(original, bucket_size), bucket_size))
        assert len(merged) == 1
        assert torch.equal(merged[0][1], original[0][1])
