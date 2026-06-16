# -*- coding: utf-8 -*-
"""Tests for the streaming safetensors writer."""

import json
import os
import struct
import tempfile
import unittest

import torch
from safetensors.torch import load_file

from trinity.utils.stream_saver import (
    DTYPE_TO_SAFETENSORS,
    StateDictMeta,
    TensorMeta,
    _compute_exact_header_size,
    save_safetensors_streaming,
)


def _meta_from_tensors(tensors: dict[str, torch.Tensor]) -> StateDictMeta:
    """Build StateDictMeta from a dict of tensors (for test convenience)."""
    meta = []
    for name, tensor in tensors.items():
        dtype_str = DTYPE_TO_SAFETENSORS.get(tensor.dtype)
        if dtype_str is not None:
            meta.append(TensorMeta(name, dtype_str, list(tensor.shape)))
        del tensor
    return meta


class TestSaveSafetensorsStreaming(unittest.TestCase):
    """Unit tests for :func:`save_safetensors_streaming`."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_basic_roundtrip(self):
        """Write several tensors and verify they load back identically."""
        tensors = {
            "layer.0.weight": torch.randn(64, 64, dtype=torch.float32),
            "layer.0.bias": torch.randn(64, dtype=torch.float32),
            "layer.1.weight": torch.randn(32, 64, dtype=torch.float32),
        }
        filepath = os.path.join(self.tmpdir, "model.safetensors")

        save_safetensors_streaming(
            tensors.items(), filepath, state_dict_meta=_meta_from_tensors(tensors)
        )

        loaded = load_file(filepath)
        self.assertEqual(set(loaded.keys()), set(tensors.keys()))
        for name, expected in tensors.items():
            torch.testing.assert_close(loaded[name], expected)

    def test_bfloat16_roundtrip(self):
        """bf16 tensors survive the streaming write."""
        tensors = {
            "w1": torch.randn(128, 128, dtype=torch.bfloat16),
            "w2": torch.randn(128, dtype=torch.bfloat16),
        }
        filepath = os.path.join(self.tmpdir, "bf16.safetensors")

        save_safetensors_streaming(
            tensors.items(), filepath, state_dict_meta=_meta_from_tensors(tensors)
        )

        loaded = load_file(filepath)
        for name, expected in tensors.items():
            torch.testing.assert_close(loaded[name], expected)

    def test_mixed_dtypes(self):
        """Different dtypes in the same file."""
        tensors = {
            "fp32": torch.randn(16, dtype=torch.float32),
            "fp16": torch.randn(16, dtype=torch.float16),
            "bf16": torch.randn(16, dtype=torch.bfloat16),
            "int8": torch.randint(-128, 127, (16,), dtype=torch.int8),
            "int32": torch.randint(0, 100, (16,), dtype=torch.int32),
        }
        filepath = os.path.join(self.tmpdir, "mixed.safetensors")

        save_safetensors_streaming(
            tensors.items(), filepath, state_dict_meta=_meta_from_tensors(tensors)
        )

        loaded = load_file(filepath)
        self.assertEqual(set(loaded.keys()), set(tensors.keys()))
        for name, expected in tensors.items():
            torch.testing.assert_close(loaded[name], expected)

    def test_empty_tensor(self):
        """A tensor with zero elements should still round-trip."""
        tensors = {
            "empty": torch.randn(0, 64, dtype=torch.float32),
            "nonempty": torch.randn(4, 64, dtype=torch.float32),
        }
        filepath = os.path.join(self.tmpdir, "empty.safetensors")

        save_safetensors_streaming(
            tensors.items(), filepath, state_dict_meta=_meta_from_tensors(tensors)
        )

        loaded = load_file(filepath)
        self.assertEqual(loaded["empty"].shape, (0, 64))
        torch.testing.assert_close(loaded["nonempty"], tensors["nonempty"])

    def test_single_tensor(self):
        """Edge case: a file with exactly one tensor."""
        tensors = {"only": torch.randn(1024, dtype=torch.float32)}
        filepath = os.path.join(self.tmpdir, "single.safetensors")

        save_safetensors_streaming(
            tensors.items(), filepath, state_dict_meta=_meta_from_tensors(tensors)
        )

        loaded = load_file(filepath)
        torch.testing.assert_close(loaded["only"], tensors["only"])

    # ------------------------------------------------------------------
    # Generator semantics
    # ------------------------------------------------------------------

    def test_generator_input(self):
        """Accepts a lazy generator, not just a dict.items()."""
        filepath = os.path.join(self.tmpdir, "gen.safetensors")

        # We need to capture the generated tensors for comparison.
        reference = {}
        # Pre-build meta so the streaming writer can size the header exactly.
        meta: StateDictMeta = [(f"param_{i}", "BF16", [32, 32]) for i in range(5)]

        def gen_with_capture():
            for i in range(5):
                t = torch.randn(32, 32, dtype=torch.bfloat16)
                name = f"param_{i}"
                reference[name] = t.clone()
                yield name, t

        save_safetensors_streaming(gen_with_capture(), filepath, state_dict_meta=meta)

        loaded = load_file(filepath)
        self.assertEqual(set(loaded.keys()), set(reference.keys()))
        for name in reference:
            torch.testing.assert_close(loaded[name], reference[name])

    # ------------------------------------------------------------------
    # rename=False (hybrid async mode)
    # ------------------------------------------------------------------

    def test_rename_false_leaves_tmp(self):
        """With rename=False, the .tmp file is left for the caller."""
        filepath = os.path.join(self.tmpdir, "model.safetensors")
        tensors = {"w": torch.randn(8, 8)}

        result = save_safetensors_streaming(
            tensors.items(),
            filepath,
            state_dict_meta=_meta_from_tensors(tensors),
            rename=False,
        )

        self.assertEqual(result, filepath + ".tmp")
        self.assertTrue(os.path.exists(filepath + ".tmp"))
        self.assertFalse(os.path.exists(filepath))

        # The .tmp file should be a valid safetensors file.
        loaded = load_file(filepath + ".tmp")
        torch.testing.assert_close(loaded["w"], tensors["w"])

    # ------------------------------------------------------------------
    # Exact header sizing via state_dict_meta
    # ------------------------------------------------------------------

    def test_exact_header_sizing(self):
        """With exact meta, a large number of tensors is sized precisely."""
        tensors = {f"very_long_tensor_name_{i:04d}": torch.randn(4) for i in range(500)}
        meta = _meta_from_tensors(tensors)
        filepath = os.path.join(self.tmpdir, "exact_header.safetensors")

        save_safetensors_streaming(tensors.items(), filepath, state_dict_meta=meta)

        loaded = load_file(filepath)
        self.assertEqual(set(loaded.keys()), set(tensors.keys()))

        # Verify header was sized exactly — no padding beyond alignment.
        exact_size = _compute_exact_header_size(meta)
        with open(filepath, "rb") as f:
            actual_header_size = struct.unpack("<Q", f.read(8))[0]
        self.assertEqual(actual_header_size, exact_size)

    def test_meta_mixed_dtypes(self):
        """state_dict_meta works with mixed dtypes."""
        tensors = {
            "fp32": torch.randn(16, dtype=torch.float32),
            "bf16": torch.randn(16, dtype=torch.bfloat16),
            "fp16": torch.randn(16, dtype=torch.float16),
        }
        meta: StateDictMeta = [
            ("fp32", "F32", [16]),
            ("bf16", "BF16", [16]),
            ("fp16", "F16", [16]),
        ]
        filepath = os.path.join(self.tmpdir, "meta_mixed.safetensors")

        save_safetensors_streaming(tensors.items(), filepath, state_dict_meta=meta)

        loaded = load_file(filepath)
        for name, expected in tensors.items():
            torch.testing.assert_close(loaded[name], expected)

    # ------------------------------------------------------------------
    # File format validation
    # ------------------------------------------------------------------

    def test_valid_safetensors_header(self):
        """The header contains valid JSON with correct dtype/shape/offsets."""
        tensors = {
            "a": torch.randn(4, 8, dtype=torch.float16),
            "b": torch.randn(16, dtype=torch.bfloat16),
        }
        filepath = os.path.join(self.tmpdir, "header.safetensors")

        save_safetensors_streaming(
            tensors.items(), filepath, state_dict_meta=_meta_from_tensors(tensors)
        )

        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size)

        header = json.loads(header_json)
        self.assertIn("a", header)
        self.assertIn("b", header)

        self.assertEqual(header["a"]["dtype"], "F16")
        self.assertEqual(header["a"]["shape"], [4, 8])

        self.assertEqual(header["b"]["dtype"], "BF16")
        self.assertEqual(header["b"]["shape"], [16])

        # Offsets should be contiguous.
        self.assertEqual(header["a"]["data_offsets"][0], 0)
        a_end = header["a"]["data_offsets"][1]
        self.assertEqual(a_end, 4 * 8 * 2)  # 4*8 elements * 2 bytes (F16)
        self.assertEqual(header["b"]["data_offsets"][0], a_end)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_unsupported_dtype_raises(self):
        """Unsupported dtypes should raise a clear error."""

        def gen():
            yield "bad", torch.randn(4, dtype=torch.complex64)

        filepath = os.path.join(self.tmpdir, "bad.safetensors")
        meta: StateDictMeta = [("bad", "CF64", [4])]  # dummy meta
        with self.assertRaises(ValueError):
            save_safetensors_streaming(gen(), filepath, state_dict_meta=meta)

    def test_missing_state_dict_meta_raises(self):
        """Calling without state_dict_meta should raise TypeError."""
        tensors = {"w": torch.randn(8, 8)}
        filepath = os.path.join(self.tmpdir, "no_meta.safetensors")
        with self.assertRaises(TypeError):
            save_safetensors_streaming(tensors.items(), filepath)  # type: ignore[call-arg]
