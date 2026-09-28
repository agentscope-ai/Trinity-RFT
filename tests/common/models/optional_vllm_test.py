"""CPU regressions for optional vLLM imports and parser delegation."""

import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

from trinity.common.models.mm_utils import vLLMMultiModalRender

_NO_VLLM = """
import builtins
original_import = builtins.__import__
def without_vllm(name, *args, **kwargs):
    if name == "vllm" or name.startswith("vllm."):
        raise ModuleNotFoundError("No module named 'vllm'", name="vllm")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_vllm
"""


class OptionalVLLMImportTest(unittest.TestCase):
    def run_without_vllm(self, script):
        result = subprocess.run(
            [sys.executable, "-c", _NO_VLLM + textwrap.dedent(script)],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_text_helpers_work_without_vllm(self):
        self.run_without_vllm(
            """
            from trinity.common.models.mm_utils import build_mm_message, has_multi_modal_content
            message = build_mm_message("hello", [], [])
            assert message == {"role": "user", "content": "hello"}
            assert not has_multi_modal_content([message])
            """
        )

    @unittest.skipUnless(importlib.util.find_spec("tinker"), "Tinker SDK is not installed")
    def test_tinker_model_imports_without_vllm(self):
        self.run_without_vllm(
            """
            from trinity.common.models.tinker_model import TinkerModel
            assert TinkerModel.__name__ == "TinkerModel"
            """
        )

    def test_vllm_renderer_reports_missing_optional_dependency(self):
        self.run_without_vllm(
            """
            from trinity.common.models.mm_utils import vLLMMultiModalRender
            try:
                vLLMMultiModalRender("unused-model-path")
            except ImportError as error:
                assert "vLLMMultiModalRender" in str(error)
                assert "'vllm' extra" in str(error)
                assert isinstance(error.__cause__, ModuleNotFoundError)
            else:
                raise AssertionError("Renderer must require vLLM")
            """
        )

    def test_renderer_preserves_unrelated_import_failures(self):
        original_import = __import__

        def broken_dependency(name, *args, **kwargs):
            if name == "vllm.config":
                raise ModuleNotFoundError(
                    "No module named 'some_dependency'", name="some_dependency"
                )
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=broken_dependency):
            with self.assertRaisesRegex(ModuleNotFoundError, "some_dependency"):
                vLLMMultiModalRender("unused-model-path")


class VLLMParserDelegationTest(unittest.IsolatedAsyncioTestCase):
    async def test_sync_and_async_parsers_receive_normalized_messages(self):
        # No model load or media download: preserve the renderer/parser boundary.
        renderer = vLLMMultiModalRender.__new__(vLLMMultiModalRender)
        renderer.model_config = object()
        renderer.mm_processor_kwargs = {"example_option": True}
        renderer._media_connector_config = {"media_io_kwargs": {"image": {"timeout": 3}}}
        messages = [
            {"role": "user", "content": [{"type": "image", "image": "https://example.org/img.png"}]}
        ]
        normalized = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.org/img.png"}}
                ],
            }
        ]
        conversation, media = [{"role": "user", "content": "image"}], {"image": object()}
        chat_utils = ModuleType("vllm.entrypoints.chat_utils")
        sync_parser = MagicMock(return_value=(conversation, media, None))
        async_parser = AsyncMock(return_value=(conversation, media, None))
        chat_utils.parse_chat_messages = sync_parser
        chat_utils.parse_chat_messages_async = async_parser
        with patch.dict(sys.modules, {"vllm.entrypoints.chat_utils": chat_utils}):
            self.assertEqual(
                renderer.process_messages(messages, content_format="openai"), (conversation, media)
            )
            self.assertEqual(
                await renderer.process_messages_async(messages, content_format="openai"),
                (conversation, media),
            )
        expected = {
            "messages": normalized,
            "model_config": renderer.model_config,
            "content_format": "openai",
            "media_io_kwargs": renderer._media_connector_config["media_io_kwargs"],
            "mm_processor_kwargs": renderer.mm_processor_kwargs,
        }
        sync_parser.assert_called_once_with(**expected)
        async_parser.assert_awaited_once_with(**expected)
        self.assertEqual(messages[0]["content"][0]["type"], "image")
