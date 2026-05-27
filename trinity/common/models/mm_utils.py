"""Utilities for processing multi-modal data (images/videos) for specific vision-language models.

Main capabilities:
- Parse prompts with media tags (`<image>` / `<video>`)
- Detect multi-modal content in conversation messages
- Normalize legacy transformers-style media parts to vLLM/OpenAI-style schema
- Build training-time multimodal inputs with model processor
- Keep client-side parsing behavior aligned with vLLM server parsing

Provides functions to:
1. Parse prompts with media tags (<image>/<video>)
2. Validate multi-modal content in conversations
3. Normalize and preprocess media inputs for inference/training
4. Construct vLLM-compatible message formats

Note:
    Processor availability is detected from local model assets
    (e.g., processor/preprocessor config files). The implementation does not
    rely on model-class-name matching.

Compatibility:
    `MultiModalRender` normalizes legacy transformers-style message parts
    (e.g., type=image with url/path/base64) into vLLM/OpenAI-style part schema
    before calling vLLM `parse_chat_messages`.
"""
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import transformers
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormat,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.inputs import MultiModalDataDict

from trinity.utils.log import get_logger

_MM_TYPE_TO_URL_FIELD = {
    "image": "image_url",
    "video": "video_url",
    "audio": "audio_url",
}

# Field order follows legacy transformers multimodal conventions in
# processing_utils.apply_chat_template.
_LEGACY_MM_VALUE_KEYS = {
    "image": ("image", "url", "path", "base64", "image_url"),
    "video": ("video", "url", "path", "video_url"),
    "audio": ("audio", "url", "path", "audio_url"),
}


def should_use_processor(model_path: str) -> bool:
    p = Path(model_path)
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        return False

    processor_files = (
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    return any((p / name).is_file() for name in processor_files)


def processor_or_tokenizer_cls(model_path: str) -> Any:
    if should_use_processor(model_path):
        return transformers.AutoProcessor
    return transformers.AutoTokenizer


def build_mm_message(
    prompt: str, images: List[Union[str, Any]], videos: List[Union[str, Any]]
) -> Dict[str, Any]:
    """Construct multi-modal message by injecting media references at tag positions in prompt.

    Parses prompt for <image>/<video> tags, replaces them with corresponding media references,
    and handles surplus media items. Extra media (beyond tag count) is prepended to content.

    Args:
        prompt: Text containing optional <image> and <video> tags as media placeholders.
                Example: "First <image> then <video> and finally <image>"
        images: List of image references (file paths, URLs, or PIL images) in order of appearance.
        videos: List of video references (file paths, URLs) in order of appearance.

    Returns:
        Message dictionary formatted for VL models:
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ...},  # Surplus media first
                {"type": "video", "video": ...},
                {"type": "text", "text": "First "},
                {"type": "image", "image": ...},  # Tag-replaced media
                ...
            ]
        }

    Raises:
        ValueError: If prompt contains more <image> tags than provided images,
                    or more <video> tags than provided videos.

    Behavior details:
        - Tags are case-sensitive and must be exact: "<image>", "<video>"
        - Empty text segments between tags are omitted
        - Surplus media (images/videos beyond tag count) appears at START of content list
        - Text segments preserve original prompt ordering around tags
    """
    content_list = []
    segments = re.split(r"(<image>|<video>)", prompt)
    img_idx, vid_idx = 0, 0
    for segment in segments:
        if segment == "<image>":
            if img_idx >= len(images):
                raise ValueError("More <image> tags in prompt than images provided.")
            content_list.append({"type": "image", "image": images[img_idx]})
            img_idx += 1
        elif segment == "<video>":
            if vid_idx >= len(videos):
                raise ValueError("More <video> tags in prompt than videos provided.")
            content_list.append({"type": "video", "video": videos[vid_idx]})
            vid_idx += 1
        elif len(segment) == 0:
            continue
        else:
            content_list.append({"type": "text", "text": segment})

    # Prepend surplus media items (not referenced by tags)
    surplus_content = []
    while img_idx < len(images):
        surplus_content.append({"type": "image", "image": images[img_idx]})
        img_idx += 1
    while vid_idx < len(videos):
        surplus_content.append({"type": "video", "video": videos[vid_idx]})
        vid_idx += 1

    content_list = surplus_content + content_list
    if len(content_list) == 1 and content_list[0]["type"] == "text":
        return {"role": "user", "content": content_list[0]["text"]}
    return {"role": "user", "content": content_list}


def has_multi_modal_content(messages: List[Dict]) -> bool:
    """Check if any message contains non-text (image/video) content.

    Inspects message content structure to detect multi-modal elements. Handles both:
    - String content (text-only, returns False)
    - List content (multi-modal candidates)

    Args:
        messages: List of conversation messages. Each message must contain a "content" field.
                  Content may be:
                  - str: Plain text message
                  - List[Dict]: Multi-modal content items (each with "type" key)

    Returns:
        True if any message contains at least one non-text content item (type != "text"),
        False otherwise.

    Example:
        >>> msg = [{"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "image", "image": "..."}]}]
        >>> has_multi_modal_content(msg)
        True
    """
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type", "text") != "text":
                    return True
    return False


class MultiModalRender(ABC):
    """
    Client-side processor that mirrors server's multimodal handling.
    """

    def __init__(self, model_path: str, *args, **kwargs):
        pass

    @abstractmethod
    def build_mm_input_for_training(
        self,
        *,
        messages: List[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        input_ids: List[List[int]] = None,
        multi_modal_data: Dict[str, List] = None,
    ) -> Dict[str, Any] | None:
        pass

    @abstractmethod
    def process_messages(
        self,
        messages: list[dict[str, Any]],
        *args,
        **kwargs,
    ):
        pass

    @abstractmethod
    async def process_messages_async(
        self,
        messages: list[dict[str, Any]],
        *args,
        **kwargs,
    ):
        pass


class vLLMMultiModalRender(MultiModalRender):
    """
    Client-side processor that mirrors vLLM server's multimodal handling.

    This class enables RL training endpoints to extract multimodal data
    with identical processing to the inference server, ensuring consistency.
    Legacy transformers-style content parts are normalized to vLLM-compatible
    OpenAI part schema before parsing.
    """

    def __init__(
        self,
        model_path: str,
        *,
        media_io_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        allowed_local_media_path: str = "",
        allowed_media_domains: Optional[list[str]] = None,
        trust_request_chat_template: bool = False,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the client-side multimodal processor.

        Args:
            model_path: Path to the model
            media_io_kwargs: Media I/O configuration (mirrors --media-io-kwargs)
            allowed_local_media_path: Path to allowed local media directory
            allowed_media_domains: List of allowed media domains
            trust_request_chat_template: Whether to trust request-provided chat template
            mm_processor_kwargs: Additional processor kwargs for multimodal processing
        """
        self.logger = get_logger(__name__)

        self.model_path = model_path
        self.media_io_kwargs = media_io_kwargs or {}
        self.mm_processor_kwargs = mm_processor_kwargs or {}

        # Initialize ModelConfig
        self.model_config = ModelConfig(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="auto",
            trust_remote_code=True,
        )

        # Initialize multimodal processor if the model supports it
        self.mm_processor = None
        if should_use_processor(model_path):
            self.mm_processor = transformers.AutoProcessor.from_pretrained(
                self.model_config.model,
                trust_remote_code=self.model_config.trust_remote_code,
            )

        # Store media connector configuration
        self._media_connector_config = {
            "media_io_kwargs": self.media_io_kwargs,
            "allowed_local_media_path": allowed_local_media_path,
            "allowed_media_domains": allowed_media_domains or [],
        }

        self.trust_request_chat_template = trust_request_chat_template

    def build_mm_input_for_training(
        self,
        *,
        messages: List[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        input_ids: List[List[int]] = None,
        multi_modal_data: Dict[str, List] = None,
    ) -> Dict[str, Any] | None:
        """Build multimodal tensors aligned with training token sequence.

        Args:
            messages: Optional chat messages used to derive token ids and/or
                multimodal payloads when not provided explicitly.
            tools: Optional tool schema list passed to
                `mm_processor.apply_chat_template` when tokenizing messages.
            input_ids: Optional token ids (shape `(seq_len,)` or
                `(batch_size, seq_len)`). When omitted, token ids are generated
                from `messages`.
            multi_modal_data: Optional multimodal payload returned by
                `process_messages` (e.g., image/video media items). When omitted,
                it is inferred from `messages`.

        Returns:
            A dictionary containing multimodal training tensors, including
            `mm_token_type_ids` and media-processor outputs (for example image
            and/or video tensors), or `None` when multimodal processing is not
            available or no multimodal content exists.

        Raises:
            ValueError: If both `input_ids` and `messages` are missing.

        Notes:
            - Requires an initialized multimodal processor (`self.mm_processor`).
              If unavailable for the current model, returns `None`.
            - 1D `input_ids` are reshaped to `(1, seq_len)` to satisfy processor
              expectations.
        """
        if self.mm_processor is None:
            return None

        if input_ids is None:
            if messages is None:
                raise ValueError("Either `messages` or `input_ids` must be provided.")
            normalized_messages = self._normalize_messages_for_vllm(messages)
            input_ids = self.mm_processor.apply_chat_template(
                normalized_messages, tools=tools, tokenize=True
            )
        input_ids = np.array(input_ids)
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)

        if multi_modal_data is None:
            if messages is not None:
                _, multi_modal_data = self.process_messages(messages)
            if multi_modal_data is None:
                return None

        def _normalize_media_item(item: Any, *, modality: str) -> Any:
            # vLLM may return wrapper objects with `.media` for images,
            # and tuples like (video_data, video_meta) for videos.
            if hasattr(item, "media"):
                return item.media
            if modality == "video" and isinstance(item, tuple) and len(item) >= 1:
                return item[0]
            return item

        multi_modal_inputs = {
            "mm_token_type_ids": torch.tensor(
                self.mm_processor.create_mm_token_type_ids(input_ids), dtype=torch.int
            )
        }
        if images := multi_modal_data.get("image", None):
            images = [_normalize_media_item(img, modality="image") for img in images]
            image_inputs = self.mm_processor.image_processor(images=images, return_tensors="pt")
            multi_modal_inputs.update(image_inputs)
        if videos := multi_modal_data.get("video", None):
            videos = [_normalize_media_item(vid, modality="video") for vid in videos]
            video_inputs = self.mm_processor.video_processor(videos=videos, return_tensors="pt")
            multi_modal_inputs.update(video_inputs)
        return multi_modal_inputs

    def process_messages(
        self,
        messages: list[dict[str, Any]],
        content_format: ChatTemplateContentFormat = "string",
    ) -> tuple[list[ConversationMessage], Optional[MultiModalDataDict],]:
        """
        Process chat messages and extract multimodal data.

        This replicates the server-side parse_chat_messages behavior.

        Args:
            messages: List of chat messages with potential multimodal content
            content_format: Chat template content format ("string" or "openai")

        Returns:
            Tuple of (conversation, mm_data) matching server output
        """
        normalized_messages = self._normalize_messages_for_vllm(messages)

        conversation, mm_data, _ = parse_chat_messages(
            messages=normalized_messages,
            model_config=self.model_config,
            content_format=content_format,
            media_io_kwargs=self._media_connector_config["media_io_kwargs"],
            mm_processor_kwargs=self.mm_processor_kwargs,
        )

        return conversation, mm_data

    async def process_messages_async(
        self,
        messages: list[dict[str, Any]],
        content_format: ChatTemplateContentFormat = "string",
    ) -> tuple[list[ConversationMessage], Optional[MultiModalDataDict],]:
        """
        Async version of process_messages for concurrent media fetching.
        """
        normalized_messages = self._normalize_messages_for_vllm(messages)

        conversation, mm_data, _ = await parse_chat_messages_async(
            messages=normalized_messages,
            model_config=self.model_config,
            content_format=content_format,
            media_io_kwargs=self._media_connector_config["media_io_kwargs"],
            mm_processor_kwargs=self.mm_processor_kwargs,
        )

        return conversation, mm_data

    def _normalize_messages_for_vllm(  # noqa: C901
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize legacy multimodal content parts for vLLM parser.

        vLLM `parse_chat_messages` accepts OpenAI-style parts (image_url,
        video_url, audio_url, image_pil, etc.). This function rewrites common
        legacy transformers-style variants into that schema.

        It also normalizes tool call arguments for tokenizer compatibility:
        OpenAI-style payloads usually store function.arguments as JSON string,
        while some chat templates expect a mapping when rendering tool calls.
        """

        def _infer_modality(part: dict[str, Any]) -> Optional[str]:
            part_type = part.get("type")
            if part_type in _LEGACY_MM_VALUE_KEYS:
                return part_type
            for modality, keys in _LEGACY_MM_VALUE_KEYS.items():
                if any(key in part for key in keys):
                    return modality
            return None

        def _extract_media_value(part: dict[str, Any], modality: str) -> tuple[Any, Optional[str]]:
            for key in _LEGACY_MM_VALUE_KEYS[modality]:
                if key not in part or part.get(key) is None:
                    continue
                value = part.get(key)
                if key == _MM_TYPE_TO_URL_FIELD[modality] and isinstance(value, dict):
                    return value.get("url"), key
                return value, key
            return None, None

        def _to_vllm_part(part: dict[str, Any], modality: str) -> dict[str, Any]:
            media_value, source_key = _extract_media_value(part, modality)
            if isinstance(media_value, str):
                # Preserve existing data URLs; upgrade raw image base64 payload.
                if (
                    modality == "image"
                    and source_key == "base64"
                    and not media_value.startswith("data:")
                ):
                    media_value = f"data:image/png;base64,{media_value}"
                url_field = _MM_TYPE_TO_URL_FIELD[modality]
                return {"type": url_field, url_field: {"url": media_value}}

            if modality == "image" and media_value is not None:
                # vLLM can parse PIL-like object through image_pil content part.
                return {"type": "image_pil", "image_pil": media_value}

            return dict(part)

        def _to_builtin(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _to_builtin(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_builtin(v) for v in value]
            if isinstance(value, tuple):
                return [_to_builtin(v) for v in value]

            # OpenAI SDK / Pydantic models
            if hasattr(value, "model_dump"):
                try:
                    return _to_builtin(value.model_dump())
                except Exception:
                    pass
            if hasattr(value, "dict"):
                try:
                    return _to_builtin(value.dict())
                except Exception:
                    pass

            return value

        def _normalize_tool_calls(tool_calls: Any) -> Any:
            if not isinstance(tool_calls, list):
                return tool_calls

            normalized_tool_calls = []
            for tool_call in tool_calls:
                tool_call = _to_builtin(tool_call)
                if not isinstance(tool_call, dict):
                    normalized_tool_calls.append(tool_call)
                    continue

                normalized_tool_call = dict(tool_call)
                function = normalized_tool_call.get("function")
                if isinstance(function, dict):
                    normalized_function = dict(function)
                    arguments = normalized_function.get("arguments")
                    if isinstance(arguments, str):
                        stripped = arguments.strip()
                        if stripped == "":
                            normalized_function["arguments"] = {}
                        else:
                            try:
                                parsed = json.loads(stripped)
                                if isinstance(parsed, dict):
                                    normalized_function["arguments"] = parsed
                            except json.JSONDecodeError:
                                # Keep original arguments string for invalid JSON.
                                pass
                    normalized_tool_call["function"] = normalized_function

                normalized_tool_calls.append(normalized_tool_call)

            return normalized_tool_calls

        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            message = _to_builtin(message)
            if not isinstance(message, dict):
                self.logger.warning(
                    "Unexpected message type for vLLM normalization: %s. Falling back to text content.",
                    type(message),
                )
                normalized_messages.append({"role": "user", "content": str(message)})
                continue

            normalized_message = dict(message)
            if "tool_calls" in normalized_message:
                normalized_message["tool_calls"] = _normalize_tool_calls(
                    normalized_message.get("tool_calls")
                )

            content = normalized_message.get("content")
            if not isinstance(content, list):
                normalized_messages.append(normalized_message)
                continue

            normalized_content = []
            for part in content:
                part = _to_builtin(part)
                if not isinstance(part, dict):
                    normalized_content.append(part)
                    continue

                modality = _infer_modality(part)
                if modality is not None:
                    normalized_content.append(_to_vllm_part(part, modality))
                    continue

                normalized_content.append(dict(part))

            normalized_message["content"] = normalized_content
            normalized_messages.append(normalized_message)

        return normalized_messages
