"""Utilities for processing multi-modal data (images/videos) for specific vision-language models.

Supported models:
- Qwen2.5-VL, Qwen3-VL series
- Kimi VL series

Provides functions to:
1. Parse prompts with media tags (<image>/<video>)
2. Validate multi-modal content in conversations
3. Preprocess media inputs for inference/training
4. Construct model-compatible message formats

Note:
    Only processors with class names containing both ("Qwen" OR "Kimi") AND "VLProcessor" are supported.
    Relies on `qwen_vl_utils.process_vision_info` for media extraction.
"""
import re
from typing import Any, Dict, List, Union


def build_multi_modal_data(
    processor: Any,
    messages: List[Dict],
) -> Dict[str, Any]:
    """Extract and preprocess vision inputs from multi-modal messages for vLLM inference.

    Processes messages containing image/video placeholders using model-specific vision utilities.
    Returns structured media inputs compatible with vLLM's multi-modal API.

    Args:
        processor: Vision-language processor instance (must have class name containing
                   ("Qwen" OR "Kimi") AND "VLProcessor").
        messages: List of conversation messages in model-expected format. Each message's "content"
                  may be a string or list of content items (text/image/video dictionaries).

    Returns:
        Dictionary containing processed media inputs with keys:
        - "image": List of processed image objects (if images exist)
        - "video": List of processed video objects (if videos exist)
        Keys are omitted when no corresponding media is present.

    Raises:
        NotImplementedError: If processor class name doesn't match supported patterns.
        ImportError: If required `qwen_vl_utils` module is unavailable.

    Example:
        >>> messages = [{"role": "user", "content": [{"type": "image", "image": "img.jpg"}, {"type": "text", "text": "Describe this"}]}]
        >>> build_multi_modal_data(processor, messages)
        {"image": [processed_image]}
    """
    processor_class_name = processor.__class__.__name__
    if (
        "Qwen" in processor_class_name or "Kimi" in processor_class_name
    ) and "VLProcessor" in processor_class_name:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        multi_modal_data = {}
        if image_inputs:
            multi_modal_data["image"] = image_inputs
        if video_inputs:
            multi_modal_data["video"] = video_inputs

        return multi_modal_data
    raise NotImplementedError(
        f"Processor '{processor_class_name}' not supported. Only Qwen/Kimi VL processors are supported."
    )


def build_mm_input_for_training(
    processor: Any, prompt: str, multi_modal_data: Dict[str, List]
) -> Dict[str, Any]:
    """Tokenize prompt and integrate processed media inputs for model training.

    Combines text prompt with preprocessed image/video data into model-ready tensor inputs.
    Handles padding and tensor conversion for training workflows.

    Args:
        processor: Vision-language processor instance (must have class name containing
                   ("Qwen" OR "Kimi") AND "VLProcessor").
        prompt: Plain text prompt WITHOUT media tags (e.g., "Describe this image").
                Media placement is handled via `multi_modal_data`, not prompt tags.
        multi_modal_data: Dictionary from `build_multi_modal_data()` containing:
                          {"image": [...], "video": [...]} (keys optional)

    Returns:
        Dictionary of model inputs including:
        - input_ids: Tokenized prompt IDs
        - attention_mask: Attention mask tensor
        - pixel_values: Processed image tensors (if images provided)
        - pixel_values_videos: Processed video tensors (if videos provided)
        All tensors converted to PyTorch format (`return_tensors="pt"`).

    Raises:
        NotImplementedError: If processor class name doesn't match supported patterns.
        ValueError: If media counts mismatch prompt expectations (handled internally by processor).

    Note:
        Prompt should NOT contain <image>/<video> tags here. Media association is managed
        through the structured `multi_modal_data` dictionary.
    """
    processor_class_name = processor.__class__.__name__
    if (
        "Qwen" in processor_class_name or "Kimi" in processor_class_name
    ) and "VLProcessor" in processor_class_name:
        inputs = processor(
            text=[prompt],
            images=multi_modal_data.get("image", None),
            videos=multi_modal_data.get("video", None),
            padding=True,
            return_tensors="pt",
        )
        return dict(inputs)
    raise NotImplementedError(
        f"Processor '{processor_class_name}' not supported. Only Qwen/Kimi VL processors are supported."
    )


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
