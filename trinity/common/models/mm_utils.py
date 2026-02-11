""""Multi-modal utilities for processing and handling multi-modal data such as images and videos.
Only support Qwen2.5/3 VL series.

Modified from: verl/utils/dataset/rl_dataset.py
"""
import re
from typing import Any, Dict, List


def build_multi_modal_data(
    processor: Any,
    messages: List[Dict],
) -> Dict[str, Any]:
    """
    Preprocess multi-modal data and build multi-modal inputs
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
    raise NotImplementedError(f"{processor_class_name} not supported")


def build_mm_input_for_training(processor: Any, prompt: str, multi_modal_data: Dict) -> Dict:
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
    raise NotImplementedError(f"{processor_class_name} not supported")


def build_mm_message(prompt: str, images: List, videos: List):
    content_list = []
    segments = re.split("(<image>|<video>)", prompt)
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

    # deal with redundant <image> and <video>
    mm_contents = []
    while img_idx < len(images):
        mm_contents.append({"type": "image", "image": images[img_idx]})
        img_idx += 1
    while vid_idx < len(videos):
        mm_contents.append({"type": "video", "video": videos[vid_idx]})
        vid_idx += 1

    content_list = mm_contents + content_list
    message = {"role": "user", "content": content_list}
    return message


def has_multi_modal_content(messages: List[Dict]):
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            for item in content:
                if item.get("type", "text") != "text":
                    return True
    return False
