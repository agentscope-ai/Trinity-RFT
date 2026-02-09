""""Multi-modal utilities for processing and handling multi-modal data such as images and videos.
Only support Qwen2.5/3 VL series.

Modified from: verl/utils/dataset/rl_dataset.py
"""
import re
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from trinity.utils.annotations import Deprecated


def build_multi_modal_data(
    processor: Any,
    messages: List[Dict],
) -> Dict[str, Any]:
    """
    Preprocess multi-modal data and build multi-modal inputs
    """
    processor_class_name = processor.__class__.__name__
    if "Qwen" in processor_class_name and "VLProcessor" in processor_class_name:
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
    if "Qwen" in processor_class_name and "VLProcessor" in processor_class_name:
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
            content_list.append({"type": "image", "image": images[img_idx]})
            img_idx += 1
        elif segment == "<video>":
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


@Deprecated
def build_multi_modal_inputs(
    prompt: str,
    images: List[Image.Image],
    videos: List[np.ndarray],
    processor: Any,
) -> Dict[str, Any]:
    """
    Preprocess multi-modal data and build multi-modal inputs
    """
    if prompt is None:
        raise ValueError("Prompt is required for build multi-modal inputs")

    multi_modal_data = {}
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos

    model_inputs = processor(
        text=[prompt],
        images=multi_modal_data.get("image", None),
        videos=multi_modal_data.get("video", None),
        return_tensors="pt",
    )

    input_ids = model_inputs.pop("input_ids")[0]
    model_inputs.pop("attention_mask")

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    return {
        "prompt": prompt,
        "prompt_token_ids": input_ids,
        "multi_modal_data": multi_modal_data,
        "multi_modal_inputs": dict(model_inputs),
    }


@Deprecated
def convert_messages_to_mm_format(messages: List[Dict]) -> List[Dict]:
    for message in messages:
        content = message["content"]
        content_list = []
        segments = re.split("(<image>|<video>)", content)
        segments = [item for item in segments if item != ""]
        for segment in segments:
            if segment == "<image>":
                content_list.append(
                    {"type": "image"}
                )  # chat template will fill the actual image data later
            elif segment == "<video>":
                content_list.append(
                    {"type": "video"}
                )  # chat template will fill the actual video data later
            elif len(segment) == 0:
                continue
            else:
                content_list.append({"type": "text", "text": segment})

        message["content"] = content_list
    return messages
