"""General utils"""

import hashlib
import json
import xml.etree.ElementTree as ET
from typing import Iterable, Optional, Tuple


def compute_stable_pack_seed(identities: Iterable[Tuple[int, int]]) -> int:
    """Compute a deterministic seed from a pack's dataset identities."""
    payload = json.dumps(sorted(identities), separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def extract_content_between_keys(
    response: str,
    key_start: str,
    key_end: str,
) -> Tuple[str, bool]:
    """Extract content in response between key_start and key_end.
    
    Strict success condition: 
        there is one and only one match for key_start / key_end, 
        and they should be in the correct order.

    Returns:
        content (str): extracted content, "null" if failed.
        success (bool): whether extraction is successful.

    TODO: consider requiring that key_end must appear at the end of response.
    """

    idx_start, idx_start_r = response.find(key_start), response.rfind(key_start)
    idx_end, idx_end_r = response.find(key_end), response.rfind(key_end)

    if (idx_start == -1) or (idx_start != idx_start_r) or (idx_end == -1) or (idx_end != idx_end_r):
        return "null", False
    
    if idx_start > idx_end:
        return "null", False
    
    return response[(idx_start + len(key_start)) : idx_end], True


def parse_xml_answer(
    response: str,
    list_tags: Optional[set[str]] = None,
) -> Tuple[Optional[dict], str]:
    """Parse one XML action wrapped in a unique ``<answer>`` block."""
    content, success = extract_content_between_keys(response, "<answer>", "</answer>")
    if not success:
        return None, "expected_exactly_one_answer_tag"

    def element_value(element: ET.Element):
        children = list(element)
        if not children:
            if element.attrib:
                return dict(element.attrib)
            return (element.text or "").strip()
        if (element.text or "").strip() or any(
            (child.tail or "").strip() for child in children
        ):
            raise ValueError("unexpected XML text")
        if element.tag in (list_tags or set()):
            return [element_value(child) for child in children]
        value = dict(element.attrib)
        for child in children:
            child_value = element_value(child)
            if child.tag in value:
                current = value[child.tag]
                value[child.tag] = (
                    current + [child_value]
                    if isinstance(current, list)
                    else [current, child_value]
                )
            else:
                value[child.tag] = child_value
        return value

    try:
        action = ET.fromstring(content)
        args = element_value(action)
        return {"action": action.tag, "args": {} if args == "" else args}, ""
    except (ET.ParseError, ValueError):
        return None, "invalid_answer_xml"
