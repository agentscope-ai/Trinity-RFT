"""General utils"""

from typing import Tuple

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
