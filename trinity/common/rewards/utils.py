from typing import Any, Dict, List


def to_rm_gallery_messages(messages: List[Dict[str, Any]]) -> Any:
    """Deprecated: was used by the removed RMGalleryFn.

    Converts a list of ``{"role": ..., "content": ...}`` dicts to
    rm_gallery ChatMessage objects.  Kept for any external callers that
    may still reference this helper; remove once confirmed unused.
    """
    from rm_gallery.core.model.message import (  # pyright: ignore[reportMissingImports]
        ChatMessage,
        MessageRole,
    )

    role_map = {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
    }

    return [ChatMessage(role=role_map[msg["role"]], content=msg["content"]) for msg in messages]
