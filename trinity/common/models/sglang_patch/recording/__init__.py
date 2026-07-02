"""SGLang generation recording adapter.

Re-exports the SGLang-specific pieces (``build_sglang_experience``,
``create_sglang_recorder``, ``setup_sglang_recording``) and the engine-agnostic
core symbols shared with the vLLM recording path
(``trinity.common.models.recording``).
"""

from trinity.buffer.store import MemoryStore, RecordStore  # noqa: F401
from trinity.common.models.recording.context import (  # noqa: F401
    RecordingContext,
    RecordingIdentityMiddleware,
    get_recording_record_key,
    get_recording_record_key_from_context,
    recording_ctx,
    skip_recording_ctx,
)
from trinity.common.models.recording.recorder import Recorder  # noqa: F401
from trinity.common.models.sglang_patch.recording.models import (  # noqa: F401
    build_sglang_experience,
)
from trinity.common.models.sglang_patch.recording.recorder import (  # noqa: F401
    create_sglang_recorder,
    patch_tokenizer_manager_for_recording,
)
from trinity.common.models.sglang_patch.recording.server import (  # noqa: F401
    setup_sglang_recording,
)

__all__ = [
    "MemoryStore",
    "RecordStore",
    "Recorder",
    "RecordingContext",
    "RecordingIdentityMiddleware",
    "build_sglang_experience",
    "create_sglang_recorder",
    "get_recording_record_key",
    "get_recording_record_key_from_context",
    "patch_tokenizer_manager_for_recording",
    "recording_ctx",
    "setup_sglang_recording",
    "skip_recording_ctx",
]
