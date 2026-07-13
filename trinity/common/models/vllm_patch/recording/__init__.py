"""Generation-recording patch for the vLLM OpenAI server.
Designed for vllm >= 0.23.0.
"""

from trinity.buffer.store import MemoryStore, RecordStore
from trinity.common.models.recording.context import (
    RecordingContext,
    RecordingIdentityMiddleware,
    get_recording_record_key_from_context,
    get_recording_request_from_context,
    recording_ctx,
    skip_recording_ctx,
)
from trinity.common.models.recording.recorder import Recorder
from trinity.common.models.vllm_patch.recording.models import build_experience
from trinity.common.models.vllm_patch.recording.recorder import (
    create_vllm_recorder,
    patch_engine_for_recording,
)
from trinity.common.models.vllm_patch.recording.server import (
    run_api_server_with_recording,
)

__all__ = [
    "MemoryStore",
    "RecordStore",
    "RecordingContext",
    "RecordingIdentityMiddleware",
    "Recorder",
    "build_experience",
    "create_vllm_recorder",
    "get_recording_record_key_from_context",
    "get_recording_request_from_context",
    "patch_engine_for_recording",
    "recording_ctx",
    "run_api_server_with_recording",
    "skip_recording_ctx",
]
