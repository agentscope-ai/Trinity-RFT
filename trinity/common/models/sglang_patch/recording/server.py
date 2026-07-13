"""Wiring that installs SGLang generation recording onto the embedded HTTP server.

Mirrors ``trinity/common/models/vllm_patch/recording/server.py:_setup_recording``:
(1) the engine wrap (``create_sglang_recorder``), (2) ``RecordingIdentityMiddleware``
— an in-process ASGI middleware reading ``Authorization: Bearer <record_key>``
into a contextvar, and (3) actor-side recording APIs over the model-owned store.

Called from ``sglang_patch.server_patch.get_api_server`` after the
``tokenizer_manager`` is created and **before** the uvicorn task starts serving,
so the middleware/router are mounted on ``app`` in time. The recorder and store
are owned by ``SGLangRolloutModel`` (passed in) so it can drain them in-process
via actor methods; they are also stashed on ``app.state`` for server-local
recording lifecycle management.
"""

import logging
from typing import Optional, Tuple

from trinity.buffer.store import RecordStore
from trinity.common.models.recording.recorder import Recorder
from trinity.common.models.recording.server import mount_recording_api
from trinity.common.models.sglang_patch.recording.recorder import create_sglang_recorder


def setup_sglang_recording(
    tokenizer_manager,
    app,
    logger: logging.Logger,
    *,
    recorder: Optional[Recorder] = None,
    store: Optional[RecordStore] = None,
    routed_experts_layout: Optional[Tuple[int, int]] = None,
) -> Recorder:
    """Wire generation recording onto the in-construction SGLang server.

    Only called when recording is on. The recorder is started here (its flusher
    task lives in the server's event loop, same loop as ``SGLangRolloutModel``).
    """
    recorder = create_sglang_recorder(
        tokenizer_manager,
        logger,
        store=store,
        recorder=recorder,
        enabled=True,
        routed_experts_layout=routed_experts_layout,
    )

    mount_recording_api(
        app,
        recorder,
        logger,
        engine_name="SGLang",
        start_recorder=True,
    )
    return recorder
