"""Engine-agnostic background recorder for generated experiences."""

import asyncio
import logging
import queue
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Optional

from trinity.buffer.store import RecordStore, get_record_key
from trinity.common.experience import Experience
from trinity.common.models.recording.context import skip_recording_ctx
from trinity.common.models.recording.merger import PrefixExperienceMerger

MODEL_VERSION_ATTR = "trinity_model_version"
TRINITY_RECORDER_ATTR = "trinity_recorder"
TRINITY_RECORD_STORE_ATTR = "trinity_record_store"

BuildExperiencesFn = Callable[..., Sequence[Experience]]

# Sentinel pushed onto the queue to request the flusher thread to exit.
_STOP = object()


class Recorder:
    """Drains finished turns into a ``RecordStore`` from a background thread.

    Engine-specific code supplies ``build_experiences``, which converts a
    finished engine output object into Trinity ``Experience`` instances.

    The CPU-bound store writes (and prefix merging) run on a dedicated daemon
    thread so they never stall the engine's event loop. Producers (engine-loop
    coroutines) push onto an unbounded, thread-safe ``queue.Queue`` and never
    block. ``MemoryStore`` is expected to be thread-safe (it guards its
    dictionaries with a lock) since engine-loop code may touch the store
    concurrently with the flusher.
    """

    def __init__(
        self,
        store: RecordStore,
        *,
        build_experiences: BuildExperiencesFn,
        enabled: bool,
        rank: int = 0,
        engine_client: Any = None,
        merge_prefix_experiences: bool = True,
    ) -> None:
        self.store = store
        self.enabled = enabled
        self.rank = rank
        self.engine_client = engine_client
        self.merge_prefix_experiences = merge_prefix_experiences
        self._build_experiences = build_experiences
        # Thread-safe unbounded queue; producers never block on put.
        self._queue: "queue.Queue[Optional[Experience]]" = queue.Queue()
        self._flusher: Optional[threading.Thread] = None
        self._pending: "set[asyncio.Task]" = set()
        self._prefix_merger = PrefixExperienceMerger(store)

    def start(self) -> None:
        """Start the background flusher thread. Idempotent."""
        if self._flusher is not None or not self.enabled:
            return
        self._flusher = threading.Thread(
            target=self._flush_loop,
            name=f"trinity-recorder-flusher-rank{self.rank}",
            daemon=True,
        )
        self._flusher.start()

    async def stop(self) -> None:
        """Drain in-flight + queued turns, then stop the flusher thread."""
        if self._flusher is None:
            return
        await self.flush()
        self._queue.put(_STOP)  # type: ignore [arg-type]
        # join() blocks; run it off the event loop.
        await asyncio.to_thread(self._flusher.join)
        self._flusher = None

    def schedule_record(self, output: Any, record_key: Optional[str], **builder_kwargs) -> None:
        """Spawn and track a record task for a finished engine output."""
        task = asyncio.create_task(self._record(output, record_key, **builder_kwargs))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def flush(self) -> None:
        """Wait until every in-flight record has been appended to the store."""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        if self._flusher is not None:
            # queue.Queue.join() is a blocking call; run it off the event loop
            # so the engine keeps scheduling while the flusher drains.
            await asyncio.to_thread(self._queue.join)

    async def _record(self, output: Any, record_key: Optional[str], **builder_kwargs) -> None:
        if skip_recording_ctx.get():
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        model_version = getattr(self.engine_client, MODEL_VERSION_ATTR, None)
        exps = self._build_experiences(
            output,
            record_key,
            timestamp=timestamp,
            model_version=model_version,
            **builder_kwargs,
        )
        for exp in exps:
            self._queue.put(exp)  # non-blocking on an unbounded queue

    def _flush_loop(self) -> None:
        while True:
            exp = self._queue.get()
            try:
                if exp is _STOP:
                    return
                self._safe_append(exp)  # type: ignore [arg-type]
            except Exception:
                logging.getLogger(__name__).exception(
                    "recording flusher failed for request %s",
                    exp.eid.suffix if exp is not _STOP else None,
                )
            finally:
                self._queue.task_done()

    def _safe_append(self, exp: Experience) -> None:
        try:
            record_key = get_record_key(exp)
            if self.merge_prefix_experiences and self._prefix_merger.try_merge(record_key, exp):
                return
            self.store.add(record_key, [exp])
            if self.merge_prefix_experiences:
                self._prefix_merger.remember(record_key, exp)
        except Exception:
            logging.getLogger(__name__).exception(
                "recording store write failed for request %s",
                exp.eid.suffix,
            )

    def forget_record(self, record_key: str) -> None:
        self._prefix_merger.forget_record(record_key)
