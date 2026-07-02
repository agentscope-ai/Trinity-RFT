"""In-memory implementation of the experience store interface."""

import logging
import threading
from collections import OrderedDict
from typing import Iterable, List

from trinity.buffer.store.base_store import ExperienceUpdate, RecordStore
from trinity.common.experience import Experience

_logger = logging.getLogger(__name__)


def parse_record_key(key: str) -> tuple[str, str, int]:
    """Parse a complete ``<batch_id>/<task_id>/<run_id>`` store key.

    ``batch_id`` may itself contain ``/`` for eval batches, for example
    ``0/eval_short/1/0`` means batch ``0/eval_short``, task ``1`` and run ``0``.
    """
    parts = key.rsplit("/", 2)
    if len(parts) != 3 or any(part == "" for part in parts):
        raise ValueError(
            f"Store key must be complete '<batch_id>/<task_id>/<run_id>', got '{key}'."
        )
    batch, task, run_text = parts
    try:
        run = int(run_text)
    except ValueError as exc:
        raise ValueError(
            f"Store key run_id must be an integer in '<batch_id>/<task_id>/<run_id>', "
            f"got '{key}'."
        ) from exc
    return batch, task, run


def get_sample_id(exp: Experience) -> str:
    """Return the short sample id used by ``MemoryStore``."""
    return exp.eid.suffix


def get_record_key(exp: Experience) -> str:
    """Return the complete store key stamped on an experience."""
    if exp.eid.batch != "" and exp.eid.task != "":
        return exp.eid.rid
    return exp.eid.suffix


class MemoryStore(RecordStore):
    """A fast in-process store backed by Python dictionaries.

    ``add``, ``overwrite`` and ``update`` require complete keys in the form
    ``<batch_id>/<task_id>/<run_id>``. ``get`` and ``remove`` also accept prefixes
    so callers can drain a batch or task at once.

    A coarse reentrant lock guards every public method: the store may be written
    to from the recorder's background flusher thread while the engine event loop
    reads/updates it (e.g. ``VLLMModel.generate`` short-circuit adds, actor
    ``block_prefix``/``update`` calls). The lock is uncontended on the fast path
    (single consumer) and only serializes the rare cross-thread overlap.
    """

    def __init__(self) -> None:
        # main storage of experiences, keyed by complete store key and sample_id
        self._records: dict[str, OrderedDict[str, Experience]] = {}
        # extra indices to support prefix-based lookups in get() and remove()
        self._batch_keys: dict[str, OrderedDict[str, None]] = {}
        self._task_keys: dict[tuple[str, str], OrderedDict[str, None]] = {}
        self._sample_to_key: dict[str, str] = {}
        # batch prefixes whose writes should be silently dropped (aborted/
        # finalized batches); see ``block_prefix``. Only grows since batch_id
        # is never reused.
        self._blocked_batches: set[str] = set()
        # Serializes all dict mutations/reads across the flusher thread and the
        # engine event loop. Reentrant so ``overwrite`` -> ``add`` works.
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return sum(len(exps) for exps in self._records.values())

    def add(self, key: str, exps: List[Experience]) -> None:
        with self._lock:
            batch, task, _ = self._parse_complete_key(key)  # validate key format
            if batch in self._blocked_batches:
                _logger.debug(
                    "Dropping write to blocked batch '%s' (key=%s, %d exps).",
                    batch,
                    key,
                    len(exps),
                )
                return
            if not exps:
                return

            records = self._records.setdefault(key, OrderedDict())
            self._index_key(batch, task, key)
            for exp in exps:
                sample_id = get_sample_id(exp)
                owner_key = self._sample_to_key.get(sample_id)
                if owner_key is not None:
                    raise ValueError(
                        f"Duplicate sample_id '{sample_id}' already exists under key '{owner_key}'."
                    )
                records[sample_id] = exp
                self._sample_to_key[sample_id] = key

    def overwrite(self, key: str, exps: List[Experience]) -> None:
        with self._lock:
            self._parse_complete_key(key)  # validate key format
            self._drop_key(key)
            self.add(key, exps)

    def replace(self, key: str, old_sample_id: str, exp: Experience) -> None:
        with self._lock:
            self._parse_complete_key(key)  # validate key format
            records = self._records.get(key)
            if records is None:
                raise KeyError(f"Key '{key}' does not exist.")
            if old_sample_id not in records:
                raise KeyError(f"sample_id '{old_sample_id}' does not exist under key '{key}'.")

            new_sample_id = get_sample_id(exp)
            owner_key = self._sample_to_key.get(new_sample_id)
            if owner_key is not None and (owner_key != key or new_sample_id != old_sample_id):
                raise ValueError(
                    f"Duplicate sample_id '{new_sample_id}' already exists under key '{owner_key}'."
                )

            items = []
            for sample_id, record in records.items():
                if sample_id == old_sample_id:
                    items.append((new_sample_id, exp))
                else:
                    items.append((sample_id, record))

            records.clear()
            records.update(items)
            self._sample_to_key.pop(old_sample_id, None)
            self._sample_to_key[new_sample_id] = key

    def update(
        self,
        key: str,
        update: ExperienceUpdate,
        sample_ids: List[str] | None,
    ) -> None:
        with self._lock:
            batch, task, run = self._parse_complete_key(key)  # validate key format
            records = self._records.get(key)
            if records is None:
                raise KeyError(f"Key '{key}' does not exist.")
            target_ids: Iterable[str] = list(records.keys()) if sample_ids is None else sample_ids
            for sample_id in target_ids:
                if sample_id not in records:
                    raise KeyError(f"sample_id '{sample_id}' does not exist under key '{key}'.")
                exp = records[sample_id]
                exp.eid.batch = batch
                exp.eid.task = task
                exp.eid.run = run
                if update.reward is not None:
                    exp.reward = update.reward
                if update.info:
                    if exp.info is None:
                        exp.info = {}
                    exp.info.update(update.info)
                if update.teacher_logprobs is not None:
                    exp.teacher_logprobs = update.teacher_logprobs

    def get(self, key: str) -> List[Experience]:
        with self._lock:
            result: List[Experience] = []
            for matched_key in self._matching_keys(key):
                result.extend(self._records[matched_key].values())
            return result

    def remove(self, key: str) -> List[Experience]:
        with self._lock:
            result: List[Experience] = []
            for matched_key in self._matching_keys(key):
                result.extend(self._drop_key(matched_key))
            return result

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._records.keys())

    def block_prefix(self, prefix: str) -> None:
        """Mark a batch prefix as blocked; future ``add``/``overwrite`` are dropped."""
        with self._lock:
            self._blocked_batches.add(prefix)

    def is_prefix_blocked(self, prefix: str) -> bool:
        """Return whether the given batch prefix is blocked."""
        with self._lock:
            return prefix in self._blocked_batches

    @staticmethod
    def _parse_complete_key(key: str) -> tuple[str, str, int]:
        """Parse a complete store key; also usable as a key-format validator."""
        return parse_record_key(key)

    def _matching_keys(self, key: str) -> list[str]:
        if key == "":
            return list(self._records.keys())
        if key in self._records:
            return [key]
        if key in self._batch_keys:
            return list(self._batch_keys[key])

        parts = key.split("/")
        if len(parts) == 1 and parts[0] != "":
            return list(self._batch_keys.get(parts[0], ()))
        if len(parts) == 2 and parts[0] != "" and parts[1] != "":
            return list(self._task_keys.get((parts[0], parts[1]), ()))

        batch, sep, task = key.rpartition("/")
        if sep and batch and task:
            return list(self._task_keys.get((batch, task), ()))
        return []

    def _drop_key(self, key: str) -> list[Experience]:
        records = self._records.pop(key, None)
        if records is None:
            return []
        batch, task, _ = self._parse_complete_key(key)
        self._unindex_key(batch, task, key)
        for sample_id in records:
            self._sample_to_key.pop(sample_id, None)
        return list(records.values())

    def _index_key(self, batch: str, task: str, key: str) -> None:
        self._batch_keys.setdefault(batch, OrderedDict())[key] = None
        self._task_keys.setdefault((batch, task), OrderedDict())[key] = None

    def _unindex_key(self, batch: str, task: str, key: str) -> None:
        batch_keys = self._batch_keys.get(batch)
        if batch_keys is not None:
            batch_keys.pop(key, None)
            if not batch_keys:
                self._batch_keys.pop(batch, None)

        task_key = (batch, task)
        task_keys = self._task_keys.get(task_key)
        if task_keys is not None:
            task_keys.pop(key, None)
            if not task_keys:
                self._task_keys.pop(task_key, None)
