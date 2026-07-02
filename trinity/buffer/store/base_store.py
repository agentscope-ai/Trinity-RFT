from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from torch import Tensor

from trinity.common.experience import Experience


@dataclass
class ExperienceUpdate:
    """Fields that may be patched onto recorded experiences after generation."""

    reward: float | None = None
    info: dict | None = None
    teacher_logprobs: Tensor | None = None


class RecordStore(ABC):
    """Abstract base class for an in-process experience store.

    The key follows the format ``<batch_id>/<task_id>/<run_id>`` and each
    experience is associated with a unique sample id.
    """

    @abstractmethod
    def add(self, key: str, exps: List[Experience]) -> None:
        """Add experiences to the store under the given complete key."""

    @abstractmethod
    def overwrite(self, key: str, exps: List[Experience]) -> None:
        """Replace all experiences under the given complete key."""

    @abstractmethod
    def replace(self, key: str, old_sample_id: str, exp: Experience) -> None:
        """Replace one experience under the given complete key."""

    @abstractmethod
    def update(self, key: str, update: ExperienceUpdate, sample_ids: List[str] | None) -> None:
        """Patch selected experiences and stamp EID fields from the complete key."""

    @abstractmethod
    def get(self, key: str) -> List[Experience]:
        """Return experiences for an exact key or prefix without removing them."""

    @abstractmethod
    def remove(self, key: str) -> List[Experience]:
        """Remove and return experiences for an exact key or prefix."""

    @abstractmethod
    def keys(self) -> list[str]:
        """Return complete keys currently stored in insertion order."""

    @abstractmethod
    def block_prefix(self, prefix: str) -> None:
        """Mark a batch prefix as blocked.

        Once a prefix is blocked, ``add`` and ``overwrite`` for any complete
        key whose batch segment matches the prefix are silently dropped.
        ``get`` and ``remove`` are unaffected. This is used to reject writes
        that race in after a batch has been aborted/finalized and its records
        deleted, so they cannot reappear as orphans.
        """

    @abstractmethod
    def is_prefix_blocked(self, prefix: str) -> bool:
        """Return whether the given batch prefix is blocked."""
