"""Writer of the buffer."""
from abc import ABC, abstractmethod
from typing import List


class BufferWriter(ABC):
    """Interface of the buffer writer."""

    @abstractmethod
    async def write(self, data: List) -> None:
        """Write to buffer."""

    @abstractmethod
    async def acquire(self) -> int:
        """Acquire the buffer writer.

        Returns:
            `int`: The reference count of the buffer after acquiring.
        """

    @abstractmethod
    async def release(self) -> int:
        """Release the buffer writer. After release, the buffer writer can not be used again.

        Returns:
            `int`: The reference count of the buffer after releasing.
        """
