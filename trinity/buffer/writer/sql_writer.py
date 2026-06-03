"""Writer of the SQL buffer."""

from trinity.buffer.buffer_writer import BufferWriter
from trinity.buffer.storage.sql import SQLExperienceStorage, SQLStorage, SQLTaskStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class SQLWriter(BufferWriter):
    """Writer of the SQL buffer."""

    def __init__(self, config: StorageConfig) -> None:
        assert config.storage_type == StorageType.SQL.value
        self.wrap_in_ray = config.wrap_in_ray
        self._storage = None
        self._async_storage = None
        self._config = config

    @property
    def storage(self):
        if self._storage is None:
            self._storage = SQLStorage.get_wrapper(self._config)
        return self._storage

    async def _get_async_storage(self):
        if self._async_storage is None:
            if self._config.schema_type is None:
                self._async_storage = SQLTaskStorage(self._config)
            else:
                self._async_storage = SQLExperienceStorage(self._config)
            await self._async_storage.prepare()
        return self._async_storage

    async def write(self, data):
        if self.wrap_in_ray:
            await self.storage.write.remote(data)
        else:
            storage = await self._get_async_storage()
            await storage.write(data)

    async def acquire(self) -> int:
        if self.wrap_in_ray:
            return await self.storage.acquire.remote()
        else:
            storage = await self._get_async_storage()
            return storage.acquire()

    async def release(self) -> int:
        if self.wrap_in_ray:
            return await self.storage.release.remote()
        else:
            storage = await self._get_async_storage()
            return storage.release()
