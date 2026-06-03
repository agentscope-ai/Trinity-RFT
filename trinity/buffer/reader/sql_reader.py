"""Reader of the SQL buffer."""

import traceback
from typing import Dict, List, Optional

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.storage.sql import SQLExperienceStorage, SQLStorage, SQLTaskStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class SQLReader(BufferReader):
    """Reader of the SQL buffer."""

    def __init__(self, config: StorageConfig) -> None:
        assert config.storage_type == StorageType.SQL.value
        self.wrap_in_ray = config.wrap_in_ray
        self.read_batch_size = config.batch_size
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

    async def read(self, batch_size: Optional[int] = None, **kwargs) -> List:
        batch_size = self.read_batch_size if batch_size is None else batch_size
        if self.wrap_in_ray:
            try:
                return await self.storage.read.remote(batch_size, **kwargs)
            except (StopIteration, StopAsyncIteration):
                raise StopAsyncIteration()
            except Exception as e:
                if "StopAsyncIteration" in traceback.format_exc():
                    raise StopAsyncIteration() from e
                raise
        else:
            storage = await self._get_async_storage()
            return await storage.read(batch_size, **kwargs)

    def state_dict(self) -> Dict:
        return {"current_index": 0}

    def load_state_dict(self, state_dict):
        return None
