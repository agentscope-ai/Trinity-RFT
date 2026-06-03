from typing import Dict, List, Set

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from trinity.buffer.schema.sql_schema import init_async_engine
from trinity.buffer.utils import async_run_with_retry_session
from trinity.common.experience import Experience
from trinity.utils.log import get_logger


class HistoryRecorder:
    """Record chat history into the database using async SQL."""

    def __init__(self, db_url: str, table_name: str):
        self.logger = get_logger()
        self._db_url = db_url
        self._table_name = table_name
        self._initialized = False

    async def prepare(self) -> None:
        if self._initialized:
            return
        engine, self.meta_cls, self.blob_cls = await init_async_engine(
            db_url=self._db_url,
            table_name=self._table_name,
            schema_type="experience",
        )
        self.session = async_sessionmaker(engine, expire_on_commit=False)
        self._initialized = True
        self.logger.info(f"Init async SQL storage at {self._db_url}")

    async def record_history(self, experiences: List[Experience]) -> None:
        """Save experiences to the database."""
        await self.prepare()

        async def operation(session: AsyncSession):
            for exp in experiences:
                meta_row = self.meta_cls.from_experience(exp)
                session.add(meta_row)
                await session.flush()
                blob_row = self.blob_cls(id=meta_row.id, experience_bytes=exp.serialize())
                session.add(blob_row)

        await async_run_with_retry_session(self.session, operation)

    async def update_reward(
        self, reward: float, msg_ids: list, run_id: int, task_id: str
    ) -> List[Experience]:
        """Update reward for given response IDs and return the updated experiences.

        Only experiences that have not been consumed (consumed == 0) will be returned.
        """
        await self.prepare()

        meta_cls = self.meta_cls
        blob_cls = self.blob_cls

        async def operation(session: AsyncSession):
            stmt = (
                select(meta_cls)
                .where(meta_cls.msg_id.in_(msg_ids), meta_cls.consumed == 0)
                .with_for_update()
            )
            result = await session.execute(stmt)
            records = result.scalars().all()

            if not records:
                return []

            ids = [record.id for record in records]

            update_stmt = (
                update(meta_cls)
                .where(meta_cls.id.in_(ids))
                .values(
                    reward=reward,
                    run_id=run_id,
                    task_id=task_id,
                    consumed=meta_cls.consumed + 1,
                )
            )
            await session.execute(update_stmt)

            blob_stmt = select(blob_cls).where(blob_cls.id.in_(ids))
            blob_result = await session.execute(blob_stmt)
            blobs = blob_result.scalars().all()
            blob_map = {b.id: b.experience_bytes for b in blobs}

            # Re-fetch meta rows to get updated values
            refresh_stmt = select(meta_cls).where(meta_cls.id.in_(ids))
            refresh_result = await session.execute(refresh_stmt)
            updated_records = refresh_result.scalars().all()

            updated_experiences = []
            for record in updated_records:
                blob_bytes = blob_map.get(record.id)
                if blob_bytes is not None:
                    updated_experiences.append(record.to_experience(blob_bytes))
            return updated_experiences

        return await async_run_with_retry_session(self.session, operation)


class MemoryHistoryRecorder:
    """
    In-memory version of HistoryRecorder for high-performance reward update and history recording.
    All data is stored in memory, and can be flushed to persistent storage as needed.
    """

    def __init__(self):
        self.logger = get_logger()
        # msg_id -> Experience
        self._exp_map: Dict[str, Experience] = {}
        # Set of msg_id that are not consumed
        self._unconsumed: Set[str] = set()

    async def record_history(self, experiences: List[Experience]) -> None:
        """Save experiences in memory."""
        for exp in experiences:
            self._exp_map[exp.eid.suffix] = exp
            if getattr(exp, "consumed", 0) == 0:
                self._unconsumed.add(exp.eid.suffix)

    async def update_reward(
        self, reward: float, msg_ids: list, run_id: int, task_id: str
    ) -> List[Experience]:
        """Update reward for given response IDs and return the updated experiences."""
        updated = []
        for msg_id in msg_ids:
            if msg_id in self._unconsumed and msg_id in self._exp_map:
                exp = self._exp_map.pop(msg_id)
                exp.reward = reward
                exp.eid.run = run_id
                exp.eid.task = task_id
                self._unconsumed.remove(msg_id)
                updated.append(exp)
        return updated
