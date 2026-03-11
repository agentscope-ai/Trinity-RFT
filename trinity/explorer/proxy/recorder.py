from typing import Dict, List, Set

from sqlalchemy.orm import sessionmaker

from trinity.buffer.schema import init_engine
from trinity.buffer.utils import retry_session
from trinity.common.experience import Experience
from trinity.utils.log import get_logger


# TODO: implement an async version in the future
class HistoryRecorder:
    """Record chat history into the database."""

    def __init__(self, db_url: str, table_name: str):
        self.logger = get_logger()
        self.engine, self.table_model_cls = init_engine(
            db_url=db_url,
            table_name=table_name,
            schema_type="experience",
        )
        self.logger.info(f"Init SQL storage at {db_url}")
        self.session = sessionmaker(bind=self.engine)

    def record_history(self, experiences: List[Experience]) -> None:
        """Save experience to the database."""
        with retry_session(self.session) as db:
            exps = [self.table_model_cls.from_experience(exp) for exp in experiences]
            db.add_all(exps)

    def update_reward(
        self, reward: float, msg_ids: list, run_id: int, task_id: str
    ) -> List[Experience]:
        """Update reward for given response IDs and return the updated experiences.

        Args:
            reward (float): The reward value to be updated.
            msg_ids (list): List of message IDs to update.
            run_id (int): The run ID associated with the experiences.
            task_id (str): The task ID associated with the experiences.

        Returns:
            List[Experience]: List of updated experiences.

        Note:
            Only experiences that have not been consumed (consumed == 0) will be returned.
            For example, if you call this method multiple times with the same msg_ids, only
            the first call will return the updated experiences; subsequent calls will return
            an empty list.
        """
        with retry_session(self.session) as db:
            # Lock and retrieve records that have not been consumed yet.
            records = (
                db.query(self.table_model_cls)
                .filter(
                    self.table_model_cls.msg_id.in_(msg_ids),
                    self.table_model_cls.consumed == 0,
                )
                .with_for_update()
                .all()
            )

            if not records:
                return []

            # Update records in memory
            for record in records:
                record.reward = reward
                record.run_id = run_id
                record.task_id = task_id
                record.consumed += 1

            # The session commit is handled by the `retry_session` context manager.
            updated_experiences = [record.to_experience() for record in records]
            return updated_experiences


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

    def record_history(self, experiences: List[Experience]) -> None:
        """Save experiences in memory."""
        for exp in experiences:
            self._exp_map[exp.eid.suffix] = exp
            if getattr(exp, "consumed", 0) == 0:
                self._unconsumed.add(exp.eid.suffix)

    def update_reward(
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
