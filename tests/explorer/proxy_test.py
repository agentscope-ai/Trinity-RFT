import os
import unittest
import uuid
from typing import List

import torch

from trinity.common.experience import EID, Experience
from trinity.explorer.proxy.recorder import HistoryRecorder


def get_dummy_experience(num: int) -> List[Experience]:
    return [
        Experience(
            eid=EID(suffix=uuid.uuid4().hex[:6]),
            tokens=torch.zeros(5),
            prompt_length=2,
            info={
                "model_version": 0,
            },
        )
        for _ in range(num)
    ]


db_path = os.path.join(os.path.dirname(__file__), "test_recorder.db")


class RecorderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    async def test_recorder(self):
        recorder = HistoryRecorder(
            db_url="sqlite:///" + db_path,
            table_name="experience",
        )
        self.assertIsInstance(recorder, HistoryRecorder)

        experiences_1 = get_dummy_experience(3)
        await recorder.record_history(experiences_1)

        msg_ids_1 = [exp.eid.suffix for exp in experiences_1]
        experiences_2 = get_dummy_experience(2)
        await recorder.record_history(experiences_2)
        updated_experiences = await recorder.update_reward(
            reward=1.0, msg_ids=msg_ids_1, run_id=1, task_id="test_task"
        )
        self.assertEqual(len(updated_experiences), 3)
        for exp in updated_experiences:
            self.assertEqual(exp.reward, 1.0)
            self.assertEqual(exp.eid.run, 1)
            self.assertEqual(str(exp.eid.task), "test_task")

        updated_experiences_empty = await recorder.update_reward(
            reward=2.0, msg_ids=["non_existing_id"], run_id=1, task_id="test_task"
        )
        self.assertEqual(len(updated_experiences_empty), 0)

        await recorder.record_history([])

        updated_experiences_2 = await recorder.update_reward(
            reward=3.0,
            msg_ids=[exp.eid.suffix for exp in experiences_2],
            run_id=2,
            task_id="test_task_2",
        )
        self.assertEqual(len(updated_experiences_2), 2)
        for exp in updated_experiences_2:
            self.assertEqual(exp.reward, 3.0)
            self.assertEqual(exp.eid.run, 2)
            self.assertEqual(str(exp.eid.task), "test_task_2")

        updated_experiences_3 = await recorder.update_reward(
            reward=4.0,
            msg_ids=[exp.eid.suffix for exp in experiences_2],
            run_id=3,
            task_id="test_task_3",
        )
        self.assertEqual(len(updated_experiences_3), 0)  # already consumed
