import unittest
import uuid

import torch

from trinity.buffer.store import ExperienceUpdate, MemoryStore
from trinity.common.experience import EID, Experience


def get_dummy_experience(num: int, request_id: str | None = None):
    request_id = request_id or uuid.uuid4().hex[:6]
    return [
        Experience(
            eid=EID(suffix=request_id if num == 1 else f"{request_id}:{i}"),
            tokens=torch.zeros(5),
            prompt_length=2,
            info={
                "sample_index": i,
                "model_version": 0,
            },
        )
        for i in range(num)
    ]


class MemoryStoreTest(unittest.TestCase):
    def test_add_update_get_remove(self):
        store = MemoryStore()
        key = "0/task_a/1"
        experiences = get_dummy_experience(3, request_id="req_a")

        store.add(key, experiences)
        self.assertEqual(len(store), 3)

        store.update(
            key,
            update=ExperienceUpdate(reward=1.0, info={"source": "reward_model"}),
            sample_ids=None,
        )
        result = store.get(key)
        self.assertEqual(len(result), 3)
        for exp in result:
            self.assertEqual(exp.reward, 1.0)
            self.assertEqual(exp.info["source"], "reward_model")
            self.assertEqual(exp.eid.batch, "0")
            self.assertEqual(exp.eid.task, "task_a")
            self.assertEqual(exp.eid.run, 1)

        removed = store.remove(key)
        self.assertEqual(len(removed), 3)
        self.assertEqual(store.get(key), [])
        self.assertEqual(len(store), 0)

    def test_update_subset_by_sample_ids(self):
        store = MemoryStore()
        key = "0/task_a/1"
        experiences = get_dummy_experience(2, request_id="req_b")

        store.add(key, experiences)
        teacher_logprobs = torch.ones(3)
        store.update(
            key,
            update=ExperienceUpdate(reward=2.0, teacher_logprobs=teacher_logprobs),
            sample_ids=["req_b:1"],
        )

        result = store.get(key)
        self.assertIsNone(result[0].reward)
        self.assertEqual(result[1].reward, 2.0)
        self.assertEqual(result[1].eid.batch, "0")
        self.assertEqual(result[1].eid.task, "task_a")
        self.assertEqual(result[1].eid.run, 1)
        torch.testing.assert_close(result[1].teacher_logprobs, teacher_logprobs)

    def test_overwrite_replaces_existing_records(self):
        store = MemoryStore()
        key = "0/task_a/1"

        store.add(key, get_dummy_experience(2, request_id="old"))
        store.overwrite(key, get_dummy_experience(1, request_id="new"))

        result = store.get(key)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].eid.suffix, "new")

    def test_prefix_get_and_remove(self):
        store = MemoryStore()
        store.add("0/task_a/0", get_dummy_experience(1, request_id="a0"))
        store.add("0/task_a/1", get_dummy_experience(2, request_id="a1"))
        store.add("0/task_b/0", get_dummy_experience(1, request_id="b0"))

        self.assertEqual(len(store.get("0/task_a")), 3)
        self.assertEqual(len(store.remove("0/task_a")), 3)
        self.assertEqual(len(store.get("0")), 1)
        self.assertEqual(store.keys(), ["0/task_b/0"])

    def test_complete_key_required_for_mutations(self):
        store = MemoryStore()
        with self.assertRaises(ValueError):
            store.add("0/task_a", get_dummy_experience(1))
        with self.assertRaises(ValueError):
            store.overwrite("0/task_a", get_dummy_experience(1))
        with self.assertRaises(ValueError):
            store.update("0/task_a", update=ExperienceUpdate(reward=1.0), sample_ids=None)
        with self.assertRaises(ValueError):
            store.add("0/task_a/not_int", get_dummy_experience(1))

    def test_duplicate_sample_id_is_rejected(self):
        store = MemoryStore()
        exp = get_dummy_experience(1, request_id="dup")
        store.add("0/task_a/0", exp)
        with self.assertRaises(ValueError):
            store.add("0/task_a/1", exp)

    def test_blocked_prefix_drops_add_and_overwrite(self):
        store = MemoryStore()
        key = "0/task_a/0"
        store.add(key, get_dummy_experience(1, request_id="pre"))
        self.assertFalse(store.is_prefix_blocked("0"))

        # Real flow: block the batch, then delete its existing records.
        store.block_prefix("0")
        self.assertTrue(store.is_prefix_blocked("0"))
        store.remove(key)
        self.assertEqual(store.get(key), [])

        # A late add on a fresh key under the blocked batch is dropped.
        store.add("0/task_a/1", get_dummy_experience(2, request_id="post"))
        self.assertEqual(store.get("0/task_a/1"), [])
        self.assertNotIn("0/task_a/1", store.keys())

        # A late overwrite is also dropped: _drop_key is a no-op (records were
        # already deleted) and add is blocked, so nothing reappears.
        store.overwrite(key, get_dummy_experience(1, request_id="overwrite"))
        self.assertEqual(store.get(key), [])
        self.assertNotIn(key, store.keys())

    def test_blocked_prefix_does_not_affect_other_batches(self):
        store = MemoryStore()
        store.block_prefix("0")
        store.add("1/task_a/0", get_dummy_experience(1, request_id="other"))
        self.assertEqual(len(store.get("1/task_a/0")), 1)

    def test_blocked_prefix_keeps_get_and_remove_working(self):
        store = MemoryStore()
        key = "0/task_a/0"
        store.add(key, get_dummy_experience(2, request_id="keep"))
        store.block_prefix("0")
        # Reads and removes still work on already-stored records.
        self.assertEqual(len(store.get(key)), 2)
        self.assertEqual(len(store.remove(key)), 2)
        self.assertEqual(store.get(key), [])


if __name__ == "__main__":
    unittest.main()
