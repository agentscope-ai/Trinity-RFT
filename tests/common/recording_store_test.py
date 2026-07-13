import unittest

import torch

from trinity.buffer.store import (
    ExperienceUpdate,
    MemoryStore,
    get_record_key,
    parse_record_key,
)
from trinity.common.experience import EID, Experience


def make_exp(request_id: str, record_key: str | None = None) -> Experience:
    info = {"sample_index": 0}
    eid = EID(suffix=request_id)
    if record_key is not None:
        batch, task, run = parse_record_key(record_key)
        eid.batch = batch
        eid.task = task
        eid.run = run
    return Experience(
        eid=eid,
        tokens=torch.zeros(5),
        prompt_length=2,
        info=info,
    )


class MemoryStoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_update_reward_sets_eid_from_record_key(self):
        store = MemoryStore()
        record_key = "0/task_a/1"
        exp = make_exp("req_a", record_key)

        store.add(get_record_key(exp), [exp])
        store.update(
            record_key,
            update=ExperienceUpdate(reward=1.5, info={"source": "reward_model"}),
            sample_ids=None,
        )
        updated = store.remove(record_key)

        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0].reward, 1.5)
        self.assertEqual(updated[0].info["source"], "reward_model")
        self.assertNotIn("run", updated[0].info)
        self.assertNotIn("task", updated[0].info)
        self.assertEqual(updated[0].eid.batch, "0")
        self.assertEqual(updated[0].eid.task, "task_a")
        self.assertEqual(updated[0].eid.run, 1)
        self.assertEqual(store.get(record_key), [])

    async def test_complete_record_key_request_lookup_and_delete(self):
        store = MemoryStore()
        record_key = "0/task_a/1"
        exp = make_exp("req_a", record_key)

        store.add(get_record_key(exp), [exp])

        self.assertEqual(store.keys(), [record_key])
        self.assertIs(_find_request(store, record_key, "req_a"), exp)

        deleted = _delete_request(store, record_key, "req_a")
        self.assertTrue(deleted)
        self.assertEqual(store.keys(), [])

    async def test_delete_request_experience_keeps_other_experiences(self):
        store = MemoryStore()
        record_key = "0/task_a/1"
        exp_a = make_exp("req_a", record_key)
        exp_b = make_exp("req_b", record_key)

        store.add(get_record_key(exp_a), [exp_a])
        store.add(get_record_key(exp_b), [exp_b])

        deleted = _delete_request(store, record_key, "req_a")

        self.assertTrue(deleted)
        remaining = store.get(record_key)
        self.assertEqual(remaining, [exp_b])

    async def test_eval_batch_record_key_allows_slash_in_batch_id(self):
        store = MemoryStore()
        record_key = "0/eval_short/1/0"
        exp = make_exp("req_eval", record_key)

        batch, task, run = parse_record_key(record_key)
        self.assertEqual(batch, "0/eval_short")
        self.assertEqual(task, "1")
        self.assertEqual(run, 0)

        store.add(get_record_key(exp), [exp])

        self.assertEqual(store.get(record_key), [exp])
        self.assertEqual(store.get("0/eval_short"), [exp])
        self.assertEqual(store.get("0/eval_short/1"), [exp])
        self.assertEqual(store.remove("0/eval_short/1"), [exp])
        self.assertEqual(store.keys(), [])


def _find_request(store: MemoryStore, record_key: str, request_id: str) -> Experience | None:
    for exp in store.get(record_key):
        if exp.eid.suffix == request_id:
            return exp
    return None


def _delete_request(store: MemoryStore, record_key: str, request_id: str) -> bool:
    kept = []
    deleted = False
    for exp in store.get(record_key):
        if exp.eid.suffix == request_id:
            deleted = True
        else:
            kept.append(exp)
    if deleted:
        if kept:
            store.overwrite(record_key, kept)
        else:
            store.remove(record_key)
    return deleted


if __name__ == "__main__":
    unittest.main()
