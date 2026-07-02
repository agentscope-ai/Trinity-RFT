import unittest

import torch

from trinity.buffer.store import ExperienceUpdate, MemoryStore, parse_record_key
from trinity.common.experience import EID, Experience
from trinity.common.models.recording.recorder import Recorder


def make_turn(
    *,
    request_id: str,
    record_key: str,
    tokens: list[int],
    prompt_length: int,
    logprobs: list[float],
    sample_index: int = 0,
) -> Experience:
    batch, task, run = parse_record_key(record_key)
    info = {"sample_index": sample_index}
    return Experience(
        eid=EID(batch=batch, task=task, run=run, suffix=request_id),
        tokens=tokens,
        prompt_length=prompt_length,
        logprobs=logprobs,
        info=info,
    )


class RecorderPrefixMergeTest(unittest.IsolatedAsyncioTestCase):
    async def test_prefix_experiences_merge_and_keep_final_sample_id(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"
        first = make_turn(
            request_id="req-1",
            record_key=record_key,
            tokens=[10, 11, 20, 21],
            prompt_length=2,
            logprobs=[-0.2, -0.3],
        )
        second = make_turn(
            request_id="req-2",
            record_key=record_key,
            tokens=[10, 11, 20, 21, 12, 13, 30, 31, 32],
            prompt_length=6,
            logprobs=[-0.4, -0.5, -0.6],
        )

        recorder._safe_append(first)
        recorder._safe_append(second)

        recorded = store.get(record_key)
        self.assertEqual(len(recorded), 1)
        merged = recorded[0]
        self.assertEqual(merged.eid.suffix, "req-2")
        self.assertEqual(merged.prompt_length, 2)
        self.assertTrue(torch.equal(merged.tokens, second.tokens))
        self.assertTrue(
            torch.equal(
                merged.action_mask,
                torch.tensor([True, True, False, False, True, True, True]),
            )
        )
        self.assertTrue(
            torch.allclose(
                merged.logprobs,
                torch.tensor([-0.2, -0.3, 0.0, 0.0, -0.4, -0.5, -0.6]),
            )
        )
        self.assertEqual(merged.info["merged_eid_suffixes"], ["req-1", "req-2"])
        self.assertEqual(merged.info["merged_sample_ids"], ["req-1", "req-2"])

        store.update(record_key, update=ExperienceUpdate(reward=1.0), sample_ids=["req-2"])
        self.assertEqual(store.get(record_key)[0].reward, 1.0)
        with self.assertRaises(KeyError):
            store.update(record_key, update=ExperienceUpdate(reward=2.0), sample_ids=["req-1"])

    async def test_non_prefix_experiences_do_not_merge(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"

        recorder._safe_append(
            make_turn(
                request_id="req-1",
                record_key=record_key,
                tokens=[10, 11, 20],
                prompt_length=2,
                logprobs=[-0.2],
            )
        )
        recorder._safe_append(
            make_turn(
                request_id="req-2",
                record_key=record_key,
                tokens=[10, 12, 30],
                prompt_length=2,
                logprobs=[-0.3],
            )
        )

        self.assertEqual(len(store.get(record_key)), 2)

    async def test_merge_head_replaces_only_matching_sample_stream(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"
        sample_zero = make_turn(
            request_id="req-1",
            record_key=record_key,
            tokens=[10, 11, 20],
            prompt_length=2,
            logprobs=[-0.2],
            sample_index=0,
        )
        sample_one_first = make_turn(
            request_id="req-2",
            record_key=record_key,
            tokens=[10, 11, 21],
            prompt_length=2,
            logprobs=[-0.3],
            sample_index=1,
        )
        sample_one_final = make_turn(
            request_id="req-3",
            record_key=record_key,
            tokens=[10, 11, 21, 12, 31],
            prompt_length=4,
            logprobs=[-0.4],
            sample_index=1,
        )

        recorder._safe_append(sample_zero)
        recorder._safe_append(sample_one_first)
        recorder._safe_append(sample_one_final)

        recorded = store.get(record_key)
        self.assertEqual(len(recorded), 2)
        self.assertEqual(recorded[0].eid.suffix, "req-1")
        self.assertEqual(recorded[1].eid.suffix, "req-3")
        self.assertTrue(
            torch.equal(
                recorded[1].action_mask,
                torch.tensor([True, False, True]),
            )
        )

    async def test_interleaved_branches_with_shared_sample_index_merge_independently(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"
        branch_a_first = make_turn(
            request_id="req-a1",
            record_key=record_key,
            tokens=[10, 11, 20],
            prompt_length=2,
            logprobs=[-0.2],
            sample_index=0,
        )
        branch_b_first = make_turn(
            request_id="req-b1",
            record_key=record_key,
            tokens=[10, 12, 21],
            prompt_length=2,
            logprobs=[-0.3],
            sample_index=0,
        )
        branch_a_final = make_turn(
            request_id="req-a2",
            record_key=record_key,
            tokens=[10, 11, 20, 13, 30],
            prompt_length=4,
            logprobs=[-0.4],
            sample_index=0,
        )
        branch_b_final = make_turn(
            request_id="req-b2",
            record_key=record_key,
            tokens=[10, 12, 21, 14, 31],
            prompt_length=4,
            logprobs=[-0.5],
            sample_index=0,
        )

        recorder._safe_append(branch_a_first)
        recorder._safe_append(branch_b_first)
        recorder._safe_append(branch_a_final)
        recorder._safe_append(branch_b_final)

        recorded = store.get(record_key)
        self.assertEqual(len(recorded), 2)
        self.assertEqual({exp.eid.suffix for exp in recorded}, {"req-a2", "req-b2"})
        merged_by_suffix = {exp.eid.suffix: exp for exp in recorded}
        self.assertEqual(
            merged_by_suffix["req-a2"].info["merged_eid_suffixes"], ["req-a1", "req-a2"]
        )
        self.assertEqual(
            merged_by_suffix["req-b2"].info["merged_eid_suffixes"], ["req-b1", "req-b2"]
        )

    async def test_multi_head_merge_uses_longest_matching_prefix(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"
        short_prefix = make_turn(
            request_id="req-short",
            record_key=record_key,
            tokens=[10, 11, 20],
            prompt_length=2,
            logprobs=[-0.2],
        )
        long_prefix = make_turn(
            request_id="req-long",
            record_key=record_key,
            tokens=[10, 11, 20, 12, 30],
            prompt_length=4,
            logprobs=[-0.3],
        )
        unrelated = make_turn(
            request_id="req-other",
            record_key=record_key,
            tokens=[10, 13, 21],
            prompt_length=2,
            logprobs=[-0.4],
        )
        final = make_turn(
            request_id="req-final",
            record_key=record_key,
            tokens=[10, 11, 20, 12, 30, 14, 40],
            prompt_length=6,
            logprobs=[-0.5],
        )

        recorder._safe_append(short_prefix)
        recorder._safe_append(long_prefix)
        recorder._safe_append(unrelated)
        recorder._safe_append(final)

        recorded = store.get(record_key)
        self.assertEqual(len(recorded), 2)
        merged = next(exp for exp in recorded if exp.eid.suffix == "req-final")
        self.assertEqual(merged.info["merged_eid_suffixes"], ["req-short", "req-long", "req-final"])
        self.assertEqual(merged.info["merged_turn_count"], 3)

    async def test_same_prompt_independent_completions_do_not_merge(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"
        short_completion = make_turn(
            request_id="req-short",
            record_key=record_key,
            tokens=[10, 11, 20, 21],
            prompt_length=2,
            logprobs=[-0.2, -0.3],
        )
        long_completion = make_turn(
            request_id="req-long",
            record_key=record_key,
            tokens=[10, 11, 20, 21, 22, 23],
            prompt_length=2,
            logprobs=[-0.4, -0.5, -0.6, -0.7],
        )

        recorder._safe_append(short_completion)
        recorder._safe_append(long_completion)

        recorded = store.get(record_key)
        self.assertEqual(len(recorded), 2)
        self.assertEqual({exp.eid.suffix for exp in recorded}, {"req-short", "req-long"})
        self.assertTrue(all("merged_turn_count" not in exp.info for exp in recorded))

    async def test_stale_merge_head_falls_back_to_append(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"
        first = make_turn(
            request_id="req-1",
            record_key=record_key,
            tokens=[10, 11, 20],
            prompt_length=2,
            logprobs=[-0.2],
        )
        second = make_turn(
            request_id="req-2",
            record_key=record_key,
            tokens=[10, 11, 20, 12, 30],
            prompt_length=4,
            logprobs=[-0.3],
        )

        recorder._safe_append(first)
        store.remove(record_key)
        recorder._safe_append(second)

        recorded = store.get(record_key)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0].eid.suffix, "req-2")
        self.assertEqual(recorded[0].prompt_length, 4)


if __name__ == "__main__":
    unittest.main()
