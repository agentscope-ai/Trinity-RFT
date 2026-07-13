"""Unit tests for RolloutCoordinator."""

import unittest
from types import SimpleNamespace

from trinity.explorer.rollout_coordinator import RolloutCoordinator
from trinity.explorer.workflow_runner import Status


class FakePipeline:
    """Minimal in-memory pipeline double for coordinator tests."""

    def __init__(self):
        """Initialize call tracking state."""

        self.process_chunk_calls = []
        self.prepare_called = False
        self.close_called = False

    async def prepare(self):
        """Record pipeline preparation."""

        self.prepare_called = True

    async def process_serialized_chunks(self, exp_chunks):
        """Record serialized chunk processing."""

        chunks = list(exp_chunks)
        self.process_chunk_calls.append(chunks)
        return {"experience_pipeline/experience_count": float(len(chunks))}

    async def close(self):
        """Record pipeline closure."""

        self.close_called = True


class FakeScheduler:
    """Minimal scheduler double for coordinator tests."""

    def __init__(self):
        """Initialize scheduler state and recorded calls."""

        self.default_timeout = 1.0
        self.started = False
        self.stopped = False
        self.schedule_calls = []
        self.scheduled_task_counts = {}
        self.cleanup_calls = []
        self.batch_results = {}
        self.get_statuses_calls = []

    async def start(self):
        """Mark the scheduler as started."""

        self.started = True

    async def stop(self):
        """Mark the scheduler as stopped."""

        self.stopped = True

    def schedule(self, tasks, batch_id):
        """Record scheduled tasks for one batch."""

        self.schedule_calls.append((batch_id, list(tasks)))
        self.scheduled_task_counts[batch_id] = len(tasks)

    async def get_statuses(
        self,
        batch_id,
        min_num=None,
        timeout=None,
        clear_timeout_tasks=True,
        return_partial_tasks=False,
    ):
        """Return only preconfigured statuses for one batch."""

        self.get_statuses_calls.append(
            {
                "batch_id": batch_id,
                "min_num": min_num,
                "timeout": timeout,
                "clear_timeout_tasks": clear_timeout_tasks,
                "return_partial_tasks": return_partial_tasks,
            }
        )
        statuses, _ = self.batch_results.pop(batch_id, ([], []))
        return statuses

    async def get_payload_results(
        self,
        batch_id,
        min_num=None,
        timeout=None,
        clear_timeout_tasks=True,
        return_partial_tasks=False,
    ):
        _ = min_num, timeout, clear_timeout_tasks, return_partial_tasks
        return self.batch_results.pop(batch_id, ([], []))

    async def cleanup_batch(self, batch_id, return_partial_tasks=False, restart_runners=True):
        """Record one scheduler cleanup request."""

        self.cleanup_calls.append(
            {
                "batch_id": batch_id,
                "return_partial_tasks": return_partial_tasks,
                "restart_runners": restart_runners,
            }
        )

    def discard_completed_results(self, batch_id):
        """Drop cached completed task results for one batch."""

        self.batch_results.pop(batch_id, None)


def _build_config(wait_after_min=0.0, return_partial_tasks=True, detailed_stats=False):
    return SimpleNamespace(
        explorer=SimpleNamespace(
            name="test_explorer",
            over_rollout=SimpleNamespace(
                wait_after_min=wait_after_min,
                return_partial_tasks=return_partial_tasks,
            ),
        ),
        monitor=SimpleNamespace(detailed_stats=detailed_stats),
    )


def _build_status(metric_value):
    return Status(
        completed_runs=1,
        total_runs=1,
        metrics=[{"run_metrics": float(metric_value)}],
    )


class CoordinatorHarness(RolloutCoordinator):
    """Coordinator subclass that injects fake owned dependencies."""

    def __init__(self, config, *, pipeline, scheduler):
        """Store doubles and delegate the main initialization to the parent."""

        self._test_pipeline = pipeline
        self._test_scheduler = scheduler
        self.discard_recorded_prefixes = []
        super().__init__(config)

    async def _init_experience_pipeline(self):
        """Return the injected fake pipeline."""

        self.experience_pipeline = self._test_pipeline
        await self.experience_pipeline.prepare()

    async def _init_scheduler(self):
        """Return the injected fake scheduler."""

        self.scheduler = self._test_scheduler

    def _init_rollout_actors(self):
        """Skip Ray actor resolution in unit tests."""

        self._rollout_actors = {}

    async def _discard_recorded_experiences(self, prefix: str) -> None:
        """Record cleanup requests without resolving real rollout actors."""

        self.discard_recorded_prefixes.append(prefix)


class TestRolloutCoordinator(unittest.IsolatedAsyncioTestCase):
    """Focused behavioral tests for the first coordinator implementation."""

    async def asyncSetUp(self):
        """Create one coordinator wired to fake owned dependencies."""

        self.scheduler = FakeScheduler()
        self.pipeline = FakePipeline()
        self.coordinator = CoordinatorHarness(
            _build_config(),
            pipeline=self.pipeline,
            scheduler=self.scheduler,
        )
        await self.coordinator.prepare()

    async def asyncTearDown(self):
        """Shutdown the coordinator after each test."""

        await self.coordinator.shutdown()

    async def test_finalize_train_batch_processes_scheduler_payloads(self):
        """Train finalize should consume batch payloads."""

        await self.coordinator.submit_batch(
            batch_id=1,
            tasks=[SimpleNamespace(is_eval=False), SimpleNamespace(is_eval=False)],
            batch_type="train",
        )

        self.scheduler.batch_results[1] = (
            [_build_status(10.0), _build_status(20.0)],
            [b"payload-0", b"payload-1"],
        )

        result = await self.coordinator.finalize_train_batch(1, timeout=1.0)
        self.assertEqual(result["finished_task_count"], 2)
        self.assertEqual(result["metrics"]["rollout/run_metrics/mean"], 15.0)
        self.assertEqual(result["metrics"]["experience_pipeline/experience_count"], 2.0)
        self.assertTrue(self.pipeline.prepare_called)
        self.assertEqual(self.pipeline.process_chunk_calls, [[b"payload-0", b"payload-1"]])
        self.assertEqual(self.coordinator.discard_recorded_prefixes[-1], "1")
        self.assertNotIn(1, self.coordinator.pending_batches)

        with self.assertRaisesRegex(KeyError, "not registered"):
            await self.coordinator.finalize_train_batch(1, timeout=1.0)

    async def test_finalize_train_batch_supports_partial_finalize(self):
        """Train finalize should allow partial completion when policy permits it."""

        await self.coordinator.submit_batch(
            batch_id=2,
            tasks=[SimpleNamespace(is_eval=False), SimpleNamespace(is_eval=False)],
            batch_type="train",
            min_wait_num=1,
        )

        self.scheduler.batch_results[2] = (
            [_build_status(7.0)],
            [b"payload-0"],
        )

        result = await self.coordinator.finalize_train_batch(2, timeout=1.0)

        self.assertEqual(result["finished_task_count"], 1)
        self.assertEqual(self.pipeline.process_chunk_calls[-1], [b"payload-0"])
        self.assertEqual(self.scheduler.cleanup_calls[-1]["batch_id"], 2)
        self.assertIn("2", self.coordinator.discard_recorded_prefixes)
        self.assertNotIn(2, self.coordinator.pending_batches)

    async def test_finalize_train_batch_times_out_without_any_results(self):
        """Train finalize should still fail when no task completes before timeout."""

        await self.coordinator.submit_batch(
            batch_id=7,
            tasks=[SimpleNamespace(is_eval=False)],
            batch_type="train",
        )

        with self.assertRaisesRegex(TimeoutError, "Timeout waiting for batch 7"):
            await self.coordinator.finalize_train_batch(7, timeout=1.0)

    async def test_finalize_eval_batch_aggregates_eval_metrics(self):
        """Eval finalize should aggregate scheduler results without pipeline writes."""

        batch_id = "3/eval_set"
        await self.coordinator.submit_batch(
            batch_id=batch_id,
            tasks=[SimpleNamespace(is_eval=True), SimpleNamespace(is_eval=True)],
            batch_type="eval",
        )
        self.scheduler.batch_results[batch_id] = (
            [_build_status(3.0), _build_status(5.0)],
            [],
        )

        result = await self.coordinator.finalize_eval_batch(batch_id, timeout=1.0)

        self.assertEqual(result["finished_task_count"], 2)
        self.assertEqual(result["metrics"]["eval/eval_set/run_metrics"], 4.0)
        self.assertEqual(self.pipeline.process_chunk_calls, [])
        self.assertEqual(self.scheduler.get_statuses_calls[0]["batch_id"], batch_id)
        self.assertEqual(self.coordinator.discard_recorded_prefixes[-1], batch_id)
        self.assertNotIn(batch_id, self.coordinator.pending_batches)

    async def test_finalize_train_batch_rejects_eval_batches_before_waiting(self):
        """Train finalize should reject an active eval batch instead of entering wait logic."""

        batch_id = "4/eval_set"
        await self.coordinator.submit_batch(
            batch_id=batch_id,
            tasks=[SimpleNamespace(is_eval=True)],
            batch_type="eval",
        )

        with self.assertRaisesRegex(ValueError, "expected train"):
            await self.coordinator.finalize_train_batch(batch_id, timeout=0.1)

    async def test_terminal_batches_are_not_reusable_after_finalize(self):
        """A finalized batch should be evicted instead of being cached for later reuse."""

        eval_batch_id = "5/eval_set"
        await self.coordinator.submit_batch(
            batch_id=eval_batch_id,
            tasks=[SimpleNamespace(is_eval=True)],
            batch_type="eval",
        )
        self.scheduler.batch_results[eval_batch_id] = ([_build_status(3.0)], [])
        await self.coordinator.finalize_eval_batch(eval_batch_id, timeout=1.0)
        self.assertEqual(self.coordinator.discard_recorded_prefixes[-1], eval_batch_id)

        with self.assertRaisesRegex(KeyError, "not registered"):
            await self.coordinator.finalize_train_batch(eval_batch_id, timeout=0.1)

        with self.assertRaisesRegex(KeyError, "not registered"):
            await self.coordinator.finalize_eval_batch(eval_batch_id, timeout=0.1)

        await self.coordinator.submit_batch(
            batch_id=6,
            tasks=[SimpleNamespace(is_eval=False)],
            batch_type="train",
        )
        self.scheduler.batch_results[6] = ([_build_status(11.0)], [b"payload-0"])
        await self.coordinator.finalize_train_batch(6, timeout=1.0)

        with self.assertRaisesRegex(KeyError, "not registered"):
            await self.coordinator.finalize_eval_batch(6, timeout=0.1)

        with self.assertRaisesRegex(KeyError, "not registered"):
            await self.coordinator.finalize_train_batch(6, timeout=0.1)

    async def test_abort_batch_marks_batch_aborted_and_evicts_it(self):
        """Abort should cleanup the batch immediately instead of caching a terminal result."""

        await self.coordinator.submit_batch(
            batch_id=4,
            tasks=[SimpleNamespace(is_eval=False), SimpleNamespace(is_eval=False)],
            batch_type="train",
        )

        await self.coordinator.abort_batch(4, reason="shutdown")

        self.assertEqual(self.scheduler.cleanup_calls[0]["batch_id"], 4)
        self.assertEqual(self.coordinator.discard_recorded_prefixes[-1], "4")
        self.assertNotIn(4, self.coordinator.pending_batches)

        with self.assertRaisesRegex(KeyError, "not registered"):
            await self.coordinator.finalize_train_batch(4, timeout=0.1)

    async def test_shutdown_closes_internal_dependencies(self):
        """Shutdown should close both owned scheduler and owned pipeline."""

        await self.coordinator.shutdown()

        self.assertTrue(self.scheduler.stopped)
        self.assertTrue(self.pipeline.close_called)
