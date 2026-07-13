"""Rollout coordinator for async batch submission and finalize."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union

import ray
from ray.actor import ActorHandle

from trinity.buffer.pipelines.experience_pipeline import ExperiencePipeline
from trinity.common.config import Config
from trinity.common.workflows import Task
from trinity.explorer.scheduler import Scheduler
from trinity.utils.log import get_logger
from trinity.utils.metrics import aggregate_eval_metrics, aggregate_metrics

BatchId = Union[int, str]
BatchType = Literal["train", "eval"]


class BatchLifecycleState(str, Enum):
    """Lifecycle states for one submitted batch."""

    PENDING = "pending"
    RUNNING = "running"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    ABORTED = "aborted"


@dataclass
class BatchState:
    """In-memory state tracked for one train or eval batch."""

    batch_id: BatchId
    batch_type: BatchType
    expected_task_count: int
    statuses: Dict[Union[int, str], Any] = field(default_factory=dict)
    min_wait_num: Optional[int] = None
    state: BatchLifecycleState = BatchLifecycleState.PENDING
    final_result: Optional[dict] = None
    finalize_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def completed_task_count(self) -> int:
        """Return the number of completed tasks tracked by status."""

        return len(self.statuses)


class RolloutCoordinator:
    """Own scheduler-side batch state and expose batch-level finalize APIs."""

    def __init__(
        self,
        config: Config,
    ):
        """Create a coordinator with internally managed scheduler and pipeline."""
        self.logger = get_logger(f"{config.explorer.name}_rollout_coordinator", in_ray_actor=True)
        self.config = config
        self.experience_pipeline = None
        self.scheduler: Optional[Scheduler] = None
        self.pending_batches: Dict[BatchId, BatchState] = {}
        self.running = False
        self.detailed_stats = getattr(getattr(config, "monitor", None), "detailed_stats", False)
        # Prepared map of rollout engine_id -> rollout actor handle, for
        # scheduler construction and recording residual cleanup.
        self._rollout_actors: Dict[int, ActorHandle] = {}

    def _init_rollout_actors(self) -> None:
        """Resolve each rollout engine's actor handle via named Ray actors.

        Mirrors ``Allocator.get_actor_name`` + ``ray.get_actor``: rollout model
        actors are named ``f"{explorer.name}_rollout_model_{engine_id}_0"``
        (node_id 0 holds the recording store).
        """
        if self._rollout_actors:
            return
        rollout_cfg = self.config.explorer.rollout_model
        name = self.config.explorer.name
        namespace = rollout_cfg.ray_namespace
        actors: Dict[int, ActorHandle] = {}
        for engine_id in range(rollout_cfg.engine_num):
            actor_name = f"{name}_rollout_model_{engine_id}_0"
            try:
                actors[engine_id] = ray.get_actor(actor_name, namespace=namespace)
            except ValueError as exc:
                raise RuntimeError(
                    "Rollout actor %s not found in namespace %s."
                    " RolloutCoordinator cannot initialize Scheduler without all rollout actors."
                    % (actor_name, namespace)
                ) from exc
        self._rollout_actors = actors

    async def prepare(self) -> None:
        """Initialize the owned pipeline and scheduler."""
        if self.running:
            return
        if not self._rollout_actors and getattr(self.config, "mode", None) != "serve":
            self._init_rollout_actors()
        if self.experience_pipeline is None:
            await self._init_experience_pipeline()
        if self.scheduler is None:
            await self._init_scheduler()
        self.running = True

    async def shutdown(self) -> None:
        """Stop background work and close owned dependencies."""
        self.running = False
        if self.scheduler is not None:
            await self.scheduler.stop()
            self.scheduler = None
        if self.experience_pipeline is not None:
            await self.experience_pipeline.close()
            self.experience_pipeline = None

    async def _init_experience_pipeline(self):
        """Create the experience pipeline owned by this coordinator actor."""
        if getattr(self.config, "mode", None) == "bench":
            return None
        self.experience_pipeline = ExperiencePipeline(self.config)
        await self.experience_pipeline.prepare()

    async def _init_scheduler(self):
        """Create the scheduler owned by this coordinator."""
        if getattr(self.config, "mode", None) == "serve":
            return
        self.scheduler = Scheduler(
            self.config,
            rollout_actors=self._rollout_actors,
        )
        await self.scheduler.start()

    def _require_scheduler(self) -> Scheduler:
        """Return the initialized scheduler."""
        assert self.scheduler is not None, "RolloutCoordinator.prepare() must be called first."
        return self.scheduler

    async def submit_batch(
        self,
        *,
        batch_id: BatchId,
        tasks: list[Task],
        batch_type: BatchType,
        min_wait_num: Optional[int] = None,
    ) -> None:
        """Register a new batch and schedule its tasks."""
        existing_state = self.pending_batches.get(batch_id)
        if existing_state is not None and existing_state.state not in {
            BatchLifecycleState.FINALIZED,
            BatchLifecycleState.ABORTED,
        }:
            raise ValueError(f"Batch {batch_id} is already active.")

        batch_state = BatchState(
            batch_id=batch_id,
            batch_type=batch_type,
            expected_task_count=len(tasks),
            min_wait_num=min_wait_num,
        )
        self.pending_batches[batch_id] = batch_state

        if tasks:
            self._require_scheduler().schedule(tasks, batch_id=batch_id)
            batch_state.state = BatchLifecycleState.RUNNING

    async def finalize_train_batch(
        self,
        batch_id: int,
        *,
        timeout: Optional[float] = None,
    ) -> dict:
        """Finalize one train batch and return aggregated metrics."""
        batch_state = self._get_batch_state(batch_id, expected_type="train")
        return await self._finalize_train_batch(batch_state, timeout=timeout)

    async def finalize_eval_batch(
        self,
        batch_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> dict:
        """Finalize one eval batch and return aggregated eval metrics."""
        batch_state = self._get_batch_state(batch_id, expected_type="eval")
        return await self._finalize_eval_batch(batch_state, timeout=timeout)

    async def _finalize_eval_batch(
        self, batch_state: BatchState, *, timeout: Optional[float]
    ) -> dict:
        """Finalize one eval batch."""

        scheduler = self._require_scheduler()
        async with batch_state.finalize_lock:
            existing_result = self._get_existing_final_result(batch_state)
            if existing_result is not None:
                return existing_result

            statuses = await scheduler.get_statuses(
                batch_id=batch_state.batch_id,
                timeout=timeout,
                return_partial_tasks=self.config.explorer.over_rollout.return_partial_tasks,
            )
            for task_id, status in enumerate(statuses):
                if task_id in batch_state.statuses:
                    continue
                batch_state.statuses[task_id] = status
            await self._discard_recorded_experiences(str(batch_state.batch_id))
            return self._finish_batch(batch_state, pipeline_metrics={})

    async def abort_batch(
        self,
        batch_id: BatchId,
        *,
        reason: str,
        keep_partial_results: bool = False,
    ) -> None:
        """Abort one batch and cleanup its running and staged state."""
        scheduler = self._require_scheduler()
        batch_state = self.pending_batches.get(batch_id)
        if batch_state is None:
            return
        if batch_state.state in {BatchLifecycleState.FINALIZED, BatchLifecycleState.ABORTED}:
            return

        self.logger.warning("Abort batch %s: %s", batch_id, reason)
        await scheduler.cleanup_batch(
            batch_id,
            return_partial_tasks=keep_partial_results,
            restart_runners=True,
        )
        scheduler.discard_completed_results(batch_id)
        await self._discard_recorded_experiences(str(batch_id))

        batch_state.state = BatchLifecycleState.ABORTED
        batch_state.final_result = self._build_batch_result(batch_state, pipeline_metrics={})
        self.pending_batches.pop(batch_id, None)

    async def process_experiences(self, payloads: list[bytes]) -> dict:
        """Process one batch of experience payloads through the pipeline."""
        if self.experience_pipeline is None:
            raise RuntimeError("Experience pipeline is not initialized.")
        if not payloads:
            return {}
        return await self.experience_pipeline.process_serialized_chunks(payloads)

    @classmethod
    def get_actor(cls, config: Config) -> ActorHandle["RolloutCoordinator"]:
        """Init rollout coordinator for the task-event-completion path."""
        return (
            ray.remote(RolloutCoordinator)
            .options(namespace=config.ray_namespace)
            .remote(
                config,
            )
        )

    def _get_batch_state(self, batch_id: BatchId, *, expected_type: BatchType) -> BatchState:
        """Return one registered batch and validate its type."""
        batch_state = self.pending_batches.get(batch_id)
        if batch_state is None:
            raise KeyError(f"Batch {batch_id} is not registered.")
        if batch_state.batch_type != expected_type:
            raise ValueError(
                f"Batch {batch_id} is {batch_state.batch_type}, expected {expected_type}."
            )
        return batch_state

    def _get_existing_final_result(self, batch_state: BatchState) -> Optional[dict]:
        """Reuse an in-flight final result or synthesize an abort result."""

        if batch_state.final_result is not None:
            return dict(batch_state.final_result)
        if batch_state.state != BatchLifecycleState.ABORTED:
            return None
        batch_state.final_result = self._build_batch_result(batch_state, pipeline_metrics={})
        return dict(batch_state.final_result)

    async def _finalize_train_batch(
        self, batch_state: BatchState, *, timeout: Optional[float]
    ) -> dict:
        """Finalize one train batch."""
        async with batch_state.finalize_lock:
            existing_result = self._get_existing_final_result(batch_state)
            if existing_result is not None:
                return existing_result

            scheduler = self._require_scheduler()
            scheduled_num = batch_state.expected_task_count
            statuses, payload_chunks = await scheduler.get_payload_results(
                batch_id=batch_state.batch_id,
                min_num=batch_state.min_wait_num,
                timeout=timeout,
                clear_timeout_tasks=False,
                return_partial_tasks=self.config.explorer.over_rollout.return_partial_tasks,
            )
            completed_count = len(statuses)
            if scheduled_num == 0:
                is_complete = True
            else:
                if completed_count == 0:
                    raise TimeoutError(f"Timeout waiting for batch {batch_state.batch_id}.")
                if batch_state.min_wait_num is None and completed_count < scheduled_num:
                    raise TimeoutError(f"Timeout waiting for batch {batch_state.batch_id}.")

                batch_state.statuses = {task_id: status for task_id, status in enumerate(statuses)}
                is_complete = completed_count >= scheduled_num

            batch_state.state = BatchLifecycleState.FINALIZING
            try:
                pipeline_metrics = await self.process_experiences(payload_chunks)
                await self._discard_recorded_experiences(str(batch_state.batch_id))
                if not is_complete:
                    await self._cleanup_train_batch_runtime(batch_state)
            except Exception:
                batch_state.state = self._get_active_batch_state(batch_state)
                raise

            return self._finish_batch(batch_state, pipeline_metrics=pipeline_metrics)

    async def _discard_recorded_experiences(self, prefix: str) -> None:
        """Block future writes and delete recorded experiences for a prefix.

        Blocking happens before deleting across all rollout ranks so that any
        in-flight write that lands after the delete is dropped by the store
        instead of reappearing as an orphan. The block flag persists on each
        rollout actor (batch_id is never reused), so the prefix stays
        unwritable for the lifetime of the process.
        """
        block_results = await asyncio.gather(
            *[
                actor.block_experience_records.remote(prefix=prefix)
                for actor in self._rollout_actors.values()
            ],
            return_exceptions=True,
        )
        delete_results = await asyncio.gather(
            *[
                actor.delete_experience_records.remote(prefix=prefix)
                for actor in self._rollout_actors.values()
            ],
            return_exceptions=True,
        )
        for result in [*block_results, *delete_results]:
            if isinstance(result, Exception):
                self.logger.error("records cleanup on rollout actor failed: %s", result)

    def _finish_batch(
        self,
        batch_state: BatchState,
        pipeline_metrics: dict,
    ) -> dict:
        """Persist one terminal result and evict the batch from active state."""
        self._require_scheduler().discard_completed_results(batch_state.batch_id)
        batch_state.state = BatchLifecycleState.FINALIZED
        batch_state.final_result = self._build_batch_result(batch_state, pipeline_metrics)
        self.pending_batches.pop(batch_state.batch_id, None)
        return dict(batch_state.final_result)

    def _get_active_batch_state(self, batch_state: BatchState) -> BatchLifecycleState:
        """Return the active lifecycle state to restore after a failed finalize attempt."""
        if batch_state.expected_task_count == 0:
            return BatchLifecycleState.PENDING
        return BatchLifecycleState.RUNNING

    async def _cleanup_train_batch_runtime(self, batch_state: BatchState) -> None:
        """Drop unfinished train work after a non-complete finalize result."""
        scheduler = self._require_scheduler()
        await scheduler.cleanup_batch(
            batch_state.batch_id,
            return_partial_tasks=False,
            restart_runners=True,
        )
        await self._discard_recorded_experiences(str(batch_state.batch_id))

    def _build_batch_result(
        self,
        batch_state: BatchState,
        pipeline_metrics: dict,
    ) -> dict:
        """Build the public finalize result returned to Explorer."""

        metrics = dict(pipeline_metrics)
        status_metrics = [
            status.metrics[0] for status in batch_state.statuses.values() if status.metrics
        ]
        if batch_state.batch_type == "train":
            if status_metrics:
                metrics.update(aggregate_metrics(status_metrics, prefix="rollout"))
            metrics["rollout/finished_task_count"] = float(batch_state.completed_task_count)
        else:
            prefix = self._eval_metric_prefix(batch_state.batch_id)
            if status_metrics:
                metrics.update(
                    aggregate_eval_metrics(
                        status_metrics,
                        prefix=prefix,
                        detailed_stats=self.detailed_stats,
                    )
                )
            metrics[f"{prefix}/finished_task_count"] = float(batch_state.completed_task_count)

        return {
            "batch_id": batch_state.batch_id,
            "batch_type": batch_state.batch_type,
            "finished_task_count": batch_state.completed_task_count,
            "metrics": metrics,
        }

    def _eval_metric_prefix(self, batch_id: BatchId) -> str:
        """Return the metric namespace prefix for one eval batch."""
        batch_name = str(batch_id)
        if "/" in batch_name:
            batch_name = batch_name.split("/", 1)[1]
        return f"eval/{batch_name}"
