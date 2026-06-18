"""Scheduler for rollout tasks."""

import asyncio
import re
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import ray

from trinity.common.config import Config
from trinity.common.models import InferenceModel
from trinity.common.workflows import Task
from trinity.explorer.workflow_runner import Status, WorkflowRunner
from trinity.utils.log import get_logger


@dataclass
class TaskWrapper:
    """A wrapper for a task.
    Each task can run multiple times (repeat_times) on same or different runners.
    """

    task: Task
    batch_id: Union[int, str]
    sub_task_num: int = 1  # number of sub tasks splitted from this task
    # if max_repeat_times_per_runner is set, one task may be splitted into multiple sub tasks
    finished_sub_task_num: int = 0
    completed_runs: int = 0
    total_runs: int = 0  # total planned runs for the whole task
    metrics: List[Dict[str, float]] = field(default_factory=list)
    experience_payloads: List[bytes] = field(default_factory=list)
    first_error: Optional[str] = None
    emitted: bool = False


@dataclass(frozen=True)
class CompletedTaskResult:
    """A completed task result stored by batch and task id."""

    batch_id: Union[int, str]
    task_id: Union[int, str]
    status: Status
    experience_payloads: List[bytes] = field(default_factory=list)


@dataclass
class RunningTaskState:
    """Per-future execution state tracked while one task is running."""

    task: TaskWrapper
    runner_id: int
    restart_runner_on_cancel: bool = True


# Adapted from verl/trainer/ppo/metric_utils.py
def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calculate_task_level_metrics(metrics: List[Dict], is_eval: bool) -> Dict[str, float]:
    """Calculate task level metrics (mean) from multiple runs of the same task.

    Args:
        metrics (`List[Dict]`): A list of metric dictionaries from multiple runs of the same task.
        is_eval (`bool`): Whether this is an evaluation task.

    Returns:
        `Dict[str, float]`: A dictionary of aggregated metrics, where each metric is averaged over all runs.
    """
    if not metrics:
        return {}
    aggregated_metrics: Dict[str, List[float]] = defaultdict(list)
    for m in metrics:
        for key, value in m.items():
            if isinstance(value, (int, float)):
                aggregated_metrics[key].append(value)
    if is_eval:
        result = {}
        for key, values in aggregated_metrics.items():
            if "time/task_execution" in key or "time/run_execution" in key:
                result[key] = sum(values) / len(values)
                continue

            n_values = len(values)
            result[f"{key}/mean@{n_values}"] = np.mean(values)
            result[f"{key}/std@{n_values}"] = np.std(values)

            if n_values > 1:
                ns = []
                n = 2
                while n < n_values:
                    ns.append(n)
                    n *= 2
                ns.append(n_values)

                for n in ns:
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                        data=values, subset_size=n, reduce_fns=[np.max, np.min], seed=42
                    )
                    result[f"{key}/best@{n}"] = bon_mean
                    result[f"{key}/worst@{n}"] = won_mean
        return result
    else:
        return {
            key: sum(values) / len(values) for key, values in aggregated_metrics.items() if values
        }


class RunnerWrapper:
    """A wrapper for a WorkflowRunner"""

    def __init__(
        self,
        runner_id: int,
        rollout_model: InferenceModel,
        auxiliary_models: List[InferenceModel],
        config: Config,
    ):
        self.logger = get_logger(__name__)
        self.runner_id = runner_id
        self.rollout_model = rollout_model
        self.auxiliary_models = auxiliary_models
        self.config = config
        self.retry_times = config.explorer.max_retry_times
        self.timeout = config.explorer.max_timeout
        self.namespace = config.ray_namespace
        self.runner = self._create_runner()
        self.state = {}

    def _create_runner(self):
        return (
            ray.remote(WorkflowRunner)
            .options(
                num_cpus=0,
                namespace=self.namespace,
                scheduling_strategy="SPREAD",
                runtime_env={
                    "env_vars": self.config.explorer.env_vars,
                },
            )
            .remote(
                self.config,
                self.rollout_model,
                self.auxiliary_models,
                self.runner_id,
            )
        )

    async def prepare(self):
        await self.runner.prepare.remote()

    async def update_state(self) -> None:
        """Get the runner state."""
        self.state = await self.runner.get_runner_state.remote()
        self.state["running_time"] = time.time() - self.state.get("begin_time", time.time())

    async def run_with_retry(
        self,
        task: TaskWrapper,
        repeat_times: int,
        run_id_base: int,
        timeout: float,
        collect_partial_runs: bool,
    ) -> Tuple[Status, bytes, int, float]:
        """
        Args:
            task (`TaskWrapper`): The task to run.
            repeat_times (`int`): The number of times to repeat the task.
            run_id_base (`int`): The base run id for this task runs.
            timeout (`float`): The timeout for each task run.

        Returns:
            `Status`: The return status of the task.
            `List`: The experiences generated by the task.
            `int`: The runner_id of current runner.
            `float`: The time taken to run the task.
        """
        last_exception_msg = None
        await self.runner.__ray_ready__.remote()
        start_time = time.time()
        status = Status(completed_runs=0, total_runs=repeat_times, metrics=list())
        exp_payload = b""
        run_task_ref = None
        task2run = replace(
            task.task,
            rollout_args=replace(
                task.task.rollout_args,
                n=repeat_times,
            ),
        )
        try:
            for attempt in range(self.retry_times + 1):
                try:
                    run_task_ref = self.runner.run_task.remote(
                        task=task2run,
                        batch_id=str(task.batch_id),
                        repeat_times=repeat_times,
                        run_id_base=run_id_base,
                        collect_partial_runs=collect_partial_runs,
                    )
                    status, exp_payload = await asyncio.wait_for(
                        run_task_ref,
                        timeout=timeout,
                    )
                    run_task_ref = None
                    if status.ok:
                        break
                    if collect_partial_runs and status.completed_runs > 0:
                        self.logger.warning(
                            "Task returned partial success; skipping retry to avoid "
                            "re-running successful runs. %s",
                            status.message,
                        )
                        break
                    else:
                        self.logger.error(status.message)
                except asyncio.TimeoutError:
                    run_task_ref = None
                    last_exception_msg = f"Timeout ({timeout} s) when running task of batch {task.batch_id} at runner {self.runner_id} at attempt {attempt + 1}: {task.task}"
                    self.logger.error(last_exception_msg)
                    status = Status(
                        completed_runs=0,
                        total_runs=repeat_times,
                        metrics=list(),
                        message=last_exception_msg,
                    )
                except asyncio.CancelledError:
                    if run_task_ref is not None:
                        ray.cancel(run_task_ref, force=False)
                    raise
                except Exception:
                    run_task_ref = None
                    last_exception_msg = traceback.format_exc()
                    self.logger.warning(
                        f"Task execution attempt {attempt + 1} failed:\n{last_exception_msg}"
                    )
                    status = Status(
                        completed_runs=0,
                        total_runs=repeat_times,
                        metrics=list(),
                        message=last_exception_msg,
                    )
        finally:
            end_time = time.time()
            status.metrics.append({"time/task_execution": end_time - start_time})
        return status, exp_payload, self.runner_id, end_time - start_time

    async def restart_runner(self):
        old_runner = self.runner
        self.runner = self._create_runner()
        await self.runner.prepare.remote()
        try:
            ray.kill(old_runner)
        except Exception:
            pass


def sort_batch_id(batch_id: Union[int, str]):
    """Priority of batch_id"""
    # TODO: avoid sort the batch_id every time
    if isinstance(batch_id, int):
        return (batch_id, 0)
    else:
        match = re.match(r"^(\d+)", batch_id)
        if match:
            num = int(match.group(1))
            return (num, 1)
        else:
            return (float("inf"), 1)


class Scheduler:
    """Scheduler for rollout tasks.

    Supports scheduling tasks to multiple runners, retrying failed tasks,
    and collecting results at different levels.
    """

    def __init__(
        self,
        config: Config,
        rollout_model: List[InferenceModel],
        auxiliary_models: Optional[List[List[InferenceModel]]] = None,
    ):
        self.logger = get_logger(__name__)
        self.config = config
        self.rollout_model = rollout_model
        self.auxiliary_models = auxiliary_models or []
        self.namespace = ray.get_runtime_context().namespace
        self.default_timeout = config.explorer.max_timeout * (config.explorer.max_retry_times + 1)
        self.max_retry_times = config.explorer.max_retry_times
        self.max_repeat_times = config.explorer.max_repeat_times_per_runner
        self.default_batch_size = config.buffer.batch_size
        self.running = False

        self.runner_num = len(rollout_model) * config.explorer.runner_per_model
        self.runners: Dict[int, RunnerWrapper] = dict()
        self.idle_runners: set[int] = set()  # runner_id of idle runners
        self.busy_runners: Dict[int, RunningTaskState] = dict()  # runner_id -> running state

        self.pending_tasks: Dict[Union[int, str], deque] = defaultdict(
            deque
        )  # batch_id -> (task, repeat_times, run_id_base)
        self.running_tasks: Dict[Union[int, str], set[asyncio.Future]] = defaultdict(
            set
        )  # batch_id -> futures
        self.task_num_map: Dict[Union[int, str], int] = defaultdict(
            int
        )  # batch_id -> tasks scheduled under this batch_id
        self.running_task_state_map: Dict[asyncio.Future, RunningTaskState] = dict()
        self.batch_is_eval_map: Dict[Union[int, str], bool] = dict()
        self.completed_tasks: Dict[
            Union[int, str], Dict[Union[int, str], CompletedTaskResult]
        ] = defaultdict(
            dict
        )  # batch_id -> results
        self.background_tasks: set[asyncio.Task] = set()

        self.scheduler_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None

        self.total_running_time = 0.0
        self.total_completed_steps = 0
        self.total_completed_sub_tasks = 0
        self.total_completed_tasks = 0

    async def _create_runner(
        self,
        runner_id: int,
    ):
        runner = RunnerWrapper(
            runner_id=runner_id,
            rollout_model=self.rollout_model[runner_id % len(self.rollout_model)],
            auxiliary_models=[
                self.auxiliary_models[j][runner_id % len(self.auxiliary_models[j])]
                for j in range(len(self.auxiliary_models))
            ],
            config=self.config,
        )
        await runner.prepare()
        self.runners[runner_id] = runner
        self.idle_runners.add(runner_id)

    async def _restart_runner(self, runner_id: int):
        """Restart a runner."""
        await self.runners[runner_id].restart_runner()

        if runner_id in self.busy_runners:
            running_state = self.busy_runners.pop(runner_id)
            task = running_state.task
            self.logger.warning(
                f"Runner {runner_id} failed to run task at batch_id {task.batch_id}: {task.task.raw_task}"
            )

        self.idle_runners.add(runner_id)
        self.logger.info(f"Runner {runner_id} restarted.")

    def _schedule_runner_restart(self, runner_id: int) -> None:
        restart_task = asyncio.create_task(self._restart_runner(runner_id))
        self.background_tasks.add(restart_task)
        restart_task.add_done_callback(self.background_tasks.discard)

    async def _scheduler_loop(self) -> None:
        self.logger.info("Scheduler loop started.")
        while self.running:
            try:
                await self._schedule_pending_tasks()
                await asyncio.sleep(0.01)
            except Exception:
                self.logger.error(f"Error in scheduler loop:\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)
        self.logger.info("Scheduler loop stopped.")

    async def _monitor_runner_state_loop(self) -> None:
        interval = self.config.explorer.runner_state_report_interval
        if interval <= 0:
            self.logger.info("Runner state monitoring loop disabled.")
            return

        self.logger.info("Runner state monitoring loop started.")
        while self.running:
            try:
                await asyncio.gather(*[runner.update_state() for runner in self.runners.values()])
                self.print_all_state()
            except Exception:
                self.logger.error(
                    f"Error in runner state monitoring loop:\n{traceback.format_exc()}"
                )
            await asyncio.sleep(interval)
        self.logger.info("Runner state monitoring loop stopped.")

    async def _schedule_pending_tasks(self) -> None:
        if not self.idle_runners:
            return

        # TODO: Support more advanced scheduling strategies
        for batch_id in sorted(self.pending_tasks.keys(), key=sort_batch_id):
            task_queue = self.pending_tasks[batch_id]

            while task_queue and self.idle_runners:
                task, repeat_times, run_id_base = task_queue.pop()
                runner_id = self.idle_runners.pop()
                future = asyncio.create_task(
                    self.runners[runner_id].run_with_retry(
                        task,
                        repeat_times=repeat_times,
                        run_id_base=run_id_base,
                        timeout=self.dynamic_timeout(),
                        collect_partial_runs=self.config.explorer.over_rollout.return_partial_tasks,
                    )
                )
                running_state = RunningTaskState(task=task, runner_id=runner_id)
                self.busy_runners[runner_id] = running_state
                self.running_task_state_map[future] = running_state
                future.add_done_callback(self.task_done_callback)
                self.running_tasks[batch_id].add(future)

            if not task_queue:
                del self.pending_tasks[batch_id]

    def task_done_callback(self, async_task: asyncio.Task):
        running_state = self.running_task_state_map.pop(async_task)
        task = running_state.task
        runner_id = running_state.runner_id
        if async_task.cancelled():
            if not running_state.restart_runner_on_cancel:
                self.busy_runners.pop(runner_id, None)
                self.idle_runners.add(runner_id)
        elif async_task.exception():
            self.logger.error(f"Task {task.task.task_id} failed: {async_task.exception()}")
            self._schedule_runner_restart(runner_id)
        else:
            status, exp_payload, runner_id, run_time = async_task.result()
            if not task.task.is_eval:
                self.total_running_time += run_time
                self.total_completed_sub_tasks += 1
            self._accumulate_task_result(task, status, exp_payload)
            self.busy_runners.pop(runner_id, None)
            self.idle_runners.add(runner_id)
            # If all sub runs in a task are completed
            if task.finished_sub_task_num == task.sub_task_num:
                if not task.task.is_eval:
                    self.total_completed_tasks += 1
                self._emit_task_result(task)
                self.logger.debug(f"Task completed (batch_id {task.batch_id}).")

        if task.batch_id in self.running_tasks:
            self.running_tasks[task.batch_id].remove(async_task)
            if not self.running_tasks[task.batch_id]:
                del self.running_tasks[task.batch_id]

    def _accumulate_task_result(
        self, task: TaskWrapper, status: Status, experience_payload: bytes
    ) -> None:
        task.finished_sub_task_num += 1
        task.completed_runs += status.completed_runs
        task.metrics.extend(status.metrics)
        if experience_payload:
            task.experience_payloads.append(experience_payload)
        if not status.ok and task.first_error is None:
            task.first_error = status.message

    def _build_task_result(self, task: TaskWrapper) -> Tuple[Status, List[bytes]]:
        if task.completed_runs < task.total_runs:
            message = f"{task.completed_runs}/{task.total_runs} runs completed successfully."
            if task.first_error:
                message = f"{message} First error: {task.first_error}"
            else:
                message = f"{message} Remaining runs were cancelled during scheduler cleanup."
        else:
            message = None
        status = Status(
            completed_runs=task.completed_runs,
            total_runs=task.total_runs,
            metrics=[calculate_task_level_metrics(task.metrics, task.task.is_eval)],
            message=message,
        )
        return status, list(task.experience_payloads)

    def _emit_task_result(self, task: TaskWrapper) -> None:
        if task.emitted:
            return
        status, experience_payloads = self._build_task_result(task)
        task_id = task.task.task_id
        completed_result = CompletedTaskResult(
            batch_id=task.batch_id,
            task_id=task_id,
            status=status,
            experience_payloads=experience_payloads,
        )
        self.completed_tasks[task.batch_id][task_id] = completed_result
        task.emitted = True

    def discard_completed_results(self, batch_id: Union[int, str]) -> None:
        """Drop cached completed results for one batch."""

        self.completed_tasks.pop(batch_id, None)
        self.task_num_map.pop(batch_id, None)
        self.batch_is_eval_map.pop(batch_id, None)

    def _collect_incomplete_tasks(self, batch_id: Union[int, str]) -> List[TaskWrapper]:
        tasks = {}
        for task, _, _ in self.pending_tasks.get(batch_id, deque()):
            tasks[id(task)] = task
        for future in self.running_tasks.get(batch_id, set()):
            running_state = self.running_task_state_map.get(future)
            if running_state is not None:
                tasks[id(running_state.task)] = running_state.task
        return list(tasks.values())

    def _emit_partial_tasks_for_batch(self, batch_id: Union[int, str]) -> None:
        for task in self._collect_incomplete_tasks(batch_id):
            if task.emitted or task.completed_runs <= 0:
                continue
            self._emit_task_result(task)
            self.logger.debug(
                f"Task partially completed and emitted (batch_id {task.batch_id}, task_id {task.task.task_id})."
            )

    def _clear_timeout_tasks(self, batch_id: Union[int, str]) -> List[asyncio.Future]:
        cancelled_futures = []
        if batch_id in self.pending_tasks:
            self.logger.info(f"Clear timeout pending tasks at batch_id {batch_id}.")
            del self.pending_tasks[batch_id]
        if batch_id in self.running_tasks:
            self.logger.info(f"Clear timeout running tasks at batch_id {batch_id}.")
            for future in self.running_tasks[batch_id]:
                cancelled_futures.append(future)
                future.cancel()
            del self.running_tasks[batch_id]
        self.task_num_map.pop(batch_id, None)
        return cancelled_futures

    def _mark_running_tasks_for_cleanup(
        self, batch_id: Union[int, str], restart_runners: bool
    ) -> None:
        for future in self.running_tasks.get(batch_id, set()):
            running_state = self.running_task_state_map.get(future)
            if running_state is not None:
                running_state.restart_runner_on_cancel = restart_runners

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        # bounded-concurrency runner prepare with in-slot retry (avoids glibc getenv races)
        _sem = asyncio.Semaphore(max(1, self.config.explorer.runner_prepare_concurrency))
        _retries = max(0, self.config.explorer.runner_prepare_max_retries)

        async def _create_limited(i: int) -> None:
            async with _sem:
                for attempt in range(_retries + 1):
                    try:
                        await self._create_runner(i)
                        return
                    except Exception as e:
                        if attempt == _retries:
                            raise
                        self.logger.warning(
                            f"Runner {i} prepare failed ({attempt + 1}/{_retries + 1}), retrying: {e}"
                        )
                        await asyncio.sleep(2.0 * (attempt + 1))

        await asyncio.gather(*[_create_limited(i) for i in range(self.runner_num)])
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        ready_refs = [runner.runner.__ray_ready__.remote() for runner in self.runners.values()]
        await asyncio.gather(*ready_refs)
        self.monitor_task = asyncio.create_task(self._monitor_runner_state_loop())
        self.logger.info(f"Starting Scheduler with {self.runner_num} runners")

    async def stop(self) -> None:
        if not self.running:
            return

        self.running = False
        all_running_futures = []
        for futures in self.running_tasks.values():
            all_running_futures.extend(futures)

        if all_running_futures:
            self.logger.info(f"Waiting for {len(all_running_futures)} running tasks to complete...")
            await asyncio.gather(*all_running_futures, return_exceptions=True)

        if self.background_tasks:
            await asyncio.gather(*list(self.background_tasks), return_exceptions=True)

        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Scheduler stopped")

    def schedule(self, tasks: List[Task], batch_id: Union[int, str]) -> None:
        """Schedule the provided tasks.

        Args:
            tasks (`List[Task]`): The tasks to schedule.
            batch_id (`Union[int, str]`):
                The id of provided tasks. In most cases, it should be current step number for
                training tasks and "<current_step_num>/<eval_taskset_name>" for eval tasks.
        """
        if not tasks:
            return
        self.batch_is_eval_map[batch_id] = tasks[0].is_eval
        self.task_num_map[batch_id] += len(tasks)
        self._split_and_submit_tasks(tasks, batch_id=batch_id)

    def _split_and_submit_tasks(self, tasks: List[Task], batch_id: Union[int, str]) -> None:
        for i, task in enumerate(tasks):
            assert task.repeat_times is not None, "Task repeat_times should not be None"
            task_wrapper = TaskWrapper(
                task=replace(task, batch_id=batch_id, task_id=i),
                batch_id=batch_id,
                total_runs=task.repeat_times,
            )
            if self.max_repeat_times is None:
                task_wrapper.sub_task_num = 1
                self.pending_tasks[batch_id].appendleft((task_wrapper, task.repeat_times, 0))
                continue
            sub_tasks = []
            for run_id_base in range(0, task.repeat_times, self.max_repeat_times):
                repeat_times = min(self.max_repeat_times, task.repeat_times - run_id_base)
                sub_tasks.append((task_wrapper, repeat_times, run_id_base))
            task_wrapper.sub_task_num = len(sub_tasks)
            self.pending_tasks[batch_id].extendleft(sub_tasks)

    def dynamic_timeout(self, timeout: Optional[float] = None) -> float:
        """Calculate dynamic timeout based on historical data."""
        max_timeout = timeout or self.default_timeout
        if not self.config.explorer.dynamic_timeout.enable:
            return max_timeout
        if (
            self.total_completed_steps < self.config.explorer.dynamic_timeout.warmup_min_steps
            or self.total_completed_sub_tasks == 0
        ):
            return max_timeout
        avg_time_per_task = self.total_running_time / self.total_completed_sub_tasks
        return min(
            max_timeout,
            avg_time_per_task * self.config.explorer.dynamic_timeout.ratio,
        )

    async def _cleanup_batch(
        self,
        batch_id: Union[int, str],
        return_partial_tasks: bool = False,
        restart_runners: bool = True,
    ) -> None:
        """Clear unfinished tasks for a batch and optionally restart associated runners."""
        if return_partial_tasks:
            self._emit_partial_tasks_for_batch(batch_id)
        self._mark_running_tasks_for_cleanup(batch_id, restart_runners=restart_runners)
        cancelled_futures = self._clear_timeout_tasks(batch_id=batch_id)
        if cancelled_futures:
            await asyncio.gather(*cancelled_futures, return_exceptions=True)
        if not restart_runners:
            return
        runners_to_restart = [
            runner_id
            for runner_id, running_state in list(self.busy_runners.items())
            if running_state.task.batch_id == batch_id
        ]
        if runners_to_restart:
            await asyncio.gather(*[self._restart_runner(rid) for rid in runners_to_restart])

    def _resolve_result_target(
        self, batch_id: Union[int, str], min_num: Optional[int]
    ) -> Tuple[int, int]:
        scheduled_num = self.task_num_map.get(batch_id, 0)
        if min_num is None:
            return scheduled_num, scheduled_num
        if min_num > scheduled_num:
            self.logger.warning(
                f"Requested min_num {min_num} is greater than scheduled tasks {scheduled_num} at batch_id {batch_id}. Adjusting min_num to {scheduled_num}."
            )
            return scheduled_num, scheduled_num
        return scheduled_num, min_num

    def _finalize_dynamic_timeout_step(
        self, batch_id: Union[int, str], scheduled_num: int, completed_count: int
    ) -> None:
        if batch_id in self.pending_tasks or batch_id in self.running_tasks:
            return
        is_eval = self.batch_is_eval_map.pop(batch_id, False)
        if not is_eval and completed_count >= scheduled_num:
            self.total_completed_steps += 1

    async def _wait_for_batch_results(
        self,
        batch_id: Union[int, str],
        min_num: int,
        scheduled_num: int,
        timeout: float,
        clear_timeout_tasks: bool,
        return_partial_tasks: bool,
    ) -> bool:
        start_time = time.time()
        min_threshold_reached_time = None
        while time.time() - start_time <= timeout:
            completed_count = len(self.completed_tasks.get(batch_id, {}))
            if completed_count >= min_num:
                min_threshold_reached_time = min_threshold_reached_time or time.time()
                if completed_count >= scheduled_num:
                    return False
                if (
                    time.time() - min_threshold_reached_time
                    >= self.config.explorer.over_rollout.wait_after_min
                ):
                    if clear_timeout_tasks:
                        await self._cleanup_batch(
                            batch_id,
                            return_partial_tasks=return_partial_tasks,
                            restart_runners=False,
                        )
                    return False
            await asyncio.sleep(0.1)
        return True

    def _collect_batch_results(self, batch_id: Union[int, str]) -> Tuple[List[Status], List[bytes]]:
        statuses = []
        payload_chunks = []
        completed_results = list(self.completed_tasks.get(batch_id, {}).values())
        for result in completed_results:
            statuses.append(result.status)
            if result.experience_payloads:
                payload_chunks.extend(result.experience_payloads)

        return statuses, payload_chunks

    async def drain_batch_payload_results(
        self, batch_id: Union[int, str]
    ) -> Tuple[List[Status], List[bytes]]:
        """Drain cached completed results for one batch."""

        statuses, payload_chunks = self._collect_batch_results(batch_id)

        if batch_id in self.completed_tasks:
            del self.completed_tasks[batch_id]

        completed_count = len(statuses)
        scheduled_num = self.task_num_map.get(batch_id, 0)
        self._finalize_dynamic_timeout_step(batch_id, scheduled_num, completed_count)
        return statuses, payload_chunks

    async def _get_batch_payload_results(
        self,
        batch_id: Union[int, str],
        *,
        min_num: Optional[int],
        timeout: Optional[float],
        clear_timeout_tasks: bool,
        return_partial_tasks: bool,
    ) -> Tuple[List[Status], List[bytes]]:
        """Wait for one batch and drain its completed payload chunks."""

        timeout = timeout or self.default_timeout
        scheduled_num, min_num = self._resolve_result_target(batch_id, min_num)

        self.logger.debug(f"Waiting for {min_num} tasks to complete...")
        timed_out = await self._wait_for_batch_results(
            batch_id=batch_id,
            min_num=min_num,
            scheduled_num=scheduled_num,
            timeout=timeout,
            clear_timeout_tasks=clear_timeout_tasks,
            return_partial_tasks=return_partial_tasks,
        )
        if timed_out:
            self.logger.error(
                f"Timed out waiting for tasks at batch {batch_id} to complete after {timeout} seconds"
            )
            if clear_timeout_tasks:
                await self._cleanup_batch(
                    batch_id,
                    return_partial_tasks=return_partial_tasks,
                    restart_runners=True,
                )

        completed_count = len(self.completed_tasks.get(batch_id, {}))
        if completed_count < min_num:
            self.logger.warning(
                f"Timeout reached, only {completed_count}/{min_num} tasks completed"
            )

        return await self.drain_batch_payload_results(batch_id)

    async def get_payload_results(
        self,
        batch_id: Union[int, str],
        min_num: Optional[int] = None,
        timeout: Optional[float] = None,
        clear_timeout_tasks: bool = True,
        return_partial_tasks: bool = False,
    ) -> Tuple[List[Status], List[bytes]]:
        """Wait for one batch and return task statuses plus serialized payload chunks."""

        return await self._get_batch_payload_results(
            batch_id=batch_id,
            min_num=min_num,
            timeout=timeout,
            clear_timeout_tasks=clear_timeout_tasks,
            return_partial_tasks=return_partial_tasks,
        )

    async def get_statuses(
        self,
        batch_id: Union[int, str],
        min_num: Optional[int] = None,
        timeout: Optional[float] = None,
        clear_timeout_tasks: bool = True,
        return_partial_tasks: bool = False,
    ) -> List[Status]:
        """Wait for one batch and return only task statuses without materializing experiences."""

        statuses, _ = await self._get_batch_payload_results(
            batch_id=batch_id,
            min_num=min_num,
            timeout=timeout,
            clear_timeout_tasks=clear_timeout_tasks,
            return_partial_tasks=return_partial_tasks,
        )
        return statuses

    async def abort_batch(
        self,
        batch_id: Union[int, str],
        return_partial_tasks: bool = False,
        restart_runners: bool = True,
    ) -> None:
        """Abort one batch and cleanup unfinished scheduler state."""
        await self._cleanup_batch(
            batch_id,
            return_partial_tasks=return_partial_tasks,
            restart_runners=restart_runners,
        )

    def has_step(self, batch_id: Union[int, str]) -> bool:
        return (
            batch_id in self.completed_tasks
            or batch_id in self.pending_tasks
            or batch_id in self.running_tasks
        )

    async def wait_all(
        self, timeout: Optional[float] = None, clear_timeout_tasks: bool = True
    ) -> None:
        """Wait for all tasks to complete without poping results. If timeout reached, raise TimeoutError.

        Args:
            timeout (`float`): timeout in seconds. Raise `TimeoutError` when no new tasks is completed within timeout.
            clear_timeout_tasks (`bool`): Whether to clear timeout tasks.
        """
        timeout = timeout or self.default_timeout
        start_time = time.time()

        self.logger.debug("Waiting for all tasks to complete...")
        last_completed_count = 0
        while time.time() - start_time < timeout:
            has_pending = bool(self.pending_tasks)
            has_running = bool(self.running_tasks)

            if not has_pending and not has_running:
                self.logger.debug("All tasks completed successfully")
                return

            completed_count = sum(len(tasks) for tasks in self.completed_tasks.values())
            if completed_count != last_completed_count:
                # flush timeout when new tasks are completed
                start_time = time.time()
                last_completed_count = completed_count

            await asyncio.sleep(0.1)

        pending_count = sum(len(tasks) for tasks in self.pending_tasks.values())
        running_count = sum(len(futures) for futures in self.running_tasks.values())
        error_msg = f"Timeout after {timeout} seconds. Still have {pending_count} pending tasks and {running_count} running tasks."
        self.logger.error(error_msg)

        if clear_timeout_tasks:
            batch_ids_to_abort = self.pending_tasks.keys() | self.running_tasks.keys()
            for batch_id in batch_ids_to_abort:
                self._clear_timeout_tasks(batch_id)
            asyncio.gather(
                *[self._restart_runner(runner_id) for runner_id in self.busy_runners.keys()]
            )

        raise TimeoutError(error_msg)

    def get_key_state(self, key: str) -> Dict:
        """Get the scheduler state.

        Args:
            key (`str`): The key of the state to get.

        Returns:
            `Dict`: A dictionary of runner ids to their state for the given key.
        """
        result = {}
        for runner in self.runners.values():
            runner_state = runner.state
            if runner_state and key in runner_state:
                result[runner.runner_id] = runner_state[key]
        return result

    def get_runner_state(self, runner_id: int) -> Dict:
        """Get the scheduler state.

        Args:
            runner_id (`int`): The id of the runner.

        Returns:
            `Dict`: The state of the runner.
        """
        runner = self.runners.get(runner_id, None)
        if runner:
            return runner.state
        else:
            return {}

    def get_all_state(self) -> Dict:
        """Get all runners' state.

        Returns:
            `Dict`: The state of all runners.
        """
        result = {}
        for runner in self.runners.values():
            runner_state = runner.state
            if runner_state:
                result[runner.runner_id] = runner_state
        return result

    def print_all_state(self) -> None:
        """Print all runners' state in a clear, aligned table format."""
        all_keys = set()
        for runner in self.runners.values():
            runner_state = runner.state
            if runner_state:
                all_keys.update(runner_state.keys())
        all_keys = sorted(all_keys)
        # Prepare header
        header = ["runner_id"] + all_keys  # type: ignore [operator]
        # Prepare rows
        rows = []
        for runner in self.runners.values():
            runner_state = runner.state or {}
            row = [str(runner.runner_id)]
            for key in all_keys:
                value = runner_state.get(key, "-")
                row.append(str(value))
            rows.append(row)
        # Calculate column widths
        col_widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
        # Print header
        header_line = " | ".join(str(h).ljust(w) for h, w in zip(header, col_widths))
        self.logger.info(header_line)
        self.logger.info("-+-".join("-" * w for w in col_widths))
        # Print each row
        for row in rows:
            line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            self.logger.info(line)
