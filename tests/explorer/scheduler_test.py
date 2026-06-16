import asyncio
import time
import unittest
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Sequence
from unittest.mock import AsyncMock, patch

import ray
import torch
from parameterized import parameterized

from tests.tools import get_template_config
from trinity.common.config import ExperienceBufferConfig
from trinity.common.constants import StorageType, SyncStyle
from trinity.common.experience import EID, Experience
from trinity.common.models.allocator import Allocator
from trinity.common.models.model import InferenceModel, ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.explorer.scheduler import Scheduler


@WORKFLOWS.register_module("dummy_workflow")
class DummyWorkflow(Workflow):
    can_repeat: bool = True

    def __init__(self, *, task, model, auxiliary_models):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.step_num = task.workflow_args.get("step_num", 1)
        self.error_type = task.raw_task.get("error_type", "")
        self.seconds = None
        if "timeout" in self.error_type:
            parts = self.error_type.split("_")
            if len(parts) > 1:
                self.seconds = int(parts[-1])
            else:
                self.seconds = 10

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    def run(self) -> List[Experience]:
        if "timeout" in self.error_type:
            time.sleep(self.seconds)
        elif self.error_type == "exception":
            raise ValueError("Exception occurred")
        elif self.error_type == "exit":
            exit(1)
        elif self.error_type == "auxiliary_models":
            assert self.auxiliary_models is not None and len(self.auxiliary_models) == 2

        exps = []
        for i in range(self.repeat_times):
            run_level_metrics = {"run_metrics": float(i + self.run_id_base)}
            run_level_exps = []
            for step in range(self.step_num):
                run_level_exps.append(
                    Experience(
                        tokens=torch.zeros(5),
                        prompt_length=2,
                        prompt_text=self.error_type or "success",
                        eid=EID(run=i + self.run_id_base, step=step),
                        info={"repeat_times": self.repeat_times},
                    )
                )
            run_level_exps[-1].metrics = run_level_metrics
            exps.extend(run_level_exps)
        return exps


@WORKFLOWS.register_module("dummy_nonrepeat_workflow")
class DummyNonRepeatWorkflow(Workflow):
    can_reset: bool = True

    def __init__(self, *, task, model, auxiliary_models):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.reset_flag = False
        self.step_num = task.workflow_args.get("step_num", 1)
        self.metrics = task.workflow_args.get("metrics", [0])

    def reset(self, task: Task):
        self.task = task
        self.reset_flag = True
        self.step_num = task.workflow_args.get("step_num", 1)
        self.metrics = task.workflow_args.get("metrics", [0])

    def run(self) -> List[Experience]:
        exps = [
            Experience(
                eid=EID(run=self.run_id_base, step=step),
                tokens=torch.zeros(5),
                prompt_length=2,
                prompt_text="success",
                info={"reset_flag": self.reset_flag},
                metrics={
                    "run_metrics": self.metrics[step % len(self.metrics)],
                },
            )
            for step in range(self.step_num)
        ]
        return exps


@WORKFLOWS.register_module("dummy_partial_snapshot_workflow")
class DummyPartialSnapshotWorkflow(Workflow):
    can_reset: bool = True

    def __init__(self, *, task, model, auxiliary_models):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.reset(task)

    def reset(self, task: Task):
        self.task = task

    def run(self) -> List[Experience]:
        actions = self.task.workflow_args.get("actions_by_repeat_times", {})
        action = actions.get(self.task.rollout_args.n, "success")

        if action.startswith("sleep_"):
            time.sleep(float(action.split("_", 1)[1]))
        elif action == "exception":
            raise ValueError("planned partial failure")

        return [
            Experience(
                eid=EID(step=0),
                tokens=torch.zeros(5),
                prompt_length=2,
                prompt_text=action,
                metrics={"run_metrics": float(self.task.rollout_args.n * 10)},
            )
        ]


@WORKFLOWS.register_module("dummy_async_partial_snapshot_workflow")
class DummyAsyncPartialSnapshotWorkflow(Workflow):
    can_reset: bool = True
    is_async: bool = True
    _run_index_by_key = defaultdict(int)

    def __init__(self, *, task, model, auxiliary_models):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.reset(task)

    def reset(self, task: Task):
        self.task = task

    async def run_async(self) -> List[Experience]:
        actions = self.task.workflow_args.get("actions_by_repeat_times", {})
        action_sequences = self.task.workflow_args.get("action_sequence_by_repeat_times", {})
        metric_sequences = self.task.workflow_args.get("metric_sequence_by_repeat_times", {})

        key = (self.task.batch_id, self.task.task_id, self.task.rollout_args.n)
        run_index = DummyAsyncPartialSnapshotWorkflow._run_index_by_key[key]
        DummyAsyncPartialSnapshotWorkflow._run_index_by_key[key] += 1

        sequence_actions = action_sequences.get(self.task.rollout_args.n)
        if sequence_actions and run_index < len(sequence_actions):
            action = sequence_actions[run_index]
        else:
            action = actions.get(self.task.rollout_args.n, "success")

        if action.startswith("sleep_"):
            await asyncio.sleep(float(action.split("_", 1)[1]))
        elif action == "exception":
            raise ValueError("planned partial failure")

        sequence = metric_sequences.get(self.task.rollout_args.n)
        if sequence and run_index < len(sequence):
            run_metric = float(sequence[run_index])
        else:
            run_metric = float(self.task.rollout_args.n * 10)

        return [
            Experience(
                eid=EID(step=0),
                tokens=torch.zeros(5),
                prompt_length=2,
                prompt_text=action,
                metrics={"run_metrics": run_metric},
            )
        ]

    def run(self):
        raise RuntimeError("This method should not be called")


@WORKFLOWS.register_module("dummy_async_workflow")
class DummyAsyncWorkflow(Workflow):
    can_repeat: bool = True
    is_async: bool = True

    def __init__(self, *, task, model, auxiliary_models):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.step_num = task.workflow_args.get("step_num", 1)

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    async def run_async(self):
        exps = []
        for i in range(self.repeat_times):
            run_level_metrics = {"run_metrics": float(i + self.run_id_base)}
            run_level_exps = []
            for step in range(self.step_num):
                run_level_exps.append(
                    Experience(
                        eid=EID(run=i + self.run_id_base, step=step),
                        tokens=torch.zeros(5),
                        prompt_length=2,
                        prompt_text="success",
                    )
                )
            run_level_exps[-1].metrics = run_level_metrics
            exps.extend(run_level_exps)
        return exps

    def run(self):
        raise RuntimeError("This method should not be called")


@WORKFLOWS.register_module("dummy_workflow_with_state")
class DummyWorkflowWithState(Workflow):
    can_repeat: bool = True
    is_async: bool = True

    def __init__(self, *, task, model: ModelWrapper, auxiliary_models):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.step_num = task.workflow_args.get("step_num", 1)

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    async def run_async(self) -> List[Experience]:
        exps = []
        for i in range(self.repeat_times):
            run_level_metrics = {"run_metrics": float(i + self.run_id_base)}
            run_level_exps = []
            for step in range(self.step_num):
                run_level_exps.append(
                    Experience(
                        eid=EID(run=i + self.run_id_base, step=step),
                        tokens=torch.zeros(5),
                        prompt_length=2,
                        prompt_text="success",
                    )
                )
            run_level_exps[-1].metrics = run_level_metrics
            self.logger.info(f"Setting workflow state to repeat_cnt={i}")
            await self.model.set_workflow_state({"repeat_cnt": i})
            await asyncio.sleep(1)
            exps.extend(run_level_exps)
        return exps


@WORKFLOWS.register_module("dummy_concurrent_workflow")
class DummyConcurrentWorkflow(Workflow):
    can_repeat: bool = False
    is_async: bool = True

    def __init__(self, *, task, model, auxiliary_models):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

    async def run_async(self) -> List[Experience]:
        await asyncio.sleep(1)

        return [
            Experience(
                eid=EID(run=self.run_id_base, step=0),
                tokens=torch.zeros(5),
                prompt_length=2,
                prompt_text="success",
            )
        ]


@ray.remote
class DummyModel(InferenceModel):
    def __init__(self):
        from trinity.common.config import InferenceModelConfig

        super().__init__(InferenceModelConfig(model_path="dummy_model"))

    def sync_model_weights(self, model_version, sync_method, timeout):
        return True

    async def prepare(self):
        return

    def get_model_version(self):
        return 0

    async def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        state_dict_meta: Optional[List] = None,
    ) -> None:
        pass

    def get_api_server_url(self) -> Optional[str]:
        return None

    async def chat(self, messages: List[Dict], lora_request=None, **kwargs) -> Sequence[Experience]:
        prompt_length = sum(len(msg["content"]) for msg in messages)
        return [
            Experience(
                tokens=torch.zeros(prompt_length + 10),
                prompt_length=prompt_length,
                logprobs=torch.zeros(10),
            )
        ]

    async def generate(self, prompt: str, lora_request=None, **kwargs) -> Sequence[Experience]:
        prompt_length = len(prompt)
        return [
            Experience(
                tokens=torch.zeros(prompt_length + 5),
                prompt_length=prompt_length,
                logprobs=torch.zeros(5),
            )
        ]


@ray.remote
class DummyAuxiliaryModel(InferenceModel):
    def sync_model_weights(self, model_version, sync_method, timeout):
        return True

    def get_model_version(self):
        return 0

    async def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        state_dict_meta: Optional[List] = None,
    ) -> None:
        pass

    def get_api_server_url(self) -> str:
        return "http://localhost:12345"


def _create_named_model_actors(config) -> List:
    allocator = Allocator(config.explorer)
    actor_handles = []

    def create_role_models(model_config, role, actor_cls) -> None:
        for engine_id in range(model_config.engine_num):
            actor_name = allocator.get_actor_name(role, engine_id, 0)
            actor_handles.append(
                actor_cls.options(
                    name=actor_name,
                    namespace=model_config.ray_namespace,
                    num_cpus=0,
                ).remote()
            )

    create_role_models(config.explorer.rollout_model, "rollout", DummyModel)
    for index, auxiliary_config in enumerate(config.explorer.auxiliary_models):
        create_role_models(auxiliary_config, f"auxiliary_{index}", DummyAuxiliaryModel)

    for actor_handle in actor_handles:
        ray.get(actor_handle.prepare.remote())
        ray.get(actor_handle.get_api_server_url.remote())

    return actor_handles


def _cleanup_named_model_actors(actor_handles: Optional[List]) -> None:
    if not actor_handles:
        return
    for actor_handle in reversed(actor_handles):
        try:
            ray.kill(actor_handle)
        except Exception:
            pass


def _align_model_namespaces(config) -> None:
    config.explorer.rollout_model.ray_namespace = config.ray_namespace
    for auxiliary_config in config.explorer.auxiliary_models:
        auxiliary_config.ray_namespace = config.ray_namespace


def _assign_test_namespace(config) -> None:
    config.ray_namespace = f"scheduler-test-{uuid.uuid4().hex}"
    _align_model_namespaces(config)


def _configure_dummy_models(config) -> None:
    for auxiliary_config in config.explorer.auxiliary_models:
        auxiliary_config.enable_openai_api = True


def generate_tasks(
    total_num: int,
    timeout_num: int = 0,
    exception_num: int = 0,
    timeout_seconds: int = 10,
    repeat_times: int = 1,
    step_num: int = 1,
    repeatable: bool = True,
):
    """Generate some tasks for testing

    Args:
        total_num: number of normal tasks
        timeout_num: number of timeout tasks
        exception_num: number of exception tasks
        timeout_seconds: the timeout for timeout tasks
        repeat_times: number of times to repeat each task
        step_num: number of steps in each task
        repeatable: whether to use repeatableworkflow
    """
    workflow = DummyWorkflow if repeatable else DummyNonRepeatWorkflow
    tasks = [
        Task(
            workflow=workflow,  # type: ignore[type-abstract]
            workflow_args={"step_num": step_num},
            repeat_times=repeat_times,
            raw_task={},
        )
        for _ in range(total_num)
    ]

    tasks.extend(
        [
            Task(
                workflow=workflow,  # type: ignore[type-abstract]
                workflow_args={"step_num": step_num},
                repeat_times=repeat_times,
                raw_task={"error_type": f"timeout_{timeout_seconds}"},
            )
            for _ in range(timeout_num)
        ]
    )

    tasks.extend(
        [
            Task(
                workflow=workflow,  # type: ignore[type-abstract]
                workflow_args={"step_num": step_num},
                repeat_times=repeat_times,
                raw_task={"error_type": "exception"},
            )
            for _ in range(exception_num)
        ]
    )

    return tasks


async def collect_results(scheduler: Scheduler, **kwargs):
    """Collect serialized payloads and materialize experiences for assertions."""

    statuses, payloads = await scheduler.get_payload_results(**kwargs)
    experiences = []
    for payload in payloads:
        experiences.extend(Experience.deserialize_many(payload))
    return statuses, experiences


class SchedulerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        DummyAsyncPartialSnapshotWorkflow._run_index_by_key.clear()
        self.config = get_template_config()
        _assign_test_namespace(self.config)
        _configure_dummy_models(self.config)
        self.config.explorer.max_retry_times = 1
        self.config.explorer.max_timeout = 5
        self.config.explorer.runner_per_model = 2
        self.config.buffer.train_batch_size = 2
        self.config.buffer.pad_token_id = 0
        self.config.buffer.explorer_output = (
            self.config.buffer.trainer_input.experience_buffer
        ) = ExperienceBufferConfig(
            name="test",
            storage_type=StorageType.QUEUE.value,
            schema_type="experience",
            path="",
        )
        self.config.buffer.trainer_input.experience_buffer.max_read_timeout = 1
        self.config.algorithm.repeat_times = 1
        self.config.check_and_update()
        self.model_actors = _create_named_model_actors(self.config)

    async def test_get_payload_results(self):
        scheduler = Scheduler(self.config)
        await scheduler.start()

        tasks = generate_tasks(8)
        scheduler.schedule(tasks, batch_id=0)

        statuses, exps = await collect_results(scheduler, batch_id=0, min_num=8, timeout=20)
        self.assertEqual(len(statuses), 8)
        self.assertEqual(len(exps), 8)
        _, exps = await collect_results(scheduler, batch_id=0, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        for result in statuses:
            self.assertTrue(result.ok)

        for batch_id in range(1, 4):
            tasks = generate_tasks(4)
            scheduler.schedule(tasks, batch_id=batch_id)

        for batch_id in range(1, 4):
            self.assertTrue(scheduler.has_step(batch_id))
            statuses, exps = await collect_results(
                scheduler, batch_id=batch_id, min_num=4, timeout=10
            )
            self.assertEqual(len(statuses), 4)
            self.assertEqual(len(exps), 4)
            self.assertFalse(scheduler.has_step(batch_id))
        _, exps = await collect_results(scheduler, batch_id=0, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        tasks = generate_tasks(3)
        scheduler.schedule(tasks, batch_id=4)
        self.assertTrue(scheduler.has_step(4))
        statuses, exps = await collect_results(scheduler, batch_id=4)
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3)
        self.assertFalse(scheduler.has_step(4))

        # test timeout
        tasks = generate_tasks(2, timeout_num=2, timeout_seconds=10)
        scheduler.schedule(tasks, batch_id=0)

        start_time = time.time()
        statuses, exps = await collect_results(scheduler, batch_id=0, min_num=4, timeout=3)
        end_time = time.time()

        self.assertLessEqual(end_time - start_time, 15)  # sync wait for runner restart
        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 2)

        # test run tasks after timeout
        tasks = generate_tasks(4)
        scheduler.schedule(tasks, batch_id=0)

        # actor restart is slow, set a big timeout
        statuses, exps = await collect_results(scheduler, batch_id=0, timeout=20)
        self.assertEqual(len(statuses), 4)

        success_count = sum(1 for r in statuses if r.ok)
        self.assertEqual(success_count, 4)
        self.assertEqual(len(exps), 4)
        _, exps = await collect_results(scheduler, batch_id=0, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        # test exception tasks
        tasks = generate_tasks(1, exception_num=3)
        scheduler.schedule(tasks, batch_id=1)
        statuses, exps = await collect_results(scheduler, batch_id=1, timeout=5)
        self.assertEqual(len(statuses), 4)

        success_count = sum(1 for r in statuses if r.ok)
        self.assertEqual(success_count, 1)
        self.assertEqual(len(exps), 1)
        _, exps = await collect_results(scheduler, batch_id=1, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        # test _cleanup_batch_and_restart_runners: part I, no clear
        tasks = generate_tasks(3, timeout_num=1, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=2)
        statuses, exps = await collect_results(
            scheduler, batch_id=2, timeout=2, clear_timeout_tasks=False
        )
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3)
        statuses, exps = await collect_results(
            scheduler, batch_id=2, timeout=2, clear_timeout_tasks=False
        )
        self.assertEqual(len(statuses), 1)
        self.assertEqual(len(exps), 1)
        #  test _cleanup_batch_and_restart_runners: part II, clear
        tasks = generate_tasks(3, timeout_num=1, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=3)
        statuses, exps = await collect_results(scheduler, batch_id=3, timeout=2)
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3)
        statuses, exps = await collect_results(scheduler, batch_id=3, timeout=2)
        self.assertEqual(len(statuses), 0)
        self.assertEqual(len(exps), 0)
        _, exps = await collect_results(scheduler, batch_id=3, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        await scheduler.stop()

    async def test_concurrent_operations(self):
        scheduler = Scheduler(self.config)
        await scheduler.start()

        async def schedule_tasks(batch_id, num_tasks):
            tasks = generate_tasks(num_tasks)
            scheduler.schedule(tasks, batch_id=batch_id)
            return await collect_results(
                scheduler, batch_id=batch_id, min_num=num_tasks, timeout=10
            )

        results = await asyncio.gather(
            schedule_tasks(0, 3),
            schedule_tasks(1, 4),
            schedule_tasks(2, 2),
        )

        self.assertEqual(len(results[0][0]), 3)
        self.assertEqual(len(results[1][0]), 4)
        self.assertEqual(len(results[2][0]), 2)

        await scheduler.stop()

    async def test_scheduler_restart_after_stop(self):
        scheduler = Scheduler(self.config)

        await scheduler.start()
        tasks = generate_tasks(2)
        scheduler.schedule(tasks, batch_id=0)
        results, exps = await collect_results(scheduler, batch_id=0, min_num=2, timeout=10)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(exps), 2)
        await scheduler.stop()

        await scheduler.start()
        tasks = generate_tasks(3, repeat_times=2)
        scheduler.schedule(tasks, batch_id=1)
        results, exps = await collect_results(scheduler, batch_id=1, min_num=3, timeout=10)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(exps), 3 * 2)
        await scheduler.stop()

    async def test_split_tasks(self):
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()
        exp_list = []

        tasks = generate_tasks(4, repeat_times=8)  # ceil(8 / 2) == 4
        scheduler.schedule(tasks, batch_id=1)
        statuses, exps = await collect_results(scheduler, batch_id=1)
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 4 * 8)
        exp_list.extend(exps)
        _, exps = await collect_results(scheduler, batch_id=1, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        tasks = generate_tasks(4, repeat_times=5)  # ceil(5 / 2) == 3
        scheduler.schedule(tasks, batch_id=2)
        statuses, exps = await collect_results(scheduler, batch_id=2)
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 4 * 5)
        exp_list.extend(exps)
        _, exps = await collect_results(scheduler, batch_id=2, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        tasks = generate_tasks(3, repeat_times=1)  # ceil(1 / 2) == 1
        scheduler.schedule(tasks, batch_id=3)
        statuses, exps = await collect_results(scheduler, batch_id=3)
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3 * 1)
        exp_list.extend(exps)
        _, exps = await collect_results(scheduler, batch_id=3, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        # test task_id, run_id and unique_id
        group_ids = [exp.eid.tid for exp in exp_list]
        self.assertEqual(len(set(group_ids)), 11)  # 4 + 4 + 3
        run_ids = [exp.eid.rid for exp in exp_list]
        self.assertEqual(len(run_ids), len(set(run_ids)))
        unique_ids = [exp.eid.uid for exp in exp_list]
        self.assertEqual(len(unique_ids), len(set(unique_ids)))

        await scheduler.stop()

    async def test_multi_step_execution(self):
        self.config.explorer.max_repeat_times_per_runner = 1
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()
        tasks = generate_tasks(2, repeat_times=4)

        n_steps = 3
        for i in range(1, n_steps + 1):
            scheduler.schedule(tasks, batch_id=i)
            statuses, exps = await collect_results(scheduler, batch_id=i)
            self.assertEqual(len(statuses), 2)
            self.assertEqual(len(exps), 2 * 4)

        await scheduler.stop()

    async def test_non_repeatable_workflow(self):
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()
        task_num, repeat_times = 5, 4
        tasks = generate_tasks(task_num, repeat_times=repeat_times, repeatable=False)

        batch_num = 2
        exp_list = []
        for i in range(1, batch_num + 1):
            scheduler.schedule(tasks, batch_id=i)
            statuses, exps = await collect_results(scheduler, batch_id=i)
            self.assertEqual(len(statuses), task_num)
            self.assertEqual(len(exps), task_num * repeat_times)
            exp_list.extend(exps)

        # test task_id, run_id and unique_id
        group_ids = [exp.eid.tid for exp in exp_list]
        self.assertEqual(len(set(group_ids)), batch_num * task_num)
        run_ids = [exp.eid.rid for exp in exp_list]
        self.assertEqual(len(set(run_ids)), batch_num * task_num * repeat_times)
        unique_ids = [exp.eid.uid for exp in exp_list]
        self.assertEqual(len(unique_ids), len(set(unique_ids)))

        # test reset used properly
        runner_num = (
            self.config.explorer.runner_per_model * self.config.explorer.max_repeat_times_per_runner
        )
        self.assertEqual(
            sum([exp.info["reset_flag"] for exp in exp_list]), len(exp_list) - runner_num
        )

    async def test_async_workflow(self):
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()
        task_num, repeat_times, step_num = 5, 4, 3
        tasks = [
            Task(
                workflow=DummyAsyncWorkflow,  # type: ignore[type-abstract]
                workflow_args={"step_num": step_num},
                repeat_times=repeat_times,
                raw_task={},
            )
            for _ in range(task_num)
        ]

        batch_num = 2
        exp_list = []
        for i in range(1, batch_num + 1):
            scheduler.schedule(tasks, batch_id=i)
            statuses, exps = await collect_results(scheduler, batch_id=i)
            self.assertEqual(len(statuses), task_num)
            self.assertEqual(len(exps), task_num * repeat_times * step_num)
            exp_list.extend(exps)

        # test task_id, run_id and unique_id
        group_ids = [exp.eid.tid for exp in exp_list]
        self.assertEqual(len(set(group_ids)), batch_num * task_num)
        run_ids = [exp.eid.rid for exp in exp_list]
        self.assertEqual(len(set(run_ids)), batch_num * task_num * repeat_times)
        unique_ids = [exp.eid.uid for exp in exp_list]
        self.assertEqual(len(unique_ids), len(set(unique_ids)))

    async def test_stepwise_experience_eid(self):
        task_num, repeat_times, step_num = 2, 4, 3
        self.config.buffer.batch_size = task_num
        self.config.buffer.train_batch_size = task_num * repeat_times * step_num
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()
        batch_num = 2

        # repeatable stepwise workflow
        tasks = generate_tasks(
            task_num, step_num=step_num, repeat_times=repeat_times, repeatable=True
        )
        exp_list = []
        for i in range(1, batch_num + 1):
            scheduler.schedule(tasks, batch_id=i)
            statuses, exps = await collect_results(scheduler, batch_id=i)
            self.assertEqual(len(statuses), task_num)
            self.assertEqual(len(exps), task_num * repeat_times * step_num)
            exp_list.extend(exps)

        # test task_id, run_id and unique_id
        group_ids = [exp.eid.tid for exp in exp_list]
        self.assertEqual(len(set(group_ids)), batch_num * task_num)
        run_ids = [exp.eid.rid for exp in exp_list]
        self.assertEqual(len(set(run_ids)), batch_num * task_num * repeat_times)
        unique_ids = [exp.eid.uid for exp in exp_list]
        self.assertEqual(len(unique_ids), len(set(unique_ids)))

        # Non-repeatable stepwise workflow
        tasks = generate_tasks(
            task_num, step_num=step_num, repeat_times=repeat_times, repeatable=False
        )
        exp_list = []
        for i in range(1, batch_num + 1):
            scheduler.schedule(tasks, batch_id=i)
            statuses, exps = await collect_results(scheduler, batch_id=i)
            self.assertEqual(len(statuses), task_num)
            self.assertEqual(len(exps), task_num * repeat_times * step_num)
            exp_list.extend(exps)

        # test task_id, run_id and unique_id
        group_ids = [exp.eid.tid for exp in exp_list]
        self.assertEqual(len(set(group_ids)), batch_num * task_num)
        run_ids = [exp.eid.rid for exp in exp_list]
        self.assertEqual(len(set(run_ids)), batch_num * task_num * repeat_times)
        unique_ids = [exp.eid.uid for exp in exp_list]
        self.assertEqual(len(unique_ids), len(set(unique_ids)))

    @parameterized.expand(
        [
            (2,),
            (None,),
        ]
    )
    async def test_metric_calculation_with_repeatable_workflow(self, max_repeat_times_per_runner):
        self.config.explorer.max_repeat_times_per_runner = max_repeat_times_per_runner
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()
        tasks = []
        tasks.extend(generate_tasks(total_num=1, step_num=1, repeat_times=4, repeatable=True))
        tasks.extend(generate_tasks(total_num=1, step_num=4, repeat_times=8, repeatable=True))
        scheduler.schedule(tasks, batch_id=0)
        statuses, exps = await collect_results(scheduler, batch_id=0)
        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 1 * 4 * 1 + 1 * 8 * 4)
        expected_run_metrics = set({1.5, 3.5})  # (0+1+2+3)/4 and (0+1+2+3+4+5+6+7)/8
        actual_run_metrics = set(status.metrics[0]["run_metrics"] for status in statuses)
        self.assertSetEqual(expected_run_metrics, actual_run_metrics)

    @parameterized.expand(
        [
            (2,),
            (None,),
        ]
    )
    async def test_metric_calculation_with_non_repeatable_workflow(
        self, max_repeat_times_per_runner
    ):
        self.config.explorer.max_repeat_times_per_runner = max_repeat_times_per_runner
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()
        tasks = []
        tasks.extend(generate_tasks(total_num=1, step_num=3, repeat_times=4, repeatable=False))
        tasks[-1].workflow_args["metrics"] = [1.0, 2.0, 3.0]
        tasks.extend(generate_tasks(total_num=1, step_num=8, repeat_times=5, repeatable=False))
        tasks[-1].workflow_args["metrics"] = [2 * i for i in range(8)]
        scheduler.schedule(tasks, batch_id=0)
        statuses, exps = await collect_results(scheduler, batch_id=0)
        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 1 * 4 * 3 + 1 * 5 * 8)
        # (1+2+3)/3 = 2.0
        # (0+2+4+6+8+10+12+14)/8 = 7.0
        self.assertSetEqual(
            set(status.metrics[0]["run_metrics"] for status in statuses), {2.0, 7.0}
        )

    async def test_over_rollout_min_wait(self):
        self.config.explorer.over_rollout.ratio = 0.5
        self.config.explorer.over_rollout.wait_after_min = 3
        self.config.explorer.max_repeat_times_per_runner = None
        self.config.buffer.batch_size = 4
        self.config.synchronizer.sync_style = SyncStyle.EXPLORER_DRIVEN
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()
        tasks = []
        tasks.extend(generate_tasks(0, timeout_num=2, repeat_times=1, timeout_seconds=1))
        tasks.extend(generate_tasks(0, timeout_num=1, repeat_times=1, timeout_seconds=3))
        tasks.extend(generate_tasks(0, timeout_num=1, repeat_times=1, timeout_seconds=6))
        scheduler.schedule(tasks, batch_id=0)
        statuses, exps = await collect_results(scheduler, batch_id=0, min_num=2)
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3 * 1)

    async def test_over_rollout_return_partial_tasks(self):
        self.config.explorer.over_rollout.ratio = 0.5
        self.config.explorer.over_rollout.wait_after_min = 0.5
        self.config.explorer.over_rollout.return_partial_tasks = True
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.synchronizer.sync_style = SyncStyle.EXPLORER_DRIVEN
        self.config.buffer.batch_size = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()

        tasks = [
            Task(
                workflow=DummyWorkflow,  # type: ignore[type-abstract]
                workflow_args={"step_num": 1},
                repeat_times=1,
                raw_task={},
            ),
            Task(
                # split into 2 sub-tasks with repeat_times=2 and 1.
                # Inside the repeat_times=2 sub-task, one internal run succeeds and
                # the other raises, so runner returns a partial subtask result.
                # The repeat_times=1 sub-task remains in sleep_5 and is later cleaned
                # up by over-rollout before it can contribute any metric.
                workflow=DummyAsyncPartialSnapshotWorkflow,  # type: ignore[type-abstract]
                workflow_args={
                    "action_sequence_by_repeat_times": {
                        2: ["success", "exception"],
                    },
                    "actions_by_repeat_times": {
                        1: "sleep_5",
                    },
                    "metric_sequence_by_repeat_times": {
                        2: [10.0, 30.0],
                    },
                },
                repeat_times=3,
                raw_task={},
            ),
        ]
        scheduler.schedule(tasks, batch_id=0)

        start_time = time.time()
        statuses, exps = await collect_results(
            scheduler,
            batch_id=0,
            min_num=1,
            timeout=2,
            return_partial_tasks=True,
        )
        end_time = time.time()

        self.assertLess(end_time - start_time, 5)
        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 2)

        full_status = next(status for status in statuses if status.total_runs == 1)
        partial_status = next(status for status in statuses if status.total_runs == 3)
        self.assertTrue(full_status.ok)
        self.assertEqual(partial_status.completed_runs, 1)
        self.assertFalse(partial_status.ok)
        metric_bearing_exps = [exp for exp in exps if exp.metrics]
        full_task_run_metrics = [
            exp.metrics["run_metrics"]
            for exp in metric_bearing_exps
            if exp.info and exp.info.get("repeat_times") == 1
        ]
        partial_task_run_metrics = [
            exp.metrics["run_metrics"]
            for exp in metric_bearing_exps
            if not (exp.info and exp.info.get("repeat_times") == 1)
        ]

        # The normal task should keep its own task-level metric, proving scheduler
        # does not mix metrics between different tasks in the same batch.
        self.assertEqual(full_task_run_metrics, [0.0])
        self.assertEqual(full_status.metrics[0]["run_metrics"], 0.0)

        # End-to-end check: the partially returned task contributes only the single
        # successful run from the partially failed repeat_times=2 sub-task. The
        # failed internal run and the cancelled repeat_times=1 sub-task must not be
        # counted when scheduler computes the task-level metric. This also guards
        # against retrying the partially successful sub-task, which would add extra
        # successful runs and change both the emitted experiences and task metric.
        # We identify partial-task experiences by excluding the known full-task
        # marker (`info["repeat_times"] == 1`) rather than assuming empty info.
        self.assertEqual(sorted(partial_task_run_metrics), [10.0])
        self.assertEqual(
            partial_status.metrics[0]["run_metrics"],
            sum(partial_task_run_metrics) / len(partial_task_run_metrics),
        )
        self.assertEqual(partial_status.metrics[0]["run_metrics"], 10.0)
        self.assertIn("1/3 runs completed successfully", partial_status.message)

        statuses, exps = await collect_results(scheduler, batch_id=0, timeout=1)
        self.assertEqual(len(statuses), 0)
        self.assertEqual(len(exps), 0)

        await scheduler.stop()

    async def test_over_rollout_async_cancelled_runner_accepts_next_batch(self):
        self.config.explorer.over_rollout.ratio = 0.5
        self.config.explorer.over_rollout.wait_after_min = 0.5
        self.config.explorer.over_rollout.return_partial_tasks = True
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.explorer.runner_per_model = 1
        self.config.synchronizer.sync_style = SyncStyle.EXPLORER_DRIVEN
        self.config.buffer.batch_size = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()

        tasks = [
            Task(
                workflow=DummyWorkflow,  # type: ignore[type-abstract]
                workflow_args={"step_num": 1},
                repeat_times=1,
                raw_task={},
            ),
            Task(
                workflow=DummyAsyncPartialSnapshotWorkflow,  # type: ignore[type-abstract]
                workflow_args={
                    "actions_by_repeat_times": {
                        2: "success",
                        1: "sleep_5",
                    }
                },
                repeat_times=3,
                raw_task={},
            ),
        ]
        scheduler.schedule(tasks, batch_id=0)

        statuses, exps = await collect_results(
            scheduler,
            batch_id=0,
            min_num=1,
            timeout=3,
            return_partial_tasks=True,
        )

        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 3)

        follow_up_tasks = generate_tasks(2)
        scheduler.schedule(follow_up_tasks, batch_id=1)
        start_time = time.time()
        next_statuses, next_exps = await collect_results(
            scheduler, batch_id=1, min_num=2, timeout=2
        )
        elapsed = time.time() - start_time

        self.assertEqual(len(next_statuses), 2)
        self.assertEqual(len(next_exps), 2)
        self.assertLess(elapsed, 1.5)

        await scheduler.stop()

    async def test_over_rollout_sync_cancel_does_not_imply_immediate_runner_reuse(self):
        self.config.mode = "explore"
        self.config.explorer.over_rollout.ratio = 0.5
        self.config.explorer.over_rollout.wait_after_min = 0.5
        self.config.explorer.over_rollout.return_partial_tasks = True
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.explorer.runner_per_model = 1
        self.config.explorer.rollout_model.engine_num = 1
        self.config.synchronizer.sync_style = SyncStyle.EXPLORER_DRIVEN
        self.config.buffer.batch_size = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()

        tasks = [
            Task(
                workflow=DummyWorkflow,  # type: ignore[type-abstract]
                workflow_args={"step_num": 1},
                repeat_times=1,
                raw_task={},
            ),
            Task(
                workflow=DummyPartialSnapshotWorkflow,  # type: ignore[type-abstract]
                workflow_args={
                    "actions_by_repeat_times": {
                        2: "success",
                        1: "sleep_2",
                    }
                },
                repeat_times=3,
                raw_task={},
            ),
        ]
        scheduler.schedule(tasks, batch_id=0)

        statuses, exps = await collect_results(
            scheduler,
            batch_id=0,
            min_num=1,
            timeout=3,
            return_partial_tasks=True,
        )

        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 3)

        follow_up_tasks = generate_tasks(2)
        scheduler.schedule(follow_up_tasks, batch_id=1)
        start_time = time.time()
        next_statuses, next_exps = await collect_results(
            scheduler,
            batch_id=1,
            min_num=2,
            timeout=0.5,
            clear_timeout_tasks=False,
        )
        elapsed = time.time() - start_time

        self.assertEqual(len(next_statuses), 0)
        self.assertEqual(len(next_exps), 0)
        self.assertGreaterEqual(elapsed, 0.5)
        self.assertTrue(scheduler.has_step(1))

        next_statuses, next_exps = await collect_results(
            scheduler, batch_id=1, min_num=2, timeout=4
        )
        self.assertEqual(len(next_statuses), 2)
        self.assertEqual(len(next_exps), 2)

        await scheduler.stop()

    async def test_timeout_cleanup_still_restarts_runner(self):
        self.config.explorer.over_rollout.wait_after_min = 100
        self.config.explorer.max_repeat_times_per_runner = None
        self.config.synchronizer.sync_style = SyncStyle.EXPLORER_DRIVEN
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()

        tasks = generate_tasks(0, timeout_num=2, repeat_times=1, timeout_seconds=10)
        scheduler.schedule(tasks, batch_id=0)

        with patch.object(scheduler, "_restart_runner", new=AsyncMock()) as restart_runner_mock:
            await collect_results(scheduler, batch_id=0, timeout=1)

        self.assertGreaterEqual(restart_runner_mock.await_count, 1)

        await scheduler.stop()

    async def test_unexpected_task_exception_restarts_runner(self):
        self.config.explorer.runner_per_model = 1
        self.config.check_and_update()
        scheduler = Scheduler(self.config)
        await scheduler.start()

        scheduler.runners[0].run_with_retry = AsyncMock(side_effect=RuntimeError("boom"))
        restart_event = asyncio.Event()

        async def fake_restart_runner(runner_id):
            scheduler.busy_runners.pop(runner_id, None)
            scheduler.idle_runners.add(runner_id)
            restart_event.set()

        with patch.object(
            scheduler, "_restart_runner", side_effect=fake_restart_runner
        ) as restart_runner_mock:
            scheduler.schedule(generate_tasks(1), batch_id=0)
            await asyncio.wait_for(restart_event.wait(), timeout=2)

        self.assertEqual(restart_runner_mock.await_count, 1)
        self.assertEqual(len(scheduler.busy_runners), 0)
        self.assertEqual(len(scheduler.idle_runners), scheduler.runner_num)
        self.assertNotIn(0, scheduler.running_tasks)

        await scheduler.stop()

    async def test_dynamic_timeout(self):
        self.config.explorer.dynamic_timeout.enable = True
        self.config.explorer.dynamic_timeout.warmup_min_steps = 1
        self.config.explorer.dynamic_timeout.ratio = 3.0
        self.config.buffer.batch_size = 4
        self.config.explorer.max_timeout = 20
        self.config.explorer.max_retry_times = 0  # no retry here
        scheduler = Scheduler(self.config)
        await scheduler.start()
        tasks = []
        tasks.extend(generate_tasks(0, timeout_num=4, repeat_times=1, timeout_seconds=1))
        for task in tasks:
            task.is_eval = True
        scheduler.schedule(
            tasks, batch_id="0/eval"
        )  # eval tasks will not count into dynamic timeout
        statuses, exps = await collect_results(scheduler, batch_id="0/eval")
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 0)
        self.assertEqual(scheduler.total_running_time, 0)
        self.assertEqual(scheduler.total_completed_steps, 0)
        self.assertEqual(scheduler.total_completed_tasks, 0)
        tasks = []
        # generate 4 tasks that will run 1 second
        tasks.extend(generate_tasks(0, timeout_num=4, repeat_times=1, timeout_seconds=1))
        scheduler.schedule(tasks, batch_id=0)  # first step will not use dynamic timeout
        statuses, exps = await collect_results(scheduler, batch_id=0)
        self.assertEqual(len(statuses), 4)
        # dynamic timeout will be set to 3.0 * 1.0 = 3.0 seconds for next step
        tasks = []
        tasks.extend(generate_tasks(0, timeout_num=4, repeat_times=1, timeout_seconds=4))
        st = time.time()
        scheduler.schedule(tasks, batch_id=1)
        statuses, exps = await collect_results(scheduler, batch_id=1)
        et = time.time()
        self.assertTrue(
            et - st < 4
        )  # should wait about 1 * 3.0 seconds, here we set 4 seconds timeout
        self.assertEqual(len(exps), 0)
        self.assertEqual(len(statuses), 4)
        # tasks take 2 seconds, which is within the dynamic timeout 3.0 * 1.0 = 3.0 seconds
        tasks = []
        tasks.extend(generate_tasks(0, timeout_num=4, repeat_times=1, timeout_seconds=2))
        scheduler.schedule(tasks, batch_id=2)
        statuses, exps = await collect_results(scheduler, batch_id=2)
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 4)

    async def test_dynamic_timeout_warmup_min_steps_uses_completed_steps(self):
        self.config.explorer.dynamic_timeout.enable = True
        self.config.explorer.dynamic_timeout.warmup_min_steps = 2
        self.config.explorer.dynamic_timeout.ratio = 1.5
        self.config.buffer.batch_size = 3
        self.config.explorer.max_timeout = 20
        self.config.explorer.max_retry_times = 0
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.check_and_update()

        scheduler = Scheduler(self.config)
        await scheduler.start()

        tasks = generate_tasks(0, timeout_num=2, repeat_times=4, timeout_seconds=1)
        scheduler.schedule(tasks, batch_id=0)
        statuses, exps = await collect_results(scheduler, batch_id=0)

        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 8)
        self.assertEqual(scheduler.total_completed_steps, 1)
        self.assertEqual(scheduler.total_completed_tasks, 2)
        self.assertEqual(scheduler.total_completed_sub_tasks, 4)
        self.assertEqual(scheduler.dynamic_timeout(), scheduler.default_timeout)

        tasks = generate_tasks(0, timeout_num=2, repeat_times=4, timeout_seconds=1)
        scheduler.schedule(tasks, batch_id=1)
        statuses, exps = await collect_results(scheduler, batch_id=1)

        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 8)
        self.assertEqual(scheduler.total_completed_steps, 2)
        self.assertAlmostEqual(scheduler.dynamic_timeout(), 1.5, delta=0.8)

        await scheduler.stop()

    async def test_collect_results_reads_payloads_returned_by_workflow_runner(self):
        scheduler = Scheduler(self.config)
        await scheduler.start()

        scheduler.schedule(generate_tasks(3, repeat_times=2), batch_id=0)

        statuses, exps = await collect_results(scheduler, batch_id=0, timeout=10)
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 6)

        await scheduler.stop()

    async def test_timeout_cleanup_keeps_completed_payloads_local(self):
        scheduler = Scheduler(self.config)
        await scheduler.start()

        scheduler.schedule(generate_tasks(1, timeout_num=1, timeout_seconds=10), batch_id=0)

        statuses, exps = await collect_results(scheduler, batch_id=0, min_num=2, timeout=1)
        self.assertEqual(len(statuses), 1)
        self.assertEqual(len(exps), 1)

        await scheduler.stop()

    async def test_eval_tasks_do_not_return_training_experiences(self):
        scheduler = Scheduler(self.config)
        await scheduler.start()

        eval_tasks = generate_tasks(2, repeat_times=2)
        for task in eval_tasks:
            task.is_eval = True

        scheduler.schedule(eval_tasks, batch_id="0/eval")
        statuses, exps = await collect_results(scheduler, batch_id="0/eval", timeout=10)

        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 0)

        await scheduler.stop()

    async def test_get_statuses_skips_payload_deserialization(self):
        scheduler = Scheduler(self.config)
        await scheduler.start()

        scheduler.schedule(generate_tasks(2, repeat_times=2), batch_id=0)

        with patch(
            "trinity.common.experience.Experience.deserialize_many",
            side_effect=AssertionError("payload deserialization should not happen"),
        ):
            statuses = await scheduler.get_statuses(batch_id=0, timeout=10)

        self.assertEqual(len(statuses), 2)

        await scheduler.stop()

    async def test_get_payload_results_keeps_payloads_serialized(self):
        scheduler = Scheduler(self.config)
        await scheduler.start()

        scheduler.schedule(generate_tasks(2, repeat_times=2), batch_id=0)

        with patch(
            "trinity.common.experience.Experience.deserialize_many",
            side_effect=AssertionError("payload deserialization should not happen"),
        ):
            statuses, payloads = await scheduler.get_payload_results(batch_id=0, timeout=10)

        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(payloads), 2)
        self.assertTrue(all(isinstance(payload, bytes) for payload in payloads))

        await scheduler.stop()

    def tearDown(self):
        try:
            _cleanup_named_model_actors(getattr(self, "model_actors", None))
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass


class TestRunnerStateCollection(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        _assign_test_namespace(self.config)
        _configure_dummy_models(self.config)
        self.config.explorer.runner_per_model = 2
        self.config.explorer.runner_state_report_interval = 1
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.check_and_update()
        self.model_actors = _create_named_model_actors(self.config)

    def tearDown(self):
        try:
            _cleanup_named_model_actors(getattr(self, "model_actors", None))
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass

    async def test_runner_state_collection(self):
        scheduler = Scheduler(self.config)
        # 4 runner in side the scheduler
        await scheduler.start()

        tasks = [
            Task(
                workflow=DummyWorkflowWithState,  # type: ignore[type-abstract]
                workflow_args={"step_num": 2},
                repeat_times=4,
                raw_task={},
            )
            for _ in range(4)
        ]
        scheduler.schedule(tasks, batch_id=0)

        async def monitor_routine():
            runner_0_state_history = defaultdict(set)
            await asyncio.sleep(self.config.explorer.runner_state_report_interval + 0.2)
            for _ in range(16):
                await asyncio.sleep(0.3)
                states = scheduler.get_all_state()
                self.assertEqual(len(states), 4)
                for state in states.values():
                    self.assertIn("workflow_id", state)
                    self.assertIn("model_version", state)
                    self.assertIn("begin_time", state)
                    self.assertIn("terminate_time", state)
                    self.assertIn("repeat_cnt", state)
                ids = scheduler.get_key_state("workflow_id")
                self.assertEqual(len(ids), 4)
                self.assertEqual(len(set(ids.values())), 4)
                runner_0_state = scheduler.get_runner_state(0)
                for key, value in runner_0_state.items():
                    runner_0_state_history[key].add(value)
            self.assertEqual(len(runner_0_state_history["repeat_cnt"]), 2)  # max_repeat_times is 2
            self.assertEqual(len(runner_0_state_history["model_version"]), 1)
            self.assertEqual(
                len(runner_0_state_history["workflow_id"]), 2
            )  # split into 2 sub tasks
            self.assertEqual(len(runner_0_state_history["begin_time"]), 2)

        await asyncio.gather(
            monitor_routine(),
            collect_results(scheduler, batch_id=0),
        )
        await scheduler.stop()
