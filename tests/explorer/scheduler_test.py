import asyncio
import time
import unittest
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import ray
import torch
from parameterized import parameterized

from tests.tools import get_template_config
from trinity.common.config import ExperienceBufferConfig
from trinity.common.constants import StorageType, SyncStyle
from trinity.common.experience import EID, Experience
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

    def sync_model(self, model_version, update_weight_args_list):
        return True

    async def prepare(self):
        return

    def get_model_version(self):
        return 0

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
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
    def sync_model(self, model_version, update_weight_args_list):
        return True

    def get_model_version(self):
        return 0

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
    ) -> None:
        pass

    def get_api_server_url(self) -> str:
        return "http://localhost:12345"


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


class SchedulerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
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

    async def test_get_results(self):
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()

        tasks = generate_tasks(8)
        scheduler.schedule(tasks, batch_id=0)

        statuses, exps = await scheduler.get_results(batch_id=0, min_num=8, timeout=20)
        self.assertEqual(len(statuses), 8)
        self.assertEqual(len(exps), 8)
        _, exps = await scheduler.get_results(batch_id=0, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        for result in statuses:
            self.assertTrue(result.ok)

        for batch_id in range(1, 4):
            tasks = generate_tasks(4)
            scheduler.schedule(tasks, batch_id=batch_id)

        for batch_id in range(1, 4):
            self.assertTrue(scheduler.has_step(batch_id))
            statuses, exps = await scheduler.get_results(batch_id=batch_id, min_num=4, timeout=10)
            self.assertEqual(len(statuses), 4)
            self.assertEqual(len(exps), 4)
            self.assertFalse(scheduler.has_step(batch_id))
        _, exps = await scheduler.get_results(batch_id=0, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        tasks = generate_tasks(3)
        scheduler.schedule(tasks, batch_id=4)
        self.assertTrue(scheduler.has_step(4))
        statuses, exps = await scheduler.get_results(batch_id=4)
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3)
        self.assertFalse(scheduler.has_step(4))

        # test timeout
        tasks = generate_tasks(2, timeout_num=2, timeout_seconds=10)
        scheduler.schedule(tasks, batch_id=0)

        start_time = time.time()
        statuses, exps = await scheduler.get_results(batch_id=0, min_num=4, timeout=3)
        end_time = time.time()

        self.assertLessEqual(end_time - start_time, 15)  # sync wait for runner restart
        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 2)

        # test run tasks after timeout
        tasks = generate_tasks(4)
        scheduler.schedule(tasks, batch_id=0)

        # actor restart is slow, set a big timeout
        statuses, exps = await scheduler.get_results(batch_id=0, timeout=20)
        self.assertEqual(len(statuses), 4)

        success_count = sum(1 for r in statuses if r.ok)
        self.assertEqual(success_count, 4)
        self.assertEqual(len(exps), 4)
        _, exps = await scheduler.get_results(batch_id=0, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        # test exception tasks
        tasks = generate_tasks(1, exception_num=3)
        scheduler.schedule(tasks, batch_id=1)
        statuses, exps = await scheduler.get_results(batch_id=1, timeout=5)
        self.assertEqual(len(statuses), 4)

        success_count = sum(1 for r in statuses if r.ok)
        self.assertEqual(success_count, 1)
        self.assertEqual(len(exps), 1)
        _, exps = await scheduler.get_results(batch_id=1, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        # test _cleanup_batch_and_restart_runners: part I, no clear
        tasks = generate_tasks(3, timeout_num=1, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=2)
        statuses, exps = await scheduler.get_results(
            batch_id=2, timeout=2, clear_timeout_tasks=False
        )
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3)
        statuses, exps = await scheduler.get_results(
            batch_id=2, timeout=2, clear_timeout_tasks=False
        )
        self.assertEqual(len(statuses), 1)
        self.assertEqual(len(exps), 1)
        #  test _cleanup_batch_and_restart_runners: part II, clear
        tasks = generate_tasks(3, timeout_num=1, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=3)
        statuses, exps = await scheduler.get_results(
            batch_id=3, timeout=2
        )
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3)
        statuses, exps = await scheduler.get_results(
            batch_id=3, timeout=2
        )
        self.assertEqual(len(statuses), 0)
        self.assertEqual(len(exps), 0)
        _, exps = await scheduler.get_results(batch_id=3, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        await scheduler.stop()

    async def test_wait_all(self):
        """Test wait all"""
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()

        tasks1 = generate_tasks(4)
        tasks2 = generate_tasks(3)
        scheduler.schedule(tasks1, batch_id=0)
        scheduler.schedule(tasks2, batch_id=1)

        start_time = time.time()
        await scheduler.wait_all(timeout=10.0)
        end_time = time.time()

        self.assertLess(end_time - start_time, 5.0)

        self.assertEqual(len(scheduler.pending_tasks), 0)
        self.assertEqual(len(scheduler.running_tasks), 0)

        status0, exps0 = await scheduler.get_results(batch_id=0, min_num=4, timeout=1)
        status1, exps1 = await scheduler.get_results(batch_id=1, min_num=3, timeout=1)
        self.assertEqual(len(status0), 4)
        self.assertEqual(len(status1), 3)

        # test timeout
        tasks = generate_tasks(2, timeout_num=2, timeout_seconds=10)
        scheduler.schedule(tasks, batch_id=0)

        start_time = time.time()
        with self.assertRaises(TimeoutError):
            await scheduler.wait_all(timeout=3.0)
        end_time = time.time()

        self.assertGreaterEqual(end_time - start_time, 2.8)
        self.assertLessEqual(end_time - start_time, 4.0)

        # test empty scenario

        start_time = time.time()
        await scheduler.wait_all(timeout=5.0)
        end_time = time.time()

        self.assertLess(end_time - start_time, 1.0)
        await scheduler.stop()

    async def test_wait_all_timeout_with_multi_batch(self):
        self.config.explorer.max_timeout = 5
        self.config.explorer.rollout_model.engine_num = 4
        self.config.explorer.runner_per_model = 1

        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()

        tasks = generate_tasks(1, timeout_num=3, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=0)
        tasks = generate_tasks(2, timeout_num=2, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=1)
        tasks = generate_tasks(3, timeout_num=1, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=2)
        start_time = time.time()
        await scheduler.wait_all()
        end_time = time.time()
        self.assertTrue(
            end_time - start_time > 9,
            f"wait time should be greater than 9, but got {end_time - start_time}",
        )

        await scheduler.stop()

    async def test_concurrent_operations(self):
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()

        async def schedule_tasks(batch_id, num_tasks):
            tasks = generate_tasks(num_tasks)
            scheduler.schedule(tasks, batch_id=batch_id)
            return await scheduler.get_results(batch_id=batch_id, min_num=num_tasks, timeout=10)

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
        scheduler = Scheduler(self.config, [DummyModel.remote()])

        await scheduler.start()
        tasks = generate_tasks(2)
        scheduler.schedule(tasks, batch_id=0)
        results, exps = await scheduler.get_results(batch_id=0, min_num=2, timeout=10)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(exps), 2)
        await scheduler.stop()

        await scheduler.start()
        tasks = generate_tasks(3, repeat_times=2)
        scheduler.schedule(tasks, batch_id=1)
        results, exps = await scheduler.get_results(batch_id=1, min_num=3, timeout=10)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(exps), 3 * 2)
        await scheduler.stop()

    async def test_scheduler_all_methods(self):
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = generate_tasks(8)
        scheduler.schedule(tasks, batch_id=0)
        self.assertTrue(scheduler.has_step(0))
        statuses, exps = await scheduler.get_results(batch_id=0, min_num=8, timeout=20)
        self.assertEqual(len(statuses), 8)
        self.assertEqual(len(exps), 8)
        scheduler.schedule(tasks, batch_id=1)
        scheduler.schedule(tasks[:4], batch_id=2)
        self.assertFalse(scheduler.has_step(0))
        statuses, exps = await scheduler.get_results(batch_id=0, min_num=8)
        self.assertFalse(scheduler.has_step(0))
        self.assertEqual(len(statuses), 0)  # batch_id 0 has no more tasks
        self.assertEqual(len(exps), 0)
        self.assertFalse(scheduler.has_step(0))
        self.assertTrue(scheduler.has_step(1))
        self.assertTrue(scheduler.has_step(2))
        await scheduler.wait_all()
        st = time.time()
        statuses, exps = await scheduler.get_results(batch_id=1)
        et = time.time()
        self.assertTrue(et - st < 1.0)
        self.assertEqual(len(statuses), 8)
        self.assertEqual(len(exps), 8)
        self.assertFalse(scheduler.has_step(1))
        self.assertTrue(scheduler.has_step(2))
        st = time.time()
        statuses, exps = await scheduler.get_results(batch_id=2)
        et = time.time()
        self.assertTrue(et - st < 1.0)
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 4)
        self.assertFalse(scheduler.has_step(2))
        await scheduler.stop()

    async def test_split_tasks(self):
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        exp_list = []

        tasks = generate_tasks(4, repeat_times=8)  # ceil(8 / 2) == 4
        scheduler.schedule(tasks, batch_id=1)
        statuses, exps = await scheduler.get_results(batch_id=1)
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 4 * 8)
        exp_list.extend(exps)
        _, exps = await scheduler.get_results(batch_id=1, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        tasks = generate_tasks(4, repeat_times=5)  # ceil(5 / 2) == 3
        scheduler.schedule(tasks, batch_id=2)
        statuses, exps = await scheduler.get_results(batch_id=2)
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 4 * 5)
        exp_list.extend(exps)
        _, exps = await scheduler.get_results(batch_id=2, min_num=1, timeout=1)
        self.assertEqual(len(exps), 0)

        tasks = generate_tasks(3, repeat_times=1)  # ceil(1 / 2) == 1
        scheduler.schedule(tasks, batch_id=3)
        statuses, exps = await scheduler.get_results(batch_id=3)
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3 * 1)
        exp_list.extend(exps)
        _, exps = await scheduler.get_results(batch_id=3, min_num=1, timeout=1)
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
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = generate_tasks(2, repeat_times=4)

        n_steps = 3
        for i in range(1, n_steps + 1):
            scheduler.schedule(tasks, batch_id=i)
            statuses, exps = await scheduler.get_results(batch_id=i)
            self.assertEqual(len(statuses), 2)
            self.assertEqual(len(exps), 2 * 4)

        await scheduler.stop()

    async def test_non_repeatable_workflow(self):
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        task_num, repeat_times = 5, 4
        tasks = generate_tasks(task_num, repeat_times=repeat_times, repeatable=False)

        batch_num = 2
        exp_list = []
        for i in range(1, batch_num + 1):
            scheduler.schedule(tasks, batch_id=i)
            statuses, exps = await scheduler.get_results(batch_id=i)
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
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
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
            statuses, exps = await scheduler.get_results(batch_id=i)
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
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        batch_num = 2

        # repeatable stepwise workflow
        tasks = generate_tasks(
            task_num, step_num=step_num, repeat_times=repeat_times, repeatable=True
        )
        exp_list = []
        for i in range(1, batch_num + 1):
            scheduler.schedule(tasks, batch_id=i)
            statuses, exps = await scheduler.get_results(batch_id=i)
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
            statuses, exps = await scheduler.get_results(batch_id=i)
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
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = []
        tasks.extend(generate_tasks(total_num=1, step_num=1, repeat_times=4, repeatable=True))
        tasks.extend(generate_tasks(total_num=1, step_num=4, repeat_times=8, repeatable=True))
        scheduler.schedule(tasks, batch_id=0)
        statuses, exps = await scheduler.get_results(batch_id=0)
        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 1 * 4 * 1 + 1 * 8 * 4)
        self.assertAlmostEqual(statuses[0].metrics[0]["run_metrics"], 1.5)  # (0+1+2+3)/4
        self.assertAlmostEqual(statuses[1].metrics[0]["run_metrics"], 3.5)  # (0+1+2+3+4+5+6+7)/8

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
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = []
        tasks.extend(generate_tasks(total_num=1, step_num=3, repeat_times=4, repeatable=False))
        tasks[-1].workflow_args["metrics"] = [1.0, 2.0, 3.0]
        tasks.extend(generate_tasks(total_num=1, step_num=8, repeat_times=5, repeatable=False))
        tasks[-1].workflow_args["metrics"] = [2 * i for i in range(8)]
        scheduler.schedule(tasks, batch_id=0)
        statuses, exps = await scheduler.get_results(batch_id=0)
        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(exps), 1 * 4 * 3 + 1 * 5 * 8)
        self.assertAlmostEqual(statuses[0].metrics[0]["run_metrics"], 2.0)  # (1+2+3)/3
        self.assertAlmostEqual(statuses[1].metrics[0]["run_metrics"], 7.0)  # (0+2+4+6+8+10+12+14)/8

    async def test_over_rollout_min_wait(self):
        self.config.explorer.over_rollout.ratio = 0.5
        self.config.explorer.over_rollout.wait_after_min = 3
        self.config.explorer.max_repeat_times_per_runner = None
        self.config.buffer.batch_size = 4
        self.config.synchronizer.sync_style = SyncStyle.EXPLORER_DRIVEN
        self.config.check_and_update()
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = []
        tasks.extend(generate_tasks(0, timeout_num=2, repeat_times=1, timeout_seconds=1))
        tasks.extend(generate_tasks(0, timeout_num=1, repeat_times=1, timeout_seconds=3))
        tasks.extend(generate_tasks(0, timeout_num=1, repeat_times=1, timeout_seconds=6))
        scheduler.schedule(tasks, batch_id=0)
        statuses, exps = await scheduler.get_results(batch_id=0, min_num=2)
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 3 * 1)

    async def test_dynamic_timeout(self):
        self.config.explorer.dynamic_timeout.enable = True
        self.config.explorer.dynamic_timeout.ratio = 3.0
        self.config.buffer.batch_size = 4
        self.config.explorer.max_timeout = 20
        self.config.explorer.max_retry_times = 0  # no retry here
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = []
        tasks.extend(generate_tasks(0, timeout_num=4, repeat_times=1, timeout_seconds=1))
        for task in tasks:
            task.is_eval = True
        scheduler.schedule(
            tasks, batch_id="0/eval"
        )  # eval tasks will not count into dynamic timeout
        statuses, exps = await scheduler.get_results(batch_id="0/eval")
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 0)
        self.assertEqual(scheduler.total_running_time, 0)
        self.assertEqual(scheduler.total_completed_tasks, 0)
        tasks = []
        # generate 4 tasks that will run 1 second
        tasks.extend(generate_tasks(0, timeout_num=4, repeat_times=1, timeout_seconds=1))
        scheduler.schedule(tasks, batch_id=0)  # first step will not use dynamic timeout
        statuses, exps = await scheduler.get_results(batch_id=0)
        self.assertEqual(len(statuses), 4)
        # dynamic timeout will be set to 3.0 * 1.0 = 3.0 seconds for next step
        tasks = []
        tasks.extend(generate_tasks(0, timeout_num=4, repeat_times=1, timeout_seconds=4))
        st = time.time()
        scheduler.schedule(tasks, batch_id=1)
        statuses, exps = await scheduler.get_results(batch_id=1)
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
        statuses, exps = await scheduler.get_results(batch_id=2)
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 4)

    def tearDown(self):
        try:
            ray.shutdown()
        except Exception:
            pass


class TestRunnerStateCollection(unittest.IsolatedAsyncioTestCase):
    async def test_runner_state_collection(self):
        ray.init(ignore_reinit_error=True)
        config = get_template_config()
        config.explorer.runner_per_model = 2
        config.explorer.runner_state_report_interval = 0.5
        config.explorer.max_repeat_times_per_runner = 2
        config.check_and_update()
        scheduler = Scheduler(config, [DummyModel.remote(), DummyModel.remote()])
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
            await asyncio.sleep(0.5)  # wait for first report
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
            scheduler.get_results(batch_id=0),
        )
