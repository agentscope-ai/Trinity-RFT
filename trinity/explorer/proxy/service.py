import asyncio
import time
from collections import deque
from typing import Dict, List, Tuple

import torch

from trinity.common.constants import RunningStatus
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.explorer.explorer import Explorer
from trinity.explorer.proxy.recorder import HistoryRecorder
from trinity.utils.log import get_logger


class ExplorerService:
    """Manages the lifecycle and operations of the Explorer API service."""

    def __init__(self, explorer: Explorer, listen_address: str = "localhost", port: int = 8010):
        self.logger = get_logger(__name__)
        self.explorer = explorer
        self.app = None
        self.port = port
        self.listen_address = listen_address
        self.running = False
        self.models: List[ModelWrapper] = [ModelWrapper(model) for model in explorer.models]
        self.min_running_model_num = explorer.config.explorer.min_running_model_num
        self.check_interval = explorer.config.explorer.service_status_check_interval
        self.max_timeout = explorer.config.explorer.max_timeout
        self.running_model_ids: deque[int] = deque()  # indices of running models
        self.model_version_map: Dict[int, int] = {}  # model index -> model version
        self.sync_task_map: Dict[asyncio.Future, int] = {}  # sync task -> model index
        self.latest_model_version = 0
        self.session_level_experience_queue: Dict[int, deque[Experience]] = {}
        self.commit_lock = asyncio.Lock()
        self.ready_experiences = deque()
        self.recorder = HistoryRecorder(
            db_url=explorer.config.explorer.db_url
            or f"sqlite:///{explorer.config.buffer.cache_dir}/proxy_history.db",
            table_name="proxy_history",
        )
        self.total_experience_count = 0
        self.ready_experience_count = 0

    async def serve(self) -> None:
        from trinity.explorer.proxy.app import run_app

        if self.running:
            self.logger.warning("Server is already running.")
            return

        self.running = True
        await asyncio.gather(*[model.prepare() for model in self.models])

        for i, _ in enumerate(self.models):
            self.running_model_ids.append(i)

        self.serve_task = asyncio.create_task(
            run_app(service=self, listen_address=self.listen_address, port=self.port)
        )
        self.sync_model_weights_task = asyncio.create_task(self.model_weights_sync_loop())

    async def model_weights_sync_loop(self) -> None:
        self.logger.info("Starting model weights synchronization loop.")
        while self.running:
            for idx in list(self.running_model_ids):
                self.model_version_map[idx] = await self.models[idx].model_version_async
                if (
                    len(self.running_model_ids)
                    > self.explorer.config.explorer.min_running_model_num
                    and self.model_version_map[idx] < self.latest_model_version
                ):
                    self.logger.info(f"Model {idx} scheduled for synchronization.")
                    self.models[idx].status = RunningStatus.REQUIRE_SYNC
                    self.running_model_ids.remove(idx)
                    asyncio.create_task(self._sync_model_weights(idx))
            # wait half interval
            await asyncio.sleep(self.check_interval / 2)
        self.logger.info("Model weights synchronization loop stopped.")

    def set_latest_model_version(self, version: int) -> None:
        if version > self.latest_model_version:
            self.latest_model_version = version
            self.logger.info(f"Updated latest model version to {version}.")

    async def _sync_model_weights(self, index: int) -> None:
        """Synchronize model weights for the given model index."""
        # wait until the model is free
        start_time = time.time()
        timeout_flag = True
        current_load = -1
        while time.time() - start_time < self.max_timeout:
            current_load = await self.models[index].get_current_load()
            if current_load == 0:
                self.logger.info(f"Model {index} begins synchronization.")
                timeout_flag = False
                break
            else:
                self.logger.info(
                    "Waiting for model %d to be free. Current load: %d", index, current_load
                )
                await asyncio.sleep(1)
        if timeout_flag:
            raise asyncio.TimeoutError(
                f"Timeout waiting for model {index} to be free for synchronization. Current load: {current_load}"
            )
        latest_version = self.latest_model_version  # capture the latest version
        # perform synchronization
        await self.models[index].sync_model_weights(latest_version)
        self.logger.info(f"Model {index} synchronized to version {latest_version}.")
        self.model_version_map[index] = await self.models[index].model_version_async
        self.models[index].status = RunningStatus.RUNNING
        self.running_model_ids.append(index)

    async def allocate_model(self, increase_count: bool = True) -> Tuple[str, int]:
        """Allocate a model for handling a request.

        Returns:
            A tuple of (model_api_address, model_version).
        """
        model_id = self.running_model_ids[0]
        model = self.models[model_id]
        if increase_count:
            model.request_count += 1
        self.running_model_ids.rotate(-1)
        if model.api_address is None:
            raise ValueError(
                "Model does not have a valid API address, please set `enable_openai_api` to `True`."
            )
        return model.api_address, self.model_version_map[model_id]

    def collect_metrics(self) -> Dict:
        metrics = {}
        for i, model in enumerate(self.models):
            metrics[f"rollout/model_{i}/total_request_count"] = model.request_count
            metrics[f"rollout/model_{i}/model_version"] = model.model_version
        metrics["rollout/total_experience_count"] = self.total_experience_count
        metrics["rollout/ready_experience_count"] = self.ready_experience_count
        return metrics

    async def record_experience(self, response, model_version: int) -> None:
        experiences = []
        for choice in response["choices"]:
            exp = Experience(
                tokens=torch.cat(
                    (
                        torch.tensor(response["prompt_token_ids"], dtype=torch.int32),
                        torch.tensor(choice["token_ids"], dtype=torch.int32),
                    )
                ),
                logprobs=(
                    torch.tensor(
                        [logprob["logprob"] for logprob in choice["logprobs"]["content"]],
                        dtype=torch.float32,
                    )
                    if "logprobs" in choice and choice["logprobs"] is not None
                    else torch.tensor([], dtype=torch.float32)
                ),
                prompt_length=len(response["prompt_token_ids"]),
            )
            exp.eid.suffix = response["id"]
            exp.info["model_version"] = model_version
            experiences.append(exp)

        self.total_experience_count += len(experiences)
        self.recorder.record_history(experiences)

    async def submit_experiences(self) -> None:
        async with self.commit_lock:
            experiences = list(self.ready_experiences)
            self.ready_experiences.clear()
            metrics = await self.explorer.experience_pipeline.process.remote(experiences)
            metrics.update(self.collect_metrics())
            self.explorer.explore_step_num += 1
            self.explorer.monitor.log(metrics, self.explorer.explore_step_num)

    async def record_feedback(self, reward: float, msg_ids: List[str], task_id: str, run_id: int):
        exps = self.recorder.update_reward(
            reward=reward,
            msg_ids=msg_ids,
            task_id=task_id,
            run_id=run_id,
        )
        self.ready_experience_count += len(exps)
        self.ready_experiences.extend(exps)

    async def shutdown(self):
        if not self.running:
            self.logger.warning("Server is not running.")
            return
        self.sync_model_weights_task.cancel()
        self.serve_task.cancel()
        try:
            await self.serve_task
        except asyncio.CancelledError:
            pass
        self.running = False
        self.logger.info("API server shutdown.")
