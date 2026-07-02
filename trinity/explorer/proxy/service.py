import asyncio
import time
from collections import deque
from typing import Dict, List, Tuple

from trinity.common.constants import RunningStatus, SyncMethod
from trinity.common.models.model import ModelWrapper
from trinity.explorer.explorer import Explorer
from trinity.utils.log import get_logger


class ExplorerService:
    """Manages the lifecycle and operations of the Explorer API service.

    The proxy is a request router + model-weight sync coordinator for serve
    mode. Experience collection used to live here (SQL-mediated
    ``/feedback``/``/commit``); it has been removed in favor of rollout
    model-side recording stores drained through actor methods. Serve-mode
    external reward reporting is therefore pending
    (see the recording refactor plan).
    """

    def __init__(self, explorer: Explorer, listen_address: str = "localhost", port: int = 8010):
        self.logger = get_logger(__name__)
        self.explorer = explorer
        self.app = None
        self.port = port
        self.listen_address = listen_address
        self.running = False
        self.models: List[ModelWrapper] = explorer.models
        self.min_running_model_num = explorer.config.explorer.min_running_model_num
        self.check_interval = explorer.config.explorer.service_status_check_interval
        self.max_timeout = explorer.config.explorer.max_timeout
        self.running_model_ids: deque[int] = deque()  # indices of running models
        self.model_version_map: Dict[int, int] = {}  # model index -> model version
        self.sync_task_map: Dict[asyncio.Future, int] = {}  # sync task -> model index
        self.latest_model_version = 0

    async def serve(self) -> None:
        from trinity.explorer.proxy.app import run_app

        if self.running:
            self.logger.warning("Server is already running.")
            return

        self.running = True
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
        await self.models[index].sync_model_weights(latest_version, method=SyncMethod.CHECKPOINT)
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
                "Model does not have a valid API address; the OpenAI API server "
                "should have been started automatically during model preparation."
            )
        return model.api_address, self.model_version_map[model_id]

    def collect_metrics(self) -> Dict:
        metrics = {}
        for i, model in enumerate(self.models):
            metrics[f"rollout/model_{i}/total_request_count"] = model.request_count
            metrics[f"rollout/model_{i}/model_version"] = model.model_version
        return metrics

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
