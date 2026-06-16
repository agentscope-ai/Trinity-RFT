"""A centralized synchronizer for coordinating explorer and trainer."""

import asyncio
import os
import shutil
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, Union

import ray

from trinity.common.config import Config
from trinity.common.constants import RunningStatus, SyncMethod
from trinity.utils.log import get_logger


class Synchronizer:
    """
    A central component to manage synchronization of models and states between
    the trainer and one or more explorers in a distributed training setup.

    Attributes:
        trainer_status: Current status of the trainer (e.g., running, waiting).
        explorer_status_counts: Dictionary tracking the number of explorers in each status.
        _ready_condition: Async condition variable for signaling state changes.
        model_state_dict: The latest model weights.
        model_version: Version number of the current model.
        checkpoint_shard_counter: Tracks how many shards are received from trainer for a specific train step.
    """

    def __init__(self, config: Config, module_ref: ray.actor.ActorHandle):
        self.logger = get_logger("synchronizer", in_ray_actor=True)
        self.config = config
        self.enable_lora = config.explorer.rollout_model.enable_lora
        self.trainer_status = RunningStatus.STOPPED
        self.explorer_status_counts: Dict[RunningStatus, int] = defaultdict(lambda: 0)
        self._ready_condition = asyncio.Condition()
        self.model_state_dict = None
        self.model_version = 0
        self.model_path = None
        self.checkpoint_shard_counter = defaultdict(lambda: 0)
        self.ref_count = 0
        self._modules = {module_ref}
        self._modules_lock = asyncio.Lock()
        asyncio.create_task(self._check_modules())
        if (
            self.config.mode != "bench"
            and self.config.synchronizer.sync_method == SyncMethod.CHECKPOINT
        ):
            asyncio.create_task(self._find_latest_state_dict())

    async def add_module(self, module_ref: ray.actor.ActorHandle) -> None:
        """Adds a module to be tracked by the synchronizer.

        Args:
            module_ref: The Ray actor handle of the module to track.
        """
        async with self._modules_lock:
            if module_ref not in self._modules:
                self._modules.add(module_ref)

    async def _check_modules(self) -> None:
        while len(self._modules) > 0:
            alive_modules = set()
            async with self._modules_lock:
                for module in self._modules:
                    try:
                        is_alive_ref = module.is_alive.remote()
                        await asyncio.wait_for(is_alive_ref, timeout=5.0)
                        alive_modules.add(module)
                    except ray.exceptions.RayActorError:
                        pass
                    except asyncio.TimeoutError:
                        ray.cancel(is_alive_ref)
                        alive_modules.add(module)
                self._modules = alive_modules
            await asyncio.sleep(1)
        self.logger.info("Synchronizer stopped.")
        try:
            ray.actor.exit_actor()
        except Exception:
            pass

    async def _find_latest_state_dict(self) -> None:
        if self.config.trainer.trainer_type == "verl":
            await self._find_verl_latest_state_dict()
        elif self.config.trainer.trainer_type == "tinker":
            await self._find_tinker_latest_state_dict()
        else:
            self.logger.warning(
                "Synchronizer does not support this trainer type. Please use `verl` or `tinker`."
            )

    async def _find_verl_latest_state_dict(self) -> None:
        default_local_dir = self.config.checkpoint_job_dir
        local_latest_state_dict_iteration = os.path.join(
            default_local_dir, "latest_state_dict_iteration.txt"
        )
        while True:
            if os.path.exists(local_latest_state_dict_iteration):
                current_model_version = self.model_version
                try:
                    with open(local_latest_state_dict_iteration, "r") as f:
                        latest_model_version = int(f.read().strip())
                except (IOError, ValueError) as e:
                    self.logger.warning(f"Failed to read or parse state dict iteration file: {e}")
                    continue
                if latest_model_version > current_model_version:
                    self.logger.info(
                        f"Synchronizer has found a new model state dict at step {latest_model_version}."
                    )
                    await self.set_model_state_dict(None, latest_model_version)
                    # remove the previous checkpoints to save disk space
                    await self._remove_previous_state_dict(current_model_version)
            await asyncio.sleep(1)

    async def _remove_previous_state_dict(self, previous_model_version: int) -> None:
        previous_state_dict_dir = os.path.join(
            self.config.checkpoint_job_dir, f"global_step_{previous_model_version}"
        )
        if os.path.exists(previous_state_dict_dir):
            # check if it's a full checkpoint, only remove checkpoints for sync
            if not os.path.exists(os.path.join(previous_state_dict_dir, ".full_checkpoint")):
                self.logger.info(
                    f"Removing previous checkpoint for sync at step {previous_model_version}."
                )
                shutil.rmtree(previous_state_dict_dir, ignore_errors=True)

    async def _find_tinker_latest_state_dict(self) -> None:
        default_local_dir = self.config.checkpoint_job_dir
        local_latest_state_dict_iteration = os.path.join(
            default_local_dir, "latest_state_dict_iteration.txt"
        )
        while True:
            if os.path.exists(local_latest_state_dict_iteration):
                try:
                    with open(local_latest_state_dict_iteration, "r") as f:
                        latest_model_version = int(f.read().strip())
                except (IOError, ValueError) as e:
                    self.logger.warning(f"Failed to read or parse state dict iteration file: {e}")
                    continue
                if latest_model_version > self.model_version:
                    self.logger.info(
                        f"Synchronizer has found a new remote tinker sampler path at step {latest_model_version}."
                    )
                    remote_path_file = os.path.join(
                        default_local_dir,
                        f"global_step_{latest_model_version}",
                        "remote_sampler_path.txt",
                    )
                    with open(remote_path_file, "r") as f:
                        remote_sampler_path = f.read().strip()
                    await self.set_model_state_dict(remote_sampler_path, latest_model_version)
            await asyncio.sleep(1)

    async def set_trainer_status(self, status: RunningStatus):
        """Update the status of the trainer."""
        async with self._ready_condition:
            self.trainer_status = status
            if status == RunningStatus.STOPPED:
                self._ready_condition.notify_all()

    def trainer_requires_sync(self) -> bool:
        """Check if the trainer is require sync."""
        return self.trainer_status == RunningStatus.REQUIRE_SYNC

    async def set_explorer_status(
        self, status: RunningStatus, old_status: Optional[RunningStatus] = None
    ):
        """
        Update the status count for an explorer.

        Args:
            status: New status of the explorer.
            old_status: Previous status if changing from one to another.
        """
        if old_status is not None:
            assert (
                old_status in self.explorer_status_counts
            ), f"Invalid explorer status {old_status}"
            assert old_status != status, f"Invalid status change from {old_status} to {status}"
            self.explorer_status_counts[old_status] -= 1
            assert (
                self.explorer_status_counts[old_status] >= 0
            ), f"Invalid status count {old_status} (new status {status})"
        if status not in self.explorer_status_counts:
            self.explorer_status_counts[status] = 0
        self.explorer_status_counts[status] += 1

    def explorer_requires_sync(self) -> bool:
        """Check if any explorer is require sync."""
        return self.explorer_status_counts[RunningStatus.REQUIRE_SYNC] > 0

    async def set_model_state_dict(
        self, model_state_dict: Union[dict, None, str], trainer_step: int
    ):
        """
        Set the new model state and update the version.

        Args:
            model_state_dict: The PyTorch model state dictionary.
            trainer_step: Step number associated with this model version.
        """
        async with self._ready_condition:
            self.model_state_dict = model_state_dict
            self.model_version = trainer_step
            # TODO: check model_path for different trainer types
            self.model_path = os.path.join(
                self.config.checkpoint_job_dir, f"global_step_{trainer_step}", "actor"
            )
            self.logger.info(f"Set model state dict version to {trainer_step}.")
            self._ready_condition.notify_all()

    def get_model_state_dict(self):
        """Return the current model state and its version."""
        return self.model_state_dict, self.model_version

    def get_model_state_dict_iterator(self) -> Iterator[Tuple[str, object]]:
        """Yield the current in-memory model state for Ray streaming consumers."""
        if self.model_state_dict is None:
            return
        if not isinstance(self.model_state_dict, dict):
            raise ValueError("Model state dict is not in expected format (dict).")
        for item in self.model_state_dict.items():
            yield item

    async def get_state_dict_meta(self):
        """
        Return metadata about the model state (names, data types, shapes).

        Returns:
            List of tuples: (name, dtype, shape).
        """
        if self.model_state_dict is None:
            return None
        if isinstance(self.model_state_dict, tuple):
            async with self._ready_condition:
                await self._ready_condition.wait_for(
                    lambda: not isinstance(self.model_state_dict, tuple)
                )
        update_weight_args_list = []
        for name, param in self.model_state_dict.items():
            update_weight_args_list.append(
                (name, str(param.dtype).split(".")[-1], tuple(param.shape))
            )
        return update_weight_args_list

    async def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        """
        Notify the explorer actor to setup weight sync group.

        This is used to initialize NCCL-based synchronization for distributed training.

        Args:
            master_address: IP address of the master node.
            master_port: Port used for synchronization.
            state_dict_meta: Metadata of the model parameters.
        """
        explorer = ray.get_actor(self.config.explorer.name, namespace=self.config.ray_namespace)
        await explorer.setup_weight_sync_group.remote(master_address, master_port)
        if state_dict_meta is not None:
            await explorer.set_state_dict_meta.remote(state_dict_meta)

    async def coordinate_weight_sync_setup(self, timeout: int = None):
        """Orchestrate NCCL weight sync group setup between Trainer and Explorer.

        1. Get rendezvous info (addr/port/meta) from Trainer
        2. Both Trainer and Explorer join the NCCL group concurrently
        3. Set state_dict_meta on Explorer for weight sync
        """
        trainer = ray.get_actor(self.config.trainer.name, namespace=self.config.ray_namespace)
        explorer = ray.get_actor(self.config.explorer.name, namespace=self.config.ray_namespace)

        addr, port, meta = await trainer.get_weight_sync_info.remote()
        world_size = self.config.synchronizer.explorer_world_size + 1  # type: ignore
        timeout = timeout or self.config.synchronizer.sync_timeout

        group_name = self.config.synchronizer.group_name
        self.logger.info(f"Coordinating weight sync setup: {addr}:{port}, world_size={world_size}")
        await asyncio.gather(
            trainer.setup_weight_sync_group.remote(addr, port, world_size, group_name, timeout),
            explorer.setup_weight_sync_group.remote(addr, port, world_size, group_name, timeout),
        )
        await explorer.set_state_dict_meta.remote(meta)
        self.logger.info("Weight sync group setup complete.")

    async def coordinate_weight_sync_teardown(self):
        """Orchestrate NCCL weight sync group teardown.

        Explorer tears down first (so no broadcasts are in-flight),
        then Trainer. Errors from dead actors are caught so that a
        crashed Explorer/Trainer does not block the remaining teardown.
        """
        self.logger.info("Coordinating weight sync teardown.")
        for role, name in [
            ("Explorer", self.config.explorer.name),
            ("Trainer", self.config.trainer.name),
        ]:
            try:
                actor = ray.get_actor(name, namespace=self.config.ray_namespace)
                await actor.teardown_weight_sync_group.remote()
            except (ray.exceptions.RayActorError, ValueError) as e:
                self.logger.warning(f"{role} already dead, skipping teardown: {e}")
        self.logger.info("Weight sync group teardown complete.")

    async def wait_new_model_state_dict(self, current_version: int) -> int:
        """
        Wait until a new model state is available.

        Args:
            current_version: Current model version known to one explorer.

        Returns:
            The new model version after it has been updated.
        """
        async with self._ready_condition:
            assert (
                self.model_version >= current_version
            ), f"The model version in Synchronizer ({self.model_version}) should be no smaller than that in Explorer ({current_version})!"
            await self.set_explorer_status(
                RunningStatus.REQUIRE_SYNC, old_status=RunningStatus.RUNNING
            )
            if self.model_version == current_version:
                if self.trainer_status != RunningStatus.STOPPED:
                    await asyncio.wait_for(
                        self._ready_condition.wait(),
                        timeout=self.config.synchronizer.sync_timeout,
                    )
            await self.set_explorer_status(
                RunningStatus.RUNNING, old_status=RunningStatus.REQUIRE_SYNC
            )
            return self.model_version

    async def notify_no_new_model_state_dict(self) -> None:
        """
        Notify the explorer that there is no new model state.
        Used for `wait_new_model_state_dict`.
        """
        async with self._ready_condition:
            self._ready_condition.notify_all()

    async def get_latest_model_version(self) -> int:
        """
        Get the latest model version available in the synchronizer.

        Returns:
            The current model version.
        """
        async with self._ready_condition:
            return self.model_version

    async def get_latest_model_path(self, use_huggingface: bool = False) -> Optional[str]:
        """
        Get the latest model path available in the synchronizer.

        Args:
            use_huggingface: Whether to return the Hugging Face model path.

        Returns:
            The current model path.
        """
        async with self._ready_condition:
            if self.model_path and use_huggingface:
                return os.path.join(self.model_path, "huggingface")
            return self.model_path

    async def ready_to_nccl_sync(self, module: str, trainer_step: int) -> Union[int, None]:
        """
        Prepare for NCCL-based synchronization between modules.

        Only supports one explorer currently.

        Args:
            module: Either 'trainer' or 'explorer'.
            trainer_step: Step number from the trainer.

        Returns:
            The model version if both sides are ready; otherwise None.
        """
        assert (
            sum(self.explorer_status_counts.values()) == 1
        ), "NCCL sync is only supported for one explorer."

        async def sync_failed():
            if module == "explorer":
                another_module = "Trainer"
                await self.set_explorer_status(
                    RunningStatus.RUNNING, old_status=RunningStatus.REQUIRE_SYNC
                )
            else:
                another_module = "Explorer"
                self.trainer_status = RunningStatus.RUNNING
            self.logger.error(f"{another_module} is not ready for model weight sync.")
            return None

        non_stop_cnt = sum(
            value
            for key, value in self.explorer_status_counts.items()
            if key != RunningStatus.STOPPED
        )
        if non_stop_cnt == 0:
            return await sync_failed()

        async with self._ready_condition:
            try:
                if module == "trainer":
                    self.model_version = trainer_step
                    self.trainer_status = RunningStatus.REQUIRE_SYNC
                    self._ready_condition.notify_all()
                    if self.explorer_status_counts[RunningStatus.REQUIRE_SYNC] != 1:
                        await asyncio.wait_for(
                            self._ready_condition.wait_for(
                                lambda: self.explorer_status_counts[RunningStatus.REQUIRE_SYNC]
                                + self.explorer_status_counts[RunningStatus.STOPPED]
                                == 1,
                            ),
                            timeout=self.config.synchronizer.sync_timeout,
                        )
                        if self.explorer_status_counts[RunningStatus.STOPPED] == 1:
                            return await sync_failed()
                    await self.set_explorer_status(
                        RunningStatus.RUNNING,
                        old_status=RunningStatus.REQUIRE_SYNC,
                    )
                elif module == "explorer":
                    await self.set_explorer_status(
                        RunningStatus.REQUIRE_SYNC, old_status=RunningStatus.RUNNING
                    )
                    self._ready_condition.notify_all()
                    if self.trainer_status != RunningStatus.REQUIRE_SYNC:
                        await asyncio.wait_for(
                            self._ready_condition.wait_for(
                                lambda: self.trainer_status
                                in {RunningStatus.REQUIRE_SYNC, RunningStatus.STOPPED},
                            ),
                            timeout=self.config.synchronizer.sync_timeout,
                        )
                        if self.trainer_status == RunningStatus.STOPPED:
                            return await sync_failed()
                    self.trainer_status = RunningStatus.RUNNING
                return self.model_version
            except asyncio.TimeoutError:
                return await sync_failed()

    @classmethod
    def get_actor(cls, config: Optional[Config] = None, namespace: Optional[str] = None):
        """
        Get or create a remote Ray actor for the Synchronizer.

        Args:
            config: Optional configuration to use for creating the actor.
            namespace: Optional Ray namespace for the actor.

        Returns:
            A reference to the Synchronizer actor.
        """
        if config is not None:
            module_ref = ray.get_runtime_context().current_actor
            synchronizer = (
                ray.remote(cls)
                .options(
                    name="synchronizer",
                    namespace=config.ray_namespace,
                    get_if_exists=True,
                    lifetime="detached",
                )
                .remote(config, module_ref=module_ref)
            )
            synchronizer.add_module.remote(module_ref)
            return synchronizer
        return ray.get_actor("synchronizer", namespace=namespace)
