"""Allocator module for managing inference engines."""

import asyncio
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import ray
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    placement_group_table,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from trinity.common.config import ExplorerConfig, InferenceModelConfig
from trinity.common.models.model import ModelWrapper
from trinity.utils.device import get_ray_resource_key
from trinity.utils.log import get_logger


@dataclass
class BundleResult:
    """Data class for storing the result of bundle allocation."""

    bundles: List[Dict[str, float]]
    actor_bundle_map: Dict[str, int]
    bundle_actor_map: Dict[int, str]


class Allocator:
    """Allocator class for managing inference engines."""

    def __init__(self, config: ExplorerConfig):
        """Initialize the Allocator."""
        self.config = config
        self.logger = get_logger(__name__)

    def get_actor_name(self, role: str, engine_id: int, node_id: int) -> str:
        """Generate a unique actor name based on the model config, engine ID, and node ID."""
        return f"{self.config.name}_{role}_model_{engine_id}_{node_id}"

    @property
    def placement_group_name(self) -> str:
        """Generate a descriptive name for the inference model placement group."""
        return f"{self.config.name}_rollout_model"

    def allocate_bundles(self) -> BundleResult:
        """Allocate bundles for the rollout model and auxiliary models based on the configuration."""
        rollout_model = self.config.rollout_model
        auxiliary_models = self.config.auxiliary_models
        model_configs = [("rollout", rollout_model)] + [
            (f"auxiliary_{index}", model) for index, model in enumerate(auxiliary_models)
        ]
        bundles: List[Dict[str, float]] = []
        actor_bundle_map: Dict[str, int] = {}
        bundle_actor_map: Dict[int, str] = {}
        bundle_id = 0
        for role, config in model_configs:
            gpus_per_bundle = config.gpu_per_engine // config.nnodes
            for engine_id in range(config.engine_num):
                for node_id in range(config.nnodes):
                    bundles.append({get_ray_resource_key(): float(gpus_per_bundle), "CPU": 1})
                    actor_name = self.get_actor_name(role, engine_id, node_id)
                    actor_bundle_map[actor_name] = bundle_id
                    bundle_actor_map[bundle_id] = actor_name
                    bundle_id += 1
        return BundleResult(
            bundles=bundles, actor_bundle_map=actor_bundle_map, bundle_actor_map=bundle_actor_map
        )

    def analyze_placement_group(self, pg: PlacementGroup, bundle_result: BundleResult):
        bundle_node_map = placement_group_table(pg)["bundles_to_node_id"]
        node_bundle_map = defaultdict(list)
        for bundle_id, node_id in bundle_node_map.items():
            node_bundle_map[node_id].append(bundle_id)
        for node_id, bundle_ids in node_bundle_map.items():
            self.logger.info("Node %s bundles:", node_id)
            for bundle_id in bundle_ids:
                actor_name = bundle_result.bundle_actor_map[bundle_id]
                self.logger.info("  > Bundle %s: Actor %s", bundle_id, actor_name)

    async def create_engine(
        self, config: InferenceModelConfig, role: str, engine_id: int
    ) -> ModelWrapper:
        config = deepcopy(config)
        config.engine_id = engine_id

        actor_bundle_lists = []
        for node_id in range(config.nnodes):
            actor_name = self.get_actor_name(role, engine_id, node_id)
            actor_bundle_lists.append((actor_name, self.bundle_result.actor_bundle_map[actor_name]))

        model_cls = None
        if config.engine_type.startswith("vllm"):
            from trinity.common.models.vllm_model import vLLMRolloutModel

            model_cls = vLLMRolloutModel
        elif config.engine_type == "sglang":
            from trinity.common.models.sglang_model import SGLangRolloutModel

            model_cls = SGLangRolloutModel
        elif config.engine_type == "tinker":
            from trinity.common.models.tinker_model import TinkerModel

            model_cls = TinkerModel
        elif config.engine_type == "external":
            return await get_external_model_wrapper(config=config)
        else:
            raise ValueError(f"Unsupported engine type: {config.engine_type}")

        self.logger.info(
            f"Creating inference_model {self.get_actor_name(role, engine_id, 0)} in {config.ray_namespace}."
        )
        return await get_model_wrapper(model_cls, config, self.pg, actor_bundle_lists)

    async def create_all_models(self) -> Tuple[List[ModelWrapper], List[List[ModelWrapper]]]:
        """Create all model actors for the rollout model and auxiliary models based on the configuration."""
        if self.config.mode == "colocate":
            return await self._create_colocate_model()

        self.bundle_result = self.allocate_bundles()
        self.pg = placement_group(
            self.bundle_result.bundles,
            strategy="PACK",
            name=self.placement_group_name,
        )
        await self.pg.ready()
        self.analyze_placement_group(self.pg, self.bundle_result)
        # create rollout_models
        tasks = []
        for engine_id in range(self.config.rollout_model.engine_num):
            tasks.append(
                asyncio.create_task(
                    self.create_engine(self.config.rollout_model, "rollout", engine_id)
                )
            )
        # create auxiliary models
        for index, auxiliary_model_config in enumerate(self.config.auxiliary_models):
            for engine_id in range(auxiliary_model_config.engine_num):
                tasks.append(
                    asyncio.create_task(
                        self.create_engine(auxiliary_model_config, f"auxiliary_{index}", engine_id)
                    )
                )
        # wait for all models to be created
        results = await asyncio.gather(*tasks)
        rollout_models: List[ModelWrapper] = results[: self.config.rollout_model.engine_num]
        auxiliary_models: List[List[ModelWrapper]] = [
            results[
                self.config.rollout_model.engine_num
                + sum(
                    self.config.auxiliary_models[i].engine_num for i in range(index)
                ) : self.config.rollout_model.engine_num
                + sum(self.config.auxiliary_models[i].engine_num for i in range(index + 1))
            ]
            for index in range(len(self.config.auxiliary_models))
        ]
        return rollout_models, auxiliary_models

    async def _create_colocate_model(
        self,
    ) -> Tuple[List[ModelWrapper], List[List[ModelWrapper]]]:
        """Create the rollout model without reserving a placement group.

        In colocate mode the trainer and rollout model share the same GPU. Reserving
        that GPU in a placement group would prevent the trainer from acquiring it.
        """
        config = deepcopy(self.config.rollout_model)
        config.engine_id = 0
        actor_name = self.get_actor_name("rollout", 0, 0)
        config.ray_actor_name = actor_name

        if not config.engine_type.startswith("vllm"):
            raise ValueError(
                f"Colocate mode only supports vLLM, but got engine type: {config.engine_type}"
            )

        from trinity.common.models.vllm_model import vLLMRolloutModel

        self.logger.info(
            "Creating colocated inference_model %s in %s.",
            actor_name,
            config.ray_namespace,
        )
        handler = (
            ray.remote(vLLMRolloutModel)
            .options(
                name=actor_name,
                num_cpus=0,
                num_gpus=0,
                namespace=config.ray_namespace,
            )
            .remote(config=config)
        )
        await handler.prepare.remote()
        server_address = await handler.get_api_server_url.remote()
        wrapper = ModelWrapper(models=[handler], config=config, api_address=server_address)
        await wrapper.prepare()
        return [wrapper], []

    def get_model(self, config: InferenceModelConfig, role: str, engine_id: int) -> ModelWrapper:
        """Get the model actor for the given role and engine ID."""
        actor_name = self.get_actor_name(role, engine_id, 0)
        try:
            model_actor = ray.get_actor(actor_name, namespace=config.ray_namespace)
            return ModelWrapper(model=model_actor, config=config)
        except ValueError:
            self.logger.error(
                "Actor %s not found in %s. Make sure the model is created.",
                actor_name,
                config.ray_namespace,
            )
            raise


async def get_model_wrapper(
    actor_cls,
    config: "InferenceModelConfig",
    pg: PlacementGroup,
    actor_bundle_list: List[Tuple[str, int]],
) -> ModelWrapper:
    """Get the Ray actor wrapper for the inference model.

    Creates Ray actors with appropriate GPU allocation and distributed communication
    setup based on the inferred launch mode.

    Args:
        actor_cls: The actor class to instantiate (e.g., vLLMRolloutModel, SGLangRolloutModel).
        config (InferenceModelConfig): The model config.
        pg (PlacementGroup): The placement group for the actors.
        actor_bundle_list (List[Tuple[str, int]]): The list of (actor_name, bundle_id) tuples
            for distributed setup.
    Returns:
        ModelWrapper: A wrapper for the model actors with distributed communication setup.
    """
    handlers = []
    for i, (actor_name, bundle_id) in enumerate(actor_bundle_list):
        engine_config = deepcopy(config)
        engine_config.ray_actor_name = actor_name
        engine_config.node_rank = i
        handlers.append(
            ray.remote(actor_cls)
            .options(
                name=actor_name,
                resources={get_ray_resource_key(): engine_config.gpu_per_engine / engine_config.nnodes},
                namespace=engine_config.ray_namespace,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_id,
                ),
            )
            .remote(config=engine_config)
        )
    if len(actor_bundle_list) > 1:
        # get master address and port from the first handler and set it to all handlers for distributed communication
        master_addr, master_port = await handlers[0].get_available_address.remote(random_port=True)
        for handler in handlers:
            await handler.set_master_addr_port.remote(master_addr, master_port)
    await asyncio.gather(*[handler.prepare.remote() for handler in handlers])
    server_address = await handlers[0].get_api_server_url.remote()
    wrapper = ModelWrapper(models=handlers, config=config, api_address=server_address)
    await wrapper.prepare()
    return wrapper


async def get_external_model_wrapper(config: InferenceModelConfig) -> ModelWrapper:
    """Get the ModelWrapper for the external model."""
    if config.engine_type != "external":
        raise ValueError("This function is only for creating external model wrapper.")
    api_address = os.environ.get(config.external_model_config.base_url_env)
    api_address = api_address.rstrip("/") if api_address else None
    api_key = os.environ.get(config.external_model_config.api_key_env, "EMPTY")
    config.api_key = api_key
    config.enable_openai_api = True
    return ModelWrapper(models=[], config=config, api_address=api_address)
