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

from trinity.common.config import (
    ExplorerConfig,
    InferenceModelConfig,
    LaunchMode,
    infer_launch_mode,
)
from trinity.common.models.model import ModelWrapper
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

    def _should_expand_dp(self, config: InferenceModelConfig) -> bool:
        """Check if SINGLE_NODE mode should expand DP into multiple independent actors.

        When nnodes>1 and DP>1, the allocator creates engine_num*DP actors,
        each running as a self-contained engine with dp_size=1 and nnodes=1.
        """
        return (
            infer_launch_mode(config.nnodes, config.data_parallel_size) == LaunchMode.SINGLE_NODE
            and config.nnodes > 1
            and config.data_parallel_size > 1
        )

    def _get_effective_engine_num(self, config: InferenceModelConfig) -> int:
        """Get the effective number of engines, accounting for SINGLE_NODE DP expansion."""
        if self._should_expand_dp(config):
            return config.engine_num * config.data_parallel_size
        return config.engine_num

    def allocate_bundles(self) -> BundleResult:
        """Allocate bundles for the rollout model and auxiliary models based on the configuration.

        Bundle allocation varies by launch mode:
        - SINGLE_NODE (nnodes=1 or DP=1): each engine gets 1 bundle with TP*DP*PP GPUs
        - SINGLE_NODE (nnodes>1 and DP>1): each engine_num*DP gets 1 bundle with TP*PP GPUs
        - HEADLESS: each engine gets nnodes bundles, each with (TP*PP)/nnodes GPUs
        """
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
            if self._should_expand_dp(config):
                # SINGLE_NODE with DP expansion: each DP rank is a separate actor
                gpus_per_bundle = config.tensor_parallel_size * config.pipeline_parallel_size
                effective_engine_num = config.engine_num * config.data_parallel_size
                for engine_id in range(effective_engine_num):
                    bundles.append({"GPU": float(gpus_per_bundle), "CPU": 1})
                    actor_name = self.get_actor_name(role, engine_id, 0)
                    actor_bundle_map[actor_name] = bundle_id
                    bundle_actor_map[bundle_id] = actor_name
                    bundle_id += 1
            else:
                # SINGLE_NODE (nnodes=1) or HEADLESS: existing logic
                gpus_per_bundle = config.gpu_per_engine // config.nnodes  # type: ignore
                for engine_id in range(config.engine_num):
                    for node_id in range(config.nnodes):
                        bundles.append({"GPU": float(gpus_per_bundle), "CPU": 1})
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
        launch_mode = infer_launch_mode(config.nnodes, config.data_parallel_size)

        actor_bundle_lists = []
        if self._should_expand_dp(config):
            # SINGLE_NODE with DP expansion: each effective engine has 1 actor
            actor_name = self.get_actor_name(role, engine_id, 0)
            actor_bundle_lists.append((actor_name, self.bundle_result.actor_bundle_map[actor_name]))
            # Set the DP rank for this expanded engine
            config.data_parallel_rank = engine_id % config.data_parallel_size
        else:
            # SINGLE_NODE (nnodes=1) or HEADLESS: cross nnodes actors per engine
            for node_id in range(config.nnodes):
                actor_name = self.get_actor_name(role, engine_id, node_id)
                actor_bundle_lists.append(
                    (actor_name, self.bundle_result.actor_bundle_map[actor_name])
                )

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
            f"Creating inference_model {self.get_actor_name(role, engine_id, 0)} "
            f"(launch_mode={launch_mode.value}) in {config.ray_namespace}."
        )
        return await get_model_wrapper(model_cls, config, self.pg, actor_bundle_lists)

    async def create_all_models(self) -> Tuple[List[ModelWrapper], List[List[ModelWrapper]]]:
        """Create all model actors for the rollout model and auxiliary models based on the configuration."""
        self.bundle_result = self.allocate_bundles()
        self.pg = placement_group(self.bundle_result.bundles, strategy="PACK")
        await self.pg.ready()
        self.analyze_placement_group(self.pg, self.bundle_result)
        # create rollout_models
        rollout_effective_num = self._get_effective_engine_num(self.config.rollout_model)
        tasks = []
        for engine_id in range(rollout_effective_num):
            tasks.append(
                asyncio.create_task(
                    self.create_engine(self.config.rollout_model, "rollout", engine_id)
                )
            )
        # create auxiliary models
        aux_effective_nums = []
        for index, auxiliary_model_config in enumerate(self.config.auxiliary_models):
            aux_effective_num = self._get_effective_engine_num(auxiliary_model_config)
            aux_effective_nums.append(aux_effective_num)
            for engine_id in range(aux_effective_num):
                tasks.append(
                    asyncio.create_task(
                        self.create_engine(auxiliary_model_config, f"auxiliary_{index}", engine_id)
                    )
                )
        # wait for all models to be created
        results = await asyncio.gather(*tasks)
        rollout_models: List[ModelWrapper] = results[:rollout_effective_num]
        offset = rollout_effective_num
        auxiliary_models: List[List[ModelWrapper]] = []
        for aux_num in aux_effective_nums:
            auxiliary_models.append(results[offset : offset + aux_num])
            offset += aux_num
        return rollout_models, auxiliary_models

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
    launch_mode = infer_launch_mode(config.nnodes, config.data_parallel_size)
    expand_dp = (
        launch_mode == LaunchMode.SINGLE_NODE
        and config.nnodes > 1
        and config.data_parallel_size > 1
    )

    for i, (actor_name, bundle_id) in enumerate(actor_bundle_list):
        engine_config = deepcopy(config)
        engine_config.ray_actor_name = actor_name

        if launch_mode == LaunchMode.HEADLESS:
            # Cross-node TP/PP: node_rank = i (0 for primary, 1+ for headless)
            engine_config.node_rank = i
            num_gpus = engine_config.gpu_per_engine / engine_config.nnodes  # type: ignore
        elif expand_dp:
            # SINGLE_NODE with DP expansion: each actor is self-contained
            engine_config.node_rank = 0
            num_gpus = engine_config.tensor_parallel_size * engine_config.pipeline_parallel_size
        else:
            # SINGLE_NODE: single actor, node_rank = 0
            engine_config.node_rank = 0
            num_gpus = engine_config.gpu_per_engine

        handlers.append(
            ray.remote(actor_cls)
            .options(
                name=actor_name,
                num_gpus=num_gpus,
                namespace=engine_config.ray_namespace,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_id,
                ),
            )
            .remote(config=engine_config)
        )

    if launch_mode == LaunchMode.HEADLESS and len(actor_bundle_list) > 1:
        # Only HEADLESS mode needs master addr/port for cross-node TP/PP coordination
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
