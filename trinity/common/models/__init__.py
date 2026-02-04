import asyncio
from collections import defaultdict
from typing import List, Tuple

import ray
from ray.util.placement_group import placement_group, placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from trinity.common.config import Config, InferenceModelConfig
from trinity.common.constants import DEBUG_NAMESPACE
from trinity.common.models.model import InferenceModel
from trinity.utils.log import get_logger


class _BundleAllocator:
    """An allocator for bundles."""

    def __init__(self, num_gpus: int) -> None:
        self.logger = get_logger(__name__, in_ray_actor=True)
        bundles = [{"GPU": 1} for _ in range(num_gpus)]
        self.pg = placement_group(bundles, strategy="PACK")
        ray.get(self.pg.ready())
        # to address https://github.com/ray-project/ray/issues/51117
        # aggregate bundles belonging to the same node
        bundle_node_map = placement_group_table(self.pg)["bundles_to_node_id"]
        node_bundle_map = defaultdict(list)
        for bundle_id, node_id in bundle_node_map.items():
            node_bundle_map[node_id].append(bundle_id)
        self.node_bundle_list = [value for value in node_bundle_map.values()]
        self.node_list = [key for key in node_bundle_map.keys()]
        self.nid = 0
        self.bid = 0

    def allocate(self, num: int) -> list:
        # allocate num bundles from current node
        if self.bid + num > len(self.node_bundle_list[self.nid]):
            raise ValueError(
                "Bundle allocation error, a tensor parallel group"
                " is allocated across multiple nodes."
            )
        bundle_list = self.node_bundle_list[self.nid][self.bid : self.bid + num]
        self.logger.info(f"Allocate bundles {bundle_list} on node {self.node_list[self.nid]}.")
        self.bid += num
        if self.bid == len(self.node_bundle_list[self.nid]):
            self.bid = 0
            self.nid += 1
        return bundle_list


def create_explorer_models(
    config: Config,
) -> Tuple[List[InferenceModel], List[List[InferenceModel]]]:
    """Create `engine_num` rollout models.

    Each model has `tensor_parallel_size` workers.
    """
    logger = get_logger(__name__)

    rollout_engines = []
    if config.explorer.rollout_model.engine_type.startswith("vllm"):
        from trinity.common.models.vllm_model import vLLMRolloutModel

        engine_cls = vLLMRolloutModel
    elif config.explorer.rollout_model.engine_type == "tinker":
        from trinity.common.models.tinker_model import TinkerModel

        engine_cls = TinkerModel
        namespace = config.ray_namespace
        rollout_engines = [
            ray.remote(engine_cls)
            .options(
                name=f"{config.explorer.name}_rollout_model_{i}",
                namespace=namespace,
            )
            .remote(
                config=config.explorer.rollout_model,
            )
            for i in range(config.explorer.rollout_model.engine_num)
        ]
        auxiliary_engines = [
            [
                ray.remote(engine_cls)
                .options(
                    name=f"{config.explorer.name}_auxiliary_model_{i}_{j}",
                    namespace=namespace,
                )
                .remote(
                    config=config.explorer.auxiliary_models[i],
                )
                for j in range(model_config.engine_num)
            ]
            for i, model_config in enumerate(config.explorer.auxiliary_models)
        ]
        return rollout_engines, auxiliary_engines
    else:
        raise ValueError(f"Unknown engine type: {config.explorer.rollout_model.engine_type}")

    if config.mode == "colocate":
        rollout_engine = (
            ray.remote(vLLMRolloutModel)
            .options(
                name=f"{config.explorer.name}_rollout_model_0",
                num_cpus=0,
                num_gpus=0,
                namespace=config.ray_namespace,
            )
            .remote(
                config=config.explorer.rollout_model,
            )
        )
        return [rollout_engine], []

    num_gpus = (
        config.explorer.rollout_model.engine_num
        * config.explorer.rollout_model.tensor_parallel_size
        + sum(
            [
                model.engine_num * model.tensor_parallel_size
                for model in config.explorer.auxiliary_models
            ]
        )
    )

    allocator = _BundleAllocator(num_gpus=num_gpus)

    # create rollout models
    rollout_engines = create_vllm_inference_models(
        config=config.explorer.rollout_model,
        allocator=allocator,
        actor_name=f"{config.explorer.name}_rollout_model",
    )

    if config.explorer.rollout_model.enable_history:
        logger.info(
            "Model History recording is enabled. Please periodically extract "
            "history via `extract_experience_from_history` to avoid out-of-memory issues."
        )

    # create auxiliary models
    auxiliary_engines = []
    for i, model_config in enumerate(config.explorer.auxiliary_models):
        engines = create_vllm_inference_models(
            config=model_config,
            allocator=allocator,
            actor_name=f"{config.explorer.name}_auxiliary_model_{i}",
        )
        auxiliary_engines.append(engines)

    return rollout_engines, auxiliary_engines


def create_vllm_inference_models(
    config: InferenceModelConfig,
    allocator: _BundleAllocator,
    actor_name: str,
) -> List:
    from trinity.common.models.vllm_model import vLLMRolloutModel

    models = []
    for i in range(config.engine_num):
        bundles_for_engine = allocator.allocate(config.tensor_parallel_size)
        config.bundle_indices = ",".join([str(bid) for bid in bundles_for_engine])
        models.append(
            ray.remote(vLLMRolloutModel)
            .options(
                name=f"{actor_name}_{i}",
                num_cpus=0,
                num_gpus=0 if config.tensor_parallel_size > 1 else 1,
                namespace=config.ray_namespace,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=allocator.pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundles_for_engine[0],
                ),
            )
            .remote(
                config=config,
            )
        )
    return models


async def create_debug_explorer_model(config: Config) -> None:
    """Create explorer inference models for debugging."""
    logger = get_logger(__name__)
    logger.info("Creating inference models for debugging...")
    # only create one engine for each model
    config.explorer.rollout_model.engine_num = 1
    for model in config.explorer.auxiliary_models:
        model.engine_num = 1
    rollout_models, auxiliary_models = create_explorer_models(config)
    # make sure models are started
    prepare_refs = [m.prepare.remote() for m in rollout_models]
    prepare_refs.extend(m.prepare.remote() for models in auxiliary_models for m in models)
    await asyncio.gather(*prepare_refs)
    logger.info(
        "----------------------------------------------------\n"
        "Inference models started successfully for debugging.\n"
        "Press Ctrl+C to exit.\n"
        "----------------------------------------------------"
    )

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")


def get_debug_explorer_model(config: Config) -> Tuple[InferenceModel, List[InferenceModel]]:
    """Get the inference models for debugging.
    The models must be created by `create_debug_explorer_model` in another process first.
    """
    import ray

    rollout_model = ray.get_actor(
        f"{config.explorer.name}_rollout_model_0", namespace=DEBUG_NAMESPACE
    )
    auxiliary_models = []
    for i in range(len(config.explorer.auxiliary_models)):
        model = ray.get_actor(
            f"{config.explorer.name}_auxiliary_model_{i}_0", namespace=DEBUG_NAMESPACE
        )
        auxiliary_models.append(model)
    return rollout_model, auxiliary_models
