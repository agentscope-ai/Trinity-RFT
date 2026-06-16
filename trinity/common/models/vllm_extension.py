# -*- coding: utf-8 -*-
"""Extensions for vLLM."""
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import ray
import torch

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    pack_tensors,
    unpack_tensor,
)

from trinity.common.models.utils import load_state_dict_iterator


@dataclass
class CheckpointWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for the checkpoint weight transfer example."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    sync_method: str
    namespace: str


@dataclass
class CheckpointWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for loading a checkpoint from disk."""

    checkpoint_path: Optional[str] = None
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS


@dataclass
class _PackedChunkMetadata:
    names: list[str]
    shapes: list[list[int]]
    dtype_names: list[str]
    tensor_sizes: list[int]


class CheckpointWeightTransferEngine(
    WeightTransferEngine[CheckpointWeightTransferInitInfo, CheckpointWeightTransferUpdateInfo]
):
    """Example engine that reads checkpoints on rank 0 and broadcasts weights.

    This is a teaching example, not a built-in backend. Register it at runtime
    with ``WeightTransferEngineFactory.register_engine`` before creating the LLM.
    """

    init_info_cls = CheckpointWeightTransferInitInfo
    update_info_cls = CheckpointWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        parallel_config: ParallelConfig,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, parallel_config, model)
        self.model_update_group: PyNcclCommunicator | None = None

    def init_transfer_engine(self, init_info: CheckpointWeightTransferInitInfo) -> None:
        dp_rank = self.parallel_config.data_parallel_index
        world_size_per_dp = self.parallel_config.world_size
        rank_within_dp = self.parallel_config.rank

        worker_rank = dp_rank * world_size_per_dp + rank_within_dp
        rank = worker_rank + init_info.rank_offset
        device = torch.accelerator.current_device_index()
        self.model_update_group = NCCLWeightTransferEngine._stateless_init_process_group(
            init_info.master_address,
            init_info.master_port,
            rank,
            init_info.world_size,
            device=device,
        )
        self._sync_method = init_info.sync_method
        self._namespace = init_info.namespace
        self._synchronizer = None

    def receive_weights(
        self,
        update_info: CheckpointWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self.model_update_group is None:
            raise RuntimeError(
                "Checkpoint weight transfer not initialized. " "Call init_transfer_engine() first."
            )

        if self.model_update_group.rank == 0:
            self._produce_checkpoint_weights(update_info, load_weights)
        else:
            self._consume_checkpoint_weights(load_weights)

    def _get_synchronizer(self):
        if self._synchronizer is None:
            from trinity.manager.synchronizer import Synchronizer

            self._synchronizer = Synchronizer.get_actor(namespace=self._namespace)
        return self._synchronizer

    def _produce_checkpoint_weights(
        self,
        update_info: CheckpointWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        assert self.model_update_group is not None

        if self._sync_method == "checkpoint":
            if update_info.checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided for checkpoint sync method")
            iterator = load_state_dict_iterator(checkpoint_dir=update_info.checkpoint_path)
        elif self._sync_method == "memory":
            synchronizer = self._get_synchronizer()
            iterator = (
                ray.get(weight_ref)
                for weight_ref in synchronizer.get_model_state_dict_iterator.remote()
            )

        def post_iter_func(item: tuple[str, torch.Tensor]) -> torch.Tensor:
            tensor = item[1]
            if tensor.device.type == "cuda":
                return tensor
            return tensor.to(device="cuda", non_blocking=True)

        while True:
            chunk = pack_tensors(
                iterator=iterator,
                post_iter_func=post_iter_func,
                buffer_size_bytes=update_info.packed_buffer_size_bytes,
            )
            if chunk is None:
                self.model_update_group.group.broadcast_obj(None, src=0)
                break

            metadata = _PackedChunkMetadata(
                names=chunk.names,
                shapes=chunk.shapes,
                dtype_names=[str(dtype).split(".")[-1] for dtype in chunk.dtypes],
                tensor_sizes=chunk.tensor_sizes,
            )
            self.model_update_group.group.broadcast_obj(metadata, src=0)
            self.model_update_group.broadcast(
                chunk.packed_tensor, src=0, stream=torch.cuda.current_stream()
            )
            load_weights(
                unpack_tensor(
                    chunk.packed_tensor,
                    metadata.names,
                    metadata.shapes,
                    chunk.dtypes,
                    metadata.tensor_sizes,
                )
            )

    def _consume_checkpoint_weights(
        self,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        assert self.model_update_group is not None

        while True:
            metadata = self.model_update_group.group.broadcast_obj(None, src=0)
            if metadata is None:
                break
            assert isinstance(metadata, _PackedChunkMetadata)

            dtypes = [getattr(torch, dtype_name) for dtype_name in metadata.dtype_names]
            packed_tensor = torch.empty(
                sum(metadata.tensor_sizes), dtype=torch.uint8, device="cuda"
            )
            self.model_update_group.broadcast(
                packed_tensor, src=0, stream=torch.cuda.current_stream()
            )
            load_weights(
                unpack_tensor(
                    packed_tensor,
                    metadata.names,
                    metadata.shapes,
                    dtypes,
                    metadata.tensor_sizes,
                )
            )

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError(
            "CheckpointWeightTransferEngine reads weights from disk on "
            "inference rank 0. Call update_weights with a checkpoint_path "
            "instead of trainer_send_weights."
        )


def register_checkpoint_weight_transfer_engine() -> None:
    """Register Trinity's checkpoint weight transfer backend with vLLM."""
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    if "checkpoint" in WeightTransferEngineFactory._registry:
        return

    WeightTransferEngineFactory.register_engine(
        name="checkpoint",
        module_path_or_cls="trinity.common.models.vllm_extension",
        class_name="CheckpointWeightTransferEngine",
    )
