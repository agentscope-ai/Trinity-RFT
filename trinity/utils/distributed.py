# -*- coding: utf-8 -*-
"""For distributed training with multiple process groups."""
import ipaddress
import socket
from abc import abstractmethod
from datetime import timedelta
from typing import Any, Optional, Union

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


def is_ipv6_address(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return isinstance(ip, ipaddress.IPv6Address)
    except ValueError:
        return False


def get_available_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_endpoint(host: str, port: int) -> str:
    if is_ipv6_address(ip_str=host):
        return f"[{host}]:{port}"
    else:
        return f"{host}:{port}"


def is_port_available(port: int, host="127.0.0.1") -> bool:
    with socket.socket() as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def init_process_group(
    host: str,
    port: int,
    group_name: str,
    backend: Union[str, Backend] = "nccl",
    timeout: Optional[float] = None,
    world_size: int = -1,
    rank: int = -1,
    pg_options: Optional[Any] = None,
    device_id: Optional[torch.device] = None,
):
    """
    This function is used to initialize the process group. It requires torch >= 2.6.0
    """

    init_method = (
        f"tcp://[{host}]:{port}" if is_ipv6_address(ip_str=host) else f"tcp://{host}:{port}"
    )

    backend = Backend(backend)

    if timeout is None:
        timeout = default_pg_timeout
    else:
        timeout = timedelta(seconds=timeout)

    # backward compatible API
    store, rank, world_size = next(rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(group_name, store)
    pg, _ = _new_process_group_helper(
        group_size=world_size,
        group_rank=rank,
        global_ranks_in_group=[],
        backend=backend,
        store=prefix_store,
        group_name=group_name,
        timeout=timeout,
        device_id=device_id,
        **{"backend_options": pg_options},
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


class WeightTransferEngine:
    @abstractmethod
    def sync_weight(self, iterator):
        """Perform the weight sync."""

    @abstractmethod
    def teardown(self):
        """Tear down the weight sync group."""

    @staticmethod
    def create(
        engine_type: str, master_address: str, master_port: int, world_size: int, group_name: str
    ):
        """Factory method to create the appropriate weight transfer engine based on the rollout engine type."""
        if engine_type == "vllm":
            return VLLMWeightTransferEngine(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                group_name=group_name,
            )
        elif engine_type == "sglang":
            return SGLangWeightTransferEngine(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                group_name=group_name,
            )
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")


class VLLMWeightTransferEngine(WeightTransferEngine):
    """A helper class to manage weight synchronization using vLLM's API.

    Device-aware: uses HCCL engine on NPU and NCCL engine on GPU. Both engines
    expose the same ``trainer_init`` / ``trainer_send_weights`` API surface, so
    the only difference is the engine class and args dataclass used.
    """

    def __init__(self, master_address: str, master_port: int, world_size: int, group_name: str):
        """Initialize the process group for weight sync with vLLM's API."""
        from trinity.utils.device import is_npu

        del group_name  # vLLM's weight transfer engines do not require a group name
        self._is_npu = is_npu()
        if self._is_npu:
            from vllm_ascend.distributed.weight_transfer.hccl_engine import (
                HCCLWeightTransferEngine,
            )

            self._engine_cls = HCCLWeightTransferEngine
            self._model_update_group = HCCLWeightTransferEngine.trainer_init(
                dict(
                    master_address=master_address,
                    master_port=master_port,
                    world_size=world_size,
                )
            )
        else:
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine,
            )

            self._engine_cls = NCCLWeightTransferEngine
            self._model_update_group = NCCLWeightTransferEngine.trainer_init(
                dict(
                    master_address=master_address,
                    master_port=master_port,
                    world_size=world_size,
                )
            )

    def sync_weight(self, iterator):
        """Perform the weight sync using vLLM's API (HCCL on NPU, NCCL on GPU)."""
        if self._is_npu:
            from vllm_ascend.distributed.weight_transfer.hccl_engine import (
                HCCLTrainerSendWeightsArgs,
            )

            self._engine_cls.trainer_send_weights(
                iterator=iterator,
                trainer_args=HCCLTrainerSendWeightsArgs(
                    group=self._model_update_group,
                    packed=True,
                ),
            )
        else:
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLTrainerSendWeightsArgs,
            )

            self._engine_cls.trainer_send_weights(
                iterator=iterator,
                trainer_args=NCCLTrainerSendWeightsArgs(
                    group=self._model_update_group,
                    packed=True,
                ),
            )

    def teardown(self):
        self._model_update_group.destroy()


class SGLangWeightTransferEngine(WeightTransferEngine):
    """A helper class to manage NCCL weight synchronization using SGLang's API."""

    def __init__(self, master_address: str, master_port: int, world_size: int, group_name: str):
        """Initialize the NCCL process group for weight sync with SGLang's API."""
        self._model_update_group = init_process_group(
            host=master_address,
            port=master_port,
            group_name=group_name,
            backend="nccl",
            world_size=world_size,
            rank=0,
        )

    def sync_weight(self, iterator):
        """Perform the NCCL weight sync using SGLang's API."""
        for _, param in iterator:
            torch.distributed.broadcast(param, src=0, group=self._model_update_group)

    def teardown(self):
        torch.distributed.destroy_process_group(self._model_update_group)
