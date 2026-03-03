# -*- coding: utf-8 -*-
"""For distributed training with multiple process groups."""
import ipaddress
import socket
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
    assert backend == "nccl", "Only nccl backend is supported for now."

    from torch.distributed.distributed_c10d import is_nccl_available

    assert is_nccl_available()

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
