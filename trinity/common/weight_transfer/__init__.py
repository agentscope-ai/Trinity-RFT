# -*- coding: utf-8 -*-
"""Bucketed model weight transfer between trainer and rollout workers.

Public API:

* :class:`NCCLSender` — trainer-side sender with double-buffered NCCL broadcast.
* :class:`NCCLReceiver` — rollout-side receiver with async and sync iterators.
* :class:`TensorMeta` — per-chunk metadata carried alongside bucket data.
"""
from trinity.common.weight_transfer.base import BaseReceiver, BaseSender, TensorMeta
from trinity.common.weight_transfer.nccl import (
    NCCLReceiver,
    NCCLSender,
    SGLangNCCLReceiver,
)

__all__ = [
    "BaseSender",
    "BaseReceiver",
    "NCCLSender",
    "NCCLReceiver",
    "SGLangNCCLReceiver",
    "TensorMeta",
]
