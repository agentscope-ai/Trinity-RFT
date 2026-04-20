# -*- coding: utf-8 -*-
"""Base reward function classes."""
from abc import ABC, abstractmethod
from typing import Dict


class RewardFn(ABC):
    """Base Reward Function Class."""

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, **kwargs) -> Dict[str, float]:
        pass
