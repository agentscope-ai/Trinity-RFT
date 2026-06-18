# -*- coding: utf-8 -*-
"""CoD (Connect-the-Dots) workflows for FrozenLake environment."""

from trinity.common.workflows.connect_the_dots.frozen_lake.workflow import (
    CoDFrozenLakeWorkflow,
)
from trinity.common.workflows.connect_the_dots.frozen_lake.workflow_obscure import (
    CoDFrozenLakeObscureWorkflow,
    ALL_PERMUTATIONS,
    get_mapping_by_index,
    get_random_mapping,
)
from trinity.common.workflows.connect_the_dots.frozen_lake.prompts import (
    load_system_prompt,
    load_user_prompt,
)

__all__ = [
    "CoDFrozenLakeWorkflow",
    "CoDFrozenLakeObscureWorkflow",
    "ALL_PERMUTATIONS",
    "get_mapping_by_index",
    "get_random_mapping",
    "load_system_prompt",
    "load_user_prompt",
]
