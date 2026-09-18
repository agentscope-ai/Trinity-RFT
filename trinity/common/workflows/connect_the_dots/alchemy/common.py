# -*- coding: utf-8 -*-
"""Shared types and data structures for Alchemy game variants."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Recipe key: tuple(sorted([e1, e2])) to handle both order and self-combination
RecipeKey = Tuple[str, str]


@dataclass
class GameInstance:
    """A complete definition of one Alchemy game."""

    target: str
    recipes: Dict[RecipeKey, str]  # {("air","air"): "pressure", ...}
    starting_inventory: Dict[str, int]  # {element: count} available at start
    all_elements: Set[str]  # All elements in this subgraph
    solution_path: List[Tuple[str, str, str]]  # [(e1, e2, result), ...]
    depth: int  # Depth of target element
    local_tiers: Dict[str, int] = None  # Tiers computed within this subgraph

    def __post_init__(self):
        if self.local_tiers is None:
            self.local_tiers = self._compute_local_tiers()
            # Filter out recipes that violate local tier ordering
            self.recipes = {
                key: result for key, result in self.recipes.items()
                if not (key[0] in self.local_tiers and key[1] in self.local_tiers and result in self.local_tiers)
                or self.local_tiers[result] > max(self.local_tiers[key[0]], self.local_tiers[key[1]])
            }

    def _compute_local_tiers(self) -> Dict[str, int]:
        """Compute tiers from recipe graph structure.

        Leaf elements (not produced by any recipe) are tier 0.
        Other elements: tier = max(tier(input1), tier(input2)) + 1.
        """
        all_ingredients: Set[str] = set()
        all_results: Set[str] = set()
        for key, result in self.recipes.items():
            all_ingredients.update(key)
            all_results.add(result)
        leaves = (all_ingredients | set(self.starting_inventory)) - all_results

        tiers = {elem: 0 for elem in leaves}
        changed = True
        while changed:
            changed = False
            for key, result in self.recipes.items():
                if result in tiers:
                    continue
                e1, e2 = key
                if e1 in tiers and e2 in tiers:
                    tiers[result] = max(tiers[e1], tiers[e2]) + 1
                    changed = True
        return tiers
