# -*- coding: utf-8 -*-
"""Procedural graph generator for Random Alchemy game.

Generates DAG-structured recipe graphs from scratch without external data.
Each graph has tiered elements, configurable branching, cross-tier connections,
and noise (distractor) elements.
"""

import random
import string
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from trinity.common.workflows.connect_the_dots.alchemy.common import (
    GameInstance,
    RecipeKey,
)


def _generate_name(rng: random.Random, existing: Set[str], length_min: int = 3, length_max: int = 5) -> str:
    """Generate a unique random element name (lowercase to match env recipe lookup)."""
    while True:
        length = rng.randint(length_min, length_max)
        name = "".join(rng.choices(string.ascii_lowercase, k=length))
        if name not in existing:
            return name


class RandomGraphGenerator:
    """Procedural DAG generator for alchemy-style games.

    Generates a complete recipe graph with:
    - Tiered elements (tier 0 = base, higher = synthesized)
    - Configurable branching and cross-tier connections
    - Noise (distractor) elements that are dead ends
    """

    def generate_graph(
        self,
        rng: random.Random,
        num_tiers: int,
        base_elements: int,
        tier_shrink: float,
        min_recipes_per_element: int,
        max_recipes_per_element: int,
        max_tier_gap: int,
        cross_tier_prob: float,
        noise_node_ratio: float,
        noise_chain_depth: int,
    ) -> Tuple[Dict[int, List[str]], Dict[RecipeKey, str], Dict[str, int], Set[str]]:
        """Generate a complete recipe graph.

        Returns:
            (tier_elements, recipes, local_tiers, noise_elements)
            - tier_elements: {tier: [element_names]} (includes noise)
            - recipes: {(e1, e2): result}
            - local_tiers: {element: tier}
            - noise_elements: set of element names that are distractors
        """
        all_names: Set[str] = set()

        # Step 1: Determine elements per tier
        tier_sizes = [base_elements]
        for t in range(1, num_tiers):
            size = max(1, round(tier_sizes[t - 1] * tier_shrink))
            tier_sizes.append(size)

        # Step 2: Generate element names per tier
        tier_elements: Dict[int, List[str]] = {}
        for t in range(num_tiers):
            tier_elements[t] = []
            for _ in range(tier_sizes[t]):
                name = _generate_name(rng, all_names)
                all_names.add(name)
                tier_elements[t].append(name)

        # Step 3: Generate recipes for each non-leaf element
        recipes: Dict[RecipeKey, str] = {}
        used_keys: Set[RecipeKey] = set()

        for t in range(1, num_tiers):
            for elem in tier_elements[t]:
                num_recipes = rng.randint(min_recipes_per_element, max_recipes_per_element)
                for _ in range(num_recipes):
                    key = self._make_recipe(
                        rng, elem, t, tier_elements, max_tier_gap,
                        cross_tier_prob, used_keys,
                    )
                    if key is not None:
                        recipes[key] = elem
                        used_keys.add(key)

        # Step 4: Add noise elements and recipes
        noise_elements: Set[str] = set()
        if noise_node_ratio > 0:
            noise_elements = self._add_noise(
                rng, tier_elements, recipes, used_keys, all_names,
                noise_node_ratio, noise_chain_depth, max_tier_gap, cross_tier_prob,
            )

        # Step 5: Compute local tiers
        local_tiers: Dict[str, int] = {}
        for t, elems in tier_elements.items():
            for e in elems:
                local_tiers[e] = t

        return tier_elements, recipes, local_tiers, noise_elements

    def _make_recipe(
        self,
        rng: random.Random,
        result_elem: str,
        result_tier: int,
        tier_elements: Dict[int, List[str]],
        max_tier_gap: int,
        cross_tier_prob: float,
        used_keys: Set[RecipeKey],
        max_attempts: int = 50,
    ) -> Optional[RecipeKey]:
        """Generate one recipe for a given result element.

        - input1: always from tier T-1
        - input2: from tier T-1 (normal) or lower tier (cross-tier, with probability)
        """
        for _ in range(max_attempts):
            # input1: must be from tier T-1
            input1 = rng.choice(tier_elements[result_tier - 1])

            # input2: cross-tier or same tier
            if rng.random() < cross_tier_prob and result_tier >= 2:
                min_tier = max(0, result_tier - max_tier_gap)
                max_input_tier = result_tier - 2  # must be lower than T-1
                if min_tier <= max_input_tier:
                    input2_tier = rng.randint(min_tier, max_input_tier)
                    input2 = rng.choice(tier_elements[input2_tier])
                else:
                    input2 = rng.choice(tier_elements[result_tier - 1])
            else:
                input2 = rng.choice(tier_elements[result_tier - 1])

            key: RecipeKey = tuple(sorted([input1, input2]))
            if key not in used_keys:
                return key

        return None  # Could not find a unique recipe

    def _add_noise(
        self,
        rng: random.Random,
        tier_elements: Dict[int, List[str]],
        recipes: Dict[RecipeKey, str],
        used_keys: Set[RecipeKey],
        all_names: Set[str],
        noise_node_ratio: float,
        noise_chain_depth: int,
        max_tier_gap: int,
        cross_tier_prob: float,
    ) -> Set[str]:
        """Add noise (distractor) elements that are dead ends.

        Returns the set of all noise element names.
        """
        num_tiers = len(tier_elements)
        # Snapshot original sizes before adding noise
        original_sizes = {t: len(elems) for t, elems in tier_elements.items()}
        all_noise: Set[str] = set()

        for depth_round in range(noise_chain_depth):
            new_noise: Dict[int, List[str]] = {}

            for t in range(1, num_tiers):
                # Number of noise nodes based on ORIGINAL tier size, not inflated
                num_noise = max(0, round(original_sizes[t] * noise_node_ratio))
                if depth_round > 0:
                    # Fewer noise nodes in deeper rounds
                    num_noise = max(0, round(num_noise * 0.5))

                for _ in range(num_noise):
                    noise_name = _generate_name(rng, all_names)
                    all_names.add(noise_name)

                    # Generate one recipe for this noise node
                    key = self._make_recipe(
                        rng, noise_name, t, tier_elements, max_tier_gap,
                        cross_tier_prob, used_keys,
                    )
                    if key is not None:
                        recipes[key] = noise_name
                        used_keys.add(key)
                        new_noise.setdefault(t, []).append(noise_name)
                        all_noise.add(noise_name)

            # Add noise nodes to tier_elements for next round
            for t, names in new_noise.items():
                tier_elements[t].extend(names)

        return all_noise

    def generate_instance(
        self,
        seed: int,
        min_num_tiers: int = 3,
        max_num_tiers: int = 4,
        min_base_elements: int = 10,
        max_base_elements: int = 15,
        tier_shrink_min: float = 0.4,
        tier_shrink_max: float = 0.6,
        min_recipes_per_element: int = 2,
        max_recipes_per_element: int = 3,
        max_tier_gap: int = 2,
        min_cross_tier_prob: float = 0.2,
        max_cross_tier_prob: float = 0.35,
        min_noise_node_ratio: float = 0.15,
        max_noise_node_ratio: float = 0.25,
        noise_chain_depth: int = 3,
        material_mode: str = "unlimited",
        min_material_mult: float = 1.5,
        max_material_mult: float = 2.0,
    ) -> GameInstance:
        """Generate a game instance for per-task scope."""
        rng = random.Random(seed)

        num_tiers = rng.randint(min_num_tiers, max_num_tiers)
        base_elements = rng.randint(min_base_elements, max_base_elements)
        tier_shrink = rng.uniform(tier_shrink_min, tier_shrink_max)
        cross_tier_prob = rng.uniform(min_cross_tier_prob, max_cross_tier_prob)
        noise_node_ratio = rng.uniform(min_noise_node_ratio, max_noise_node_ratio)

        tier_elements, recipes, local_tiers, noise_elements = self.generate_graph(
            rng, num_tiers, base_elements, tier_shrink,
            min_recipes_per_element, max_recipes_per_element,
            max_tier_gap, cross_tier_prob, noise_node_ratio, noise_chain_depth,
        )

        # Target from highest tier (noise excluded)
        valid_targets = [e for e in tier_elements[num_tiers - 1] if e not in noise_elements]
        if not valid_targets:
            # Fallback: try lower tiers
            for t in range(num_tiers - 2, 0, -1):
                valid_targets = [e for e in tier_elements[t] if e not in noise_elements]
                if valid_targets:
                    break
        target = rng.choice(valid_targets)

        return self._build_instance(
            rng, target, tier_elements, recipes, local_tiers,
            material_mode, min_material_mult, max_material_mult,
        )

    def generate_instance_for_pack(
        self,
        pack_seed: int,
        task_seed: int,
        task_idx: int = 0,
        num_tasks: int = 8,
        min_num_tiers: int = 3,
        max_num_tiers: int = 4,
        min_base_elements: int = 10,
        max_base_elements: int = 15,
        tier_shrink_min: float = 0.4,
        tier_shrink_max: float = 0.6,
        min_recipes_per_element: int = 2,
        max_recipes_per_element: int = 3,
        max_tier_gap: int = 2,
        min_cross_tier_prob: float = 0.2,
        max_cross_tier_prob: float = 0.35,
        min_noise_node_ratio: float = 0.15,
        max_noise_node_ratio: float = 0.25,
        noise_chain_depth: int = 3,
        material_mode: str = "unlimited",
        min_material_mult: float = 1.5,
        max_material_mult: float = 2.0,
    ) -> GameInstance:
        """Generate a game instance for per-pack scope.

        All tasks in a pack share the same graph. Each task gets a different target
        selected via weighted sampling (higher tiers more likely, no duplicates).
        """
        pack_rng = random.Random(pack_seed)

        num_tiers = pack_rng.randint(min_num_tiers, max_num_tiers)
        base_elements = pack_rng.randint(min_base_elements, max_base_elements)
        tier_shrink = pack_rng.uniform(tier_shrink_min, tier_shrink_max)
        cross_tier_prob = pack_rng.uniform(min_cross_tier_prob, max_cross_tier_prob)
        noise_node_ratio = pack_rng.uniform(min_noise_node_ratio, max_noise_node_ratio)

        tier_elements, recipes, local_tiers, noise_elements = self.generate_graph(
            pack_rng, num_tiers, base_elements, tier_shrink,
            min_recipes_per_element, max_recipes_per_element,
            max_tier_gap, cross_tier_prob, noise_node_ratio, noise_chain_depth,
        )

        # Group valid (non-noise) elements by tier for target selection
        tier_groups: Dict[int, List[str]] = {}
        for t in range(1, num_tiers):
            candidates = [e for e in tier_elements[t] if e not in noise_elements]
            if candidates:
                tier_groups[t] = candidates

        # Tier-weighted target selection, shuffled across tasks
        ordered_targets = self._select_targets(pack_rng, tier_groups, num_tasks)

        target = ordered_targets[task_idx % len(ordered_targets)]

        task_rng = random.Random(task_seed)
        return self._build_instance(
            task_rng, target, tier_elements, recipes, local_tiers,
            material_mode, min_material_mult, max_material_mult,
        )

    def _select_targets(
        self,
        rng: random.Random,
        tier_groups: Dict[int, List[str]],
        num_tasks: int,
    ) -> List[str]:
        """Select targets via tier-weighted sampling without replacement.

        First selects a tier (weight = tier + 1), then picks a random element
        from that tier. Skips tiers that are exhausted.
        """
        # Deep copy to avoid mutating
        available = {t: list(elems) for t, elems in tier_groups.items()}
        ordered: List[str] = []

        for _ in range(num_tasks):
            # Filter exhausted tiers
            active = {t: elems for t, elems in available.items() if elems}
            if not active:
                break

            # Weighted tier selection
            tiers = list(active.keys())
            weights = [t + 1 for t in tiers]
            chosen_tier = rng.choices(tiers, weights=weights, k=1)[0]

            # Pick random element from chosen tier, remove it
            elem = rng.choice(active[chosen_tier])
            available[chosen_tier].remove(elem)
            ordered.append(elem)

        rng.shuffle(ordered)
        return ordered

    def _build_instance(
        self,
        rng: random.Random,
        target: str,
        tier_elements: Dict[int, List[str]],
        recipes: Dict[RecipeKey, str],
        local_tiers: Dict[str, int],
        material_mode: str,
        min_material_mult: float,
        max_material_mult: float,
    ) -> GameInstance:
        """Build a GameInstance for a given target."""
        all_elements = set()
        for elems in tier_elements.values():
            all_elements.update(elems)

        # Trace one solution path
        solution_path = self._trace_solution(target, recipes)

        # Starting inventory: all tier 0 elements, same quantity each
        if material_mode == "unlimited":
            starting_inventory = {e: 1 for e in tier_elements[0]}
        else:
            # Uniform count based on solution path needs
            base_counts = self._compute_base_counts(solution_path)
            max_needed = max(base_counts.values()) if base_counts else 1
            material_mult = rng.uniform(min_material_mult, max_material_mult)
            uniform_count = max(max_needed, int(max_needed * material_mult + 0.5))
            starting_inventory = {e: uniform_count for e in tier_elements[0]}

        target_tier = local_tiers.get(target, 0)

        return GameInstance(
            target=target,
            recipes=recipes,
            starting_inventory=starting_inventory,
            all_elements=all_elements,
            solution_path=solution_path,
            depth=target_tier,
            local_tiers=local_tiers,
        )

    @staticmethod
    def _trace_solution(
        target: str,
        recipes: Dict[RecipeKey, str],
    ) -> List[Tuple[str, str, str]]:
        """Trace one solution path from base elements to target."""
        reverse: Dict[str, List[RecipeKey]] = defaultdict(list)
        for key, result in recipes.items():
            reverse[result].append(key)

        path: List[Tuple[str, str, str]] = []
        visited: Set[str] = set()

        def _expand(elem: str):
            if elem in visited:
                return
            visited.add(elem)
            available = reverse.get(elem, [])
            if not available:
                return
            chosen = available[0]
            e1, e2 = chosen
            _expand(e1)
            _expand(e2)
            path.append((e1, e2, elem))

        _expand(target)
        return path

    @staticmethod
    def _compute_base_counts(
        solution_path: List[Tuple[str, str, str]],
    ) -> Dict[str, int]:
        """Compute minimum base material counts for the solution path."""
        counts: Dict[str, int] = defaultdict(int)
        available: Dict[str, int] = defaultdict(int)

        for e1, e2, result in solution_path:
            for elem in [e1, e2]:
                if available.get(elem, 0) > 0:
                    available[elem] -= 1
                else:
                    counts[elem] += 1
            available[result] = available.get(result, 0) + 1

        return dict(counts)
