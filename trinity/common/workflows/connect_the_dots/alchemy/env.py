# -*- coding: utf-8 -*-
"""Gymnasium environment for the Alchemy crafting game."""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym

from trinity.common.workflows.connect_the_dots.alchemy.common import GameInstance, RecipeKey


class AlchemyEnv(gym.Env):
    """Alchemy crafting game environment.

    Supports two material modes:
    - "limited": elements are consumed on every attempt (success or failure).
    - "unlimited": elements are never consumed; new discoveries are added to inventory.

    Game ends when target is synthesized, round limit is reached, or (limited only)
    materials run out.
    """

    metadata = {"render_modes": ["text"]}

    def __init__(self, game_instance: GameInstance, max_rounds: int = 35,
                 material_mode: str = "limited", render_mode: str = "text"):
        super().__init__()
        self.game = game_instance
        self.max_rounds = max_rounds
        self.material_mode = material_mode
        self.render_mode = render_mode
        self.inventory: Dict[str, int] = {}
        self.discovered_recipes: List[Tuple[str, str, str]] = []
        self.failed_combinations: List[Tuple[str, str]] = []
        self.current_round: int = 0
        self._last_result: Optional[str] = None
        self._invalid_action: bool = False
        self._success: bool = False

    def reset(self, seed=None, options=None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed=seed)
        if self.material_mode == "unlimited":
            self.inventory = {k: 1 for k in self.game.starting_inventory}
        else:
            self.inventory = dict(self.game.starting_inventory)
        self.discovered_recipes = []
        self.failed_combinations = []
        self.current_round = 0
        self._last_result = None
        self._invalid_action = False
        self._success = False
        return self.render(), self._get_info()

    def step(self, action: Tuple[str, str]) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        raw1, raw2 = action
        self.current_round += 1
        self._last_result = None
        self._invalid_action = False

        # Normalize to inventory keys (case-insensitive)
        inv_lower = {k.lower(): k for k in self.inventory}
        elem1 = inv_lower.get(raw1.lower(), raw1)
        elem2 = inv_lower.get(raw2.lower(), raw2)

        # Validate elements exist in inventory
        if self.material_mode == "unlimited":
            # Unlimited: just check both elements exist (no quantity requirement)
            if self.inventory.get(elem1, 0) < 1 or self.inventory.get(elem2, 0) < 1:
                self._invalid_action = True
        else:
            if elem1 == elem2:
                if self.inventory.get(elem1, 0) < 2:
                    self._invalid_action = True
            else:
                if self.inventory.get(elem1, 0) < 1 or self.inventory.get(elem2, 0) < 1:
                    self._invalid_action = True

        if self._invalid_action:
            return self._finish_step()

        # Consume ingredients (limited mode only)
        if self.material_mode == "limited":
            if elem1 == elem2:
                self.inventory[elem1] -= 2
            else:
                self.inventory[elem1] -= 1
                self.inventory[elem2] -= 1
            self.inventory = {k: v for k, v in self.inventory.items() if v > 0}

        # Lookup recipe
        key: RecipeKey = tuple(sorted([elem1.lower(), elem2.lower()]))
        result = self.game.recipes.get(key)

        if result is not None:
            self._last_result = result
            if self.material_mode == "unlimited":
                self.inventory[result] = 1
            else:
                self.inventory[result] = self.inventory.get(result, 0) + 1
            self.discovered_recipes.append((elem1, elem2, result))
        else:
            self.failed_combinations.append((elem1, elem2))

        return self._finish_step()

    def _finish_step(self) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self._success = self.game.target in self.inventory
        terminated = self._success
        truncated = (
            not terminated
            and (self.current_round >= self.max_rounds or not self.get_feasible_actions())
        )
        reward = 1.0 if self._success else 0.0
        return self.render(), reward, terminated, truncated, self._get_info()

    def get_feasible_actions(self) -> List[Tuple[str, str]]:
        elems = sorted(self.inventory.keys())
        pairs = []
        for i, e1 in enumerate(elems):
            for e2 in elems[i:]:
                if self.material_mode == "unlimited":
                    # Unlimited: any pair is feasible as long as elements exist
                    pairs.append((e1, e2))
                else:
                    if e1 == e2:
                        if self.inventory[e1] >= 2:
                            pairs.append((e1, e2))
                    else:
                        pairs.append((e1, e2))
        return pairs

    def render(self) -> str:
        inv_str = ", ".join(f"{e} x{c}" for e, c in sorted(self.inventory.items()))
        lines = [
            f"Target: {self.game.target}",
            f"Round: {self.current_round}/{self.max_rounds}",
            f"Inventory: {inv_str}" if inv_str else "Inventory: (empty)",
        ]
        if self.discovered_recipes:
            lines.append("Discovered recipes:")
            for e1, e2, r in self.discovered_recipes:
                lines.append(f"  {e1} + {e2} = {r}")
        if self.failed_combinations:
            lines.append("Failed combinations:")
            for e1, e2 in self.failed_combinations:
                lines.append(f"  {e1} + {e2}")
        return "\n".join(lines)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "inventory": dict(self.inventory),
            "discovered_recipes": list(self.discovered_recipes),
            "failed_combinations": list(self.failed_combinations),
            "feasible_actions_count": len(self.get_feasible_actions()),
            "target": self.game.target,
            "current_round": self.current_round,
            "success": self._success,
            "last_result": self._last_result,
            "invalid_action": self._invalid_action,
        }
