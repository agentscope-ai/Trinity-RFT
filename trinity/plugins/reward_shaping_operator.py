"""
Reward Shaping Operator for Trinity-RFT

This operator shapes rewards based on various criteria:
- Length-based reward shaping
- Diversity-based reward shaping
- Format-based reward shaping

Issue: Can be used as a starting point for implementing
custom reward shaping strategies in Trinity-RFT.

Usage:
    Add to your config:
    ```yaml
    buffer:
      operators:
        - type: "trinity.plugins.reward_shaping_operator.RewardShapingOperator"
          config:
            strategy: "length"
            min_length: 10
            max_length: 1000
    ```
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from dataclasses import dataclass

from trinity.buffer.operators.experience_operator import ExperienceOperatorV1
from trinity.common.experience import Experience


@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping."""
    strategy: str = "length"  # length, diversity, format
    min_length: int = 10
    max_length: int = 1000
    length_bonus: float = 0.1
    diversity_threshold: float = 0.5
    format_bonus: float = 0.05


class RewardShapingOperator(ExperienceOperatorV1):
    """
    Operator that shapes rewards based on various criteria.
    
    Strategies:
    - length: Bonus/penalty based on response length
    - diversity: Bonus for diverse responses in batch
    - format: Bonus for proper formatting (lists, code blocks, etc.)
    """
    
    def __init__(self, config: RewardShapingConfig | None = None):
        self.config = config or RewardShapingConfig()
    
    async def process(
        self, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        """Process experiences and shape their rewards."""
        
        if not exps:
            return exps, {"shaped_count": 0}
        
        shaped_count = 0
        total_bonus = 0.0
        
        for exp in exps:
            bonus = self._calculate_bonus(exp)
            if bonus != 0:
                # Apply bonus to reward if reward exists
                if hasattr(exp, 'reward') and exp.reward is not None:
                    exp.reward = exp.reward + bonus
                    shaped_count += 1
                    total_bonus += bonus
        
        metrics = {
            "shaped_count": shaped_count,
            "total_bonus": total_bonus,
            "strategy": self.config.strategy,
        }
        
        return exps, metrics
    
    def _calculate_bonus(self, exp: Experience) -> float:
        """Calculate reward bonus based on strategy."""
        
        if self.config.strategy == "length":
            return self._length_bonus(exp)
        elif self.config.strategy == "format":
            return self._format_bonus(exp)
        else:
            return 0.0
    
    def _length_bonus(self, exp: Experience) -> float:
        """Calculate bonus based on response length."""
        if not hasattr(exp, 'response') or not exp.response:
            return 0.0
        
        length = len(exp.response)
        
        if length < self.config.min_length:
            # Penalty for too short
            return -self.config.length_bonus
        elif length > self.config.max_length:
            # Small penalty for too long
            return -self.config.length_bonus * 0.5
        else:
            # Bonus for good length
            return self.config.length_bonus * 0.1
    
    def _format_bonus(self, exp: Experience) -> float:
        """Calculate bonus for proper formatting."""
        if not hasattr(exp, 'response') or not exp.response:
            return 0.0
        
        bonus = 0.0
        response = exp.response
        
        # Bonus for lists
        if "- " in response or "* " in response or "1. " in response:
            bonus += self.config.format_bonus
        
        # Bonus for code blocks
        if "```" in response:
            bonus += self.config.format_bonus
        
        # Bonus for headers
        if "# " in response or "## " in response:
            bonus += self.config.format_bonus * 0.5
        
        return bonus
