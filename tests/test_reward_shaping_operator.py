"""
Tests for RewardShapingOperator
"""

import pytest
from unittest.mock import MagicMock

from trinity.plugins.reward_shaping_operator import (
    RewardShapingOperator,
    RewardShapingConfig,
)


class MockExperience:
    """Mock Experience object for testing."""
    def __init__(self, response="", reward=0.0):
        self.response = response
        self.reward = reward


@pytest.mark.asyncio
async def test_length_bonus_short():
    """Test penalty for short responses."""
    config = RewardShapingConfig(strategy="length", min_length=10)
    operator = RewardShapingOperator(config)
    
    exp = MockExperience(response="short", reward=1.0)
    result, metrics = await operator.process([exp])
    
    assert result[0].reward < 1.0  # Should have penalty
    assert metrics["shaped_count"] == 1


@pytest.mark.asyncio
async def test_length_bonus_good():
    """Test bonus for good length responses."""
    config = RewardShapingConfig(strategy="length", min_length=10, max_length=1000)
    operator = RewardShapingOperator(config)
    
    exp = MockExperience(response="This is a good length response for testing.", reward=1.0)
    result, metrics = await operator.process([exp])
    
    assert result[0].reward > 1.0  # Should have bonus
    assert metrics["shaped_count"] == 1


@pytest.mark.asyncio
async def test_format_bonus():
    """Test bonus for formatted responses."""
    config = RewardShapingConfig(strategy="format")
    operator = RewardShapingOperator(config)
    
    exp = MockExperience(response="# Header\n- Item 1\n- Item 2\n```python\nprint('hello')\n```", reward=1.0)
    result, metrics = await operator.process([exp])
    
    assert result[0].reward > 1.0  # Should have bonus
    assert metrics["shaped_count"] == 1


@pytest.mark.asyncio
async def test_empty_list():
    """Test with empty experience list."""
    operator = RewardShapingOperator()
    result, metrics = await operator.process([])
    
    assert result == []
    assert metrics["shaped_count"] == 0
