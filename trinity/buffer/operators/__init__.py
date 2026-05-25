from trinity.buffer.operators.experience_operator import (
    ExperienceOperator,
    ExperienceOperatorV1,
    create_operators,
)
from trinity.utils.registry import Registry

EXPERIENCE_OPERATORS: Registry = Registry(
    "experience_operators",
    default_mapping={
        "reward_filter": "trinity.buffer.operators.filters.reward_filter.RewardFilter",
        "reward_std_filter": "trinity.buffer.operators.filters.reward_filter.RewardSTDFilter",
        "dapo_dynamic_sampling": "trinity.buffer.operators.filters.reward_filter.DAPODynamicSamplingFilter",
        "mask_response_truncated": "trinity.buffer.operators.filters.reward_filter.MaskResponseTruncatedOperator",
        "reward_shaping_mapper": "trinity.buffer.operators.mappers.reward_shaping_mapper.RewardShapingMapper",
        "pass_rate_calculator": "trinity.buffer.operators.mappers.pass_rate_calculator.PassRateCalculator",
        "data_juicer": "trinity.buffer.operators.data_juicer_operator.DataJuicerOperator",
        "invalid_reward_filter": "trinity.buffer.operators.filters.reward_filter.InvalidRewardFilter",
    },
)

__all__ = [
    "ExperienceOperator",
    "ExperienceOperatorV1",
    "create_operators",
    "EXPERIENCE_OPERATORS",
]
