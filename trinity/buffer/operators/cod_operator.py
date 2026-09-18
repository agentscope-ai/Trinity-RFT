from typing import List, Tuple

from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience


class OverwriteRewardWithReturns(ExperienceOperator):
    """Overwrite exp.reward with exp.info["returns_norm"], dedicated to CoD-PPO algorithm."""

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        result_exps = [exp for exp in exps]
        for exp in result_exps:
            exp.reward = exp.info["returns_norm"]
        return result_exps, {}
