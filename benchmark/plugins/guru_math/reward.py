from typing import Optional

from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.rewards.math_reward import MathBoxedRewardFn


@REWARD_FUNCTIONS.register_module("math_boxed_reward_naive_dapo")
class NaiveDapoRewardFn(MathBoxedRewardFn):
    def __call__(  # type: ignore
        self,
        response: str,
        truth: Optional[str] = None,
        with_think: Optional[bool] = False,
        format_score_coef: Optional[float] = 0.1,
        **kwargs,
    ) -> dict[str, float]:
        from trinity.common.rewards.naive_dapo_score import compute_score

        score, _ = compute_score(response, truth)  # type: ignore
        return {"accuracy": score, "format_score": 0}


@REWARD_FUNCTIONS.register_module("math_boxed_reward_prime_math")
class PrimeMathRewardFn(MathBoxedRewardFn):
    def __call__(  # type: ignore
        self,
        response: str,
        truth: Optional[str] = None,
        with_think: Optional[bool] = False,
        format_score_coef: Optional[float] = 0.1,
        **kwargs,
    ) -> dict[str, float]:
        from verl.utils.reward_score.prime_math import compute_score

        res = compute_score(response, truth)
        return {"accuracy": res["score"], "format_score": 0}
