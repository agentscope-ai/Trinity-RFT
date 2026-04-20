"""OpenJudge reward function classes."""

import asyncio
import os
from typing import Any, Dict, List, Optional

from trinity.common.experience import Experience
from trinity.common.rewards.reward_fn import RewardFn


class OpenJudgeRewardFn(RewardFn):
    """Reward Function using OpenJudge multi-grader pipeline.

    Args:
        grader_configs: Dict mapping grader name to a GraderConfig, a
            BaseGrader instance, or a (BaseGrader, mapper) tuple.
            When *None* a default pair (CorrectnessGrader + RelevanceGrader)
            is used so the class is usable out of the box.
        model_name: Default judge model for any grader that needs one.
        max_concurrency: Passed to GradingRunner.
        score_aggregation: How to combine per-grader scores into the final
            ``reward`` key.  ``"mean"`` (default) or ``"sum"``.
        judge_api_base_url_env: Env-var holding the judge API base URL.
        judge_api_key_env: Env-var holding the judge API key.
    """

    def __init__(
        self,
        grader_configs: Optional[Dict[str, Any]] = None,
        model_name: str = "qwen3-32b",
        max_concurrency: int = 8,
        score_aggregation: str = "mean",
        judge_api_base_url_env: str = "OPENAI_BASE_URL",
        judge_api_key_env: str = "OPENAI_API_KEY",
        **kwargs,
    ):
        try:
            from openjudge.models.openai_chat_model import (  # pyright: ignore[reportMissingImports]
                OpenAIChatModel,
            )
            from openjudge.runner.grading_runner import (  # pyright: ignore[reportMissingImports]
                GradingRunner,
            )
        except ImportError as e:
            raise ImportError(
                "OpenJudge dependencies are not installed. "
                "Please install with `pip install -e .[openjudge]`."
            ) from e

        self.score_aggregation = score_aggregation

        if grader_configs is None:
            from openjudge.graders.common.correctness import (  # pyright: ignore[reportMissingImports]
                CorrectnessGrader,
            )
            from openjudge.graders.common.relevance import (  # pyright: ignore[reportMissingImports]
                RelevanceGrader,
            )

            judge_base_url = os.getenv(judge_api_base_url_env, "")
            if not judge_base_url:
                raise ValueError(f"Judge base URL is missing. Set env `{judge_api_base_url_env}`.")
            model_kwargs: Dict[str, Any] = {
                "model": model_name,
                "base_url": judge_base_url,
                "api_key": os.getenv(judge_api_key_env, ""),
            }
            model = OpenAIChatModel(**model_kwargs)
            grader_configs = {
                "correctness": CorrectnessGrader(model=model),
                "relevance": RelevanceGrader(model=model),
            }

        self.runner = GradingRunner(
            grader_configs=grader_configs,
            max_concurrency=max_concurrency,
            show_progress=False,
        )

    def __call__(  # type: ignore[override]
        self,
        experience: Any,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, float]:
        """Evaluate a single experience and return a reward dict."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acall(experience, messages, **kwargs))

        raise RuntimeError(
            "OpenJudgeRewardFn.__call__ cannot be used inside a running event loop. "
            "Use `await reward_fn.acall(...)` in async workflows."
        )

    async def acall(  # type: ignore[override]
        self,
        experience: Any,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, float]:
        """Async evaluation for event-loop contexts."""
        merged_messages = list(messages)
        if not merged_messages or merged_messages[-1].get("role") != "assistant":
            merged_messages.append(
                {
                    "role": "assistant",
                    "content": str(getattr(experience, "response_text", "") or ""),
                }
            )

        data = {"messages": merged_messages}
        batch_results = await self.runner.arun([data])
        return self._extract_reward(batch_results)

    def _extract_reward(self, batch_results: Dict[str, Any]) -> Dict[str, float]:
        from openjudge.graders.schema import (  # pyright: ignore[reportMissingImports]
            GraderError,
            GraderScore,
        )

        reward_dict: Dict[str, float] = {}
        scores: List[float] = []

        for grader_name, grader_results in batch_results.items():
            if not grader_results:
                continue
            result = grader_results[0]
            if isinstance(result, GraderScore):
                reward_dict[f"{grader_name}_score"] = result.score
                scores.append(result.score)
            elif isinstance(result, GraderError):
                reward_dict[f"{grader_name}_score"] = 0.0
                scores.append(0.0)

        if scores:
            reward_dict["reward"] = (
                sum(scores) / len(scores) if self.score_aggregation == "mean" else sum(scores)
            )
        else:
            reward_dict["reward"] = 0.0

        return reward_dict


class TrajectoryAccuracyGrader(OpenJudgeRewardFn):
    """Single-grader reward using OpenJudge TrajectoryAccuracyGrader.

    Args:
        reward_name: Logical name for this reward (used for logging/registry).
        model_name: Judge model passed to OpenAIChatModel.
        normalize_score: When True, linearly maps the raw score to [0, 1]
            using score_min / score_max.
        score_min: Lower bound for normalisation (default 1.0).
        score_max: Upper bound for normalisation (default 3.0).
        judge_api_base_url_env: Env-var holding the judge API base URL.
        judge_api_key_env: Env-var holding the judge API key.
    """

    def __init__(
        self,
        reward_name: str = "openjudge_trajectory_accuracy_reward",
        model_name: str = "qwen3-max",
        normalize_score: bool = True,
        score_min: float = 1.0,
        score_max: float = 3.0,
        judge_api_base_url_env: str = "OPENAI_BASE_URL",
        judge_api_key_env: str = "OPENAI_API_KEY",
        **kwargs,
    ):
        try:
            from openjudge.graders.agent.trajectory.trajectory_accuracy import (
                TrajectoryAccuracyGrader as _TrajectoryAccuracyGrader,  # pyright: ignore[reportMissingImports]
            )
            from openjudge.models.openai_chat_model import (  # pyright: ignore[reportMissingImports]
                OpenAIChatModel,
            )
        except ImportError as e:
            raise ImportError(
                "OpenJudge dependencies are not installed. "
                "Please install with `pip install -e .[openjudge]`."
            ) from e

        judge_base_url = os.getenv(judge_api_base_url_env, "")
        if not judge_base_url:
            raise ValueError(f"Judge base URL is missing. Set env `{judge_api_base_url_env}`.")
        judge_model = OpenAIChatModel(
            model=kwargs.get("judge_model_name", model_name),
            base_url=judge_base_url,
            api_key=os.getenv(judge_api_key_env, ""),
            temperature=kwargs.get("temperature", 0.0),
        )

        super().__init__(
            grader_configs={"trajectory": _TrajectoryAccuracyGrader(model=judge_model)},
            max_concurrency=kwargs.get("max_concurrency", 8),
            judge_api_base_url_env=judge_api_base_url_env,
            judge_api_key_env=judge_api_key_env,
        )

        self.reward_name = reward_name
        self.normalize_score = normalize_score
        self.score_min = float(score_min)
        self.score_max = float(score_max)

    def __call__(  # type: ignore[override]
        self,
        experience: Experience,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, float]:
        return super().__call__(experience, messages, **kwargs)

    def _extract_reward(self, batch_results: Dict[str, Any]) -> Dict[str, float]:
        reward_dict = super()._extract_reward(batch_results)

        if not self.normalize_score or self.score_max <= self.score_min:
            return reward_dict

        raw = reward_dict.get("reward", 0.0)
        normalized = (raw - self.score_min) / (self.score_max - self.score_min)
        reward_dict["reward"] = max(0.0, min(1.0, normalized))
        return reward_dict
