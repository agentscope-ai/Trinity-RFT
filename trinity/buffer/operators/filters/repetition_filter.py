"""Filter out experiences whose generated response is highly repetitive."""

from typing import Dict, List, Sequence, Tuple

from trinity.buffer.operators import ExperienceOperatorV1
from trinity.common.experience import Experience


def compute_repeat_ratio(tokens: Sequence[int], ngram_size: int) -> float:
    """Compute the duplicate n-gram ratio of a token sequence.

    The ratio is defined as ``1 - unique_ngrams / total_ngrams`` over all sliding
    windows of ``ngram_size`` tokens. It is ``0.0`` when every n-gram is distinct and
    approaches ``1.0`` when the sequence keeps repeating the same short pattern.

    Args:
        tokens: Token ids of the generated response.
        ngram_size: Size of the sliding window.

    Returns:
        float: Duplicate n-gram ratio in ``[0, 1]``. Sequences shorter than
        ``ngram_size`` have no n-grams and score ``0.0``.
    """
    total = len(tokens) - ngram_size + 1
    if total <= 0:
        return 0.0
    unique = len({tuple(tokens[i : i + ngram_size]) for i in range(total)})
    return 1.0 - unique / total


class RepetitionFilter(ExperienceOperatorV1):
    """
    Drop experiences whose response has collapsed into repetition.

    Repetition collapse is a common failure mode of on-policy rollouts: the policy keeps
    emitting the same pattern instead of terminating. Such trajectories carry almost no
    learning signal, so this operator removes them before they enter the training buffer.

    Only the model-generated part of the experience is inspected: tokens before
    ``prompt_length`` are ignored, and when ``action_mask`` is available (multi-turn
    experiences) tokens with a zero mask (e.g. environment observations) are ignored too.
    An experience is dropped when ``duplicate_ngrams / total_ngrams > max_repeat_ratio``.
    Empty responses and responses shorter than ``min_response_tokens`` are always kept,
    since the ratio is not meaningful on very short sequences.
    """

    def __init__(
        self,
        ngram_size: int = 4,
        max_repeat_ratio: float = 0.5,
        min_response_tokens: int = 32,
    ) -> None:
        """Initialize the repetition filter.

        Args:
            ngram_size: Sliding window size (in tokens) used to measure repetition.
            max_repeat_ratio: Experiences whose duplicate n-gram ratio is strictly greater
                than this value are dropped. ``0.0`` drops any response containing a
                repeated n-gram, ``1.0`` disables filtering.
            min_response_tokens: Responses with fewer generated tokens than this are kept
                without being scored.
        """
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be >= 1, got {ngram_size}")
        if not 0.0 <= max_repeat_ratio <= 1.0:
            raise ValueError(f"max_repeat_ratio must be in [0, 1], got {max_repeat_ratio}")
        if min_response_tokens < 0:
            raise ValueError(f"min_response_tokens must be >= 0, got {min_response_tokens}")
        self.ngram_size = ngram_size
        self.max_repeat_ratio = max_repeat_ratio
        self.min_response_tokens = min_response_tokens

    @staticmethod
    def _response_tokens(exp: Experience) -> List[int]:
        """Extract the model-generated token ids of an experience."""
        if exp.tokens is None:
            return []
        response = exp.tokens[exp.prompt_length :]
        if exp.action_mask is not None:
            mask = exp.action_mask.bool()
            length = min(len(response), len(mask))
            response = response[:length][mask[:length]]
        return response.tolist()

    async def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        """Drop experiences whose response exceeds the repetition threshold.

        Args:
            exps: Experiences to process.

        Returns:
            Tuple[List[Experience], Dict]: Kept experiences and filtering metrics.
        """
        kept = []
        ratios = []
        skipped_empty = 0
        skipped_short = 0
        for exp in exps:
            tokens = self._response_tokens(exp)
            if len(tokens) == 0:
                skipped_empty += 1
                kept.append(exp)
                continue
            if len(tokens) < self.min_response_tokens:
                skipped_short += 1
                kept.append(exp)
                continue
            ratio = compute_repeat_ratio(tokens, self.ngram_size)
            ratios.append(ratio)
            if ratio > self.max_repeat_ratio:
                continue
            kept.append(exp)
        metrics: Dict[str, float] = {
            "filtered_count": len(exps) - len(kept),
            "skipped_empty_count": skipped_empty,
            "skipped_short_count": skipped_short,
        }
        if ratios:
            metrics["repeat_ratio/mean"] = sum(ratios) / len(ratios)
            metrics["repeat_ratio/max"] = max(ratios)
        return kept, metrics
