"""Data selectors."""
from typing import Dict, List

import numpy as np
import torch

from trinity.buffer.reader.file_reader import _HFBatchReader
from trinity.buffer.selector.difficulty_estimator import InterpolationBetaPREstimator
from trinity.common.config import DataSelectorConfig
from trinity.utils.annotations import Experimental
from trinity.utils.log import get_logger


@Experimental
class BaseSelector:
    """
    Abstract base class defining the interface for custom data selection strategies.

    A selector determines which samples (by index) are selected from the dataset
    during training. It enables flexible sampling beyond simple
    sequential or random access, supporting active learning, curriculum learning,
    or difficulty-based sampling in the future.

    Subclasses must implement:
        - get_indices: returns list of indices for next batch
        - update: updates internal state using feedback (e.g., loss values, mean rewards, etc.)
        - state_dict / load_state_dict: for saving/loading selector state (checkpointing)
    """

    def __init__(self, data_source: _HFBatchReader, config: DataSelectorConfig):
        self.data_source = data_source
        self.config = config

    def get_indices(self, batch_size: int, return_extra_info: bool = False) -> List[int]:
        """
        Select a batch of sample indices from the dataset.

        Args:
            batch_size (int): Number of indices to return
            return_extra_info (bool): If True, may return additional metadata (future use)

        Returns:
            List[int]: Selected indices into the dataset
        """
        raise NotImplementedError

    def feedback(self, indices: List[int], values: List[float]) -> None:
        """
        Update internal state based on feedback (e.g., model loss, accuracy).

        This allows adaptive selectors (like hard example mining) to learn over time.

        Args:
            indices (List[int]): Previously selected indices
            values (List[float]): Feedback values corresponding to those indices
        """
        raise NotImplementedError

    def state_dict(self) -> Dict:
        """
        Return serializable state of the selector for checkpointing.

        Returns:
            Dict: State information (e.g., current position, etc.)
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Restore selector state from a saved dictionary.

        Args:
            state_dict (Dict): Output from state_dict()
        """
        raise NotImplementedError


class SequentialSelector(BaseSelector):
    """
    Selects data sequentially in fixed order across epochs.

    Example: [0,1,2,...,B-1], then [B,B+1,...,2B-1], etc.
    """

    def __init__(self, data_source: _HFBatchReader, config: DataSelectorConfig):
        super().__init__(data_source, config)
        self.dataset_size = data_source.dataset_size
        self.current_index = 0

    def get_indices(self, batch_size: int, return_extra_info: bool = False) -> List[int]:
        start = self.current_index % self.dataset_size
        end = start + batch_size
        self.current_index += batch_size
        if end <= self.dataset_size:
            return list(range(start, end))
        return list(range(start, self.dataset_size)) + list(range(0, end - self.dataset_size))

    def feedback(self, indices: List[int], values: List[float]) -> None:
        # No-op: sequential selection doesn't adapt based on feedback
        pass

    def state_dict(self) -> Dict:
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        self.current_index = state_dict.get("current_index", 0)


class ShuffleSelector(BaseSelector):
    """
    Shuffles dataset once per epoch and iterates through it sequentially.

    Each epoch uses a different permutation of a subset of the full dataset.
    When one epoch ends, a new shuffle is triggered.
    Mimics standard PyTorch DataLoader with shuffle=True.
    """

    def __init__(self, data_source: _HFBatchReader, config: DataSelectorConfig):
        super().__init__(data_source, config)
        self.dataset_size = data_source.dataset_size  # Total available samples
        self.current_index = 0  # Progress tracker
        self.seed = config.seed  # For reproducible shuffling
        self.orders = self._get_orders()  # Current shuffled index order

    def _get_orders(self) -> List[int]:
        """
        Generate a new shuffled order for the current epoch.

        Uses NumPy's PCG64 random generator seeded by epoch number for reproducibility.
        Ensures different shuffle per epoch while being deterministic if seed is fixed.
        """
        rng = np.random.default_rng(self.seed + self.current_index // self.dataset_size)
        return rng.permutation(self.dataset_size).tolist()

    def get_indices(self, batch_size: int, return_extra_info: bool = False) -> List[int]:
        start = self.current_index % self.dataset_size
        end = start + batch_size
        if end <= self.dataset_size:
            ret = self.orders[start:end]
            # At end of epoch, reshuffle for next epoch
            if end == self.dataset_size:
                self.orders = self._get_orders()
        else:
            ret = self.orders[start:]
            # At end of epoch, reshuffle for next epoch
            self.orders = self._get_orders()
            ret += self.orders[: (end - self.dataset_size)]
        self.current_index += batch_size
        return ret

    def feedback(self, indices: List[int], values: List[float]) -> None:
        # No-op: static shuffling does not adapt
        pass

    def state_dict(self) -> Dict:
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        self.current_index = state_dict.get("current_index", 0)
        self.orders = self._get_orders()


class RandomSelector(BaseSelector):
    """
    Uniformly samples batches randomly with replacement *per batch*.

    Unlike ShuffleSelector, there is no concept of an epoch — every batch is independently sampled.
    Can result in repeated samples within an epoch. Suitable for online or stochastic training regimes.
    """

    def __init__(self, data_source: _HFBatchReader, config: DataSelectorConfig):
        super().__init__(data_source, config)
        self.dataset_size = data_source.dataset_size
        self.current_index = 0
        self.seed = config.seed

    def get_indices(self, batch_size, return_extra_info=False):
        # Seed varies per batch to ensure repeatability across runs
        rng = np.random.default_rng(self.seed + self.current_index)
        selected_indices = rng.choice(self.dataset_size, batch_size, replace=False)
        self.current_index += batch_size
        if return_extra_info:
            return selected_indices, {}
        else:
            return selected_indices

    def feedback(self, indices: List[int], values: List[float]) -> None:
        # No-op: basic random selection doesn't adapt
        pass

    def state_dict(self) -> Dict:
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        self.current_index = state_dict.get("current_index", 0)


class OfflineEasy2HardSelector(BaseSelector):
    """
    Selects samples in an 'easy-to-hard' curriculum based on pre-defined difficulty features.

    This selector assumes that higher feature values indicate easier examples.
    It sorts all data once at initialization by descending feature value(s), then sequentially
    serves batches from easy → hard over epochs. The sorting is fixed (offline), so no online
    adaptation occurs during training.

    Useful for curriculum learning where sample difficulty is estimated ahead of time
    (e.g., via teacher model confidence, length, BLEU score, etc.).
    """

    def __init__(self, data_source, config: DataSelectorConfig):
        super().__init__(data_source, config)
        self.logger = get_logger("offline_easy2hard_selector")

        # Extract specified feature columns (e.g., 'loss', 'confidence') used to estimate difficulty
        feature_keys = config.feature_keys
        self.features = np.concatenate(
            [np.array(list(data_source.dataset[k]))[:, None] for k in feature_keys], axis=1
        )
        # Shape: (N, len(feature_keys)) — one row per sample, one column per feature

        # Append index to each feature vector for tracking original positions after sorting
        features_with_index = [list(self.features[i]) + [i] for i in range(len(self.features))]

        # Sort by feature values in descending order → highest (easiest) first
        features_with_index = sorted(features_with_index)[::-1]
        self.logger.debug(f"OfflineEasy2HardSelector, sorted {features_with_index[:20]}")

        # Store the sorted order of indices (from easiest to hardest)
        self.sorted_index = np.array([i[-1] for i in features_with_index])

        # Number of samples per epoch (may be less than full dataset size)
        self.dataset_size = data_source.dataset_size
        self.current_index = 0

    def feedback(self, indices: List[int], values: List[float]) -> None:
        # No-op: this selector does not adapt based on runtime feedback
        pass

    def get_indices(self, batch_size, return_extra_info=False):
        """
        Returns next batch of indices in curriculum order (easy → hard).

        Batches are taken sequentially from the pre-sorted list. When epoch ends,
        it wraps around to the beginning (i.e., restarts curriculum).
        """
        start = self.current_index % self.dataset_size
        end = start + batch_size
        if end <= self.dataset_size:
            selected_indices = self.sorted_index[start:end]
        else:
            selected_indices = np.concatenate(
                [self.sorted_index[start:], self.sorted_index[: (end - self.dataset_size)]]
            )
        self.current_index += batch_size
        if not return_extra_info:
            return selected_indices
        else:
            extra_info = {
                "indices": selected_indices.tolist(),
                "feat1": self.features[selected_indices, 0].tolist(),
                "feat2": self.features[selected_indices, 1].tolist(),
            }
            return selected_indices, extra_info

    def state_dict(self) -> Dict:
        """
        Save current position in the curriculum for checkpointing.
        Allows resuming from same point in the easy→hard progression.
        """
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        """
        Restore progress through the curriculum from saved state.
        """
        self.current_index = state_dict.get("current_index", 0)


class DifficultyBasedSelector(BaseSelector):
    """
    Adaptive difficulty-based selector using probabilistic modeling of sample difficulty.

    Uses `InterpolationBetaPREstimator` to model each sample's probability of success (PR),
    updated with observed feedback (e.g., loss, accuracy). Then selects samples close to
    a target reward (e.g., 1.0 for perfect performance), implementing a form of
    *targeted difficulty sampling* — focusing on items near the edge of model capability.

    Supports both greedy selection (`tau=0`) and stochastic sampling (`tau>0`).
    """

    def __init__(self, data_source, config: DataSelectorConfig) -> None:
        super().__init__(data_source, config)
        self.logger = get_logger("difficulty_based_selector")

        # Initialize difficulty estimator using two features (assumed: e.g., correctness & uncertainty)
        self.diff_estimator = self.build_diff_estimator(
            data_source.dataset, config.feature_keys, config.kwargs
        )
        self.current_index = 0
        self.seed = config.seed

        self.do_sample = config.kwargs.get(
            "do_sample", False
        )  # Whether to sample PR during estimation
        self.target_reward = config.kwargs.get("target_reward", 1.0)  # Desired performance level
        self.tau = config.kwargs.get("tau", 1.0)  # Temperature for sampling distribution

    def build_diff_estimator(self, dataset, feature_keys: List[str], config: dict):
        """
        Constructs a Beta-distribution-based difficulty estimator from features.

        Expects exactly two feature keys (e.g., ['correct', 'uncertainty']), which are concatenated
        into a feature matrix and passed to InterpolationBetaPREstimator for modeling P(success).
        """
        self.logger.debug(f"{config=}")
        if len(feature_keys) != 2:
            raise ValueError(
                f"DifficultyBasedSelector requires exactly 2 feature keys, but got {len(feature_keys)}."
            )
        features = np.concatenate(
            [np.array(list(dataset[k]))[:, None] for k in feature_keys], axis=1
        )
        self.logger.debug(f"{features.shape=}")
        self.logger.debug(f"{features[:5]=}")
        adaptive_rho = config.get("adaptive_rho", False)
        return InterpolationBetaPREstimator(
            features=features,
            m=config.get("m", 16),
            lamb=config.get("lamb", 0.2),
            rho=config.get("rho", 0.2),
            adaptive_rho=adaptive_rho,
        )

    def feedback(self, indices: List[int], values: List[float]) -> None:
        """
        Updates the difficulty estimator with observed performance on selected samples.

        Args:
            indices (List[int]): Previously selected sample indices
            values (List[float]): Observed rewards/scores (e.g., accuracy, BLEU) for those samples
        """
        self.diff_estimator.update(indices, values)

    def get_scores(self) -> List[float]:
        """
        Computes selection scores: negative distance between predicted PR and target reward.

        Samples whose predicted performance is closest to `target_reward` receive highest scores.
        Encourages selection of "just right" difficulty samples (neither too easy nor too hard).
        """
        rng = np.random.default_rng(self.seed + self.current_index)
        predicted_pr = self.diff_estimator.predict_pr(rng=rng, do_sample=self.do_sample)
        scores = -np.abs(self.target_reward - predicted_pr)
        return scores

    def get_indices(self, batch_size, return_extra_info=False):
        """
        Selects batch of indices based on difficulty proximity to target.

        If tau == 0: take top-k highest scoring samples (greedy).
        Else: sample stochastically using softmax(logits / tau).
        """
        sampling_scores = self.get_scores()
        sampling_scores = torch.from_numpy(sampling_scores)
        if self.tau == 0:
            selected_indices = torch.topk(sampling_scores, batch_size).indices
        else:
            sampling_logits = sampling_scores / self.tau
            sampling_logits -= sampling_logits.max()
            sampling_probabilities = torch.softmax(sampling_logits, dim=0)
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.current_index)
            selected_indices = torch.multinomial(
                sampling_probabilities,
                batch_size,
                replacement=False,
                generator=rng,
            )
        self.logger.debug(f"{selected_indices=}")
        self.logger.debug(f"{sampling_scores=}")
        self.logger.debug(f"{sampling_scores[selected_indices]=}")
        self.current_index += batch_size

        if return_extra_info:
            selected_indices_list = selected_indices.tolist()
            alphas = self.diff_estimator.alphas[selected_indices_list]
            betas = self.diff_estimator.betas[selected_indices_list]
            point_est = alphas / (alphas + betas)
            extra_info = {
                "indices": selected_indices_list,
                "scores": sampling_scores[selected_indices].tolist(),
                "alphas": alphas.tolist(),
                "betas": betas.tolist(),
                "point": point_est.tolist(),
            }
            return selected_indices, extra_info
        else:
            return selected_indices

    def state_dict(self) -> Dict:
        """
        Save current state for checkpointing.
        Only tracks sampling progress; actual difficulty estimates are in diff_estimator.
        """
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        """
        Restore selector state from checkpoint.
        """
        self.current_index = state_dict.get("current_index", 0)
