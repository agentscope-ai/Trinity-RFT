# -*- coding: utf-8 -*-
"""Algorithm classes."""

from abc import ABC, ABCMeta, abstractmethod
from typing import Dict

from trinity.common.config import Config
from trinity.common.constants import SyncMethod
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class ConstantMeta(ABCMeta):
    def __setattr__(cls, name, value):
        if name in cls.__dict__:
            raise AttributeError(f"{name} is already defined in {cls.__name__}")
        return super().__setattr__(name, value)


class AlgorithmType(ABC, metaclass=ConstantMeta):
    use_critic: bool  # whether to use critic model

    use_reference: bool  # whether to use reference model

    compute_advantage_in_trainer: bool  # whether to compute advantage in trainer
    # For algorithms that rely on experience grouping,
    # we recommend set this value to False

    can_balance_batch: bool  # balance batch in trainer

    schema: str  # schema of training data

    @classmethod
    @abstractmethod
    def default_config(cls) -> Dict:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls._name

    @classmethod
    def check_config(cls, config: Config) -> None:
        pass


class SFTAlgorithm(AlgorithmType):
    """SFT Algorithm."""

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "sft"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "sample_strategy": "default",
            "policy_loss_fn": "sft",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }

    @classmethod
    def check_config(cls, config: Config) -> None:
        if config.mode == "train":
            if (
                config.buffer.trainer_input.experience_buffer is None
                or not config.buffer.trainer_input.experience_buffer.path
            ):
                raise ValueError(
                    "`buffer.trainer_input.experience_buffer.path` is required when `algorithm.algorithm_type == sft`"
                )
        elif config.mode in ["both", "explore"]:
            raise ValueError(f"SFT does not support `{config.mode}` mode")

        if config.synchronizer.sync_method != SyncMethod.CHECKPOINT:
            config.synchronizer.sync_method = SyncMethod.CHECKPOINT
            logger.warning(
                "SFT only supports checkpoint synchronization, set `synchronizer.sync_method` to `checkpoint`."
            )

        config.synchronizer.sync_interval = config.trainer.save_interval


class PPOAlgorithm(AlgorithmType):
    """PPO Algorithm."""

    use_critic: bool = True
    use_reference: bool = True
    compute_advantage_in_trainer: bool = True
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 1,
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "advantage_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class GRPOAlgorithm(AlgorithmType):
    """GRPO algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "grpo",
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class ReinforcePlusPlusAlgorithm(AlgorithmType):
    """Reinforce++ algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = True
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "reinforceplusplus",
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class RLOOAlgorithm(AlgorithmType):
    """RLOO algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = True
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "rloo",
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class OPMDAlgorithm(AlgorithmType):
    """OPMD algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "opmd",
            "sample_strategy": "default",
            "policy_loss_fn": "opmd",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class AsymREAlgorithm(AlgorithmType):
    """AsymRE algorithm."""

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "sample_strategy": "default",
            "policy_loss_fn": "opmd",
            "advantage_fn": "asymre",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }


class DPOAlgorithm(AlgorithmType):
    """DPO algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = False
    schema: str = "dpo"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "sample_strategy": "default",
            "policy_loss_fn": "dpo",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }

    @classmethod
    def check_config(cls, config: Config) -> None:
        if config.mode == "train":
            if (
                config.buffer.trainer_input.experience_buffer is None
                or not config.buffer.trainer_input.experience_buffer.path
            ):
                raise ValueError(
                    "`buffer.trainer_input.experience_buffer.path` is required when `algorithm.algorithm_type == dpo`"
                )
        elif config.mode in ["both", "explore"]:
            raise ValueError(f"DPO does not support `{config.mode}` mode")

        if config.synchronizer.sync_method != SyncMethod.CHECKPOINT:
            config.synchronizer.sync_method = SyncMethod.CHECKPOINT
            logger.warning(
                "DPO only supports checkpoint synchronization, set `synchronizer.sync_method` to `checkpoint`."
            )
        config.synchronizer.sync_interval = config.trainer.save_interval
        if config.algorithm.repeat_times != 2:
            config.algorithm.repeat_times = 2  # Fake repeat times
        if config.algorithm.kl_loss_fn in {"none", None}:
            config.algorithm.kl_loss_fn = "k2"
            logger.warning("DPO must use KL loss. Set `algorithm.kl_loss_fn` to `k2`")


class TOPRAlgorithm(AlgorithmType):
    """TOPR algorithm. See https://arxiv.org/pdf/2503.14286v1"""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "reinforce",  # or simply use grpo
            "sample_strategy": "default",
            "policy_loss_fn": "topr",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class CISPOAlgorithm(AlgorithmType):
    """CISPO algorithm. See https://arxiv.org/abs/2506.13585"""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "grpo",
            "sample_strategy": "default",
            "policy_loss_fn": "cispo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class GSPOAlgorithm(AlgorithmType):
    """GSPO algorithm. See https://arxiv.org/pdf/2507.18071"""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "grpo",
            "sample_strategy": "default",
            "policy_loss_fn": "gspo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class SAPOAlgorithm(AlgorithmType):
    """SAPO (Soft Adaptive Policy Optimization) algorithm.

    SAPO uses a smooth, temperature-controlled soft gate instead of hard clipping
    to stabilize training while maintaining effective learning.
    """

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "grpo",
            "sample_strategy": "default",
            "policy_loss_fn": "sapo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class MIXAlgorithm(AlgorithmType):
    """MIX algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    use_rollout: bool = True
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 8,
            "advantage_fn": "grpo",
            "policy_loss_fn": "mix",
            "sample_strategy": "mix",
            "entropy_loss_fn": "mix",
        }


class MIXCHORDAlgorithm(AlgorithmType):
    """MIX algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    use_rollout: bool = True
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 8,
            "policy_loss_fn": "mix_chord",
            "advantage_fn": "grpo",
            "sample_strategy": "mix",
            "entropy_loss_fn": "mix",
        }


class RAFTAlgorithm(AlgorithmType):
    """RAFT Algorithm.
    This algorithm is conceptually similar to Supervised Fine-Tuning (SFT)
    but is designed to work with `experience` schema from rollouts.
    """

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "sample_strategy": "default",
            "policy_loss_fn": "sft",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }


class sPPOAlgorithm(AlgorithmType):
    """sPPO Algorithm."""

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "sample_strategy": "default",
            "policy_loss_fn": "sppo",
            "advantage_fn": "opmd",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }


class RECAlgorithm(AlgorithmType):
    """REC Algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "sample_strategy": "default",
            "policy_loss_fn": "rec",
            "advantage_fn": "rec",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class MultiStepGRPOAlgorithm(AlgorithmType):
    """Multi-Step GRPO Algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 8,
            "advantage_fn": "step_wise_grpo",
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


class OnPolicyDistillAlgorithm(AlgorithmType):
    """On-Policy Distillation Algorithm.

    Reference: Tinker library.

    Workflow stores teacher_logprobs in experience.info["teacher_logprobs"].
    Trainer's advantage_fn computes: advantages = teacher_logprobs - student_logprobs
    Trainer uses:
        importance_sampling loss if no clipping is needed
        ppo loss if clipping is needed, for better stability
    """

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = True  # advantage_fn computes from teacher_logprobs
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 8,
            "advantage_fn": "on_policy_distill",
            "advantage_fn_args": {"kl_coef": 1.0},
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",  # or importance_sampling if no clipping is needed
            "kl_penalty_fn": "none",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }


class JSDAlgorithm(AlgorithmType):
    """JSD (Jensen-Shannon Divergence) Algorithm.

    Uses JSD between teacher and student for distillation.
    Same structure as On-Policy Distill but with JSD advantage function.
    """

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = True  # advantage_fn computes JSD from teacher_logprobs
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 8,
            "advantage_fn": "jsd",
            "advantage_fn_args": {"kl_coef": 1.0, "lambda_coef": 0.5},
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }
