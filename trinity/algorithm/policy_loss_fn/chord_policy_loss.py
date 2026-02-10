"""Implements the CHORD policy loss function."""

import math
from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.policy_loss_fn.ppo_policy_loss import PPOPolicyLossFn
from trinity.algorithm.policy_loss_fn.sft_loss import SFTLossFn
from trinity.algorithm.utils import aggregate_loss


def mu_schedule_function(
    global_step: int, mu_warmup_steps: int, mu_decay_steps: int, mu_peak: float, mu_valley: float
) -> float:
    """
    Computes a cosine decay schedule with a warmup phase for the mu parameter.
    """
    # Warmup
    if global_step < mu_warmup_steps:
        return (global_step / mu_warmup_steps) * mu_peak

    # Decay
    if global_step >= (mu_warmup_steps + mu_decay_steps):
        return mu_valley

    adjusted_step = global_step - mu_warmup_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * adjusted_step / mu_decay_steps))
    decayed_mu = (mu_peak - mu_valley) * cosine_decay + mu_valley
    return decayed_mu


class SFTISLossFn(PolicyLossFn):
    """
    SFT loss with importance sampling
    """

    def __init__(self, backend: str = "verl", loss_agg_mode: str = "token-mean") -> None:
        super().__init__(backend=backend)
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        token_prob = torch.exp(logprob)
        sft_loss = aggregate_loss(
            -logprob * token_prob.detach(), action_mask, loss_agg_mode=self.loss_agg_mode
        )
        return sft_loss, {"sft_is_loss": sft_loss.detach().item()}

    @classmethod
    def default_args(cls):
        return {
            "loss_agg_mode": "token-mean",
        }


def phi_function(token_prob):
    """
    The phi function downweights token with extreme probability.
    Feel free to modify this function.
    """
    return token_prob * (1 - token_prob)


class SFTPhiLossFn(PolicyLossFn):
    """
    SFT loss with transformed phi function
    """

    def __init__(
        self, backend: str = "verl", loss_agg_mode: str = "token-mean", cutoff_prob: float = 1.0
    ) -> None:
        super().__init__(backend=backend)
        self.loss_agg_mode = loss_agg_mode
        self.cutoff_prob = cutoff_prob
        assert 0.0 <= self.cutoff_prob <= 1.0

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        token_prob = torch.exp(logprob)
        if self.cutoff_prob < 1.0:
            logprob = torch.clamp(logprob, max=math.log(self.cutoff_prob))

        weighted_phi = phi_function(token_prob)

        sft_loss = aggregate_loss(
            -logprob * weighted_phi.detach(), action_mask, loss_agg_mode=self.loss_agg_mode
        )
        return sft_loss, {"sft_phi_loss": sft_loss.detach().item()}

    @classmethod
    def default_args(cls):
        return {
            "loss_agg_mode": "token-mean",
            "cutoff_prob": 1.0,
        }


class MIXCHORDPolicyLossFn(PolicyLossFn):
    """Implements a mixed policy loss combining GRPO and SFT losses.

    This loss function applies different loss components to data based on whether
    it comes from an expert or not, as indicated by `expert_mask`. It combines:

    - GRPO loss (self.grpo_loss_fn) for non-expert data
    - SFT loss (self.sft_loss_fn) for expert data
        the weight of SFT loss is globally controled by `mu_schedule` function
        the tokenwise weights are calculated using different SFT loss formulas

    The per-sample weights are normalized using either `experience_per_gpu` or
    `gradient_accumulation`, depending on whether dynamic batch sizing is enabled,
    to ensure consistent weighting across different batches of the same type experiences.
    """

    def __init__(
        self,
        backend: str = "verl",
        mu_warmup_steps: int = 0,
        mu_decay_steps: int = 0,
        mu_peak: float = 0.1,
        mu_valley: float = 0.1,
        enable_phi_function: bool = True,
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
        use_dynamic_bsz: Optional[bool] = None,
        ppo_mini_batch_size: int = 1,
        ppo_micro_batch_size_per_gpu: int = 1,
        ngpus_trainer: int = 1,
        train_batch_size_usual: int = 1,
        train_batch_size_expert: int = 1,
        loss_agg_mode: str = "token-mean",
        sft_loss_agg_mode: Optional[str] = None,
        grpo_loss_agg_mode: Optional[str] = None,
    ) -> None:
        super().__init__(backend=backend)
        self.mu_warmup_steps = mu_warmup_steps
        self.mu_decay_steps = mu_decay_steps
        self.mu_peak = mu_peak
        self.mu_valley = mu_valley
        self.enable_phi_function = enable_phi_function
        self.use_dynamic_bsz = use_dynamic_bsz
        self.experience_per_gpu = ppo_mini_batch_size // ngpus_trainer
        self.gradient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu
        self.train_batch_size_usual = train_batch_size_usual // ngpus_trainer
        self.train_batch_size_expert = train_batch_size_expert // ngpus_trainer
        self.grpo_loss_fn = PPOPolicyLossFn(
            clip_range=clip_range,
            clip_range_low=clip_range_low,
            clip_range_high=clip_range_high,
            loss_agg_mode=grpo_loss_agg_mode or loss_agg_mode,
        )
        if enable_phi_function:
            self.sft_loss_fn = SFTPhiLossFn(loss_agg_mode=sft_loss_agg_mode or loss_agg_mode)
        else:
            self.sft_loss_fn = SFTLossFn(loss_agg_mode=sft_loss_agg_mode or loss_agg_mode)

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        expert_mask: torch.Tensor,
        step: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        assert (
            len(expert_mask) == logprob.shape[0]
        ), f"Error: {len(expert_mask)=} != {logprob.shape[0]=}"

        assert len(step) == logprob.shape[0], f"Error: {len(step)=} != {logprob.shape[0]=}"

        assert (
            step.max().item() == step.min().item()
        ), f"Error: {step.max().item()} != {step.min().item()}"
        current_step = step.max().item()

        n_usual_exp = torch.sum(~expert_mask).item()
        n_expert_exp = torch.sum(expert_mask).item()

        if self.use_dynamic_bsz:
            per_micro_batch_weight_usual = self.experience_per_gpu / (
                logprob.shape[0] * self.train_batch_size_usual
            )
            per_micro_batch_weight_expert = self.experience_per_gpu / (
                logprob.shape[0] * self.train_batch_size_expert
            )
        else:
            per_micro_batch_weight_usual = self.gradient_accumulation / self.train_batch_size_usual  # type: ignore
            per_micro_batch_weight_expert = self.gradient_accumulation / self.train_batch_size_expert  # type: ignore

        grpo_loss, grpo_metrics = self.grpo_loss_fn(
            logprob[~expert_mask],
            old_logprob[~expert_mask],
            action_mask[~expert_mask],
            advantages[~expert_mask],
            **kwargs,
        )
        grpo_loss = grpo_loss * n_usual_exp * per_micro_batch_weight_usual
        grpo_metrics = {
            k: v * n_usual_exp * per_micro_batch_weight_usual for k, v in grpo_metrics.items()
        }

        # SFT Loss (expert)
        sft_loss, sft_metrics = self.sft_loss_fn(
            logprob[expert_mask],
            action_mask[expert_mask],
        )
        sft_loss = sft_loss * n_expert_exp * per_micro_batch_weight_expert
        sft_metrics = {
            k: v * n_expert_exp * per_micro_batch_weight_expert for k, v in sft_metrics.items()
        }

        mu = mu_schedule_function(
            current_step, self.mu_warmup_steps, self.mu_decay_steps, self.mu_peak, self.mu_valley
        )

        loss = (1 - mu) * grpo_loss + mu * sft_loss

        metrics = {f"usual/{k}": v for k, v in grpo_metrics.items()}
        metrics.update({f"expert/{k}": v for k, v in sft_metrics.items()})
        metrics.update({"loss": loss.item(), "mu": mu})

        return loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        """
        mu_warmup_steps: int, mu_decay_steps: int, mu_peak: float, mu_valley: float
        """
        return {
            "mu_warmup_steps": 0,
            "mu_decay_steps": 0,
            "mu_peak": 0.1,
            "mu_valley": 0.1,
            "clip_range": 0.2,
            "enable_phi_function": True,
            "loss_agg_mode": "token-mean",
        }
