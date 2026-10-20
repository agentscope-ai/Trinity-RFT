"""SAPO policy loss function.
Soft Adaptive Policy Optimization (SAPO) is a reinforcement learning algorithm
that uses a smooth, temperature-controlled soft gate instead of hard clipping.

Refer to the SAPO paper for details. https://arxiv.org/abs/2511.20347
"""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


class SAPOPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        tau_pos: float = 1.0,
        tau_neg: float = 1.05,
        loss_agg_mode: str = "token-mean",
    ) -> None:
        """Initialize SAPO policy loss function.

        Args:
            backend: The training framework/backend to use (e.g., "verl")
            tau_pos: Temperature for positive advantages (τ_pos), default 1.0
            tau_neg: Temperature for negative advantages (τ_neg), default 1.05, should be >= tau_pos
            loss_agg_mode: Mode for aggregating loss across tokens
        """
        super().__init__(backend=backend)
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.loss_agg_mode = loss_agg_mode

        # Validate that tau_neg > tau_pos for stability
        assert self.tau_neg >= self.tau_pos, (
            f"tau_neg ({self.tau_neg}) should be >= tau_pos ({self.tau_pos}) "
            "for better training stability"
        )

    def soft_gate_function(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """Compute the soft gate function f_{i,t}(x).

        The soft gate function is defined as:
            f_{i,t}(x) = σ(τ_{i,t} * (x - 1)) * 4 / τ_{i,t}

        where:
            - σ is the sigmoid function
            - τ_{i,t} is the asymmetric temperature (tau_pos or tau_neg)
            - x is the importance sampling ratio r_{i,t}(θ)

        Args:
            ratio: Token-level importance sampling ratio r_{i,t}(θ)
            advantages: Normalized advantage function Â_i (same for all tokens in a sequence)

        Returns:
            The soft gate values for each token
        """
        # Select temperature based on advantage sign
        # tau_i,t = tau_pos if A_i > 0, else tau_neg
        tau = torch.where(
            advantages > 0,
            torch.tensor(self.tau_pos, device=ratio.device, dtype=ratio.dtype),
            torch.tensor(self.tau_neg, device=ratio.device, dtype=ratio.dtype),
        )

        # Compute sigmoid(tau * (ratio - 1))
        sigmoid_input = tau * (ratio - 1)
        sigmoid_output = torch.sigmoid(sigmoid_input)

        # Compute the soft gate: sigma(tau * (x - 1)) * 4 / tau
        soft_gate = sigmoid_output * (4.0 / tau)

        return soft_gate

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute SAPO policy loss.

        The SAPO objective function is:
            J(θ) = E[1/G Σ 1/|y_i| Σ f_{i,t}(r_{i,t}(θ)) * Â_i]

        We minimize the negative of this objective.

        Args:
            logprob: Log probabilities from current policy π_θ
            old_logprob: Log probabilities from old policy π_{θ_old}
            action_mask: Mask indicating valid tokens
            advantages: Group-normalized advantage function

        Returns:
            loss: The computed policy loss (negative of objective)
            metrics: Dictionary of metrics for logging
        """
        # Compute token-level importance sampling ratio
        # r_{i,t}(θ) = π_θ(y_{i,t}|q, y_{i,<t}) / π_{θ_old}(y_{i,t}|q, y_{i,<t})
        negative_approx_kl = logprob - old_logprob
        ratio = torch.exp(negative_approx_kl)

        # Compute approximate KL divergence for monitoring
        ppo_kl = masked_mean(-negative_approx_kl, action_mask)

        # Compute soft gate function
        soft_gate = self.soft_gate_function(ratio, advantages)

        # SAPO loss: -E[f_{i,t}(r_{i,t}) * Â_i]
        # Differentiate f(r) through the ratio: grad = f'(r) * r * A * grad(log pi), i.e. the weight
        # 4 p (1 - p) * r of the paper (Eq. 7-8), which peaks at r = 1 and decays away from it.
        sapo_loss = -advantages * soft_gate

        # Aggregate loss across tokens
        loss = aggregate_loss(sapo_loss, action_mask, loss_agg_mode=self.loss_agg_mode)

        # Compute metrics for logging
        avg_soft_gate = masked_mean(soft_gate, action_mask)
        avg_ratio = masked_mean(ratio, action_mask)

        # Compute fraction of tokens with positive/negative advantages
        pos_adv_frac = masked_mean((advantages > 0).float(), action_mask)

        metrics = {
            "sapo_loss": loss.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "avg_soft_gate": avg_soft_gate.detach().item(),
            "avg_ratio": avg_ratio.detach().item(),
            "pos_adv_frac": pos_adv_frac.detach().item(),
        }

        return loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        """Get default initialization arguments for SAPO.

        Default configuration (from the SAPO paper):
            - tau_pos: 1.0 (temperature for positive advantages)
            - tau_neg: 1.05 (temperature for negative advantages)
            - loss_agg_mode: "token-mean" (average over tokens)

        The asymmetric temperatures (tau_neg > tau_pos) help stabilize training
        by more aggressively suppressing updates from tokens with negative advantages.

        Returns:
            Dictionary of default arguments
        """
        return {
            "tau_pos": 1.0,
            "tau_neg": 1.05,
            "loss_agg_mode": "token-mean",
        }
