# -*- coding: utf-8 -*-
"""Advantage computation for Clip_B
Ref: https://arxiv.org/pdf/2602.03392"""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Tuple

import torch

if TYPE_CHECKING:
    from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn


class ClipBAdvantageFn(AdvantageFn):
    """Clip_B advantage: keep all positive-advantage tokens,
    one-side clip negative-advantage tokens by entropy signal."""

    def __init__(
        self,
        epsilon: float = 1e-6,
        mu: float = 2.5,
    ) -> None:
        self.epsilon = epsilon
        self.mu = mu

    def __call__(
        self,
        exps: "DataProto",
        **kwargs,
    ) -> Tuple["DataProto", Dict]:
        """
        Compute advantage for Clip_B.
        exps should contain the following fields:
        - token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        - response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        - uid: `(torch.Tensor)`
            shape: (bs,)
        - rollout_log_probs: `(torch.Tensor)`
            shape: (bs, response_length)
        - entropys: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns:
            exps: DataProto with advantages and returns added
            metrics: Dict with clipping metrics
        """
        token_level_rewards = exps.batch["token_level_rewards"]
        response_mask = exps.batch["response_mask"]
        index = exps.non_tensor_batch["uid"]

        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}
        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])

            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
                    id2std[idx] = torch.tensor(1.0, dtype=scores.dtype, device=scores.device)
                elif len(id2score[idx]) > 1:
                    group_scores = torch.stack(id2score[idx]).to(
                        dtype=scores.dtype, device=scores.device
                    )
                    id2mean[idx] = torch.mean(group_scores)
                    id2std[idx] = torch.std(group_scores)
                else:
                    raise ValueError(f"no score in prompt index: {idx}")

            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + self.epsilon)
            scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

        exps.batch["advantages"] = scores
        exps.batch["returns"] = scores.clone()

        # --- BEGIN: token filtering logic ---
        # Use recomputed logprobs & entropy from current model (not rollout)
        LP = exps.batch["rollout_log_probs"]  # [B, T], recomputed logprobs
        H = exps.batch["entropys"]  # [B, T], recomputed entropy
        M = response_mask  # [B, T], mask of valid tokens
        p = LP.exp()  # [B, T], probability of valid tokens
        S = p * (H + LP)  # [B, T], indicator

        # Detach for constructing clip mask (no gradient needed)
        xS = S.detach().to(torch.float32)  # [B, T]
        m = M.to(torch.float32)  # [B, T]

        # Masked global mean & variance (population variance, denominator = n)
        n = m.sum().clamp_min(1.0)
        ES = (xS * m).sum() / n  # scalar
        varS = ((xS - ES) ** 2 * m).sum() / n  # scalar
        stdS = varS.sqrt()  # scalar

        # Centered signal
        z = xS - ES  # [B, T]

        # if stdS is too small, keep all tokens; otherwise
        # keep all positive-advantage tokens; one-side clip negative-advantage tokens
        if stdS.item() < 1e-12:
            keep = torch.ones_like(M, dtype=M.dtype)  # all kept
        else:
            A = exps.batch["advantages"].detach().to(torch.float32)  # [B, T]
            pos_mask = A > 0
            neg_mask = A < 0

            keep_pos = torch.ones_like(pos_mask, dtype=torch.bool)  # positive: all kept
            keep_neg = z >= -(self.mu * stdS)  # negative: lower-side clip
            keep_zero = torch.ones_like(pos_mask, dtype=torch.bool)  # zero: all kept

            keep_bool = torch.where(pos_mask, keep_pos, torch.where(neg_mask, keep_neg, keep_zero))
            keep = keep_bool.to(M.dtype)

        M_clipped = M * keep
        exps.batch["response_mask"] = M_clipped
        # --- END: token filtering logic ---

        # Monitoring metrics
        total_tokens = m.sum().clamp_min(1.0)
        frac_clipped = 1.0 - (M_clipped.to(torch.float32).sum() / total_tokens).item()

        A = exps.batch["advantages"].detach().to(torch.float32)
        pos_mask = (A > 0).to(M.dtype)
        neg_mask = (A < 0).to(M.dtype)
        total_pos = (M * pos_mask).to(torch.float32).sum().clamp_min(1.0)
        total_neg = (M * neg_mask).to(torch.float32).sum().clamp_min(1.0)
        frac_clipped_pos = 1.0 - ((M_clipped * pos_mask).to(torch.float32).sum() / total_pos).item()
        frac_clipped_neg = 1.0 - ((M_clipped * neg_mask).to(torch.float32).sum() / total_neg).item()

        metrics = {
            "frac_clipped": frac_clipped,
            "frac_clipped_pos": frac_clipped_pos,
            "frac_clipped_neg": frac_clipped_neg,
            "ES": ES.item(),
            "varS": varS.item(),
        }
        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 1e-6,
            "mu": 2.5,
        }
