# -*- coding: utf-8 -*-
"""Trinity policy loss function for veRL's engine-based training API.
This module provides a loss function compatible with veRL's
BaseEngine.forward_backward_batch() interface, replacing the old
DataParallelPPOActor.update_policy() approach.
The loss function signature expected by veRL's engine:
    def loss_fn(model_output, data: TensorDict, dp_group=None) -> (loss, metrics)
"""

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import no_padding_2_padding

from trinity.algorithm import ENTROPY_LOSS_FN, KL_FN, POLICY_LOSS_FN
from trinity.algorithm.kl_fn.kl_fn import DummyKLFn
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import AlgorithmConfig


class TrinityPolicyLoss:
    """Picklable policy loss callable for veRL's engine API.
    Wraps Trinity's POLICY_LOSS_FN, KL_FN, and ENTROPY_LOSS_FN registries
    into a single callable that can be serialized by Ray and sent to remote
    workers via set_loss_fn().
    """

    def __init__(self, algo_config: AlgorithmConfig):
        self.policy_loss_fn = POLICY_LOSS_FN.get(algo_config.policy_loss_fn)(
            backend="verl", **algo_config.policy_loss_fn_args
        )
        self.kl_loss_fn = KL_FN.get(algo_config.kl_loss_fn)(**algo_config.kl_loss_fn_args)
        self.entropy_loss_fn = ENTROPY_LOSS_FN.get(algo_config.entropy_loss_fn)(
            **algo_config.entropy_loss_fn_args
        )
        self.calculate_entropy = algo_config.entropy_loss_fn != "none"
        self.loss_agg_mode = algo_config.loss_agg_mode
        self.use_kl_loss = not isinstance(self.kl_loss_fn, DummyKLFn)

    def __call__(
        self,
        model_output: dict,
        data: TensorDict,
        dp_group=None,
    ) -> tuple[torch.Tensor, dict]:
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        entropy = model_output.get("entropy", None)
        if entropy is not None:
            entropy = no_padding_2_padding(entropy, data)

        fields = ["response_mask"]
        fields.extend(self.policy_loss_fn.select_keys)
        if self.use_kl_loss:
            fields.append("ref_log_prob")
            if "old_log_probs" in data.keys():
                fields.append("old_log_probs")
        fields = list(dict.fromkeys(fields))
        padded_data = data.select(*fields).to_padded_tensor()

        response_mask = padded_data["response_mask"].to(bool)
        model_inputs = {"response_mask": response_mask}
        for key in fields:
            if key == "response_mask":
                continue
            if key in padded_data.keys():
                model_inputs[key] = padded_data[key]

        metrics = {}

        pg_loss, pg_loss_metrics = self.policy_loss_fn(
            logprob=log_prob,
            batch_num_tokens=tu.get_non_tensor_data(
                data=data, key="batch_num_tokens", default=None
            ),
            dp_size=tu.get_non_tensor_data(data=data, key="dp_size", default=None),
            **model_inputs,
        )
        prefix_metrics(src_metrics=pg_loss_metrics, prefix="actor", dst_metrics=metrics)
        policy_loss = pg_loss

        if self.calculate_entropy and entropy is not None:
            entropy_loss, entropy_loss_metrics = self.entropy_loss_fn(
                entropy=entropy,
                action_mask=response_mask,
                loss_agg_mode=self.loss_agg_mode,
                **model_inputs,
            )
            prefix_metrics(src_metrics=entropy_loss_metrics, prefix="actor", dst_metrics=metrics)
            policy_loss = policy_loss - entropy_loss

        if self.use_kl_loss:
            kl_loss, kl_loss_metrics = self.kl_loss_fn.calculate_kl_loss(
                logprob=log_prob,
                ref_logprob=model_inputs.get("ref_log_prob", None),
                response_mask=response_mask,
                loss_agg_mode=self.loss_agg_mode,
                old_logprob=model_inputs.get("old_log_probs", None),
            )
            prefix_metrics(src_metrics=kl_loss_metrics, prefix="actor", dst_metrics=metrics)
            policy_loss = policy_loss + kl_loss

        # final_loss: the unscaled combined loss (pg - entropy + kl), aligned
        # with the old dp_actor's actor/final_loss metric.  veRL's engine also
        # reports "loss" (sum across micro-batches, DP-averaged), but that has
        # different scaling semantics.
        metrics["final_loss"] = policy_loss.detach().item()

        return policy_loss, metrics

    def __repr__(self) -> str:
        return (
            f"TrinityPolicyLoss(policy={self.policy_loss_fn.__class__.__name__}, "
            f"kl={self.kl_loss_fn.__class__.__name__}, "
            f"entropy={self.entropy_loss_fn.__class__.__name__})"
        )


def build_trinity_loss(algo_config: AlgorithmConfig) -> TrinityPolicyLoss:
    """Build a TrinityPolicyLoss instance for veRL's engine API."""
    return TrinityPolicyLoss(algo_config)
