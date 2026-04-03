# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor.
Modified from https://github.com/volcengine/verl/blob/v0.7.1/verl/workers/actor/dp_actor.py
"""

import logging
import os

import torch
from torch import nn
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor as DPActor

from trinity.algorithm import ENTROPY_LOSS_FN, KL_FN, POLICY_LOSS_FN
from trinity.algorithm.entropy_loss_fn.entropy_loss_fn import DummyEntropyLossFn
from trinity.algorithm.kl_fn.kl_fn import DummyKLFn
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import AlgorithmConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(DPActor):
    def __init__(
        self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config, actor_module, actor_optimizer)
        self.policy_loss_fn = None
        self.kl_loss_fn = None
        self.entropy_loss_fn = None

    def set_algorithm(self, algorithm_config: AlgorithmConfig):
        self.loss_agg_mode = algorithm_config.loss_agg_mode
        self.policy_loss_fn = POLICY_LOSS_FN.get(algorithm_config.policy_loss_fn)(
            backend="verl", **algorithm_config.policy_loss_fn_args
        )
        self.kl_loss_fn = KL_FN.get(algorithm_config.kl_loss_fn)(**algorithm_config.kl_loss_fn_args)
        self.entropy_loss_fn = ENTROPY_LOSS_FN.get(algorithm_config.entropy_loss_fn)(
            **algorithm_config.entropy_loss_fn_args
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):  # noqa: C901
        # make sure we are in training mode
        self.actor_module.train()

        # temperature must be in the data.meta_info to avoid silent error
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        select_keys = [
            "input_ids",
            "position_ids",
            "attention_mask",
            "responses",
            "response_mask",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        select_keys.extend(self.policy_loss_fn.select_keys)
        if not isinstance(self.kl_loss_fn, DummyKLFn):
            select_keys.append("ref_log_prob")
        # rollout_is_weights will be used in policy loss
        # rollout_log_probs is equal to old_log_prob in Trinity
        select_keys = list(set(select_keys))

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        # EXPERIMENTAL: apply loss scale fix
        do_fix_actor_microbatch_loss_scale = self.config.fix_actor_microbatch_loss_scale and (
            self.loss_agg_mode == "token-mean"
        )

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if do_fix_actor_microbatch_loss_scale:
                    # calculate the total number of response tokens in the minibatch
                    mini_batch_token_num = torch.sum(
                        mini_batch.batch["response_mask"].to(get_device_id())
                    ).item()

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {
                        **micro_batch.batch,
                        **micro_batch.non_tensor_batch,
                        "pad_token_id": pad_token_id,
                    }
                    response_mask = model_inputs["response_mask"]
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    # all return: (bsz, response_length)
                    calculate_entropy = self.entropy_loss_fn != DummyEntropyLossFn
                    outputs = self._forward_micro_batch(
                        micro_batch=model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    pg_loss, pg_loss_metrics = self.policy_loss_fn(  # type: ignore
                        logprob=log_prob, **model_inputs
                    )
                    prefix_metrics(
                        src_metrics=pg_loss_metrics, prefix="actor", dst_metrics=micro_batch_metrics
                    )

                    # TODO: to be check
                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import (
                            compute_rollout_corr_metrics_from_logprobs,
                        )

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    # compute entropy loss from entropy
                    entropy_loss, entropy_loss_metrics = self.entropy_loss_fn(  # type: ignore
                        entropy=entropy,
                        action_mask=response_mask,
                        loss_agg_mode=self.loss_agg_mode,
                        **model_inputs,
                    )
                    prefix_metrics(
                        src_metrics=entropy_loss_metrics,
                        prefix="actor",
                        dst_metrics=micro_batch_metrics,
                    )

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss

                    kl_loss, kl_loss_metrics = self.kl_loss_fn.calculate_kl_loss(
                        logprob=log_prob,
                        ref_logprob=model_inputs.get("ref_log_prob", None),
                        response_mask=response_mask,
                        loss_agg_mode=self.loss_agg_mode,
                        old_logprob=model_inputs.get("old_log_probs", None),
                    )
                    prefix_metrics(
                        src_metrics=kl_loss_metrics,
                        prefix="actor",
                        dst_metrics=micro_batch_metrics,
                    )
                    policy_loss = policy_loss + kl_loss

                    # set loss scale for the microbatch
                    if not do_fix_actor_microbatch_loss_scale:
                        # original implementation of microbatch loss scale
                        if self.config.use_dynamic_bsz:
                            loss_scale = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        else:
                            loss_scale = 1.0 / self.gradient_accumulation
                    else:
                        # EXPERIMENTAL: fix for token-mean loss aggregation
                        # scale microbatch loss according to the number of tokens (rather than sequences)
                        loss_scale = torch.sum(response_mask).item() / (mini_batch_token_num + 1e-6)

                    loss = policy_loss * loss_scale
                    micro_batch_metrics["actor/final_loss"] = loss.detach().item()
                    if "actor/kl_loss" in micro_batch_metrics:
                        micro_batch_metrics["actor/kl_loss"] *= loss_scale
                    if "actor/pg_loss" in micro_batch_metrics:
                        micro_batch_metrics["actor/pg_loss"] *= loss_scale

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
