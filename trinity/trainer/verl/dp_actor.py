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

import torch
import verl.utils.torch_functional as verl_F
from torch import nn
from verl import DataProto
from verl.utils.attention_utils import (
    index_first_axis,
    pad_input,
    rearrange,
    unpad_input,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor.dp_actor import DataParallelPPOActor as DPActor

from trinity.algorithm import ENTROPY_LOSS_FN, KL_FN, POLICY_LOSS_FN
from trinity.algorithm.kl_fn.kl_fn import DummyKLFn
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import AlgorithmConfig
from trinity.utils.log import get_logger

__all__ = ["DataParallelPPOActor"]

logger = get_logger(in_ray_actor=True)


def get_seq_idx(cu_seqlens: torch.Tensor, total_nnz: int) -> torch.Tensor:
    """
    Build `seq_idx` from `cu_seqlens`, mapping each packed position to its
    original sequence id.

    Args:
        cu_seqlens: Shape (batch + 1,). Cumulative sequence lengths from
            `unpad_input`.
            For example, [0, 3, 7, 10] means sequence 0 has length 3,
            sequence 1 has length 4, and sequence 2 has length 3.
        total_nnz: Total number of packed tokens, i.e. `cu_seqlens[-1]`.

    Returns:
        Shape (total_nnz,), where each position is the original sequence id
        (0-indexed). For example, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2].
    """
    device = cu_seqlens.device
    batch_size = cu_seqlens.shape[0] - 1
    seq_idx = torch.zeros(total_nnz, dtype=torch.int32, device=device)

    # Use cu_seqlens differences: place 1 at each sequence start index, then
    # apply cumsum to recover sequence ids.
    # Example: cu_seqlens = [0, 3, 7, 10]
    # Set 1 at indices [3, 7], then cumsum -> [0,0,0,1,1,1,1,2,2,2]
    seq_idx.scatter_(
        dim=0,
        # Start index of each new sequence (exclude the last endpoint).
        index=cu_seqlens[1:-1].long(),
        src=torch.ones(batch_size - 1, dtype=torch.int32, device=device),
    )
    seq_idx = seq_idx.cumsum(dim=0, dtype=torch.int32)

    return seq_idx


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
        self.calculate_entropy = algorithm_config.entropy_loss_fn != "none"

    def _forward_micro_batch(  # noqa: C901
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import (
                    forward_micro_batch_with_prefix_grouper,
                )

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get(
                        "entropy_from_logits_with_chunking", False
                    ),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                seq_idx = get_seq_idx(
                    cu_seqlens=cu_seqlens,
                    total_nnz=cu_seqlens[-1].item(),
                )

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(
                            rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                        )
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import (
                        process_multi_modal_inputs_for_minicpmo,
                    )

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(
                    input_ids_rmpad, shifts=-1, dims=1
                )  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        (
                            input_ids_rmpad,
                            position_ids_rmpad,
                            pad_size,
                        ) = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                    if pad_size > 0:
                        seq_idx = torch.cat(
                            [
                                seq_idx,
                                torch.full_like(seq_idx[:pad_size], fill_value=seq_idx[-1].item()),
                            ],
                            dim=0,
                        )
                        cu_seqlens[-1] += pad_size

                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(
                    0
                )  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {
                    "seq_idx": seq_idx.unsqueeze(0).to(torch.int32),
                    "cu_seqlens": cu_seqlens,
                }
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[
                    :, -response_length - 1 : -1
                ]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[
                        :, -response_length - 1 : -1, :
                    ]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(
                                verl_F.entropy_from_logits, logits
                            )
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits
                            )
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

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
                    )
                    torch.distributed.all_reduce(
                        mini_batch_token_num, op=torch.distributed.ReduceOp.SUM
                    )
                    if mini_batch_token_num == 0:
                        mini_batch_token_num += 1e-6  # to avoid division by zero

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
                    outputs = self._forward_micro_batch(
                        micro_batch=model_inputs,
                        temperature=temperature,
                        calculate_entropy=self.calculate_entropy,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if self.calculate_entropy else None

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
                        cur_token_num = torch.sum(response_mask.to(get_device_id()))
                        loss_scale = (
                            cur_token_num
                            / mini_batch_token_num
                            * torch.distributed.get_world_size()
                        )
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

                    micro_batch_metrics = {
                        key: (value.detach().item() if isinstance(value, torch.Tensor) else value)
                        for key, value in micro_batch_metrics.items()
                    }
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
