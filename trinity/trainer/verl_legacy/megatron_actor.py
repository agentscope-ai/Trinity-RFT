# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer

Modified from https://github.com/volcengine/verl/blob/v0.7.1/verl/workers/actor/megatron_actor.py
"""

import os
from functools import partial
from typing import Iterable

import torch
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.utils import VocabUtility
from verl import DataProto
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    merge_router_topk_indices,
    reorder_and_merge_vpp_layers,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_megatron_mtp_loss, unwrap_model
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.megatron_workers import MegatronPPOActor as OldMegatronPPOActor
from verl.workers.megatron_workers import logger

from trinity.algorithm import ENTROPY_LOSS_FN, KL_FN, POLICY_LOSS_FN
from trinity.algorithm.kl_fn.kl_fn import DummyKLFn
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import AlgorithmConfig


class _VocabParallelLogProbsAndEntropy(torch.autograd.Function):
    """Compute TP-sharded target log-probs and entropy in one pass.

    This avoids the verl #1970 failure mode where entropy saves logits for
    backward and Megatron cross-entropy later mutates the same logits buffer
    in-place. The implementation keeps the entropy-safe path local to the
    calculate_entropy branch and reuses a single fp32 buffer to limit peak
    memory.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1)

        if vocab_parallel_logits.dtype == torch.float32:
            logits = vocab_parallel_logits.clone()
        else:
            logits = vocab_parallel_logits.float()
        tp_group = mpu.get_tensor_model_parallel_group()

        logits_max = logits.max(dim=-1).values
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )

        logits.sub_(logits_max.unsqueeze(dim=-1))
        partition_vocab_size = logits.size(-1)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size,
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_tensor_model_parallel_world_size(),
        )

        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        logits_2d = logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d].clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )

        # Reuse the fp32 buffer through exp -> softmax to avoid keeping
        # normalized_logits / exp_logits / softmax as separate large tensors.
        logits.exp_()
        sum_exp_logits = logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )

        log_sum_exp = sum_exp_logits.log()
        logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        softmax = logits

        # Consume the softmax-weighted logits reduction immediately so the
        # temporary product tensor does not live longer than necessary.
        sum_softmax_times_logits = mul_reduce(softmax, vocab_parallel_logits)
        torch.distributed.all_reduce(
            sum_softmax_times_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
        entropy = logits_max + log_sum_exp - sum_softmax_times_logits

        ctx.partition_vocab_size = partition_vocab_size
        ctx.save_for_backward(softmax, target_mask, masked_target_1d, entropy)

        return predicted_logits - log_sum_exp, entropy

    @staticmethod
    def backward(ctx, grad_log_probs, grad_entropy):
        softmax, target_mask, masked_target_1d, entropy = ctx.saved_tensors
        grad_input = softmax
        if grad_entropy is not None:
            # Keep only one temporary vocab-sized tensor in backward: reuse the
            # saved softmax buffer for grad_input and materialize log_softmax
            # only long enough to build the entropy coefficient.
            log_softmax = softmax.log()
            log_softmax.add_(entropy.unsqueeze(dim=-1))
            log_softmax.mul_(grad_entropy.unsqueeze(dim=-1))
            if grad_log_probs is not None:
                log_softmax.add_(grad_log_probs.unsqueeze(dim=-1))
            grad_input.mul_(log_softmax)
            grad_input.mul_(-1)
        elif grad_log_probs is not None:
            grad_input.mul_(grad_log_probs.unsqueeze(dim=-1))
            grad_input.mul_(-1)
        else:
            grad_input.zero_()

        if grad_log_probs is not None:
            grad_2d = grad_input.view(-1, ctx.partition_vocab_size)
            arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
            grad_2d[arange_1d, masked_target_1d] += grad_log_probs.reshape(-1) * (
                ~target_mask
            ).view(-1).to(grad_input.dtype)

        return grad_input, None


def vocab_parallel_log_probs_and_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _VocabParallelLogProbsAndEntropy.apply(vocab_parallel_logits, target)


class MegatronPPOActor(OldMegatronPPOActor):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        super().__init__(*args, **kwargs)
        self.policy_loss_fn = None
        self.kl_loss_fn = None
        self.entropy_loss_fn = None
        self.calculate_entropy = False

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

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = [
            "input_ids",
            "position_ids",
            "attention_mask",
            "responses",
            "response_mask",
        ]

        if getattr(self, "use_prefix_grouper", False) and "prompts" in data.batch.keys():
            select_keys.append("prompts")

        if self.policy_loss_fn is not None and hasattr(self.policy_loss_fn, "select_keys"):
            select_keys.extend(self.policy_loss_fn.select_keys)

        if not isinstance(self.kl_loss_fn, DummyKLFn):
            select_keys.append("ref_log_prob")

        if self.enable_routing_replay:
            select_keys.append("routed_experts")

        select_keys = list(set(select_keys))

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if getattr(self, "use_prefix_grouper", False) and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(
            batch_keys=select_keys,
            non_tensor_batch_keys=non_tensor_select_keys,
        )

        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch(  # noqa: C901
        self,
        data: DataProto,
        forward_only=False,
        post_process_fn=None,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
        mini_batch_size=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        data.to(get_device_id())
        data.batch = data.batch.contiguous()
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch[
                "multi_modal_inputs"
            ]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)

        if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][
                :, 0
            ]  # mcore patch recompute qwen2vl's pos ids during forward

        indices = None
        temperature = data.meta_info["temperature"]
        if use_dynamic_bsz:
            assert (
                max_token_len is not None
            ), "max_token_len must be set when use_dynamic_bsz is True"
            dp_group = mpu.get_data_parallel_group()
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = (
                    self.tf_config.microbatch_group_size_per_vp_stage
                )
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                    dp_group=dp_group,
                )
                assert (
                    len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0
                ), (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    max_token_len=max_token_len,
                    dp_group=dp_group,
                )
            total_seqlen = max_token_len
        else:
            assert (
                micro_batch_size is not None
            ), "micro_batch_size is needed to be passed in when not using dynamic batch size"
            micro_batches = mini_batch.batch.split(micro_batch_size)
            seq_len = micro_batches[0]["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            log_probs = None
            entropy = None
            if isinstance(output, dict):
                log_probs = output["log_probs"]
                if "entropy" in output:
                    entropy = output["entropy"]
            else:
                assert isinstance(output, torch.Tensor)
                log_probs = output

            device = log_probs.device
            metrics = {}
            if forward_only:
                if post_process_fn is None:
                    pass
                    # metrics["logits"] = output
                else:
                    stats = post_process_fn(output, data)
                    metrics.update(stats)
                if not calculate_entropy:
                    return torch.tensor(1.0, device=device), metrics

            responses = data["responses"]
            response_length = responses.size(1)
            response_mask = data["response_mask"].to(bool)

            # compute policy loss
            log_prob = log_probs[:, -response_length - 1 : -1].contiguous()
            ret_entropy = None
            stats = {}
            if not forward_only:
                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                pg_loss, pg_loss_metrics = self.policy_loss_fn(  # type: ignore
                    logprob=log_prob, **data
                )
                prefix_metrics(src_metrics=pg_loss_metrics, prefix="actor", dst_metrics=stats)

                # TODO: to be check
                # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                rollout_log_prob = data.get("rollout_log_probs", None)
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
                    stats.update(rollout_corr_metrics)

                policy_loss = pg_loss

            if calculate_entropy:
                entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
                if not forward_only:
                    # compute entropy loss from entropy
                    entropy_loss, entropy_loss_metrics = self.entropy_loss_fn(  # type: ignore
                        entropy=entropy,
                        action_mask=response_mask,
                        loss_agg_mode=self.loss_agg_mode,
                        **data,
                    )
                    prefix_metrics(
                        src_metrics=entropy_loss_metrics,
                        prefix="actor",
                        dst_metrics=stats,
                    )

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss
                else:
                    ret_entropy = entropy

            if forward_only:
                policy_loss = torch.tensor(1.0, device=device)
            else:
                kl_loss, kl_loss_metrics = self.kl_loss_fn.calculate_kl_loss(
                    logprob=log_prob,
                    ref_logprob=data.get("ref_log_prob", None),
                    response_mask=response_mask,
                    loss_agg_mode=self.loss_agg_mode,
                    old_logprob=data.get("old_log_probs", None),
                )
                prefix_metrics(
                    src_metrics=kl_loss_metrics,
                    prefix="actor",
                    dst_metrics=stats,
                )
                policy_loss = policy_loss + kl_loss

                # return loss and stats

            # apply scale on log
            if "actor/kl_loss" in stats:
                stats["actor/kl_loss"] /= n_micro_batch
            if "actor/pg_loss" in stats:
                stats["actor/pg_loss"] /= n_micro_batch
            append_to_dict(metrics, stats)
            return policy_loss, [metrics, ret_entropy]

        def forward_step(batch_iter, model, return_schedule_plan: bool = False):
            """
            Args:
                batch_iter: the batch iterator
                model: the model
                return_schedule_plan: whether to return the schedule plan, for 1f1b overlap
            """
            if return_schedule_plan:
                assert (
                    self.tf_config.overlap_moe_expert_parallel_comm
                ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                # TODO: Fix this
                assert (
                    not calculate_entropy
                ), "calculate_entropy must be disabled to return the schedule plan"
                from megatron.core.models.gpt.gpt_model import GPTModel

                assert isinstance(model, GPTModel), "model must be a GPTModel"
                assert (
                    self.use_fused_kernels
                ), "use_fused_kernels must be enabled to return the schedule plan"
                # TODO: support VLM with MoE
                from verl.models.mcore.model_forward_1f1b_overlap import (
                    gptmodel_forward_1f1b_overlap,
                )

            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            unwrapped_model = unwrap_model(model)
            if hasattr(unwrapped_model, "vp_stage"):
                vp_rank = unwrapped_model.vp_stage
            else:
                vp_rank = 0

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(
                    batch["multi_modal_inputs"], indices
                )
            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
                router_instance_list = RouterReplayHelper.get_micro_batch_router_list(
                    self.tf_config, vp_rank
                )
                for router in router_instance_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

            if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
                layers_topk_idx = batch["routed_experts"]
                set_router_replay_data(layers_topk_idx, attention_mask, self.tf_config, vp_rank)

            from verl.models.mcore import (
                get_mcore_forward_fn,
                get_mcore_forward_fused_fn,
            )

            if self.use_fused_kernels:
                forward_fn = get_mcore_forward_fused_fn(self.hf_config)
                if return_schedule_plan:
                    forward_fn = gptmodel_forward_1f1b_overlap
                # return dict of [logits, entropy]
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=label,
                    labels_mask=label_mask,
                    temperature=temperature,
                    multi_modal_inputs=multi_modal_inputs,
                )
            else:
                forward_fn = get_mcore_forward_fn(self.hf_config)

                def logits_processor(logits, label, label_mask):
                    assert logits.shape[:2] == label.shape[:2]
                    assert label.shape == label_mask.shape
                    logits.div_(temperature)
                    ret = {}
                    if calculate_entropy:
                        # Only use the safe path when entropy is enabled. This avoids
                        # issue #1970 without changing the default Megatron CE path.
                        log_probs, entropy = vocab_parallel_log_probs_and_entropy(logits, label)
                        ret["entropy"] = entropy
                    else:
                        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret["log_probs"] = log_probs
                    return ret

                logits_processor_args = {"label": label, "label_mask": label_mask}
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    multi_modal_inputs=multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                    data_format="thd" if self.config.megatron.use_remove_padding else "bshd",
                    mtp_config=None if forward_only else getattr(self, "mtp_config", None),
                )

            if forward_only:
                meta_info = None
            else:
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                meta_info = {
                    "clip_ratio": self.config.clip_ratio,
                    "entropy_coeff": self.config.entropy_coeff,
                    "clip_ratio_c": clip_ratio_c,
                }

            if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
                merge_router_topk_indices(
                    attention_mask,
                    input_ids,
                    self.mini_layer_topk_idx_list,  # type: ignore
                    self.tf_config,
                    vp_rank,
                )

            if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
                router_instance_list = RouterReplayHelper.get_micro_batch_router_list(
                    self.tf_config, vp_rank
                )
                for router in router_instance_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                # config = self.actor_module[0].module.module.config
                vp_size = len(self.actor_module)
                microbatch_group_size_per_vp_stage = (
                    self.tf_config.microbatch_group_size_per_vp_stage
                )
                bs = n_micro_batch
                losses_reduced["mini_layer_topk_idx_tensor"] = reorder_and_merge_vpp_layers(
                    self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage  # type: ignore
                )
            else:
                losses_reduced["mini_layer_topk_idx_tensor"] = torch.cat(
                    self.mini_layer_topk_idx_list, dim=0  # type: ignore
                )
            self.mini_layer_topk_idx_list = []

        if (
            not forward_only
            and getattr(self, "mtp_config", None) is not None
            and self.mtp_config.enable_train
        ):
            losses_reduced["mtp_losses"] = [get_megatron_mtp_loss(n_micro_batch)]

        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto]) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        for data in dataloader:
            if self.config.router_replay.mode in ["R2", "R3"]:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = (
                    self.config.ppo_max_token_len_per_gpu
                    * self.config.megatron.context_parallel_size
                )
            metric_micro_batch = self.forward_backward_batch(
                data,
                calculate_entropy=self.calculate_entropy,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=self.config.ppo_mini_batch_size,
            )
            mtp_losses = metric_micro_batch.get("mtp_losses", None)
            if mtp_losses is not None:
                for mtp_metrics_dict in mtp_losses:
                    append_to_dict(metrics, mtp_metrics_dict)
            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(
                    metrics, metric[0]
                )  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError

            if self.config.router_replay.mode in ["R2", "R3"]:
                RouterReplay.clear_global_router_replay_action()
                RouterReplay.clear_global_indices()

        # add empty cache after each compute
        self.actor_optimizer.zero_grad()
        get_torch_device().empty_cache()
        return metrics
