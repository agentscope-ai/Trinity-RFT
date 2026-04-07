"""Utils for ccompatibility issues with verl."""

import os
from logging import Logger
from typing import List

import numpy as np
import torch
from transformers import PreTrainedModel
from verl import DataProto
from verl.trainer.ppo.metric_utils import _compute_response_info
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

from trinity.common.config import Config
from trinity.common.experience import (
    Experience,
    gather_action_masks,
    gather_attention_masks,
    gather_response_attrs,
    gather_token_ids,
    split_dpo_experience_to_single_turn,
)


def to_data_proto(
    experiences: List[Experience], pad_token_id: int, model: PreTrainedModel, logger: Logger
) -> DataProto:  # noqa: C901
    """Convert List[Experience] to verl DataProto."""
    assert len(experiences) > 0, "No experiences provided."
    if experiences[0].experience_type == "dpo":
        experiences = split_dpo_experience_to_single_turn(experiences)
    max_prompt_length = max([exp.prompt_length for exp in experiences])
    max_response_length = max([len(exp.tokens) - exp.prompt_length for exp in experiences])  # type: ignore

    attention_mask = gather_attention_masks(
        experiences, max_prompt_length, max_response_length
    ).long()
    cumsum = torch.cumsum(attention_mask, dim=-1)
    position_ids = torch.clip(cumsum - 1, 0, None).long()
    tokens = gather_token_ids(
        experiences, max_prompt_length, max_response_length, pad_token_id
    ).long()
    batch_dict = {
        "uid": np.array([exp.eid.tid for exp in experiences]),
        "unique_ids": np.array([exp.eid.uid for exp in experiences]),
        "position_ids": position_ids,
        "input_ids": tokens,
        "responses": tokens[:, max_prompt_length:],
        "attention_mask": attention_mask,
        "response_mask": gather_action_masks(experiences, max_response_length),
    }

    have_reward = all(exp.reward is not None for exp in experiences)
    have_token_level_reward = all(exp.token_level_reward is not None for exp in experiences)
    if have_reward or have_token_level_reward:
        assert all(exp.logprobs is not None for exp in experiences), "No logprobs provided."
        if have_token_level_reward:
            if have_reward:
                logger.warning(
                    "Both experiences.rewards and experiences.token_level_rewards are provided. "
                    "Using experiences.token_level_rewards."
                )
            token_level_rewards = gather_response_attrs(
                experiences, "token_level_reward", max_response_length
            )
        else:
            token_level_rewards = torch.zeros(attention_mask.shape, dtype=torch.float32)
            eos_mask_idx = cumsum.argmax(dim=-1)
            token_level_rewards[torch.arange(len(experiences)), eos_mask_idx] = torch.tensor(
                [exp.reward for exp in experiences],
                dtype=torch.float32,
            )
            token_level_rewards = token_level_rewards[:, max_prompt_length:]
        batch_dict.update(
            {
                "token_level_scores": token_level_rewards,
                "rollout_log_probs": gather_response_attrs(
                    experiences, "logprobs", max_response_length
                ),
            }
        )

    for attr in ["advantages", "returns", "teacher_logprobs"]:
        if all(getattr(exp, attr, None) is not None for exp in experiences):
            batch_dict[attr] = gather_response_attrs(experiences, attr, max_response_length)

    if hasattr(model, "get_rope_index"):
        # used for multi-modal model
        import inspect

        # Adapted from verl/experimental/agent_loop/agent_loop.py
        position_ids_list, multi_modal_inputs = [], []
        for idx, exp in enumerate(experiences):
            mm_inputs = exp.multi_modal_inputs or {}
            input_ids = batch_dict["input_ids"][idx].unsqueeze(0)
            attention_mask = batch_dict["attention_mask"][idx].unsqueeze(0)

            get_rope_index_sig = inspect.signature(model.get_rope_index)
            get_rope_index_kwargs = {}
            for key in get_rope_index_sig.parameters:
                if key in {"self", "input_ids", "attention_mask", "kwargs"}:
                    continue
                elif key == "mm_token_type_ids":
                    pad_data = torch.zeros_like(input_ids)
                    if key in mm_inputs:
                        data = mm_inputs.pop(key)
                        start = max_prompt_length - exp.prompt_length
                        end = start + data.size(1)
                        pad_data[:, start:end] = data
                    get_rope_index_kwargs[key] = pad_data
                else:
                    get_rope_index_kwargs[key] = mm_inputs.get(key, None)

            vision_position_ids, _ = model.get_rope_index(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **get_rope_index_kwargs,
            )  # (3, 1, seq_len)
            vision_position_ids = vision_position_ids.squeeze(1)  # (3, seq_len)

            text_position_ids = batch_dict["position_ids"][idx].unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat(
                (text_position_ids, vision_position_ids), dim=0
            )  # (4, seq_length)
            position_ids_list.append(position_ids)  # (4, seq_length)
            multi_modal_inputs.append(mm_inputs)

        batch_dict["position_ids"] = torch.stack(
            position_ids_list, dim=0
        ).long()  # (bs, 4, seq_length)
        batch_dict["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

    custom_fields_set = set(tuple(exp.custom_fields) for exp in experiences)
    if len(custom_fields_set) == 1:
        custom_fields = list(custom_fields_set)[0]
        for custom_field in custom_fields:
            batch_dict[custom_field.destination_field] = torch.tensor(
                [exp.info[custom_field.source_field] for exp in experiences],
                dtype=custom_field.data_type,
            )
    else:
        raise ValueError("Custom fields are not consistent across experiences.")
    meta_info = {
        "model_versions": np.array([exp.info.get("model_version", 0) for exp in experiences])
    }
    return DataProto.from_single_dict(batch_dict, meta_info=meta_info)


def compute_data_metrics(batch: DataProto) -> dict:
    """
    Computes various metrics from a batch of data for PPO training.
    Modified from verl.trainer.ppo.metric_utils.compute_data_metrics

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values
            - critic/vf_explained_var: Explained variance of the value function
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    metrics = {}

    if "token_level_rewards" in batch.batch and "token_level_scores" in batch.batch:
        sequence_score = batch.batch["token_level_scores"].sum(-1)
        sequence_reward = batch.batch["token_level_rewards"].sum(-1)
        metrics.update(
            {
                # score
                "critic/score/mean": torch.mean(sequence_score).detach().item(),
                "critic/score/max": torch.max(sequence_score).detach().item(),
                "critic/score/min": torch.min(sequence_score).detach().item(),
                # reward
                "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
                "critic/rewards/max": torch.max(sequence_reward).detach().item(),
                "critic/rewards/min": torch.min(sequence_reward).detach().item(),
            }
        )

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]
    metrics.update(
        {
            # response length
            "response_length/mean": torch.mean(response_length).detach().item(),
            "response_length/max": torch.max(response_length).detach().item(),
            "response_length/min": torch.min(response_length).detach().item(),
            "response_length/clip_ratio": torch.mean(
                torch.eq(response_length, max_response_length).float()
            )
            .detach()
            .item(),
            # prompt length
            "prompt_length/mean": torch.mean(prompt_length).detach().item(),
            "prompt_length/max": torch.max(prompt_length).detach().item(),
            "prompt_length/min": torch.min(prompt_length).detach().item(),
            "prompt_length/clip_ratio": torch.mean(
                torch.eq(prompt_length, max_prompt_length).float()
            )
            .detach()
            .item(),
        }
    )

    if "advantages" in batch.batch:
        # adv
        advantages = batch.batch["advantages"]
        if response_mask.numel() > 0:
            valid_adv = torch.masked_select(advantages, response_mask)
        else:
            valid_adv = torch.zeros(1)
        metrics.update(
            {
                # adv
                "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
                "critic/advantages/max": torch.max(valid_adv).detach().item(),
                "critic/advantages/min": torch.min(valid_adv).detach().item(),
            }
        )
    if "returns" in batch.batch:
        # returns
        returns = batch.batch["returns"]
        if response_mask.numel() > 0:
            valid_returns = torch.masked_select(returns, response_mask)
        else:
            valid_returns = torch.zeros(1)
        metrics.update(
            {
                "critic/returns/mean": torch.mean(valid_returns).detach().item(),
                "critic/returns/max": torch.max(valid_returns).detach().item(),
                "critic/returns/min": torch.min(valid_returns).detach().item(),
            }
        )

    return metrics


def get_latest_hf_checkpoint_path(config: Config):
    """Get the latest huggingface checkpoint path"""
    if config.trainer.trainer_type != "verl":
        raise ValueError("This function is only for verl trainer.")
    checkpoint_dir = find_latest_ckpt_path(config.checkpoint_job_dir)
    hf_checkpoint_dir = os.path.join(checkpoint_dir, "actor", "huggingface")
    if not os.path.exists(hf_checkpoint_dir):
        raise ValueError(f"No huggingface checkpoint found in {hf_checkpoint_dir}")
    return hf_checkpoint_dir


# modified from verl/utils/fsdp_utils.py:apply_fsdp2
# bug fix for transformers v5
def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    import torch.nn as nn
    from verl.utils.fsdp_utils import (
        CPUOffloadPolicy,
        fully_shard,
        maybe_patch_fsdp_module,
    )

    assert (
        CPUOffloadPolicy is not None
    ), "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


# modified from verl/utils/seqlen_balancing.py:rearrange_micro_batches
def rearrange_micro_batches(
    batch,
    max_token_len,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    """
    Split a batch into micro-batches by total token count, with optional DP sync and padding.

    Args:
        batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
        max_token_len (int): max sum of attention_mask per micro-batch.
        dp_group (optional): torch.distributed group for data-parallel sync.
        num_batches_divided_by (optional): virtual pipeline parallel size, for megatron.
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
        min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).
        use_dynamic_bsz_balance (bool, optional): balance the computational workload between micro-batches

    Returns:
        List[TensorDict]: the micro-batches.
        List[List[int]]: index lists mapping each micro-batch back to original positions.
    """
    from torch import distributed as dist
    from verl.utils import tensordict_utils as tu
    from verl.utils.device import get_device_name
    from verl.utils.seqlen_balancing import (
        calculate_workload,
        ceildiv,
        get_seqlen_balanced_partitions,
        roundup_divisible,
    )

    # this is per local micro_bsz
    input_ids = batch["input_ids"]
    if input_ids.is_nested:
        seq_len_effective: torch.Tensor = input_ids.offsets().diff()
    else:
        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    max_seq_len = seq_len_effective.max().item()

    assert (
        max_token_len >= max_seq_len
    ), f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    total_seqlen = seq_len_effective.sum().item()
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    if min_num_micro_batch is not None:
        # used to support pp
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    if dist.is_initialized() and same_micro_num_in_dp:
        num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()
    if num_batches_divided_by is not None:
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    assert num_micro_batches <= len(seq_len_effective)

    # note that seq_len_effective is a GPU tensor. We need to make it a list to avoid D2H!
    workloads = calculate_workload(seq_len_effective).cpu().tolist()
    micro_bsz_idx = get_seqlen_balanced_partitions(workloads, num_micro_batches, equal_size=False)

    if use_dynamic_bsz_balance:
        # Use the sum of squared sequence lengths to approximate attention computation workload
        micro_bsz_idx.sort(
            key=lambda partition: (
                sum(workloads[idx] for idx in partition),
                partition[0] if partition else 0,
            ),
            reverse=True,
        )
        # Place smaller micro-batches at both ends to reduce the bubbles exposed during the warm-up and cool-down.
        micro_bsz_idx = micro_bsz_idx[::2][::-1] + micro_bsz_idx[1::2]

    micro_batches = []

    for partition in micro_bsz_idx:
        curr_micro_batch = tu.index_select_tensor_dict(batch, partition)
        micro_batches.append(curr_micro_batch)

    return micro_batches, micro_bsz_idx


# add rope_theta to hf config for backward compatibility, can be removed after verl is updated
def patch_rope_theta_in_hf_config(hf_config):
    if not hasattr(hf_config, "rope_theta"):
        if hasattr(hf_config, "rope_parameters"):
            rope_parameters = hf_config.rope_parameters
        elif hasattr(hf_config, "text_config") and hasattr(
            hf_config.text_config, "rope_parameters"
        ):
            rope_parameters = hf_config.text_config.rope_parameters
        else:
            rope_parameters = {}

        rope_theta = rope_parameters.get("rope_theta", None)
        if rope_theta is not None:
            setattr(hf_config, "rope_theta", rope_theta)
