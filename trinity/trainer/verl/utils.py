"""Utils for ccompatibility issues with verl."""

import os
from logging import Logger
from typing import List

import numpy as np
import torch
from transformers import ProcessorMixin
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
    experiences: List[Experience], pad_token_id: int, processor: ProcessorMixin, logger: Logger
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

    if processor is not None:
        import inspect

        # Adapted from verl/experimental/agent_loop/agent_loop.py
        position_ids_list, multi_modal_inputs = [], []
        for idx, exp in enumerate(experiences):
            mm_inputs = exp.multi_modal_inputs or {}
            input_ids = batch_dict["input_ids"][idx].unsqueeze(0)
            attention_mask = batch_dict["attention_mask"][idx].unsqueeze(0)

            get_rope_index_sig = inspect.signature(processor.get_rope_index)
            get_rope_index_kwargs = {}
            for key in mm_inputs.keys():
                if key in get_rope_index_sig.parameters:
                    get_rope_index_kwargs[key] = mm_inputs[key]

            vision_position_ids, _ = processor.get_rope_index(
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


# modified from verl/workers/fsdp_workers.py:ActorRolloutRefWorker._build_model_optimizer
def get_model_class(hf_config):
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
    )

    has_remote_code = hasattr(hf_config, "auto_map") and any(
        hf_config.architectures[0] in val for val in hf_config.auto_map.values()
    )
    if has_remote_code:
        auto_class = next(
            k for k, v in hf_config.auto_map.items() if hf_config.architectures[0] in v
        )
        match auto_class:
            case "AutoModelForVision2Seq":
                model_class = AutoModelForVision2Seq
            case "AutoModelForCausalLM":
                model_class = AutoModelForCausalLM
            case "AutoModelForImageTextToText":
                model_class = AutoModelForImageTextToText
            case _:
                model_class = AutoModel
    else:
        if type(hf_config) in AutoModelForVision2Seq._model_mapping.keys():
            model_class = AutoModelForVision2Seq
        elif type(hf_config) in AutoModelForCausalLM._model_mapping.keys():
            model_class = AutoModelForCausalLM
        elif type(hf_config) in AutoModelForImageTextToText._model_mapping.keys():
            model_class = AutoModelForImageTextToText
        else:
            model_class = AutoModel
    return model_class
