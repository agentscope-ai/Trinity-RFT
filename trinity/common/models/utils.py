# -*- coding: utf-8 -*-
import os
import re
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from trinity.common.config import TrainerConfig
from trinity.utils.log import get_logger


def tokenize_and_mask_messages_hf(
    tokenizer: Any,
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    chat_template: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Calculate the assistant token mask with `chat_template`.

    Args:
        tokenizer (Any): The tokenizer.
        messages (List[dict]): Messages with `role` and `content` fields.
        tools (Optional[List[dict]]): The list of tool dictionaries.
        chat_template (str): The chat template with `{% generation %}` symbol.

    Returns:
        `torch.Tensor`: The token_ids (sequence_length)
        `torch.Tensor`: Assistant_masks (sequence_length).
        `int`: Prompt length.
    """
    token_dict = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        padding=False,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    # find the first assistant token, the tokens before are prompt tokens
    prompt_length = torch.argmax(token_dict["assistant_masks"][0]).item()
    return token_dict["input_ids"][0], token_dict["assistant_masks"][0], prompt_length


def tokenize_and_mask_messages_default(
    tokenizer: Any,
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    chat_template: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Calculate the assistant token mask.

    Args:
        tokenizer (Any): The tokenizer.
        messages (List[dict]): Messages with `role` and `content` fields.
        tools (Optional[List[dict]]): The list of tool dictionaries.
        chat_template (str): The chat template with `{% generation %}` symbol.

    Returns:
        `torch.Tensor`: The token_ids (sequence_length)
        `torch.Tensor`: Assistant_masks (sequence_length).
        `int`: Prompt length.

    Note:
        This method is based on the assumption that as the number of chat rounds increases,
        the tokens of the previous round are exactly the prefix tokens of the next round.
        If the assumption is not met, the function may produce incorrect results.
        Please check the chat template before using this method.
    """

    tokens = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        padding=False,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    assistant_token_mask = torch.zeros(tokens.shape[1], dtype=torch.int)
    for idx, message in enumerate(messages):
        if message["role"] == "assistant":
            prompt_token_ids = tokenizer.apply_chat_template(
                messages[:idx],
                tools=tools,
                chat_template=chat_template,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_length = prompt_token_ids.shape[1]
            prompt_response_token_ids = tokenizer.apply_chat_template(
                messages[: idx + 1],
                tools=tools,
                chat_template=chat_template,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_response_length = prompt_response_token_ids.shape[1]
            assistant_token_mask[prompt_length:prompt_response_length] = 1
    prompt_length = torch.argmax(assistant_token_mask).item()
    return tokens[0], assistant_token_mask, prompt_length


def get_action_mask_method(chat_template: Optional[str] = None) -> Callable:
    """Get the action mask method according to the chat template.

    Args:
        chat_template (str): The chat template. If { % generation % } is present, use HF tokenizer's `return_assistant_tokens_mask`.

    Returns:
        The action mask method.
    """
    if chat_template is None:
        return tokenize_and_mask_messages_default
    # check if the chat template contains `{% generation %}` symbol
    elif re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template):
        return tokenize_and_mask_messages_hf
    else:
        return tokenize_and_mask_messages_default


def get_checkpoint_dir_with_step_num(
    checkpoint_root_path: str,
    trainer_type: str = "verl",
    step_num: Optional[int] = None,
    raise_error: bool = True,
) -> Tuple[str, int]:
    """Get the checkpoint directory from a root checkpoint directory.

    Args:
        checkpoint_root_path (str): The root checkpoint directory.
        trainer_type (str): The trainer type. Only support "verl" for now.
        step_num (Optional[int], optional): The step number. If specified,
            load the checkpoint with the specified step number. If None,
            load the latest checkpoint. Defaults to None.
        raise_error (bool): Whether to raise an error if the checkpoint does not exist.

    Returns:
        Tuple[str, int]: The checkpoint directory and the step number of the checkpoint.
            If the checkpoint does not exist and `raise_error` is False, return (None, 0).
    """
    if trainer_type == "verl":
        return get_verl_checkpoint_info(
            checkpoint_path=checkpoint_root_path, step_num=step_num, raise_error=raise_error
        )
    else:
        raise NotImplementedError(f"Unsupported trainer type {trainer_type}")


def get_latest_state_dict(
    checkpoint_root_path: str,
    trainer_type: str = "verl",
) -> Tuple[str, int]:
    """Get the latest state dict from a root checkpoint directory.

    Args:
        checkpoint_root_path (str): The root checkpoint directory.

    Returns:
        Tuple[str, int]: The state dict path and the iteration of the state dict.
            If the state dict does not exist, return (None, 0).
    """
    if trainer_type != "verl":
        raise NotImplementedError(f"Unsupported trainer type {trainer_type}")
    latest_state_dict_iteration_path = os.path.join(
        checkpoint_root_path, "latest_state_dict_iteration.txt"
    )
    if os.path.exists(latest_state_dict_iteration_path):
        with open(latest_state_dict_iteration_path, "r", encoding="utf-8") as f:
            iteration = f.read().strip()
            state_dict_path = os.path.join(
                checkpoint_root_path, f"global_step_{iteration}", "actor"
            )
            return state_dict_path, int(iteration)
    return None, 0  # type: ignore


def load_state_dict(checkpoint_dir: str, config: TrainerConfig) -> Union[dict, Tuple[str, str]]:
    """Load state dict from a checkpoint dir.

    Args:
        checkpoint_dir (str): The checkpoint directory.
        trainer_type (str): The trainer type. Only support "verl" for now.

    Returns:
        Union[dict, Tuple[str, str]]: The state dict. If the checkpoint uses
            megatron dist checkpointing, return a tuple of (method, checkpoint_dir).
    """
    if config.trainer_type == "verl":
        strategy = config.trainer_strategy
        if strategy in {"fsdp", "fsdp2"}:
            return load_fsdp_state_dict_from_verl_checkpoint(checkpoint_dir)
        elif strategy == "megatron":
            actor_config = config.trainer_config.actor_rollout_ref.actor
            if (
                actor_config.megatron.use_dist_checkpointing
                or not actor_config.megatron.use_mbridge
            ):
                return "megatron", checkpoint_dir
            else:  # hf checkpointing
                return load_huggingface_state_dict(os.path.join(checkpoint_dir, "huggingface"))
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    else:
        raise NotImplementedError(f"Unsupported trainer type {config.trainer_type}")


def get_verl_checkpoint_info(
    checkpoint_path: str, step_num: Optional[int] = None, raise_error: bool = True
) -> Tuple[str, int]:
    """Get the checkpoint directory from a Verl root checkpoint directory.

    Args:
        checkpoint_path (str): The root checkpoint directory.
        step_num (Optional[int], optional): The step number. If specified,
            load the checkpoint with the specified step number. If None,
            load the latest checkpoint. Defaults to None.
        raise_error (bool): Whether to raise an error if the checkpoint does not exist.

    Returns:
        Tuple[str, int]: The checkpoint directory and the step number of the checkpoint.
    """
    if step_num is None:
        # load latest checkpoint
        iteration_file = os.path.join(checkpoint_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(iteration_file):
            with open(
                iteration_file, "r", encoding="utf-8"
            ) as f:  # TODO: this file may be modified simultaneously
                iteration = f.read().strip()
                return os.path.join(checkpoint_path, f"global_step_{iteration}"), int(iteration)
        elif raise_error:
            raise FileNotFoundError(f"No iteration file found in {checkpoint_path}")
        else:
            return None, 0  # type: ignore
    else:
        # load specific iteration checkpoint
        path = os.path.join(checkpoint_path, f"global_step_{step_num}")
        if not os.path.exists(path) and raise_error:
            raise FileNotFoundError(f"Checkpoint {path} not found")
        return path, step_num


# modified from verl/model_merger/fsdp_model_merger.py
def load_fsdp_state_dict_from_verl_checkpoint(checkpoint_path: str) -> dict:  # noqa: C901
    """Load state dict from a Verl checkpoint."""
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    logger = get_logger(__name__)
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        trust_remote_code=False,
        is_value_model=False,
        local_dir=checkpoint_path,
        hf_model_config_path=os.path.join(checkpoint_path, "huggingface"),
    )
    merger = FSDPModelMerger(config)

    world_size = merger._get_world_size()
    rank_zero_state_dict = merger._load_rank_zero_state_dict(world_size)

    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank_zero_state_dict, world_size)
    logger.info(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    logger.info(f"Processing model shards with {total_shards} {mesh_shape} in total")

    merged_state_dict = merger._load_and_merge_state_dicts(
        world_size, total_shards, mesh_shape, mesh_dim_names
    )
    return merged_state_dict


def load_huggingface_state_dict(checkpoint_path: str):
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint_path)
    return model.state_dict()


def get_megatron_converter(checkpoint_path: str):
    import builtins
    from contextlib import contextmanager

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.megatron_model_merger import MegatronModelMerger

    # modified from verl/model_merger/megatron_model_merger.py
    class MegatronStateDictConverter(MegatronModelMerger):
        def __init__(self, config: ModelMergerConfig):
            torch.distributed.init_process_group = lambda *args, **kwargs: None
            torch.distributed.get_rank = lambda: 0
            torch.distributed.get_world_size = lambda: 1
            self.logger = get_logger(__name__)
            with self._redirect_print_to_logger():
                super().__init__(config)

        @contextmanager
        def _redirect_print_to_logger(self):
            original_print = builtins.print

            def logger_print(*args, **kwargs):
                message = " ".join(str(arg) for arg in args)
                self.logger.debug(message)

            builtins.print = logger_print
            try:
                yield
            finally:
                builtins.print = original_print

        def get_state_dict(self, checkpoint_path):
            self.config.local_dir = checkpoint_path
            from verl.utils.megatron_utils import get_dist_checkpoint_path

            with self._redirect_print_to_logger():
                model_ckpt_path = get_dist_checkpoint_path(self.config.local_dir)

                model_state_dict = self._load_state_dicts(model_ckpt_path)
                merged_state_dict = self._merge_state_dicts(model_state_dict)
            del model_state_dict
            return merged_state_dict

    config = ModelMergerConfig(
        operation="merge",
        backend="megatron",
        tie_word_embedding=False,
        trust_remote_code=False,
        is_value_model=False,
        local_dir=checkpoint_path,
        hf_model_config_path=os.path.join(checkpoint_path, "huggingface"),
        use_cpu_initialization=True,
    )
    converter = MegatronStateDictConverter(config)
    return converter
