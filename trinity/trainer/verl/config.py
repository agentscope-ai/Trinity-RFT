# -*- coding: utf-8 -*-
"""veRL 0.8 configuration builder for Trinity-RFT.

This module provides `build_verl_config()`, the single entry point that
converts Trinity's `Config` (global_config) into the `DictConfig` required
by veRL 0.8's `ActorRolloutRefWorker` and `TrainingWorker`.

Design principle (P1):
  - VERLTrainer uses `global_config` for all Trinity-level logic.
  - `build_verl_config()` is called **once** to produce the minimal `DictConfig`
    needed at the Worker/Engine boundary.
  - Only fields that veRL workers/engines actually consume are included.

The DictConfig structure must match what `ActorRolloutRefWorker.__init__`
and `ActorRolloutRefWorker.init_model()` expect.  Every nested section
that corresponds to a `BaseConfig` subclass **must** contain a `_target_`
field pointing to the fully-qualified Python class path, because
`omega_conf_to_dataclass()` (Mode 1 — no `dataclass_type` argument) uses
`hydra.utils.instantiate()` which requires `_target_` for recursive
instantiation.

Config sections and their target types:
  config.model        → HFModelConfig
  config.actor        → FSDPActorConfig | McoreActorConfig
  config.ref          → FSDPActorConfig | McoreActorConfig  (subset)
  config.rollout      → RolloutConfig
  config.critic       → FSDPCriticConfig | McoreCriticConfig
  config.global_profiler → dict (plain dict, not a dataclass)
"""
from __future__ import annotations

import sys
from dataclasses import is_dataclass
from typing import List, Optional, Union, get_args, get_origin

from omegaconf import DictConfig, OmegaConf

from trinity.algorithm import ALGORITHM_TYPE
from trinity.common.config import Config
from trinity.common.config import OptimizerConfig as TrinityOptimizerConfig
from trinity.utils.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers for injecting `_target_` into config dicts
# ---------------------------------------------------------------------------


def _resolve_dc_type(type_hint):
    """Resolve the concrete BaseConfig dataclass type from a field annotation.

    Handles plain types, Optional[T], and Union[T, None].
    Returns None if the type is not a BaseConfig subclass.
    """
    from verl.base_config import BaseConfig

    # Direct dataclass type
    if (
        is_dataclass(type_hint)
        and isinstance(type_hint, type)
        and issubclass(type_hint, BaseConfig)
    ):
        return type_hint

    # Handle Optional[T] / Union[T, None]
    origin = get_origin(type_hint)
    if origin is Union:
        for arg in get_args(type_hint):
            if arg is type(None):
                continue
            if is_dataclass(arg) and isinstance(arg, type) and issubclass(arg, BaseConfig):
                return arg

    return None


def _inject_targets(config, dataclass_type, type_overrides=None, skip_fields=None):
    """Recursively inject `_target_` into a config dict based on the dataclass type hierarchy.

    Args:
        config: The config dict to inject _target_ into.
        dataclass_type: The verl dataclass type that this config corresponds to.
        type_overrides: Dict mapping field name → concrete dataclass type, for fields
            where the annotation type doesn't match the desired concrete type
            (e.g., ActorConfig.optim: OptimizerConfig → FSDPOptimizerConfig).
        skip_fields: Set of field names to skip (no _target_ injection).
            Used for fields like CriticConfig.model_config where we don't want
            Hydra to recursively instantiate the nested config.
    """
    if type_overrides is None:
        type_overrides = {}
    if skip_fields is None:
        skip_fields = set()

    # Set _target_ at this level
    config["_target_"] = f"{dataclass_type.__module__}.{dataclass_type.__name__}"

    # Walk all fields of the dataclass (including inherited fields)
    for f in dataclass_type.__dataclass_fields__.values():
        if f.name in skip_fields:
            continue
        if f.name not in config:
            continue

        # Determine the concrete type for this field
        if f.name in type_overrides:
            ft = type_overrides[f.name]
        else:
            ft = _resolve_dc_type(f.type)

        if ft is None:
            continue

        nested = config[f.name]
        if isinstance(nested, dict):
            _inject_targets(nested, ft)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_verl_config(global_config: Config) -> DictConfig:  # noqa: C901
    """Build the veRL 0.8 DictConfig from Trinity's global Config.

    This produces the *minimal* DictConfig that ActorRolloutRefWorker and
    TrainingWorker need.  All Trinity-level logic (algorithm, advantage,
    KL penalty, etc.) stays in VERLTrainer using global_config.

    The resulting DictConfig has the following top-level keys:
      - model:       HFModelConfig fields
      - actor:       FSDPActorConfig or McoreActorConfig fields
      - ref:         actor-style config for reference model
      - rollout:     RolloutConfig fields
      - critic:      CriticConfig fields (for VERLTrainer._init_workers)
      - global_profiler: plain dict
    """
    cfg = global_config
    strategy = cfg.trainer.trainer_strategy  # "fsdp", "fsdp2", or "megatron"

    total_training_steps = cfg.trainer.total_steps or sys.maxsize
    algorithm = ALGORITHM_TYPE.get(cfg.algorithm.algorithm_type)
    use_critic = algorithm.use_critic

    # ====================================================================
    # 1. Model config (HFModelConfig fields)
    # ====================================================================
    model = _build_model_config(cfg)

    # ====================================================================
    # 2. Actor config
    # ====================================================================
    actor = _build_actor_config(cfg, strategy, total_training_steps)

    # ====================================================================
    # 3. Ref config
    # ====================================================================
    ref = _build_ref_config(cfg, strategy)

    # ====================================================================
    # 4. Rollout config
    # ====================================================================
    rollout = _build_rollout_config(cfg)

    # ====================================================================
    # 5. Critic config
    # ====================================================================
    critic = _build_critic_config(cfg, strategy, use_critic, total_training_steps)

    # ====================================================================
    # 6. Global profiler
    # ====================================================================
    global_profiler = {
        "tool": None,
        "enable": False,
        "all_ranks": False,
        "ranks": [],
        "save_path": "outputs/profile",
        "tool_config": None,
        "global_tool_config": None,
        "steps": None,
        "profile_continuous_steps": False,
    }

    # ====================================================================
    # 7. Assemble the DictConfig
    # ====================================================================
    verl_dict = {
        "model": model,
        "actor": actor,
        "ref": ref,
        "rollout": rollout,
        "critic": critic,
        "global_profiler": global_profiler,
    }

    # Merge user overrides from trainer_config (if any)
    user_overrides = _normalize_trainer_config(cfg.trainer.trainer_config)
    if user_overrides:
        _strip_incompatible_engine_keys(user_overrides, strategy)
        _deep_merge(verl_dict, user_overrides, FROZEN_KEYS)

    return OmegaConf.create(verl_dict, flags={"allow_objects": True})


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_model_config(cfg: Config) -> dict:
    """Build HFModelConfig-compatible dict with `_target_`."""
    from verl.workers.config.model import HFModelConfig

    model = {
        "path": cfg.model.model_path,
        "use_shm": False,
        "trust_remote_code": cfg.model.trust_remote_code,
        # LoRA fields
        "lora_rank": 0,
        "lora_alpha": 16,
        "target_modules": "all-linear",
        "exclude_modules": None,
        "lora_adapter_path": None,
        # Other HFModelConfig fields
        "enable_gradient_checkpointing": True,
        "use_remove_padding": cfg.trainer.use_remove_padding,
        "use_fused_kernels": False,
        "fused_kernel_options": {},
        "custom_chat_template": cfg.model.custom_chat_template,
        "enable_activation_offload": False,
        "external_lib": None,
        "override_config": {},
        "mtp": {"enable": False},
    }

    # Apply LoRA config if present
    if cfg.model.lora_configs is not None:
        lora_config = cfg.model.lora_configs[0]
        model["lora_rank"] = lora_config.lora_rank
        model["lora_alpha"] = lora_config.lora_alpha
        model["target_modules"] = lora_config.target_modules
        model["exclude_modules"] = lora_config.exclude_modules
        if not lora_config.is_dummy:
            model["lora_adapter_path"] = lora_config.path

    # Rope config — only scalar overrides go through override_config.
    # rope_scaling (a nested dict) is NOT supported by veRL's update_model_config
    # and is not needed for training (the model's config.json is authoritative).
    # It is primarily relevant for the Explorer/vLLM inference side.
    if cfg.model.rope_theta is not None:
        model["override_config"]["rope_theta"] = cfg.model.rope_theta  # type: ignore

    _inject_targets(model, HFModelConfig)
    return model


def _build_actor_config(cfg: Config, strategy: str, total_training_steps: int) -> dict:
    """Build ActorConfig-compatible dict with `_target_`."""
    from verl.workers.config.actor import FSDPActorConfig, McoreActorConfig
    from verl.workers.config.engine import FSDPEngineConfig, McoreEngineConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig, McoreOptimizerConfig

    is_fsdp = strategy.startswith("fsdp")
    is_megatron = strategy.startswith("megatron")

    if is_fsdp:
        dc_type = FSDPActorConfig
        type_overrides = {
            "optim": FSDPOptimizerConfig,
            "engine": FSDPEngineConfig,
        }
    else:
        dc_type = McoreActorConfig
        type_overrides = {
            "optim": McoreOptimizerConfig,
            "engine": McoreEngineConfig,
        }

    actor = {
        "strategy": strategy,
        "ppo_mini_batch_size": cfg.buffer.train_batch_size,
        "ppo_micro_batch_size_per_gpu": None,
        "ppo_micro_batch_size": None,
        "use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "ppo_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "ppo_infer_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "entropy_coeff": 0,
        "use_kl_loss": cfg.algorithm.kl_loss_fn != "none",
        "kl_loss_coef": 0.001,
        "kl_loss_type": "low_var_kl",
        "ppo_epochs": 1,
        "shuffle": False,
        "data_loader_seed": 42,
        "loss_agg_mode": cfg.algorithm.loss_agg_mode or "token-mean",
        "loss_scale_factor": None,
        "use_prefix_grouper": False,
        "use_torch_compile": True,
        "freeze_vision_tower": False,
        "use_fused_kernels": False,
        "rollout_n": cfg.algorithm.repeat_times,
        "policy_loss": {
            "loss_mode": "vanilla",
            "rollout_correction": cfg.algorithm.rollout_correction or {"bypass_mode": True},
        },
        "router_replay": {"mode": "R3" if cfg.algorithm.enable_router_replay else "disabled"},
        "profiler": _build_profiler_config(),
        "checkpoint": _build_checkpoint_config(),
        "optim": _build_optimizer_config(cfg.algorithm.optimizer, strategy, total_training_steps),
    }

    # Strategy-specific fields
    router_replay_mode = "R3" if cfg.algorithm.enable_router_replay else "disabled"
    if is_fsdp:
        actor["grad_clip"] = cfg.trainer.grad_clip
        actor["ulysses_sequence_parallel_size"] = cfg.trainer.ulysses_sequence_parallel_size
        actor["entropy_from_logits_with_chunking"] = False
        actor["entropy_checkpointing"] = False
        actor["fsdp_config"] = _build_fsdp_engine_config(cfg, strategy, router_replay_mode)
        actor["use_remove_padding"] = cfg.trainer.use_remove_padding
        actor["use_rollout_log_probs"] = False
        actor["calculate_sum_pi_squared"] = False
    elif is_megatron:
        actor["megatron"] = _build_mcore_engine_config(cfg, router_replay_mode)
        actor["load_weight"] = True
        actor["use_rollout_log_probs"] = False

    _inject_targets(actor, dc_type, type_overrides=type_overrides)
    return actor


def _build_ref_config(cfg: Config, strategy: str) -> dict:
    """Build ref-config dict with `_target_` (subset of actor config)."""
    from verl.workers.config.actor import FSDPActorConfig, McoreActorConfig
    from verl.workers.config.engine import FSDPEngineConfig, McoreEngineConfig

    is_fsdp = strategy.startswith("fsdp")
    is_megatron = strategy.startswith("megatron")

    if is_fsdp:
        dc_type = FSDPActorConfig
        type_overrides = {"engine": FSDPEngineConfig}
    else:
        dc_type = McoreActorConfig
        type_overrides = {"engine": McoreEngineConfig}

    # NOTE: use log_prob_* naming — engine_workers.py renames these to ppo_*
    # before calling omega_conf_to_dataclass().
    ref = {
        "strategy": strategy,
        "rollout_n": cfg.algorithm.repeat_times,
        "log_prob_micro_batch_size_per_gpu": None,
        "log_prob_use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "log_prob_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "use_prefix_grouper": False,
        "profiler": _build_profiler_config(),
        "router_replay": {"mode": "disabled"},
        "checkpoint": _build_checkpoint_config(save_contents=["model"], load_contents=["model"]),
    }

    # Strategy-specific fields
    if is_fsdp:
        ref["ulysses_sequence_parallel_size"] = cfg.trainer.ulysses_sequence_parallel_size
        ref["entropy_from_logits_with_chunking"] = False
        ref["entropy_checkpointing"] = False
        ref["fsdp_config"] = {**_build_fsdp_engine_config(cfg, strategy), "forward_only": True}
        ref["use_remove_padding"] = cfg.trainer.use_remove_padding
        ref["use_rollout_log_probs"] = False
    elif is_megatron:
        ref["load_weight"] = True
        ref["use_rollout_log_probs"] = False
        ref["megatron"] = {**_build_mcore_engine_config(cfg), "forward_only": True}

    _inject_targets(ref, dc_type, type_overrides=type_overrides)
    return ref


def _build_rollout_config(cfg: Config) -> dict:
    """Build RolloutConfig-compatible dict with `_target_`."""
    from verl.workers.config.rollout import RolloutConfig

    # Get temperature from taskset or default
    temperature = 1.0
    if cfg.buffer.explorer_input.tasksets:
        temperature = cfg.buffer.explorer_input.tasksets[0].rollout_args.temperature

    rollout = {
        "name": "auto",
        "mode": "async",
        "temperature": temperature,
        "n": cfg.algorithm.repeat_times,
        # log prob settings mirror actor settings
        "log_prob_micro_batch_size": None,
        "log_prob_micro_batch_size_per_gpu": None,
        "log_prob_use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "log_prob_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        # Multi-turn / val (Trinity doesn't use these)
        "val_kwargs": {"do_sample": False},
        "multi_turn": {"enable": False},
        "checkpoint_engine": {
            "backend": "naive",
            "update_weights_bucket_megabytes": 2048,
            "engine_kwargs": {},
        },
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "enable_sleep_mode": True,
    }

    _inject_targets(rollout, RolloutConfig)
    return rollout


def _build_critic_config(
    cfg: Config, strategy: str, use_critic: bool, total_training_steps: int
) -> dict:
    """Build CriticConfig-compatible dict with `_target_`.

    Key design decisions for the critic config:
    - `model_config` is NOT a dataclass field in CriticConfig (it's only in
      _mutable_fields).  It cannot be passed to __init__ as a kwarg.  The
      trainer.py code creates HFModelConfig manually.
    - `model` is a dataclass field typed as `HFModelConfig = None`.  We omit
      it (let it default to None) to avoid Hydra instantiating an HFModelConfig
      during omega_conf_to_dataclass().
    - `engine` is inherited from CriticConfig and set by __post_init__:
        FSDPCriticConfig.__post_init__ sets  self.engine = self.fsdp
        McoreCriticConfig.__post_init__ sets self.engine = self.megatron
      So we include `fsdp` / `megatron` in the dict and let __post_init__
      wire up `engine` automatically.
    """
    from verl.workers.config.critic import FSDPCriticConfig, McoreCriticConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig, McoreOptimizerConfig

    is_fsdp = strategy.startswith("fsdp")
    is_megatron = strategy.startswith("megatron")

    if is_fsdp:
        dc_type = FSDPCriticConfig
        type_overrides = {
            "optim": FSDPOptimizerConfig,
        }
    else:
        dc_type = McoreCriticConfig
        type_overrides = {
            "optim": McoreOptimizerConfig,
        }

    critic = {
        "enable": use_critic,
        "strategy": strategy,
        "ppo_mini_batch_size": cfg.buffer.train_batch_size,
        "ppo_micro_batch_size_per_gpu": None,
        "ppo_micro_batch_size": None,
        "use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "ppo_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "ppo_infer_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "ppo_epochs": 1,
        "shuffle": True,
        "cliprange_value": 0.5,
        "loss_agg_mode": "token-mean",
        "data_loader_seed": 42,
        "rollout_n": cfg.algorithm.repeat_times,
        "profiler": _build_profiler_config(),
        "optim": _build_critic_optimizer_config(strategy, total_training_steps),
        "checkpoint": _build_checkpoint_config(),
    }

    # Strategy-specific fields
    if is_fsdp:
        critic["grad_clip"] = cfg.trainer.grad_clip
        critic["ulysses_sequence_parallel_size"] = cfg.trainer.ulysses_sequence_parallel_size
        critic["forward_micro_batch_size"] = 1
        critic["forward_micro_batch_size_per_gpu"] = 1
        critic["forward_max_token_len_per_gpu"] = cfg.trainer.max_token_len_per_gpu
        # fsdp field — FSDPCriticConfig.__post_init__ sets self.engine = self.fsdp
        critic["fsdp"] = _build_fsdp_engine_config(cfg, strategy)
    elif is_megatron:
        critic["load_weight"] = True
        critic["nccl_timeout"] = 600
        # megatron field — McoreCriticConfig.__post_init__ sets self.engine = self.megatron
        critic["megatron"] = _build_mcore_engine_config(cfg)

    _inject_targets(critic, dc_type, type_overrides=type_overrides)
    return critic


# ---------------------------------------------------------------------------
# Sub-config builders (return plain dicts — _inject_targets handles _target_)
# ---------------------------------------------------------------------------


def _build_fsdp_engine_config(
    cfg: Config, strategy: str, router_replay_mode: str = "disabled"
) -> dict:
    """Build FSDPEngineConfig-compatible dict."""
    return {
        "param_offload": cfg.trainer.param_offload,
        "optimizer_offload": cfg.trainer.optimizer_offload,
        "offload_policy": cfg.trainer.offload_policy,
        "reshard_after_forward": True,
        "wrap_policy": {"min_num_params": 0},
        "fsdp_size": -1,
        "forward_prefetch": False,
        "model_dtype": "fp32",
        "dtype": "bfloat16",
        "mixed_precision": {},
        "ulysses_sequence_parallel_size": cfg.trainer.ulysses_sequence_parallel_size,
        "strategy": strategy,  # "fsdp" or "fsdp2"
        "use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "use_remove_padding": cfg.trainer.use_remove_padding,
        "use_fused_kernels": False,
        "router_replay": {"mode": router_replay_mode},
    }


def _build_mcore_engine_config(cfg: Config, router_replay_mode: str = "disabled") -> dict:
    """Build McoreEngineConfig-compatible dict."""
    mg = cfg.trainer.megatron
    return {
        "strategy": "megatron",
        "param_offload": cfg.trainer.param_offload,
        "optimizer_offload": cfg.trainer.optimizer_offload,
        "grad_offload": cfg.trainer.grad_offload,
        "forward_only": False,
        "dtype": "bfloat16",
        "use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "use_remove_padding": cfg.trainer.use_remove_padding,
        "use_fused_kernels": False,
        "seed": 42,
        # Mcore-specific parallelism
        "tensor_model_parallel_size": mg.tensor_model_parallel_size,
        "expert_model_parallel_size": mg.expert_model_parallel_size,
        "expert_tensor_parallel_size": mg.expert_tensor_parallel_size,
        "pipeline_model_parallel_size": mg.pipeline_model_parallel_size,
        "virtual_pipeline_model_parallel_size": mg.virtual_pipeline_model_parallel_size,
        "context_parallel_size": mg.context_parallel_size,
        "sequence_parallel": mg.sequence_parallel,
        "use_distributed_optimizer": True,
        "use_dist_checkpointing": False,
        "dist_checkpointing_path": None,
        "dist_ckpt_optim_fully_reshardable": False,
        "distrib_optim_fully_reshardable_mem_efficient": False,
        "use_mbridge": True,
        "vanilla_mbridge": True,
        "override_ddp_config": {},
        "override_transformer_config": {
            "recompute_granularity": "full",
            "recompute_modules": ["core_attn"],
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
        "override_mcore_model_config": {},
        "router_replay": {"mode": router_replay_mode},
    }


def _build_optimizer_config(
    trinity_optim: TrinityOptimizerConfig, strategy: str, total_training_steps: int
) -> dict:
    """Build veRL OptimizerConfig-compatible dict from Trinity's OptimizerConfig."""
    is_fsdp = strategy.startswith("fsdp")

    optim = {
        "lr": trinity_optim.lr,
        "lr_warmup_steps_ratio": trinity_optim.lr_warmup_steps_ratio,
        "lr_warmup_steps": trinity_optim.lr_warmup_steps,
        "total_training_steps": total_training_steps,
        "weight_decay": trinity_optim.weight_decay,
        "betas": list(trinity_optim.betas),
        "clip_grad": trinity_optim.clip_grad,
    }

    if is_fsdp:
        # FSDP uses FSDPOptimizerConfig
        optim["optimizer"] = _map_optimizer_name_fsdp(trinity_optim.optimizer_type)
        optim["optimizer_impl"] = "torch.optim"
        optim["min_lr_ratio"] = trinity_optim.min_lr_ratio
        optim["lr_scheduler_type"] = trinity_optim.lr_scheduler_type
        optim["num_cycles"] = 0.5
        optimizer_overrides = {
            key: value
            for key, value in {
                "fused": trinity_optim.fused,
                "foreach": trinity_optim.foreach,
            }.items()
            if value is not None
        }
        optim["override_optimizer_config"] = optimizer_overrides or None
        optim["zero_indexed_step"] = True
    else:
        # Megatron uses McoreOptimizerConfig
        optim["optimizer"] = trinity_optim.optimizer_type
        optim["lr_warmup_init"] = trinity_optim.min_lr_ratio * trinity_optim.lr
        optim["lr_decay_steps"] = total_training_steps
        optim["lr_decay_style"] = trinity_optim.lr_scheduler_type
        optim["min_lr"] = trinity_optim.min_lr_ratio * trinity_optim.lr
        optim["weight_decay_incr_style"] = "constant"
        optim["lr_wsd_decay_style"] = "exponential"
        optim["lr_wsd_decay_steps"] = None
        optim["use_checkpoint_opt_param_scheduler"] = False
        optim["override_optimizer_config"] = None

    return optim


def _build_critic_optimizer_config(strategy: str, total_training_steps: int) -> dict:
    """Build a default optimizer config for the critic model."""
    is_fsdp = strategy.startswith("fsdp")
    optim = {
        "lr": 1e-5,
        "lr_warmup_steps_ratio": 0.0,
        "lr_warmup_steps": -1,
        "total_training_steps": total_training_steps,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "clip_grad": 1.0,
    }
    if is_fsdp:
        optim["optimizer"] = "AdamW"
        optim["optimizer_impl"] = "torch.optim"
        optim["min_lr_ratio"] = 0.01
        optim["lr_scheduler_type"] = "constant"
        optim["num_cycles"] = 0.5
        optim["override_optimizer_config"] = None
        optim["zero_indexed_step"] = True
    else:
        optim["optimizer"] = "adam"
        optim["lr_warmup_init"] = 0.0
        optim["lr_decay_steps"] = total_training_steps
        optim["lr_decay_style"] = "constant"
        optim["min_lr"] = 0.0
        optim["weight_decay_incr_style"] = "constant"
        optim["lr_wsd_decay_style"] = "exponential"
        optim["lr_wsd_decay_steps"] = None
        optim["use_checkpoint_opt_param_scheduler"] = False
        optim["override_optimizer_config"] = None

    return optim


def _build_profiler_config() -> dict:
    """Build a disabled ProfilerConfig-compatible dict for worker-level profiler."""
    return {
        "tool": None,
        "enable": False,
        "all_ranks": False,
        "ranks": [],
        "save_path": "outputs/profile",
        "tool_config": None,
        "global_tool_config": None,
    }


def _build_checkpoint_config(
    save_contents: Optional[List[str]] = None,
    load_contents: Optional[List[str]] = None,
) -> dict:
    """Build CheckpointConfig-compatible dict."""
    if save_contents is None:
        save_contents = ["model", "optimizer", "extra"]
    if load_contents is None:
        load_contents = ["model", "optimizer", "extra"]
    return {
        "save_contents": save_contents,
        "load_contents": load_contents,
        "async_save": False,
        "mbridge_config": {
            "distributed_filesystem": True,
            "memory_efficient": True,
            "strict": False,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_optimizer_name_fsdp(name: str) -> str:
    """Map Trinity's optimizer name to the class name FSDPOptimizerConfig expects."""
    mapping = {
        "adam": "AdamW",
        "adamw": "AdamW",
        "sgd": "SGD",
    }
    return mapping.get(name, name)


# ---------------------------------------------------------------------------
# User trainer_config override support
# ---------------------------------------------------------------------------

# Keys derived from global_config — user overrides for these are ignored.
FROZEN_KEYS = {
    # model
    "model.path",
    "model.trust_remote_code",
    "model.use_remove_padding",
    "model.custom_chat_template",
    "model.lora_rank",
    "model.lora_alpha",
    "model.target_modules",
    "model.exclude_modules",
    "model.lora_adapter_path",
    # actor
    "actor.strategy",
    "actor.ppo_mini_batch_size",
    "actor.use_dynamic_bsz",
    "actor.ppo_max_token_len_per_gpu",
    "actor.ppo_infer_max_token_len_per_gpu",
    "actor.use_kl_loss",
    "actor.loss_agg_mode",
    "actor.rollout_n",
    "actor.grad_clip",
    "actor.ulysses_sequence_parallel_size",
    "actor.use_remove_padding",
    # ref
    "ref.strategy",
    "ref.rollout_n",
    "ref.log_prob_use_dynamic_bsz",
    "ref.log_prob_max_token_len_per_gpu",
    "ref.ulysses_sequence_parallel_size",
    "ref.use_remove_padding",
    # rollout
    "rollout.temperature",
    "rollout.n",
    "rollout.log_prob_use_dynamic_bsz",
    "rollout.log_prob_max_token_len_per_gpu",
    # critic
    "critic.enable",
    "critic.strategy",
    "critic.ppo_mini_batch_size",
    "critic.use_dynamic_bsz",
    "critic.ppo_max_token_len_per_gpu",
    "critic.ppo_infer_max_token_len_per_gpu",
    "critic.rollout_n",
    "critic.grad_clip",
    "critic.ulysses_sequence_parallel_size",
    "critic.forward_max_token_len_per_gpu",
}

_MERGEABLE_KEYS = ("model", "actor", "ref", "rollout", "critic")

# Engine config keys that are strategy-specific.
# FSDP actor/ref use "fsdp_config"; FSDP critic uses "fsdp".
# Megatron actor/ref/critic all use "megatron".
_FSDP_ENGINE_KEYS = ("fsdp_config", "fsdp")
_MEGATRON_ENGINE_KEYS = ("megatron",)


def _strip_incompatible_engine_keys(overrides: dict, strategy: str) -> None:
    """Remove engine config keys that don't belong to the active strategy.

    When ``trainer_config`` carries overrides from a different strategy
    (e.g. ``actor.megatron`` while the strategy is ``fsdp2``), the extra
    keys would fail at ``omega_conf_to_dataclass`` time because the target
    dataclass doesn't define them.
    """
    is_fsdp = strategy.startswith("fsdp")
    drop_keys = _MEGATRON_ENGINE_KEYS if is_fsdp else _FSDP_ENGINE_KEYS

    for section in ("actor", "ref", "critic"):
        sub = overrides.get(section)
        if isinstance(sub, dict):
            for key in drop_keys:
                if key in sub:
                    logger.warning(
                        "Stripping trainer_config override '%s.%s' — "
                        "incompatible with strategy '%s'",
                        section,
                        key,
                        strategy,
                    )
                    del sub[key]


def _normalize_trainer_config(trainer_config) -> dict:
    """Normalize user trainer_config to verl flat structure.

    Accepts the old ``veRLConfig`` layout (with ``actor_rollout_ref`` wrapper)
    or the flat layout (``model``/``actor``/``ref``/``rollout``/``critic`` at
    the top level).  Returns a plain dict with only the mergeable top-level
    keys.
    """
    if not trainer_config:
        return {}

    if hasattr(trainer_config, "to_container"):
        tc = OmegaConf.to_container(trainer_config, resolve=True)
    elif not isinstance(trainer_config, dict):
        tc = OmegaConf.to_container(OmegaConf.structured(trainer_config), resolve=True)
    else:
        tc = dict(trainer_config)

    result: dict = {}

    # Unwrap actor_rollout_ref → flat model/actor/ref/rollout
    if "actor_rollout_ref" in tc:
        arr = tc.pop("actor_rollout_ref")
        if isinstance(arr, dict):
            for key in _MERGEABLE_KEYS:
                if key in arr:
                    result[key] = arr[key]

    # Direct top-level keys (critic lives outside actor_rollout_ref)
    if "critic" in tc:
        result["critic"] = tc["critic"]

    # Also accept the flat format where user writes model/actor/… at top level
    for key in _MERGEABLE_KEYS:
        if key in tc and key not in result:
            result[key] = tc[key]

    return result


def _deep_merge(base: dict, overrides: dict, frozen: set, prefix: str = "") -> None:
    """Recursively deep-merge *overrides* into *base*, skipping frozen keys."""
    for key, value in overrides.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if full_key in frozen:
            logger.warning(
                "Ignoring trainer_config override for '%s' — "
                "this field is controlled by global_config",
                full_key,
            )
            continue

        if key == "_target_":
            continue

        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value, frozen, full_key)
        else:
            base[key] = value
