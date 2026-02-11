import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from omegaconf import OmegaConf
from verl.workers.config import PolicyLossConfig, RouterReplayConfig

from trinity.algorithm import ALGORITHM_TYPE
from trinity.common.config import Config, SynchronizerConfig, set_if_none
from trinity.common.constants import EXPLORER_NAME
from trinity.common.patch import kimi_vl_monkey_patch_decorator
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class Data:
    train_batch_size: int = 1024  # kept to pass RayPPOTrainer._validate_config
    trust_remote_code: bool = False


@dataclass
class FusedKernelOptions:
    impl_backend: Optional[str] = None


@dataclass
class ActorModel:
    path: str = ""
    external_lib: Optional[str] = None
    override_config: Dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = True
    use_fused_kernels: bool = False
    fused_kernel_options: FusedKernelOptions = field(default_factory=FusedKernelOptions)
    custom_chat_template: Optional[str] = None
    enable_activation_offload: bool = False
    use_shm: bool = False
    trust_remote_code: bool = False  # Whether to enable loading a remote code model

    # lora configs
    lora_rank: int = 0  # The rank of the LoRA model, default to 0. If lora_rank > 0, LoRA module is enabled in trainer
    lora_alpha: int = 32
    target_modules: Optional[str] = "all-linear"
    exclude_modules: Optional[str] = None
    lora_adapter_path: Optional[str] = None

    # rope configs
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None


@dataclass
class Optim:
    # For actor, most fields are set in algorithm.optimizer
    # For critic, you can set trainer_config.critic.optim
    optimizer: str = "adam"
    optimizer_impl: str = "torch.optim"
    lr: float = 1e-6
    lr_warmup_steps: int = -1
    lr_warmup_steps_ratio: float = 0.0
    min_lr_ratio: Optional[float] = 0.0
    warmup_style: Optional[str] = None  # deprecated !
    lr_scheduler_type: str = "constant"
    total_training_steps: int = -1  # ! DO NOT SET, use trainer.total_steps
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    clip_grad: float = 1.0
    lr_warmup_init: Optional[float] = None  # 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: Optional[str] = None  # "constant"
    min_lr: Optional[float] = None  # 0.0
    weight_decay: float = 0.01
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


@dataclass
class WrapPolicy:
    min_num_params: int = 0


@dataclass
class FSDPConfig:
    _target_: str = "verl.workers.config.FSDPEngineConfig"  # DO NOT SET
    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False
    reshard_after_forward: bool = True
    wrap_policy: WrapPolicy = field(default_factory=WrapPolicy)
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: Optional[str] = None
    dtype: str = "bfloat16"
    mixed_precision: dict = field(default_factory=dict)


@dataclass
class Checkpoint:
    load_contents: List[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    save_contents: List[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False  # TODO: testing async save


@dataclass
class OverrideTransformerConfig:
    recompute_granularity: Optional[str] = "full"
    recompute_modules: List[str] = field(default_factory=lambda: ["core_attn"])
    recompute_method: Optional[str] = "uniform"
    recompute_num_layers: Optional[int] = 1


@dataclass
class MegatronConfig:
    param_offload: bool = False
    grad_offload: bool = False
    optimizer_offload: bool = False
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: Optional[str] = None
    seed: int = 42
    override_ddp_config: dict = field(default_factory=dict)
    override_transformer_config: OverrideTransformerConfig = field(
        default_factory=OverrideTransformerConfig
    )
    use_mbridge: bool = False
    dtype: str = "bfloat16"
    use_remove_padding: bool = True


@dataclass
class ProfileConfig:
    use_profile: bool = False
    profile_ranks: Optional[List[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None


@dataclass
class Actor:
    strategy: Optional[str] = None
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: int = 1
    use_dynamic_bsz: Optional[bool] = None
    ppo_max_token_len_per_gpu: Optional[int] = None
    fix_actor_microbatch_loss_scale: Optional[bool] = None  # EXPERIMENTAL
    grad_clip: Optional[float] = None
    ppo_epochs: int = 1
    shuffle: bool = False
    ulysses_sequence_parallel_size: Optional[int] = None
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    optim: Optim = field(default_factory=Optim)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    profiler: dict = field(default_factory=dict)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    # do not set
    loss_agg_mode: str = "token-mean"
    clip_ratio: float = 0.2
    clip_ratio_low: Optional[float] = None
    clip_ratio_high: Optional[float] = None
    entropy_coeff: float = 0.001
    use_kl_loss: bool = False


@dataclass
class Ref:
    strategy: Optional[str] = None
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: int = 1
    log_prob_use_dynamic_bsz: Optional[bool] = None
    log_prob_max_token_len_per_gpu: Optional[int] = None
    ulysses_sequence_parallel_size: Optional[int] = None
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    checkpoint: Checkpoint = field(
        default_factory=lambda: Checkpoint(load_contents=["model"], save_contents=["model"])
    )
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    load_weight: bool = True
    profiler: dict = field(default_factory=dict)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)


@dataclass
class _ValKwargs:
    do_sample: bool = False


@dataclass
class _MultiTurn:
    enable: bool = False


@dataclass
class Rollout:
    # do not set
    val_kwargs: _ValKwargs = field(default_factory=_ValKwargs)
    multi_turn: _MultiTurn = field(default_factory=_MultiTurn)
    temperature: float = 1.0
    n: int = 1  # > 1 for grpo
    log_prob_use_dynamic_bsz: Optional[bool] = None
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_max_token_len_per_gpu: Optional[int] = None


@dataclass
class ActorRolloutRef:
    hybrid_engine: bool = True
    model: ActorModel = field(default_factory=ActorModel)
    actor: Actor = field(default_factory=Actor)
    ref: Ref = field(default_factory=Ref)
    rollout: Rollout = field(default_factory=Rollout)
    nccl_timeout: float = 600  # ! DO NOT SET, it will be set by `config.synchronizer.sync_timeout`
    synchronizer: Optional[SynchronizerConfig] = None
    explorer_name: str = EXPLORER_NAME


@dataclass
class CriticModel:
    path: str = ""
    tokenizer_path: str = ""
    override_config: Dict[str, str] = field(default_factory=dict)
    external_lib: Optional[str] = None
    trust_remote_code: bool = False  # Whether to enable loading a remote code model
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = True
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)

    # rope configs
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None


@dataclass
class Critic:
    enable: bool = False
    strategy: Optional[str] = None
    optim: Optim = field(default_factory=Optim)
    model: CriticModel = field(default_factory=CriticModel)
    ppo_mini_batch_size: int = 0
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: int = 1
    forward_micro_batch_size: Optional[int] = None
    forward_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: Optional[bool] = None
    ppo_max_token_len_per_gpu: Optional[int] = None
    forward_max_token_len_per_gpu: Optional[int] = None
    ulysses_sequence_parallel_size: Optional[int] = None
    ppo_epochs: int = 1
    shuffle: bool = False
    grad_clip: Optional[float] = None
    cliprange_value: float = 0.0
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    rollout_n: int = 1
    loss_agg_mode: str = "token-mean"
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    nccl_timeout: float = 600  # ! DO NOT SET, it will be set by `config.synchronizer.sync_timeout`
    ray_namespace: str = ""  # automatically generated
    profiler: dict = field(default_factory=dict)


@dataclass
class _RewardModel:
    input_tokenizer: Optional[str] = None
    path: str = ""
    external_lib: Optional[str] = None
    use_remove_padding: bool = False
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class RewardModel:
    enable: bool = False
    strategy: Optional[str] = None
    model: _RewardModel = field(default_factory=_RewardModel)
    micro_batch_size_per_gpu: int = 1
    max_length: Optional[int] = None
    ulysses_sequence_parallel_size: int = 1
    use_dynamic_bsz: bool = False
    forward_max_token_len_per_gpu: int = 0
    reward_manager: str = "naive"
    use_reward_loop: bool = True


@dataclass
class CustomRewardFunction:
    path: Optional[str] = None
    name: str = "compute_score"


@dataclass
class KL_Ctrl:
    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: float = 10000
    target_kl: float = 0.1


@dataclass
class RolloutCorrection:
    rollout_is: Optional[str] = None
    rollout_is_threshold: float = 2.0
    rollout_rs: Optional[str] = None
    rollout_rs_threshold: Optional[float] = None
    rollout_rs_threshold_lower: Optional[float] = None
    rollout_token_veto_threshold: Optional[float] = None
    # Because rollout and training in Trinity runs separately,
    # rollout_is_batch_normalize is default to True
    bypass_mode: bool = True
    loss_type: str = "ppo_clip"
    rollout_is_batch_normalize: bool = False


@dataclass
class Algorithm:
    rollout_correction: RolloutCorrection = field(default_factory=RolloutCorrection)
    # ! DO NOT SET gamma or lam below; they are kept here merely for compatibility with verl,
    # and their values will be overwritten by those in AlgorithmConfig.advantage_fn_args
    # if they are really needed (e.g., for GAE advantage/returns computation)
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KL_Ctrl = field(default_factory=KL_Ctrl)


@dataclass
class Trainer:
    balance_batch: bool = True
    total_epochs: int = 30
    total_training_steps: Optional[
        int
    ] = None  # ! DO NOT SET, use trainer.total_steps in global_config
    project_name: str = ""
    group_name: str = ""
    experiment_name: str = ""
    logger: List[str] = field(default_factory=list)
    val_generations_to_log_to_wandb: int = 0
    nnodes: int = 0
    n_gpus_per_node: int = 0
    save_freq: int = 0
    resume_mode: str = "auto"
    resume_from_path: str = ""
    test_freq: int = 0
    critic_warmup: int = 0
    default_hdfs_dir: Optional[str] = None
    remove_previous_ckpt_in_save: bool = False  # deprecated
    del_local_ckpt_after_load: bool = False
    default_local_dir: str = ""
    val_before_train: bool = False
    training_rollout_mode: str = "parallel"
    enable_exp_buffer: bool = True
    sync_freq: int = 0
    max_actor_ckpt_to_keep: Optional[int] = None
    max_critic_ckpt_to_keep: Optional[int] = None
    device: str = "cuda"  # default to cuda


@dataclass
class veRLConfig:
    data: Data = field(default_factory=Data)
    actor_rollout_ref: ActorRolloutRef = field(default_factory=ActorRolloutRef)
    critic: Critic = field(default_factory=Critic)
    reward_model: RewardModel = field(default_factory=RewardModel)
    custom_reward_function: CustomRewardFunction = field(default_factory=CustomRewardFunction)
    algorithm: Algorithm = field(default_factory=Algorithm)
    trainer: Trainer = field(default_factory=Trainer)
    global_profiler: dict = field(default_factory=dict)
    synchronizer: Optional[SynchronizerConfig] = None
    enable_preview: bool = True

    @kimi_vl_monkey_patch_decorator
    def _check_parallel_config(
        self,
        obj: Union[Actor, Ref, Critic],
        component_name: str,
        model_config: Union[ActorModel, CriticModel],
        fsdp_config: FSDPConfig,
        world_size: int,
        sp_attr: str = "ulysses_sequence_parallel_size",
    ) -> None:
        if obj.strategy.startswith("fsdp"):
            # check sequence parallelism
            sp_size = getattr(obj, sp_attr, 1)
            if world_size % sp_size != 0:
                raise ValueError(
                    f"The number of trainer GPUs ({world_size}) must be "
                    f"divisible by `{component_name}.{sp_attr}` ({sp_size}). "
                    f"Please change `trainer.{sp_attr}` or "
                    f"`{component_name}.{sp_attr}` in verl config to a reasonable value."
                )

            try:
                import transformers

                hf_config = transformers.AutoConfig.from_pretrained(
                    model_config.path, trust_remote_code=model_config.trust_remote_code
                )
                num_attention_heads = hf_config.num_attention_heads
            except Exception:
                num_attention_heads = None

            if num_attention_heads and num_attention_heads % sp_size != 0:
                raise ValueError(
                    f"The number of attention heads ({num_attention_heads}) must be "
                    f"divisible by `trainer.ulysses_sequence_parallel_size` ({sp_size})."
                    f"Please change `trainer.{sp_attr}` or "
                    f"`{component_name}.{sp_attr}` in verl config to a reasonable value."
                )

            fsdp_size = fsdp_config.fsdp_size
            if fsdp_size <= 0 or fsdp_size >= world_size:
                fsdp_size = fsdp_config.fsdp_size = world_size
            if world_size % fsdp_size != 0:
                raise ValueError(
                    f"The number of GPUs ({world_size}) must be "
                    f"divisible by `{component_name}.fsdp_config.fsdp_size` ({fsdp_size}). "
                    f"Please change `{component_name}.fsdp_config.fsdp_size` in verl config "
                    f"to a reasonable value."
                )
        else:
            # TODO: add check for megatron strategy
            pass

    def _adjust_token_len_if_needed(
        self,
        obj,
        config: Config,
        component_name: str,
        token_len_attr: str = "ppo_max_token_len_per_gpu",
        sp_attr: str = "ulysses_sequence_parallel_size",
    ) -> None:
        """
        Helper to adjust token length per GPU if current setting is too small.

        Ensures: token_len * seq_parallel >= config.model.max_model_len * 2
        """
        current_token_len = getattr(obj, token_len_attr)
        seq_parallel = getattr(obj, sp_attr)
        required_min = config.model.max_model_len * 2  # type: ignore

        if current_token_len * seq_parallel < required_min:
            new_token_len = math.ceil(required_min / seq_parallel)
            setattr(obj, token_len_attr, new_token_len)
            logger.warning(
                f"{component_name}.{token_len_attr} is automatically set to {new_token_len} "
                f"to match model.max_model_len ({config.model.max_model_len}). If you face OOM issues, "
                "please set `model.max_model_len` to a smaller value."
            )

    def synchronize_config(self, config: Config) -> None:  # noqa: C901
        """Synchronize config."""
        # Trainer Config
        self.trainer.nnodes = config.cluster.trainer_node_num
        self.trainer.n_gpus_per_node = config.cluster.trainer_gpu_num_per_node
        self.trainer.total_training_steps = config.trainer.total_steps or sys.maxsize
        self.trainer.sync_freq = config.synchronizer.sync_interval
        self.trainer.save_freq = config.trainer.save_interval
        self.trainer.project_name = config.project
        self.trainer.group_name = config.group
        self.trainer.experiment_name = config.name
        self.trainer.default_local_dir = config.checkpoint_job_dir
        if config.trainer.max_checkpoints_to_keep is not None:
            self.trainer.max_actor_ckpt_to_keep = config.trainer.max_checkpoints_to_keep
            self.trainer.max_critic_ckpt_to_keep = config.trainer.max_checkpoints_to_keep
        if not config.continue_from_checkpoint:
            self.trainer.resume_mode = "disable"
        else:
            self.trainer.resume_mode = "auto"

        # kept to pass RayPPOTrainer._validate_config
        self.data.train_batch_size = config.buffer.train_batch_size
        self.data.trust_remote_code = config.model.trust_remote_code

        self.synchronizer = config.synchronizer
        self.actor_rollout_ref.nccl_timeout = config.synchronizer.sync_timeout
        self.actor_rollout_ref.synchronizer = config.synchronizer
        self.actor_rollout_ref.explorer_name = config.explorer.name
        algorithm = ALGORITHM_TYPE.get(config.algorithm.algorithm_type)
        self.critic.enable = algorithm.use_critic
        self.critic.nccl_timeout = config.synchronizer.sync_timeout
        self.critic.ray_namespace = config.synchronizer.ray_namespace

        # Actor / Rollout Config
        actor_config = self.actor_rollout_ref.actor
        rollout_config = self.actor_rollout_ref.rollout
        actor_model_config = self.actor_rollout_ref.model
        actor_optim = actor_config.optim
        actor_model_config.path = config.model.model_path
        for attr in ["trust_remote_code", "custom_chat_template", "rope_scaling", "rope_theta"]:
            setattr(actor_model_config, attr, getattr(config.model, attr))
        actor_optim.total_training_steps = self.trainer.total_training_steps
        actor_config.ppo_mini_batch_size = config.buffer.train_batch_size
        rollout_config.temperature = (
            config.buffer.explorer_input.tasksets[0].rollout_args.temperature
            if config.buffer.explorer_input.tasksets
            else 1.0
        )
        rollout_config.n = config.algorithm.repeat_times
        for actor_attr, trainer_attr in [
            ("grad_clip",) * 2,
            ("use_dynamic_bsz",) * 2,
            ("fix_actor_microbatch_loss_scale",) * 2,
            ("ulysses_sequence_parallel_size",) * 2,
            ("ppo_max_token_len_per_gpu", "max_token_len_per_gpu"),
            ("strategy", "trainer_strategy"),
        ]:
            set_if_none(actor_config, actor_attr, getattr(config.trainer, trainer_attr))
        self._check_parallel_config(
            obj=actor_config,
            component_name="actor",
            model_config=actor_model_config,
            fsdp_config=actor_config.fsdp_config,
            world_size=config.cluster.trainer_gpu_num,
        )
        self._adjust_token_len_if_needed(
            obj=self.actor_rollout_ref.actor,
            config=config,
            component_name="actor",
        )

        # Ref Config
        ref_config = self.actor_rollout_ref.ref
        for ref_attr, trainer_attr in [
            ("log_prob_use_dynamic_bsz", "use_dynamic_bsz"),
            ("log_prob_max_token_len_per_gpu", "max_token_len_per_gpu"),
            ("ulysses_sequence_parallel_size",) * 2,
            ("strategy", "trainer_strategy"),
        ]:
            set_if_none(ref_config, ref_attr, getattr(config.trainer, trainer_attr))
        self._check_parallel_config(
            obj=ref_config,
            component_name="ref",
            model_config=actor_model_config,
            fsdp_config=ref_config.fsdp_config,
            world_size=config.cluster.trainer_gpu_num,
        )
        self._adjust_token_len_if_needed(
            obj=self.actor_rollout_ref.ref,
            config=config,
            component_name="ref",
            token_len_attr="log_prob_max_token_len_per_gpu",
        )

        # Critic config
        critic_optim = self.critic.optim
        self.critic.model.path = config.model.critic_model_path
        self.critic.model.tokenizer_path = config.model.critic_model_path
        self.critic.model.rope_scaling = config.model.rope_scaling
        self.critic.model.rope_theta = config.model.rope_theta
        self.critic.ppo_mini_batch_size = config.buffer.train_batch_size
        self.critic.rollout_n = config.algorithm.repeat_times
        critic_optim.total_training_steps = self.trainer.total_training_steps
        for critic_attr, trainer_attr in [
            ("grad_clip",) * 2,
            ("use_dynamic_bsz",) * 2,
            ("ulysses_sequence_parallel_size",) * 2,
            ("strategy", "trainer_strategy"),
            ("ppo_max_token_len_per_gpu", "max_token_len_per_gpu"),
        ]:
            set_if_none(self.critic, critic_attr, getattr(config.trainer, trainer_attr))
        self._check_parallel_config(
            obj=self.critic,
            component_name="critic",
            model_config=self.critic.model,
            fsdp_config=self.critic.model.fsdp_config,
            world_size=config.cluster.trainer_gpu_num,
        )
        self._adjust_token_len_if_needed(
            obj=self.critic,
            config=config,
            component_name="critic",
        )
        set_if_none(
            self.critic, "forward_max_token_len_per_gpu", self.critic.ppo_max_token_len_per_gpu
        )

        # LoRA related config
        if config.model.lora_configs is not None:
            lora_config = config.model.lora_configs[0]
            for attr in ["lora_rank", "lora_alpha", "target_modules", "exclude_modules"]:
                setattr(actor_model_config, attr, getattr(lora_config, attr))
            if not lora_config.is_dummy:
                actor_model_config.lora_adapter_path = lora_config.path
            if actor_config.strategy not in ["fsdp", "fsdp2"]:
                logger.warning(
                    f"Lora is only supported for fsdp and fsdp2, but got {actor_config.strategy} instead, changed to fsdp."
                )
                actor_config.strategy = "fsdp"
            if self.critic.strategy not in ["fsdp", "fsdp2"]:
                logger.warning(
                    f"Lora is only supported for fsdp and fsdp2, but got {self.critic.strategy} instead, changed to fsdp."
                )
                self.critic.strategy = "fsdp"

        # Algorithm related config
        optim_config = config.algorithm.optimizer
        for field_name in optim_config.__dataclass_fields__:
            field_value = getattr(optim_config, field_name)
            if field_name == "optimizer_type":
                setattr(actor_optim, "optimizer", field_value)
            elif hasattr(actor_optim, field_name):
                setattr(actor_optim, field_name, field_value)
        # ensure megatron optimizer config compatibility
        set_if_none(actor_optim, "lr_warmup_init", optim_config.min_lr_ratio * optim_config.lr)
        set_if_none(actor_optim, "lr_decay_steps", self.trainer.total_training_steps)
        set_if_none(actor_optim, "lr_decay_style", optim_config.lr_scheduler_type)
        set_if_none(actor_optim, "min_lr", optim_config.min_lr_ratio * optim_config.lr)
        set_if_none(critic_optim, "lr_warmup_init", 0.0)
        set_if_none(critic_optim, "lr_decay_steps", self.trainer.total_training_steps)
        set_if_none(critic_optim, "lr_decay_style", "constant")
        set_if_none(critic_optim, "min_lr", 0.0)
        # fix optimizer type for fsdp
        if config.trainer.trainer_strategy.startswith("fsdp"):
            optim_map = {
                "adam": "AdamW",
                "adamw": "AdamW",
                "sgd": "SGD",
            }
            actor_optim.optimizer = optim_map.get(actor_optim.optimizer, actor_optim.optimizer)
            critic_optim.optimizer = optim_map.get(critic_optim.optimizer, critic_optim.optimizer)
        actor_config.use_kl_loss = config.algorithm.kl_loss_fn != "none"
        self.algorithm.use_kl_in_reward = config.algorithm.kl_penalty_fn != "none"
        # TODO (yanxi): it seems that adv_estimator now only affects whether use_critic is set to
        # True or False in RayPPOTrainer.__init__() (and hence in VerlPPOTrainerWrapper).
        # Need to double check whether this is indeed the case,
        # and see if adv_estimator can be removed completely.

        if config.algorithm.algorithm_type == "dpo":  # for DPO
            logger.warning("DPO micro batch size is doubled for computing loss.")
            actor_config.ppo_micro_batch_size_per_gpu *= 2
            ref_config.log_prob_micro_batch_size_per_gpu *= 2

        # check rollout config (only works for lora)
        for rollout_attr, actor_attr in [
            ("log_prob_use_dynamic_bsz", "use_dynamic_bsz"),
            ("log_prob_micro_batch_size", "ppo_micro_batch_size"),
            ("log_prob_micro_batch_size_per_gpu", "ppo_micro_batch_size_per_gpu"),
            ("log_prob_max_token_len_per_gpu", "ppo_max_token_len_per_gpu"),
        ]:
            set_if_none(rollout_config, rollout_attr, getattr(actor_config, actor_attr))

        # TODO: check other fields
        self.enable_preview = config.trainer.enable_preview


def load_config(config_path: str) -> veRLConfig:
    schema = OmegaConf.structured(veRLConfig)
    yaml_config = OmegaConf.load(config_path)
    try:
        config = OmegaConf.merge(schema, yaml_config)
        return OmegaConf.to_object(config)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e
