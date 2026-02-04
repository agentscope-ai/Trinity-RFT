# -*- coding: utf-8 -*-
"""Configs for RFT."""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from trinity.common.constants import (
    EXPLORER_NAME,
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    LOG_NODE_IP_ENV_VAR,
    PLUGIN_DIRS_ENV_VAR,
    TRAINER_NAME,
    PromptType,
    SaveStrategy,
    StorageType,
    SyncMethod,
    SyncStyle,
)
from trinity.utils.annotations import Experimental
from trinity.utils.log import get_logger

logger = get_logger(__name__)


def set_if_none(obj, attr, val):
    if getattr(obj, attr, None) is None:
        setattr(obj, attr, val)


@dataclass
class FormatConfig:
    """Configuration for data formatting"""

    # for sft / dpo
    prompt_type: PromptType = PromptType.MESSAGES

    # for plaintext input
    prompt_key: str = "prompt"  # user prompt
    response_key: str = "response"  # assistant response
    system_prompt_key: Optional[str] = None  # If set, use the provided system prompt
    system_prompt: Optional[str] = None  # has lower priority than system_prompt_key

    # for message list input
    messages_key: str = "message"

    # for tools
    tools_key: str = "tools"
    image_key: Optional[str] = None  # used for multi-modal data
    video_key: Optional[str] = None  # used for multi-modal data

    reply_prefix: Optional[str] = None

    # for sample-level task controlling
    workflow_key: str = ""
    reward_fn_key: str = ""

    # for dpo dataset
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"

    # for multi-turn sft
    enable_concatenated_multi_turn: bool = False

    # for sft / dpo, if None, use model.custom_chat_template
    chat_template: Optional[str] = None


@dataclass
class GenerationConfig:
    temperature: Optional[float] = None  # 1.0
    top_p: Optional[float] = None  # 1.0
    top_k: int = -1  # -1 means disabled
    logprobs: Optional[int] = None  # 0  # vLLM return `logprobs + 1` elements
    max_tokens: Optional[int] = None  # if None, use model.max_response_tokens
    # repeat each task for `n` times
    # ! DO NOT SET, it will be set by `algorithm.repeat_times` or `buffer.explorer_input.eval_tasksets[i].repeat_times`
    n: int = 1


@dataclass
class OptimizerConfig:
    lr: float = 1e-6
    lr_warmup_steps: int = -1
    lr_warmup_steps_ratio: float = 0.0
    min_lr_ratio: float = 0.0
    warmup_style: Optional[str] = None  # deprecated !
    lr_scheduler_type: str = "constant"
    optimizer_type: str = "adam"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.01
    clip_grad: float = 1.0


@dataclass
class LoRAConfig:
    """LoRA config, only effective for rollout model, not for auxiliary models."""

    name: Optional[str] = None
    path: Optional[str] = None
    base_model_name: Optional[str] = None
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dtype: str = "auto"
    target_modules: str = "all-linear"
    exclude_modules: Optional[str] = None
    is_dummy: bool = False  # DO NOT SET, automatically set


@Experimental
@dataclass
class TaskSelectorConfig:
    """Data selector config."""

    selector_type: Optional[str] = "sequential"

    # For shuffle
    seed: int = 42

    # Estimator Config
    feature_keys: List[str] = field(default_factory=lambda: [])
    kwargs: dict = field(default_factory=dict)


@dataclass
class ReplayBufferConfig:
    """Config for replay buffer used in StorageType.QUEUE."""

    enable: bool = False
    priority_fn: str = "linear_decay"
    reuse_cooldown_time: Optional[float] = None
    priority_fn_args: Dict = field(default_factory=lambda: {"decay": 2.0})


@dataclass
class OverRolloutConfig:
    """Config for over-rollout in explorer."""

    ratio: float = 0.0  # explorer will only wait for (1 - over_rollout.ratio) * batch_size of tasks at each step
    wait_after_min: float = 30.0  # wait 30 s after reaching minimum task threshold
    # more settings will be added in the future
    # e.g., postpone tasks into the next step if not finished in time


@dataclass
class DynamicTimeoutConfig:
    """Config for dynamic timeout in explorer."""

    enable: bool = False
    ratio: float = 3.0  # the timeout for each step will be min(max_timeout, average_time_per_task * dynamic_timeout.ratio)


@dataclass
class StorageConfig:
    """Storage config for both taskset and experience buffer.
    Not visible to users directly. Please use ExperienceBufferConfig or TasksetConfig instead.
    """

    name: str = ""
    storage_type: str = StorageType.FILE.value
    path: Optional[str] = None
    repeat_times: Optional[int] = None

    # For continuing training
    index: int = 0

    # used for StorageType.FILE
    split: str = "train"
    subset_name: Optional[str] = None
    format: FormatConfig = field(default_factory=FormatConfig)

    # used for StorageType.QUEUE
    capacity: int = 10000
    max_read_timeout: float = 1800
    replay_buffer: Optional[ReplayBufferConfig] = field(default_factory=ReplayBufferConfig)

    # used for StorageType.SQL
    max_retry_times: int = 3
    max_retry_interval: int = 1

    # used for rollout tasks
    default_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None
    rollout_args: GenerationConfig = field(default_factory=GenerationConfig)
    workflow_args: dict = field(default_factory=dict)
    reward_fn_args: dict = field(default_factory=dict)
    task_selector: TaskSelectorConfig = field(default_factory=TaskSelectorConfig)

    # enable progress bar (tqdm) for _HFBatchReader
    enable_progress_bar: Optional[bool] = False

    # get storage from existing experiment
    ray_namespace: Optional[str] = None

    # ! DO NOT SET except you know what you are doing
    wrap_in_ray: bool = True

    # ! DO NOT SET, automatically set
    schema_type: Optional[str] = None

    # ! DO NOT SET, automatically set from buffer.total_epochs
    total_epochs: int = 1  # automatically set

    # ! DO NOT SET, automatically set from buffer.total_steps
    total_steps: Optional[int] = None  # automatically set

    # ! DO NOT SET, automatically set from buffer.batch_size / train_batch_size
    batch_size: int = 0

    # ! DO NOT SET, automatically set from model.model_path
    tokenizer_path: Optional[str] = None

    # ! DO NOT SET,  automatically set corresponding to train/eval
    is_eval: bool = False


@dataclass
class TasksetConfig:
    name: str = ""
    storage_type: str = StorageType.FILE.value
    path: Optional[str] = None

    default_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None
    rollout_args: GenerationConfig = field(default_factory=GenerationConfig)
    workflow_args: dict = field(default_factory=dict)
    reward_fn_args: dict = field(default_factory=dict)
    task_selector: TaskSelectorConfig = field(default_factory=TaskSelectorConfig)

    # used for StorageType.FILE
    split: str = "train"
    subset_name: Optional[str] = None
    format: FormatConfig = field(default_factory=FormatConfig)

    # used for StorageType.SQL
    max_retry_times: int = 3
    max_retry_interval: int = 1

    enable_progress_bar: bool = False

    # ! This setting is only valid for `eval_taskset`; for other taskset, it will be overridden by `algorithm.repeat_times`.
    repeat_times: int = 1
    # ! DO NOT SET, automatically load from checkpoint
    index: int = 0
    # ! DO NOT SET, automatically set based on train/eval
    is_eval: bool = False
    # ! DO NOT SET, automatically set from buffer.batch_size
    batch_size: int = 0
    # ! DO NOT SET, automatically set from buffer.total_epochs
    total_epochs: int = 1  # automatically set
    # ! DO NOT SET, automatically set from buffer.total_steps
    total_steps: Optional[int] = None  # automatically set
    # ! DO NOT SET, automatically set form ray_namespace
    ray_namespace: Optional[str] = None

    def to_storage_config(self) -> StorageConfig:
        storage_config = StorageConfig(
            name=self.name,
            storage_type=self.storage_type,
            path=self.path,
            task_selector=self.task_selector,
            repeat_times=self.repeat_times,
            split=self.split,
            subset_name=self.subset_name,
            format=self.format,
            max_retry_times=self.max_retry_times,
            max_retry_interval=self.max_retry_interval,
            default_workflow_type=self.default_workflow_type,
            default_reward_fn_type=self.default_reward_fn_type,
            rollout_args=self.rollout_args,
            workflow_args=self.workflow_args,
            reward_fn_args=self.reward_fn_args,
            enable_progress_bar=self.enable_progress_bar,
            index=self.index,
            is_eval=self.is_eval,
            batch_size=self.batch_size,
            total_epochs=self.total_epochs,
            total_steps=self.total_steps,
            ray_namespace=self.ray_namespace,
        )
        return storage_config


@dataclass
class ExperienceBufferConfig:
    """Storage Config for trainer input experience buffer."""

    name: str = ""
    storage_type: str = StorageType.QUEUE.value
    path: Optional[str] = None

    # used for StorageType.QUEUE
    capacity: int = 10000
    max_read_timeout: float = 1800
    replay_buffer: Optional[ReplayBufferConfig] = field(default_factory=ReplayBufferConfig)

    # used for StorageType.SQL
    max_retry_times: int = 3
    max_retry_interval: int = 1

    # used for StorageType.FILE
    split: str = "train"
    subset_name: Optional[str] = None
    format: FormatConfig = field(default_factory=FormatConfig)
    enable_progress_bar: Optional[bool] = False

    # ! DO NOT SET, automatically set
    schema_type: Optional[str] = None
    # ! DO NOT SET
    index: int = 0
    # ! DO NOT SET, automatically set from buffer.batch_size
    batch_size: int = 0
    # ! DO NOT SET, automatically set from model.model_path
    tokenizer_path: Optional[str] = None
    # ! DO NOT SET, automatically set from buffer.total_epochs
    total_epochs: int = 1  # automatically set
    # ! DO NOT SET, automatically set from buffer.total_steps
    total_steps: Optional[int] = None  # automatically set
    # ! DO NOT SET, automatically set form ray_namespace
    ray_namespace: Optional[str] = None

    def to_storage_config(self) -> StorageConfig:
        storage_config = StorageConfig(
            name=self.name,
            storage_type=self.storage_type,
            path=self.path,
            capacity=self.capacity,
            max_read_timeout=self.max_read_timeout,
            replay_buffer=self.replay_buffer,
            max_retry_times=self.max_retry_times,
            max_retry_interval=self.max_retry_interval,
            split=self.split,
            subset_name=self.subset_name,
            format=self.format,
            enable_progress_bar=self.enable_progress_bar,
            schema_type=self.schema_type,
            index=self.index,
            batch_size=self.batch_size,
            tokenizer_path=self.tokenizer_path,
            total_epochs=self.total_epochs,
            total_steps=self.total_steps,
            ray_namespace=self.ray_namespace,
        )
        return storage_config


@dataclass
class OperatorConfig:
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@Experimental
@dataclass
class ExperiencePipelineConfig:
    """Config for experience pipeline.

    Experience Pipeline is used to pre-process rollout experiences for better training.
    """

    # The list of experience operators to apply, operators will be applied in the order they are defined
    operators: List[OperatorConfig] = field(default_factory=list)
    save_input: bool = True  # whether to save the input experiences
    # the path to save the input experiences, can be a jsonl file or a sqlite database file
    input_save_path: Optional[str] = None

    # The following fields are experimental, do not set them unless you know what you are doing

    # A dictionary of input buffers, buffers are indexed by their names.
    # users only need to set extra buffers here
    inputs: Dict[str, ExperienceBufferConfig] = field(default_factory=dict)
    # The output buffer will automatically set to the trainer input buffer, so we do not need to set it here.
    output: Optional[ExperienceBufferConfig] = None


@Experimental
@dataclass
class TaskPipelineConfig:
    """Config for task pipeline.

    Task Pipeline is used to pre-process raw tasks for better exploring. Currently, we only support using
    Data-Juicer operators for task pipeline.
    """

    # The list of data-juicer operators to apply, operators will be applied in the order they are defined
    operators: List[OperatorConfig] = field(default_factory=list)
    # number of process
    num_process: int = 4
    # The path to the Data-Juicer config file. If set, operators and num_process will be ignored
    config_path: Optional[str] = None

    # Raw input tasksets. Currently, task pipeline only support local file as inputs,
    # e.g., /path/to/file.jsonl or /path/to/file.parquet, not a directory or huggingface path
    inputs: List[str] = field(default_factory=list)
    # Output task buffer, if not set, use `buffer.explorer_input.taskset`. In most cases, users do not need to set this field.
    output: Optional[TasksetConfig] = None

    # The list of fields extracted from the input tasksets and processed into the output taskset
    target_fields: List[str] = field(default_factory=list)

    # weights for priority computing. Usually including 4 types of weights:
    # - difficulty
    # - diversity
    # - usage_frequency
    # - quality
    priority_weights: Dict[str, float] = field(default_factory=dict)

    # number of samples to select after task pipeline. -1 means all
    top_k: int = -1


@Experimental
@dataclass
class DataProcessorConfig:
    """Data Processor config"""

    # support two types of data pipelines for now
    # 1. For task. Data preprocessing from raw dataset to the task set
    task_pipeline: Optional[TaskPipelineConfig] = None
    # 2. For experience. Data processing for rollouts
    experience_pipeline: Optional[ExperiencePipelineConfig] = field(
        default_factory=ExperiencePipelineConfig
    )


@dataclass
class TinkerConfig:
    enable: bool = False
    rank: int = 16  # lora rank
    seed: Optional[int] = None
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    base_url: Optional[str] = None


@dataclass
class ModelConfig:
    # source model path
    model_path: str = ""
    critic_model_path: str = ""

    custom_chat_template: Optional[str] = None
    chat_template_path: Optional[
        str
    ] = None  # path to the chat template file, e.g., jinja2 type; overrides `custom_chat_template` if set

    # rollout args
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    logprobs: int = 0

    # the total number of tokens the model can handle
    max_model_len: Optional[int] = None

    # Note: the following fields are only for the `chat`/`generate` methods in `InferenceModel`
    # if you are using openai API, please set them when calling the API.

    # the maximum number of tokens for the prompt
    max_prompt_tokens: Optional[int] = None
    # the maximum number of tokens for the response
    max_response_tokens: Optional[int] = None
    # the minimum number of tokens for the response
    min_response_tokens: int = 0
    # whether to truncate the prompt; if set to True, the prompt will be truncated to `max_prompt_tokens` tokens;
    # not applicable for OpenAI API
    enable_prompt_truncation: bool = True
    # repetition penalty for response generation
    repetition_penalty: float = 1.0

    # lora config
    lora_configs: Optional[List[LoRAConfig]] = None
    fully_sharded_loras: bool = False
    max_cpu_loras: Optional[int] = None

    # rope config
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None

    # tinker config
    tinker: TinkerConfig = field(default_factory=TinkerConfig)


@dataclass
class InferenceModelConfig:
    # ! DO NOT SET in explorer.rollout_model, automatically set from config.model.model_path
    model_path: Optional[str] = None
    name: Optional[str] = None

    engine_type: str = "vllm"
    engine_num: int = 1
    tensor_parallel_size: int = 1
    use_v1: bool = True
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    seed: int = 42

    # rollout args, ! DO NOT SET
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    logprobs: Optional[int] = None

    # if not set, use `model.max_model_len`
    max_model_len: Optional[int] = None
    # if not set, use `model.max_prompt_tokens`
    max_prompt_tokens: Optional[int] = None
    # if not set, use `model.max_response_tokens`
    max_response_tokens: Optional[int] = None
    # if not set, use `model.min_response_tokens`
    min_response_tokens: Optional[int] = None
    # if not set, use `model.enable_prompt_truncation`
    enable_prompt_truncation: Optional[bool] = None
    # If not set, use `model.repetition_penalty`
    repetition_penalty: Optional[float] = None
    # used for testing very long response generation, do not set it unless you know what you are doing
    ignore_eos: bool = False

    # override chat template in model
    chat_template: Optional[str] = None

    # For Qwen3
    enable_thinking: Optional[bool] = None

    # For history recording
    enable_history: bool = False

    # For OpenAI API
    enable_openai_api: bool = False
    enable_log_requests: bool = False  # whether to enable request logging in vLLM API server

    # For tool calls in OpenAI API
    enable_auto_tool_choice: bool = False

    tool_call_parser: Optional[str] = None

    reasoning_parser: Optional[str] = None

    # ! DO NOT SET
    bundle_indices: str = ""
    ray_namespace: Optional[str] = None
    cuda_visible_devices: Optional[str] = None

    # ! DO NOT SET, automatically set from model.lora_configs
    enable_lora: bool = False
    enable_runtime_lora_updating: bool = False
    lora_modules: Optional[List[Dict]] = None
    lora_kwargs: Optional[dict] = field(default_factory=dict)

    # ! DO NOT SET, rope config
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None


@dataclass
class AlgorithmConfig:
    """Config for algorithm."""

    algorithm_type: str = "ppo"
    # for GRPO-like algorithms, repeat each task for `repeat_times` times
    repeat_times: int = 1

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # the strategy for sampling experiences from the buffer
    sample_strategy: Optional[str] = None
    sample_strategy_args: Optional[dict] = None

    advantage_fn: Optional[str] = None  # "ppo"
    # If not set, use AdvantageFn.default_args()
    advantage_fn_args: Optional[dict] = None

    kl_penalty_fn: Optional[str] = None  # "none"  # set to "none" to disable kl penalty in reward
    # If not set, use kl_penalty_fn.default_args()
    kl_penalty_fn_args: Optional[dict] = None

    policy_loss_fn: Optional[str] = None  # "ppo"
    # If not set, use PolicyLossFn.default_args()
    policy_loss_fn_args: Optional[dict] = None

    kl_loss_fn: Optional[str] = None  # "k2"  # set to "none" to disable kl loss
    # If not set, use kl_loss_fn.default_args()
    kl_loss_fn_args: Optional[dict] = None

    entropy_loss_fn: Optional[str] = None  # "default"
    # If not set, use entropy_loss_fn.default_args()
    entropy_loss_fn_args: Optional[dict] = None

    # aggregation mode for losses: 'token-mean' or 'seq-mean-token-sum' or 'seq-mean-token-mean' or 'seq-mean-token-sum-norm'
    # If not set, use 'token-mean'
    loss_agg_mode: Optional[str] = None


@dataclass
class ClusterConfig:
    """Config for the cluster."""

    ray_address: str = "auto"
    node_num: int = 0
    gpu_per_node: int = 0

    # ! DO NOT SET
    total_gpu_num: int = 0
    rollout_gpu_num: int = 0
    auxiliary_model_gpu_num: int = 0
    explorer_gpu_num: int = 0
    trainer_gpu_num: int = 0
    trainer_node_num: int = 0
    trainer_gpu_num_per_node: int = 0


@Experimental
@dataclass
class ExplorerInput:
    """Config for explorer input."""

    taskset: Optional[TasksetConfig] = None
    tasksets: List[TasksetConfig] = field(default_factory=list)
    eval_tasksets: List[TasksetConfig] = field(default_factory=list)
    # The following args provide default values for the corresponding args in `taskset` and `eval_tasksets`
    default_workflow_type: Optional[str] = None
    default_eval_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None


@dataclass
class TrainerInput:
    """Config for trainer input."""

    # The main experience buffer to be used in trainer
    # Commonly, it is also the output buffer of the Explorer
    experience_buffer: Optional[ExperienceBufferConfig] = None

    # Some auxiliary buffers to facilitate training (e.g., data mixing)
    auxiliary_buffers: Dict[str, ExperienceBufferConfig] = field(default_factory=dict)


@dataclass
class BufferConfig:
    """Config for buffer."""

    batch_size: int = 1
    train_batch_size: int = 0  # default to `batch_size` * `algorithm.n`
    total_epochs: int = 1
    total_steps: Optional[int] = None

    # for explorer
    explorer_input: ExplorerInput = field(default_factory=ExplorerInput)

    # for trainer
    trainer_input: TrainerInput = field(default_factory=TrainerInput)

    # ! DO NOT SET FOLLOWING FIELDS
    explorer_output: Optional[StorageConfig] = None  # automatically set
    tokenizer_path: Optional[str] = None  # automatically set
    pad_token_id: Optional[int] = None  # automatically set
    cache_dir: Optional[str] = None  # automatically set


@dataclass
class ExplorerConfig:
    """Config for explorer."""

    name: str = EXPLORER_NAME
    # for workflow runner
    # number of workflow runners.
    runner_per_model: int = 8  # number of runners per each rollout model
    max_timeout: int = 1800  # wait each task for 30 minutes at most
    max_retry_times: int = 2  # retry each task for 2 times if it fails or timeout
    env_vars: dict = field(default_factory=dict)  # environment variables for workflow runner

    # Workflow Runner Configs for tasks requiring group execution
    # how to run a group of tasks in a single workflow runner
    # "sequential": run tasks one by one, no requirements on workflow design, but have lower throughput
    # "asynchronous": run tasks asynchronously, requires the workflow to be designed with async/await
    #   syntax, and no sharing of state between tasks
    # "multi-threading": run tasks using multi-threading, requires the workflow to be thread-safe,
    #   and no sharing of state between tasks
    concurrent_mode: str = "sequential"
    # the number of time to repeat each task in a single workflow runner
    # we recommend setting this only when using "sequential" concurrent_mode
    max_repeat_times_per_runner: Optional[int] = None

    runner_num: Optional[int] = None  # ! Deprecated

    # for inference models
    # for rollout model
    rollout_model: InferenceModelConfig = field(default_factory=InferenceModelConfig)
    # for other models used in the custom workflows
    auxiliary_models: List[InferenceModelConfig] = field(default_factory=list)

    # for evaluation
    eval_interval: int = 100
    eval_on_startup: bool = True  # evalulate at step 0

    # for benchmark
    bench_on_latest_checkpoint: bool = False  # only benchmark the latest checkpoint

    # for serve mode proxy
    proxy_port: int = 8010
    # listen on all interfaces by default
    listen_address: str = "0.0.0.0"
    # check the running status of the server every 60 seconds
    service_status_check_interval: int = 60
    # keep at least 1 model in running status
    min_running_model_num: int = 1
    # db url for proxy history recorder, if not set, use proxy_history.db in buffer cache dir
    db_url: Optional[str] = None

    # Experimental feature
    over_rollout: OverRolloutConfig = field(default_factory=OverRolloutConfig)
    dynamic_timeout: DynamicTimeoutConfig = field(default_factory=DynamicTimeoutConfig)
    # report runner state every `runner_state_report_interval` seconds, 0 to disable
    runner_state_report_interval: int = 0


@dataclass
class TrainerConfig:
    name: str = TRAINER_NAME
    trainer_type: str = "verl"
    trainer_strategy: str = "fsdp"  # "fsdp", "fsdp2" or "megatron"
    save_interval: int = 0
    enable_preview: bool = True  # enable rollout preview in wandb
    total_steps: Optional[
        int
    ] = None  # total training steps, training stops when reaching this step, None means no limit

    save_hf_checkpoint: str = "last"  # whether to save checkpoint in HuggingFace format
    # "always": save all checkpoints in HF format
    # "never": never save checkpoint in HF format
    # "last": only save the last checkpoint in HF format

    # trainer configs
    grad_clip: float = 1.0
    use_dynamic_bsz: bool = True
    # if None, automatically set to ceil(2 * model.max_model_len / ulysses_sequence_parallel_size)
    max_token_len_per_gpu: Optional[int] = None
    ulysses_sequence_parallel_size: int = 1  # sp size
    fix_actor_microbatch_loss_scale: bool = False  # EXPERIMENTAL
    # TODO: extract more train-related params from underlying trainer engine

    save_strategy: SaveStrategy = SaveStrategy.UNRESTRICTED
    max_checkpoints_to_keep: Optional[int] = None

    trainer_config: Any = field(default_factory=dict)
    trainer_config_path: str = ""  # deprecated, use `trainer_config` instead


@dataclass
class MonitorConfig:
    # TODO: support multiple monitors (List[str])
    monitor_type: str = "tensorboard"
    # the default args for monitor
    monitor_args: Optional[Dict] = None
    # whether to return detailed stats (mean, std, max, min) for evaluation metrics
    detailed_stats: bool = False
    # whether to enable ray timeline profile
    # the output file will be saved to `cache_dir/timeline.json`
    enable_ray_timeline: bool = False
    # ! DO NOT SET, automatically generated as checkpoint_job_dir/monitor
    cache_dir: str = ""


@dataclass
class SynchronizerConfig:
    """Configs for model weight synchronization."""

    sync_method: SyncMethod = SyncMethod.NCCL
    sync_style: SyncStyle = SyncStyle.FIXED
    # sync weights every `sync_interval` steps
    sync_interval: int = 1
    # allow explorer to run `sync_offset` steps before sync
    sync_offset: int = 0
    # waiting for `sync_timeout` seconds before timeout in `nccl` method
    sync_timeout: int = 3600
    # wait for the lastest checkpoint to be ready  # TODO: to be used
    wait_for_checkpoint: bool = False

    # ! DO NOT SET, automatically calculated
    explorer_world_size: Optional[int] = None
    ray_namespace: str = ""


@dataclass
class DataJuicerServiceConfig:
    """Config for Data-Juicer.

    Please update `trinity.service.data_juicer.server.server.py` correspondingly if you change the fields here.
    """

    # the url of the Data-Juicer server
    server_url: Optional[str] = None

    # whether to start Data-Juicer server automatically
    auto_start: bool = False

    # the following fields are only used when `auto_start` is True
    # the port of the Data-Juicer server, if not set, a random port will be used
    port: Optional[int] = None
    # the hostname will be automatically set to "localhost" so we do not need to set it here


@dataclass
class ServiceConfig:
    """Configs for outside services."""

    data_juicer: Optional[DataJuicerServiceConfig] = None


@dataclass
class LogConfig:
    """Configs for logger."""

    level: str = "INFO"  # default log level (DEBUG, INFO, WARNING, ERROR)
    group_by_node: bool = False  # whether to group logs by node IP in Ray cluster
    # ! DO NOT SET, automatically generated as <checkpoint_root_dir>/<project>/<name>/log
    save_dir: str = ""


@dataclass
class StageConfig:
    """Configs for a stage."""

    stage_name: str
    mode: Optional[str] = None
    algorithm: Optional[AlgorithmConfig] = None
    buffer: Optional[BufferConfig] = None
    data_processor: Optional[DataProcessorConfig] = None
    explorer: Optional[ExplorerConfig] = None
    trainer: Optional[TrainerConfig] = None


@dataclass
class Config:
    """Global Configuration"""

    mode: str = "both"  # `explore`, `train`, `both`, `bench`, `serve` or `colocate`
    project: str = "Trinity-RFT"
    group: str = ""
    name: str = "rft"
    # the root dir for checkpoints
    checkpoint_root_dir: str = ""
    # ! DO NOT SET, automatically generated as `checkpoint_root_dir/project/name`
    checkpoint_job_dir: str = ""
    # If not set, automatically generated as f"{config.project}-{config.name}"
    ray_namespace: str = ""
    # whether to continue training from the last checkpoint in checkpoint_job_dir (if any)
    continue_from_checkpoint: bool = True
    # whether to checks GPU memory usage and suggests changes to configs.
    ignore_validator_suggestions: bool = False

    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    data_processor: DataProcessorConfig = field(default_factory=DataProcessorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    explorer: ExplorerConfig = field(default_factory=ExplorerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    synchronizer: SynchronizerConfig = field(default_factory=SynchronizerConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # configurations for different training stages
    stages: List[StageConfig] = field(default_factory=list)

    def save(self, config_path: str) -> None:
        """Save config to file."""
        with open(config_path, "w", encoding="utf-8") as f:
            OmegaConf.save(self, f)

    def __iter__(self):
        """Iterate over configs with each stage applied in order.

        Yields:
            Config: The config after applying each stage.
        """
        for stage in self.stages:
            new_config = deepcopy(self)
            for field_name in stage.__dataclass_fields__:
                stage_value = getattr(stage, field_name)
                if stage_value is not None and hasattr(new_config, field_name):
                    setattr(new_config, field_name, stage_value)
            if stage.stage_name:
                new_config.name = f"{self.name}/{stage.stage_name}"
            # set trainer.save_hf_checkpoint to "last" to make sure next stage can load from HF checkpoint
            new_config.trainer.save_hf_checkpoint = "last"
            new_config.stages = []
            yield new_config

    def check_and_update(self) -> Config:
        """Check and update the config."""
        from trinity.common.config_validator import validators

        # validate
        for validator in validators:
            validator.validate(self)
        return self

    def flatten(self) -> Dict[str, Any]:
        """Flatten the config into a single-level dict with dot-separated keys for nested fields."""

        def _flatten(obj, parent_key="", sep="."):
            items = {}
            if hasattr(obj, "__dataclass_fields__"):
                obj = vars(obj)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    items.update(_flatten(v, new_key, sep=sep))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                    items.update(_flatten(v, new_key, sep=sep))
            elif isinstance(obj, Enum):
                items[parent_key] = obj.value
            else:
                items[parent_key] = obj
            return items

        return _flatten(self)

    def get_envs(self) -> Dict[str, str]:
        """Get the environment variables from the config."""
        envs = {
            PLUGIN_DIRS_ENV_VAR: os.getenv(PLUGIN_DIRS_ENV_VAR, ""),
            LOG_LEVEL_ENV_VAR: self.log.level,
            LOG_DIR_ENV_VAR: self.log.save_dir,
            LOG_NODE_IP_ENV_VAR: "1" if self.log.group_by_node else "0",
        }
        if self.model.tinker.base_url:
            envs["TINKER_BASE_URL"] = self.model.tinker.base_url
        return envs


def load_config(config_path: str) -> Config:
    """Load the configuration from the given path."""
    # TODO: add test
    schema = OmegaConf.structured(Config)
    yaml_config = OmegaConf.load(config_path)
    try:
        config = OmegaConf.merge(schema, yaml_config)
        return OmegaConf.to_object(config)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e
