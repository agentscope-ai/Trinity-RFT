# -*- coding: utf-8 -*-
"""veRL Trainer Class (engine-based)

Standalone trainer wrapper for Trinity-RFT, using veRL's engine-based worker
architecture. Does NOT inherit from veRL's RayPPOTrainer (deprecated) or
PPOTrainer (includes rollout management that conflicts with Trinity's
decoupled Explorer/Buffer/Trainer architecture).

Only the training-step mechanics are used: compute_log_prob, update_actor,
update_critic, update_weights.
"""
import asyncio
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

import ray
import torch
import transformers
from accelerate import init_empty_weights
from omegaconf import DictConfig
from ray.actor import ActorHandle
from verl import DataProto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.utils import Role
from verl.utils import hf_processor, hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.fs import copy_to_local
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.workers.config.model import HFModelConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from trinity.algorithm import ADVANTAGE_FN, ALGORITHM_TYPE, KL_FN
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import Config
from trinity.common.constants import SaveStrategy
from trinity.common.experience import Experience
from trinity.trainer.trainer import TrainEngineWrapper
from trinity.trainer.verl.checkpoint import CheckpointCoordinator
from trinity.trainer.verl.config import build_verl_config
from trinity.trainer.verl.utils import compute_data_metrics, to_data_proto
from trinity.trainer.verl.workers import TrinityActorRolloutRefWorker
from trinity.utils.log import get_logger


class CheckpointMonitor:
    def __init__(
        self, save_strategy: SaveStrategy, default_local_dir: str, default_hdfs_dir: str = None
    ):
        self.logger = get_logger("checkpoint_monitor", in_ray_actor=True)
        self.default_local_dir = default_local_dir
        self.default_hdfs_dir = default_hdfs_dir
        self.local_latest_checkpointed_iteration = os.path.join(
            default_local_dir, "latest_checkpointed_iteration.txt"
        )
        self.local_latest_state_dict_iteration = os.path.join(
            default_local_dir, "latest_state_dict_iteration.txt"
        )
        self.checkpoint_counter = defaultdict(int)
        self.state_dict_counter = defaultdict(int)
        self.checkpoint_steps = set()
        self.state_dict_steps = set()
        self.latest_checkpoint_step = 0
        self.latest_state_dict_step = 0

        self.save_strategy = save_strategy
        self.condition = asyncio.Condition()
        self.current_identifier = 0
        self.saving_count = 0

    def update_latest_checkpoint_step(self, step: int):
        assert step >= self.latest_checkpoint_step
        if step == self.latest_checkpoint_step:
            return
        self.latest_checkpoint_step = step
        with open(self.local_latest_checkpointed_iteration, "w") as f:
            f.write(str(step))
        if step in self.state_dict_counter:
            assert self.state_dict_counter[step] == 0
            self.update_latest_state_dict_step(step)

        # Upload checkpoint to hdfs
        if self.default_hdfs_dir is not None:
            local_path = os.path.join(self.default_local_dir, f"global_step_{step}")
            hdfs_path = os.path.join(self.default_hdfs_dir, f"global_step_{step}")
            self.logger.info(f"Uploading checkpoint to {hdfs_path}")
            from verl.utils import hdfs_io

            hdfs_io.makedirs(hdfs_path, exist_ok=True)
            hdfs_io.copy(src=local_path, dst=hdfs_path, dirs_exist_ok=True)
        self.logger.info(f"Checkpoint at step {step} saved.")

    def update_latest_state_dict_step(self, step: int):
        assert step >= self.latest_state_dict_step
        if step == self.latest_state_dict_step:
            return
        self.latest_state_dict_step = step
        with open(self.local_latest_state_dict_iteration, "w") as f:
            f.write(str(step))

    async def register_thread_count(
        self,
        step: int,
        *,
        state_dict_thread_count: int = 0,
        checkpoint_thread_count: int = 0,
    ):
        if state_dict_thread_count != 0:
            self.state_dict_counter[step] += state_dict_thread_count
        if checkpoint_thread_count != 0:
            self.checkpoint_counter[step] += checkpoint_thread_count

    async def monitor_step(self, step: int, is_state_dict: bool = False):
        if is_state_dict:
            self.state_dict_steps.add(step)
            if self.state_dict_counter[step] == 0:
                self.update_latest_state_dict_step(step)
        else:
            self.checkpoint_steps.add(step)
            if self.checkpoint_counter[step] == 0 and self.state_dict_counter[step] == 0:
                self.update_latest_checkpoint_step(step)

    async def notify_started(self, node_id: str, job_id: str):
        if self.save_strategy == SaveStrategy.SINGLE_THREAD:
            identifier = self.current_identifier + 1
        elif self.save_strategy == SaveStrategy.SINGLE_PROCESS:
            identifier = f"{node_id}_{job_id}"
        elif self.save_strategy == SaveStrategy.SINGLE_NODE:
            identifier = node_id
        elif self.save_strategy == SaveStrategy.UNRESTRICTED:
            return
        else:
            raise ValueError(f"Invalid save strategy: {self.save_strategy}")

        async with self.condition:
            if identifier != self.current_identifier and self.saving_count > 0:
                await self.condition.wait_for(lambda: self.saving_count == 0)
            self.current_identifier = identifier
            self.saving_count += 1

    async def notify_finished(self, step: int, is_state_dict: bool = False):
        async with self.condition:
            self.saving_count -= 1
            self.condition.notify_all()
        if is_state_dict:
            self.state_dict_counter[step] -= 1
            if (
                step in self.state_dict_steps or step in self.checkpoint_steps
            ) and self.state_dict_counter[step] == 0:
                self.update_latest_state_dict_step(step)
                if step in self.checkpoint_steps and self.checkpoint_counter[step] == 0:
                    self.update_latest_checkpoint_step(step)
        else:
            self.checkpoint_counter[step] -= 1
            if (
                step in self.checkpoint_steps
                and self.checkpoint_counter[step] == 0
                and self.state_dict_counter[step] == 0
            ):
                self.update_latest_checkpoint_step(step)

    @classmethod
    def get_actor(
        cls,
        namespace: str,
        save_strategy: Optional[SaveStrategy] = None,
        default_local_dir: Optional[str] = None,
        default_hdfs_dir: Optional[str] = None,
    ):
        return (
            ray.remote(cls)
            .options(
                name="checkpoint_monitor",
                namespace=namespace,
                get_if_exists=True,
            )
            .remote(
                save_strategy=save_strategy,
                default_local_dir=default_local_dir,
                default_hdfs_dir=default_hdfs_dir,
            )
        )


class VERLTrainer(TrainEngineWrapper):
    """Standalone veRL trainer wrapper for Trinity-RFT.

    Manages engine-based worker groups for actor/critic/ref training without
    inheriting from veRL's PPOTrainer (which bundles rollout logic that Trinity
    handles separately). Modified from veRL's TaskRunner and PPOTrainer, with a
    focus on training mechanics and metrics collection.
    """

    GLOBAL_POOL_ID = "global_pool"

    def __init__(self, global_config: Config):
        self.logger = get_logger(__name__, in_ray_actor=True)
        self.logger.info(
            f"Initializing verl Trainer with {global_config.trainer.trainer_strategy} backend"
        )
        self.global_config = global_config
        self.config: DictConfig = build_verl_config(global_config)
        # ---------------
        # Algorithm Setup
        # ---------------
        self.algorithm_config = global_config.algorithm
        self.algorithm = ALGORITHM_TYPE.get(self.algorithm_config.algorithm_type)
        if self.algorithm.compute_advantage_in_trainer:
            self.advantage_fn = ADVANTAGE_FN.get(self.algorithm_config.advantage_fn)(
                **self.algorithm_config.advantage_fn_args
            )
            self.kl_fn = KL_FN.get(self.algorithm_config.kl_penalty_fn)(
                **self.algorithm_config.kl_penalty_fn_args
            )
        # -------------------------------
        # Workers and Resource Pool Setup
        # -------------------------------
        self.mapping: Dict[Role, str] = {}
        self.role_worker_mapping: Dict[Role, ActorHandle] = {}

        # Add Actor
        lora_rank = self.config.model.lora_rank
        # if lora is enabled, use ref in actor
        self.ref_in_actor = lora_rank > 0 or self.config.model.lora_adapter_path is not None
        self.use_reference_policy = self.algorithm.use_reference
        self.use_critic = self.algorithm.use_critic
        role = (
            Role.ActorRolloutRef
            if self.use_reference_policy and not self.ref_in_actor
            else Role.ActorRollout
        )
        self.role_worker_mapping[role] = ray.remote(TrinityActorRolloutRefWorker)  # type: ignore
        self.mapping[role] = self.GLOBAL_POOL_ID

        # Add Critic
        if self.algorithm.use_critic:
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)  # type: ignore
            self.mapping[Role.Critic] = self.GLOBAL_POOL_ID

        # TODO: use a global resource manager for both explorer and trainer
        resource_pool_spec = {
            self.GLOBAL_POOL_ID: [self.global_config.cluster.trainer_gpu_num_per_node]
            * self.global_config.cluster.trainer_node_num
        }
        # Trinity do not need reward / distillation model workers, so we only create one global pool for actor and critic
        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=self.mapping  # type: ignore
        )

        # ----------------
        # Tokenizer Setup
        # ----------------
        local_path = copy_to_local(
            global_config.model.model_path,
            use_shm=self.config.model.use_shm,
        )
        trust_remote_code = global_config.model.trust_remote_code
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
        # TODO: remove empty model after dataproto conversion is fixed
        hf_config = transformers.AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        with init_empty_weights():
            self.empty_model = transformers.AutoModel.from_config(
                hf_config, trust_remote_code=trust_remote_code
            )
        # Trinity do not need data loader in trainer, so we do not initialize it here

        # Training steps config
        self.total_training_steps = global_config.trainer.total_steps or sys.maxsize
        # we only support cuda for now
        self.device_name = "cuda"
        checkpoint_monitor = CheckpointMonitor.get_actor(
            namespace=global_config.synchronizer.ray_namespace,
            save_strategy=global_config.trainer.save_strategy,
            default_local_dir=self.global_config.checkpoint_job_dir,
            default_hdfs_dir=None,
        )
        self.checkpoint_coordinator = CheckpointCoordinator(checkpoint_monitor)
        self._saved_checkpoint_paths: list[str] = []
        self._init_workers()

    def _init_workers(self):
        """Initialize distributed training workers using Ray backend."""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # Actor+Rollout
        actor_role = (
            Role.ActorRolloutRef
            if Role.ActorRolloutRef in self.role_worker_mapping
            else Role.ActorRollout
        )
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config,
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # Critic (engine-based TrainingWorker)
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # Build HFModelConfig for critic manually.
            # model_config is NOT a dataclass field in CriticConfig (only in
            # _mutable_fields), so it cannot be in the DictConfig passed to
            # omega_conf_to_dataclass. We construct it here from global_config.
            gcfg = self.global_config
            critic_model_path = gcfg.model.critic_model_path or gcfg.model.model_path
            # Only scalar overrides go through override_config. rope_scaling
            # (a nested dict) is not supported by veRL's update_model_config
            # and is not needed for training — the model's config.json is
            # authoritative. It is primarily relevant for Explorer/vLLM.
            override_config = {}
            if gcfg.model.rope_theta is not None:
                override_config["rope_theta"] = gcfg.model.rope_theta
            critic_model_config = HFModelConfig(
                path=critic_model_path,
                use_shm=False,
                trust_remote_code=gcfg.model.trust_remote_code,
                enable_gradient_checkpointing=True,
                use_remove_padding=gcfg.trainer.use_remove_padding,
                override_config=override_config,
            )
            critic_engine_config = critic_cfg.engine
            critic_optim_config = critic_cfg.optim
            critic_checkpoint_config = critic_cfg.checkpoint

            # Set engine infer/training token length from critic config
            critic_engine_config.infer_max_token_len_per_gpu = (
                critic_cfg.ppo_infer_max_token_len_per_gpu
            )
            critic_engine_config.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu

            critic_worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_model_config,
                engine_config=critic_engine_config,
                optimizer_config=critic_optim_config,
                checkpoint_config=critic_checkpoint_config,
            )

            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=critic_worker_cfg
            )
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # Spawn worker groups
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if self.config.global_profiler.steps is not None:
            wg_kwargs["profile_steps"] = self.config.global_profiler.steps
            # Only require nsight worker options when tool is nsys
            if self.config.global_profiler.tool == "nsys":
                # TODO: setup nsys config
                # assert (
                #     OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                #     is not None
                # ), "worker_nsight_options must be set when using nsys with profile_steps"
                # wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                #     OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                # )
                pass
        wg_kwargs["device_name"] = self.device_name
        self.logger.info(f"worker group kwargs: {wg_kwargs}")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.logger.info(f"create worker group {spawn_wg.keys()}")

        # Initialize critic
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            self.logger.info("critic model engine initialized")

        # Initialize actor and ref model
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        self.logger.info("actor and ref model engine initialized")
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # Checkpoint saving is handled by worker-level save_checkpoint / save_state_dict
        # methods (delegating to strategy-specific helpers in fsdp_engine/megatron_engine).
        # No separate CheckPointEngineManager is needed in the current architecture.
        self.logger.info("workers initialized")
        # Trinity do not need reward model / distillation model / agent loop, so we do not initialize them here

    # ------------------------------------------------------------------
    # TrainEngineWrapper interface
    # ------------------------------------------------------------------

    @property
    def train_step_num(self) -> int:
        return self.global_steps

    async def prepare(self):
        self.actor_rollout_wg.set_trinity_config(
            self.algorithm_config, self.global_config.synchronizer.ray_namespace
        )
        self.global_steps = 0
        self._load_checkpoint()

    async def get_weight_sync_info(self):
        results = self.actor_rollout_wg.get_weight_sync_info()
        for r in results:
            if r is not None:
                return r
        raise RuntimeError("Failed to get weight sync info from rank 0")

    async def setup_weight_sync_group(
        self,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        timeout: int,
        bucket_size_mb: int = 500,
        per_tensor: bool = False,
    ):
        self.actor_rollout_wg.setup_weight_sync_group(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            group_name=group_name,
            timeout=timeout,
            bucket_size_mb=bucket_size_mb,
            per_tensor=per_tensor,
        )

    async def teardown_weight_sync_group(self):
        self.actor_rollout_wg.teardown_weight_sync_group()

    async def train_step(self, batch_exps: List[Experience]) -> Dict:  # noqa: C901
        batch = to_data_proto(
            batch_exps, self.tokenizer.pad_token_id, self.empty_model, self.logger
        )
        metrics = {}
        self.global_steps += 1
        timing_raw = {}

        with marked_timer("step", timing_raw):
            batch.meta_info["temperature"] = self.global_config.model.temperature

            if self.algorithm.can_balance_batch and self.global_config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1
            ).tolist()

            # Extract images_seqlens for multi-modal MFU computation
            images_seqlens_all = []
            for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
                if "image_grid_thw" not in multi_modal_input:
                    continue
                images_seqlens = multi_modal_input.get("images_seqlens", None)
                if images_seqlens is None:
                    continue
                images_seqlens_all.extend(images_seqlens.tolist())
            if images_seqlens_all:
                batch.meta_info["images_seqlens"] = images_seqlens_all

            # Old log-probs computation:
            # - bypass_old_logprobs=True (default): use rollout_log_probs directly
            #   (2 policies: π_rollout, π_θ)
            # - bypass_old_logprobs=False: recompute old_log_probs as proximal anchor
            #   (3 policies: π_rollout, π_old, π_θ)
            if self.global_config.algorithm.bypass_old_logprobs:
                if "rollout_log_probs" in batch.batch:
                    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
            else:
                # TODO: Recompute path not yet fully validated with the
                # new engine-based worker. Use bypass_old_logprobs=True for now.
                if (
                    "model_versions" in batch.meta_info
                    and (batch.meta_info["model_versions"] != self.global_steps - 1).any()
                ):
                    self.logger.warning(
                        f"model_versions mismatch: {batch.meta_info['model_versions']} vs {self.global_steps - 1}"
                    )
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    batch = self._compute_old_log_prob(batch, metrics)

            if self.use_reference_policy:
                with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                    batch = self._compute_ref_log_prob(batch)

            if self.use_critic:
                with marked_timer("values", timing_raw):
                    batch = self._compute_values(batch)

            if self.algorithm.compute_advantage_in_trainer:
                with marked_timer("adv", timing_raw):
                    batch, kl_metrics = self.kl_fn.apply_kl_penalty_to_reward(batch)
                    metrics.update(prefix_metrics(kl_metrics, prefix="advantage"))
                    batch, adv_metrics = self.advantage_fn(batch)
                    metrics.update(prefix_metrics(adv_metrics, prefix="advantage"))
            else:
                if "token_level_scores" in batch.batch.keys():
                    assert "token_level_rewards" not in batch.batch.keys()
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # TODO: add Rollout correction

            # Update critic
            if self.algorithm.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    batch = self._update_critic(batch, metrics)

            # Update actor
            if self.global_config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch = self._update_actor(batch, metrics)

        # Collect metrics
        metrics.update(compute_data_metrics(batch=batch))
        timing_metrics = compute_timing_metrics(batch=batch, timing_raw=timing_raw)
        metrics.update({k.replace("timing_s/", "time/"): v for k, v in timing_metrics.items()})
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(
            compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
        )
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

        return metrics

    async def save_checkpoint(
        self, block_until_saved: bool = False, save_as_hf: bool = False
    ) -> None:
        local_global_step_folder = os.path.join(
            self.global_config.checkpoint_job_dir, f"global_step_{self.global_steps}"
        )
        os.makedirs(local_global_step_folder, exist_ok=True)
        with open(os.path.join(local_global_step_folder, ".full_checkpoint"), "w") as f:
            f.write("")

        actor_local_path = os.path.join(local_global_step_folder, "actor")

        self.actor_rollout_wg.save_checkpoint(
            local_path=actor_local_path,
            global_step=self.global_steps,
            save_as_hf=save_as_hf,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            self.critic_wg.save_checkpoint(
                local_path=critic_local_path,
                global_step=self.global_steps,
            )

        # Manage checkpoint retention at the VERLTrainer level (not inside
        # the engine) to avoid veRL's double-counting bug when the same path
        # is registered multiple times.
        self._retain_checkpoints(local_global_step_folder)

        await self.checkpoint_coordinator.register_and_monitor(
            self.global_steps, state_dict_thread_count=1
        )

        if block_until_saved:
            await self.wait_for_save()

    async def wait_for_save(self) -> None:
        self.actor_rollout_wg.wait_on_save_thread()

    def _retain_checkpoints(self, new_path: str) -> None:
        """Register a checkpoint path and remove old ones beyond the retention limit.

        Deduplicates: if the same path was already the latest registered (e.g.
        save_state_dict and save_checkpoint at the same step), it is not
        counted twice.
        """
        import shutil

        new_path = os.path.abspath(new_path)
        max_keep = self.global_config.trainer.max_checkpoints_to_keep

        # Dedup: skip if this is already the latest registered path
        if self._saved_checkpoint_paths and self._saved_checkpoint_paths[-1] == new_path:
            return
        self._saved_checkpoint_paths.append(new_path)

        if not max_keep or max_keep <= 0:
            return
        while len(self._saved_checkpoint_paths) > max_keep:
            old_path = self._saved_checkpoint_paths.pop(0)
            if os.path.exists(old_path):
                self.logger.info(f"Removing old checkpoint: {old_path}")
                shutil.rmtree(old_path, ignore_errors=True)

    async def sync_weight_nccl(self) -> None:
        self.actor_rollout_wg.sync_weight_nccl()

    async def upload_state_dict(self):
        self.actor_rollout_wg.upload_state_dict(self.global_steps)

    async def save_state_dict(self):
        actor_local_path = os.path.join(
            self.global_config.checkpoint_job_dir, f"global_step_{self.global_steps}", "actor"
        )
        self.actor_rollout_wg.save_state_dict(
            actor_local_path,
            global_step=self.global_steps,
        )
        await self.checkpoint_coordinator.register_and_monitor(
            self.global_steps, is_state_dict=True, state_dict_thread_count=1
        )

    # ------------------------------------------------------------------
    # Internal training methods
    # ------------------------------------------------------------------

    def _get_dp_size(self) -> int:
        """Get data parallel size from actor worker group."""
        role = "actor"
        wg = self.actor_rollout_wg
        if role not in wg._dispatch_info:
            dp_rank_mapping = wg._query_dispatch_info(role)
            wg._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = wg._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics: dict):
        """Reorder batch so each DP rank gets similar total tokens."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        dp_size = self._get_dp_size()

        global_partition_lst = get_seqlen_balanced_partitions(
            workload_lst, k_partitions=dp_size, equal_size=True
        )
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition

        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(),
            partitions=global_partition_lst,
            prefix="global_seqlen",
        )
        metrics.update(global_balance_stats)

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        metadata["temperature"] = batch.meta_info.get(
            "temperature", self.global_config.model.temperature
        )
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        log_probs = tu.get(output, "log_probs")
        log_probs = no_padding_2_padding(log_probs, batch_td)
        batch.batch["ref_log_prob"] = log_probs.float()
        return batch

    def _compute_old_log_prob(self, batch: DataProto, metrics: Dict) -> DataProto:
        # NOTE: This recompute path is not yet fully validated for verl.
        # Callers should prefer bypass_old_logprobs=True until this is stable.
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_sum_pi_squared = self.config.actor.calculate_sum_pi_squared
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
            temperature=batch.meta_info.get("temperature", self.global_config.model.temperature),
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        output_metrics = tu.get(output, "metrics")
        old_log_prob_mfu = output_metrics.get("mfu") if output_metrics is not None else None

        entropy = no_padding_2_padding(entropy, batch_td).float()
        log_probs = no_padding_2_padding(log_probs, batch_td).float()
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td).float()

        batch.batch["old_log_probs"] = log_probs
        batch.batch["entropy"] = entropy
        # for backward compatibility, remove the above line in the future
        batch.batch["entropys"] = entropy

        if routed_experts is not None:
            if "routed_experts" in batch.batch:
                raise ValueError(
                    "Detected conflicting router replay configuration: "
                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                    "cannot be enabled simultaneously. "
                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                    "it should not be set when using R2 mode."
                )
            batch.batch["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            batch.batch["sum_pi_squared"] = sum_pi_squared

        response_masks = batch.batch["response_mask"]
        entropy_agg = agg_loss(
            loss_mat=entropy,
            loss_mask=response_masks,
            loss_agg_mode=self.global_config.algorithm.loss_agg_mode or "token-mean",
            loss_scale_factor=None,  # Keep None for now
        )
        metrics["actor/entropy"] = entropy_agg.detach().item()
        if old_log_prob_mfu is not None:
            metrics["perf/mfu/actor_infer"] = old_log_prob_mfu

        if "rollout_log_probs" in batch.batch:
            from verl.utils.debug.metrics import calculate_debug_metrics

            metrics.update(calculate_debug_metrics(batch))

        return batch

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        batch.batch["values"] = values.float()
        return batch

    def _update_actor(self, batch: DataProto, metrics: Dict) -> DataProto:
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.algorithm_config.entropy_loss_fn != "none"
        ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
        ppo_epochs = self.config.actor.ppo_epochs
        seed = self.config.actor.data_loader_seed
        shuffle = self.config.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            temperature=batch.meta_info.get("temperature", self.global_config.model.temperature),
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        if "actor/mfu" in actor_output:
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        metrics.update(reduce_metrics(actor_output))
        return batch

    def _update_critic(self, batch: DataProto, metrics: Dict) -> DataProto:
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )
        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        if "critic/mfu" in output:
            output["perf/mfu/critic"] = output.pop("critic/mfu")
        metrics.update(reduce_metrics(output))
        return batch

    def _load_checkpoint(self):
        if not self.global_config.continue_from_checkpoint:
            self.logger.info("Training from scratch (continue_from_checkpoint=False)")
            return

        checkpoint_folder = self.global_config.checkpoint_job_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if global_step_folder is None:
            self.logger.info("Training from scratch (checkpoint folder not found)")
            return

        self.logger.info(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        self.logger.info(f"Setting global step to {self.global_steps}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=False)
        if self.use_critic:
            critic_path = os.path.join(global_step_folder, "critic")
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=False)
