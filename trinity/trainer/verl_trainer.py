# -*- coding: utf-8 -*-
"""veRL Trainer Class

Modified from verl/trainer/ppo/ray_trainer.py
"""
import asyncio
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import ray
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayClassWithInitArgs,
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    create_colocated_worker_cls,
)
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.metric import reduce_metrics
from verl.workers.config import FSDPEngineConfig

from trinity.algorithm import ADVANTAGE_FN, ALGORITHM_TYPE, KL_FN
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import Config
from trinity.common.constants import SaveStrategy
from trinity.common.experience import Experience
from trinity.trainer.trainer import TrainEngineWrapper
from trinity.trainer.verl.utils import compute_data_metrics, to_data_proto
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


class VerlPPOTrainerWrapper(RayPPOTrainer, TrainEngineWrapper):
    """A wrapper for verl.trainer.ppo.RayPPOTrainer."""

    def __init__(
        self,
        global_config: Config,
    ):
        self.logger = get_logger(__name__, in_ray_actor=True)
        train_config = global_config.trainer
        config = OmegaConf.structured(train_config.trainer_config)
        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # processor for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from trinity.trainer.verl.fsdp_workers import (
                ActorRolloutRefWorker,
                CriticWorker,
            )

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from trinity.trainer.verl.megatron_workers import (
                ActorRolloutRefWorker,
                CriticWorker,
            )

        else:
            raise NotImplementedError

        self.checkpoint_monitor = CheckpointMonitor.get_actor(
            namespace=global_config.synchronizer.ray_namespace,
            save_strategy=global_config.trainer.save_strategy,
            default_local_dir=config.trainer.default_local_dir,
            default_hdfs_dir=config.trainer.default_hdfs_dir,
        )

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )
        self.algorithm_config = global_config.algorithm

        # specify advantage function for various rft algorithms
        self.algorithm = ALGORITHM_TYPE.get(self.algorithm_config.algorithm_type)
        if self.algorithm.compute_advantage_in_trainer:
            self.advantage_fn = ADVANTAGE_FN.get(self.algorithm_config.advantage_fn)(
                **self.algorithm_config.advantage_fn_args
            )
            self.kl_fn = KL_FN.get(self.algorithm_config.kl_penalty_fn)(
                **self.algorithm_config.kl_penalty_fn_args
            )
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor=processor,
        )
        self.init_workers()

    def init_workers(self):  # noqa: C901
        """Initialize distributed training workers using Ray backend.


        Creates:

        1. Ray resource pools from configuration

        2. Worker groups for each role (actor, critic, etc.)

        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        actor_role = (
            Role.ActorRolloutRef
            if Role.ActorRolloutRef in self.role_worker_mapping
            else Role.ActorRollout
        )
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            critic_cfg = self.config.critic

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = (
                        critic_cfg.ppo_infer_max_token_len_per_gpu
                    )
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=critic_cfg
            )
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs[
                "ray_wait_register_center_timeout"
            ] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                    )
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                    )
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

    @property
    def train_step_num(self) -> int:
        return self.global_steps

    async def prepare(self):
        self.actor_rollout_wg.setup_weight_sync_group()
        self.actor_rollout_wg.set_algorithm(self.algorithm_config)

        # The global step counter, initialized to 0
        # It represents the total number of training steps completed so far
        # We increment this counter at the beginning of each training step
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        # Do not use verl's dataloader
        self.train_dataloader = None
        self.total_training_steps = self.config.trainer.total_training_steps or sys.maxsize
        self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
        self.config.critic.optim.total_training_steps = self.total_training_steps

    async def save_state_dict(self):  # checkpoint sync
        actor_local_path = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", "actor"
        )
        self.actor_rollout_wg.save_state_dict(
            actor_local_path,
            global_step=self.global_steps,
        )
        await self.checkpoint_monitor.monitor_step.remote(self.global_steps, is_state_dict=True)

    async def upload_state_dict(self):  # state dict sync
        self.actor_rollout_wg.upload_state_dict(self.global_steps)

    async def train_step(self, batch_exps: List[Experience]) -> Dict:  # noqa C901
        batch = to_data_proto(batch_exps, self.tokenizer.pad_token_id, self.processor, self.logger)
        metrics = {}
        self.global_steps += 1
        timing_raw = {}

        with marked_timer("step", timing_raw):
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            if self.algorithm.can_balance_batch and self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)  # TODO this may affect multi-turn

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1
            ).tolist()

            # Operating Mode Selection:
            # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
            # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
            #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                "bypass_mode", False
            )
            if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                if "rollout_log_probs" in batch.batch:
                    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                    apply_bypass_mode(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
            else:  # Recompute old_log_probs  TODO: to be check
                if (batch.meta_info["model_versions"] != self.global_steps - 1).any():
                    self.logger.warning(
                        f"model_versions mismatch: {batch.meta_info['model_versions']} vs {self.global_steps - 1}"
                    )
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    actor_config = self.config.actor_rollout_ref.actor
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=response_masks,
                        loss_agg_mode=actor_config.loss_agg_mode,
                        loss_scale_factor=actor_config.loss_scale_factor,
                    )
                    old_log_prob_metrics = {
                        "actor/entropy": entropy_agg.detach().item(),
                        "perf/mfu/actor_infer": old_log_prob_mfu,
                    }
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)
                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        from verl.utils.debug.metrics import calculate_debug_metrics

                        metrics.update(calculate_debug_metrics(batch))

            if self.algorithm.use_reference:  # ref_logprob may not be used
                # compute reference log_prob
                with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                    ref_log_prob = self._compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            if self.algorithm.use_critic:
                with marked_timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            if self.algorithm.compute_advantage_in_trainer:
                with marked_timer("adv", timing_raw):
                    # compute kl penalty
                    batch, kl_metrics = self.kl_fn.apply_kl_penalty_to_reward(batch)
                    metrics.update(prefix_metrics(kl_metrics, prefix="critic"))
                    # compute advantages, executed on the driver process
                    batch, _ = self.advantage_fn(batch)
            else:
                # skip token_level_scores for sft/dpo
                if "token_level_scores" in batch.batch.keys():
                    assert "token_level_rewards" not in batch.batch.keys()
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # TODO: to be check
            # Compute rollout correction: IS weights, rejection sampling, and metrics
            # Only runs in decoupled mode (computes once per batch using stable π_old)
            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
            if (
                rollout_corr_config is not None
                and "rollout_log_probs" in batch.batch
                and not bypass_recomputing_logprobs  # Only in decoupled mode
            ):
                from verl.trainer.ppo.rollout_corr_helper import (
                    compute_rollout_correction_and_add_to_batch,
                )

                # Compute IS weights, apply rejection sampling, compute metrics
                batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                    batch, rollout_corr_config
                )
                # IS and off-policy metrics already have rollout_corr/ prefix
                metrics.update(is_metrics)

            # update critic
            if self.algorithm.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    critic_output = self._update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if (
                not self.algorithm.use_critic
                or self.config.trainer.critic_warmup <= self.global_steps
            ):
                # update actor
                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self._update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

        # collect metrics
        metrics.update(compute_data_metrics(batch=batch))
        timing_metrics = compute_timing_metrics(batch=batch, timing_raw=timing_raw)
        metrics.update({k.replace("timing_s/", "time/"): v for k, v in timing_metrics.items()})
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(
            compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
        )

        return metrics

    async def save_checkpoint(
        self, block_until_saved: bool = False, save_as_hf: bool = False
    ) -> None:
        await self._save_checkpoint(save_as_hf=save_as_hf)
        if block_until_saved:
            self.actor_rollout_wg.wait_on_save_thread()
            if self.algorithm and self.algorithm.use_critic:
                self.critic_wg.wait_on_save_thread()

    async def _save_checkpoint(self, save_as_hf: bool = False):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        # save a flag to indicate this is a full checkpoint dir
        # make sure this flag is created before notifying the synchronizer
        # to avoid the synchronizer recognizing it as a state_dict-only checkpoint
        # TODO: use a better way to indicate full checkpoint
        os.makedirs(local_global_step_folder, exist_ok=True)
        flag_path = os.path.join(local_global_step_folder, ".full_checkpoint")
        with open(flag_path, "w") as f:
            f.write("")

        self.logger.info(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get(
            "remove_previous_ckpt_in_save", False
        )
        if remove_previous_ckpt_in_save:
            self.logger.warning(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            global_step=self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
            save_as_hf=save_as_hf,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            self.critic_wg.save_checkpoint(
                critic_local_path,
                global_step=self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        await self.checkpoint_monitor.monitor_step.remote(self.global_steps)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                self.logger.info("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(
                    self.config.trainer.resume_from_path, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_from_path
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        self.logger.info(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        self.logger.info(f"Setting global step to {self.global_steps}")
        self.logger.info(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

    def sync_weight(self) -> None:
        self.actor_rollout_wg.sync_weight()
