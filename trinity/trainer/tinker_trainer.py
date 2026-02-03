import math
import os
import sys
from typing import Dict, List

import ray
import tinker
import torch
from tinker import types

from trinity.algorithm import ALGORITHM_TYPE
from trinity.algorithm.advantage_fn import ADVANTAGE_FN
from trinity.algorithm.entropy_loss_fn import ENTROPY_LOSS_FN
from trinity.algorithm.entropy_loss_fn.entropy_loss_fn import DummyEntropyLossFn
from trinity.algorithm.kl_fn import KL_FN
from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.tinker.utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    to_tinker_input,
)
from trinity.trainer.trainer import TrainEngineWrapper
from trinity.utils.log import get_logger
from trinity.utils.timer import Timer


class TinkerTrainerWrapper(TrainEngineWrapper):
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("tinker_trainer")
        self._init_algorithm()
        self.synchronizer = Synchronizer.get_actor(namespace=self.config.synchronizer.ray_namespace)

    def _init_algorithm(self):
        self.algorithm = ALGORITHM_TYPE.get(self.config.algorithm.algorithm_type)
        self.algorithm_config = algorithm_config = self.config.algorithm
        if self.algorithm.compute_advantage_in_trainer:
            self.advantage_fn = ADVANTAGE_FN.get(algorithm_config.advantage_fn)(
                **algorithm_config.advantage_fn_args
            )
            self.kl_fn = KL_FN.get(algorithm_config.kl_penalty_fn)(
                **algorithm_config.kl_penalty_fn_args
            )
            # TODO
            raise NotImplementedError(
                "`compute_advantage_in_trainer` is not implemented yet in tinker"
            )
        self.loss_agg_mode = algorithm_config.loss_agg_mode
        self.policy_loss_fn = POLICY_LOSS_FN.get(algorithm_config.policy_loss_fn)(
            backend="tinker", **algorithm_config.policy_loss_fn_args
        )
        self.kl_loss_fn = KL_FN.get(algorithm_config.kl_loss_fn)(**algorithm_config.kl_loss_fn_args)
        self.entropy_loss_fn = ENTROPY_LOSS_FN.get(algorithm_config.entropy_loss_fn)(
            **algorithm_config.entropy_loss_fn_args
        )

        # EXPERIMENTAL: apply loss scale fix
        self.do_fix_actor_microbatch_loss_scale = (
            self.config.trainer.fix_actor_microbatch_loss_scale
            and (self.loss_agg_mode == "token-mean")
        )

        self.lr_scheduler_type = algorithm_config.optimizer.lr_scheduler_type
        self.total_steps = self.config.trainer.total_steps or sys.maxsize
        self.num_warmup_steps = algorithm_config.optimizer.lr_warmup_steps
        if self.num_warmup_steps < 0:
            self.num_warmup_steps = int(
                algorithm_config.optimizer.lr_warmup_steps_ratio * self.total_steps
            )
        self.min_lr_ratio = algorithm_config.optimizer.min_lr_ratio
        assert 0.0 <= self.min_lr_ratio <= 1.0
        self.logger.info(
            f"Total steps: {self.total_steps if self.total_steps != sys.maxsize else 'unlimited'},"
            f" num_warmup_steps: {self.num_warmup_steps}"
        )

        if self.lr_scheduler_type not in {"constant", "cosine"}:
            raise NotImplementedError(
                f"LR scheduler type {self.lr_scheduler_type} is not supported"
            )

    @property
    def _current_lr_factor(self):
        train_step_num = self._train_step_num
        # warmup
        if train_step_num < self.num_warmup_steps:
            factor = float(train_step_num) / float(max(1.0, self.num_warmup_steps))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * factor
            return factor

        # decay
        if train_step_num >= self.total_steps:
            progress = 1.0
        else:
            progress = float(train_step_num - self.num_warmup_steps) / float(
                max(1.0, self.total_steps - self.num_warmup_steps)
            )
        if self.lr_scheduler_type == "constant":
            factor = 1.0
        elif self.lr_scheduler_type == "cosine":
            num_cycles = 0.5  # TODO: may add to config
            factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * factor
        return max(self.min_lr_ratio, factor)

    @property
    def current_learning_rate(self):
        return self._current_lr_factor * self.algorithm_config.optimizer.lr

    @property
    def adam_params(self):
        return types.AdamParams(
            learning_rate=self.current_learning_rate,
            beta1=self.algorithm_config.optimizer.betas[0],
            beta2=self.algorithm_config.optimizer.betas[1],
            # eps is currently not in config
            weight_decay=self.algorithm_config.optimizer.weight_decay,
            grad_clip_norm=self.config.trainer.grad_clip,
        )

    async def prepare(self):
        self.service_client = tinker.ServiceClient()
        self.checkpoint_manager = self.service_client.create_rest_client()
        name_prefix_list = [self.config.project, self.config.group, self.config.name]
        self.tinker_checkpoint_name_prefix = "-".join(
            [prefix for prefix in name_prefix_list if prefix]
        )
        self.default_local_dir = self.config.checkpoint_job_dir

        self.local_latest_checkpointed_iteration = os.path.join(
            self.config.checkpoint_job_dir, "latest_checkpointed_iteration.txt"
        )
        self.local_latest_state_dict_iteration = os.path.join(
            self.config.checkpoint_job_dir, "latest_state_dict_iteration.txt"
        )

        if os.path.exists(self.local_latest_checkpointed_iteration):
            with open(self.local_latest_checkpointed_iteration, "r") as f:
                self._train_step_num = self.latest_remote_checkpoint_step = int(f.read().strip())
            checkpoint_file_path = os.path.join(
                self.default_local_dir,
                f"global_step_{self._train_step_num}",
                "remote_checkpoint_path.txt",
            )
            with open(checkpoint_file_path, "r") as f:
                self.latest_remote_checkpoint_path = f.read().strip()
            self.actor_client = (
                await self.service_client.create_training_client_from_state_with_optimizer_async(
                    path=self.latest_remote_checkpoint_path,
                )
            )
        else:
            self.actor_client = await self.service_client.create_lora_training_client_async(
                base_model=self.config.model.model_path,
                rank=self.config.model.tinker.rank,
                seed=self.config.model.tinker.seed,
                train_mlp=self.config.model.tinker.train_mlp,
                train_attn=self.config.model.tinker.train_attn,
                train_unembed=self.config.model.tinker.train_unembed,
            )
            self.latest_remote_checkpoint_step = 0
            self.latest_remote_checkpoint_path = None
            self._train_step_num = 0
        self.model_info = await self.actor_client.get_info_async()

        if os.path.exists(self.local_latest_state_dict_iteration):
            with open(self.local_latest_state_dict_iteration, "r") as f:
                self.latest_remote_sampler_step = int(f.read().strip())
            sampler_file_path = os.path.join(
                self.default_local_dir,
                f"global_step_{self.latest_remote_sampler_step}",
                "remote_sampler_path.txt",
            )
            with open(sampler_file_path, "r") as f:
                self.latest_remote_sampler_path = f.read().strip()
        else:
            self.latest_remote_sampler_step = None
            self.latest_remote_sampler_path = None

        self.ref_client = await self.service_client.create_sampling_client_async(
            base_model=self.config.model.model_path,
        )

    @property
    def train_step_num(self) -> int:
        """Get the current training step number."""
        return self._train_step_num

    def _loss_func(
        self, batch: list[types.Datum], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = 0.0
        metrics = {}
        assert len(self.model_inputs_list) == len(
            logprobs
        ), "len(self.model_inputs_list) must equal to len(logprobs)"
        for model_inputs, logprob in zip(self.model_inputs_list, logprobs):
            micro_batch_metrics = {}
            response_mask = model_inputs["action_mask"]
            logprob = logprob[-response_mask.shape[0] :]

            pg_loss, pg_loss_metrics = self.policy_loss_fn(logprob=logprob, **model_inputs)
            prefix_metrics(
                src_metrics=pg_loss_metrics, prefix="actor", dst_metrics=micro_batch_metrics
            )

            if self.entropy_loss_fn != DummyEntropyLossFn:
                entropy = -(logprob * logprob.exp())
            else:
                entropy = None
            # compute entropy loss from entropy
            entropy_loss, entropy_loss_metrics = self.entropy_loss_fn(  # type: ignore
                entropy=entropy,
                **model_inputs,
                loss_agg_mode=self.loss_agg_mode,
            )
            prefix_metrics(
                src_metrics=entropy_loss_metrics,
                prefix="actor",
                dst_metrics=micro_batch_metrics,
            )

            # compute kl loss
            kl_loss, kl_loss_metrics = self.kl_loss_fn.calculate_kl_loss(
                logprob=logprob,
                ref_logprob=model_inputs["ref_logprob"],
                response_mask=response_mask,
                loss_agg_mode=self.loss_agg_mode,
                old_logprob=model_inputs["old_logprob"],
            )
            prefix_metrics(
                src_metrics=kl_loss_metrics,
                prefix="actor",
                dst_metrics=micro_batch_metrics,
            )

            # compute policy loss
            policy_loss = pg_loss - entropy_loss + kl_loss
            loss_scale = 1.0
            if not self.do_fix_actor_microbatch_loss_scale:
                loss_scale /= len(logprobs)
            loss = policy_loss * loss_scale
            total_loss = total_loss + loss
            micro_batch_metrics["actor/final_loss"] = loss.detach().item()

            # update metrics
            for key, val in micro_batch_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(val)

        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        return total_loss, avg_metrics

    async def train_step(self, batch_exps: List[Experience]) -> Dict:
        """Training one step.

        Args:
            batch (List[Experience]): A batch of experiences to train.

        Returns:
            Dict: Metrics of the training step.
        """
        batch, batch_input_tokens, model_inputs_list = to_tinker_input(batch_exps, self.logger)
        self.model_inputs_list = model_inputs_list
        timing_raw = {}
        metrics = {}
        self._train_step_num += 1

        with Timer(timing_raw, "step"):
            if self.algorithm.use_reference:  # ref_logprob may not be used
                import asyncio

                ref_logprobs = await asyncio.gather(
                    *[
                        self.ref_client.compute_logprobs_async(input_tokens)
                        for input_tokens in batch_input_tokens
                    ]
                )
                for model_inputs, ref_logprob in zip(model_inputs_list, ref_logprobs):
                    response_length = model_inputs["action_mask"].shape[0]
                    model_inputs["ref_logprob"] = torch.tensor(ref_logprob[-response_length:])

            if self.algorithm.compute_advantage_in_trainer:
                # TODO: following is verl format, which is not compatible with tinker
                raise NotImplementedError(
                    "`compute_advantage_in_trainer` is not implemented yet in tinker"
                )
            else:
                # skip token_level_scores for sft/dpo
                for model_inputs in model_inputs_list:
                    if "token_level_scores" in model_inputs:
                        assert "token_level_rewards" not in model_inputs
                        model_inputs["token_level_rewards"] = model_inputs["token_level_scores"]

            # update actor
            with Timer(timing_raw, "update_actor"):
                fwdbwd_future = await self.actor_client.forward_backward_custom_async(
                    batch, self._loss_func
                )
                optim_future = await self.actor_client.optim_step_async(self.adam_params)
                fwdbwd_result = await fwdbwd_future
                optim_result = await optim_future
                metrics.update(fwdbwd_result.metrics)
                if optim_result.metrics:
                    metrics.update(optim_result.metrics)

        # collect metrics
        metrics.update(compute_data_metrics(batch=self.model_inputs_list))
        timing_metrics = compute_timing_metrics(batch=self.model_inputs_list, timing_raw=timing_raw)
        metrics.update({k.replace("timing_s/", "time/"): v for k, v in timing_metrics.items()})
        metrics.update(
            compute_throughout_metrics(batch=self.model_inputs_list, timing_raw=timing_raw)
        )

        return metrics

    async def save_checkpoint(
        self, block_until_saved: bool = False, save_as_hf: bool = False
    ) -> None:
        """Save the checkpoint."""
        if self.train_step_num == self.latest_remote_checkpoint_step:
            return
        self.latest_remote_checkpoint_step = self.train_step_num
        checkpoint_name = f"{self.tinker_checkpoint_name_prefix}-state-{self.train_step_num}"
        save_state_future = await self.actor_client.save_state_async(checkpoint_name)
        self.latest_remote_checkpoint_path = (await save_state_future).path
        local_path = os.path.join(
            self.default_local_dir,
            f"global_step_{self.train_step_num}",
        )
        os.makedirs(local_path, exist_ok=True)

        # save a flag to indicate this is a full checkpoint dir
        # make sure this flag is created before notifying the synchronizer
        # to avoid the synchronizer recognizing it as a state_dict-only checkpoint
        # TODO: use a better way to indicate full checkpoint
        flag_path = os.path.join(local_path, ".full_checkpoint")
        with open(flag_path, "w") as f:
            f.write("")

        remote_checkpoint_path = os.path.join(local_path, "remote_checkpoint_path.txt")
        with open(remote_checkpoint_path, "w") as f:
            f.write(self.latest_remote_checkpoint_path)

        with open(self.local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.train_step_num))

    def sync_weight(self) -> None:
        """Sync the model weight."""
        raise NotImplementedError("Tinker trainer does not support NCCL sync")

    async def upload_state_dict(self) -> None:
        """Upload the state dict to Synchronizer."""
        await self.save_state_dict()
        ray.get(
            self.synchronizer.set_model_state_dict.remote(
                self.latest_remote_sampler_path, self.train_step_num
            )
        )

    async def save_state_dict(self) -> None:
        """Only save the model state dict for Synchronizer."""
        if self.train_step_num == self.latest_remote_sampler_step:
            return
        self.stale_remote_sampler_step = self.latest_remote_sampler_step
        self.latest_remote_sampler_step = self.train_step_num
        current_checkpoint_name = (
            f"{self.tinker_checkpoint_name_prefix}-sampler-{self.train_step_num}"
        )
        save_weights_future = await self.actor_client.save_weights_for_sampler_async(
            current_checkpoint_name
        )
        self.latest_remote_sampler_path = (await save_weights_future).path
        if self.stale_remote_sampler_step is not None:
            stale_checkpoint_name = (
                f"{self.tinker_checkpoint_name_prefix}-sampler-{self.stale_remote_sampler_step}"
            )
            try:
                await self.checkpoint_manager.delete_checkpoint_async(
                    self.model_info.model_id, stale_checkpoint_name
                )
            except Exception:
                self.logger.warning(f"Failed to remove stale state_dict {stale_checkpoint_name}")
        local_path = os.path.join(
            self.default_local_dir,
            f"global_step_{self.train_step_num}",
        )
        os.makedirs(local_path, exist_ok=True)
        remote_sampler_path = os.path.join(local_path, "remote_sampler_path.txt")
        with open(remote_sampler_path, "w") as f:
            f.write(self.latest_remote_sampler_path)

        with open(self.local_latest_state_dict_iteration, "w") as f:
            f.write(str(self.train_step_num))
