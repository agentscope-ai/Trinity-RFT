# -*- coding: utf-8 -*-
"""Custom vLLM Worker."""
import ray
import torch
import torch.distributed

from trinity.common.models.vllm_patch.worker_patch import patch_vllm_prompt_logprobs
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.distributed import init_process_group
from trinity.utils.log import get_logger


class WorkerExtension:
    def apply_patches(self):
        """Apply necessary patches to vLLM."""
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        patch_vllm_moe_model_weight_loader(self.model_runner.model)
        patch_vllm_prompt_logprobs(self.model_runner)

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        state_dict_meta: list = None,
        explorer_name: str = None,
        namespace: str = None,
    ):
        """Init torch process group for model weights update"""
        rank = torch.distributed.get_rank()
        self.logger = get_logger(f"vllm_worker_{rank}")

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"
        self._state_dict_meta = state_dict_meta
        self._weight_update_rank = rank + rank_offset
        self.logger.info(
            f"vLLM starting init_process_group:\n"
            f"  > address={master_address}:{master_port}\n"
            f"  > rank={rank}\n"
            f"  > rank_offset={rank_offset}\n"
            f"  > world_size={world_size}"
        )
        self._model_update_group = init_process_group(
            host=master_address,
            port=master_port,
            group_name=group_name,
            backend=backend,
            timeout=timeout,
            world_size=world_size,
            rank=self._weight_update_rank,
        )
        self.logger.info("vLLM init_process_group finished.")
        self._explorer_name = explorer_name
        self._namespace = namespace
        self.synchronizer = Synchronizer.get_actor(namespace=self._namespace)
        self._checkpoint_converter = None

    def update_weight(self):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if self._weight_update_rank == 0:
            state_dict, model_version = ray.get(self.synchronizer.get_model_state_dict.remote())
            if isinstance(state_dict, tuple):
                # currently only megatron return a tuple
                method, checkpoint_dir = state_dict
                if method == "megatron":
                    if self._checkpoint_converter is None:
                        from trinity.common.models.utils import get_megatron_converter

                        self._checkpoint_converter = get_megatron_converter(checkpoint_dir)
                    state_dict = self._checkpoint_converter.get_state_dict(checkpoint_dir)
                else:
                    raise NotImplementedError(f"{method} is not supported")
                ray.get(self.synchronizer.set_model_state_dict.remote(state_dict, model_version))
        if self._state_dict_meta is None:
            self._state_dict_meta = ray.get(self.synchronizer.get_state_dict_meta.remote())

        def _weight_iterator():
            for name, dtype_str, shape in self._state_dict_meta:
                if self._weight_update_rank == 0:
                    weight = state_dict[name]
                    weight = weight.to(self.device)
                else:
                    dtype = getattr(torch, dtype_str)
                    weight = torch.empty(shape, dtype=dtype, device=self.device)
                torch.distributed.broadcast(weight, 0, group=self._model_update_group)
                yield (name, weight)

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.model_runner.vllm_config):
            self.model_runner.reload_weights(
                weights_iterator=_weight_iterator(),
                is_checkpoint_format=True,
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
