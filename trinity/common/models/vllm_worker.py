# -*- coding: utf-8 -*-
"""Custom vLLM Worker."""
import logging

import ray
import torch
import torch.distributed

from trinity.common.models.vllm_patch.worker_patch import patch_vllm_prompt_logprobs
from trinity.common.weight_transfer import NCCLReceiver, NCCLSender
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.distributed import init_process_group
from trinity.utils.log import get_logger


def _suppress_layerwise_reload_warnings() -> None:
    """Silence benign vLLM layerwise reload warnings during weight sync."""
    try:
        logger = logging.getLogger("vllm.model_executor.model_loader.reload.layerwise")
        if logger is not None:
            logger.setLevel(logging.ERROR)
    except Exception:  # pragma: no cover - best-effort suppression
        pass


class WorkerExtension:
    def apply_patches(self):
        """Apply necessary patches to vLLM."""
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        patch_vllm_moe_model_weight_loader(self.model_runner.model)
        patch_vllm_prompt_logprobs(self.model_runner)
        _suppress_layerwise_reload_warnings()

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        explorer_name: str = None,
        namespace: str = None,
        zmq_ip: str = None,
        zmq_port: int = None,
        bucket_size_mb: int = 500,
    ):
        """Init torch process group for model weights update"""
        rank = torch.distributed.get_rank()
        self.logger = get_logger(f"vllm_worker_{rank}")

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"
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

        # Set up bucketed weight transfer for NCCL or intra-explorer sync.
        self._weight_receiver = None
        self._weight_sender = None
        if bucket_size_mb > 0:
            bucket_size = bucket_size_mb * 1024 * 1024
            if zmq_ip is not None and zmq_port is not None:
                # NCCL mode: all workers are Receivers (Sender is on Trainer).
                self._weight_receiver = NCCLReceiver(
                    pg=self._model_update_group,
                    bucket_size=bucket_size,
                    zmq_ip=zmq_ip,
                    zmq_port=zmq_port,
                )
                self.logger.info(
                    f"NCCLReceiver ready "
                    f"(ZMQ: {zmq_ip}:{zmq_port}, bucket_size={bucket_size_mb}MB)"
                )
            elif self._weight_update_rank == 0 and world_size > 1:
                # CHECKPOINT/MEMORY mode: rank 0 creates a Sender for
                # bucketed intra-explorer broadcast.  Receivers on rank 1+
                # are created in a second phase via setup_weight_receiver().
                self._weight_sender = NCCLSender()
                self._weight_sender.prepare(self._model_update_group, bucket_size)
                self.logger.info(
                    f"Intra-explorer NCCLSender ready "
                    f"(ZMQ: {self._weight_sender.zmq_info}, "
                    f"bucket_size={bucket_size_mb}MB)"
                )

    def get_weight_sender_zmq_info(self):
        """Return Sender's ZMQ info for Receiver setup (Phase 2).

        Only rank 0 has a Sender (created during ``init_process_group``
        for CHECKPOINT/MEMORY mode).  Other ranks return ``None``.
        """
        if hasattr(self, "_weight_sender") and self._weight_sender is not None:
            return self._weight_sender.zmq_info
        return None

    def setup_weight_receiver(self, zmq_ip, zmq_port, bucket_size_mb):
        """Create :class:`NCCLReceiver` on non-rank-0 workers.

        Phase 2 of intra-explorer bucketed weight transfer setup.
        Called after ``init_process_group`` has created the NCCL group
        and the Sender on rank 0.
        """
        if (
            self._weight_update_rank != 0
            and zmq_ip is not None
            and bucket_size_mb > 0
            and self._weight_receiver is None
        ):
            self._weight_receiver = NCCLReceiver(
                pg=self._model_update_group,
                bucket_size=bucket_size_mb * 1024 * 1024,
                zmq_ip=zmq_ip,
                zmq_port=zmq_port,
            )
            self.logger.info(
                f"Intra-explorer NCCLReceiver ready "
                f"(ZMQ: {zmq_ip}:{zmq_port}, bucket_size={bucket_size_mb}MB)"
            )

    def teardown_process_group(self):
        """Destroy the NCCL process group and finalize sender/receiver."""
        if hasattr(self, "_weight_sender") and self._weight_sender is not None:
            self._weight_sender.finalize()
            self._weight_sender = None
        if hasattr(self, "_weight_receiver") and self._weight_receiver is not None:
            self._weight_receiver.finalize()
            self._weight_receiver = None
        if hasattr(self, "_model_update_group") and self._model_update_group is not None:
            torch.distributed.destroy_process_group(self._model_update_group)
            self._model_update_group = None

    def _load_state_dict_from_disk(self):
        """Load model weights from disk using the latest model path.

        Uses :func:`load_state_dict` to auto-detect the format (safetensors,
        FSDP shards, HuggingFace, or Megatron) and returns a ``dict``.
        """
        model_path = ray.get(self.synchronizer.get_latest_model_path.remote())
        from trinity.common.models.utils import load_state_dict

        result = load_state_dict(model_path)
        if isinstance(result, tuple):
            return self._resolve_checkpoint_ref(*result)
        return result

    def _resolve_checkpoint_ref(self, method, checkpoint_dir):
        """Resolve a ``(method, path)`` checkpoint reference to a state dict."""
        if method == "megatron":
            if self._checkpoint_converter is None:
                from trinity.common.models.utils import get_megatron_converter

                self._checkpoint_converter = get_megatron_converter(checkpoint_dir)
            return self._checkpoint_converter.get_state_dict(checkpoint_dir)
        elif method == "huggingface":
            from trinity.common.models.utils import load_huggingface_state_dict

            return load_huggingface_state_dict(checkpoint_dir)
        else:
            raise NotImplementedError(f"{method} is not supported")

    def update_weight(self):
        """Broadcast weight to all vllm workers from source rank 0 (actor model).

        Rank 0 obtains the state dict (from disk or from Synchronizer), then
        distributes it to all workers:

        1. **Sender** (rank 0, multi-worker): bucketed broadcast + local load.
        2. **Receiver** (rank 1+, multi-worker): bucketed receive.
        3. **Single-worker** (world_size=1): direct load, no broadcast needed.
        """
        state_dict = None
        if self._weight_update_rank == 0:
            result, model_version = ray.get(self.synchronizer.get_model_state_dict.remote())
            if result is None:
                # CHECKPOINT mode: load directly from disk (safetensors / FSDP / HF / Megatron)
                state_dict = self._load_state_dict_from_disk()
            elif isinstance(result, tuple):
                # Legacy: (method, checkpoint_dir) reference for lazy loading
                state_dict = self._resolve_checkpoint_ref(*result)
            else:
                # MEMORY mode: state_dict provided directly by Synchronizer
                state_dict = result

        from vllm.config import set_current_vllm_config

        if self._weight_sender is not None:
            # Rank 0 with intra-explorer Sender: bucketed broadcast to
            # other workers, then load into own model directly.
            assert state_dict is not None
            self._weight_sender.send(state_dict.items())
            with set_current_vllm_config(self.model_runner.vllm_config):
                self.model_runner.reload_weights(
                    weights_iterator=(
                        (name, param.to(self.device)) for name, param in state_dict.items()
                    ),
                    is_checkpoint_format=True,
                )
        elif self._weight_receiver is not None:
            # Rank 1+ with intra-explorer Receiver: bucketed receive.
            weights_iter = self._weight_receiver.receive()
            with set_current_vllm_config(self.model_runner.vllm_config):
                self.model_runner.reload_weights(
                    weights_iterator=weights_iter,
                    is_checkpoint_format=True,
                )
        else:
            # Single-worker (world_size=1): direct load, no broadcast needed.
            assert state_dict is not None
            with set_current_vllm_config(self.model_runner.vllm_config):
                self.model_runner.reload_weights(
                    weights_iterator=(
                        (name, param.to(self.device)) for name, param in state_dict.items()
                    ),
                    is_checkpoint_format=True,
                )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def update_weight_nccl(self):
        """Receive weights via bucketed NCCL broadcast (NCCLReceiver).

        Both verl and verl_legacy trainers now provide ZMQ metadata,
        so a Receiver is always available for NCCL weight sync.
        """
        assert self._weight_receiver is not None, "NCCLReceiver must be set up for NCCL weight sync"
        weights_iter = self._weight_receiver.receive()

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.model_runner.vllm_config):
            self.model_runner.reload_weights(
                weights_iterator=weights_iter,
                is_checkpoint_format=True,
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
