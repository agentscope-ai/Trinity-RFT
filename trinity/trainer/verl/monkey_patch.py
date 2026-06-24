from types import MethodType
from typing import Optional

import torch
import verl.utils.torch_functional as verl_F
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import index_first_axis, unpad_input
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.ulysses import ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngine,
    FSDPEngineWithLMHead,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.workers.utils.padding import build_attention_mask_from_nested


# from https://github.com/verl-project/verl/pull/5886
# Remove this patch once the fix is released in veRL
def left_right_2_no_padding(data: TensorDict) -> TensorDict:
    """
    Convert TensorDict from left-right padding to no-padding format.

    Args:
        data: TensorDict with "input_ids", "attention_mask", "response_mask", "position_ids"

    Returns:
        data: TensorDict with
        - Tensor includes NestedTensors like "input_ids", "loss_mask", "position_ids"
        - NonTensorData includes "max_seq_len", "max_response_len", "indices"

    Note:
    1. the return input_ids/position_ids/loss_mask are nested tensor.
    2. we will remove "attention_mask", "response" in the return data, but "response_mask" is kept.
    """
    assert "input_ids" in data, "input_ids is required in left-right padding data"
    assert "attention_mask" in data, "attention_mask is required in left-right padding data"
    assert "response_mask" in data, "response_mask is required in left-right padding data"
    assert "position_ids" in data, "position_ids is required in left-right padding data"

    input_ids = data.pop("input_ids")
    attention_mask = data["attention_mask"]
    response_mask = data["response_mask"]
    position_ids = data["position_ids"]  # (bs, seq_len) or # (bs, 4, seq_len)

    max_seq_len, max_response_len = input_ids.shape[1], response_mask.shape[1]
    tu.assign_non_tensor_data(data, "max_seq_len", max_seq_len)
    tu.assign_non_tensor_data(data, "max_response_len", max_response_len)

    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    tu.assign_non_tensor_data(data, "indices", indices)

    input_ids_nested = torch.nested.nested_tensor_from_jagged(
        input_ids_rmpad.squeeze(-1), offsets=cu_seqlens
    )

    position_ids_list = []
    num_pos_components = (
        0  # 0 means 1D position_ids, >0 means multi-component (e.g. 4 for Qwen3.5/Qwen2-VL)
    )
    for i in range(attention_mask.shape[0]):
        curr_mask = attention_mask[i].bool()
        curr_pos_ids = position_ids[i]
        if curr_pos_ids.dim() == 1:  # (seq_len,)
            valid_ids = curr_pos_ids[curr_mask]
        else:  # (num_components, seq_len) — flatten to 1D for nested tensor compatibility
            # 3D jagged nested tensors have broken unbind() and to_padded_tensor() in PyTorch
            # (see pytorch/pytorch#153238), so we flatten to 1D and reshape back in prepare_model_inputs
            num_pos_components = curr_pos_ids.shape[0]
            valid_ids = (
                curr_pos_ids[:, curr_mask].T.contiguous().flatten()
            )  # (valid_len * num_components,)
        position_ids_list.append(valid_ids)
    position_ids_nested = torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)
    if num_pos_components > 0:
        tu.assign_non_tensor_data(data, "num_pos_components", num_pos_components)

    data["input_ids"] = input_ids_nested
    data["position_ids"] = position_ids_nested
    data["loss_mask"] = data["response_mask"]

    routed_experts = data.get("routed_experts", None)
    if routed_experts is not None and not routed_experts.is_nested:
        if routed_experts.max() <= 255:
            routed_experts = routed_experts.to(torch.uint8)
        routed_experts_rmpad = index_first_axis(routed_experts.unsqueeze(-1).flatten(0, 1), indices)
        routed_experts_nested = torch.nested.nested_tensor_from_jagged(
            routed_experts_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        data["routed_experts"] = routed_experts_nested

    # (bsz, seqlen, topk)
    teacher_logprobs = data.get("teacher_logprobs", None)
    teacher_ids = data.get("teacher_ids", None)
    if teacher_logprobs is not None and teacher_ids is not None:
        teacher_logprobs_rmpad = index_first_axis(
            teacher_logprobs.unsqueeze(-1).flatten(0, 1), indices
        )
        teacher_ids_rmpad = index_first_axis(teacher_ids.unsqueeze(-1).flatten(0, 1), indices)
        teacher_logprobs_nested = torch.nested.nested_tensor_from_jagged(
            teacher_logprobs_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        teacher_ids_nested = torch.nested.nested_tensor_from_jagged(
            teacher_ids_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        data["teacher_logprobs"] = teacher_logprobs_nested
        data["teacher_ids"] = teacher_ids_nested

    return data


# from https://github.com/verl-project/verl/pull/6604
# Remove this patch once the fix is released in veRL
def save_checkpoint(
    self,
    local_path: str,
    hdfs_path: Optional[str] = None,
    global_step: int = 0,
    max_ckpt_to_keep: Optional[int] = None,
    **kwargs,
) -> None:
    """
    Save FSDP checkpoint, handling parameter offload as needed.
    """
    origin_module_device = next(self.module.parameters()).device.type
    if (self._is_offload_param or origin_module_device == "cpu") and not getattr(
        self, "_uses_fsdp2_cpu_offload_policy", False
    ):
        load_fsdp_model_to_gpu(self.module)

    self.checkpoint_manager.save_checkpoint(
        local_path=local_path,
        hdfs_path=hdfs_path,
        global_step=global_step,
        max_ckpt_to_keep=max_ckpt_to_keep,
    )

    torch.distributed.barrier()
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.module)


# ---------------------------------------------------------------------------
# Patch: prepare_model_inputs with seq_idx / cu_seqlens for packed sequences
# ---------------------------------------------------------------------------
# Needed by models with linear-attention layers (e.g. Qwen3.5 GateDeltaNet)
# that require ``seq_idx`` and ``cu_seqlens`` in the model forward kwargs.
# Remove this patch once veRL upstream adds native support.


def get_seq_idx(cu_seqlens: torch.Tensor, total_nnz: int) -> torch.Tensor:
    """Build ``seq_idx`` from ``cu_seqlens``, mapping each packed position to its
    original sequence id.

    Args:
        cu_seqlens: Shape ``(batch + 1,)``. Cumulative sequence lengths.
        total_nnz: Total number of packed tokens, i.e. ``cu_seqlens[-1]``.

    Returns:
        Shape ``(total_nnz,)``, where each position is the original sequence id
        (0-indexed). For example, cu_seqlens=[0,3,7,10] -> [0,0,0,1,1,1,1,2,2,2].
    """
    device = cu_seqlens.device
    batch_size = cu_seqlens.shape[0] - 1
    seq_idx = torch.zeros(total_nnz, dtype=torch.int32, device=device)
    seq_idx.scatter_(
        dim=0,
        index=cu_seqlens[1:-1].long(),
        src=torch.ones(batch_size - 1, dtype=torch.int32, device=device),
    )
    seq_idx = seq_idx.cumsum(dim=0, dtype=torch.int32)
    return seq_idx


def prepare_model_inputs(self, micro_batch: TensorDict):
    """Rewritten ``FSDPEngineWithLMHead.prepare_model_inputs`` that injects
    ``seq_idx`` and ``cu_seqlens`` into model_inputs for packed-sequence
    models (e.g. Qwen3.5 GateDeltaNet).

    This is a full rewrite (not a wrapper) so that the Ulysses SP pad_size
    adjustment on ``seq_idx`` / ``cu_seqlens`` is handled inline, right after
    ``ulysses_pad_and_slice_inputs`` returns ``pad_size``.
    """
    use_remove_padding = tu.get_non_tensor_data(
        data=micro_batch, key="use_remove_padding", default=True
    )
    pad_mode = tu.get_non_tensor_data(
        data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING
    )
    use_fused_kernels = tu.get_non_tensor_data(
        data=micro_batch, key="use_fused_kernels", default=False
    )
    temperature = micro_batch["temperature"]
    temperature_item = temperature
    if use_fused_kernels:
        assert not isinstance(
            temperature, torch.Tensor
        ), "use_fused_kernels does not support per sample temperature yet"
    assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

    multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
    input_ids = micro_batch["input_ids"]
    position_ids = micro_batch["position_ids"]

    if not isinstance(temperature, torch.Tensor):
        temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

    temperature = temperature.to(torch.float32)
    assert temperature.shape[0] == input_ids.shape[0]

    # args used to get outputs
    output_args = {}

    if use_remove_padding:
        # ---- compute cu_seqlens & seq_idx from nested input_ids ----
        cu_seqlens = input_ids.offsets().to(torch.int32)  # (batch+1,)
        total_nnz = cu_seqlens[-1].item()
        seq_idx = get_seq_idx(cu_seqlens, total_nnz)  # (total_nnz,)

        # support per sample temperature
        temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()  # (total_nnz,)
        temperature_rmpad = temperature_rmpad.unsqueeze(0)  # (1, total_nnz)

        if pad_mode == DatasetPadMode.NO_PADDING:
            input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
            # https://github.com/verl-project/verl/pull/5886
            num_pos_components = tu.get_non_tensor_data(
                data=micro_batch, key="num_pos_components", default=0
            )
            if num_pos_components > 0:
                # position_ids stored as flattened 1D nested tensor: (total_nnz * num_components,)
                # reshape to (num_components, 1, total_nnz)
                flat_pos = position_ids.values()  # (total_nnz * num_components,)
                total_nnz = flat_pos.shape[0] // num_pos_components
                position_ids_rmpad = flat_pos.view(total_nnz, num_pos_components).T.unsqueeze(1)
            else:
                position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)
        else:
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        # for compute the log_prob
        input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

        # pad and slice the inputs if sp > 1
        if self.use_ulysses_sp:
            is_vlm_model = hasattr(
                getattr(self.module, "module", self.module).config, "vision_config"
            )
            if is_vlm_model:
                # vlm model's inputs will be sliced after embedding
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                    input_ids_rmpad,
                    position_ids_rmpad=position_ids_rmpad,
                    sp_size=self.ulysses_sequence_parallel_size,
                )
            else:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad,
                    position_ids_rmpad=position_ids_rmpad,
                    sp_size=self.ulysses_sequence_parallel_size,
                    skip_position_ids_rmpad=getattr(self, "_veomni_handles_position_ids", False),
                )
            input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                input_ids_rmpad_rolled,
                position_ids_rmpad=None,
                sp_size=self.ulysses_sequence_parallel_size,
            )

            temperature_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                temperature_rmpad,
                position_ids_rmpad=None,
                sp_size=self.ulysses_sequence_parallel_size,
                pad_value=1,
            )

            output_args["pad_size"] = pad_size

            # ---- adjust seq_idx & cu_seqlens for Ulysses SP padding ----
            if pad_size > 0:
                seq_idx = torch.cat(
                    [
                        seq_idx,
                        torch.full(
                            (pad_size,),
                            seq_idx[-1].item(),
                            dtype=seq_idx.dtype,
                            device=seq_idx.device,
                        ),
                    ],
                    dim=0,
                )
                cu_seqlens = cu_seqlens.clone()
                cu_seqlens[-1] += pad_size

        input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
        temperature_rmpad = temperature_rmpad.squeeze(0)
        output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
        output_args["temperature_rmpad"] = temperature_rmpad

        # only pass input_ids and position_ids to enable flash_attn_varlen
        max_seq_len = cu_seqlens.diff().max()
        model_inputs = {
            "input_ids": input_ids_rmpad,
            "attention_mask": None,
            "position_ids": position_ids_rmpad,
            # seq_idx & cu_seqlens for packed-sequence linear attention
            "seq_idx": seq_idx.unsqueeze(0).to(torch.int32),
            "cu_seq_lens_q": cu_seqlens,
            "cu_seq_lens_k": cu_seqlens,
            "max_length_q": max_seq_len,
            "max_length_k": max_seq_len,
        }

    else:
        if pad_mode == DatasetPadMode.NO_PADDING:
            input_ids = micro_batch["input_ids"]
            position_ids = micro_batch["position_ids"]
            pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            batch_size = micro_batch.batch_size[0]
            seq_len_effective = input_ids.offsets().diff()
            max_seq_len = int(seq_len_effective.max().item())

            input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
            # we store the per sample temperature
            output_args["temperature"] = temperature

            input_ids = torch.nested.to_padded_tensor(
                input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
            )
            # https://github.com/verl-project/verl/pull/5886
            num_pos_components = tu.get_non_tensor_data(
                data=micro_batch, key="num_pos_components", default=0
            )
            if num_pos_components > 0:
                # position_ids stored as flattened 1D nested: each sample has (seq_len * num_components,)
                # pad to (batch, max_seq_len * num_components), then reshape to (num_components, batch, max_seq_len)
                position_ids = (
                    torch.nested.to_padded_tensor(
                        position_ids,
                        padding=0,
                        output_size=(batch_size, max_seq_len * num_pos_components),
                    )
                    .view(batch_size, max_seq_len, num_pos_components)
                    .permute(2, 0, 1)
                )
            else:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, max_seq_len)
                )

            attention_mask = build_attention_mask_from_nested(
                input_ids=micro_batch["input_ids"], max_seq_len=max_seq_len
            )

            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

        else:
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

    extra_args = {}
    if use_fused_kernels:
        extra_args["temperature"] = temperature_item
        extra_args["return_dict"] = True
        if use_remove_padding:
            extra_args["shift_labels"] = output_args["input_ids_rmpad_rolled"].unsqueeze(0)

    model_inputs.update(multi_modal_inputs)
    model_inputs.update(extra_args)

    return model_inputs, output_args


def patch_verl_engine(engine):
    if engine is None:
        return
    if isinstance(engine, FSDPEngine) and not getattr(engine, "_patched", False):
        engine.save_checkpoint = MethodType(save_checkpoint, engine)
        # Patch prepare_model_inputs to inject seq_idx/cu_seqlens for
        # packed-sequence models (e.g. Qwen3.5 GateDeltaNet).
        if isinstance(engine, FSDPEngineWithLMHead):
            engine.prepare_model_inputs = MethodType(prepare_model_inputs, engine)
        setattr(engine, "_patched", True)
