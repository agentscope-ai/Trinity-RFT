import warnings
from types import MethodType
from typing import Optional

import torch
import verl.utils.torch_functional as verl_F
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import index_first_axis, unpad_input
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.import_utils import is_trl_available
from verl.utils.model import extract_multi_modal_inputs, patch_valuehead_model
from verl.utils.transformers_compat import get_auto_model_for_vision2seq
from verl.utils.ulysses import ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngine,
    FSDPEngineWithLMHead,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.workers.utils.padding import build_attention_mask_from_nested

from trinity.trainer.verl_legacy.monkey_patch import apply_monkey_patch

AutoModelForVision2Seq = get_auto_model_for_vision2seq()


def load_valuehead_model(local_path, torch_dtype, model_config, trust_remote_code, use_meta=False):
    from transformers import AutoModelForCausalLM, AutoModelForTokenClassification

    # When ``use_meta`` is True (non-rank-0 processes under FSDP2), build the model
    # on the meta device from the config instead of loading pretrained weights, so
    # that FSDP2 can later broadcast rank-0's materialized weights. ``from_config``
    # is used in place of ``from_pretrained`` and (for the trl value-head path) the
    # wrapper is instantiated directly to skip checkpoint state-dict loading. Both
    # branches mirror rank 0's structure because ``from_config`` raises the same
    # ``ValueError`` as ``from_pretrained`` when the config is not in the auto
    # mapping (e.g. VLMs), keeping the try/except fallback consistent across ranks.
    try:
        if use_meta:
            model = AutoModelForTokenClassification.from_config(
                config=model_config,
                dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
        return model
    except BaseException as e:
        if not is_trl_available():
            raise RuntimeError(
                f"model({local_path}) is not a value head model, please install trl to make it valid"
            ) from e

    assert is_trl_available()

    from trl import AutoModelForCausalLMWithValueHead

    if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
        module_class = AutoModelForVision2Seq
    else:
        module_class = AutoModelForCausalLM
    if use_meta:
        ori_model = module_class.from_config(
            config=model_config,
            dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
        )
    else:
        ori_model = module_class.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=model_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
        )
    # vlm models
    if hasattr(model_config, "text_config"):
        ori_model.config.hidden_size = model_config.text_config.hidden_size
    if use_meta:
        # Instantiate the wrapper directly on the meta device; skip
        # ``from_pretrained`` so no checkpoint state-dict is loaded — FSDP2 will
        # broadcast the materialized weights from rank 0. ``_init_weights`` is a
        # no-op for the default (``None``) ``v_head_init_strategy``.
        model = AutoModelForCausalLMWithValueHead(ori_model)
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(ori_model)
    patch_valuehead_model(model)
    return model


def _build_module(self):
    from verl.utils.model import get_hf_auto_model_class
    from verl.utils.torch_dtypes import PrecisionType

    torch_dtype = self.engine_config.model_dtype

    if torch_dtype is None:
        # if it is training, we force torch_dtype to fp32
        torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

    torch_dtype = PrecisionType.to_dtype(torch_dtype)

    major_capability, _ = torch.cuda.get_device_capability(0)
    use_meta = (
        (self.rank != 0 if self.device_mesh is None else self.device_mesh.get_coordinate()[-1] != 0)
        if self.engine_config.strategy == "fsdp2" and major_capability >= 9
        else False
    )

    init_context = torch.device("meta") if use_meta else torch.device("cpu")

    with init_context, warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if self.model_config.model_type == "language_model":
            auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)

            loading_kwargs = dict(
                dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            if use_meta:
                module = auto_class.from_config(**loading_kwargs)
            else:
                module = auto_class.from_pretrained(
                    pretrained_model_name_or_path=self.model_config.local_path,
                    **loading_kwargs,
                )
        else:
            assert (
                self.model_config.model_type == "value_model"
            ), f"Unsupported model type: {self.model_config.model_type}"
            self.model_config.hf_config.num_labels = 1
            self.model_config.hf_config.classifier_dropout = 0.0
            self.model_config.hf_config.hidden_dropout = "0"
            self.model_config.hf_config.summary_dropout_prob = 0.0
            module = load_valuehead_model(
                local_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                model_config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
                use_meta=use_meta,
            )

        use_liger = self.model_config.use_liger
        # Apply Liger kernel; disable fused_linear_cross_entropy (conflicts with verl's forward patching)
        if use_liger:
            from liger_kernel.transformers.monkey_patch import (
                _apply_liger_kernel_to_instance,
            )

            _apply_liger_kernel_to_instance(
                model=module,
                fused_linear_cross_entropy=False,
                swiglu=True,
            )

        fused_kernel_options = self.model_config.fused_kernel_options
        fused_kernels_backend = (
            fused_kernel_options.get("impl_backend", None)
            if fused_kernel_options is not None
            else None
        )

        use_fused_kernels = self.model_config.use_fused_kernels
        apply_monkey_patch(
            model=module,
            use_remove_padding=self.use_remove_padding,
            ulysses_sp_size=self.ulysses_sequence_parallel_size,
            use_fused_kernels=use_fused_kernels,
            fused_kernels_backend=fused_kernels_backend,
        )

        if self.model_config.enable_gradient_checkpointing:
            module.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    return module


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
                curr_pos_ids[:, curr_mask].contiguous().flatten()
            )  # (num_components * valid_len,)
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
                # position_ids stored as flattened 1D nested tensor: (num_components * total_nnz,)
                # reshape to (num_components, 1, total_nnz)
                flat_pos = position_ids.values()  # (num_components * total_nnz,)
                position_ids_rmpad = flat_pos.view(num_pos_components, -1).unsqueeze(1)
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
                # position_ids stored as flattened 1D nested: each sample has (num_components * seq_len,)
                # pad to (batch, num_components * max_seq_len), then reshape to (num_components, batch, max_seq_len)
                position_ids = (
                    torch.nested.to_padded_tensor(
                        position_ids,
                        padding=0,
                        output_size=(batch_size, num_pos_components * max_seq_len),
                    )
                    .view(batch_size, num_pos_components, max_seq_len)
                    .permute(1, 0, 2)
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
        engine._build_module = MethodType(_build_module, engine)
        engine.save_checkpoint = MethodType(save_checkpoint, engine)
        # Patch prepare_model_inputs to inject seq_idx/cu_seqlens for
        # packed-sequence models (e.g. Qwen3.5 GateDeltaNet).
        if isinstance(engine, FSDPEngineWithLMHead):
            engine.prepare_model_inputs = MethodType(prepare_model_inputs, engine)
        setattr(engine, "_patched", True)
