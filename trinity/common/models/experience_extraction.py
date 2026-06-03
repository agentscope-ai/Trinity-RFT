import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybase64
import torch
from torch import Tensor
from transformers import AutoConfig

from trinity.common.experience import Experience
from trinity.common.models.mm_utils import combine_output_token_ids


def get_routed_experts_layout(
    model_path: str, trust_remote_code: bool = True
) -> Optional[Tuple[int, int]]:
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    text_config = getattr(hf_config, "text_config", hf_config)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    topk = getattr(text_config, "num_experts_per_tok", None)
    if num_layers is None or topk is None:
        return None
    return int(num_layers), int(topk)


def decode_sglang_routed_experts(
    routed_experts_value: Any,
    total_tokens: int,
    layout: Tuple[int, int],
) -> Optional[Tensor]:
    if routed_experts_value is None:
        return None
    if isinstance(routed_experts_value, torch.Tensor):
        return routed_experts_value.to(torch.uint8)
    if not isinstance(routed_experts_value, str):
        return torch.tensor(routed_experts_value, dtype=torch.uint8)

    decoded = pybase64.b64decode_as_bytearray(routed_experts_value)
    routed_experts = torch.frombuffer(decoded, dtype=torch.int32)
    num_layers, topk = layout
    seq_length = max(total_tokens - 1, 0)
    expected_numel = seq_length * num_layers * topk
    if routed_experts.numel() != expected_numel:
        raise ValueError(
            "Unexpected routed_experts size from SGLang: "
            f"expected {expected_numel} elements for shape ({seq_length}, {num_layers}, {topk}), "
            f"got {routed_experts.numel()}"
        )
    return routed_experts.reshape(seq_length, num_layers, topk).to(torch.uint8)


def decode_vllm_routed_experts(routed_experts_value: str | None) -> Optional[Tensor]:
    if routed_experts_value is None:
        return None

    decoded = pybase64.b64decode_as_bytearray(routed_experts_value)
    routed_experts = np.load(io.BytesIO(decoded), allow_pickle=False)
    return torch.as_tensor(routed_experts, dtype=torch.uint8)


def convert_api_output_to_experience(
    output,
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
    routed_experts_layout: Optional[Tuple[int, int]] = None,
) -> List[Experience]:
    """Convert a non-stream API output to a list of experiences.

    Args:
        output: Completion output from API client.
        multi_modal_inputs: Optional training-time multimodal tensors aligned
            with the prompt tokens.
        routed_experts_layout: Optional `(num_layers, topk)` layout used to
            decode routed experts.
    """
    return _convert_completion_output_to_experience(
        output,
        multi_modal_inputs=multi_modal_inputs,
        routed_experts_layout=routed_experts_layout,
    )


class HistoryRecordingStream:  # TODO: add multi-modal support
    def __init__(self, stream, history: List[Experience], is_async: bool = False) -> None:
        self._stream = stream
        self._history = history
        self._chunks = []
        self._recorded = False
        self._is_async = is_async
        if is_async:
            self._iterator = stream.__aiter__()
        else:
            self._iterator = iter(stream)

    def __iter__(self):
        if self._is_async:
            raise TypeError("Use 'async for' for async streams.")
        return self

    def __next__(self):
        if self._is_async:
            raise TypeError("Use 'async for' for async streams.")
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._record_history_once()
            raise
        self._chunks.append(chunk)
        return chunk

    def close(self) -> None:
        if self._is_async:
            raise TypeError("Use 'aclose' for async streams.")
        self._record_history_once()
        close_fn = getattr(self._stream, "close", None)
        if callable(close_fn):
            close_fn()

    def __aiter__(self):
        if not self._is_async:
            raise TypeError("Use 'for' for sync streams.")
        return self

    async def __anext__(self):
        if not self._is_async:
            raise TypeError("Use 'for' for sync streams.")
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._record_history_once()
            raise
        self._chunks.append(chunk)
        return chunk

    async def aclose(self) -> None:
        if not self._is_async:
            raise TypeError("Use 'close' for sync streams.")
        self._record_history_once()
        close_fn = getattr(self._stream, "aclose", None)
        if callable(close_fn):
            close_result = close_fn()
            if hasattr(close_result, "__await__"):
                await close_result
            return
        close_fn = getattr(self._stream, "close", None)
        if callable(close_fn):
            close_fn()

    def _record_history_once(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        if self._chunks:
            self._history.extend(_convert_stream_chunks_to_experience(self._chunks))

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _convert_completion_output_to_experience(
    output,
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
    routed_experts_layout: Optional[Tuple[int, int]] = None,
) -> List[Experience]:
    return [
        Experience(
            tokens=torch.cat(
                (
                    torch.tensor(output.prompt_token_ids, dtype=torch.int32),
                    torch.tensor(choice.token_ids, dtype=torch.int32),
                )
            ),
            logprobs=extract_logprobs(choice),
            prompt_length=len(output.prompt_token_ids),
            response_text=getattr(choice.message, "content", None),
            routed_experts=_extract_completion_routed_experts(
                output,
                choice,
                total_tokens=len(output.prompt_token_ids) + len(choice.token_ids),
                routed_experts_layout=routed_experts_layout,
            ),
            multi_modal_inputs=combine_output_token_ids(choice.token_ids, multi_modal_inputs),
        )
        for choice in output.choices
    ]


def _convert_stream_chunks_to_experience(chunks: Sequence[Any]) -> List[Experience]:
    prompt_token_ids: Optional[List[int]] = None
    by_choice: Dict[int, Dict[str, Any]] = {}

    for chunk in chunks:
        if prompt_token_ids is None and hasattr(chunk, "prompt_token_ids"):
            chunk_prompt_token_ids = getattr(chunk, "prompt_token_ids", None)
            if chunk_prompt_token_ids is not None:
                prompt_token_ids = list(chunk_prompt_token_ids)

        for choice in getattr(chunk, "choices", []) or []:
            idx = getattr(choice, "index", 0)
            if idx not in by_choice:
                by_choice[idx] = {
                    "token_ids": [],
                    "logprobs": [],
                    "response_text_parts": [],
                }
            data = by_choice[idx]

            token_ids = getattr(choice, "token_ids", None)
            if token_ids is not None:
                data["token_ids"].extend(token_ids)

            choice_logprobs = getattr(choice, "logprobs", None)
            if (
                choice_logprobs is not None
                and getattr(choice_logprobs, "content", None) is not None
            ):
                for token_logprob in choice_logprobs.content:
                    data["logprobs"].append(token_logprob.logprob)
                    if token_ids is None:
                        token_id = getattr(token_logprob, "token_id", None)
                        if token_id is not None:
                            data["token_ids"].append(token_id)

            delta = getattr(choice, "delta", None)
            if delta is not None:
                delta_content = getattr(delta, "content", None)
                if isinstance(delta_content, str) and len(delta_content) > 0:
                    data["response_text_parts"].append(delta_content)

    prompt_token_ids = prompt_token_ids or []
    exps: List[Experience] = []
    for idx in sorted(by_choice.keys()):
        data = by_choice[idx]
        response_token_ids = data["token_ids"]
        if len(response_token_ids) == 0:
            continue
        response_text = "".join(data["response_text_parts"])
        exps.append(
            Experience(
                tokens=torch.tensor(prompt_token_ids + response_token_ids, dtype=torch.int32),
                logprobs=torch.tensor(data["logprobs"], dtype=torch.float32),
                prompt_length=len(prompt_token_ids),
                response_text=response_text,
            )
        )
    return exps


def _extract_completion_routed_experts(
    output,
    choice,
    total_tokens: int,
    routed_experts_layout: Optional[Tuple[int, int]] = None,
) -> Optional[Tensor]:
    routed_experts_value = getattr(choice, "routed_experts", None)
    if routed_experts_value is not None:
        try:
            return decode_vllm_routed_experts(routed_experts_value)
        except (ValueError, OSError):
            return None

    if routed_experts_layout is None:
        return None

    if not hasattr(output, "sglext") or "routed_experts" not in output.sglext:
        return None
    routed_experts_value = output.sglext.get("routed_experts", None)
    try:
        return decode_sglang_routed_experts(
            routed_experts_value,
            total_tokens,
            layout=routed_experts_layout,
        )
    except ValueError:
        return None


def extract_logprobs(choice) -> Tensor:
    if not hasattr(choice, "logprobs") or choice.logprobs is None:
        return torch.tensor([], dtype=torch.float32)
    return torch.tensor(
        [logprob.logprob for logprob in choice.logprobs.content],
        dtype=torch.float32,
    )
