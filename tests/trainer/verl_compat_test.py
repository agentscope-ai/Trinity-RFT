"""CPU regression tests for the verl 0.9 worker configuration/input contracts."""
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

pytest.importorskip("verl", minversion="0.9.0")

from verl.trainer.config import CheckpointConfig  # noqa: E402
from verl.utils.config import omega_conf_to_dataclass  # noqa: E402
from verl.workers.config.checkpoint import McoreCheckpointConfig  # noqa: E402
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead  # noqa: E402

from trinity.common.config import Config  # noqa: E402
from trinity.trainer.verl import monkey_patch  # noqa: E402
from trinity.trainer.verl.config import (  # noqa: E402
    _build_actor_config,
    _build_critic_config,
    _build_ref_config,
)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "megatron"])
@pytest.mark.parametrize("role", ["actor", "ref", "critic"])
def test_worker_checkpoint_and_disabled_profiler_configs(strategy, role):
    """Build the exact nested dataclasses consumed by each worker role."""
    config = Config()
    sections = {
        "actor": _build_actor_config(config, strategy, total_training_steps=10),
        "ref": _build_ref_config(config, strategy),
        "critic": _build_critic_config(config, strategy, use_critic=True, total_training_steps=10),
    }
    section = sections[role]
    checkpoint = omega_conf_to_dataclass(OmegaConf.create(section["checkpoint"]))
    expected_contents = ["model"] if role == "ref" else ["model", "optimizer", "extra"]
    assert checkpoint.save_contents == expected_contents
    assert checkpoint.load_contents == expected_contents
    assert checkpoint.async_save is False
    if strategy == "megatron":
        assert isinstance(checkpoint, McoreCheckpointConfig)
        assert checkpoint.mbridge_config == {
            "distributed_filesystem": True,
            "memory_efficient": True,
            "strict": False,
        }
    else:
        assert type(checkpoint) is CheckpointConfig
        assert "mbridge_config" not in section["checkpoint"]

    profiler = omega_conf_to_dataclass(OmegaConf.create(section["profiler"]))
    assert profiler.enable is False
    assert profiler.tool is None
    # TrainingWorker performs this lookup even when profiling is disabled.
    assert profiler.tool_config.get(profiler.tool, {}) == {}


def make_micro_batch():
    """Create two packed CPU sequences without loading a model or starting Ray."""
    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.tensor([1, 2, 3, 4, 5]), offsets=torch.tensor([0, 3, 5])
    )
    position_ids = torch.nested.nested_tensor_from_jagged(
        torch.tensor([0, 1, 2, 0, 1]), offsets=input_ids.offsets()
    )
    return TensorDict(
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "temperature": torch.tensor([1.0, 2.0]),
        },
        batch_size=[2],
    )


def test_packed_outputs_without_sequence_parallelism(monkeypatch):
    """SP=1 round-trips through verl's output preparation with zero padding."""
    # Some test environments have flash-attn installed; keep this a CPU-only test.
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "1")
    batch = make_micro_batch()
    engine = SimpleNamespace(use_ulysses_sp=False)
    model_inputs, output_args = monkey_patch.prepare_model_inputs(engine, batch)
    assert output_args["pad_size"] == 0
    torch.testing.assert_close(
        model_inputs["cu_seq_lens_q"], torch.tensor([0, 3, 5], dtype=torch.int32)
    )
    torch.testing.assert_close(
        model_inputs["seq_idx"], torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int32)
    )

    engine._gather_and_unpad_packed = mock.Mock(side_effect=lambda value, pad_size: value)
    logits = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) / 10
    output = FSDPEngineWithLMHead.prepare_model_outputs(
        engine, SimpleNamespace(logits=logits), output_args, batch, logits_processor_func=None
    )
    expected = torch.log_softmax(logits.squeeze(0) / torch.tensor([1, 1, 1, 2, 2])[:, None], dim=-1)
    expected = expected.gather(1, torch.tensor([2, 3, 4, 5, 1])[:, None]).squeeze(1)
    torch.testing.assert_close(output["log_probs"].values(), expected)
    torch.testing.assert_close(output["log_probs"].offsets(), batch["input_ids"].offsets())
    assert engine._gather_and_unpad_packed.call_args.args[1] == 0


@pytest.mark.parametrize("is_vlm", [False, True])
def test_sequence_parallel_padding_is_preserved(is_vlm):
    """Nonzero SP padding still extends sequence metadata for text and VLM paths."""
    config = SimpleNamespace(vision_config={}) if is_vlm else SimpleNamespace()
    engine = SimpleNamespace(
        use_ulysses_sp=True, ulysses_sequence_parallel_size=2, module=SimpleNamespace(config=config)
    )

    def pad_inputs(inputs, position_ids_rmpad=None, **kwargs):
        padding = 1
        padded_inputs = torch.nn.functional.pad(
            inputs, (0, padding), value=kwargs.get("pad_value", 0)
        )
        padded_positions = (
            None
            if position_ids_rmpad is None
            else torch.nn.functional.pad(position_ids_rmpad, (0, padding))
        )
        return padded_inputs, padded_positions, padding

    with mock.patch.object(monkey_patch, "ulysses_pad", side_effect=pad_inputs), mock.patch.object(
        monkey_patch, "ulysses_pad_and_slice_inputs", side_effect=pad_inputs
    ):
        model_inputs, output_args = monkey_patch.prepare_model_inputs(engine, make_micro_batch())

    assert output_args["pad_size"] == 1
    assert output_args["input_ids_rmpad_rolled"].shape == (6,)
    torch.testing.assert_close(
        model_inputs["cu_seq_lens_q"], torch.tensor([0, 3, 6], dtype=torch.int32)
    )
    torch.testing.assert_close(
        model_inputs["seq_idx"], torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int32)
    )
