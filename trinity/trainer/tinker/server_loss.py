"""Build inputs and metrics for the optional server-side Trinity PPO loss."""

from typing import Dict, List

import torch
from tinker import types


def with_trinity_ppo_inputs(
    batch: List[types.Datum], model_inputs: List[dict], kl_coef: float
) -> List[types.Datum]:
    """Align response-only tensors with the shifted target tokens.

    Args:
        batch: Datums produced by `to_tinker_input`.
        model_inputs: Corresponding response masks, old logprobs and advantages.
        kl_coef: Nonzero when reference logprobs must be included.

    Returns:
        New datums with PPO inputs; the input datums are not mutated.

    Raises:
        ValueError: Required response data is absent or inconsistent.
    """
    if len(batch) != len(model_inputs):
        raise ValueError("Server loss needs one set of model inputs per datum.")
    output = []
    for datum, inputs in zip(batch, model_inputs):
        mask = inputs["action_mask"].to(dtype=torch.float32, device="cpu")
        target_length = len(datum.loss_fn_inputs["target_tokens"].data)
        response_length = mask.numel()
        if mask.ndim != 1 or not 0 < response_length <= target_length:
            raise ValueError("Server loss requires a nonempty response mask matching the targets.")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("trinity_ppo requires a binary action mask.")

        def padded(key: str, allow_scalar: bool = False) -> torch.Tensor:
            if key not in inputs:
                raise ValueError(f"trinity_ppo requires {key}.")
            values = torch.as_tensor(inputs[key], dtype=torch.float32, device="cpu").reshape(-1)
            if allow_scalar and values.numel() == 1:
                values = values.expand(response_length)
            if values.numel() != response_length:
                raise ValueError(f"{key} must have one value per response token.")
            values = values.masked_fill(~mask.bool(), 0)
            if not torch.isfinite(values).all():
                raise ValueError(f"{key} must be finite on active response tokens.")
            result = torch.zeros(target_length, dtype=torch.float32)
            result[-response_length:] = values
            return result

        loss_inputs = dict(datum.loss_fn_inputs)
        # Use the actual action mask, including masked response tokens, rather
        # than inferring it from sequence lengths or nonzero advantages.
        weights = torch.zeros(target_length, dtype=torch.float32)
        weights[-response_length:] = mask
        loss_inputs["weights"] = weights
        loss_inputs["logprobs"] = padded("old_logprob")
        loss_inputs["advantages"] = padded("advantages", allow_scalar=True)
        if kl_coef > 0:
            loss_inputs["ref_logprobs"] = padded("ref_logprob")
        output.append(types.Datum(model_input=datum.model_input, loss_fn_inputs=loss_inputs))
    return output


def trinity_ppo_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Derive token-weighted diagnostics from additive server statistics."""
    output = {}
    if "loss:sum" in metrics:
        output["actor/final_loss"] = metrics["loss:sum"]
    count = metrics.get("trinity/response_tokens:sum", 0)
    if count <= 0:
        return output
    for source, destination in (
        ("trinity/ratio_sum:sum", "actor/ratio_mean"),
        ("trinity/clipped_tokens:sum", "actor/ratio_clip_fraction"),
        ("trinity/kl_sum:sum", "actor/kl_token_mean"),
    ):
        if source in metrics:
            output[destination] = metrics[source] / count
    if "trinity/ratio_squared_sum:sum" in metrics and "actor/ratio_mean" in output:
        output["actor/ratio_var"] = max(
            0.0,
            metrics["trinity/ratio_squared_sum:sum"] / count - output["actor/ratio_mean"] ** 2,
        )
    return output
