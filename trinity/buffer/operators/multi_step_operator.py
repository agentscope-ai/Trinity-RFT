from typing import List, Tuple

import torch

from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import EID, Experience, group_by
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class MultiStepPadding(ExperienceOperator):
    """
    Padding experiences of one run to the max step.

    Note: This operator assumes that the reward is already calculated and stored in the Experience object.
    """

    def __init__(self, max_steps: int = 0):
        self.max_steps = max_steps

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        """Padding each rollout to the max step."""
        logger.debug(f"Processing {len(exps)} experiences")
        total_num_placeholder_exps = 0
        all_exps = []

        task_exps = group_by(exps, "task")
        for _, task_exp in task_exps.items():
            run_exps = group_by(task_exp, "run")
            for _, exps_same_run in run_exps.items():
                if len(exps_same_run) == 0:
                    continue
                num_placeholder_exps = 0
                if len(exps_same_run) < self.max_steps:
                    num_placeholder_exps = self.max_steps - len(exps_same_run)
                    # Calculate average response length to keep metrics unchanged
                    assert all(
                        exp.tokens is not None for exp in exps_same_run
                    ), "Tokens are not provided"
                    response_lengths = [
                        len(exp.tokens) - exp.prompt_length for exp in exps_same_run  # type: ignore
                    ]
                    avg_response_length = int(sum(response_lengths) / len(response_lengths))
                    # Ensure at least 1 to avoid zero-length response
                    avg_response_length = max(avg_response_length, 1)

                    # Use the first experience as a template
                    template_exp = exps_same_run[0]
                    prompt_length = template_exp.prompt_length

                    # Create tokens with average response length
                    # Keep the prompt part, pad the response part to average length
                    prompt_tokens = template_exp.tokens[:prompt_length]  # type: ignore
                    # Use the last token of prompt as padding token for response part
                    pad_token = prompt_tokens[-1] if len(prompt_tokens) > 0 else 0
                    response_tokens = torch.full(
                        (avg_response_length,),
                        pad_token,
                        dtype=template_exp.tokens.dtype,  # type: ignore
                    )
                    avg_tokens = torch.cat([prompt_tokens, response_tokens])
                    avg_logprobs = (
                        torch.zeros(avg_response_length, dtype=torch.float32)
                        if template_exp.logprobs is not None
                        else None
                    )
                    assert all(
                        exp.reward is not None for exp in exps_same_run
                    ), "Rewards are not provided"
                    rewards = [exp.reward for exp in exps_same_run if exp.reward is not None]
                    avg_reward = sum(rewards) / len(rewards)

                    template_eid = template_exp.eid

                    empty_experiences = [
                        Experience(
                            eid=EID(
                                batch=template_eid.batch,
                                task=template_eid.task,
                                run=template_eid.run,
                                step=-1,
                            ),  # -1 means placeholder
                            tokens=avg_tokens,
                            logprobs=avg_logprobs,
                            prompt_length=prompt_length,
                            action_mask=torch.zeros(avg_response_length, dtype=torch.bool),
                            truncate_status="prompt_truncated",  # TODO: merge with the following
                            info={"status": "placeholder"},  # TODO: use another field
                            reward=avg_reward,
                        )
                        for _ in range(num_placeholder_exps)
                    ]
                    logger.debug(f"Adding {num_placeholder_exps} placeholder experiences")
                    # Put empty at the beginning, as the adv is computed using the last exp
                    exps_same_run = empty_experiences + exps_same_run
                    all_exps.extend(exps_same_run)
                else:
                    all_exps.extend(exps_same_run)
                total_num_placeholder_exps += num_placeholder_exps
        metrics = {"total_num_placeholder_exps": total_num_placeholder_exps}
        logger.debug(f"After padding: {len(all_exps)}")
        return all_exps, metrics
