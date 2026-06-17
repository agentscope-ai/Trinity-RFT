"""Utils for CoD (prompting and parsing)."""

import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from trinity.common.workflows.connect_the_dots.utils import extract_content_between_keys


def compute_part_summary(exps: List) -> dict:
    """Compute summary statistics (accuracy per interaction step) for a list of experiences."""
    step_rewards: Dict[int, List[float]] = defaultdict(list)
    all_rewards = []
    for exp in exps:
        reward = exp.reward
        all_rewards.append(reward)
        step_id = exp.eid.step
        if step_id is not None:
            step_rewards[step_id].append(reward)

    num_samples = len(all_rewards)
    overall_acc = sum(all_rewards) / num_samples if num_samples > 0 else 0.0

    per_step_acc = {}
    for sid in sorted(step_rewards.keys()):
        rs = step_rewards[sid]
        per_step_acc[str(sid)] = round(sum(rs) / len(rs), 4)

    return {
        "overall_acc": round(overall_acc, 4),
        "num_samples": num_samples,
        "per_interaction_step_acc": per_step_acc,
    }


HINTS_EXAMPLE_PROMPT = """As an example, hints might look like the following:
    Hints: when solving a mathematical calculation task, it is often beneficial to
    - Start with a list of known conditions;
    - Create a high-level plan for problem solving;
    - Constantly verify previous calculation steps;
    - Return the final answer only when sufficiently confident.
"""


class CoDPrompts:
    @staticmethod
    def sys_prompt_gen_hint_iteratively(hint_example: bool = False) -> str:
        """System prompt for iterative hint generation."""
        prompt = """You are a helpful assistant and a cross-task learning agent. Your responsibility is to solve **a sequence of different-but-related tasks** within the same environment, while **transferring informative hints across tasks** that can help achieve better task-solving performance.
        
## Problem setting

**In the current session, your job is to update the hints.**

To be concrete, the user provides:
1. Previous hints that were used to guide task solving;
2. A new task-solving trajectory.

Based on this information, you need to generate updated hints that can assist in solving future tasks within the same environment more effectively, by
- Preserving useful guidance and information from the previous hints, while discarding incorrect ones;
- Incorporating new lessons and information that can be learned from the new trajectory.

## General principles

- You are in an environment that might be stationary or non-stationary, and relation between tasks in this environment is initially unknown. One general principle is thus to generate **informative** hints that can serve as useful priors when solving a new task. If you are uncertain about the correctness or usefulness of certain hints, feel free to include them in your response and briefly mention uncertainty.
- There are at least two major categories of useful hints: (1) revealed information / clues about the environment that were initially unknown; (2) task-solving techniques with validated efficacy. With that said, your updated hints can certainly go beyond these two categories.

## Response format

First think about what hints to return, then provide your final answer. Format your complete response as follows:
```
[THINKING]
--- Start of updated hints ---
[HINTS]
--- End of updated hints ---
```
where [THINKING] and [HINTS] should be replaced with your actual thinking process and updated hints, respectively.

Other requirements:
- Keep your thinking process concise. Avoid overthinking.
- Keep your updated hints concise if possible, containing the most critical and helpful information. Avoid generic and uninformative statements. Avoid repeating the system prompt within the task-solving trajectory.
- Make sure that your updated hints can be potentially helpful for other related tasks, rather than specific to the task in the provided trajectory.
- Use standard Markdown format for your updated hints. Simpler structures (e.g., bullet points) are preferred. If you do need to use sections, you should start from the third level "###", while avoiding "#" and "##".
"""
        if hint_example:
            prompt = prompt + HINTS_EXAMPLE_PROMPT
        return prompt

    @staticmethod
    def user_prompt_gen_hint_iteratively(
        prev_hint: str,
        trajectory: str,
        reward: float,
        feedback: str,
    ) -> str:
        """User prompt for iterative hint generation.

        Args:
            prev_hint: Previous hints (empty string if first iteration)
            trajectory: The solution trajectory (includes task description and model responses)
            reward: The reward received (e.g., 1.0 for success, 0.0 for failure)
            feedback: Environment feedback (e.g., final state, error message)
        """
        if prev_hint:
            prev_hint_section = f"""--- Previous hints ---

{prev_hint}

"""
        else:
            prev_hint_section = """--- Previous hints ---

(No previous hint available)

"""

        prompt = f"""{prev_hint_section}
--- A new task-solving trajectory ---

{trajectory}

Reward: {reward}

Environment feedback: {feedback}

--- Your job ---

Now, please think through it and return your updated hints based on the above information, following the requirements in the system prompt.
"""
        # !!! For "cheating" in environments like frozenlake-obscure, add:
        # "Note that there were initially some hidden clues about the environment. If you can reveal some of them based on the provided trajectory, include them in your updated hints."

        return prompt

    @staticmethod
    def extract_hint(response: str) -> Tuple[str, bool]:
        """Extract hints from response.

        Returns: extracted hint (str) and indicator of successful parsing (bool)
        """

        """Response format from sys_prompt_gen_hint_iteratively:
        ```
        {your thinking process here}
        --- Start of updated hints ---
        {your updated hints, formatted as bullet points}
        --- End of updated hints ---
        ```
        """
        key_start = "--- Start of updated hints ---"
        key_end = "--- End of updated hints ---"
        content, success = extract_content_between_keys(response, key_start, key_end)

        if success:
            return "Hints: " + content, True
        return "Hints: no hint available.", False

        # # this version works, but easily lose control over response / hint length
        # key = "### Updated hints:"
        # idx = response.rfind(key)
        # if idx != -1:
        #     return "Hints: " + response[(idx + len(key)) :]
        # return "Hints: no hint available."


# Inline warning prepended to per-trajectory ground-truth reference when
# the workflow surfaces it via the feedback string. Kept brief on purpose;
# do not enumerate transferability hints — let the system prompt drive that.
GT_LEAKAGE_WARNING = (
    "The ground-truth reference below is case-specific. "
    "Updated hints will be applied to other cases, so abstract "
    "transferable insights and do not copy case-specific facts."
)


def format_gt_block_for_feedback(gt_reference: str) -> str:
    """Format a case-specific GT reference as a feedback suffix block.

    Workflows that want to surface a per-trajectory oracle signal to
    the iterative hint generator should append this block to their
    feedback string (via exp.info["feedback"]).

    Returns an empty string if `gt_reference` is empty, so callers can
    unconditionally concatenate without dragging in a warning that
    points at nothing.
    """
    if not gt_reference:
        return ""
    return (
        f"\n\n{GT_LEAKAGE_WARNING}\n\n"
        f"--- Ground truth reference (case-specific) ---\n{gt_reference}"
    )
