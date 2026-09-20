# -*- coding: utf-8 -*-
"""
CoD workflow for Random Alchemy crafting game.

Uses procedurally generated recipe graphs instead of Little Alchemy data.
Element names are already random, so no obfuscation needed.
"""

from typing import List, Optional, Tuple

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.connect_the_dots.alchemy.env import AlchemyEnv
from trinity.common.workflows.connect_the_dots.alchemy.prompts import (
    load_system_prompt,
    load_user_prompt,
)
from trinity.common.workflows.connect_the_dots.alchemy.random_graph import (
    RandomGraphGenerator,
)
from trinity.common.workflows.connect_the_dots.base_workflow import (
    AsyncCoDMultiStepWorkflow,
)
from trinity.common.workflows.connect_the_dots.utils import extract_content_between_keys
from trinity.common.workflows.workflow import Task


def parse_action(response: str) -> Optional[Tuple[str, str]]:
    """Parse action from model response."""
    key_start, key_end = "<answer>", "</answer>"
    content, success = extract_content_between_keys(response, key_start, key_end)
    if not success:
        return None
    action_str = content.strip()
    parts = action_str.split("+")
    if len(parts) != 2:
        return None
    elem1 = parts[0].strip()
    elem2 = parts[1].strip()
    if not elem1 or not elem2:
        return None
    return (elem1, elem2)


class CoDRandomAlchemyWorkflow(AsyncCoDMultiStepWorkflow):
    """CoD Random Alchemy workflow using procedurally generated graphs."""

    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
        use_openai_client: bool = False,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
            use_openai_client=use_openai_client,
        )

        workflow_args = task.workflow_args if hasattr(task, "workflow_args") else {}
        self.max_rounds = workflow_args.get("max_rounds", 15)
        self.max_response_tokens_restraint = workflow_args.get(
            "max_response_tokens_restraint", None
        )
        self.show_recipes = workflow_args.get("show_recipes", False)
        self.show_elements = workflow_args.get("show_elements", False)
        self.show_tiers = workflow_args.get("show_tiers", True)
        self.material_mode = workflow_args.get("material_mode", "unlimited")
        self.scope = workflow_args.get("scope", "per_task")

        # Graph generation params
        self.min_num_tiers = workflow_args.get("min_num_tiers", 3)
        self.max_num_tiers = workflow_args.get("max_num_tiers", 4)
        self.min_base_elements = workflow_args.get("min_base_elements", 10)
        self.max_base_elements = workflow_args.get("max_base_elements", 15)
        self.tier_shrink_min = workflow_args.get("tier_shrink_min", 0.4)
        self.tier_shrink_max = workflow_args.get("tier_shrink_max", 0.6)
        self.min_recipes_per_element = workflow_args.get("min_recipes_per_element", 2)
        self.max_recipes_per_element = workflow_args.get("max_recipes_per_element", 3)
        self.max_tier_gap = workflow_args.get("max_tier_gap", 2)
        self.min_cross_tier_prob = workflow_args.get("min_cross_tier_prob", 0.2)
        self.max_cross_tier_prob = workflow_args.get("max_cross_tier_prob", 0.35)
        self.min_noise_node_ratio = workflow_args.get("min_noise_node_ratio", 0.15)
        self.max_noise_node_ratio = workflow_args.get("max_noise_node_ratio", 0.25)
        self.noise_chain_depth = workflow_args.get("noise_chain_depth", 3)
        self.min_material_mult = workflow_args.get("min_material_mult", 1.5)
        self.max_material_mult = workflow_args.get("max_material_mult", 2.0)

        # Raw task params
        self.raw_task = task.raw_task if hasattr(task, "raw_task") else {}
        self.seed = self.raw_task.get("seed", 42)

        # Generate game instance
        gen = RandomGraphGenerator()
        gen_kwargs = dict(
            min_num_tiers=self.min_num_tiers,
            max_num_tiers=self.max_num_tiers,
            min_base_elements=self.min_base_elements,
            max_base_elements=self.max_base_elements,
            tier_shrink_min=self.tier_shrink_min,
            tier_shrink_max=self.tier_shrink_max,
            min_recipes_per_element=self.min_recipes_per_element,
            max_recipes_per_element=self.max_recipes_per_element,
            max_tier_gap=self.max_tier_gap,
            min_cross_tier_prob=self.min_cross_tier_prob,
            max_cross_tier_prob=self.max_cross_tier_prob,
            min_noise_node_ratio=self.min_noise_node_ratio,
            max_noise_node_ratio=self.max_noise_node_ratio,
            noise_chain_depth=self.noise_chain_depth,
            material_mode=self.material_mode,
            min_material_mult=self.min_material_mult,
            max_material_mult=self.max_material_mult,
        )

        if self.scope == "per_pack":
            pack_seed = self.raw_task.get("pack_seed", self.seed)
            task_idx = self.raw_task.get("task_idx", 0)
            num_tasks = self.raw_task.get("pack_size", 8)
            self.game_instance = gen.generate_instance_for_pack(
                pack_seed=pack_seed,
                task_seed=self.seed,
                task_idx=task_idx,
                num_tasks=num_tasks,
                **gen_kwargs,
            )
        else:
            self.game_instance = gen.generate_instance(
                seed=self.seed,
                **gen_kwargs,
            )

        # Create environment
        self.env = AlchemyEnv(self.game_instance, max_rounds=self.max_rounds,
                              material_mode=self.material_mode)

        # State
        self.done: bool = False
        self.final_reward: float = 0.0
        self.current_step: int = 0
        self.action_feedback: Optional[str] = None
        self.early_termination_by_format_issue: bool = False

    def _build_system_prompt(self) -> str:
        target = self.game_instance.target

        kwargs = {
            "show_recipes": self.show_recipes,
            "show_elements": self.show_elements,
            "show_tiers": self.show_tiers,
            "obfuscate_names": True,  # names are random, same prompt style
            "material_mode": self.material_mode,
            "target": target,
        }
        if self.show_recipes:
            kwargs["recipes"] = self.game_instance.recipes
        if self.show_elements:
            kwargs["all_elements"] = sorted(self.game_instance.all_elements)
        if self.show_tiers:
            kwargs["tier_info"] = self._build_tier_info()

        sys_prompt = load_system_prompt(**kwargs)

        sys_prompt = self._augment_system_prompt(sys_prompt)
        return sys_prompt

    def _build_tier_info(self) -> str:
        local_tiers = self.game_instance.local_tiers or {}
        tier_groups: dict = {}
        for elem, tier in sorted(local_tiers.items(), key=lambda x: x[1]):
            tier_groups.setdefault(tier, []).append(elem)

        lines = []
        for tier in sorted(tier_groups.keys()):
            elems = tier_groups[tier]
            # label = f"tier {tier}" if tier == 0 else f"tier {tier} (needs {tier} synthesis step{'s' if tier > 1 else ''})"
            label = f"tier {tier}"
            lines.append(f"- {label}: {', '.join(elems)}")
        target = self.game_instance.target
        target_tier = local_tiers.get(target, "?")
        lines.append(f"\nThe target element '{target}' is at tier {target_tier}.")
        return "\n".join(lines)

    def _build_action_feedback(
        self,
        action: Tuple[str, str],
        result: Optional[str],
        invalid: bool,
    ) -> str:
        e1, e2 = action
        if invalid:
            inv = self.env.inventory
            if e1 not in inv and e2 not in inv:
                return f"'{e1}' and '{e2}' are not in your inventory."
            elif e1 not in inv:
                return f"'{e1}' is not in your inventory."
            elif e2 not in inv:
                return f"'{e2}' is not in your inventory."
            elif e1 == e2:
                return f"Not enough '{e1}' (need 2, have {inv.get(e1, 0)})."
            return f"Not enough materials for '{e1}' + '{e2}'."
        if result is not None:
            return f"You combined {e1} + {e2} and created: {result}!"
        if self.material_mode == "unlimited":
            return f"{e1} + {e2} — nothing happened."
        return f"{e1} + {e2} — nothing happened. Materials consumed."

    async def run_async(self) -> List[Experience]:
        obs, info = self.env.reset()
        self.done = False
        self.final_reward = 0.0
        self.current_step = 0
        self.action_feedback = None
        self.early_termination_by_format_issue = False

        self.memory.clear()
        self.memory.append({"role": "system", "content": self._build_system_prompt()})

        return await super().run_async()

    async def step_async(self, step_num: int) -> Tuple[bool, List[Experience]]:
        if self.done:
            return False, []

        target = self.game_instance.target
        inv = self.env.inventory

        user_content = load_user_prompt(
            current_round=step_num + 1,
            max_rounds=self.max_rounds,
            inventory=inv,
            target=target,
            action_feedback=self.action_feedback,
            material_mode=self.material_mode,
        )

        if self.icl_examples and step_num == 0:
            user_content = f"{user_content}\n\nHere are some reference examples:\n\n{self.icl_examples}"

        self.memory.append({"role": "user", "content": user_content})

        if self.reply_prefix:
            self.memory.append({"role": "assistant", "content": self.reply_prefix})

        experiences = await self.model.chat_async(self.memory)
        response_text = experiences[0].response_text
        self.memory.append({"role": "assistant", "content": response_text})

        sys_prompt = self.memory[0]["content"] if self.memory and self.memory[0]["role"] == "system" else ""
        for exp in experiences:
            exp.info["sys_prompt"] = sys_prompt
            exp.info["user_prompt"] = user_content

        action = parse_action(response_text)

        if action is None:
            self.action_feedback = (
                "Invalid format: could not parse combination. "
                "Expected format: {your reasoning process here}<answer>Element1 + Element2</answer>. Game over."
            )
            self.early_termination_by_format_issue = True
            self.done = True
            self.current_step = step_num + 1
            return False, experiences

        e1, e2 = action

        # Case-insensitive match to inventory
        inv_lower = {x.lower(): x for x in self.env.inventory}
        e1_canonical = inv_lower.get(e1.lower(), e1)
        e2_canonical = inv_lower.get(e2.lower(), e2)

        obs, reward, terminated, truncated, info = self.env.step(
            (e1_canonical, e2_canonical)
        )

        self.action_feedback = self._build_action_feedback(
            (e1_canonical, e2_canonical),
            info.get("last_result"),
            invalid=info.get("invalid_action", False),
        )

        self.done = terminated or truncated
        self.current_step = step_num + 1

        if terminated and reward > 0:
            self.final_reward = reward

        return not self.done, experiences

    def _get_feedback(self) -> str:
        target = self.game_instance.target

        if target in self.env.inventory:
            return f"Success! Synthesized {target}."
        if self.early_termination_by_format_issue:
            return self.action_feedback
        if len(self.env.get_feasible_actions()) == 0:
            return (
                f"Failed: Ran out of materials. Could not synthesize {target}. "
                f"Used {self.current_step} rounds."
            )
        return (
            f"Failed: Ran out of rounds ({self.current_step}/{self.max_rounds}). "
            f"Could not synthesize {target}."
        )

    @property
    def max_step_num(self) -> int:
        return self.max_rounds
