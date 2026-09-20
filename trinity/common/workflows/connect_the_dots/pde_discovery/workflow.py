# -*- coding: utf-8 -*-
"""CoD PDE discovery workflow.

This is a lightweight implementation of the PDE discovery demo in
``pde_discovery_in_CoD``. The environment accepts XML actions in plain-text
responses, keeps raw sampled points in a stateful pack buffer, and
lets the outer CoD workflow autonomously update the context after each task.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.connect_the_dots.base_workflow import (
    AsyncCoDMultiStepWorkflow,
)
from trinity.common.workflows.connect_the_dots.pde_discovery import candidate as candidate_utils
from trinity.common.workflows.connect_the_dots.pde_discovery import ground_truth
from trinity.common.workflows.connect_the_dots.pde_discovery import pde_numeric
from trinity.common.workflows.connect_the_dots.pde_discovery import regression
from trinity.common.workflows.connect_the_dots.pde_discovery.candidate import (
    DEFAULT_DICTIONARY,
)
from trinity.common.workflows.connect_the_dots.pde_discovery.prompts import (
    load_system_prompt,
    load_user_prompt,
)
from trinity.common.workflows.connect_the_dots.utils import parse_xml_answer
from trinity.common.workflows.workflow import Task


KAPPA_CAP = regression.KAPPA_CAP
XML_LIST_TAGS = {"dictionary", "candidate_equations", "uncertain_terms"}

logger = logging.getLogger(__name__)


@dataclass
class PDEPackState:
    """Persistent state shared by tasks in one CoD pack."""

    datasets: Dict[str, dict] = field(default_factory=dict)
    sample_counter: int = 0
    reaction_terms: List[str] = field(default_factory=list)
    reaction_coefficients: Dict[str, float] = field(default_factory=dict)
    ground_truth_template_index: Optional[int] = None
    environment_seed: Optional[int] = None
    latest_regression_equation: str = ""
    preferred_equation: str = ""
    preferred_support: List[str] = field(default_factory=list)
    preferred_coefficients: Dict[str, float] = field(default_factory=dict)
    dense_field: Optional[dict] = None
    last_context_update: Dict[str, object] = field(default_factory=dict)
    evidence_history: List[dict] = field(default_factory=list)
    next_task_position: int = 1


def _normalize_point_grid(
    raw_grid: object,
) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(raw_grid, dict):
        return None

    raw_points = raw_grid.get("point")
    if isinstance(raw_points, dict):
        raw_points = [raw_points]
    if not isinstance(raw_points, list):
        return None

    try:
        x_values = [float(point["x"]) for point in raw_points]
        t_values = [float(point["t"]) for point in raw_points]
    except (KeyError, TypeError, ValueError):
        return None
    if not x_values or not t_values:
        return None
    if any(not 0.0 < value < 1.0 for value in x_values):
        return None
    if any(not 0.0 <= value <= 1.0 for value in t_values):
        return None
    if len(x_values) != len(t_values):
        return None
    return list(zip(x_values, t_values))


class CoDPDEDiscoveryWorkflow(AsyncCoDMultiStepWorkflow):
    """CoD workflow for discovering nonlinear PDE reaction terms."""

    is_async: bool = True
    # use weakref to avoid memory leakage
    _PACK_STATES: weakref.WeakValueDictionary[str, PDEPackState] = (
        weakref.WeakValueDictionary()
    )

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
        self.reset(task)

    def reset(self, task: Task):
        if isinstance(getattr(task, "workflow_args", None), dict):
            task.workflow_args.setdefault("context_compression_mode", "keep_all")
        super().reset(task)
        args = task.workflow_args if hasattr(task, "workflow_args") else {}
        self.dictionary_terms = args.get("dictionary_terms", DEFAULT_DICTIONARY)
        self.min_reaction_terms = max(1, int(args.get("min_reaction_terms", 1)))
        self.max_reaction_terms = max(
            self.min_reaction_terms,
            int(args.get("max_reaction_terms", 2)),
        )
        self.dictionary_terms = [
            str(term) for term in self.dictionary_terms if str(term) in DEFAULT_DICTIONARY
        ] or list(DEFAULT_DICTIONARY)
        self.ground_truth_family = str(
            args.get("ground_truth_family", ground_truth.DEFAULT_HIDDEN_GT_FAMILY_FILE)
        )
        self.initial_condition_shape = (
            pde_numeric.InitialConditionShapeConfig.from_mapping(
                args.get("initial_condition_shape")
            )
        )
        if "pde_state_abs_limit" not in args:
            raise ValueError("pde_state_abs_limit workflow argument is required")
        self.pde_state_abs_limit = float(args["pde_state_abs_limit"])
        if (
            not math.isfinite(self.pde_state_abs_limit)
            or self.pde_state_abs_limit <= 0.0
        ):
            raise ValueError("pde_state_abs_limit must be finite and positive")
        self.hidden_gt_templates = self._load_hidden_gt_templates()
        self.initial_amplitude_upper = (
            ground_truth.calibrate_family_initial_amplitude_upper(
                self.hidden_gt_templates,
                self.min_reaction_terms,
                self.max_reaction_terms,
                state_abs_limit=self.pde_state_abs_limit,
            )
        )
        self.max_steps = args.get("max_steps", 10)
        self.point_budget = int(args.get("point_budget", 15))
        self.default_blind_penalty = args.get("blind_submission_penalty", 100.0)
        self.kappa_threshold = float(args.get("kappa_threshold", 100.0))
        if not math.isfinite(self.kappa_threshold) or self.kappa_threshold <= 0.0:
            raise ValueError("kappa_threshold must be a finite positive number")
        self.noise_level = float(args.get("noise_level", 0.002))
        if not math.isfinite(self.noise_level) or self.noise_level < 0.0:
            raise ValueError("noise_level must be a finite non-negative number")
        self.pde_grid_size = int(args.get("pde_grid_size", 257))
        self.pde_time_steps = int(args.get("pde_time_steps", 2001))
        self.seed = int(self.raw_task.get("seed", 42))
        self.task_position = (
            int(self.raw_task.get("task_idx", task.index.get("index", 0))) + 1
        )
        self.pack_size = int(self.raw_task.get("pack_size", args.get("pack_size", 1)))
        if self.pack_size <= 0:
            raise ValueError("PDE task pack_size must be positive")
        self.trajectory_count = self.pack_size
        trajectory_index = (max(1, self.task_position) - 1) % self.trajectory_count
        self.current_trajectory_id = f"traj_{trajectory_index}"
        self.pack_key = self._pack_key()

        if self.task_position == 1:
            self.pack_state = PDEPackState()
            self._PACK_STATES[self.pack_key] = self.pack_state
        elif self.pack_key in self._PACK_STATES:
            self.pack_state = self._PACK_STATES[self.pack_key]
        else:
            raise ValueError(
                "PDE pack tasks must be processed sequentially from task_idx=0"
            )
        if self.task_position != self.pack_state.next_task_position:
            raise ValueError(
                "PDE pack task_idx values must be sequential without gaps or repeats"
            )
        self._ensure_hidden_reaction()
        self.pack_state.next_task_position += 1

        self.budget_remaining = self.point_budget
        self.done = False
        self.current_step = 0
        self.action_feedback: Optional[str] = None
        self.final_reward = 0.0
        self.last_sampled_max_error: Optional[float] = None
        self.hidden_formula_l1_error: Optional[float] = None
        self.hidden_formula_extra_support_penalty: Optional[float] = None
        self.hidden_formula_reward_loss: Optional[float] = None
        self.last_regression_dataset_id = "merged_all"
        self.last_regression_dictionary = []
        self.last_candidate_diagnostics: List[dict] = []
        self.has_task_regression = False
        self.has_task_explicit_sampling = False
        self.has_task_context_update = False
        self.early_termination_by_format_issue = False
        self.dump_trajectories = bool(args.get("dump_trajectories", True))
        self.trajectory_dump_path = self._trajectory_dump_path(args)

    def _pack_key(self) -> str:
        if "pack_seed" in self.raw_task:
            # Runtime-injected pack_seed is the complete pack identity. Ignore
            # legacy dataset pack_id so pre-generated rows remain safe when
            # shuffled, mixed with another domain, or repacked at a new size.
            return f"pde-seed-{self.raw_task['pack_seed']}"
        return f"pde-pack-{self.task.batch_id}-{self.task.task_id}"

    def _environment_seed(self) -> int:
        """Return an optional benchmark-controlled seed hidden from the model."""
        return int(
            self.raw_task.get(
                "pde_environment_seed",
                self.raw_task.get("pack_seed", self.seed),
            )
        )

    def _support_size_hint(self) -> str:
        lower = max(1, min(self.min_reaction_terms, len(self.dictionary_terms)))
        upper = max(lower, min(self.max_reaction_terms, len(self.dictionary_terms)))
        if lower == upper:
            return f"exactly {lower}"
        return f"{lower} to {upper}"

    def _ensure_hidden_reaction(self) -> None:
        """Create the hidden sparse PDE reaction for this CoD pack.

        The target is deliberately kept out of task rows and prompts.  It is
        regenerated from the pack seed inside the environment so the agent can
        only infer it through sampled-data feedback.
        """
        raw_template_index = self.raw_task.get("ground_truth_template_index")
        template_index = (
            int(raw_template_index) if raw_template_index is not None else None
        )
        environment_seed = self._environment_seed()
        if self.pack_state.reaction_coefficients:
            if template_index != self.pack_state.ground_truth_template_index:
                raise ValueError(
                    "Every task in a PDE eval pack must use the same "
                    "ground_truth_template_index"
                )
            if environment_seed != self.pack_state.environment_seed:
                raise ValueError(
                    "Every task in a PDE eval pack must use the same "
                    "pde_environment_seed"
                )
            return
        pack_seed = environment_seed
        seed_sequence = np.random.SeedSequence(
            [pack_seed, pde_numeric.GROUND_TRUTH_STREAM]
        )
        rng = np.random.default_rng(seed_sequence)
        min_terms = max(1, min(self.min_reaction_terms, len(self.dictionary_terms)))
        max_terms = max(min_terms, min(self.max_reaction_terms, len(self.dictionary_terms)))
        terms: List[str] = []
        coefficients: Dict[str, float] = {}
        max_attempts = ground_truth.GT_STABILITY_MAX_ATTEMPTS
        for attempt in range(max_attempts):
            terms, coefficients = self._sample_hidden_reaction(
                rng,
                min_terms,
                max_terms,
                template_index=template_index,
            )
            if self._is_stable_reaction_candidate(coefficients):
                break
        else:
            logger.warning(
                "No stable hidden PDE reaction found after %d attempts; "
                "using the last sampled template candidate.",
                max_attempts,
            )

        self.pack_state.reaction_terms = terms
        self.pack_state.reaction_coefficients = coefficients
        self.pack_state.ground_truth_template_index = template_index
        self.pack_state.environment_seed = environment_seed

    def _load_hidden_gt_templates(self) -> List[dict]:
        return ground_truth.load_hidden_gt_templates(
            self.dictionary_terms,
            family=self.ground_truth_family,
        )

    def _sample_hidden_reaction(
        self,
        rng: np.random.Generator,
        min_terms: int,
        max_terms: int,
        template_index: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        return ground_truth.sample_hidden_reaction(
            rng,
            self.hidden_gt_templates,
            min_terms,
            max_terms,
            template_index=template_index,
        )

    def _is_stable_reaction_candidate(
        self,
        coefficients: Dict[str, float],
    ) -> bool:
        return ground_truth.is_stable_reaction_candidate(
            coefficients=coefficients,
            pde_grid_size=self.pde_grid_size,
            pde_time_steps=self.pde_time_steps,
            trajectory_count=self.trajectory_count,
            pack_seed=self._environment_seed(),
            initial_amplitude_upper=self.initial_amplitude_upper,
            initial_condition_shape=self.initial_condition_shape,
            state_abs_limit=self.pde_state_abs_limit,
        )

    def _trajectory_dump_path(self, args: dict) -> str:
        root = args.get("checkpoint_job_dir") or os.path.join(
            "logs", "research_cod", "pde_discovery"
        )
        dump_dir = os.path.join(str(root), "trajectory_dumps")
        raw_uid = str(self.raw_task.get("uid", f"task_{self.task.task_id}"))
        safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_uid)
        safe_pack = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.pack_key)
        file_name = (
            f"{safe_pack}_{safe_uid}_pos{self.task_position}_"
            f"pid{os.getpid()}_obj{id(self)}.jsonl"
        )
        return os.path.join(dump_dir, file_name)

    def _dump_trajectory_record(self, record: dict) -> None:
        if not self.dump_trajectories:
            return
        try:
            os.makedirs(os.path.dirname(self.trajectory_dump_path), exist_ok=True)
            payload = {
                "uid": self.raw_task.get("uid"),
                "pack_key": self.pack_key,
                "task_position": self.task_position,
                "event": record.get("event"),
                "step": record.get("step"),
                "budget_remaining": self.budget_remaining,
                "latest_regression_equation": self.pack_state.latest_regression_equation,
                "current_equation": self.pack_state.preferred_equation,
                "reward_snapshot": self.final_reward,
                "format_error": self.early_termination_by_format_issue,
                **record,
            }
            with open(self.trajectory_dump_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write PDE trajectory dump: %s", exc)

    async def run_async(self) -> List[Experience]:
        self.memory.clear()
        sys_prompt = load_system_prompt(
            dictionary_terms=self.dictionary_terms,
            support_size_hint=self._support_size_hint(),
        )
        sys_prompt = self._augment_system_prompt(sys_prompt)
        self.memory.append({"role": "system", "content": sys_prompt})
        try:
            return await super().run_async()
        except BaseException:
            self._PACK_STATES.pop(self.pack_key, None)
            raise

    async def step_async(self, step_num: int) -> Tuple[bool, List[Experience]]:
        if self.done:
            return False, []

        user_content = load_user_prompt(
            task_position=self.task_position,
            pack_size=self.pack_size,
            current_step=step_num + 1,
            max_steps=self.max_steps,
            point_budget=self.point_budget,
            budget_remaining=self.budget_remaining,
            task_description=self.task_desc or "",
            available_datasets=sorted(self.pack_state.datasets.keys()),
            last_sampled_max_error=self.last_sampled_max_error,
            evidence_memory=self.pack_state.last_context_update.get(
                "evidence_memory", {}
            ),
            action_feedback=self.action_feedback,
        )
        if self.icl_examples and step_num == 0:
            user_content = (
                f"{user_content}\n\nHere are some reference examples:\n\n"
                f"{self.icl_examples}"
            )

        self.memory.append({"role": "user", "content": user_content})
        if self.reply_prefix:
            self.memory.append({"role": "assistant", "content": self.reply_prefix})

        experiences = await self.model.chat_async(self.memory, **self.rollout_args)
        response_text = experiences[0].response_text or ""
        self.memory.append({"role": "assistant", "content": response_text})

        sys_prompt = self.memory[0]["content"] if self.memory else ""
        for exp in experiences:
            exp.info["sys_prompt"] = sys_prompt
            exp.info["user_prompt"] = user_content

        action, parse_error = parse_xml_answer(response_text, XML_LIST_TAGS)
        if action is None:
            response_excerpt = response_text.replace("\n", "\\n")[:500]
            logger.warning(
                "PDE discovery format error (%s). "
                "Response excerpt: %s",
                parse_error,
                response_excerpt,
            )
            for exp in experiences:
                exp.info["pde_parse_error"] = parse_error
                exp.info["pde_parse_error_response_excerpt"] = response_excerpt
            self.action_feedback = (
                "Invalid format: expected exactly one XML action wrapped in "
                "<answer>...</answer>. Game over."
            )
            self.final_reward = 0.0
            self.done = True
            self.current_step = step_num + 1
            self.early_termination_by_format_issue = True
            terminated = True
            format_error = True
            feedback = self.action_feedback
        else:
            feedback, terminated, format_error = self._execute_action(action)
            self.action_feedback = feedback
            self.done = terminated
            self.current_step = step_num + 1
            self.early_termination_by_format_issue = format_error
            if format_error:
                self.final_reward = 0.0

        self._dump_trajectory_record(
            {
                "event": "step",
                "step": step_num + 1,
                "response": response_text,
                "action": action,
                "action_feedback": feedback,
                "terminated": terminated,
                "format_error": format_error,
                "parse_error": parse_error,
                "reward_snapshot": self.final_reward,
            }
        )
        return not self.done and self.current_step < self.max_steps, experiences

    def _execute_action(self, payload: dict) -> Tuple[str, bool, bool]:
        action = payload.get("action")
        args = payload.get("args", {})
        if not isinstance(args, dict):
            return "Invalid action arguments. Game over.", True, True

        handlers = {
            "sample_pde_data": self._action_sample_pde_data,
            "summarize_pack_evidence": self._action_summarize_pack_evidence,
            "run_sparse_regression": self._action_run_sparse_regression,
            "update_scientific_context": self._action_update_scientific_context,
        }
        handler = handlers.get(action)
        if handler is None:
            return f"Invalid action: unknown action '{action}'. Game over.", True, True
        return handler(args)

    def _action_summarize_pack_evidence(self, args: dict) -> Tuple[str, bool, bool]:
        reveal_value = args.get("reveal_equations", "false")
        if str(reveal_value).lower() not in {"true", "false"}:
            return "Error: reveal_equations must be true or false.", False, False
        reveal_equations = str(reveal_value).lower() == "true"
        history = list(self.pack_state.evidence_history)
        if not history:
            return (
                "Pack evidence summary: no prior task evidence has been recorded "
                "in this pack. Start with current-task measurements.",
                False,
                False,
            )

        slots: Dict[str, dict] = {}
        for entry in history:
            equation = str(entry.get("preferred_equation") or "").strip()
            if not equation:
                continue
            if equation not in slots:
                slots[equation] = {
                    "slot_id": f"H{len(slots) + 1}",
                    "equation": equation,
                    "tasks": set(),
                }
            slot = slots[equation]
            if entry.get("task_position") is not None:
                slot["tasks"].add(str(entry["task_position"]))

        if not slots:
            return (
                "Pack evidence summary: prior notes exist, but no tested "
                "candidate slots are available. Use current-task actions to build evidence.",
                False,
                False,
            )

        lines = [
            "Pack evidence summary; historical hypotheses, not ground truth.",
        ]
        for equation, slot in slots.items():
            pieces = [
                f"{slot['slot_id']}: tasks={','.join(sorted(slot['tasks'])) or 'unknown'}",
            ]
            if reveal_equations:
                pieces.append(f"agent_recorded_equation={slot['equation']}")
            lines.append("; ".join(pieces) + ".")

        if not reveal_equations:
            lines.append(
                "Call summarize_pack_evidence with reveal_equations=true only if "
                "you need to actively reuse prior agent-recorded candidates."
            )
        return "\n".join(lines), False, False

    def _action_sample_pde_data(self, args: dict) -> Tuple[str, bool, bool]:
        has_points = args.get("points") is not None
        has_point_grid = args.get("point_grid") is not None
        if has_points:
            return (
                "Error: explicit points are not supported. Use point_grid with "
                "one point element per numeric x and t coordinate pair.",
                False,
                False,
            )
        grid_points = _normalize_point_grid(args.get("point_grid"))
        if not has_point_grid:
            return (
                "Error: sample_pde_data supports only point_grid. Provide "
                "one point element per numeric x and t coordinate pair.",
                False,
                False,
            )
        if has_point_grid and grid_points is None:
            return (
                "Error: each point in point_grid must contain numeric x and t "
                "values with 0 < x < 1 and 0 <= t <= 1.",
                False,
                False,
            )

        grid_sample_points = grid_points or []
        num_points = len(grid_sample_points)
        if num_points <= 0:
            return "Error: point_grid must produce at least one point.", False, False
        if num_points > self.budget_remaining:
            return (
                f"Error: requested {num_points} points but only "
                f"{self.budget_remaining} budget remains.",
                False,
                False,
            )

        trajectory_id = self.current_trajectory_id
        self.pack_state.sample_counter += 1
        dataset_id = f"ds_t{self.task_position}_{self.pack_state.sample_counter}"
        data = self._sample_grid_points(grid_sample_points, trajectory_id)
        x_values = data["x"]
        t_values = data["t"]
        region = {
            "x": [float(np.min(x_values)), float(np.max(x_values))],
            "t": [float(np.min(t_values)), float(np.max(t_values))],
        }
        sampling_mode = "point_grid"
        self.has_task_explicit_sampling = True
        data["region"] = region
        data["sampling_strategy"] = sampling_mode
        data["task_position"] = self.task_position
        data["trajectory_id"] = np.array([trajectory_id] * num_points, dtype=object)
        self.pack_state.datasets[dataset_id] = data
        self.budget_remaining -= num_points

        t = data["t"]
        summary = (
            f"Sampled {num_points} points. Dataset: {dataset_id}. "
            f"Point budget remaining: {self.budget_remaining}/{self.point_budget}. "
            f"Sampling mode: {sampling_mode}. "
            f"Region: x in [{region['x'][0]:.3g}, {region['x'][1]:.3g}], "
            f"t in [{region['t'][0]:.3g}, {region['t'][1]:.3g}]. "
            f"mean t = {np.mean(t):.3g}. "
            f"Dataset coverage: {self._coverage_report(data)}."
        )
        return summary, False, False

    def _action_run_sparse_regression(self, args: dict) -> Tuple[str, bool, bool]:
        dataset_id = str(args.get("dataset_id", "merged_all"))
        data = self._resolve_dataset(dataset_id)
        if data is None:
            return f"Error: dataset_id '{dataset_id}' was not found.", False, False

        raw_dictionary = args.get("dictionary")
        raw_candidates = args.get("candidate_equations")
        has_candidates = isinstance(raw_candidates, list) and bool(raw_candidates)
        if has_candidates and len(raw_candidates) > 8:
            return "Error: candidate_equations accepts at most 8 equations.", False, False
        if raw_dictionary is None and not has_candidates:
            return (
                "Error: run_sparse_regression requires an agent-specified "
                "dictionary or candidate_equations.",
                False,
                False,
            )
        if raw_dictionary is not None and (
            not isinstance(raw_dictionary, list) or not raw_dictionary
        ):
            return (
                "Error: dictionary must be a non-empty list of supported terms.",
                False,
                False,
            )
        dictionary = self._sanitize_dictionary(raw_dictionary)
        if raw_dictionary is not None and not any(
            str(term) in self.dictionary_terms for term in raw_dictionary
        ):
            return (
                "Error: dictionary did not contain any supported basis terms.",
                False,
                False,
            )
        allowed_supports = None
        if has_candidates:
            candidate_strings = [
                str(candidate).strip()
                for candidate in raw_candidates[:8]
                if str(candidate).strip()
            ]
            allowed_supports = []
            for candidate in candidate_strings:
                support = self._candidate_support(candidate, dictionary)
                if support:
                    allowed_supports.append(support)
            if not allowed_supports:
                return (
                    "Error: candidate_equations did not contain any supported "
                    "dictionary terms.",
                    False,
                    False,
                )
            if raw_dictionary is None:
                support_terms = {
                    term for support in allowed_supports for term in support
                }
                dictionary = [
                    term
                    for term in self.dictionary_terms
                    if term in support_terms
                ]
        if not dictionary:
            return (
                "Error: run_sparse_regression has no selected supported terms "
                "to score.",
                False,
                False,
            )
        alpha_arg = args.get("alpha")
        threshold_arg = args.get("threshold")
        try:
            alpha = float(0.05 if alpha_arg is None else alpha_arg)
            threshold = float(0.05 if threshold_arg is None else threshold_arg)
        except (TypeError, ValueError):
            return (
                "Error: alpha and threshold must be numeric when provided.",
                False,
                False,
            )
        if not all(
            math.isfinite(value) and value >= 0.0 for value in (alpha, threshold)
        ):
            return (
                "Error: alpha and threshold must be finite non-negative numbers.",
                False,
                False,
            )
        result = self._regression_result(
            data["u"],
            data["y"],
            data["u"],
            data["y"],
            dictionary,
            alpha,
            threshold,
            allowed_supports=allowed_supports,
        )
        result["feedback"] += (
            f" Fit dataset coverage: {self._coverage_report(data)}. "
            "Objective scope: current sampled dataset only; use it as a local "
            "fit and conditioning check before committing an equation. "
            f"{self._point_error_report(data, result['coefficients'])}"
        )
        self.last_sampled_max_error = result["penalty"]
        self.last_regression_dataset_id = dataset_id
        self.last_regression_dictionary = list(dictionary)
        self.last_candidate_diagnostics = result.get("candidate_diagnostics", [])
        self.pack_state.latest_regression_equation = result["equation"]
        self.has_task_regression = True
        return result["feedback"], False, False

    def _task_local_observation_stats(
        self,
        data: Optional[dict],
    ) -> dict:
        if data is None:
            return {"n": 0, "u_span": 0.0}
        u_values = np.asarray(data.get("u", []), dtype=float)
        u_values = u_values[np.isfinite(u_values)]
        if len(u_values) == 0:
            return {"n": 0, "u_span": 0.0}
        return {
            "n": int(len(u_values)),
            "u_span": float(np.max(u_values) - np.min(u_values)),
        }

    def _structured_evidence_memory(
        self,
        *,
        preferred_diagnostics: dict,
        observation_stats: dict,
    ) -> dict:
        preferred_error = float(
            preferred_diagnostics.get("training_max_error", KAPPA_CAP)
        )

        return {
            "task_evidence_status": "recorded",
            "observation_summary": {
                "n": observation_stats.get("n", 0),
                "u_span": observation_stats.get("u_span", 0.0),
                "relative_margin": preferred_diagnostics.get("relative_margin", 0.0),
                "preferred_sampled_max_error": preferred_error,
            },
        }

    def _action_update_scientific_context(self, args: dict) -> Tuple[str, bool, bool]:
        short_note = re.sub(r"\s+", " ", str(args.get("note", "") or "")).strip()
        preferred_equation = str(args.get("preferred_equation", "")).strip()
        uncertain_terms = args.get("uncertain_terms", [])
        if not isinstance(uncertain_terms, list):
            uncertain_terms = [str(uncertain_terms)]
        if len(uncertain_terms) > 4:
            return "Error: uncertain_terms accepts at most 4 terms.", False, False
        uncertain_terms = [
            re.sub(r"\s+", " ", str(term or "")).strip()
            for term in uncertain_terms
            if str(term or "").strip()
        ]
        if not preferred_equation:
            return (
                "Error: update_scientific_context requires preferred_equation. "
                "Choose the equation you want to finalize from the sampled-data "
                "diagnostics. This may be the "
                "latest regression candidate or another candidate scored by "
                "run_sparse_regression.",
                False,
                False,
            )
        for field_name, equation in (
            ("preferred_equation", preferred_equation),
        ):
            symbolic_tokens = candidate_utils.symbolic_coefficient_tokens(equation)
            if symbolic_tokens:
                tokens = ", ".join(symbolic_tokens)
                return (
                    f"Error: {field_name} contains symbolic coefficient(s): "
                    f"{tokens}. update_scientific_context requires explicit "
                    "numeric coefficients from a tested candidate, not "
                    "placeholders such as c1/c2/a/b.",
                    False,
                    False,
                )
        if not self.has_task_explicit_sampling:
            return (
                "Error: update_scientific_context requires current-task "
                "sample_pde_data evidence in the current task. Choose "
                "point_grid coordinates, sample them, then run sparse "
                "regression before updating scientific context.",
                False,
                False,
            )
        if not self.has_task_regression:
            return (
                "Error: update_scientific_context requires at least one "
                "run_sparse_regression call in the current task. Run regression "
                "on an available dataset first so the task has a current "
                "equation and sampled-data evidence.",
                False,
                False,
            )
        preferred_coefficients = self._candidate_coefficients(preferred_equation)
        if not preferred_coefficients:
            return (
                "Error: preferred_equation did not parse into dictionary terms. "
                "Use an explicit equation built from supported dictionary terms "
                "with numeric coefficients.",
                False,
                False,
            )
        evaluated_candidates = [
            self._candidate_coefficients(str(item.get("equation", "")))
            for item in self.last_candidate_diagnostics
        ]
        if not any(
            all(
                math.isclose(
                    preferred_coefficients.get(term, 0.0),
                    candidate.get(term, 0.0),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                for term in self.dictionary_terms
            )
            for candidate in evaluated_candidates
        ):
            return (
                "Error: preferred_equation must match a candidate scored by the "
                "latest run_sparse_regression call.",
                False,
                False,
            )
        data = self._resolve_dataset(self.last_regression_dataset_id)
        if data is None:
            data = self._resolve_dataset("merged_all")
        preferred_diagnostics = self._candidate_objective_diagnostics(
            preferred_equation,
            data,
            self.last_regression_dictionary,
        )
        preferred_support = list(preferred_diagnostics["support"])
        observation_stats = self._task_local_observation_stats(data)
        reward_diagnostics = preferred_diagnostics
        self.last_sampled_max_error = float(reward_diagnostics["step_penalty"])
        hidden_reward_diagnostics = self._hidden_formula_reward_diagnostics(
            preferred_equation
        )
        self.hidden_formula_l1_error = float(
            hidden_reward_diagnostics["normalized_l1_error"]
        )
        self.hidden_formula_extra_support_penalty = float(
            hidden_reward_diagnostics["extra_support_penalty"]
        )
        self.hidden_formula_reward_loss = float(hidden_reward_diagnostics["loss"])
        self.pack_state.preferred_equation = preferred_equation
        self.pack_state.preferred_support = preferred_support
        self.pack_state.preferred_coefficients = dict(
            preferred_diagnostics["coefficients"]
        )
        self.final_reward = float(hidden_reward_diagnostics["reward"])
        evidence_memory = self._structured_evidence_memory(
            preferred_diagnostics=preferred_diagnostics,
            observation_stats=observation_stats,
        )
        preferred_diagnostics_summary = {
            "training_max_error": reward_diagnostics["training_max_error"],
            "condition_penalty": reward_diagnostics["condition_penalty"],
            "relative_margin": reward_diagnostics["relative_margin"],
            "second_best_equation": reward_diagnostics["second_best_equation"],
            "step_penalty": reward_diagnostics["step_penalty"],
        }
        self.pack_state.last_context_update = {
            "evidence_memory": evidence_memory,
            "preferred_equation": preferred_equation,
            "preferred_diagnostics": preferred_diagnostics_summary,
            "latest_regression_equation": self.pack_state.latest_regression_equation,
            "note": short_note,
            "uncertain_terms": [str(term) for term in uncertain_terms],
        }
        self.pack_state.evidence_history.append(
            {
                "task_position": self.task_position,
                "evidence_memory": dict(evidence_memory),
                "preferred_equation": preferred_equation,
            }
        )
        self.pack_state.evidence_history = self.pack_state.evidence_history[-self.pack_size :]
        self.has_task_context_update = True
        return (
            "Task terminated. "
            f"reward_equation={preferred_equation}; "
            f"sampled-data max error={reward_diagnostics['training_max_error']:.3g}; "
            f"condition penalty={reward_diagnostics['condition_penalty']:.3g}; "
            f"closest different-support candidate={reward_diagnostics['second_best_equation']}; "
            f"relative margin={reward_diagnostics['relative_margin']:.3g}.",
            True,
            False,
        )

    def _sample_grid_points(
        self,
        points: List[Tuple[float, float]],
        trajectory_id: str,
    ) -> dict:
        seed_sequence = np.random.SeedSequence(
            [
                self.seed,
                self.task_position,
                self.pack_state.sample_counter,
            ]
        )
        rng = np.random.default_rng(seed_sequence)
        self._ensure_dense_field()
        x = np.array([point[0] for point in points], dtype=float)
        t = np.array([point[1] for point in points], dtype=float)
        return self._sample_values_at_points(x, t, trajectory_id, rng)

    def _sample_values_at_points(
        self,
        x: np.ndarray,
        t: np.ndarray,
        trajectory_id: str,
        rng: np.random.Generator,
    ) -> dict:
        # The regression target is the scheme-consistent reaction residual,
        # sampled from the dense hidden PDE field rather than direct oracle f(u).
        u = self._interpolate_dense_field("u", x, t, trajectory_id)
        y = self._interpolate_dense_field("y", x, t, trajectory_id)
        point_count = len(x)
        u += rng.normal(0.0, self.noise_level, point_count)
        y += rng.normal(0.0, self.noise_level * 0.25, point_count)
        return {"x": x, "t": t, "u": u, "y": y}

    def _ensure_dense_field(self) -> None:
        if self.pack_state.dense_field is not None:
            return

        nx = max(17, self.pde_grid_size)
        nt = max(51, self.pde_time_steps)
        if nx % 2 == 0:
            nx += 1
        x_grid = np.linspace(0.0, 1.0, nx)
        t_grid = np.linspace(0.0, 1.0, nt)
        dx = float(x_grid[1] - x_grid[0])
        dt = float(t_grid[1] - t_grid[0])
        r = dt / (dx * dx)

        interior = nx - 2
        lower = -r * np.ones(interior - 1, dtype=float)
        diag = (1.0 + 2.0 * r) * np.ones(interior, dtype=float)
        upper = -r * np.ones(interior - 1, dtype=float)

        trajectories = {}
        for traj_idx in range(self.trajectory_count):
            trajectory_id = f"traj_{traj_idx}"
            trajectories[trajectory_id] = self._simulate_dense_trajectory(
                x_grid,
                t_grid,
                dx,
                dt,
                lower,
                diag,
                upper,
                traj_idx,
            )

        default_field = trajectories["traj_0"]
        self.pack_state.dense_field = {
            "trajectories": trajectories,
            "x_grid": x_grid,
            "t_grid": t_grid,
            "u": default_field["u"],
            "y": default_field["y"],
        }

    def _simulate_dense_trajectory(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        dx: float,
        dt: float,
        lower: np.ndarray,
        diag: np.ndarray,
        upper: np.ndarray,
        trajectory_index: int,
    ) -> dict:
        return pde_numeric.simulate_dense_trajectory(
            x_grid=x_grid,
            t_grid=t_grid,
            dx=dx,
            dt=dt,
            lower=lower,
            diag=diag,
            upper=upper,
            trajectory_index=trajectory_index,
            reaction_fn=self._reaction_value,
            pack_seed=self._environment_seed(),
            trajectory_count=self.trajectory_count,
            initial_amplitude_upper=self.initial_amplitude_upper,
            initial_condition_shape=self.initial_condition_shape,
            state_abs_limit=self.pde_state_abs_limit,
        )

    def _interpolate_dense_field(
        self,
        field_name: str,
        x: np.ndarray,
        t: np.ndarray,
        trajectory_id: str = "traj_0",
    ) -> np.ndarray:
        self._ensure_dense_field()
        return pde_numeric.interpolate_dense_field(
            self._trajectory_field(trajectory_id),
            field_name,
            x,
            t,
        )

    def _trajectory_field(self, trajectory_id: str) -> dict:
        self._ensure_dense_field()
        dense_field = self.pack_state.dense_field or {}
        trajectories = dense_field.get("trajectories")
        if isinstance(trajectories, dict) and trajectory_id in trajectories:
            return trajectories[trajectory_id]
        return dense_field

    def _sanitize_dictionary(self, dictionary: object) -> List[str]:
        return candidate_utils.sanitize_dictionary(dictionary, self.dictionary_terms)

    def _reaction_value(
        self, u: np.ndarray, coefficients: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        return candidate_utils.reaction_value(
            u,
            coefficients or self.pack_state.reaction_coefficients,
        )

    def _resolve_dataset(self, dataset_id: str) -> Optional[dict]:
        if dataset_id in self.pack_state.datasets:
            return self.pack_state.datasets[dataset_id]
        if dataset_id == "merged_all" or dataset_id == "merged":
            ids = sorted(self.pack_state.datasets)
        else:
            match = re.fullmatch(r"merged_t(\d+)_t(\d+)", dataset_id)
            if not match:
                return None
            lo, hi = int(match.group(1)), int(match.group(2))
            ids = [
                dsid
                for dsid, data in self.pack_state.datasets.items()
                if lo <= int(data.get("task_position", 0)) <= hi
            ]
        if not ids:
            return None
        merged = {
            key: np.concatenate([self.pack_state.datasets[dsid][key] for dsid in ids])
            for key in ["x", "t", "u", "y"]
        }
        if all("trajectory_id" in self.pack_state.datasets[dsid] for dsid in ids):
            merged["trajectory_id"] = np.concatenate(
                [self.pack_state.datasets[dsid]["trajectory_id"] for dsid in ids]
            )
        return merged

    def _coverage_report(self, data: dict) -> str:
        u = np.asarray(data.get("u", []), dtype=float)
        if len(u) == 0:
            return "empty dataset"
        x = np.asarray(data.get("x", []), dtype=float)
        t = np.asarray(data.get("t", []), dtype=float)
        pieces = [
            f"n={len(u)}",
            f"u=[{np.min(u):.3g},{np.max(u):.3g}]",
        ]
        if len(x):
            pieces.append(f"x_span={np.max(x) - np.min(x):.3g}")
        if len(t):
            pieces.append(f"t_span={np.max(t) - np.min(t):.3g}")
        return "; ".join(pieces)

    def _point_error_report(self, data: dict, coefficients: Dict[str, float]) -> str:
        x = np.asarray(data.get("x", []), dtype=float)
        t = np.asarray(data.get("t", []), dtype=float)
        u = np.asarray(data.get("u", []), dtype=float)
        y = np.asarray(data.get("y", []), dtype=float)
        if not (len(x) == len(t) == len(u) == len(y)) or len(u) == 0:
            return "Sampled-point errors: unavailable."
        pred = self._reaction_value(u, coefficients=coefficients)
        errors = np.abs(y - pred)
        rows = [
            (
                f"(x={float(x_i):.4g}, t={float(t_i):.4g}, "
                f"u={float(u_i):.4g}, y={float(y_i):.4g}, "
                f"pred={float(pred_i):.4g}, abs_error={float(err_i):.4g})"
            )
            for x_i, t_i, u_i, y_i, pred_i, err_i in zip(x, t, u, y, pred, errors)
        ]
        return "Sampled-point errors: " + "; ".join(rows) + "."

    def _hidden_formula_reward_diagnostics(self, candidate: str) -> dict:
        """Compute the training-only reward from hidden coefficient vectors."""
        pred_coefficients = self._candidate_coefficients(candidate)
        target_coefficients = self.pack_state.reaction_coefficients
        target_l1 = sum(abs(float(value)) for value in target_coefficients.values())
        diff_l1 = 0.0
        for term in self.dictionary_terms:
            diff_l1 += abs(
                float(pred_coefficients.get(term, 0.0))
                - float(target_coefficients.get(term, 0.0))
            )
        normalized_l1 = diff_l1 / max(1e-8, target_l1)
        pred_support_size = sum(
            1 for value in pred_coefficients.values() if abs(float(value)) > 1e-12
        )
        target_support_size = len(self.pack_state.reaction_terms)
        extra_support_penalty = float(
            max(0, pred_support_size - target_support_size)
        )
        loss = float(normalized_l1 + extra_support_penalty)
        return {
            "normalized_l1_error": float(normalized_l1),
            "extra_support_penalty": extra_support_penalty,
            "loss": loss,
            "reward": float(1.0 / (1.0 + loss)),
        }

    def _regression_result(
        self,
        u: np.ndarray,
        y: np.ndarray,
        u_objective: np.ndarray,
        y_objective: np.ndarray,
        dictionary: list,
        alpha: float,
        threshold: float,
        allowed_supports: Optional[List[List[str]]] = None,
    ) -> dict:
        return regression.sparse_regression_result(
            u=u,
            y=y,
            u_objective=u_objective,
            y_objective=y_objective,
            dictionary=dictionary,
            alpha=alpha,
            threshold=threshold,
            max_reaction_terms=self.max_reaction_terms,
            default_blind_penalty=self.default_blind_penalty,
            kappa_threshold=self.kappa_threshold,
            allowed_supports=allowed_supports,
        )

    def _candidate_coefficients(self, candidate: str) -> Dict[str, float]:
        return candidate_utils.candidate_coefficients(
            candidate,
            self.dictionary_terms,
        )

    def _candidate_support(self, candidate: str, dictionary: List[str]) -> List[str]:
        return candidate_utils.candidate_support(
            candidate,
            dictionary,
            self.dictionary_terms,
        )

    def _candidate_objective_diagnostics(
        self,
        candidate: str,
        sampled_data: dict,
        dictionary: List[str],
    ) -> dict:
        return regression.candidate_objective_diagnostics(
            candidate=candidate,
            data=sampled_data,
            objective_data=sampled_data,
            dictionary=dictionary,
            dictionary_terms=self.dictionary_terms,
            last_candidate_diagnostics=self.last_candidate_diagnostics,
            kappa_threshold=self.kappa_threshold,
        )

    def _get_feedback(self) -> str:
        if self.early_termination_by_format_issue:
            return self.action_feedback or "Invalid action."
        return "\n".join(
            [
                "Latest sampled-data diagnostic: "
                f"{self.last_sampled_max_error if self.last_sampled_max_error is not None else 'none'}",
                (
                    "Latest finalized equation: "
                    f"{self.pack_state.preferred_equation or 'none'}"
                ),
                (
                    "Latest regression candidate: "
                    f"{self.pack_state.latest_regression_equation or 'none'}"
                ),
            ]
        )

    def _hidden_accuracy_metrics(self) -> dict:
        if not self.has_task_context_update:
            return {
                "structure_recovery": 0.0,
                "coefficient_relative_error": 1.0,
            }
        coeffs = self.pack_state.preferred_coefficients
        target = self.pack_state.reaction_coefficients
        target_support = set(self.pack_state.reaction_terms)
        support = self.pack_state.preferred_support
        structure_recovery = float(set(support) == target_support)
        coefficient_error = 1.0
        if structure_recovery:
            coefficient_error = sum(
                abs(coeffs.get(term, 0.0) - target[term])
                / max(1e-6, abs(target[term]))
                for term in target_support
            ) / max(1, len(target_support))
        return {
            "structure_recovery": structure_recovery,
            "coefficient_relative_error": coefficient_error,
        }

    async def reward_async(self, exps: List[Experience]) -> float:
        if not self.early_termination_by_format_issue and not self.has_task_context_update:
            self.final_reward = 0.0
            if self.last_sampled_max_error is None:
                self.last_sampled_max_error = self.default_blind_penalty
            self.action_feedback = (
                self.action_feedback or ""
            ) + "\nTask ended without update_scientific_context; no scientific context was committed."
        hidden_metrics = self._hidden_accuracy_metrics()
        reward = await super().reward_async(exps)
        if not exps:
            if self.task_position >= self.pack_size:
                self._PACK_STATES.pop(self.pack_key, None)
            return reward

        metrics = exps[-1].metrics or {}
        metrics.update(
            {
                "pde_structure_recovery": hidden_metrics["structure_recovery"],
                "pde_coefficient_relative_error": hidden_metrics[
                    "coefficient_relative_error"
                ],
                "pde_sample_efficiency_points": float(
                    self.point_budget - self.budget_remaining
                ),
                "pde_last_sampled_max_error": float(
                    self.last_sampled_max_error
                    if self.last_sampled_max_error is not None
                    else self.default_blind_penalty
                ),
                "pde_hidden_formula_normalized_l1_error": float(
                    self.hidden_formula_l1_error
                    if self.hidden_formula_l1_error is not None
                    else self.default_blind_penalty
                ),
                "pde_hidden_formula_extra_support_penalty": float(
                    self.hidden_formula_extra_support_penalty
                    if self.hidden_formula_extra_support_penalty is not None
                    else 0.0
                ),
                "pde_hidden_formula_reward_loss": float(
                    self.hidden_formula_reward_loss
                    if self.hidden_formula_reward_loss is not None
                    else self.default_blind_penalty
                ),
                "pde_ground_truth_support_size": float(
                    len(self.pack_state.reaction_terms)
                ),
                "pde_ground_truth_template_index": float(
                    self.pack_state.ground_truth_template_index
                    if self.pack_state.ground_truth_template_index is not None
                    else -1
                ),
                "pde_ground_truth_template_instance": float(
                    self.raw_task.get("ground_truth_template_instance", -1)
                ),
                "pde_eval_pack_index": float(
                    self.raw_task.get("eval_pack_index", -1)
                ),
            }
        )
        support_size = len(self.pack_state.reaction_terms)
        metrics.update(
            {
                f"pde_structure_recovery_support_size_{support_size}": hidden_metrics[
                    "structure_recovery"
                ],
                f"pde_hidden_formula_l1_support_size_{support_size}": float(
                    self.hidden_formula_l1_error
                    if self.hidden_formula_l1_error is not None
                    else self.default_blind_penalty
                ),
                f"pde_hidden_formula_reward_support_size_{support_size}": float(
                    self.final_reward
                ),
            }
        )
        template_index = self.pack_state.ground_truth_template_index
        if template_index is not None:
            metrics.update(
                {
                    f"pde_structure_recovery_template_{template_index}": hidden_metrics[
                        "structure_recovery"
                    ],
                    f"pde_hidden_formula_l1_template_{template_index}": float(
                        self.hidden_formula_l1_error
                        if self.hidden_formula_l1_error is not None
                        else self.default_blind_penalty
                    ),
                    f"pde_hidden_formula_reward_template_{template_index}": float(
                        self.final_reward
                    ),
                }
            )
        exps[-1].metrics = metrics
        self._dump_trajectory_record(
            {
                "event": "final",
                "step": self.current_step,
                "reward": reward,
                "final_reward": self.final_reward,
                "feedback": self._get_feedback(),
                "metrics": dict(metrics),
            }
        )
        if self.task_position >= self.pack_size:
            self._PACK_STATES.pop(self.pack_key, None)
        return reward

    def _compress_memory(self) -> None:
        super()._compress_memory()
        old_feedback_cutoff = max(1, self.current_step - 1)
        seen_user_messages = 0
        for msg in self.memory:
            if msg.get("role") != "user":
                continue
            seen_user_messages += 1
            content = msg.get("content", "")
            if seen_user_messages >= old_feedback_cutoff or "Action feedback:" not in content:
                continue
            msg["content"] = (
                content.split("Action feedback:")[0]
                + "Action feedback:\n[History compressed.]"
            )

    @property
    def max_step_num(self) -> int:
        return self.max_steps
