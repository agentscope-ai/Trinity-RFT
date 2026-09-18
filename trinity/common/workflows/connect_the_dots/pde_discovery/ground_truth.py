# -*- coding: utf-8 -*-
"""Hidden ground-truth sampling for PDE discovery."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from .candidate import (
    DEFAULT_DICTIONARY,
    reaction_value,
)
from . import pde_numeric


DEFAULT_GT_FAMILY_DIR = os.path.join(
    os.path.dirname(__file__),
    "ground_truth_families",
)
DEFAULT_HIDDEN_GT_FAMILY_FILE = "physical_general.json"
COEFFICIENT_ABS_MIN = 0.5
COEFFICIENT_ABS_MAX = 2.0
GT_STABILITY_MAX_ATTEMPTS = 128
GT_STABILITY_GRID_SIZE = 65
GT_STABILITY_TIME_STEPS = 401


def _serialized_templates(templates: List[dict]) -> tuple:
    return tuple(
        (
            tuple(template["terms"]),
            tuple(
                (term, float(template.get("signs", {}).get(term, 1.0)))
                for term in template["terms"]
            ),
        )
        for template in templates
    )


@lru_cache(maxsize=None)
def _calibrated_family_initial_amplitude_upper(
    serialized_templates: tuple,
    min_terms: int,
    max_terms: int,
    state_abs_limit: float,
) -> float:
    """Find one conservative amplitude bound shared by a GT family."""
    nx = GT_STABILITY_GRID_SIZE
    nt = GT_STABILITY_TIME_STEPS
    x_grid = np.linspace(0.0, 1.0, nx)
    dt = 1.0 / (nt - 1)
    dx = 1.0 / (nx - 1)
    ratio = dt / (dx * dx)
    interior = nx - 2
    lower = -ratio * np.ones(interior - 1, dtype=float)
    diag = (1.0 + 2.0 * ratio) * np.ones(interior, dtype=float)
    upper = -ratio * np.ones(interior - 1, dtype=float)

    # Cover the corners of the allowed coefficient box for every eligible support.
    coefficient_vertices = []
    for terms, raw_signs in serialized_templates:
        if not min_terms <= len(terms) <= max_terms:
            continue
        signs = dict(raw_signs)
        for magnitudes in product(
            (COEFFICIENT_ABS_MIN, COEFFICIENT_ABS_MAX),
            repeat=len(terms),
        ):
            coefficient_vertices.append(
                {
                    term: (1.0 if signs[term] >= 0.0 else -1.0) * magnitude
                    for term, magnitude in zip(terms, magnitudes)
                }
            )
    if not coefficient_vertices:
        raise ValueError("Ground-truth family has no templates to calibrate")

    state_limit = float(state_abs_limit)
    if not np.isfinite(state_limit) or state_limit <= 0.0:
        raise ValueError("state_abs_limit must be finite and positive")
    # The first Dirichlet mode is the broadest profile and receives the least
    # diffusion damping among the admissible sine modes.
    base_profile = np.sin(np.pi * x_grid)

    def family_is_stable(amplitude: float) -> bool:
        for coefficients in coefficient_vertices:
            state = amplitude * base_profile
            for _ in range(nt - 1):
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    rhs = state[1:-1] + dt * reaction_value(
                        state[1:-1], coefficients
                    )
                if not np.all(np.isfinite(rhs)):
                    return False
                next_state = pde_numeric.solve_tridiagonal(
                    lower, diag, upper, rhs
                )
                if (
                    not np.all(np.isfinite(next_state))
                    or np.any(np.abs(next_state) >= state_limit)
                ):
                    return False
                state[1:-1] = next_state
        return True

    safe = 0.0
    unsafe = state_limit
    resolution = state_limit / (GT_STABILITY_GRID_SIZE - 1)
    while unsafe - safe > resolution:
        midpoint = (safe + unsafe) / 2.0
        if family_is_stable(midpoint):
            safe = midpoint
        else:
            unsafe = midpoint
    if safe <= 0.0:
        raise ValueError("Ground-truth family has no positive stable amplitude")
    return safe


def calibrate_family_initial_amplitude_upper(
    templates: List[dict],
    min_terms: int,
    max_terms: int,
    state_abs_limit: float,
) -> float:
    """Return a cached, ground-truth-independent bound for one family."""
    return _calibrated_family_initial_amplitude_upper(
        _serialized_templates(templates),
        int(min_terms),
        int(max_terms),
        float(state_abs_limit),
    )


def load_hidden_gt_templates(
    dictionary_terms: List[str],
    family: str = DEFAULT_HIDDEN_GT_FAMILY_FILE,
) -> List[dict]:
    family_file = str(family).strip()
    if family_file.endswith(".json"):
        family_file = family_file[:-5]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", family_file):
        raise ValueError(f"Invalid hidden ground-truth family: {family!r}")
    path = os.path.join(DEFAULT_GT_FAMILY_DIR, f"{family_file}.json")
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    raw_templates = payload.get("templates", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_templates, list):
        raise ValueError(f"Hidden GT family file has no template list: {path}")

    templates = []
    for raw_template in raw_templates:
        if not isinstance(raw_template, dict):
            continue
        terms = [
            str(term)
            for term in raw_template.get("terms", [])
            if str(term) in DEFAULT_DICTIONARY and str(term) in dictionary_terms
        ]
        terms = list(dict.fromkeys(terms))
        if not terms:
            continue
        raw_signs = raw_template.get("signs", {})
        signs = {}
        if isinstance(raw_signs, dict):
            for term in terms:
                try:
                    sign = float(raw_signs.get(term, 0.0))
                except (TypeError, ValueError):
                    sign = 0.0
                signs[term] = 1.0 if sign >= 0.0 else -1.0
        templates.append({"terms": terms, "signs": signs})
    if not templates:
        raise ValueError(f"Hidden GT family file has no usable templates: {path}")
    return templates


def sample_hidden_reaction(
    rng: np.random.Generator,
    templates: List[dict],
    min_terms: int,
    max_terms: int,
    template_index: int | None = None,
) -> Tuple[List[str], Dict[str, float]]:
    eligible = [
        template
        for template in templates
        if min_terms <= len(template["terms"]) <= max_terms
    ]
    if not eligible:
        raise ValueError(
            "Hidden GT family has no templates compatible with "
            f"min_reaction_terms={min_terms}, max_reaction_terms={max_terms}."
        )
    if template_index is None:
        template = eligible[int(rng.integers(0, len(eligible)))]
    else:
        if not 0 <= template_index < len(eligible):
            raise ValueError(
                f"ground_truth_template_index={template_index} is outside the "
                f"eligible template range [0, {len(eligible) - 1}]"
            )
        template = eligible[template_index]
    terms = sorted(list(template["terms"]), key=DEFAULT_DICTIONARY.index)
    signs = template.get("signs", {})
    coefficients = {}
    for term in terms:
        sign = float(signs.get(term, 0.0))
        if abs(sign) <= 1e-12:
            sign = 1.0 if rng.random() < 0.5 else -1.0
        magnitude = round(
            float(rng.uniform(COEFFICIENT_ABS_MIN, COEFFICIENT_ABS_MAX)),
            2,
        )
        coefficients[term] = (1.0 if sign >= 0.0 else -1.0) * magnitude
    return terms, coefficients


def is_stable_reaction_candidate(
    coefficients: Dict[str, float],
    pde_grid_size: int,
    pde_time_steps: int,
    trajectory_count: int,
    pack_seed: int,
    initial_amplitude_upper: float,
    initial_condition_shape: pde_numeric.InitialConditionShapeConfig,
    state_abs_limit: float,
) -> bool:
    nx = max(17, min(pde_grid_size, GT_STABILITY_GRID_SIZE))
    nt = max(17, min(pde_time_steps, GT_STABILITY_TIME_STEPS))
    x_grid = np.linspace(0.0, 1.0, nx)
    t_grid = np.linspace(0.0, 1.0, nt)
    dx = float(x_grid[1] - x_grid[0])
    dt = float(t_grid[1] - t_grid[0])
    r = dt / (dx * dx)

    interior = nx - 2
    lower = -r * np.ones(interior - 1, dtype=float)
    diag = (1.0 + 2.0 * r) * np.ones(interior, dtype=float)
    upper = -r * np.ones(interior - 1, dtype=float)

    clip_limit = float(state_abs_limit)
    if not np.isfinite(clip_limit) or clip_limit <= 0.0:
        raise ValueError("state_abs_limit must be finite and positive")
    for traj_idx in range(trajectory_count):
        u_grid = np.zeros((nt, nx), dtype=float)
        u_grid[0] = pde_numeric.initial_condition(
            x_grid,
            pack_seed=pack_seed,
            trajectory_index=traj_idx,
            trajectory_count=trajectory_count,
            amplitude_upper=initial_amplitude_upper,
            shape_config=initial_condition_shape,
        )
        u_grid[:, 0] = 0.0
        u_grid[:, -1] = 0.0
        if not np.all(np.isfinite(u_grid[0])):
            return False

        for step in range(nt - 1):
            rhs_raw = u_grid[step, 1:-1] + dt * reaction_value(
                u_grid[step, 1:-1],
                coefficients,
            )
            if (
                not np.all(np.isfinite(rhs_raw))
                or np.any(rhs_raw <= -clip_limit)
                or np.any(rhs_raw >= clip_limit)
            ):
                return False
            u_next = pde_numeric.solve_tridiagonal(lower, diag, upper, rhs_raw)
            if (
                not np.all(np.isfinite(u_next))
                or np.any(u_next <= -clip_limit)
                or np.any(u_next >= clip_limit)
            ):
                return False
            u_grid[step + 1, 1:-1] = u_next
    return True
