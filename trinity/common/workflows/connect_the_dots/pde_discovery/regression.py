# -*- coding: utf-8 -*-
"""Sparse-regression helpers for PDE discovery.

This module follows the SINDy-style sparse-identification pattern: build a
candidate-library matrix, select sparse supports, then refit coefficients on
the selected terms. This style of sparse regression is commonly used for
data-driven governing-equation and PDE discovery.

References: Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
"Discovering governing equations from data: Sparse identification of nonlinear
dynamical systems." arXiv preprint arXiv:1509.03580 (SINDy); Rudy, Samuel H.,
et al. "Data-driven discovery of partial differential equations." arXiv
preprint arXiv:1609.06401 (PDE-FIND).
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from . import candidate as candidate_utils


KAPPA_CAP = 1e12
DEBIASED_REFIT_ALPHA = 1e-8


def max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 1.0
    return float(np.max(np.abs(y_true - y_pred)))


def relative_error_margin(second_best_error: float, best_error: float) -> float:
    gap = max(0.0, float(second_best_error) - float(best_error))
    if best_error > 0.0:
        return float(gap / float(best_error))
    return float("inf") if gap > 0.0 else 0.0


def effective_kappa(u: np.ndarray, dictionary: List[str]) -> float:
    theta, _ = candidate_utils.dictionary_matrix(u, dictionary)
    if theta.shape[1] == 0:
        return KAPPA_CAP
    scales = np.linalg.norm(theta, axis=0)
    valid = scales > 1e-12
    if np.count_nonzero(valid) == 0:
        return KAPPA_CAP
    theta_norm = theta[:, valid] / scales[valid]
    try:
        singular_values = np.linalg.svd(theta_norm, compute_uv=False)
    except np.linalg.LinAlgError:
        return KAPPA_CAP
    if len(singular_values) == 0:
        return KAPPA_CAP
    sigma_min = float(singular_values[-1])
    sigma_max = float(singular_values[0])
    if sigma_min <= 1e-12:
        return KAPPA_CAP
    return min(float(sigma_max / sigma_min), KAPPA_CAP)


def sparse_regression_result(
    u: np.ndarray,
    y: np.ndarray,
    u_objective: np.ndarray,
    y_objective: np.ndarray,
    dictionary: list,
    alpha: float,
    threshold: float,
    max_reaction_terms: int,
    default_blind_penalty: float,
    kappa_threshold: float,
    allowed_supports: Optional[List[List[str]]] = None,
) -> dict:
    theta, terms = candidate_utils.dictionary_matrix(u, dictionary)
    if theta.shape[1] == 0:
        target_y = y_objective if len(y_objective) else y
        max_error = max_abs_error(target_y, np.zeros_like(target_y, dtype=float))
        penalty = max_error + float(default_blind_penalty)
        return {
            "support": [],
            "coefficients": {},
            "equation": "0",
            "max_error": max_error,
            "condition_ratio": 1.0,
            "kappa": KAPPA_CAP,
            "penalty": penalty,
            "reward_penalty": penalty,
            "relative_margin": 0.0,
            "second_best_max_error": float("nan"),
            "second_best_equation": "not evaluated: no valid dictionary terms",
            "diagnostics": {},
            "candidate_diagnostics": [],
            "feedback": "Sparse regression failed: no valid dictionary terms.",
        }

    dictionary_kappa = effective_kappa(u, terms)
    scales = np.linalg.norm(theta, axis=0)
    valid = scales > 1e-12
    theta_valid = theta[:, valid]
    terms_valid = [term for term, keep in zip(terms, valid) if keep]
    scales_valid = scales[valid]
    theta_scaled = theta_valid / scales_valid
    theta_objective_valid, _ = candidate_utils.dictionary_matrix(u_objective, terms_valid)
    active = np.ones(theta_scaled.shape[1], dtype=bool)
    coef_scaled_full = np.zeros(theta_scaled.shape[1], dtype=float)
    ridge = max(alpha, 0.0)

    for _ in range(8):
        if not np.any(active):
            break
        design = theta_scaled[:, active]
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        rhs = design.T @ y
        try:
            coef_active = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef_active = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        coef_unscaled_active = coef_active / scales_valid[active]
        keep_active = np.abs(coef_unscaled_active) >= threshold
        coef_scaled_full[:] = 0.0
        coef_scaled_full[np.where(active)[0]] = coef_active
        if np.all(keep_active):
            break
        active_indices = np.where(active)[0]
        active[active_indices[~keep_active]] = False

    hard_cap_terms = max(1, min(max_reaction_terms, len(terms_valid)))
    coefficient_threshold = max(float(threshold), 0.0)

    def objective(
        max_error_value: float,
        support_terms: List[str],
    ) -> Tuple[float, float, float]:
        support_kappa = effective_kappa(u, support_terms) if support_terms else dictionary_kappa
        condition_ratio = max(
            0.0, math.log10(support_kappa) - math.log10(kappa_threshold)
        )
        return (
            max_error_value,
            support_kappa,
            condition_ratio,
        )

    def fit_support(indices: List[int]) -> dict:
        support_indices = sorted(set(indices))
        coef_unscaled = np.zeros(theta_scaled.shape[1], dtype=float)

        if support_indices:
            design = theta_valid[:, support_indices]
            rhs = design.T @ y
            try:
                if DEBIASED_REFIT_ALPHA > 0.0:
                    gram = design.T @ design + DEBIASED_REFIT_ALPHA * np.eye(
                        design.shape[1]
                    )
                    coef_selected = np.linalg.solve(gram, rhs)
                else:
                    coef_selected = np.linalg.lstsq(design, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                coef_selected = np.linalg.lstsq(design, y, rcond=None)[0]

            kept_pairs = [
                (idx, float(coef))
                for idx, coef in zip(support_indices, coef_selected)
                if abs(float(coef)) >= coefficient_threshold
            ]
            kept_indices = [idx for idx, _ in kept_pairs]
            if kept_indices != support_indices:
                return fit_support(kept_indices)
            for idx, coef in kept_pairs:
                coef_unscaled[idx] = coef

        coefficients = {
            terms_valid[idx]: round(float(coef_unscaled[idx]), 3)
            for idx in support_indices
        }
        # Score the same rounded equation exposed to the agent.
        equation = candidate_utils.format_equation(coefficients)
        coefficients = candidate_utils.candidate_coefficients(equation, terms_valid)
        coef_unscaled = np.array(
            [coefficients.get(term, 0.0) for term in terms_valid], dtype=float
        )
        pred_objective = theta_objective_valid @ coef_unscaled
        max_error_value = max_abs_error(y_objective, pred_objective)
        score, support_kappa, condition_ratio = objective(
            max_error_value,
            list(coefficients),
        )
        return {
            "indices": support_indices,
            "support": list(coefficients),
            "coefficients": coefficients,
            "coef_unscaled": coef_unscaled,
            "max_error": max_error_value,
            "condition_ratio": condition_ratio,
            "kappa": support_kappa,
            "penalty": score,
        }

    initial_indices = [
        int(idx)
        for idx, keep in enumerate(active)
        if keep and abs(float(coef_scaled_full[idx] / scales_valid[idx])) >= coefficient_threshold
    ]
    if len(initial_indices) > hard_cap_terms:
        seed_coefficients = coef_scaled_full / scales_valid
        initial_indices = sorted(
            initial_indices,
            key=lambda idx: abs(float(seed_coefficients[idx])),
            reverse=True,
        )[:hard_cap_terms]

    fit_cache = {}

    def cached_fit(indices: List[int]) -> dict:
        key = tuple(sorted(set(indices)))
        if key not in fit_cache:
            fit_cache[key] = fit_support(list(key))
        return fit_cache[key]

    candidate_supports = set()
    if allowed_supports is not None:
        term_to_index = {term: idx for idx, term in enumerate(terms_valid)}
        for support_terms in allowed_supports:
            support_indices = [
                term_to_index[term] for term in support_terms if term in term_to_index
            ]
            candidate_supports.add(tuple(sorted(set(support_indices))))
    else:
        if initial_indices:
            candidate_supports.add(tuple(sorted(initial_indices)))
        for target_size in range(1, hard_cap_terms + 1):
            for support_indices in combinations(range(len(terms_valid)), target_size):
                candidate_supports.add(tuple(support_indices))
    if not candidate_supports:
        candidate_supports.add(tuple())

    support_candidates = [
        cached_fit(list(candidate_support))
        for candidate_support in sorted(candidate_supports)
    ]

    def add_relative_margin(candidate: dict) -> dict:
        candidate = dict(candidate)
        support_key = tuple(candidate["support"])
        alternatives = [
            other
            for other in support_candidates
            if tuple(other["support"]) != support_key
        ]
        if alternatives:
            second_best = min(alternatives, key=lambda item: item["max_error"])
            second_best_max_error = float(second_best["max_error"])
            second_best_equation = candidate_utils.format_equation(
                second_best["coefficients"]
            )
            relative_margin = relative_error_margin(
                second_best_max_error,
                float(candidate["max_error"]),
            )
        else:
            second_best_max_error = float("nan")
            second_best_equation = "not evaluated: no alternative support supplied"
            relative_margin = 0.0
        candidate.update(
            {
                "relative_margin": float(relative_margin),
                "second_best_max_error": second_best_max_error,
                "second_best_equation": second_best_equation,
                "reward_penalty": float(candidate["max_error"]),
                "diagnostics": {
                    "condition_ratio": float(candidate["condition_ratio"]),
                    "relative_margin": float(relative_margin),
                    "second_best_max_error": second_best_max_error,
                    "second_best_equation": second_best_equation,
                },
                "penalty": float(candidate["max_error"]),
            }
        )
        return candidate

    scored_candidates = [add_relative_margin(candidate) for candidate in support_candidates]
    scored_candidates.sort(key=lambda item: item["penalty"])
    best = scored_candidates[0]
    equation = candidate_utils.format_equation(best["coefficients"])
    selection_scope = (
        "agent-specified refit set"
        if allowed_supports is not None
        else "agent-specified dictionary"
    )
    selection_note = (
        f" Scored {len(support_candidates)} support(s) from the {selection_scope}."
    )
    if len({tuple(candidate["support"]) for candidate in support_candidates}) < 2:
        selection_note += (
            " No different-support alternative was supplied, so relative_margin is "
            "reported as 0."
        )

    candidate_diagnostics = [
        {
            "equation": candidate_utils.format_equation(candidate["coefficients"]),
            "support": list(candidate["support"]),
            "training_max_error": float(candidate["max_error"]),
            "condition_penalty": float(candidate["condition_ratio"]),
            "relative_margin": float(candidate["relative_margin"]),
            "second_best_equation": candidate["second_best_equation"],
            "diagnostics": dict(candidate["diagnostics"]),
            "step_penalty": float(candidate["reward_penalty"]),
        }
        for candidate in sorted(scored_candidates, key=lambda item: item["penalty"])
    ]
    candidate_summary = (
        f"best equation={equation}; "
        f"sampled-data max error={best['max_error']:.3g}; "
        f"condition penalty={best['condition_ratio']:.3g}; "
        f"closest different-support candidate={best['second_best_equation']}; "
        f"relative margin={best['relative_margin']:.3g}"
    )
    feedback = (
        "Sparse regression with thresholded ridge screening, hard-capped support "
        "enumeration, and debiased coefficient refit. "
        f"Selected-support kappa = {best['kappa']:.3g}; "
        f"full-dictionary kappa = {dictionary_kappa:.3g}. "
        f"Best fit: f(u) = {equation}. "
        f"sampled-data max error = {best['max_error']:.3g}. "
        f"Coefficient threshold = {coefficient_threshold:.3g}; "
        f"Non-zero terms: {len(best['support'])}. "
        f"condition penalty term = {best['condition_ratio']:.3g}. "
        f"Closest scored different-support equation {best['second_best_equation']} "
        f"has max error = {best['second_best_max_error']:.3g}; "
        f"relative margin = {best['relative_margin']:.3g}. "
        f"{selection_note} "
        f"Candidate sampled-data diagnostics: {candidate_summary}. "
        "This tool updates only the latest regression candidate; the scientific "
        "conclusion inherited by future tasks is set only by update_scientific_context."
    )
    return {
        "support": best["support"],
        "coefficients": best["coefficients"],
        "equation": equation,
        "max_error": best["max_error"],
        "condition_ratio": best["condition_ratio"],
        "relative_margin": best["relative_margin"],
        "second_best_max_error": best["second_best_max_error"],
        "second_best_equation": best["second_best_equation"],
        "kappa": best["kappa"],
        "penalty": best["reward_penalty"],
        "reward_penalty": best["reward_penalty"],
        "diagnostics": dict(best["diagnostics"]),
        "candidate_diagnostics": candidate_diagnostics,
        "feedback": feedback,
    }


def candidate_objective_diagnostics(
    candidate: str,
    data: dict,
    objective_data: dict,
    dictionary: List[str],
    dictionary_terms: List[str],
    last_candidate_diagnostics: List[dict],
    kappa_threshold: float,
) -> dict:
    coefficients = candidate_utils.candidate_coefficients(candidate, dictionary_terms)
    support = [term for term in dictionary if abs(coefficients.get(term, 0.0)) > 0.0]
    pred_objective = candidate_utils.reaction_value(objective_data["u"], coefficients)
    max_error = max_abs_error(objective_data["y"], pred_objective)
    support_kappa = effective_kappa(data["u"], support) if support else KAPPA_CAP
    condition_penalty = max(
        0.0,
        math.log10(support_kappa) - math.log10(kappa_threshold),
    )
    support_key = tuple(support)
    alternatives = [
        item
        for item in last_candidate_diagnostics
        if tuple(item.get("support", [])) != support_key
    ]
    if alternatives:
        second_best = min(
            alternatives,
            key=lambda item: item["training_max_error"],
        )
        second_best_max_error = float(second_best["training_max_error"])
        second_best_equation = str(second_best["equation"])
        relative_margin = relative_error_margin(second_best_max_error, max_error)
    else:
        second_best_max_error = float("nan")
        second_best_equation = "not evaluated: no alternative support supplied"
        relative_margin = 0.0

    step_penalty = max_error
    diagnostics = {
        "condition_ratio": condition_penalty,
        "relative_margin": relative_margin,
        "second_best_max_error": second_best_max_error,
        "second_best_equation": second_best_equation,
    }
    return {
        "equation": candidate,
        "support": support,
        "coefficients": coefficients,
        "training_max_error": max_error,
        "condition_penalty": condition_penalty,
        "relative_margin": relative_margin,
        "second_best_max_error": second_best_max_error,
        "second_best_equation": second_best_equation,
        "step_penalty": step_penalty,
        "reward_penalty": step_penalty,
        "diagnostics": diagnostics,
    }
