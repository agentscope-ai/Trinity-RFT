# -*- coding: utf-8 -*-
"""Dictionary and candidate-equation utilities for PDE discovery."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_DICTIONARY = [
    "u",
    "u**2",
    "u**3",
    "u**4",
    "u**5",
    "u**7",
    "sin(u)",
    "sin(2*u)",
    "exp(u)-1",
    "log(1+u**2)",
    "tanh(u)",
    "u/(1+u**2)",
    "u**2/(1+u**2)",
    "u**4/(1+u**4)",
    "u/(1+u+u**2)",
]

SYMBOLIC_COEFFICIENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:c\d+|coef(?:ficient)?\d*|a|b|alpha|beta|gamma)"
    r"(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)


def term_pattern(term: str) -> str:
    if term == "u":
        return r"u(?!\*\*)"
    escaped = re.escape(term)
    if "+" in term or "-" in term:
        # format_equation wraps compound terms to preserve expression grouping.
        return rf"(?:{escaped}|\({escaped}\))"
    return escaped


def symbolic_coefficient_tokens(equation: object) -> List[str]:
    return sorted(set(SYMBOLIC_COEFFICIENT_RE.findall(str(equation or ""))))


def basis_values(term: str, u: np.ndarray) -> np.ndarray:
    if term == "u":
        return u
    if term == "u**2":
        return u**2
    if term == "u**3":
        return u**3
    if term == "u**4":
        return u**4
    if term == "u**5":
        return u**5
    if term == "u**7":
        return u**7
    if term == "sin(u)":
        return np.sin(u)
    if term == "sin(2*u)":
        return np.sin(2.0 * u)
    if term == "exp(u)-1":
        return np.expm1(u)
    if term == "log(1+u**2)":
        return np.log1p(u**2)
    if term == "tanh(u)":
        return np.tanh(u)
    if term == "u/(1+u**2)":
        return u / (1.0 + u**2)
    if term == "u**2/(1+u**2)":
        return (u**2) / (1.0 + u**2)
    if term == "u**4/(1+u**4)":
        return (u**4) / (1.0 + u**4)
    if term == "u/(1+u+u**2)":
        return u / (1.0 + u + u**2)
    raise ValueError(f"Unsupported dictionary term: {term}")


def dictionary_matrix(
    u: np.ndarray,
    dictionary: List[str],
) -> Tuple[np.ndarray, List[str]]:
    columns = []
    terms = []
    for term in dictionary:
        try:
            basis = basis_values(str(term), u)
        except ValueError:
            continue
        if np.all(np.isfinite(basis)):
            columns.append(basis.astype(float))
            terms.append(str(term))
    if not columns:
        return np.zeros((len(u), 0), dtype=float), []
    return np.column_stack(columns), terms


def sanitize_dictionary(dictionary: object, fallback_terms: List[str]) -> List[str]:
    if not isinstance(dictionary, list) or not dictionary:
        return list(fallback_terms)
    sanitized = []
    for term in dictionary:
        term = str(term)
        if term in DEFAULT_DICTIONARY and term not in sanitized:
            sanitized.append(term)
    return sanitized or list(fallback_terms)


def reaction_value(u: np.ndarray, coefficients: Dict[str, float]) -> np.ndarray:
    values = np.zeros_like(u, dtype=float)
    for term, coef in coefficients.items():
        try:
            values = values + coef * basis_values(term, u)
        except ValueError:
            continue
    return values


def format_equation(coefficients: Dict[str, float]) -> str:
    if not coefficients:
        return "0"

    def format_term(term: str) -> str:
        if "+" in term or "-" in term:
            return f"({term})"
        return term

    pieces = [
        f"{coef:.3g}*{format_term(term)}"
        for term, coef in coefficients.items()
    ]
    return pieces[0] + "".join(
        f" {'+' if piece[0] != '-' else '-'} {piece.lstrip('-')}"
        for piece in pieces[1:]
    )


def candidate_coefficients(
    candidate: str,
    dictionary_terms: List[str],
) -> Dict[str, float]:
    text = str(candidate or "").strip()
    if "=" in text:
        left, right = text.split("=", 1)
        # Accept only the equation form used by the environment.
        if not re.fullmatch(r"\s*f\s*\(\s*u\s*\)\s*", left, re.IGNORECASE):
            return {}
        text = right

    coefficients: Dict[str, float] = {}
    remaining = re.sub(r"\s+", "", text)
    for term in sorted(dictionary_terms, key=len, reverse=True):
        pattern = (
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
            r"\*"
            + term_pattern(term)
        )
        while match := re.search(pattern, remaining, flags=re.IGNORECASE):
            # Sum repeated terms rather than silently keeping only one.
            coefficients[term] = coefficients.get(term, 0.0) + float(match.group(1))
            start, end = match.span()
            remaining = remaining[:start] + (" " * (end - start)) + remaining[end:]

    # Reject unsupported terms instead of silently dropping them.
    if remaining.replace(" ", "") not in {"", "+", "-"}:
        return {}
    return {
        term: coefficient
        for term, coefficient in coefficients.items()
        if abs(coefficient) > 1e-12
    }


def candidate_support(
    candidate: str,
    dictionary: List[str],
    dictionary_terms: List[str],
) -> List[str]:
    coefficient_support = candidate_coefficients(candidate, dictionary_terms)
    if coefficient_support:
        return [term for term in dictionary if term in coefficient_support]
    compact = candidate.replace(" ", "")
    support = []
    for term in sorted(dictionary, key=len, reverse=True):
        pattern = term_pattern(term)
        if re.search(pattern, compact, flags=re.IGNORECASE):
            support.append(term)
            compact = re.sub(pattern, "", compact, flags=re.IGNORECASE)
    support_set = set(support)
    return [term for term in dictionary if term in support_set]
