# -*- coding: utf-8 -*-
"""Numerical helpers for the PDE discovery workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np


GROUND_TRUTH_STREAM = 0
INITIAL_AMPLITUDE_STREAM = 1
INITIAL_SHAPE_STREAM = 2


@dataclass(frozen=True)
class InitialConditionShapeConfig:
    """Configurable complexity prior for random smooth initial conditions."""

    mode_count_range: tuple[int, int]

    def __post_init__(self) -> None:
        value = self.mode_count_range
        if len(value) != 2 or value[0] < 1 or value[1] < value[0]:
            raise ValueError("mode_count_range must be a positive [min, max] pair")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object] | None,
    ) -> "InitialConditionShapeConfig":
        if raw is None:
            raise ValueError("initial_condition_shape configuration is required")
        allowed = {"mode_count_range"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                "Unsupported initial_condition_shape keys: "
                + ", ".join(sorted(unknown))
            )
        missing = allowed - set(raw)
        if missing:
            raise ValueError(
                "Missing initial_condition_shape keys: "
                + ", ".join(sorted(missing))
            )

        def count_range(name: str) -> tuple[int, int]:
            value = raw[name]
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != 2
            ):
                raise ValueError(f"{name} must be a two-element list")
            return int(value[0]), int(value[1])

        return cls(mode_count_range=count_range("mode_count_range"))


def initial_condition_amplitudes(
    pack_seed: int,
    trajectory_count: int,
    amplitude_upper: float,
) -> list[float]:
    """Sample one amplitude from each stratum below a calibrated upper bound."""
    count = max(1, int(trajectory_count))
    upper = float(amplitude_upper)
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("initial amplitude upper bound must be finite and positive")
    band_edges = np.linspace(0.0, upper, count + 1)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(pack_seed), INITIAL_AMPLITUDE_STREAM])
    )
    amplitudes = [
        float(rng.uniform(band_edges[index], band_edges[index + 1]))
        for index in range(count)
    ]
    rng.shuffle(amplitudes)
    return amplitudes


def _normalized_modes(
    x_grid: np.ndarray,
    rng: np.random.Generator,
    mode_count: int,
) -> np.ndarray:
    domain_length = float(x_grid[-1] - x_grid[0])
    if domain_length <= 0.0:
        raise ValueError("initial-condition grid must span a positive domain")
    coordinate = (x_grid - x_grid[0]) / domain_length
    modes = np.zeros_like(x_grid, dtype=float)
    coefficients = rng.normal(size=mode_count)
    for mode, coefficient in enumerate(coefficients, start=1):
        modes += coefficient * np.sin(mode * np.pi * coordinate)
    scale = float(np.max(np.abs(modes)))
    return modes / scale if scale > 1e-12 else modes


def _random_initial_condition(
    x_grid: np.ndarray,
    rng: np.random.Generator,
    amplitude: float,
    config: InitialConditionShapeConfig,
) -> np.ndarray:
    domain_length = float(x_grid[-1] - x_grid[0])
    if domain_length <= 0.0:
        raise ValueError("initial-condition grid must span a positive domain")
    envelope = np.sin(np.pi * (x_grid - x_grid[0]) / domain_length)
    mode_count = int(
        rng.integers(
            config.mode_count_range[0],
            config.mode_count_range[1] + 1,
        )
    )
    modes = _normalized_modes(x_grid, rng, mode_count)
    shape = envelope * (modes - float(np.min(modes)))
    shape_scale = max(float(np.max(shape)), 1e-12)
    u0 = amplitude * shape / shape_scale
    u0[0] = 0.0
    u0[-1] = 0.0
    return u0


def initial_condition(
    x_grid: np.ndarray,
    pack_seed: int,
    trajectory_index: int = 0,
    trajectory_count: int = 1,
    amplitude_upper: float | None = None,
    shape_config: InitialConditionShapeConfig | None = None,
) -> np.ndarray:
    seed_sequence = np.random.SeedSequence(
        [int(pack_seed), INITIAL_SHAPE_STREAM, int(trajectory_index)]
    )
    rng = np.random.default_rng(seed_sequence)
    if amplitude_upper is None:
        raise ValueError("initial conditions require a calibrated amplitude upper bound")
    amplitudes = initial_condition_amplitudes(
        pack_seed,
        trajectory_count,
        amplitude_upper=amplitude_upper,
    )
    if not 0 <= trajectory_index < len(amplitudes):
        raise ValueError(
            f"trajectory_index={trajectory_index} is outside "
            f"trajectory_count={trajectory_count}"
        )
    if shape_config is None:
        raise ValueError("initial_condition_shape configuration is required")
    return _random_initial_condition(
        x_grid,
        rng,
        amplitudes[trajectory_index],
        shape_config,
    )


def solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    n = len(diag)
    c_prime = np.zeros(max(0, n - 1), dtype=float)
    d_prime = np.zeros(n, dtype=float)
    denom = diag[0]
    if n > 1:
        c_prime[0] = upper[0] / denom
    d_prime[0] = rhs[0] / denom
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom
    solution = np.zeros(n, dtype=float)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


def second_derivative_x(values: np.ndarray, dx: float) -> np.ndarray:
    deriv2 = np.zeros_like(values)
    deriv2[:, 1:-1] = (
        values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
    ) / (dx * dx)
    deriv2[:, 0] = deriv2[:, 1]
    deriv2[:, -1] = deriv2[:, -2]
    return deriv2


def scheme_consistent_reaction_residual(
    u_grid: np.ndarray,
    dx: float,
    dt: float,
) -> np.ndarray:
    y_grid = np.zeros_like(u_grid, dtype=float)
    if len(u_grid) < 2:
        return y_grid
    u_xx_next = second_derivative_x(u_grid[1:], dx)
    y_grid[:-1, 1:-1] = (
        (u_grid[1:, 1:-1] - u_grid[:-1, 1:-1]) / dt
        - u_xx_next[:, 1:-1]
    )
    return y_grid


def simulate_dense_trajectory(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    dx: float,
    dt: float,
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    trajectory_index: int,
    reaction_fn: Callable[[np.ndarray], np.ndarray],
    pack_seed: int,
    trajectory_count: int = 1,
    initial_amplitude_upper: float | None = None,
    initial_condition_shape: InitialConditionShapeConfig | None = None,
    state_abs_limit: float | None = None,
) -> dict:
    if state_abs_limit is None or not np.isfinite(state_abs_limit) or state_abs_limit <= 0.0:
        raise ValueError("state_abs_limit must be finite and positive")
    nx = len(x_grid)
    nt = len(t_grid)
    u_grid = np.zeros((nt, nx), dtype=float)
    u_grid[0] = initial_condition(
        x_grid,
        pack_seed,
        trajectory_index,
        trajectory_count=trajectory_count,
        amplitude_upper=initial_amplitude_upper,
        shape_config=initial_condition_shape,
    )
    u_grid[:, 0] = 0.0
    u_grid[:, -1] = 0.0

    for step in range(nt - 1):
        rhs = u_grid[step, 1:-1] + dt * reaction_fn(u_grid[step, 1:-1])
        rhs = np.clip(rhs, -state_abs_limit, state_abs_limit)
        u_next = solve_tridiagonal(lower, diag, upper, rhs)
        u_grid[step + 1, 1:-1] = np.clip(
            u_next,
            -state_abs_limit,
            state_abs_limit,
        )

    reaction_residual = scheme_consistent_reaction_residual(u_grid, dx, dt)
    # The final state has no forward step, so evaluate f(u) directly.
    reaction_residual[-1, 1:-1] = reaction_fn(u_grid[-1, 1:-1])
    return {
        "x_grid": x_grid,
        "t_grid": t_grid,
        "u": u_grid,
        "y": reaction_residual,
    }


def interpolate_dense_field(
    field: dict,
    field_name: str,
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    x_grid = field["x_grid"]
    t_grid = field["t_grid"]
    values = field[field_name]

    xi = np.searchsorted(x_grid, x, side="right") - 1
    ti = np.searchsorted(t_grid, t, side="right") - 1
    xi = np.clip(xi, 0, len(x_grid) - 2)
    ti = np.clip(ti, 0, len(t_grid) - 2)
    x0 = x_grid[xi]
    x1 = x_grid[xi + 1]
    t0 = t_grid[ti]
    t1 = t_grid[ti + 1]
    wx = (x - x0) / np.maximum(x1 - x0, 1e-12)
    wt = (t - t0) / np.maximum(t1 - t0, 1e-12)

    v00 = values[ti, xi]
    v01 = values[ti, xi + 1]
    v10 = values[ti + 1, xi]
    v11 = values[ti + 1, xi + 1]
    return (
        (1.0 - wt) * ((1.0 - wx) * v00 + wx * v01)
        + wt * ((1.0 - wx) * v10 + wx * v11)
    )
