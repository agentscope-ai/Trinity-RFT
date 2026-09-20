"""Grid-navigation environment used by the CoD research workflow."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Position = Tuple[int, int]


def _generate_landscape_costs(
    *,
    rng: np.random.Generator,
    size: int,
    num_components: int,
    min_scale: float,
    max_scale: float,
) -> np.ndarray:
    """Generate a spatially correlated integer cost landscape."""
    if num_components < 2:
        raise ValueError("landscape_num_components must be at least 2.")
    if not 0.0 < min_scale <= max_scale:
        raise ValueError("landscape scales must satisfy 0 < min_scale <= max_scale.")

    coordinates = np.linspace(0.0, 1.0, size)
    row_grid, col_grid = np.meshgrid(coordinates, coordinates, indexing="ij")
    landscape = np.zeros((size, size), dtype=np.float64)

    for _ in range(num_components):
        center_row, center_col = rng.uniform(0.0, 1.0, size=2)
        long_scale = float(rng.uniform(min_scale, max_scale))
        short_scale = float(rng.uniform(min_scale, long_scale))
        angle = float(rng.uniform(0.0, np.pi))
        amplitude = float(rng.uniform(0.8, 1.2))

        row_delta = row_grid - center_row
        col_delta = col_grid - center_col
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        long_delta = cos_angle * row_delta + sin_angle * col_delta
        short_delta = -sin_angle * row_delta + cos_angle * col_delta
        landscape += amplitude * np.exp(
            -0.5 * ((long_delta / long_scale) ** 2 + (short_delta / short_scale) ** 2)
        )

    # Positive Gaussian components form costly hills over a connected low-cost
    # background. Squaring the normalized height makes routes through valleys cheap
    # without flattening the spatial differences through a rank transformation.
    landscape_min = float(landscape.min())
    landscape_span = float(landscape.max() - landscape_min)
    if landscape_span == 0.0:
        return np.zeros_like(landscape, dtype=np.int64)
    normalized = (landscape - landscape_min) / landscape_span
    costs = np.rint(99.0 * np.square(normalized))
    return costs.astype(np.int64)


@dataclass
class GridNavigationPackState:
    """The cost map and observations shared by all tasks in one CoD pack."""

    pack_seed: int
    costs: np.ndarray
    revealed: np.ndarray

    @classmethod
    def generate(
        cls,
        pack_seed: int,
        grid_min_size: int,
        grid_max_size: int,
        landscape_num_components: int = 6,
        landscape_min_scale: float = 0.10,
        landscape_max_scale: float = 0.35,
    ) -> "GridNavigationPackState":
        """Generate a deterministic square cost map for one pack."""
        rng = np.random.default_rng(pack_seed)
        size = int(rng.integers(grid_min_size, grid_max_size + 1))
        costs = _generate_landscape_costs(
            rng=rng,
            size=size,
            num_components=landscape_num_components,
            min_scale=landscape_min_scale,
            max_scale=landscape_max_scale,
        )
        return cls(
            pack_seed=pack_seed,
            costs=costs,
            revealed=np.zeros_like(costs, dtype=bool),
        )

    @property
    def size(self) -> int:
        """Return the side length of the square grid."""
        return int(self.costs.shape[0])

    def reveal(self, path: list[Position], radius: int) -> None:
        """Reveal the Chebyshev neighborhood of every entered cell."""
        for row, col in path:
            row_start = max(0, row - radius)
            row_end = min(self.size, row + radius + 1)
            col_start = max(0, col - radius)
            col_end = min(self.size, col + radius + 1)
            self.revealed[row_start:row_end, col_start:col_end] = True


_PACK_STATES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def get_pack_state(
    *,
    pack_seed: int,
    task_idx: int,
    grid_min_size: int,
    grid_max_size: int,
    landscape_num_components: int = 6,
    landscape_min_scale: float = 0.10,
    landscape_max_scale: float = 0.35,
) -> GridNavigationPackState:
    """Create the state for task 0, and reuse it for later tasks in the pack."""
    key = pack_seed
    if task_idx == 0:
        state = GridNavigationPackState.generate(
            pack_seed=pack_seed,
            grid_min_size=grid_min_size,
            grid_max_size=grid_max_size,
            landscape_num_components=landscape_num_components,
            landscape_min_scale=landscape_min_scale,
            landscape_max_scale=landscape_max_scale,
        )
        _PACK_STATES[key] = state
        return state
    return _PACK_STATES[key]


@dataclass(frozen=True)
class GridNavigationTask:
    """Task-specific start, goal, and exact round count."""

    start: Position
    goal: Position
    num_rounds: int


def _sample_other_coordinate(rng: np.random.Generator, current: int, size: int) -> int:
    value = int(rng.integers(0, size - 1))
    return value + 1 if value >= current else value


def generate_task(
    *,
    task_seed: int,
    grid_size: int,
    min_rounds: int,
    max_rounds: int,
) -> GridNavigationTask:
    """Generate a task together with an implicit exact-length witness path."""
    rng = np.random.default_rng(task_seed)
    num_rounds = int(rng.integers(min_rounds, max_rounds + 1))

    while True:
        start = (
            int(rng.integers(0, grid_size)),
            int(rng.integers(0, grid_size)),
        )
        current = start
        for _ in range(num_rounds):
            row, col = current
            if int(rng.integers(0, 2)) == 0:
                current = (row, _sample_other_coordinate(rng, col, grid_size))
            else:
                current = (_sample_other_coordinate(rng, row, grid_size), col)
        if current != start:
            return GridNavigationTask(start=start, goal=current, num_rounds=num_rounds)


class GridNavigationEnv:
    """One fixed-round navigation task over a shared cost map."""

    def __init__(
        self,
        *,
        pack_state: GridNavigationPackState,
        task: GridNavigationTask,
        reveal_radius: int,
    ):
        """Initialize one task over an existing pack state."""
        self.pack_state = pack_state
        self.task = task
        self.reveal_radius = reveal_radius
        self.reset_task()

    def reset_task(self) -> None:
        """Reset task-local state while retaining pack observations."""
        self.current_position = self.task.start
        self.rounds_taken = 0
        self.entered_costs: list[int] = []
        self.last_path: list[Position] = []

    @property
    def normalized_loss(self) -> float:
        """Return the visit-weighted mean cell cost normalized to [0, 1]."""
        if not self.entered_costs:
            return 0.0
        return float(sum(self.entered_costs) / (100.0 * len(self.entered_costs)))

    def validate_destination(self, destination: Position) -> Optional[str]:
        """Return an error for an illegal rook move, otherwise None."""
        row, col = destination
        cur_row, cur_col = self.current_position
        if not (0 <= row < self.pack_state.size and 0 <= col < self.pack_state.size):
            return f"Destination ({row},{col}) is outside the grid."
        if destination == self.current_position:
            return "The destination must differ from the current position."
        if row != cur_row and col != cur_col:
            return "The destination must be in the same row or the same column."
        return None

    def path_to(self, destination: Position) -> list[Position]:
        """List entered cells, excluding the origin and including the destination."""
        cur_row, cur_col = self.current_position
        dest_row, dest_col = destination
        if cur_row == dest_row:
            step = 1 if dest_col > cur_col else -1
            return [(cur_row, col) for col in range(cur_col + step, dest_col + step, step)]
        step = 1 if dest_row > cur_row else -1
        return [(row, cur_col) for row in range(cur_row + step, dest_row + step, step)]

    def step(self, destination: Position) -> dict:
        """Execute one previously validated move."""
        path = self.path_to(destination)
        step_costs = [int(self.pack_state.costs[position]) for position in path]
        self.entered_costs.extend(step_costs)
        self.pack_state.reveal(path, self.reveal_radius)
        self.current_position = destination
        self.rounds_taken += 1
        self.last_path = path

        done = self.rounds_taken == self.task.num_rounds
        success = done and self.current_position == self.task.goal
        reward = 1.0 - self.normalized_loss if success else 0.0
        return {
            "path": path,
            "normalized_loss": self.normalized_loss,
            "done": done,
            "success": success,
            "reward": reward,
        }

    def render(self) -> str:
        """Render row and column labels with known costs and question marks."""
        header = "     " + "".join(f"{col:>3}" for col in range(self.pack_state.size))
        rows = [header]
        for row in range(self.pack_state.size):
            cells = []
            for col in range(self.pack_state.size):
                value = (
                    str(int(self.pack_state.costs[row, col]))
                    if self.pack_state.revealed[row, col]
                    else "?"
                )
                cells.append(f"{value:>3}")
            rows.append(f"{row:>3}  " + "".join(cells))
        return "\n".join(rows)
