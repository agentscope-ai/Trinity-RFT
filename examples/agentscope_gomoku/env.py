from typing import Optional

from .utils import board_to_string, check_winner


class GomokuEnv:
    """Simple Gomoku environment for an agentic workflow."""

    EMPTY = "."
    AGENT = "X"
    OPPONENT = "O"

    def __init__(
        self,
        size: int = 5,
        win_length: int = 4,
    ):
        if size < win_length:
            raise ValueError("Board size must be >= win length.")

        self.size = size
        self.win_length = win_length
        self.board: list[list[str]] = []
        self.done = False
        self.winner: Optional[str] = None

        self.reset()

    def reset(self) -> str:
        """Reset the board and return the initial observation."""

        self.board = [
            [self.EMPTY for _ in range(self.size)]
            for _ in range(self.size)
        ]

        self.done = False
        self.winner = None

        return self.render()

    def render(self) -> str:
        """Return the current board as an LLM-friendly string."""

        return board_to_string(self.board)

    def _is_valid_move(self, row: int, col: int) -> bool:
        """Check whether a move is inside the board and on an empty cell."""

        return (
            0 <= row < self.size
            and 0 <= col < self.size
            and self.board[row][col] == self.EMPTY
        )

    def _is_draw(self) -> bool:
        """Return True when there are no empty cells left."""

        return all(
            cell != self.EMPTY
            for row in self.board
            for cell in row
        )

    def _opponent_move(self) -> Optional[tuple[int, int]]:
        """Make a simple opponent move.

        The opponent chooses the first available cell.
        This keeps the environment deterministic and lightweight.
        """

        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == self.EMPTY:
                    self.board[row][col] = self.OPPONENT
                    return row, col

        return None

    def step(
        self,
        action: tuple[int, int],
    ) -> tuple[str, float, bool, dict]:
        """Apply the agent's move and return observation, reward, done, info."""

        if self.done:
            return (
                self.render(),
                0.0,
                True,
                {"reason": "game_already_finished"},
            )

        row, col = action

        if not self._is_valid_move(row, col):
            return (
                self.render(),
                -1.0,
                False,
                {"reason": "invalid_move"},
            )

        # Agent places X.
        self.board[row][col] = self.AGENT

        # Check whether the agent won.
        if check_winner(
            self.board,
            self.AGENT,
            self.win_length,
        ):
            self.done = True
            self.winner = self.AGENT

            return (
                self.render(),
                1.0,
                True,
                {"reason": "agent_won"},
            )

        # Check whether the board is full.
        if self._is_draw():
            self.done = True

            return (
                self.render(),
                0.0,
                True,
                {"reason": "draw"},
            )

        # Opponent makes its move.
        opponent_move = self._opponent_move()

        if opponent_move is not None:
            opponent_row, opponent_col = opponent_move

            if check_winner(
                self.board,
                self.OPPONENT,
                self.win_length,
            ):
                self.done = True
                self.winner = self.OPPONENT

                return (
                    self.render(),
                    -1.0,
                    True,
                    {
                        "reason": "opponent_won",
                        "opponent_move": opponent_move,
                    },
                )

        if self._is_draw():
            self.done = True

            return (
                self.render(),
                0.0,
                True,
                {"reason": "draw"},
            )

        return (
            self.render(),
            0.0,
            False,
            {
                "reason": "continue",
                "opponent_move": opponent_move,
            },
        )

    def finished(self) -> bool:
        """Return whether the game has ended."""

        return self.done

    def success(self) -> bool:
        """Return whether the agent won the game."""

        return self.winner == self.AGENT