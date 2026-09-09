from typing import Literal

from pydantic import BaseModel, Field


SYSTEM_PROMPT = """You are an AI agent playing Gomoku.

Gomoku Rules:
- The board is represented using:
  . = empty
  X = your stones
  O = opponent stones
- You are X.
- Players take turns placing one stone on an empty cell.
- Get 4 stones in a row horizontally, vertically, or diagonally to win.
- You must choose an empty cell.

The board coordinates are zero-indexed:
(row, column)

You must output your next move inside triple backticks.

Example:
```2,3```

Only output one valid move in the format:
row,column
"""


class GomokuAction(BaseModel):
    row: int = Field(description="Zero-based row of the move.")
    col: int = Field(description="Zero-based column of the move.")


VALID_PLAYERS = Literal["X", "O"]


def check_winner(
    board: list[list[str]],
    player: str,
    win_length: int = 4,
) -> bool:
    """Return True if player has win_length stones in a row."""

    size = len(board)

    directions = [
        (0, 1),   # horizontal
        (1, 0),   # vertical
        (1, 1),   # diagonal
        (1, -1),  # reverse diagonal
    ]

    for row in range(size):
        for col in range(size):
            if board[row][col] != player:
                continue

            for row_step, col_step in directions:
                count = 0
                current_row = row
                current_col = col

                while (
                    0 <= current_row < size
                    and 0 <= current_col < size
                    and board[current_row][current_col] == player
                ):
                    count += 1

                    if count >= win_length:
                        return True

                    current_row += row_step
                    current_col += col_step

    return False


def board_to_string(board: list[list[str]]) -> str:
    """Convert the board to an LLM-friendly text representation."""

    header = "  " + " ".join(str(i) for i in range(len(board)))

    rows = [
        f"{i} " + " ".join(row)
        for i, row in enumerate(board)
    ]

    return header + "\n" + "\n".join(rows)