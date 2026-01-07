"""Connect4 game state implementation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import Cell, Player

if TYPE_CHECKING:
    from ..game_config import GameConfig
    from ..storage import GameMetadata


@dataclass
class Connect4State:
    """Pure Python Connect4 state for evaluation."""

    board: list[int]  # 42 cells (7x6), column-major: board[col * 6 + row]
    current_player: Player
    _done: bool = False
    _winner: int | None = None

    WIDTH = 7
    HEIGHT = 6

    @property
    def done(self) -> bool:
        return self._done

    @property
    def winner(self) -> int | None:
        return self._winner

    @classmethod
    def new(cls) -> "Connect4State":
        return cls(board=[0] * 42, current_player=Player.FIRST)

    def copy(self) -> "Connect4State":
        return Connect4State(
            board=self.board.copy(),
            current_player=self.current_player,
            _done=self._done,
            _winner=self._winner,
        )

    def _get_cell(self, col: int, row: int) -> int:
        """Get cell value (row 0 is bottom)."""
        return self.board[col * self.HEIGHT + row]

    def _set_cell(self, col: int, row: int, value: int) -> None:
        """Set cell value."""
        self.board[col * self.HEIGHT + row] = value

    def _column_height(self, col: int) -> int:
        """Return the number of pieces in a column."""
        for row in range(self.HEIGHT):
            if self._get_cell(col, row) == Cell.EMPTY:
                return row
        return self.HEIGHT

    def legal_moves(self) -> list[int]:
        """Return indices of columns that aren't full."""
        if self._done:
            return []
        return [
            col for col in range(self.WIDTH) if self._column_height(col) < self.HEIGHT
        ]

    def legal_moves_mask(self) -> list[float]:
        """Return mask where 1.0 = legal, 0.0 = illegal."""
        if self._done:
            return [0.0] * self.WIDTH
        return [
            1.0 if self._column_height(col) < self.HEIGHT else 0.0
            for col in range(self.WIDTH)
        ]

    def make_move(self, col: int) -> None:
        """Drop a piece in the given column."""
        if self._done:
            raise ValueError("Game is already over")
        row = self._column_height(col)
        if row >= self.HEIGHT:
            raise ValueError(f"Column {col} is full")

        self._set_cell(col, row, self.current_player)

        # Check for winner
        if self._check_winner(col, row, self.current_player):
            self._done = True
            self._winner = self.current_player
        elif all(self._column_height(c) == self.HEIGHT for c in range(self.WIDTH)):
            # Draw (board full)
            self._done = True
            self._winner = None
        else:
            # Switch player
            if self.current_player == Player.FIRST:
                self.current_player = Player.SECOND
            else:
                self.current_player = Player.FIRST

    def _check_winner(self, col: int, row: int, player: Player) -> bool:
        """Check if the last move at (col, row) creates a win."""
        directions = [
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        ]  # horizontal, vertical, diagonals
        for dc, dr in directions:
            count = 1
            # Count in positive direction
            c, r = col + dc, row + dr
            while (
                0 <= c < self.WIDTH
                and 0 <= r < self.HEIGHT
                and self._get_cell(c, r) == player
            ):
                count += 1
                c += dc
                r += dr
            # Count in negative direction
            c, r = col - dc, row - dr
            while (
                0 <= c < self.WIDTH
                and 0 <= r < self.HEIGHT
                and self._get_cell(c, r) == player
            ):
                count += 1
                c -= dc
                r -= dr
            if count >= 4:
                return True
        return False

    def to_observation(self, config: "GameConfig | GameMetadata") -> np.ndarray:
        """Convert to neural network observation format."""
        obs = np.zeros(config.obs_size, dtype=np.float32)

        board_size = config.board_width * config.board_height
        for i, cell in enumerate(self.board):
            if cell == Cell.FIRST:
                obs[i] = 1.0
            elif cell == Cell.SECOND:
                obs[i + board_size] = 1.0

        # Legal moves mask
        for col in range(self.WIDTH):
            if self._column_height(col) < self.HEIGHT and not self._done:
                obs[config.legal_mask_offset + col] = 1.0

        # Current player
        player_offset = config.legal_mask_offset + config.num_actions
        if self.current_player == Player.FIRST:
            obs[player_offset] = 1.0
        else:
            obs[player_offset + 1] = 1.0

        return obs

    def display(self) -> str:
        """Return a string representation of the board."""
        symbols = {Cell.EMPTY: ".", Cell.FIRST: "R", Cell.SECOND: "Y"}
        rows = []
        for row in range(self.HEIGHT - 1, -1, -1):
            cells = [
                symbols[Cell(self._get_cell(col, row))] for col in range(self.WIDTH)
            ]
            rows.append(" ".join(cells))
        rows.append("-" * (self.WIDTH * 2 - 1))
        rows.append(" ".join(str(i) for i in range(self.WIDTH)))
        return "\n".join(rows)
