"""TicTacToe game state implementation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import Cell, Player

if TYPE_CHECKING:
    from ..game_config import GameConfig
    from ..storage import GameMetadata


@dataclass
class TicTacToeState:
    """Pure Python TicTacToe state for evaluation."""

    board: list[int]  # 9 cells, 0=empty, 1=first, 2=second
    current_player: Player
    _done: bool = False
    _winner: int | None = None  # None=ongoing/draw, 1=first wins, 2=second wins

    @property
    def done(self) -> bool:
        return self._done

    @property
    def winner(self) -> int | None:
        return self._winner

    @classmethod
    def new(cls) -> "TicTacToeState":
        return cls(board=[0] * 9, current_player=Player.FIRST)

    def copy(self) -> "TicTacToeState":
        return TicTacToeState(
            board=self.board.copy(),
            current_player=self.current_player,
            _done=self._done,
            _winner=self._winner,
        )

    def legal_moves(self) -> list[int]:
        """Return indices of empty cells."""
        if self._done:
            return []
        return [i for i, cell in enumerate(self.board) if cell == Cell.EMPTY]

    def legal_moves_mask(self) -> list[float]:
        """Return mask where 1.0 = legal, 0.0 = illegal."""
        if self._done:
            return [0.0] * 9
        return [1.0 if cell == Cell.EMPTY else 0.0 for cell in self.board]

    def make_move(self, pos: int) -> None:
        """Make a move at the given position."""
        if self._done:
            raise ValueError("Game is already over")
        if self.board[pos] != Cell.EMPTY:
            raise ValueError(f"Position {pos} is not empty")

        self.board[pos] = self.current_player

        # Check for winner
        if self._check_winner(self.current_player):
            self._done = True
            self._winner = self.current_player
        elif all(cell != Cell.EMPTY for cell in self.board):
            # Draw
            self._done = True
            self._winner = None
        else:
            # Switch player
            if self.current_player == Player.FIRST:
                self.current_player = Player.SECOND
            else:
                self.current_player = Player.FIRST

    def _check_winner(self, player: Player) -> bool:
        """Check if the given player has won."""
        b = self.board
        p = player

        # Rows
        if b[0] == b[1] == b[2] == p:
            return True
        if b[3] == b[4] == b[5] == p:
            return True
        if b[6] == b[7] == b[8] == p:
            return True

        # Columns
        if b[0] == b[3] == b[6] == p:
            return True
        if b[1] == b[4] == b[7] == p:
            return True
        if b[2] == b[5] == b[8] == p:
            return True

        # Diagonals
        if b[0] == b[4] == b[8] == p:
            return True
        if b[2] == b[4] == b[6] == p:
            return True

        return False

    def to_observation(self, config: "GameConfig | GameMetadata") -> np.ndarray:
        """Convert to neural network observation format."""
        obs = np.zeros(config.obs_size, dtype=np.float32)

        # Board view: positions 0-8 for first player, 9-17 for second
        board_size = config.board_width * config.board_height
        for i, cell in enumerate(self.board):
            if cell == Cell.FIRST:
                obs[i] = 1.0
            elif cell == Cell.SECOND:
                obs[i + board_size] = 1.0

        # Legal moves mask starting at legal_mask_offset
        for i, cell in enumerate(self.board):
            if cell == Cell.EMPTY and not self._done:
                obs[config.legal_mask_offset + i] = 1.0

        # Current player: 2 floats at end
        player_offset = config.legal_mask_offset + config.num_actions
        if self.current_player == Player.FIRST:
            obs[player_offset] = 1.0
        else:
            obs[player_offset + 1] = 1.0

        return obs

    def display(self) -> str:
        """Return a string representation of the board."""
        symbols = {Cell.EMPTY: ".", Cell.FIRST: "X", Cell.SECOND: "O"}
        rows = []
        for row in range(3):
            cells = [symbols[Cell(self.board[row * 3 + col])] for col in range(3)]
            rows.append(" ".join(cells))
        return "\n".join(rows)
