"""Evaluator for measuring trained model performance against baselines.

This module provides evaluation capabilities to measure how well a trained
model plays games compared to random play. It loads game configuration from
the replay database (like the Rust actor does), making it self-describing
and supporting multiple games.

Usage (defaults assume running from trainer/ directory):
    python -m trainer.evaluator --model ../data/models/latest.onnx --games 100
    python -m trainer.evaluator --db ../data/replay.db --env-id connect4 --games 100
"""

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

from .game_config import GameConfig, get_config
from .replay import GameMetadata, ReplayBuffer

logger = logging.getLogger(__name__)


class Player(IntEnum):
    """Player identifiers matching game implementations."""

    FIRST = 1  # First player (X in TicTacToe, Red in Connect4)
    SECOND = 2  # Second player (O in TicTacToe, Yellow in Connect4)


class Cell(IntEnum):
    """Cell states."""

    EMPTY = 0
    FIRST = 1  # First player's piece
    SECOND = 2  # Second player's piece


def get_game_metadata_or_config(
    db_path: str | None, env_id: str
) -> GameConfig | GameMetadata:
    """Load game configuration from database or fall back to hardcoded config.

    This follows the same pattern as trainer.py, making the evaluator self-describing
    by reading metadata from the replay database when available.

    Args:
        db_path: Path to replay database. If None or doesn't exist, uses fallback.
        env_id: Environment ID (e.g., "tictactoe", "connect4").

    Returns:
        GameConfig or GameMetadata with game configuration.
    """
    if db_path and Path(db_path).exists():
        try:
            with ReplayBuffer(db_path, validate_schema=False) as replay:
                metadata = replay.get_metadata(env_id)
                if metadata:
                    logger.info(f"Loaded game metadata from database for {env_id}")
                    return metadata
                logger.warning(f"No metadata in database for {env_id}, using fallback")
        except Exception as e:
            logger.warning(f"Failed to read database metadata: {e}, using fallback")

    return get_config(env_id)


class GameState(Protocol):
    """Protocol for game state implementations."""

    @property
    def done(self) -> bool:
        """Return True if game is over."""
        ...

    @property
    def winner(self) -> int | None:
        """Return winner (1=first player, 2=second player, None=draw/ongoing)."""
        ...

    @property
    def current_player(self) -> Player:
        """Return current player to move."""
        ...

    def legal_moves(self) -> list[int]:
        """Return list of legal move indices."""
        ...

    def legal_moves_mask(self) -> list[float]:
        """Return mask where 1.0 = legal, 0.0 = illegal."""
        ...

    def make_move(self, pos: int) -> None:
        """Make a move at the given position."""
        ...

    def to_observation(self, config: GameConfig | GameMetadata) -> np.ndarray:
        """Convert to neural network observation format."""
        ...

    def display(self) -> str:
        """Return a string representation for debugging."""
        ...

    def copy(self) -> "GameState":
        """Return a copy of the state."""
        ...


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

    def to_observation(self, config: GameConfig | GameMetadata) -> np.ndarray:
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

    def to_observation(self, config: GameConfig | GameMetadata) -> np.ndarray:
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


def create_game_state(env_id: str) -> GameState:
    """Create a new game state for the given environment.

    Args:
        env_id: Environment ID (e.g., "tictactoe", "connect4").

    Returns:
        New game state instance.

    Raises:
        ValueError: If env_id is not supported.
    """
    if env_id == "tictactoe":
        return TicTacToeState.new()
    elif env_id == "connect4":
        return Connect4State.new()
    else:
        raise ValueError(f"Unsupported game for evaluation: {env_id}")


class Policy(Protocol):
    """Protocol for action selection policies."""

    def select_action(self, state: GameState, config: GameConfig | GameMetadata) -> int:
        """Select an action given the current state and game config."""
        ...

    @property
    def name(self) -> str:
        """Return the policy name for logging."""
        ...


class RandomPolicy:
    """Uniformly random policy over legal moves."""

    @property
    def name(self) -> str:
        return "Random"

    def select_action(self, state: GameState, config: GameConfig | GameMetadata) -> int:
        legal = state.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")
        return random.choice(legal)


class OnnxPolicy:
    """Policy using an ONNX neural network model.

    Uses game configuration (from database or fallback) to correctly parse
    observations and policy outputs for different games.
    """

    def __init__(self, model_path: str, temperature: float = 0.0):
        """
        Args:
            model_path: Path to ONNX model file.
            temperature: Sampling temperature. 0 = greedy (argmax).
        """
        self.model_path = model_path
        self.temperature = temperature

        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.policy_output = self.session.get_outputs()[0].name
        self.value_output = self.session.get_outputs()[1].name

    @property
    def name(self) -> str:
        return f"ONNX({Path(self.model_path).name})"

    def select_action(self, state: GameState, config: GameConfig | GameMetadata) -> int:
        """Select an action using the neural network policy.

        Args:
            state: Current game state.
            config: Game configuration (from database metadata or fallback).

        Returns:
            Selected action index.
        """
        legal = state.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        # Get observation using config for correct encoding
        obs = state.to_observation(config)
        obs_batch = obs.reshape(1, -1)

        # Run inference
        outputs = self.session.run(
            [self.policy_output, self.value_output],
            {self.input_name: obs_batch},
        )
        policy_logits = outputs[0][0]  # Shape: (num_actions,)

        # Mask illegal moves
        legal_mask = state.legal_moves_mask()
        masked_logits = np.where(
            np.array(legal_mask) == 1.0,
            policy_logits,
            -np.inf,
        )

        if self.temperature == 0.0:
            # Greedy selection
            return int(np.argmax(masked_logits))
        else:
            # Sample with temperature
            logits = masked_logits / self.temperature
            logits = logits - np.max(logits)  # Numerical stability
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            return int(np.random.choice(config.num_actions, p=probs))

    def get_value(self, state: GameState, config: GameConfig | GameMetadata) -> float:
        """Get the value estimate for a state.

        Args:
            state: Current game state.
            config: Game configuration for observation encoding.

        Returns:
            Value estimate from the network.
        """
        obs = state.to_observation(config)
        obs_batch = obs.reshape(1, -1)

        outputs = self.session.run(
            [self.value_output],
            {self.input_name: obs_batch},
        )
        # Handle both (1, 1) and (1,) output shapes from different ONNX exports
        value = outputs[0]
        if value.ndim == 2:
            return float(value[0, 0])
        else:
            return float(value[0])


@dataclass
class MatchResult:
    """Result of a single game."""

    winner: int | None  # 1=player1, 2=player2, None=draw
    moves: int
    player1_as: Player  # Which color player1 played


@dataclass
class EvalResults:
    """Aggregated evaluation results."""

    env_id: str
    player1_name: str
    player2_name: str
    games_played: int
    player1_wins: int
    player2_wins: int
    draws: int
    player1_wins_as_first: int
    player1_wins_as_second: int
    player2_wins_as_first: int
    player2_wins_as_second: int
    avg_game_length: float

    @property
    def player1_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.player1_wins / self.games_played

    @property
    def player2_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.player2_wins / self.games_played

    @property
    def draw_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.draws / self.games_played

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"{'=' * 50}",
            f"Evaluation Results: {self.player1_name} vs {self.player2_name}",
            f"Game: {self.env_id}",
            f"{'=' * 50}",
            f"Games played: {self.games_played}",
            "",
            f"{self.player1_name}:",
            f"  Wins: {self.player1_wins} ({self.player1_win_rate:.1%})",
            f"    As first player: {self.player1_wins_as_first}",
            f"    As second player: {self.player1_wins_as_second}",
            "",
            f"{self.player2_name}:",
            f"  Wins: {self.player2_wins} ({self.player2_win_rate:.1%})",
            f"    As first player: {self.player2_wins_as_first}",
            f"    As second player: {self.player2_wins_as_second}",
            "",
            f"Draws: {self.draws} ({self.draw_rate:.1%})",
            f"Average game length: {self.avg_game_length:.1f} moves",
            f"{'=' * 50}",
        ]
        return "\n".join(lines)


def play_game(
    player1: Policy,
    player2: Policy,
    player1_as: Player,
    env_id: str,
    config: GameConfig | GameMetadata,
    verbose: bool = False,
) -> MatchResult:
    """Play a single game between two policies.

    Args:
        player1: First policy to evaluate.
        player2: Second policy (opponent).
        player1_as: Which color player1 plays (FIRST or SECOND).
        env_id: Environment ID for creating game state.
        config: Game configuration (from database or fallback).
        verbose: Print game moves.

    Returns:
        MatchResult with winner and game length.
    """
    state = create_game_state(env_id)
    moves = 0

    # Assign policies to player slots
    if player1_as == Player.FIRST:
        first_policy, second_policy = player1, player2
    else:
        first_policy, second_policy = player2, player1

    if verbose:
        role = "first" if player1_as == Player.FIRST else "second"
        logger.info(f"Game start: {player1.name} as {role}")
        logger.info(f"\n{state.display()}")

    while not state.done:
        # Select current player's policy
        policy = first_policy if state.current_player == Player.FIRST else second_policy

        # Get action (pass config for observation encoding)
        action = policy.select_action(state, config)
        state.make_move(action)
        moves += 1

        if verbose:
            logger.info(f"\n{policy.name} plays position {action}")
            logger.info(f"\n{state.display()}")

    # Determine winner from player1's perspective
    if state.winner is None:
        winner = None
    elif state.winner == player1_as:
        winner = 1
    else:
        winner = 2

    if verbose:
        if winner == 1:
            logger.info(f"{player1.name} wins!")
        elif winner == 2:
            logger.info(f"{player2.name} wins!")
        else:
            logger.info("Draw!")

    return MatchResult(winner=winner, moves=moves, player1_as=player1_as)


def evaluate(
    player1: Policy,
    player2: Policy,
    env_id: str,
    config: GameConfig | GameMetadata,
    num_games: int = 100,
    verbose: bool = False,
) -> EvalResults:
    """Run evaluation between two policies.

    Each policy plays half the games as first player and half as second
    to account for first-mover advantage.

    Args:
        player1: Policy to evaluate (typically the trained model).
        player2: Opponent policy (typically random).
        env_id: Environment ID for creating game states.
        config: Game configuration (from database or fallback).
        num_games: Total number of games to play.
        verbose: Print individual game details.

    Returns:
        EvalResults with aggregated statistics.
    """
    results = EvalResults(
        env_id=env_id,
        player1_name=player1.name,
        player2_name=player2.name,
        games_played=0,
        player1_wins=0,
        player2_wins=0,
        draws=0,
        player1_wins_as_first=0,
        player1_wins_as_second=0,
        player2_wins_as_first=0,
        player2_wins_as_second=0,
        avg_game_length=0.0,
    )

    total_moves = 0
    games_per_side = num_games // 2

    # Play half as first player, half as second
    for game_num in range(num_games):
        # Alternate sides
        player1_as = Player.FIRST if game_num < games_per_side else Player.SECOND

        result = play_game(
            player1, player2, player1_as, env_id, config, verbose=verbose
        )

        results.games_played += 1
        total_moves += result.moves

        if result.winner == 1:
            results.player1_wins += 1
            if player1_as == Player.FIRST:
                results.player1_wins_as_first += 1
            else:
                results.player1_wins_as_second += 1
        elif result.winner == 2:
            results.player2_wins += 1
            if player1_as == Player.FIRST:
                results.player2_wins_as_first += 1
            else:
                results.player2_wins_as_second += 1
        else:
            results.draws += 1

        # Progress logging
        if not verbose and (game_num + 1) % 10 == 0:
            logger.info(
                f"Progress: {game_num + 1}/{num_games} games "
                f"({results.player1_name}: {results.player1_wins}W, "
                f"{results.player2_name}: {results.player2_wins}W, "
                f"{results.draws}D)"
            )

    results.avg_game_length = total_moves / max(1, results.games_played)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model against random play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="../data/models/latest.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="../data/replay.db",
        help="Path to replay database (for loading game metadata)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="tictactoe",
        choices=["tictactoe", "connect4"],
        help="Game environment to evaluate",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print individual game moves",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load game configuration from database (preferred) or fallback to hardcoded
    # This follows the same pattern as the Rust actor, making the evaluator self-describing
    config = get_game_metadata_or_config(args.db, args.env_id)
    logger.info(
        f"Game config for {args.env_id}: "
        f"board={config.board_width}x{config.board_height}, "
        f"actions={config.num_actions}, obs_size={config.obs_size}"
    )

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    logger.info(f"Loading model: {model_path}")

    try:
        model_policy = OnnxPolicy(str(model_path), temperature=args.temperature)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    random_policy = RandomPolicy()

    logger.info(
        f"Running {args.games} games: {model_policy.name} vs {random_policy.name}"
    )
    logger.info(f"Environment: {args.env_id}")

    # Run evaluation with game configuration
    results = evaluate(
        player1=model_policy,
        player2=random_policy,
        env_id=args.env_id,
        config=config,
        num_games=args.games,
        verbose=args.verbose,
    )

    # Print results
    print(results.summary())

    # Interpretation
    print()
    if results.player1_win_rate > 0.7:
        print("✓ Model is significantly better than random play!")
    elif results.player1_win_rate > 0.5:
        print("~ Model is slightly better than random play.")
    elif results.player1_win_rate > 0.3:
        print("✗ Model is roughly equivalent to random play.")
    else:
        print("✗ Model is worse than random play!")

    # Game-specific analysis
    if args.env_id == "tictactoe" and results.draw_rate > 0.8:
        print(
            "\nNote: High draw rate suggests defensive play, which is optimal for TicTacToe."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
