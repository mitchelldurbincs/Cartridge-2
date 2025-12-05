"""Evaluator for measuring trained model performance against baselines.

This module provides evaluation capabilities to measure how well a trained
model plays TicTacToe compared to random play. It runs many games between
different policies and reports win/loss/draw statistics.

Usage:
    python -m trainer.evaluator --model data/models/latest.onnx --games 100
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

logger = logging.getLogger(__name__)


class Player(IntEnum):
    """Player identifiers matching TicTacToe implementation."""
    X = 1  # First player
    O = 2  # Second player


class Cell(IntEnum):
    """Cell states."""
    EMPTY = 0
    X = 1
    O = 2


@dataclass
class GameState:
    """Pure Python TicTacToe state for evaluation."""
    board: list[int]  # 9 cells, 0=empty, 1=X, 2=O
    current_player: Player
    done: bool = False
    winner: int | None = None  # None=ongoing/draw, 1=X wins, 2=O wins

    @classmethod
    def new(cls) -> "GameState":
        return cls(board=[0] * 9, current_player=Player.X)

    def copy(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            done=self.done,
            winner=self.winner,
        )

    def legal_moves(self) -> list[int]:
        """Return indices of empty cells."""
        if self.done:
            return []
        return [i for i, cell in enumerate(self.board) if cell == Cell.EMPTY]

    def legal_moves_mask(self) -> list[float]:
        """Return mask where 1.0 = legal, 0.0 = illegal."""
        if self.done:
            return [0.0] * 9
        return [1.0 if cell == Cell.EMPTY else 0.0 for cell in self.board]

    def make_move(self, pos: int) -> None:
        """Make a move at the given position."""
        if self.done:
            raise ValueError("Game is already over")
        if self.board[pos] != Cell.EMPTY:
            raise ValueError(f"Position {pos} is not empty")

        self.board[pos] = self.current_player

        # Check for winner
        if self._check_winner(self.current_player):
            self.done = True
            self.winner = self.current_player
        elif all(cell != Cell.EMPTY for cell in self.board):
            # Draw
            self.done = True
            self.winner = None
        else:
            # Switch player
            self.current_player = Player.O if self.current_player == Player.X else Player.X

    def _check_winner(self, player: Player) -> bool:
        """Check if the given player has won."""
        b = self.board
        p = player

        # Rows
        if b[0] == b[1] == b[2] == p: return True
        if b[3] == b[4] == b[5] == p: return True
        if b[6] == b[7] == b[8] == p: return True

        # Columns
        if b[0] == b[3] == b[6] == p: return True
        if b[1] == b[4] == b[7] == p: return True
        if b[2] == b[5] == b[8] == p: return True

        # Diagonals
        if b[0] == b[4] == b[8] == p: return True
        if b[2] == b[4] == b[6] == p: return True

        return False

    def to_observation(self) -> np.ndarray:
        """Convert to neural network observation format (29 floats)."""
        obs = np.zeros(29, dtype=np.float32)

        # Board view: 18 floats (positions 0-8 for X, 9-17 for O)
        for i, cell in enumerate(self.board):
            if cell == Cell.X:
                obs[i] = 1.0
            elif cell == Cell.O:
                obs[i + 9] = 1.0

        # Legal moves mask: 9 floats (positions 18-26)
        for i, cell in enumerate(self.board):
            if cell == Cell.EMPTY and not self.done:
                obs[18 + i] = 1.0

        # Current player: 2 floats (positions 27-28)
        if self.current_player == Player.X:
            obs[27] = 1.0
        else:
            obs[28] = 1.0

        return obs

    def display(self) -> str:
        """Return a string representation of the board."""
        symbols = {Cell.EMPTY: ".", Cell.X: "X", Cell.O: "O"}
        rows = []
        for row in range(3):
            cells = [symbols[Cell(self.board[row * 3 + col])] for col in range(3)]
            rows.append(" ".join(cells))
        return "\n".join(rows)


class Policy(Protocol):
    """Protocol for action selection policies."""

    def select_action(self, state: GameState) -> int:
        """Select an action given the current state."""
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

    def select_action(self, state: GameState) -> int:
        legal = state.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")
        return random.choice(legal)


class OnnxPolicy:
    """Policy using an ONNX neural network model."""

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

    def select_action(self, state: GameState) -> int:
        legal = state.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        # Get observation
        obs = state.to_observation()
        obs_batch = obs.reshape(1, -1)

        # Run inference
        outputs = self.session.run(
            [self.policy_output, self.value_output],
            {self.input_name: obs_batch},
        )
        policy_logits = outputs[0][0]  # Shape: (9,)

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
            return int(np.random.choice(9, p=probs))

    def get_value(self, state: GameState) -> float:
        """Get the value estimate for a state."""
        obs = state.to_observation()
        obs_batch = obs.reshape(1, -1)

        outputs = self.session.run(
            [self.value_output],
            {self.input_name: obs_batch},
        )
        return float(outputs[0][0][0])


@dataclass
class MatchResult:
    """Result of a single game."""
    winner: int | None  # 1=player1, 2=player2, None=draw
    moves: int
    player1_as: Player  # Which color player1 played


@dataclass
class EvalResults:
    """Aggregated evaluation results."""
    player1_name: str
    player2_name: str
    games_played: int
    player1_wins: int
    player2_wins: int
    draws: int
    player1_wins_as_x: int
    player1_wins_as_o: int
    player2_wins_as_x: int
    player2_wins_as_o: int
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
            f"{'=' * 50}",
            f"Games played: {self.games_played}",
            f"",
            f"{self.player1_name}:",
            f"  Wins: {self.player1_wins} ({self.player1_win_rate:.1%})",
            f"    As X (first): {self.player1_wins_as_x}",
            f"    As O (second): {self.player1_wins_as_o}",
            f"",
            f"{self.player2_name}:",
            f"  Wins: {self.player2_wins} ({self.player2_win_rate:.1%})",
            f"    As X (first): {self.player2_wins_as_x}",
            f"    As O (second): {self.player2_wins_as_o}",
            f"",
            f"Draws: {self.draws} ({self.draw_rate:.1%})",
            f"Average game length: {self.avg_game_length:.1f} moves",
            f"{'=' * 50}",
        ]
        return "\n".join(lines)


def play_game(player1: Policy, player2: Policy, player1_as: Player, verbose: bool = False) -> MatchResult:
    """Play a single game between two policies.

    Args:
        player1: First policy to evaluate.
        player2: Second policy (opponent).
        player1_as: Which color player1 plays (X or O).
        verbose: Print game moves.

    Returns:
        MatchResult with winner and game length.
    """
    state = GameState.new()
    moves = 0

    # Assign policies to colors
    if player1_as == Player.X:
        x_policy, o_policy = player1, player2
    else:
        x_policy, o_policy = player2, player1

    if verbose:
        logger.info(f"Game start: {player1.name} as {'X' if player1_as == Player.X else 'O'}")
        logger.info(f"\n{state.display()}")

    while not state.done:
        # Select current player's policy
        policy = x_policy if state.current_player == Player.X else o_policy

        # Get action
        action = policy.select_action(state)
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
    num_games: int = 100,
    verbose: bool = False,
) -> EvalResults:
    """Run evaluation between two policies.

    Each policy plays half the games as X (first) and half as O (second)
    to account for first-mover advantage.

    Args:
        player1: Policy to evaluate (typically the trained model).
        player2: Opponent policy (typically random).
        num_games: Total number of games to play.
        verbose: Print individual game details.

    Returns:
        EvalResults with aggregated statistics.
    """
    results = EvalResults(
        player1_name=player1.name,
        player2_name=player2.name,
        games_played=0,
        player1_wins=0,
        player2_wins=0,
        draws=0,
        player1_wins_as_x=0,
        player1_wins_as_o=0,
        player2_wins_as_x=0,
        player2_wins_as_o=0,
        avg_game_length=0.0,
    )

    total_moves = 0
    games_per_side = num_games // 2

    # Play half as X, half as O
    for game_num in range(num_games):
        # Alternate sides: even games as X, odd as O
        player1_as = Player.X if game_num < games_per_side else Player.O

        result = play_game(player1, player2, player1_as, verbose=verbose)

        results.games_played += 1
        total_moves += result.moves

        if result.winner == 1:
            results.player1_wins += 1
            if player1_as == Player.X:
                results.player1_wins_as_x += 1
            else:
                results.player1_wins_as_o += 1
        elif result.winner == 2:
            results.player2_wins += 1
            if player1_as == Player.X:
                results.player2_wins_as_x += 1
            else:
                results.player2_wins_as_o += 1
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
        default="./data/models/latest.onnx",
        help="Path to ONNX model file",
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

    logger.info(f"Running {args.games} games: {model_policy.name} vs {random_policy.name}")

    # Run evaluation
    results = evaluate(
        player1=model_policy,
        player2=random_policy,
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

    # TicTacToe-specific analysis
    if results.draw_rate > 0.8:
        print("\nNote: High draw rate suggests defensive play, which is optimal for TicTacToe.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
