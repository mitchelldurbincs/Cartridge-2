"""Game configuration registry for the trainer.

This module provides game-specific configuration that the trainer needs
for neural network architecture and observation parsing.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class GameConfig:
    """Game configuration for the trainer."""

    # Game identity
    env_id: str
    display_name: str

    # Board dimensions
    board_width: int
    board_height: int

    # Neural network dimensions
    num_actions: int
    obs_size: int
    legal_mask_offset: int

    # Network architecture hints
    hidden_size: int = 128

    @property
    def board_size(self) -> int:
        """Total number of board cells."""
        return self.board_width * self.board_height

    @property
    def legal_mask_bits(self) -> int:
        """Bitmask for extracting legal moves from info bits."""
        return (1 << self.num_actions) - 1


# Game configuration registry
GAME_CONFIGS: Dict[str, GameConfig] = {
    "tictactoe": GameConfig(
        env_id="tictactoe",
        display_name="Tic-Tac-Toe",
        board_width=3,
        board_height=3,
        num_actions=9,
        obs_size=29,
        legal_mask_offset=18,
        hidden_size=128,
    ),
    "connect4": GameConfig(
        env_id="connect4",
        display_name="Connect 4",
        board_width=7,
        board_height=6,
        num_actions=7,
        obs_size=100,  # Placeholder - will be updated
        legal_mask_offset=86,  # Placeholder - will be updated
        hidden_size=256,
    ),
    "othello": GameConfig(
        env_id="othello",
        display_name="Othello",
        board_width=8,
        board_height=8,
        num_actions=64,
        obs_size=195,  # Placeholder - will be updated
        legal_mask_offset=129,  # Placeholder - will be updated
        hidden_size=512,
    ),
}


def get_config(env_id: str) -> GameConfig:
    """Get the game configuration for a given environment ID.

    Args:
        env_id: Environment identifier (e.g., "tictactoe", "connect4")

    Returns:
        GameConfig for the specified game.

    Raises:
        ValueError: If the game is not registered.
    """
    if env_id not in GAME_CONFIGS:
        available = ", ".join(GAME_CONFIGS.keys())
        raise ValueError(
            f"Unknown game: {env_id}. Available games: {available}"
        )
    return GAME_CONFIGS[env_id]


def list_games() -> list[str]:
    """List all registered game IDs."""
    return list(GAME_CONFIGS.keys())


if __name__ == "__main__":
    # Simple test
    for env_id in list_games():
        config = get_config(env_id)
        print(f"{config.display_name}:")
        print(f"  Board: {config.board_width}x{config.board_height}")
        print(f"  Actions: {config.num_actions}")
        print(f"  Obs size: {config.obs_size}")
        print(f"  Legal mask offset: {config.legal_mask_offset}")
        print()
