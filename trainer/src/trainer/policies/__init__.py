"""Policy implementations for game playing.

This module provides different policies for selecting actions in games,
used primarily for model evaluation.
"""

from typing import TYPE_CHECKING, Protocol

from .onnx import OnnxPolicy
from .random import RandomPolicy

if TYPE_CHECKING:
    from ..game_config import GameConfig
    from ..games import GameState
    from ..storage import GameMetadata


class Policy(Protocol):
    """Protocol for action selection policies."""

    def select_action(
        self, state: "GameState", config: "GameConfig | GameMetadata"
    ) -> int:
        """Select an action given the current state and game config."""
        ...

    @property
    def name(self) -> str:
        """Return the policy name for logging."""
        ...


__all__ = [
    "OnnxPolicy",
    "Policy",
    "RandomPolicy",
]
