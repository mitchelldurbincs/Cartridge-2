"""Random policy implementation."""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..game_config import GameConfig
    from ..games import GameState
    from ..storage import GameMetadata


class RandomPolicy:
    """Uniformly random policy over legal moves."""

    @property
    def name(self) -> str:
        return "Random"

    def select_action(
        self, state: "GameState", config: "GameConfig | GameMetadata"
    ) -> int:
        legal = state.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")
        return random.choice(legal)
