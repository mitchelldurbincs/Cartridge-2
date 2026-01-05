"""SQLite replay buffer for loading training data.

This module provides read access to the replay buffer populated by the Rust actor.
The schema matches actor/src/replay.rs exactly.

BACKWARD COMPATIBILITY:
    This module now re-exports classes from trainer.storage for backward compatibility.
    New code should use `from trainer.storage import create_replay_buffer` instead.

    For K8s deployments, configure the storage backend in config.toml or via
    environment variables:
        CARTRIDGE_STORAGE_BACKEND=postgres
        CARTRIDGE_POSTGRES_URL=postgresql://user:pass@host:5432/cartridge

Schema (from Rust):
    CREATE TABLE transitions (
        id TEXT PRIMARY KEY,
        env_id TEXT NOT NULL,
        episode_id TEXT NOT NULL,
        step_number INTEGER NOT NULL,
        state BLOB NOT NULL,
        action BLOB NOT NULL,
        next_state BLOB NOT NULL,
        observation BLOB NOT NULL,
        next_observation BLOB NOT NULL,
        reward REAL NOT NULL,
        done INTEGER NOT NULL,
        timestamp INTEGER NOT NULL,
        policy_probs BLOB,              -- MCTS visit distribution (f32[num_actions])
        mcts_value REAL DEFAULT 0.0,    -- MCTS value estimate at position
        game_outcome REAL,              -- Propagated outcome (+1/-1/0) from player's perspective
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )

Training targets:
    - Policy targets: MCTS visit count distributions (normalized) from the actor.
      Falls back to one-hot action encoding if policy_probs is not available.
    - Value targets: Game outcomes (win=+1, loss=-1, draw=0) propagated from
      terminal states. Each position is labeled with the final outcome from
      that position's player's perspective.
"""

import sqlite3
from pathlib import Path

import numpy as np

# Re-export from new storage module for backward compatibility
from trainer.storage.base import GameMetadata, Transition
from trainer.storage.sqlite import (
    EXPECTED_COLUMNS,
    SchemaError,
    SqliteReplayBuffer,
    create_empty_db,
)

# Alias for backward compatibility - ReplayBuffer now uses SqliteReplayBuffer
# For new code, use: from trainer.storage import create_replay_buffer
ReplayBuffer = SqliteReplayBuffer

# Re-export all public symbols for backward compatibility
__all__ = [
    "ReplayBuffer",
    "GameMetadata",
    "Transition",
    "SchemaError",
    "EXPECTED_COLUMNS",
    "create_empty_db",
    "insert_test_transitions",
]


def insert_test_transitions(db_path: str | Path, count: int = 100) -> None:
    """Insert synthetic test transitions for development.

    Creates random TicTacToe-like transitions with MCTS policy data.
    """
    import struct
    import uuid

    conn = sqlite3.connect(str(db_path))

    for i in range(count):
        episode_id = str(uuid.uuid4())

        # Create synthetic observation (29 f32 values)
        obs = np.zeros(29, dtype=np.float32)
        # Random board positions
        obs[:9] = np.random.choice([0.0, 1.0], size=9, p=[0.7, 0.3])  # X positions
        obs[9:18] = np.random.choice([0.0, 1.0], size=9, p=[0.7, 0.3])  # O positions
        # Legal moves (positions not occupied)
        obs[18:27] = 1.0 - np.maximum(obs[:9], obs[9:18])
        # Current player
        obs[27] = 1.0 if i % 2 == 0 else 0.0
        obs[28] = 0.0 if i % 2 == 0 else 1.0

        obs_bytes = obs.tobytes()

        # Random state (11 bytes for TicTacToe)
        state = bytes([0] * 9 + [1, 0])

        # Random action (u32 position 0-8)
        action_idx = np.random.randint(0, 9)
        action_bytes = struct.pack("<I", action_idx)

        # Random reward (-1, 0, or 1)
        reward = float(np.random.choice([-1.0, 0.0, 1.0], p=[0.3, 0.4, 0.3]))

        # Generate synthetic MCTS policy (soft distribution over legal moves)
        legal_mask = obs[18:27]
        policy = np.zeros(9, dtype=np.float32)
        if legal_mask.sum() > 0:
            raw_probs = np.random.random(9).astype(np.float32) * legal_mask
            policy = raw_probs / (raw_probs.sum() + 1e-8)
        else:
            # No legal moves - use uniform distribution to avoid NaN in training
            policy = np.ones(9, dtype=np.float32) / 9.0
        policy_bytes = policy.tobytes()

        # Random MCTS value estimate
        mcts_value = float(np.random.uniform(-1.0, 1.0))

        # Game outcome (same as reward for terminal, random for non-terminal)
        game_outcome = (
            reward if reward != 0 else float(np.random.choice([-1.0, 0.0, 1.0]))
        )

        conn.execute(
            """
            INSERT INTO transitions
            (id, env_id, episode_id, step_number, state, action, next_state,
             observation, next_observation, reward, done, timestamp,
             policy_probs, mcts_value, game_outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"test-{i}",
                "tictactoe",
                episode_id,
                i % 9,
                state,
                action_bytes,
                state,
                obs_bytes,
                obs_bytes,
                reward,
                1 if reward != 0 else 0,
                i,
                policy_bytes,
                mcts_value,
                game_outcome,
            ),
        )

    conn.commit()
    conn.close()
