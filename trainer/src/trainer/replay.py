"""SQLite replay buffer for loading training data.

This module provides read access to the replay buffer populated by the Rust actor.
The schema matches actor/src/replay.rs exactly.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


# Expected columns in the transitions table (order matters for validation)
EXPECTED_COLUMNS = [
    ("id", "TEXT"),
    ("env_id", "TEXT"),
    ("episode_id", "TEXT"),
    ("step_number", "INTEGER"),
    ("state", "BLOB"),
    ("action", "BLOB"),
    ("next_state", "BLOB"),
    ("observation", "BLOB"),
    ("next_observation", "BLOB"),
    ("reward", "REAL"),
    ("done", "INTEGER"),
    ("timestamp", "INTEGER"),
    ("policy_probs", "BLOB"),
    ("mcts_value", "REAL"),
    ("game_outcome", "REAL"),
]


class SchemaError(Exception):
    """Raised when the database schema doesn't match expected format."""

    pass


@dataclass
class GameMetadata:
    """Game metadata stored in the replay database by the actor.

    This makes the database self-describing, allowing the trainer to
    dynamically configure itself based on the game being trained.
    """

    env_id: str
    display_name: str
    board_width: int
    board_height: int
    num_actions: int
    obs_size: int
    legal_mask_offset: int
    player_count: int

    @property
    def board_size(self) -> int:
        """Total number of board cells."""
        return self.board_width * self.board_height

    @property
    def legal_mask_end(self) -> int:
        """End index of legal mask in observation."""
        return self.legal_mask_offset + self.num_actions


@dataclass
class Transition:
    """A single transition from the replay buffer."""

    id: str
    env_id: str
    episode_id: str
    step_number: int
    state: bytes
    action: bytes
    next_state: bytes
    observation: bytes
    next_observation: bytes
    reward: float
    done: bool
    timestamp: int
    policy_probs: bytes | None  # f32[num_actions] MCTS visit distribution
    mcts_value: float  # MCTS value estimate at this position
    game_outcome: float | None  # Final game outcome from this player's perspective (+1/-1/0)


class ReplayBuffer:
    """Read-only interface to the SQLite replay buffer.

    This class provides random sampling of transitions for training,
    matching the schema created by the Rust actor.

    Raises:
        FileNotFoundError: If the database file doesn't exist.
        SchemaError: If the database schema doesn't match expected format.
    """

    def __init__(self, db_path: str | Path, validate_schema: bool = True):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Replay database not found: {db_path}")

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        if validate_schema:
            self._validate_schema()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "ReplayBuffer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _validate_schema(self) -> None:
        """Validate that the database has the expected schema.

        Raises:
            SchemaError: If the schema doesn't match expectations.
        """
        # Check if transitions table exists
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transitions'"
        )
        if cursor.fetchone() is None:
            raise SchemaError(
                "Table 'transitions' not found in database. "
                "Ensure the Rust actor has initialized the replay buffer."
            )

        # Get column info
        cursor = self._conn.execute("PRAGMA table_info(transitions)")
        columns = {row[1]: row[2].upper() for row in cursor.fetchall()}

        # Check for expected columns
        missing = []
        type_mismatches = []
        for col_name, expected_type in EXPECTED_COLUMNS:
            if col_name not in columns:
                missing.append(col_name)
            elif not columns[col_name].startswith(expected_type):
                type_mismatches.append(
                    f"{col_name}: expected {expected_type}, got {columns[col_name]}"
                )

        if missing:
            raise SchemaError(
                f"Missing columns in transitions table: {missing}. "
                "The Rust actor schema may have changed. "
                "Expected columns: {[c[0] for c in EXPECTED_COLUMNS]}"
            )

        if type_mismatches:
            raise SchemaError(
                f"Column type mismatches in transitions table: {type_mismatches}"
            )

    def count(self) -> int:
        """Get total number of transitions in the buffer."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM transitions")
        return cursor.fetchone()[0]

    def get_metadata(self, env_id: str | None = None) -> GameMetadata | None:
        """Get game metadata from the database.

        Args:
            env_id: Specific game to get metadata for. If None, returns metadata
                    for the first game found (useful when DB has only one game).

        Returns:
            GameMetadata if found, None otherwise.
        """
        # Check if game_metadata table exists
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_metadata'"
        )
        if cursor.fetchone() is None:
            return None

        if env_id:
            cursor = self._conn.execute(
                """SELECT env_id, display_name, board_width, board_height,
                          num_actions, obs_size, legal_mask_offset, player_count
                   FROM game_metadata WHERE env_id = ?""",
                (env_id,),
            )
        else:
            cursor = self._conn.execute(
                """SELECT env_id, display_name, board_width, board_height,
                          num_actions, obs_size, legal_mask_offset, player_count
                   FROM game_metadata LIMIT 1"""
            )

        row = cursor.fetchone()
        if row is None:
            return None

        return GameMetadata(
            env_id=row[0],
            display_name=row[1],
            board_width=row[2],
            board_height=row[3],
            num_actions=row[4],
            obs_size=row[5],
            legal_mask_offset=row[6],
            player_count=row[7],
        )

    def list_metadata(self) -> list[GameMetadata]:
        """List all game metadata in the database."""
        # Check if game_metadata table exists
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_metadata'"
        )
        if cursor.fetchone() is None:
            return []

        cursor = self._conn.execute(
            """SELECT env_id, display_name, board_width, board_height,
                      num_actions, obs_size, legal_mask_offset, player_count
               FROM game_metadata"""
        )

        return [
            GameMetadata(
                env_id=row[0],
                display_name=row[1],
                board_width=row[2],
                board_height=row[3],
                num_actions=row[4],
                obs_size=row[5],
                legal_mask_offset=row[6],
                player_count=row[7],
            )
            for row in cursor.fetchall()
        ]

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample random transitions for training.

        Uses SQLite ORDER BY RANDOM() for uniform sampling.
        """
        cursor = self._conn.execute(
            """
            SELECT id, env_id, episode_id, step_number, state, action, next_state,
                   observation, next_observation, reward, done, timestamp,
                   policy_probs, mcts_value, game_outcome
            FROM transitions
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (batch_size,),
        )

        return [
            Transition(
                id=row["id"],
                env_id=row["env_id"],
                episode_id=row["episode_id"],
                step_number=row["step_number"],
                state=row["state"],
                action=row["action"],
                next_state=row["next_state"],
                observation=row["observation"],
                next_observation=row["next_observation"],
                reward=row["reward"],
                done=bool(row["done"]),
                timestamp=row["timestamp"],
                policy_probs=row["policy_probs"],
                mcts_value=row["mcts_value"] or 0.0,
                game_outcome=row["game_outcome"],
            )
            for row in cursor.fetchall()
        ]

    def sample_batch_tensors(
        self, batch_size: int, num_actions: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Sample transitions and return as numpy arrays ready for training.

        Args:
            batch_size: Number of transitions to sample.
            num_actions: Number of actions for the game (e.g., 9 for TicTacToe, 7 for Connect4).
                         Get this from GameMetadata.num_actions.

        Returns:
            Tuple of (observations, policy_targets, value_targets) as numpy arrays,
            or None if not enough data.

            - observations: Shape (batch, obs_size) - game observations
            - policy_targets: Shape (batch, num_actions) - MCTS visit distributions
            - value_targets: Shape (batch,) - game outcomes (+1/-1/0) from player's perspective
        """
        transitions = self.sample(batch_size)
        if len(transitions) < batch_size:
            return None

        observations = []
        policy_targets = []
        value_targets = []

        for t in transitions:
            # Parse observation (game-specific size, determined by metadata)
            obs = np.frombuffer(t.observation, dtype=np.float32)
            observations.append(obs)

            # Use MCTS policy distribution as target if available
            if t.policy_probs is not None and len(t.policy_probs) > 0:
                policy = np.frombuffer(t.policy_probs, dtype=np.float32)
                # Ensure correct shape
                if len(policy) != num_actions:
                    # Fallback to one-hot if shape mismatch
                    action_idx = int.from_bytes(t.action, byteorder="little")
                    policy = np.zeros(num_actions, dtype=np.float32)
                    policy[action_idx] = 1.0
            else:
                # Fallback to one-hot action if no MCTS data (backward compat)
                action_idx = int.from_bytes(t.action, byteorder="little")
                policy = np.zeros(num_actions, dtype=np.float32)
                policy[action_idx] = 1.0
            policy_targets.append(policy)

            # Use game_outcome as value target (propagated from terminal state)
            # This gives proper AlphaZero training signal at every position
            # Falls back to mcts_value for backward compatibility with old data
            if t.game_outcome is not None:
                value_targets.append(t.game_outcome)
            else:
                value_targets.append(t.mcts_value)

        return (
            np.array(observations, dtype=np.float32),
            np.array(policy_targets, dtype=np.float32),
            np.array(value_targets, dtype=np.float32),
        )

    def iter_batches(
        self, batch_size: int, num_actions: int, max_batches: int | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate through random batches.

        Args:
            batch_size: Number of transitions per batch.
            num_actions: Number of actions for the game.
            max_batches: Maximum number of batches to yield.

        Yields batches until max_batches is reached or buffer is exhausted.
        """
        batches_yielded = 0
        while True:
            if max_batches is not None and batches_yielded >= max_batches:
                break

            batch = self.sample_batch_tensors(batch_size, num_actions)
            if batch is None:
                break

            yield batch
            batches_yielded += 1

    def cleanup(self, window_size: int) -> int:
        """Delete old transitions to maintain a sliding window.

        Returns the number of deleted transitions.
        """
        cursor = self._conn.execute(
            """
            DELETE FROM transitions
            WHERE id NOT IN (
                SELECT id FROM transitions
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            (window_size,),
        )
        self._conn.commit()
        return cursor.rowcount


def create_empty_db(db_path: str | Path) -> None:
    """Create an empty replay database with the correct schema.

    Useful for testing without the Rust actor.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transitions (
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
            policy_probs BLOB,
            mcts_value REAL DEFAULT 0.0,
            game_outcome REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON transitions(timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_transitions_episode ON transitions(episode_id)"
    )
    conn.commit()
    conn.close()


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
        policy_bytes = policy.tobytes()

        # Random MCTS value estimate
        mcts_value = float(np.random.uniform(-1.0, 1.0))

        # Game outcome (same as reward for terminal, random for non-terminal)
        game_outcome = reward if reward != 0 else float(np.random.choice([-1.0, 0.0, 1.0]))

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
