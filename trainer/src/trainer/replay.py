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

import logging
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

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
    game_outcome: (
        float | None
    )  # Final game outcome from this player's perspective (+1/-1/0)


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

        # Use timeout to handle concurrent access from actor, and WAL mode for
        # better concurrency (allows readers while writer is active)
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
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

    def count(self, env_id: str | None = None) -> int:
        """Get total number of transitions in the buffer.

        Args:
            env_id: Optional environment ID to filter by. If None, counts all transitions.
        """
        if env_id is not None:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM transitions WHERE env_id = ?", (env_id,)
            )
        else:
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

    def sample(self, batch_size: int, env_id: str | None = None) -> list[Transition]:
        """Sample random transitions for training.

        Uses efficient rowid-based sampling when the table is dense enough,
        falling back to ORDER BY RANDOM() for sparse tables or when rowid
        sampling doesn't return enough results.

        Args:
            batch_size: Number of transitions to sample.
            env_id: Optional environment ID to filter by. If None, samples from all games.
        """
        # Try efficient rowid-based sampling first
        result = self._sample_by_rowid(batch_size, env_id)
        if len(result) >= batch_size:
            return result[:batch_size]

        # Fall back to ORDER BY RANDOM() if rowid sampling didn't get enough
        return self._sample_fallback(batch_size, env_id)

    def _sample_by_rowid(
        self, batch_size: int, env_id: str | None = None
    ) -> list[Transition]:
        """Sample using rowid-based random selection. O(k) for dense tables.

        This is much faster than ORDER BY RANDOM() which is O(n log n).
        For a table with 100K rows, this can be 10-100x faster.
        """
        # Get rowid bounds and count
        if env_id is not None:
            cursor = self._conn.execute(
                """SELECT MIN(rowid), MAX(rowid), COUNT(*)
                   FROM transitions WHERE env_id = ?""",
                (env_id,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM transitions"
            )

        row = cursor.fetchone()
        min_rowid, max_rowid, total = row

        if total == 0 or min_rowid is None:
            return []

        if total <= batch_size:
            # Return all rows if we don't have enough
            return self._fetch_all_transitions(env_id)

        # Calculate table density (ratio of filled rows to rowid range)
        rowid_range = max_rowid - min_rowid + 1
        density = total / rowid_range

        # If table is too sparse (<30% filled), fall back to ORDER BY RANDOM()
        # because we'd need too many rowid samples to get enough valid rows
        if density < 0.3:
            return []  # Signal to use fallback

        # Generate random rowids with oversampling to account for gaps
        # Higher oversample for lower density
        oversample_factor = min(3.0, 1.0 / density + 0.5)
        sample_size = min(int(batch_size * oversample_factor * 1.5), rowid_range)

        # Generate unique random rowids
        if sample_size >= rowid_range:
            random_rowids = list(range(min_rowid, max_rowid + 1))
        else:
            random_rowids = random.sample(range(min_rowid, max_rowid + 1), sample_size)

        # Build query with IN clause
        placeholders = ",".join("?" * len(random_rowids))

        if env_id is not None:
            query = f"""
                SELECT id, env_id, episode_id, step_number, state, action, next_state,
                       observation, next_observation, reward, done, timestamp,
                       policy_probs, mcts_value, game_outcome
                FROM transitions
                WHERE rowid IN ({placeholders}) AND env_id = ?
                LIMIT ?
            """
            params = (*random_rowids, env_id, batch_size)
        else:
            query = f"""
                SELECT id, env_id, episode_id, step_number, state, action, next_state,
                       observation, next_observation, reward, done, timestamp,
                       policy_probs, mcts_value, game_outcome
                FROM transitions
                WHERE rowid IN ({placeholders})
                LIMIT ?
            """
            params = (*random_rowids, batch_size)

        cursor = self._conn.execute(query, params)
        return self._rows_to_transitions(cursor.fetchall())

    def _sample_fallback(
        self, batch_size: int, env_id: str | None = None
    ) -> list[Transition]:
        """Fallback sampling using ORDER BY RANDOM(). O(n log n) but always works."""
        if env_id is not None:
            cursor = self._conn.execute(
                """
                SELECT id, env_id, episode_id, step_number, state, action, next_state,
                       observation, next_observation, reward, done, timestamp,
                       policy_probs, mcts_value, game_outcome
                FROM transitions
                WHERE env_id = ?
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (env_id, batch_size),
            )
        else:
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

        return self._rows_to_transitions(cursor.fetchall())

    def _fetch_all_transitions(self, env_id: str | None = None) -> list[Transition]:
        """Fetch all transitions (used when total <= batch_size)."""
        if env_id is not None:
            cursor = self._conn.execute(
                """
                SELECT id, env_id, episode_id, step_number, state, action, next_state,
                       observation, next_observation, reward, done, timestamp,
                       policy_probs, mcts_value, game_outcome
                FROM transitions WHERE env_id = ?
                """,
                (env_id,),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT id, env_id, episode_id, step_number, state, action, next_state,
                       observation, next_observation, reward, done, timestamp,
                       policy_probs, mcts_value, game_outcome
                FROM transitions
                """
            )
        return self._rows_to_transitions(cursor.fetchall())

    def _rows_to_transitions(self, rows: list) -> list[Transition]:
        """Convert database rows to Transition objects."""
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
            for row in rows
        ]

    def sample_batch_tensors(
        self, batch_size: int, num_actions: int, env_id: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Sample transitions and return as numpy arrays ready for training.

        Uses pre-allocated numpy arrays to avoid intermediate list allocations,
        reducing memory overhead by 10-20%.

        Args:
            batch_size: Number of transitions to sample.
            num_actions: Number of actions for the game (e.g., 9 for TicTacToe, 7 for Connect4).
                         Get this from GameMetadata.num_actions.
            env_id: Optional environment ID to filter by. If None, samples from all games.

        Returns:
            Tuple of (observations, policy_targets, value_targets) as numpy arrays,
            or None if not enough data.

            - observations: Shape (batch, obs_size) - game observations
            - policy_targets: Shape (batch, num_actions) - MCTS visit distributions
            - value_targets: Shape (batch,) - game outcomes (+1/-1/0) from player's perspective
        """
        transitions = self.sample(batch_size, env_id=env_id)
        if len(transitions) < batch_size:
            return None

        # Determine observation size from first transition
        first_obs = np.frombuffer(transitions[0].observation, dtype=np.float32)
        obs_size = len(first_obs)

        # Pre-allocate arrays (avoids intermediate list allocations)
        observations = np.empty((batch_size, obs_size), dtype=np.float32)
        policy_targets = np.empty((batch_size, num_actions), dtype=np.float32)
        value_targets = np.empty(batch_size, dtype=np.float32)

        # Fill first observation (already parsed)
        observations[0] = first_obs

        # Process first transition's policy and value
        t = transitions[0]
        if t.policy_probs is not None and len(t.policy_probs) > 0:
            policy = np.frombuffer(t.policy_probs, dtype=np.float32)
            if len(policy) == num_actions:
                policy_targets[0] = policy
            else:
                # Policy shape mismatch - log warning and fall back to one-hot
                logger.warning(
                    f"Policy shape mismatch: got {len(policy)}, expected {num_actions}. "
                    f"Falling back to one-hot for transition {t.id}"
                )
                policy_targets[0] = 0.0
                action_idx = int.from_bytes(t.action, byteorder="little")
                policy_targets[0, action_idx] = 1.0
        else:
            policy_targets[0] = 0.0
            action_idx = int.from_bytes(t.action, byteorder="little")
            policy_targets[0, action_idx] = 1.0

        # Clamp value targets to valid range [-1, 1] to prevent training instability
        raw_value = t.game_outcome if t.game_outcome is not None else t.mcts_value
        value_targets[0] = np.clip(raw_value, -1.0, 1.0)

        # Process remaining transitions
        for i in range(1, batch_size):
            t = transitions[i]

            # Parse observation directly into pre-allocated array
            observations[i] = np.frombuffer(t.observation, dtype=np.float32)

            # Use MCTS policy distribution as target if available
            if t.policy_probs is not None and len(t.policy_probs) > 0:
                policy = np.frombuffer(t.policy_probs, dtype=np.float32)
                if len(policy) == num_actions:
                    policy_targets[i] = policy
                else:
                    # Policy shape mismatch - log warning and fall back to one-hot
                    logger.warning(
                        f"Policy shape mismatch: got {len(policy)}, expected {num_actions}. "
                        f"Falling back to one-hot for transition {t.id}"
                    )
                    policy_targets[i] = 0.0
                    action_idx = int.from_bytes(t.action, byteorder="little")
                    policy_targets[i, action_idx] = 1.0
            else:
                # Fallback to one-hot action if no MCTS data (backward compat)
                policy_targets[i] = 0.0
                action_idx = int.from_bytes(t.action, byteorder="little")
                policy_targets[i, action_idx] = 1.0

            # Use game_outcome as value target (propagated from terminal state)
            # Falls back to mcts_value for backward compatibility with old data
            # Clamp to valid range [-1, 1] to prevent training instability
            raw_value = t.game_outcome if t.game_outcome is not None else t.mcts_value
            value_targets[i] = np.clip(raw_value, -1.0, 1.0)

        return observations, policy_targets, value_targets

    def iter_batches(
        self,
        batch_size: int,
        num_actions: int,
        max_batches: int | None = None,
        env_id: str | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate through random batches.

        Args:
            batch_size: Number of transitions per batch.
            num_actions: Number of actions for the game.
            max_batches: Maximum number of batches to yield.
            env_id: Optional environment ID to filter by. If None, samples from all games.

        Yields batches until max_batches is reached or buffer is exhausted.
        """
        batches_yielded = 0
        while True:
            if max_batches is not None and batches_yielded >= max_batches:
                break

            batch = self.sample_batch_tensors(
                batch_size, num_actions=num_actions, env_id=env_id
            )
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

    def clear_transitions(self) -> int:
        """Delete all transitions from the buffer.

        Preserves game_metadata table. Used for synchronized AlphaZero
        training where each iteration trains only on data from the current model.

        Returns the number of deleted transitions.
        """
        cursor = self._conn.execute("DELETE FROM transitions")
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
