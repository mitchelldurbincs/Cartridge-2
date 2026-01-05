"""SQLite backend for replay buffer storage.

This is the default backend for local development and single-machine training.
It wraps the existing SQLite-based replay buffer implementation.
"""

import random
import sqlite3
from pathlib import Path

import numpy as np

from trainer.storage.base import GameMetadata, ReplayBufferBase, Transition

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


class SqliteReplayBuffer(ReplayBufferBase):
    """SQLite-backed replay buffer implementation.

    This is the default backend for local development. It provides
    efficient random sampling and supports concurrent reads from
    the trainer while the actor writes.
    """

    def __init__(self, db_path: str | Path, validate_schema: bool = True):
        """Initialize SQLite replay buffer.

        Args:
            db_path: Path to the SQLite database file.
            validate_schema: Whether to validate the schema on connect.

        Raises:
            FileNotFoundError: If the database file doesn't exist.
            SchemaError: If the database schema doesn't match expected format.
        """
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

    def _validate_schema(self) -> None:
        """Validate that the database has the expected schema."""
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transitions'"
        )
        if cursor.fetchone() is None:
            raise SchemaError(
                "Table 'transitions' not found in database. "
                "Ensure the Rust actor has initialized the replay buffer."
            )

        cursor = self._conn.execute("PRAGMA table_info(transitions)")
        columns = {row[1]: row[2].upper() for row in cursor.fetchall()}

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
                f"Expected columns: {[c[0] for c in EXPECTED_COLUMNS]}"
            )

        if type_mismatches:
            raise SchemaError(
                f"Column type mismatches in transitions table: {type_mismatches}"
            )

    def count(self, env_id: str | None = None) -> int:
        """Get total number of transitions in the buffer."""
        if env_id is not None:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM transitions WHERE env_id = ?", (env_id,)
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM transitions")
        return cursor.fetchone()[0]

    def get_metadata(self, env_id: str | None = None) -> GameMetadata | None:
        """Get game metadata from the database."""
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
        """Sample random transitions for training."""
        result = self._sample_by_rowid(batch_size, env_id)
        if len(result) >= batch_size:
            return result[:batch_size]

        return self._sample_fallback(batch_size, env_id)

    def _sample_by_rowid(
        self, batch_size: int, env_id: str | None = None
    ) -> list[Transition]:
        """Sample using rowid-based random selection. O(k) for dense tables."""
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
            return self._fetch_all_transitions(env_id)

        rowid_range = max_rowid - min_rowid + 1
        density = total / rowid_range

        if density < 0.3:
            return []  # Signal to use fallback

        oversample_factor = min(3.0, 1.0 / density + 0.5)
        sample_size = min(int(batch_size * oversample_factor * 1.5), rowid_range)

        if sample_size >= rowid_range:
            random_rowids = list(range(min_rowid, max_rowid + 1))
        else:
            random_rowids = random.sample(range(min_rowid, max_rowid + 1), sample_size)

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
        """Fallback sampling using ORDER BY RANDOM()."""
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

        Optimized implementation using pre-allocated arrays.
        """
        transitions = self.sample(batch_size, env_id=env_id)
        if len(transitions) < batch_size:
            return None

        first_obs = np.frombuffer(transitions[0].observation, dtype=np.float32)
        obs_size = len(first_obs)

        observations = np.empty((batch_size, obs_size), dtype=np.float32)
        policy_targets = np.empty((batch_size, num_actions), dtype=np.float32)
        value_targets = np.empty(batch_size, dtype=np.float32)

        observations[0] = first_obs

        t = transitions[0]
        if t.policy_probs is not None and len(t.policy_probs) > 0:
            policy = np.frombuffer(t.policy_probs, dtype=np.float32)
            if len(policy) == num_actions:
                policy_targets[0] = policy
            else:
                policy_targets[0] = 0.0
                action_idx = int.from_bytes(t.action, byteorder="little")
                policy_targets[0, action_idx] = 1.0
        else:
            policy_targets[0] = 0.0
            action_idx = int.from_bytes(t.action, byteorder="little")
            policy_targets[0, action_idx] = 1.0

        value_targets[0] = (
            t.game_outcome if t.game_outcome is not None else t.mcts_value
        )

        for i in range(1, batch_size):
            t = transitions[i]
            observations[i] = np.frombuffer(t.observation, dtype=np.float32)

            if t.policy_probs is not None and len(t.policy_probs) > 0:
                policy = np.frombuffer(t.policy_probs, dtype=np.float32)
                if len(policy) == num_actions:
                    policy_targets[i] = policy
                else:
                    policy_targets[i] = 0.0
                    action_idx = int.from_bytes(t.action, byteorder="little")
                    policy_targets[i, action_idx] = 1.0
            else:
                policy_targets[i] = 0.0
                action_idx = int.from_bytes(t.action, byteorder="little")
                policy_targets[i, action_idx] = 1.0

            value_targets[i] = (
                t.game_outcome if t.game_outcome is not None else t.mcts_value
            )

        return observations, policy_targets, value_targets

    def clear_transitions(self) -> int:
        """Delete all transitions from the buffer."""
        cursor = self._conn.execute("DELETE FROM transitions")
        self._conn.commit()
        return cursor.rowcount

    def vacuum(self) -> None:
        """Run VACUUM to reclaim disk space."""
        self._conn.execute("VACUUM")

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
