"""Statistics tracking and persistence for training.

This module provides dataclasses for tracking training and evaluation
statistics, along with functions for loading and saving stats with
atomic writes to prevent corruption during concurrent reads.
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default history bounds
DEFAULT_MAX_HISTORY = 2000  # Store up to 2000 training steps for full history visualization
DEFAULT_MAX_EVAL_HISTORY = 50


@dataclass
class EvalStats:
    """Evaluation statistics from a single evaluation run."""

    step: int = 0
    win_rate: float = 0.0
    draw_rate: float = 0.0
    loss_rate: float = 0.0
    games_played: int = 0
    avg_game_length: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "win_rate": self.win_rate,
            "draw_rate": self.draw_rate,
            "loss_rate": self.loss_rate,
            "games_played": self.games_played,
            "avg_game_length": self.avg_game_length,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalStats":
        """Create EvalStats from a dictionary."""
        return cls(
            step=data.get("step", 0),
            win_rate=data.get("win_rate", 0.0),
            draw_rate=data.get("draw_rate", 0.0),
            loss_rate=data.get("loss_rate", 0.0),
            games_played=data.get("games_played", 0),
            avg_game_length=data.get("avg_game_length", 0.0),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class TrainerStats:
    """Training statistics for web visualization."""

    step: int = 0
    total_steps: int = 0
    total_loss: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    learning_rate: float = 0.0
    samples_seen: int = 0
    replay_buffer_size: int = 0
    last_checkpoint: str = ""
    timestamp: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)
    _max_history: int = DEFAULT_MAX_HISTORY

    # Environment being trained
    env_id: str = ""

    # Evaluation metrics
    last_eval: EvalStats | None = None
    eval_history: list[dict] = field(default_factory=list)
    _max_eval_history: int = DEFAULT_MAX_EVAL_HISTORY

    def append_history(self, entry: dict) -> None:
        """Append to history, maintaining max length bound."""
        self.history.append(entry)
        if len(self.history) > self._max_history:
            # Remove oldest entries to stay within bound
            self.history = self.history[-self._max_history :]

    def append_eval(self, eval_stats: EvalStats) -> None:
        """Append evaluation result to history."""
        self.last_eval = eval_stats
        self.eval_history.append(eval_stats.to_dict())
        if len(self.eval_history) > self._max_eval_history:
            self.eval_history = self.eval_history[-self._max_eval_history :]

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "total_loss": self.total_loss,
            "value_loss": self.value_loss,
            "policy_loss": self.policy_loss,
            "learning_rate": self.learning_rate,
            "samples_seen": self.samples_seen,
            "replay_buffer_size": self.replay_buffer_size,
            "last_checkpoint": self.last_checkpoint,
            "timestamp": self.timestamp,
            "history": self.history,  # Already bounded on append
            "env_id": self.env_id,
            "last_eval": self.last_eval.to_dict() if self.last_eval else None,
            "eval_history": self.eval_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainerStats":
        """Create TrainerStats from a dictionary."""
        stats = cls(
            step=data.get("step", 0),
            total_steps=data.get("total_steps", 0),
            total_loss=data.get("total_loss", 0.0),
            value_loss=data.get("value_loss", 0.0),
            policy_loss=data.get("policy_loss", 0.0),
            learning_rate=data.get("learning_rate", 0.0),
            samples_seen=data.get("samples_seen", 0),
            replay_buffer_size=data.get("replay_buffer_size", 0),
            last_checkpoint=data.get("last_checkpoint", ""),
            timestamp=data.get("timestamp", 0.0),
            history=data.get("history", []),
            env_id=data.get("env_id", ""),
            eval_history=data.get("eval_history", []),
        )

        # Parse last_eval if present
        if data.get("last_eval"):
            stats.last_eval = EvalStats.from_dict(data["last_eval"])

        return stats


def load_stats(path: str | Path) -> TrainerStats:
    """Load stats from a JSON file.

    Args:
        path: Path to the stats JSON file.

    Returns:
        TrainerStats loaded from file, or empty TrainerStats if file doesn't exist
        or is invalid.
    """
    stats_path = Path(path)
    if not stats_path.exists():
        return TrainerStats()

    try:
        with open(stats_path) as f:
            data = json.load(f)

        stats = TrainerStats.from_dict(data)

        logger.info(
            f"Loaded existing stats: {len(stats.eval_history)} eval records, "
            f"{len(stats.history)} training records"
        )
        return stats

    except Exception as e:
        logger.warning(f"Failed to load existing stats: {e}")
        return TrainerStats()


def write_stats(stats: TrainerStats, path: str | Path) -> None:
    """Write stats to a JSON file using atomic write-then-rename.

    This ensures readers never see a partially written file, even if
    the write is interrupted or a reader accesses the file concurrently.

    Args:
        stats: TrainerStats to write.
        path: Path to write the stats JSON file.

    Raises:
        Exception: If the write fails (temp file is cleaned up on error).
    """
    stats_path = Path(path)

    # Ensure parent directory exists
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file then rename (atomic on most filesystems)
    temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=stats_path.parent)
    try:
        with os.fdopen(temp_fd, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        os.replace(temp_path, stats_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
