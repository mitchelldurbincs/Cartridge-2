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
DEFAULT_MAX_HISTORY = 10000  # Max training history entries
DEFAULT_MAX_EVAL_HISTORY = 50

# Tiered history retention thresholds
# Recent data is kept at full resolution, older data is downsampled
RECENT_STEPS_THRESHOLD = 1000  # Keep all entries within last 1000 steps
MEDIUM_STEPS_THRESHOLD = 10000  # Downsample to every 100 steps for 1000-10000 range
RECENT_RESOLUTION = 1  # Keep every entry in recent range
MEDIUM_RESOLUTION = 100  # Keep every 100th step in medium range
OLD_RESOLUTION = 500  # Keep every 500th step for older data


def _downsample_history(history: list[dict], current_step: int) -> list[dict]:
    """Downsample history using tiered retention strategy.

    Args:
        history: List of history entries, each with a "step" key.
        current_step: The current training step (used to determine age).

    Returns:
        Downsampled history list preserving recent data at full resolution
        and older data at reduced resolution.
    """
    if not history:
        return history

    result = []
    for entry in history:
        step = entry.get("step", 0)
        age = current_step - step

        if age <= RECENT_STEPS_THRESHOLD:
            # Recent: keep all entries
            result.append(entry)
        elif age <= MEDIUM_STEPS_THRESHOLD:
            # Medium age: keep entries at MEDIUM_RESOLUTION intervals
            if step % MEDIUM_RESOLUTION == 0:
                result.append(entry)
        else:
            # Old: keep entries at OLD_RESOLUTION intervals
            if step % OLD_RESOLUTION == 0:
                result.append(entry)

    return result


@dataclass
class EvalStats:
    """Evaluation statistics from a single evaluation run.

    Supports both model-vs-random and model-vs-best (gatekeeper) evaluations.
    """

    step: int = 0
    win_rate: float = 0.0
    draw_rate: float = 0.0
    loss_rate: float = 0.0
    games_played: int = 0
    avg_game_length: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # Model-vs-model evaluation fields
    opponent: str = "random"  # "random" or "best"
    opponent_iteration: int | None = None  # Which iteration the best model came from
    became_new_best: bool = False  # Whether current model replaced best
    current_iteration: int = 0  # Current model's iteration

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "win_rate": self.win_rate,
            "draw_rate": self.draw_rate,
            "loss_rate": self.loss_rate,
            "games_played": self.games_played,
            "avg_game_length": self.avg_game_length,
            "timestamp": self.timestamp,
            "opponent": self.opponent,
            "opponent_iteration": self.opponent_iteration,
            "became_new_best": self.became_new_best,
            "current_iteration": self.current_iteration,
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
            opponent=data.get("opponent", "random"),
            opponent_iteration=data.get("opponent_iteration"),
            became_new_best=data.get("became_new_best", False),
            current_iteration=data.get("current_iteration", 0),
        )


@dataclass
class BestModelInfo:
    """Information about the current best (gatekeeper) model."""

    iteration: int = 0  # Which iteration this model came from
    step: int = 0  # Training step when it became best
    win_rate_when_promoted: float = 0.0  # Win rate that earned it the spot
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "step": self.step,
            "win_rate_when_promoted": self.win_rate_when_promoted,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BestModelInfo":
        """Create BestModelInfo from a dictionary."""
        return cls(
            iteration=data.get("iteration", 0),
            step=data.get("step", 0),
            win_rate_when_promoted=data.get("win_rate_when_promoted", 0.0),
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

    # Environment being trained
    env_id: str = ""

    # History bounds
    _max_history: int = DEFAULT_MAX_HISTORY

    # Evaluation metrics
    last_eval: EvalStats | None = None
    eval_history: list[dict] = field(default_factory=list)
    _max_eval_history: int = DEFAULT_MAX_EVAL_HISTORY

    # Best model (gatekeeper) tracking
    best_model: BestModelInfo | None = None

    def append_history(self, entry: dict) -> None:
        """Append to history with tiered retention and max size bound.

        Uses a tiered downsampling strategy to keep recent data at full
        resolution while preserving coarse historical data:
        - Last 1000 steps: full resolution (every logged step)
        - 1000-10000 steps ago: every 100th step
        - 10000+ steps ago: every 500th step

        After downsampling, enforces _max_history as an absolute bound,
        keeping the most recent entries.
        """
        self.history.append(entry)
        current_step = entry.get("step", 0)
        self.history = _downsample_history(self.history, current_step)

        # Enforce absolute max history bound
        if len(self.history) > self._max_history:
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
            "best_model": self.best_model.to_dict() if self.best_model else None,
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

        # Parse best_model if present
        if data.get("best_model"):
            stats.best_model = BestModelInfo.from_dict(data["best_model"])

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
