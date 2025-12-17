#!/usr/bin/env python3
"""Synchronized AlphaZero Training Loop.

This script implements the classic AlphaZero iteration pattern:
1. Clear replay buffer (start fresh with current model)
2. Run actor for N episodes (self-play data generation)
3. Train for M steps on the generated data
4. Export new model, repeat

This ensures each training iteration only uses data from the current model,
avoiding the noise that comes from mixing data from many model generations.

Usage:
    python scripts/train_loop.py --iterations 100 --episodes 500 --steps 1000

    # With custom paths
    python scripts/train_loop.py \
        --iterations 50 \
        --episodes 200 \
        --steps 500 \
        --env-id connect4 \
        --data-dir ./data

    # Resume from a specific iteration
    python scripts/train_loop.py --iterations 100 --start-iteration 25
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add trainer to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "trainer" / "src"))

from trainer.replay import ReplayBuffer, create_empty_db


@dataclass
class IterationStats:
    """Statistics for a single training iteration."""

    iteration: int
    episodes_generated: int
    transitions_generated: int
    training_steps: int
    actor_time_seconds: float
    trainer_time_seconds: float
    total_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TrainLoopConfig:
    """Configuration for the synchronized training loop."""

    # Iteration settings
    iterations: int = 100
    start_iteration: int = 1
    episodes_per_iteration: int = 500
    steps_per_iteration: int = 1000

    # Environment
    env_id: str = "tictactoe"

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    actor_binary: Path | None = None  # Auto-detect if None

    # Actor settings
    actor_log_interval: int = 50

    # Trainer settings
    batch_size: int = 64
    learning_rate: float = 1e-3
    checkpoint_interval: int = 100
    eval_interval: int = 0  # Disabled by default for speed
    eval_games: int = 50
    device: str = "cpu"

    # Logging
    log_level: str = "INFO"

    @property
    def replay_db_path(self) -> Path:
        return self.data_dir / "replay.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def stats_path(self) -> Path:
        return self.data_dir / "stats.json"

    @property
    def loop_stats_path(self) -> Path:
        return self.data_dir / "loop_stats.json"


class TrainLoop:
    """Synchronized AlphaZero training loop orchestrator."""

    def __init__(self, config: TrainLoopConfig):
        self.config = config
        self.logger = logging.getLogger("train_loop")
        self.iteration_history: list[IterationStats] = []
        self._shutdown_requested = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, requesting shutdown...")
        self._shutdown_requested = True

    def _find_actor_binary(self) -> Path:
        """Find the actor binary, preferring release build."""
        if self.config.actor_binary:
            return self.config.actor_binary

        # Look for actor binary in standard locations
        base = Path(__file__).parent.parent
        candidates = [
            base / "actor" / "target" / "release" / "actor",
            base / "actor" / "target" / "debug" / "actor",
            base / "target" / "release" / "actor",
            base / "target" / "debug" / "actor",
        ]

        for candidate in candidates:
            if candidate.exists():
                self.logger.info(f"Found actor binary: {candidate}")
                return candidate

        raise FileNotFoundError(
            f"Actor binary not found. Searched: {candidates}. "
            "Run 'cd actor && cargo build --release' first."
        )

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

    def _clear_replay_buffer(self) -> int:
        """Clear all transitions from the replay buffer.

        Returns the number of deleted transitions.
        """
        if not self.config.replay_db_path.exists():
            self.logger.info("Replay database doesn't exist, creating empty one")
            create_empty_db(self.config.replay_db_path)
            return 0

        with ReplayBuffer(self.config.replay_db_path) as replay:
            deleted = replay.clear_transitions()
            self.logger.info(f"Cleared {deleted} transitions from replay buffer")
            return deleted

    def _get_transition_count(self) -> int:
        """Get the current number of transitions in the replay buffer."""
        if not self.config.replay_db_path.exists():
            return 0
        with ReplayBuffer(self.config.replay_db_path) as replay:
            return replay.count(env_id=self.config.env_id)

    def _run_actor(self, num_episodes: int) -> tuple[bool, float]:
        """Run the actor for a specified number of episodes.

        Returns (success, elapsed_seconds).
        """
        actor_binary = self._find_actor_binary()

        cmd = [
            str(actor_binary),
            "--env-id",
            self.config.env_id,
            "--max-episodes",
            str(num_episodes),
            "--replay-db-path",
            str(self.config.replay_db_path),
            "--data-dir",
            str(self.config.data_dir),
            "--log-interval",
            str(self.config.actor_log_interval),
            "--log-level",
            "info",
        ]

        self.logger.info(f"Starting actor: {' '.join(cmd)}")
        start_time = time.time()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output in real-time
            while True:
                if self._shutdown_requested:
                    self.logger.warning("Shutdown requested, terminating actor...")
                    process.terminate()
                    process.wait(timeout=5)
                    return False, time.time() - start_time

                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    # Forward actor output with prefix
                    print(f"[actor] {line.rstrip()}")

            return_code = process.wait()
            elapsed = time.time() - start_time

            if return_code != 0:
                self.logger.error(f"Actor exited with code {return_code}")
                return False, elapsed

            self.logger.info(f"Actor completed in {elapsed:.1f}s")
            return True, elapsed

        except Exception as e:
            self.logger.error(f"Actor failed: {e}")
            return False, time.time() - start_time

    def _run_trainer(self, num_steps: int, start_step: int) -> tuple[bool, float]:
        """Run the trainer for a specified number of steps.

        Returns (success, elapsed_seconds).
        """
        # Run trainer as a subprocess to keep things clean
        trainer_dir = Path(__file__).parent.parent / "trainer"

        cmd = [
            sys.executable,
            "-m",
            "trainer",
            "--db",
            str(self.config.replay_db_path),
            "--model-dir",
            str(self.config.models_dir),
            "--stats",
            str(self.config.stats_path),
            "--env-id",
            self.config.env_id,
            "--steps",
            str(num_steps),
            "--start-step",
            str(start_step),
            "--batch-size",
            str(self.config.batch_size),
            "--lr",
            str(self.config.learning_rate),
            "--checkpoint-interval",
            str(self.config.checkpoint_interval),
            "--device",
            self.config.device,
            "--max-wait",
            "60",  # Short timeout since we know data exists
            "--log-level",
            "INFO",
        ]

        # Add evaluation settings
        if self.config.eval_interval > 0:
            cmd.extend(["--eval-interval", str(self.config.eval_interval)])
            cmd.extend(["--eval-games", str(self.config.eval_games)])
        else:
            cmd.extend(["--eval-interval", "0"])

        self.logger.info(f"Starting trainer: {' '.join(cmd)}")
        start_time = time.time()

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(trainer_dir / "src")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(trainer_dir),
                env=env,
            )

            # Stream output in real-time
            while True:
                if self._shutdown_requested:
                    self.logger.warning("Shutdown requested, terminating trainer...")
                    process.terminate()
                    process.wait(timeout=5)
                    return False, time.time() - start_time

                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    # Forward trainer output with prefix
                    print(f"[trainer] {line.rstrip()}")

            return_code = process.wait()
            elapsed = time.time() - start_time

            if return_code != 0:
                self.logger.error(f"Trainer exited with code {return_code}")
                return False, elapsed

            self.logger.info(f"Trainer completed in {elapsed:.1f}s")
            return True, elapsed

        except Exception as e:
            self.logger.error(f"Trainer failed: {e}")
            return False, time.time() - start_time

    def _save_loop_stats(self) -> None:
        """Save iteration history to a JSON file."""
        stats = {
            "config": {
                "env_id": self.config.env_id,
                "episodes_per_iteration": self.config.episodes_per_iteration,
                "steps_per_iteration": self.config.steps_per_iteration,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
            "iterations": [
                {
                    "iteration": s.iteration,
                    "episodes": s.episodes_generated,
                    "transitions": s.transitions_generated,
                    "steps": s.training_steps,
                    "actor_time": s.actor_time_seconds,
                    "trainer_time": s.trainer_time_seconds,
                    "total_time": s.total_time_seconds,
                    "timestamp": s.timestamp,
                }
                for s in self.iteration_history
            ],
        }

        with open(self.config.loop_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    def run_iteration(self, iteration: int) -> IterationStats | None:
        """Run a single training iteration.

        Returns iteration stats, or None if shutdown was requested.
        """
        iter_start = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ITERATION {iteration}")
        self.logger.info(f"{'='*60}")

        # Step 1: Clear replay buffer
        self.logger.info("Step 1: Clearing replay buffer...")
        self._clear_replay_buffer()

        if self._shutdown_requested:
            return None

        # Step 2: Run actor for N episodes
        self.logger.info(
            f"Step 2: Running actor for {self.config.episodes_per_iteration} episodes..."
        )
        actor_success, actor_time = self._run_actor(self.config.episodes_per_iteration)

        if not actor_success:
            if self._shutdown_requested:
                return None
            self.logger.error("Actor failed, aborting iteration")
            return None

        transitions_count = self._get_transition_count()
        self.logger.info(f"Actor generated {transitions_count} transitions")

        if self._shutdown_requested:
            return None

        # Step 3: Train for M steps
        # Calculate start step for checkpoint naming continuity
        start_step = (iteration - 1) * self.config.steps_per_iteration
        self.logger.info(
            f"Step 3: Training for {self.config.steps_per_iteration} steps..."
        )
        trainer_success, trainer_time = self._run_trainer(
            self.config.steps_per_iteration, start_step
        )

        if not trainer_success:
            if self._shutdown_requested:
                return None
            self.logger.error("Trainer failed, aborting iteration")
            return None

        total_time = time.time() - iter_start

        stats = IterationStats(
            iteration=iteration,
            episodes_generated=self.config.episodes_per_iteration,
            transitions_generated=transitions_count,
            training_steps=self.config.steps_per_iteration,
            actor_time_seconds=actor_time,
            trainer_time_seconds=trainer_time,
            total_time_seconds=total_time,
        )

        self.logger.info(f"\nIteration {iteration} complete:")
        self.logger.info(f"  Episodes: {stats.episodes_generated}")
        self.logger.info(f"  Transitions: {stats.transitions_generated}")
        self.logger.info(f"  Actor time: {stats.actor_time_seconds:.1f}s")
        self.logger.info(f"  Trainer time: {stats.trainer_time_seconds:.1f}s")
        self.logger.info(f"  Total time: {stats.total_time_seconds:.1f}s")

        return stats

    def run(self) -> None:
        """Run the full training loop."""
        self._ensure_directories()

        self.logger.info(f"Starting synchronized AlphaZero training")
        self.logger.info(f"  Environment: {self.config.env_id}")
        self.logger.info(f"  Iterations: {self.config.iterations}")
        self.logger.info(f"  Episodes per iteration: {self.config.episodes_per_iteration}")
        self.logger.info(f"  Steps per iteration: {self.config.steps_per_iteration}")
        self.logger.info(f"  Data directory: {self.config.data_dir}")

        loop_start = time.time()

        for iteration in range(
            self.config.start_iteration,
            self.config.start_iteration + self.config.iterations,
        ):
            if self._shutdown_requested:
                self.logger.warning("Shutdown requested, stopping loop")
                break

            stats = self.run_iteration(iteration)
            if stats:
                self.iteration_history.append(stats)
                self._save_loop_stats()

        total_time = time.time() - loop_start
        completed = len(self.iteration_history)

        self.logger.info(f"\n{'='*60}")
        self.logger.info("TRAINING COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Completed iterations: {completed}")
        self.logger.info(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")

        if self.iteration_history:
            total_episodes = sum(s.episodes_generated for s in self.iteration_history)
            total_transitions = sum(
                s.transitions_generated for s in self.iteration_history
            )
            total_steps = sum(s.training_steps for s in self.iteration_history)
            self.logger.info(f"Total episodes: {total_episodes}")
            self.logger.info(f"Total transitions: {total_transitions}")
            self.logger.info(f"Total training steps: {total_steps}")


def parse_args() -> TrainLoopConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Synchronized AlphaZero Training Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python scripts/train_loop.py --iterations 50 --episodes 200 --steps 500

    # Connect4 with GPU
    python scripts/train_loop.py --env-id connect4 --device cuda --iterations 100

    # Resume from iteration 25
    python scripts/train_loop.py --iterations 100 --start-iteration 25
        """,
    )

    # Iteration settings
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1,
        help="Starting iteration number (default: 1)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Episodes per iteration (default: 500)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps per iteration (default: 1000)",
    )

    # Environment
    parser.add_argument(
        "--env-id",
        type=str,
        default="tictactoe",
        help="Environment ID: tictactoe, connect4 (default: tictactoe)",
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)",
    )
    parser.add_argument(
        "--actor-binary",
        type=str,
        default=None,
        help="Path to actor binary (auto-detect if not specified)",
    )

    # Actor settings
    parser.add_argument(
        "--actor-log-interval",
        type=int,
        default=50,
        help="Actor log interval in episodes (default: 50)",
    )

    # Trainer settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Steps between checkpoints (default: 100)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Steps between evaluations, 0 to disable (default: 0)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="Games per evaluation (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Training device (default: cpu)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    return TrainLoopConfig(
        iterations=args.iterations,
        start_iteration=args.start_iteration,
        episodes_per_iteration=args.episodes,
        steps_per_iteration=args.steps,
        env_id=args.env_id,
        data_dir=Path(args.data_dir),
        actor_binary=Path(args.actor_binary) if args.actor_binary else None,
        actor_log_interval=args.actor_log_interval,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        device=args.device,
        log_level=args.log_level,
    )


def main() -> None:
    """Main entry point."""
    config = parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    loop = TrainLoop(config)
    loop.run()


if __name__ == "__main__":
    main()
