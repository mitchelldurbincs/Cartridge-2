"""Synchronized AlphaZero Training Loop Orchestrator.

This module implements the classic AlphaZero iteration pattern:
1. Clear replay buffer (start fresh with current model)
2. Run actor for N episodes (self-play data generation)
3. Train for M steps on the generated data
4. Run evaluation against random baseline
5. Export new model, repeat

This ensures each training iteration only uses data from the current model,
avoiding the noise that comes from mixing data from many model generations.

Usage:
    # As a module
    python -m trainer.orchestrator --iterations 100 --episodes 500 --steps 1000

    # Via console entry point
    trainer-loop --iterations 100 --episodes 500 --steps 1000

Environment Variables:
    ACTOR_BINARY: Path to actor binary (default: auto-detect)
    DATA_DIR: Base data directory (default: ./data)
    ALPHAZERO_*: Override any CLI argument via ALPHAZERO_<ARG_NAME>
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

from .evaluator import OnnxPolicy, RandomPolicy, evaluate as run_eval, create_game_state
from .game_config import get_config
from .replay import ReplayBuffer, create_empty_db, GameMetadata

logger = logging.getLogger(__name__)


@dataclass
class IterationStats:
    """Statistics for a single training iteration."""

    iteration: int
    episodes_generated: int
    transitions_generated: int
    training_steps: int
    actor_time_seconds: float
    trainer_time_seconds: float
    eval_time_seconds: float
    total_time_seconds: float
    eval_win_rate: float | None = None
    eval_draw_rate: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LoopConfig:
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
    actor_binary: Path | None = None  # Auto-detect or use ACTOR_BINARY env var

    # Actor settings
    actor_log_interval: int = 50

    # Trainer settings
    batch_size: int = 64
    learning_rate: float = 1e-3
    checkpoint_interval: int = 100
    device: str = "cpu"

    # Evaluation settings - enabled by default
    eval_interval: int = 1  # Run evaluation every N iterations (0 to disable)
    eval_games: int = 50  # Games per evaluation

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

    @property
    def eval_stats_path(self) -> Path:
        return self.data_dir / "eval_stats.json"


def get_env_with_prefix(key: str, default: str | None = None) -> str | None:
    """Get environment variable with ALPHAZERO_ prefix."""
    return os.environ.get(f"ALPHAZERO_{key}", default)


def get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    val = get_env_with_prefix(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            logger.warning(f"Invalid integer for ALPHAZERO_{key}: {val}, using default {default}")
    return default


def get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    val = get_env_with_prefix(key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            logger.warning(f"Invalid float for ALPHAZERO_{key}: {val}, using default {default}")
    return default


class Orchestrator:
    """Synchronized AlphaZero training loop orchestrator."""

    def __init__(self, config: LoopConfig):
        self.config = config
        self.iteration_history: list[IterationStats] = []
        self.eval_history: list[dict] = []
        self._shutdown_requested = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, requesting shutdown...")
        self._shutdown_requested = True

    def _find_actor_binary(self) -> Path:
        """Find the actor binary, preferring release build.

        Checks in order:
        1. config.actor_binary (if explicitly set)
        2. ACTOR_BINARY environment variable
        3. Standard cargo build locations
        """
        # Check explicit config
        if self.config.actor_binary:
            if self.config.actor_binary.exists():
                return self.config.actor_binary
            raise FileNotFoundError(f"Configured actor binary not found: {self.config.actor_binary}")

        # Check environment variable
        env_path = os.environ.get("ACTOR_BINARY")
        if env_path:
            path = Path(env_path)
            if path.exists():
                logger.info(f"Using actor binary from ACTOR_BINARY: {path}")
                return path
            raise FileNotFoundError(f"ACTOR_BINARY not found: {env_path}")

        # Auto-detect from standard locations
        # Use the module's location to find the project root
        module_dir = Path(__file__).parent
        # trainer/src/trainer -> project root
        project_root = module_dir.parent.parent.parent

        candidates = [
            # Docker location
            Path("/app/actor"),
            # Workspace-level target
            project_root / "target" / "release" / "actor",
            project_root / "target" / "debug" / "actor",
            # Actor-specific target
            project_root / "actor" / "target" / "release" / "actor",
            project_root / "actor" / "target" / "debug" / "actor",
        ]

        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Found actor binary: {candidate}")
                return candidate

        raise FileNotFoundError(
            f"Actor binary not found. Searched: {[str(c) for c in candidates]}. "
            "Set ACTOR_BINARY environment variable or run 'cd actor && cargo build --release'."
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
            logger.info("Replay database doesn't exist, creating empty one")
            create_empty_db(self.config.replay_db_path)
            return 0

        with ReplayBuffer(self.config.replay_db_path) as replay:
            deleted = replay.clear_transitions()
            logger.info(f"Cleared {deleted} transitions from replay buffer")
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
            "--env-id", self.config.env_id,
            "--max-episodes", str(num_episodes),
            "--replay-db-path", str(self.config.replay_db_path),
            "--data-dir", str(self.config.data_dir),
            "--log-interval", str(self.config.actor_log_interval),
            "--log-level", "info",
        ]

        logger.info(f"Starting actor: {' '.join(cmd)}")
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
                    logger.warning("Shutdown requested, terminating actor...")
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
                logger.error(f"Actor exited with code {return_code}")
                return False, elapsed

            logger.info(f"Actor completed in {elapsed:.1f}s")
            return True, elapsed

        except Exception as e:
            logger.error(f"Actor failed: {e}")
            return False, time.time() - start_time

    def _run_trainer(self, num_steps: int, start_step: int) -> tuple[bool, float]:
        """Run the trainer for a specified number of steps.

        Returns (success, elapsed_seconds).
        """
        # Import locally to avoid circular imports in some edge cases
        from .trainer import Trainer, TrainerConfig

        config = TrainerConfig(
            db_path=str(self.config.replay_db_path),
            model_dir=str(self.config.models_dir),
            stats_path=str(self.config.stats_path),
            env_id=self.config.env_id,
            steps=num_steps,
            start_step=start_step,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            checkpoint_interval=self.config.checkpoint_interval,
            device=self.config.device,
            max_wait=60.0,  # Short timeout since we know data exists
            eval_interval=0,  # Disable trainer's built-in eval, we do it ourselves
        )

        logger.info(f"Starting trainer for {num_steps} steps (start_step={start_step})")
        start_time = time.time()

        try:
            trainer = Trainer(config)
            stats = trainer.train()
            elapsed = time.time() - start_time

            logger.info(f"Trainer completed in {elapsed:.1f}s, final loss: {stats.total_loss:.4f}")
            return True, elapsed

        except Exception as e:
            logger.error(f"Trainer failed: {e}")
            return False, time.time() - start_time

    def _run_evaluation(self, iteration: int) -> tuple[float | None, float | None, float]:
        """Run evaluation against random baseline.

        Returns (win_rate, draw_rate, elapsed_seconds).
        """
        model_path = self.config.models_dir / "latest.onnx"

        if not model_path.exists():
            logger.warning(f"Model not found for evaluation: {model_path}")
            return None, None, 0.0

        start_time = time.time()
        logger.info(f"Running evaluation ({self.config.eval_games} games)...")

        try:
            # Load game config
            config = get_config(self.config.env_id)

            # Create policies
            model_policy = OnnxPolicy(str(model_path), temperature=0.0)
            random_policy = RandomPolicy()

            # Run evaluation
            results = run_eval(
                player1=model_policy,
                player2=random_policy,
                env_id=self.config.env_id,
                config=config,
                num_games=self.config.eval_games,
                verbose=False,
            )

            elapsed = time.time() - start_time

            logger.info(
                f"Evaluation: {results.player1_name} vs {results.player2_name} - "
                f"Win: {results.player1_win_rate:.1%}, Draw: {results.draw_rate:.1%}, "
                f"Loss: {results.player2_win_rate:.1%}"
            )

            # Save to eval history
            eval_record = {
                "iteration": iteration,
                "win_rate": results.player1_win_rate,
                "draw_rate": results.draw_rate,
                "loss_rate": results.player2_win_rate,
                "games": results.games_played,
                "timestamp": datetime.now().isoformat(),
            }
            self.eval_history.append(eval_record)
            self._save_eval_stats()

            return results.player1_win_rate, results.draw_rate, elapsed

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None, None, time.time() - start_time

    def _save_loop_stats(self) -> None:
        """Save iteration history to a JSON file."""
        stats = {
            "config": {
                "env_id": self.config.env_id,
                "episodes_per_iteration": self.config.episodes_per_iteration,
                "steps_per_iteration": self.config.steps_per_iteration,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "eval_interval": self.config.eval_interval,
                "eval_games": self.config.eval_games,
            },
            "iterations": [
                {
                    "iteration": s.iteration,
                    "episodes": s.episodes_generated,
                    "transitions": s.transitions_generated,
                    "steps": s.training_steps,
                    "actor_time": s.actor_time_seconds,
                    "trainer_time": s.trainer_time_seconds,
                    "eval_time": s.eval_time_seconds,
                    "total_time": s.total_time_seconds,
                    "eval_win_rate": s.eval_win_rate,
                    "eval_draw_rate": s.eval_draw_rate,
                    "timestamp": s.timestamp,
                }
                for s in self.iteration_history
            ],
        }

        with open(self.config.loop_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    def _save_eval_stats(self) -> None:
        """Save evaluation history to a separate file for web UI."""
        with open(self.config.eval_stats_path, "w") as f:
            json.dump({"evaluations": self.eval_history}, f, indent=2)

    def run_iteration(self, iteration: int) -> IterationStats | None:
        """Run a single training iteration.

        Returns iteration stats, or None if shutdown was requested.
        """
        iter_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")

        # Step 1: Clear replay buffer
        logger.info("Step 1: Clearing replay buffer...")
        self._clear_replay_buffer()

        if self._shutdown_requested:
            return None

        # Step 2: Run actor for N episodes
        logger.info(f"Step 2: Running actor for {self.config.episodes_per_iteration} episodes...")
        actor_success, actor_time = self._run_actor(self.config.episodes_per_iteration)

        if not actor_success:
            if self._shutdown_requested:
                return None
            logger.error("Actor failed, aborting iteration")
            return None

        transitions_count = self._get_transition_count()
        logger.info(f"Actor generated {transitions_count} transitions")

        if self._shutdown_requested:
            return None

        # Step 3: Train for M steps
        start_step = (iteration - 1) * self.config.steps_per_iteration
        logger.info(f"Step 3: Training for {self.config.steps_per_iteration} steps...")
        trainer_success, trainer_time = self._run_trainer(
            self.config.steps_per_iteration, start_step
        )

        if not trainer_success:
            if self._shutdown_requested:
                return None
            logger.error("Trainer failed, aborting iteration")
            return None

        if self._shutdown_requested:
            return None

        # Step 4: Run evaluation (if enabled)
        eval_time = 0.0
        win_rate = None
        draw_rate = None

        should_eval = (
            self.config.eval_interval > 0
            and iteration % self.config.eval_interval == 0
        )

        if should_eval:
            logger.info(f"Step 4: Running evaluation...")
            win_rate, draw_rate, eval_time = self._run_evaluation(iteration)
        else:
            if self.config.eval_interval > 0:
                logger.info(f"Step 4: Skipping evaluation (next at iteration {iteration + (self.config.eval_interval - iteration % self.config.eval_interval)})")
            else:
                logger.info("Step 4: Evaluation disabled (ALPHAZERO_EVAL_INTERVAL=0)")

        total_time = time.time() - iter_start

        stats = IterationStats(
            iteration=iteration,
            episodes_generated=self.config.episodes_per_iteration,
            transitions_generated=transitions_count,
            training_steps=self.config.steps_per_iteration,
            actor_time_seconds=actor_time,
            trainer_time_seconds=trainer_time,
            eval_time_seconds=eval_time,
            total_time_seconds=total_time,
            eval_win_rate=win_rate,
            eval_draw_rate=draw_rate,
        )

        logger.info(f"\nIteration {iteration} complete:")
        logger.info(f"  Episodes: {stats.episodes_generated}")
        logger.info(f"  Transitions: {stats.transitions_generated}")
        logger.info(f"  Actor time: {stats.actor_time_seconds:.1f}s")
        logger.info(f"  Trainer time: {stats.trainer_time_seconds:.1f}s")
        if win_rate is not None:
            logger.info(f"  Eval win rate: {win_rate:.1%}")
        logger.info(f"  Total time: {stats.total_time_seconds:.1f}s")

        return stats

    def run(self) -> None:
        """Run the full training loop."""
        self._ensure_directories()

        # Log configuration with evaluation status prominently
        logger.info("=" * 60)
        logger.info("Synchronized AlphaZero Training")
        logger.info("=" * 60)
        logger.info(f"Environment: {self.config.env_id}")
        logger.info(f"Iterations: {self.config.iterations}")
        logger.info(f"Episodes per iteration: {self.config.episodes_per_iteration}")
        logger.info(f"Steps per iteration: {self.config.steps_per_iteration}")
        logger.info(f"Data directory: {self.config.data_dir}")

        # Prominently show evaluation settings
        if self.config.eval_interval > 0:
            logger.info(f"Evaluation: ENABLED (every {self.config.eval_interval} iteration(s), {self.config.eval_games} games)")
        else:
            logger.info("Evaluation: DISABLED (set ALPHAZERO_EVAL_INTERVAL > 0 to enable)")

        logger.info("=" * 60)

        loop_start = time.time()

        for iteration in range(
            self.config.start_iteration,
            self.config.start_iteration + self.config.iterations,
        ):
            if self._shutdown_requested:
                logger.warning("Shutdown requested, stopping loop")
                break

            stats = self.run_iteration(iteration)
            if stats:
                self.iteration_history.append(stats)
                self._save_loop_stats()

        total_time = time.time() - loop_start
        completed = len(self.iteration_history)

        logger.info(f"\n{'='*60}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Completed iterations: {completed}")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")

        if self.iteration_history:
            total_episodes = sum(s.episodes_generated for s in self.iteration_history)
            total_transitions = sum(s.transitions_generated for s in self.iteration_history)
            total_steps = sum(s.training_steps for s in self.iteration_history)
            logger.info(f"Total episodes: {total_episodes}")
            logger.info(f"Total transitions: {total_transitions}")
            logger.info(f"Total training steps: {total_steps}")

            # Report final evaluation if available
            final_with_eval = [s for s in self.iteration_history if s.eval_win_rate is not None]
            if final_with_eval:
                final = final_with_eval[-1]
                logger.info(f"Final evaluation: {final.eval_win_rate:.1%} win rate vs random")


def parse_args() -> LoopConfig:
    """Parse command line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(
        description="Synchronized AlphaZero Training Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
    ALPHAZERO_ENV_ID          Game environment (default: tictactoe)
    ALPHAZERO_ITERATIONS      Number of training iterations
    ALPHAZERO_EPISODES        Episodes per iteration
    ALPHAZERO_STEPS           Training steps per iteration
    ALPHAZERO_BATCH_SIZE      Training batch size
    ALPHAZERO_LR              Learning rate
    ALPHAZERO_DEVICE          Training device (cpu/cuda/mps)
    ALPHAZERO_EVAL_INTERVAL   Evaluate every N iterations (0 to disable)
    ALPHAZERO_EVAL_GAMES      Games per evaluation
    ACTOR_BINARY              Path to actor binary
    DATA_DIR                  Base data directory

Examples:
    # Basic training with evaluation every iteration
    python -m trainer.orchestrator --iterations 50 --episodes 200 --steps 500

    # Connect4 with GPU
    python -m trainer.orchestrator --env-id connect4 --device cuda --iterations 100

    # Disable evaluation for faster training
    python -m trainer.orchestrator --eval-interval 0

    # Via Docker
    ALPHAZERO_EVAL_INTERVAL=1 docker compose up alphazero
        """,
    )

    # Get defaults from environment variables
    default_env_id = get_env_with_prefix("ENV_ID", "tictactoe")
    default_data_dir = os.environ.get("DATA_DIR", "./data")

    # Iteration settings
    parser.add_argument(
        "--iterations", type=int,
        default=get_env_int("ITERATIONS", 100),
        help="Number of training iterations (env: ALPHAZERO_ITERATIONS)",
    )
    parser.add_argument(
        "--start-iteration", type=int,
        default=get_env_int("START_ITERATION", 1),
        help="Starting iteration number (env: ALPHAZERO_START_ITERATION)",
    )
    parser.add_argument(
        "--episodes", type=int,
        default=get_env_int("EPISODES", 500),
        help="Episodes per iteration (env: ALPHAZERO_EPISODES)",
    )
    parser.add_argument(
        "--steps", type=int,
        default=get_env_int("STEPS", 1000),
        help="Training steps per iteration (env: ALPHAZERO_STEPS)",
    )

    # Environment
    parser.add_argument(
        "--env-id", type=str,
        default=default_env_id,
        help="Environment ID: tictactoe, connect4 (env: ALPHAZERO_ENV_ID)",
    )

    # Paths
    parser.add_argument(
        "--data-dir", type=str,
        default=default_data_dir,
        help="Data directory (env: DATA_DIR)",
    )
    parser.add_argument(
        "--actor-binary", type=str,
        default=os.environ.get("ACTOR_BINARY"),
        help="Path to actor binary (env: ACTOR_BINARY)",
    )

    # Actor settings
    parser.add_argument(
        "--actor-log-interval", type=int,
        default=get_env_int("ACTOR_LOG_INTERVAL", 50),
        help="Actor log interval in episodes",
    )

    # Trainer settings
    parser.add_argument(
        "--batch-size", type=int,
        default=get_env_int("BATCH_SIZE", 64),
        help="Training batch size (env: ALPHAZERO_BATCH_SIZE)",
    )
    parser.add_argument(
        "--lr", type=float,
        default=get_env_float("LR", 1e-3),
        help="Learning rate (env: ALPHAZERO_LR)",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int,
        default=get_env_int("CHECKPOINT_INTERVAL", 100),
        help="Steps between checkpoints (env: ALPHAZERO_CHECKPOINT_INTERVAL)",
    )
    parser.add_argument(
        "--device", type=str,
        default=get_env_with_prefix("DEVICE", "cpu"),
        choices=["cpu", "cuda", "mps"],
        help="Training device (env: ALPHAZERO_DEVICE)",
    )

    # Evaluation settings - NOTE: default is 1 (enabled)
    parser.add_argument(
        "--eval-interval", type=int,
        default=get_env_int("EVAL_INTERVAL", 1),
        help="Evaluate every N iterations, 0 to disable (env: ALPHAZERO_EVAL_INTERVAL)",
    )
    parser.add_argument(
        "--eval-games", type=int,
        default=get_env_int("EVAL_GAMES", 50),
        help="Games per evaluation (env: ALPHAZERO_EVAL_GAMES)",
    )

    # Logging
    parser.add_argument(
        "--log-level", type=str,
        default=get_env_with_prefix("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    args = parser.parse_args()

    return LoopConfig(
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


def main() -> int:
    """Main entry point for the orchestrator."""
    config = parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        orchestrator = Orchestrator(config)
        orchestrator.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
