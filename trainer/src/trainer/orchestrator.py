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

Configuration:
    Settings are loaded from config.toml (see central_config.py).
    CLI arguments override config file values.
    Legacy ALPHAZERO_* environment variables are still supported.
"""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .central_config import get_config as get_central_config
from .evaluator import OnnxPolicy, RandomPolicy
from .evaluator import evaluate as run_eval
from .game_config import get_config as get_game_config
from .replay import ReplayBuffer, create_empty_db

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
    actor_log_interval: int = 200

    # MCTS simulation ramping: start_sims + (iteration-1) * sim_ramp_rate, capped at max_sims
    mcts_start_sims: int = 200  # Simulations for first iteration
    mcts_max_sims: int = 800  # Maximum simulations (reached after ramping)
    mcts_sim_ramp_rate: int = 30  # Simulations to add per iteration

    # Temperature schedule: after temp_threshold moves, reduce temperature for exploitation
    # Set to 0 to disable (always use temp=1.0)
    # Recommended: ~60% of typical game length (tictactoe=5, connect4=20, othello=30)
    temp_threshold: int = 0

    # Trainer settings
    batch_size: int = 64
    learning_rate: float = 1e-3
    checkpoint_interval: int = 100
    device: str = "cpu"

    # Evaluation settings - enabled by default
    eval_interval: int = 1  # Run evaluation every N iterations (0 to disable)
    eval_games: int = 50  # Games per evaluation
    eval_win_threshold: float = 0.55  # Win rate needed to become new best
    eval_vs_random: bool = True  # Also evaluate against random baseline

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

    @property
    def best_model_path(self) -> Path:
        return self.models_dir / "best.onnx"

    @property
    def best_model_info_path(self) -> Path:
        return self.data_dir / "best_model.json"

    def get_num_simulations(self, iteration: int) -> int:
        """Calculate MCTS simulations for given iteration (ramping schedule).

        Starts at mcts_start_sims and increases by mcts_sim_ramp_rate per iteration,
        capped at mcts_max_sims.
        """
        sims = self.mcts_start_sims + (iteration - 1) * self.mcts_sim_ramp_rate
        return min(sims, self.mcts_max_sims)


class Orchestrator:
    """Synchronized AlphaZero training loop orchestrator."""

    def __init__(self, config: LoopConfig):
        self.config = config
        self.iteration_history: list[IterationStats] = []
        self.eval_history: list[dict] = []
        self._shutdown_requested = False

        # Best model (gatekeeper) tracking
        self.best_model_iteration: int | None = None
        self._load_best_model_info()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Auto-resume from previous state if start_iteration is default (1)
        self._auto_resume_if_needed()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, requesting shutdown...")
        self._shutdown_requested = True

    def _load_best_model_info(self) -> None:
        """Load best model info from disk if it exists."""
        if not self.config.best_model_info_path.exists():
            return

        try:
            with open(self.config.best_model_info_path) as f:
                data = json.load(f)
            self.best_model_iteration = data.get("iteration")
            logger.info(
                f"Loaded best model info: iteration {self.best_model_iteration}"
            )
        except Exception as e:
            logger.warning(f"Failed to load best model info: {e}")

    def _save_best_model_info(self, iteration: int, win_rate: float) -> None:
        """Save best model info to disk."""
        data = {
            "iteration": iteration,
            "win_rate_when_promoted": win_rate,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.config.best_model_info_path, "w") as f:
            json.dump(data, f, indent=2)

    def _promote_to_best(self, iteration: int, win_rate: float) -> None:
        """Copy current model to best.onnx and update tracking."""
        current_path = self.config.models_dir / "latest.onnx"
        if not current_path.exists():
            logger.warning("Cannot promote: latest.onnx not found")
            return

        shutil.copy(current_path, self.config.best_model_path)
        self.best_model_iteration = iteration
        self._save_best_model_info(iteration, win_rate)
        logger.info(
            f"🏆 New best model! Iteration {iteration} with {win_rate:.1%} win rate"
        )

    def _auto_resume_if_needed(self) -> None:
        """Auto-resume from loop_stats.json if start_iteration is default (1).

        This allows training to automatically continue from where it left off
        after a restart (e.g., docker-compose down/up).
        """
        # Only auto-resume if start_iteration is the default value (1)
        if self.config.start_iteration != 1:
            logger.debug(
                f"start_iteration={self.config.start_iteration} (not default), "
                "skipping auto-resume"
            )
            return

        # Check if loop_stats.json exists
        if not self.config.loop_stats_path.exists():
            logger.debug("No loop_stats.json found, starting fresh")
            return

        try:
            with open(self.config.loop_stats_path) as f:
                saved_state = json.load(f)

            iterations = saved_state.get("iterations", [])
            if not iterations:
                logger.debug("loop_stats.json has no completed iterations")
                return

            # Find the last completed iteration
            last_iteration = max(it.get("iteration", 0) for it in iterations)
            if last_iteration <= 0:
                return

            # Resume from the next iteration
            new_start = last_iteration + 1
            logger.info(
                f"Auto-resuming from iteration {new_start} "
                f"(found {len(iterations)} completed iterations in loop_stats.json)"
            )
            self.config.start_iteration = new_start

            # Also restore iteration history for continuity
            for it_data in iterations:
                stats = IterationStats(
                    iteration=it_data.get("iteration", 0),
                    episodes_generated=it_data.get("episodes", 0),
                    transitions_generated=it_data.get("transitions", 0),
                    training_steps=it_data.get("steps", 0),
                    actor_time_seconds=it_data.get("actor_time", 0.0),
                    trainer_time_seconds=it_data.get("trainer_time", 0.0),
                    eval_time_seconds=it_data.get("eval_time", 0.0),
                    total_time_seconds=it_data.get("total_time", 0.0),
                    eval_win_rate=it_data.get("eval_win_rate"),
                    eval_draw_rate=it_data.get("eval_draw_rate"),
                    timestamp=it_data.get("timestamp", ""),
                )
                self.iteration_history.append(stats)

            # Also restore eval history if available
            if self.config.eval_stats_path.exists():
                try:
                    with open(self.config.eval_stats_path) as f:
                        eval_data = json.load(f)
                    self.eval_history = eval_data.get("evaluations", [])
                    logger.debug(
                        f"Restored {len(self.eval_history)} evaluation records"
                    )
                except Exception as e:
                    logger.warning(f"Failed to restore eval history: {e}")

        except Exception as e:
            logger.warning(f"Failed to load loop_stats.json for auto-resume: {e}")

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
            raise FileNotFoundError(
                f"Configured actor binary not found: {self.config.actor_binary}"
            )

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
            # VACUUM to reclaim disk space and prevent database file growth
            # from degrading SQLite performance over many iterations
            replay.vacuum()
            logger.info(f"Cleared {deleted} transitions from replay buffer (vacuumed)")
            return deleted

    def _get_transition_count(self) -> int:
        """Get the current number of transitions in the replay buffer."""
        if not self.config.replay_db_path.exists():
            return 0
        with ReplayBuffer(self.config.replay_db_path) as replay:
            return replay.count(env_id=self.config.env_id)

    def _run_actor(self, num_episodes: int, iteration: int) -> tuple[bool, float]:
        """Run the actor for a specified number of episodes.

        Args:
            num_episodes: Number of self-play episodes to run.
            iteration: Current training iteration (for simulation ramping).

        Returns (success, elapsed_seconds).
        """
        actor_binary = self._find_actor_binary()

        # Calculate MCTS simulations based on iteration (ramping schedule)
        num_simulations = self.config.get_num_simulations(iteration)

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
            "--num-simulations",
            str(num_simulations),
            "--temp-threshold",
            str(self.config.temp_threshold),
        ]

        logger.info(
            f"Starting actor: {num_simulations} sims (iter {iteration}), "
            f"temp_threshold={self.config.temp_threshold}"
        )
        logger.info(f"Command: {' '.join(cmd)}")
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
                    print(f"[actor] {line.rstrip()}", flush=True)

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

        # Calculate total training steps across all iterations for continuous LR decay.
        # This enables proper AlphaZero-style training where LR decays smoothly
        # over the entire training run, not per-iteration.
        lr_total_steps = self.config.iterations * self.config.steps_per_iteration

        config = TrainerConfig(
            db_path=str(self.config.replay_db_path),
            model_dir=str(self.config.models_dir),
            stats_path=str(self.config.stats_path),
            env_id=self.config.env_id,
            total_steps=num_steps,
            start_step=start_step,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            checkpoint_interval=self.config.checkpoint_interval,
            device=self.config.device,
            max_wait=60.0,  # Short timeout since we know data exists
            eval_interval=0,  # Disable trainer's built-in eval, we do it ourselves
            lr_total_steps=lr_total_steps,  # Continuous LR decay across iterations
        )

        logger.info(f"Starting trainer for {num_steps} steps (start_step={start_step})")
        start_time = time.time()

        try:
            trainer = Trainer(config)
            stats = trainer.train()
            elapsed = time.time() - start_time

            logger.info(
                f"Trainer completed in {elapsed:.1f}s, final loss: {stats.total_loss:.4f}"
            )
            return True, elapsed

        except Exception as e:
            logger.error(f"Trainer failed: {e}")
            return False, time.time() - start_time

    def _run_evaluation(
        self, iteration: int
    ) -> tuple[float | None, float | None, float]:
        """Run evaluation against best model and optionally random baseline.

        Returns (win_rate_vs_best, draw_rate_vs_best, elapsed_seconds).
        """
        model_path = self.config.models_dir / "latest.onnx"

        if not model_path.exists():
            logger.warning(f"Model not found for evaluation: {model_path}")
            return None, None, 0.0

        start_time = time.time()
        config = get_game_config(self.config.env_id)

        try:
            # Use small temperature for evaluation to avoid deterministic games
            # With temp=0, identical models play the exact same game 50 times
            eval_temperature = 0.2
            current_policy = OnnxPolicy(str(model_path), temperature=eval_temperature)

            # --- Model vs Best (Gatekeeper) Evaluation ---
            vs_best_win_rate = None
            vs_best_draw_rate = None
            became_new_best = False

            if not self.config.best_model_path.exists():
                # First iteration: current becomes best
                logger.info("No best model yet - promoting current model to best")
                self._promote_to_best(iteration, 1.0)
                vs_best_win_rate = 1.0
                vs_best_draw_rate = 0.0
                became_new_best = True
            else:
                # Evaluate against best
                logger.info(
                    f"Evaluating vs best model (iter {self.best_model_iteration}) "
                    f"- {self.config.eval_games} games..."
                )
                best_policy = OnnxPolicy(
                    str(self.config.best_model_path), temperature=eval_temperature
                )

                vs_best_results = run_eval(
                    player1=current_policy,
                    player2=best_policy,
                    env_id=self.config.env_id,
                    config=config,
                    num_games=self.config.eval_games,
                    verbose=False,
                )

                vs_best_win_rate = vs_best_results.player1_win_rate
                vs_best_draw_rate = vs_best_results.draw_rate

                logger.info(
                    f"vs Best (iter {self.best_model_iteration}): "
                    f"Win {vs_best_win_rate:.1%}, Draw {vs_best_draw_rate:.1%}, "
                    f"Loss {vs_best_results.player2_win_rate:.1%}"
                )

                # Check if current model beats threshold
                if vs_best_win_rate > self.config.eval_win_threshold:
                    self._promote_to_best(iteration, vs_best_win_rate)
                    became_new_best = True
                else:
                    logger.info(
                        f"Current model did not beat threshold "
                        f"({vs_best_win_rate:.1%} <= {self.config.eval_win_threshold:.1%})"
                    )

            # --- Model vs Random Evaluation (optional) ---
            vs_random_win_rate = None
            vs_random_draw_rate = None

            if self.config.eval_vs_random:
                logger.info(f"Evaluating vs random - {self.config.eval_games} games...")
                random_policy = RandomPolicy()

                vs_random_results = run_eval(
                    player1=current_policy,
                    player2=random_policy,
                    env_id=self.config.env_id,
                    config=config,
                    num_games=self.config.eval_games,
                    verbose=False,
                )

                vs_random_win_rate = vs_random_results.player1_win_rate
                vs_random_draw_rate = vs_random_results.draw_rate

                logger.info(
                    f"vs Random: Win {vs_random_win_rate:.1%}, "
                    f"Draw {vs_random_draw_rate:.1%}, "
                    f"Loss {vs_random_results.player2_win_rate:.1%}"
                )

            elapsed = time.time() - start_time

            # Save to eval history (includes both evaluations)
            eval_record = {
                "iteration": iteration,
                "step": iteration * self.config.steps_per_iteration,
                # Model vs Best results
                "vs_best_win_rate": vs_best_win_rate,
                "vs_best_draw_rate": vs_best_draw_rate,
                "vs_best_opponent_iteration": self.best_model_iteration,
                "became_new_best": became_new_best,
                # Model vs Random results (if enabled)
                "vs_random_win_rate": vs_random_win_rate,
                "vs_random_draw_rate": vs_random_draw_rate,
                # Metadata
                "games": self.config.eval_games,
                "timestamp": datetime.now().isoformat(),
            }
            self.eval_history.append(eval_record)
            self._save_eval_stats()
            self._update_stats_with_eval()

            return vs_best_win_rate, vs_best_draw_rate, elapsed

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
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

    def _update_stats_with_eval(self) -> None:
        """Update stats.json with evaluation results so frontend can display them.

        The frontend reads stats.json and expects `last_eval`, `eval_history`,
        and `best_model` fields to display evaluation metrics.
        """
        if not self.eval_history:
            return

        try:
            # Read existing stats.json
            stats_data = {}
            if self.config.stats_path.exists():
                with open(self.config.stats_path) as f:
                    stats_data = json.load(f)

            # Convert eval_history to the format expected by frontend
            formatted_history = []
            for record in self.eval_history:
                formatted_history.append(
                    {
                        "step": record.get(
                            "step",
                            record.get("iteration", 0)
                            * self.config.steps_per_iteration,
                        ),
                        "current_iteration": record.get("iteration", 0),
                        # Model vs Best results
                        "opponent": "best",
                        "opponent_iteration": record.get("vs_best_opponent_iteration"),
                        "win_rate": record.get("vs_best_win_rate", 0.0),
                        "draw_rate": record.get("vs_best_draw_rate", 0.0),
                        "loss_rate": 1.0
                        - record.get("vs_best_win_rate", 0.0)
                        - record.get("vs_best_draw_rate", 0.0),
                        "became_new_best": record.get("became_new_best", False),
                        # Model vs Random results
                        "vs_random_win_rate": record.get("vs_random_win_rate"),
                        "vs_random_draw_rate": record.get("vs_random_draw_rate"),
                        # Metadata
                        "games_played": record.get("games", 0),
                        "timestamp": time.time(),
                    }
                )

            # Update stats with eval data
            if formatted_history:
                stats_data["last_eval"] = formatted_history[-1]
                stats_data["eval_history"] = formatted_history

            # Add best model info
            if self.best_model_iteration is not None:
                stats_data["best_model"] = {
                    "iteration": self.best_model_iteration,
                    "step": self.best_model_iteration * self.config.steps_per_iteration,
                }

            # Write back atomically
            temp_path = self.config.stats_path.with_suffix(".json.tmp")
            with open(temp_path, "w") as f:
                json.dump(stats_data, f, indent=2)
            temp_path.replace(self.config.stats_path)

            logger.debug(
                f"Updated stats.json with {len(formatted_history)} eval records"
            )

        except Exception as e:
            logger.warning(f"Failed to update stats.json with eval results: {e}")

    def run_iteration(self, iteration: int) -> IterationStats | None:
        """Run a single training iteration.

        Returns iteration stats, or None if shutdown was requested.
        """
        iter_start = time.time()
        num_simulations = self.config.get_num_simulations(iteration)
        logger.info("=" * 60)
        logger.info(f"ITERATION {iteration} | MCTS simulations: {num_simulations}")
        logger.info("=" * 60)

        # Step 1: Clear replay buffer
        logger.info("Step 1: Clearing replay buffer...")
        self._clear_replay_buffer()

        if self._shutdown_requested:
            return None

        # Step 2: Run actor for N episodes
        logger.info(
            f"Step 2: Running actor for {self.config.episodes_per_iteration} episodes..."
        )
        actor_success, actor_time = self._run_actor(
            self.config.episodes_per_iteration, iteration
        )

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
            self.config.eval_interval > 0 and iteration % self.config.eval_interval == 0
        )

        if should_eval:
            logger.info("Step 4: Running evaluation...")
            win_rate, draw_rate, eval_time = self._run_evaluation(iteration)
        else:
            if self.config.eval_interval > 0:
                remaining = (
                    self.config.eval_interval - iteration % self.config.eval_interval
                )
                next_eval = iteration + remaining
                logger.info(
                    f"Step 4: Skipping evaluation (next at iteration {next_eval})"
                )
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

        # Show resume status
        if self.iteration_history:
            logger.info(
                f"RESUMING from iteration {self.config.start_iteration} "
                f"({len(self.iteration_history)} previous iterations loaded)"
            )
        else:
            logger.info(f"Starting from iteration {self.config.start_iteration}")

        logger.info(f"Target iterations: {self.config.iterations}")
        logger.info(f"Episodes per iteration: {self.config.episodes_per_iteration}")
        logger.info(f"Steps per iteration: {self.config.steps_per_iteration}")
        logger.info(f"Data directory: {self.config.data_dir}")

        # Show MCTS simulation ramping settings
        start_sims = self.config.mcts_start_sims
        max_sims = self.config.mcts_max_sims
        ramp_rate = self.config.mcts_sim_ramp_rate
        iters_to_max = (
            max(1, (max_sims - start_sims) // ramp_rate) if ramp_rate > 0 else 1
        )
        logger.info(
            f"MCTS simulations: {start_sims} -> {max_sims} "
            f"(+{ramp_rate}/iter, reaches max at iter {iters_to_max})"
        )

        # Show temperature schedule
        if self.config.temp_threshold > 0:
            logger.info(
                f"Temperature schedule: temp=1.0 for moves 0-{self.config.temp_threshold - 1}, "
                f"temp=0.1 after move {self.config.temp_threshold}"
            )
        else:
            logger.info("Temperature schedule: DISABLED (temp=1.0 for all moves)")

        # Prominently show evaluation settings
        if self.config.eval_interval > 0:
            interval = self.config.eval_interval
            games = self.config.eval_games
            threshold = self.config.eval_win_threshold
            logger.info(
                f"Evaluation: ENABLED (every {interval} iteration(s), {games} games)"
            )
            logger.info(f"  Gatekeeper: win rate > {threshold:.0%} to become new best")
            if self.config.eval_vs_random:
                logger.info("  Also evaluating vs random baseline")
            if self.best_model_iteration:
                logger.info(
                    f"  Current best model: iteration {self.best_model_iteration}"
                )
        else:
            logger.info(
                "Evaluation: DISABLED (set ALPHAZERO_EVAL_INTERVAL > 0 to enable)"
            )

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
        total_in_history = len(self.iteration_history)
        # Count only iterations completed in this session
        new_completed = total_in_history - (self.config.start_iteration - 1)

        logger.info(f"\n{'='*60}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        if self.config.start_iteration > 1:
            logger.info(
                f"Completed iterations this session: {new_completed} "
                f"(total in history: {total_in_history})"
            )
        else:
            logger.info(f"Completed iterations: {total_in_history}")
        logger.info(f"Session time: {total_time:.1f}s ({total_time/3600:.2f}h)")

        if self.iteration_history:
            total_episodes = sum(s.episodes_generated for s in self.iteration_history)
            total_transitions = sum(
                s.transitions_generated for s in self.iteration_history
            )
            total_steps = sum(s.training_steps for s in self.iteration_history)
            logger.info(f"Total episodes: {total_episodes}")
            logger.info(f"Total transitions: {total_transitions}")
            logger.info(f"Total training steps: {total_steps}")

            # Report final evaluation if available
            final_with_eval = [
                s for s in self.iteration_history if s.eval_win_rate is not None
            ]
            if final_with_eval:
                final = final_with_eval[-1]
                logger.info(
                    f"Final evaluation: {final.eval_win_rate:.1%} win rate vs random"
                )


def parse_args() -> LoopConfig:
    """Parse command line arguments with config.toml defaults.

    Priority (highest to lowest):
    1. CLI arguments
    2. Environment variables (CARTRIDGE_* and legacy ALPHAZERO_*)
    3. config.toml values
    4. Built-in defaults
    """
    # Load central config (includes env var overrides)
    cfg = get_central_config()

    parser = argparse.ArgumentParser(
        description="Synchronized AlphaZero Training Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
    Settings are loaded from config.toml with the following override priority:
    1. CLI arguments (highest)
    2. Environment variables (CARTRIDGE_* or legacy ALPHAZERO_*)
    3. config.toml values
    4. Built-in defaults (lowest)

Examples:
    # Basic training with evaluation every iteration
    python -m trainer.orchestrator --iterations 50 --episodes 200 --steps 500

    # Connect4 with GPU
    python -m trainer.orchestrator --env-id connect4 --device cuda --iterations 100

    # Disable evaluation for faster training
    python -m trainer.orchestrator --eval-interval 0

    # Via Docker (uses config.toml mounted to /app/config.toml)
    docker compose up alphazero
        """,
    )

    # Iteration settings (defaults from central config)
    parser.add_argument(
        "--iterations",
        type=int,
        default=cfg.training.iterations,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=cfg.training.start_iteration,
        help="Starting iteration number",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=cfg.training.episodes_per_iteration,
        help="Episodes per iteration",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=cfg.training.steps_per_iteration,
        help="Training steps per iteration",
    )

    # Environment
    parser.add_argument(
        "--env-id",
        type=str,
        default=cfg.common.env_id,
        help="Environment ID: tictactoe, connect4",
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=cfg.common.data_dir,
        help="Data directory",
    )
    parser.add_argument(
        "--actor-binary",
        type=str,
        default=os.environ.get("ACTOR_BINARY"),
        help="Path to actor binary (env: ACTOR_BINARY)",
    )

    # Actor settings
    parser.add_argument(
        "--actor-log-interval",
        type=int,
        default=cfg.actor.log_interval,
        help="Actor log interval in episodes",
    )

    # MCTS simulation ramping settings
    parser.add_argument(
        "--mcts-start-sims",
        type=int,
        default=200,
        help="MCTS simulations for first iteration (default: 200)",
    )
    parser.add_argument(
        "--mcts-max-sims",
        type=int,
        default=cfg.mcts.num_simulations,
        help="Maximum MCTS simulations after ramping (default: from config)",
    )
    parser.add_argument(
        "--mcts-sim-ramp-rate",
        type=int,
        default=30,
        help="MCTS simulations to add per iteration (default: 30)",
    )

    # Temperature schedule
    parser.add_argument(
        "--temp-threshold",
        type=int,
        default=cfg.mcts.temp_threshold,
        help="Move number to reduce temperature (0=disabled). "
        "Recommended: tictactoe=5, connect4=15, othello=30",
    )

    # Trainer settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.training.batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg.training.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=cfg.training.checkpoint_interval,
        help="Steps between checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.training.device,
        choices=["cpu", "cuda", "mps"],
        help="Training device",
    )

    # Evaluation settings
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=cfg.evaluation.interval,
        help="Evaluate every N iterations, 0 to disable",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=cfg.evaluation.games,
        help="Games per evaluation",
    )
    parser.add_argument(
        "--eval-win-threshold",
        type=float,
        default=cfg.evaluation.win_threshold,
        help="Win rate threshold to become new best model (0.55 = 55%%)",
    )
    parser.add_argument(
        "--eval-vs-random",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=cfg.evaluation.eval_vs_random,
        help="Also evaluate against random baseline (true/false)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default=cfg.common.log_level.upper(),
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
        mcts_start_sims=args.mcts_start_sims,
        mcts_max_sims=args.mcts_max_sims,
        mcts_sim_ramp_rate=args.mcts_sim_ramp_rate,
        temp_threshold=args.temp_threshold,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        eval_win_threshold=args.eval_win_threshold,
        eval_vs_random=args.eval_vs_random,
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
