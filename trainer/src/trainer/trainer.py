"""Training loop with ONNX export and stats tracking.

This module provides the main training loop that:
1. Loads batches from the SQLite replay buffer
2. Trains the AlphaZero-style network
3. Exports ONNX checkpoints using atomic write-then-rename
4. Writes stats.json for web visualization

Training targets:
    - Policy targets: MCTS visit count distributions (soft targets) from the actor.
    - Value targets: Game outcomes (win=+1, loss=-1, draw=0) propagated from
      terminal states. Each position is labeled with the final outcome from
      that player's perspective, giving meaningful signal at every position.
"""

import dataclasses
import glob
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import onnx
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .game_config import GameConfig, get_config
from .network import AlphaZeroLoss, PolicyValueNetwork, create_network
from .replay import GameMetadata, ReplayBuffer
from .evaluator import evaluate, OnnxPolicy, RandomPolicy

logger = logging.getLogger(__name__)

# Constants for backoff/wait behavior
DEFAULT_WAIT_INTERVAL = 2.0  # seconds between checks
DEFAULT_MAX_WAIT = 300.0  # 5 minutes max wait
LOG_EVERY_N_WAITS = 5  # Log waiting message every N intervals


def cli_field(
    default,
    *,
    cli: str | None = None,
    help: str = "",
    choices: list | None = None,
    action: str | None = None,
):
    """Create a dataclass field with CLI metadata."""

    metadata: dict[str, object] = {}
    if cli is not None:
        metadata["cli"] = cli
        metadata["help"] = help
        if choices:
            metadata["choices"] = choices
        if action:
            metadata["action"] = action

    return field(default=default, metadata=metadata)


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    db_path: str = cli_field(
        "./data/replay.db", cli="--db", help="Path to SQLite replay database"
    )
    model_dir: str = cli_field(
        "./data/models", cli="--model-dir", help="Directory for ONNX model checkpoints"
    )
    stats_path: str = cli_field(
        "./data/stats.json", cli="--stats", help="Path to write stats.json for web polling"
    )

    # Training hyperparameters
    batch_size: int = cli_field(64, cli="--batch-size", help="Batch size for training")
    learning_rate: float = cli_field(1e-3, cli="--lr", help="Learning rate")
    weight_decay: float = cli_field(1e-4, cli="--weight-decay", help="Weight decay")
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0

    # Gradient clipping (0 = disabled)
    grad_clip_norm: float = cli_field(
        1.0, cli="--grad-clip", help="Gradient clipping max norm (0 to disable)"
    )

    # Learning rate schedule
    use_lr_scheduler: bool = cli_field(
        True,
        cli="--no-lr-schedule",
        action="store_false",
        help="Disable cosine annealing LR scheduler",
    )
    lr_min_ratio: float = cli_field(
        0.1, cli="--lr-min-ratio", help="Final LR as ratio of initial LR"
    )

    # Training schedule
    total_steps: int = cli_field(1000, cli="--steps", help="Total training steps")
    checkpoint_interval: int = cli_field(
        100, cli="--checkpoint-interval", help="Steps between checkpoint saves"
    )
    stats_interval: int = cli_field(10, cli="--stats-interval", help="Steps between stats updates")
    log_interval: int = cli_field(10, cli="--log-interval", help="Steps between log messages")

    # Checkpoint management
    max_checkpoints: int = cli_field(
        10, cli="--max-checkpoints", help="Maximum number of checkpoints to keep"
    )

    # Wait/backoff settings
    wait_interval: float = cli_field(
        DEFAULT_WAIT_INTERVAL,
        cli="--wait-interval",
        help="Seconds between checks when waiting for data",
    )
    max_wait: float = cli_field(
        DEFAULT_MAX_WAIT,
        cli="--max-wait",
        help="Max seconds to wait for DB/data (0 = wait forever)",
    )

    # Step offset for continuous training (checkpoint naming)
    start_step: int = cli_field(
        0, cli="--start-step", help="Starting step number for checkpoint naming"
    )

    # Evaluation settings
    eval_interval: int = cli_field(
        100, cli="--eval-interval", help="Steps between evaluations (0 to disable)"
    )
    eval_games: int = cli_field(
        50, cli="--eval-games", help="Number of games per evaluation"
    )

    # Stats history settings
    max_history_length: int = 100

    # Environment
    env_id: str = cli_field("tictactoe", cli="--env-id", help="Environment ID")
    device: str = cli_field(
        "cpu", cli="--device", choices=["cpu", "cuda", "mps"], help="Device to train on"
    )

    @classmethod
    def configure_parser(cls, parser) -> None:
        """Add CLI arguments to parser based on field metadata."""

        for f in fields(cls):
            cli_flag = f.metadata.get("cli")
            if not cli_flag:
                continue

            kwargs: dict[str, object] = {
                "help": f.metadata.get("help", ""),
            }

            if f.default is not dataclasses.MISSING:
                kwargs["default"] = f.default

            action = f.metadata.get("action")
            if action:
                kwargs["action"] = action
                kwargs.pop("default", None)
            else:
                if f.type in (int, float, str):
                    kwargs["type"] = f.type

            if f.metadata.get("choices"):
                kwargs["choices"] = f.metadata["choices"]

            parser.add_argument(cli_flag, **kwargs)

    @classmethod
    def from_args(cls, args) -> "TrainerConfig":
        """Construct a TrainerConfig from parsed argparse args."""

        config_kwargs: dict[str, object] = {}
        for f in fields(cls):
            cli_flag = f.metadata.get("cli")
            if not cli_flag:
                continue

            arg_name = cli_flag.lstrip("-").replace("-", "_")
            if hasattr(args, arg_name):
                config_kwargs[f.name] = getattr(args, arg_name)

        return cls(**config_kwargs)


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
    _max_history: int = 100  # Bound history length

    # Evaluation metrics
    last_eval: EvalStats | None = None
    eval_history: list[dict] = field(default_factory=list)
    _max_eval_history: int = 50  # Keep last 50 evaluations

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
            "last_eval": self.last_eval.to_dict() if self.last_eval else None,
            "eval_history": self.eval_history,
        }


class WaitTimeout(Exception):
    """Raised when waiting for a condition exceeds max_wait."""

    pass


class Trainer:
    """AlphaZero-style trainer for game agents."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Get game configuration
        self.game_config = get_config(config.env_id)

        # Create model directory
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(config.stats_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize network
        self.network = create_network(config.env_id)
        self.network.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Initialize LR scheduler (cosine annealing)
        self.scheduler = None
        if config.use_lr_scheduler:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.total_steps,
                eta_min=config.learning_rate * config.lr_min_ratio,
            )

        # Initialize loss function
        self.loss_fn = AlphaZeroLoss(
            value_weight=config.value_loss_weight,
            policy_weight=config.policy_loss_weight,
        )

        # Stats tracking
        self.stats = TrainerStats(total_steps=config.total_steps)
        self.stats._max_history = config.max_history_length
        self.samples_seen = 0

        # Rolling window for averaging (last 100 steps)
        self._recent_losses: list[dict[str, float]] = []
        self._rolling_window = 100

        # Checkpoint tracking
        self.checkpoints: list[Path] = []

    def _wait_with_backoff(
        self, condition_fn, description: str, check_interval: float | None = None
    ) -> None:
        """Wait for a condition with exponential backoff and timeout.

        Args:
            condition_fn: Callable returning True when condition is met.
            description: Human-readable description for logging.
            check_interval: Override default wait interval.

        Raises:
            WaitTimeout: If max_wait is exceeded (and max_wait > 0).
        """
        interval = check_interval or self.config.wait_interval
        max_wait = self.config.max_wait
        start_time = time.time()
        wait_count = 0

        while not condition_fn():
            elapsed = time.time() - start_time
            if max_wait > 0 and elapsed >= max_wait:
                raise WaitTimeout(
                    f"Timed out waiting for {description} after {elapsed:.1f}s"
                )

            wait_count += 1
            if wait_count % LOG_EVERY_N_WAITS == 1:
                if max_wait > 0:
                    remaining = max_wait - elapsed
                    logger.info(
                        f"Waiting for {description}... "
                        f"(elapsed: {elapsed:.1f}s, timeout in: {remaining:.1f}s)"
                    )
                else:
                    logger.info(
                        f"Waiting for {description}... (elapsed: {elapsed:.1f}s)"
                    )

            time.sleep(interval)

    def train(self) -> TrainerStats:
        """Run the training loop.

        Returns:
            Final training statistics.

        Raises:
            WaitTimeout: If max_wait is exceeded waiting for database or data.
        """
        logger.info(f"Starting training for {self.config.total_steps} steps")
        logger.info(f"Loading replay buffer from {self.config.db_path}")
        if self.config.grad_clip_norm > 0:
            logger.info(f"Gradient clipping enabled: max_norm={self.config.grad_clip_norm}")
        if self.scheduler:
            logger.info(
                f"LR scheduler enabled: cosine annealing "
                f"{self.config.learning_rate} -> {self.config.learning_rate * self.config.lr_min_ratio}"
            )

        # Wait for replay buffer to exist with proper backoff
        db_path = Path(self.config.db_path)
        if not db_path.exists():
            self._wait_with_backoff(
                lambda: db_path.exists(),
                f"replay database at {db_path}",
            )
            logger.info("Replay database found, opening connection")

        with ReplayBuffer(self.config.db_path) as replay:
            env_id = self.config.env_id

            # Try to get game metadata from database (preferred, self-describing)
            db_metadata = replay.get_metadata(env_id)
            if db_metadata:
                logger.info(f"Using game metadata from database for {env_id}")
                # Override game_config with values from database
                self.game_config = GameConfig(
                    env_id=db_metadata.env_id,
                    display_name=db_metadata.display_name,
                    board_width=db_metadata.board_width,
                    board_height=db_metadata.board_height,
                    num_actions=db_metadata.num_actions,
                    obs_size=db_metadata.obs_size,
                    legal_mask_offset=db_metadata.legal_mask_offset,
                )
            else:
                logger.warning(
                    f"No metadata in database for {env_id}, using fallback config"
                )

            buffer_size = replay.count(env_id=env_id)
            logger.info(f"Replay buffer contains {buffer_size} transitions for {env_id}")

            # Wait for enough data with proper backoff
            if buffer_size < self.config.batch_size:
                self._wait_with_backoff(
                    lambda: replay.count(env_id=env_id) >= self.config.batch_size,
                    f"sufficient data ({self.config.batch_size} samples for {env_id})",
                )
                buffer_size = replay.count(env_id=env_id)
                logger.info(f"Replay buffer now has {buffer_size} transitions for {env_id}")

            self.stats.replay_buffer_size = buffer_size

            # Training loop
            consecutive_skips = 0
            start_step = self.config.start_step
            for step in range(1, self.config.total_steps + 1):
                global_step = start_step + step
                self.stats.step = global_step

                # Sample batch (filter by env_id and use correct num_actions)
                batch = replay.sample_batch_tensors(
                    self.config.batch_size,
                    num_actions=self.game_config.num_actions,
                    env_id=env_id,
                )
                if batch is None:
                    consecutive_skips += 1
                    if consecutive_skips % LOG_EVERY_N_WAITS == 1:
                        logger.warning(
                            f"Not enough data for batch (need {self.config.batch_size}), "
                            f"sleeping {self.config.wait_interval}s..."
                        )
                    time.sleep(self.config.wait_interval)
                    continue

                consecutive_skips = 0
                observations, policy_targets, value_targets = batch
                self.samples_seen += len(observations)

                # Train step
                metrics = self._train_step(observations, policy_targets, value_targets)

                # Step LR scheduler
                if self.scheduler:
                    self.scheduler.step()

                # Update stats
                self.stats.total_loss = metrics["loss/total"]
                self.stats.value_loss = metrics["loss/value"]
                self.stats.policy_loss = metrics["loss/policy"]
                self.stats.learning_rate = self.optimizer.param_groups[0]["lr"]
                self.stats.samples_seen = self.samples_seen
                self.stats.timestamp = time.time()
                self.stats.replay_buffer_size = replay.count(env_id=env_id)

                # Track recent losses for rolling average
                self._recent_losses.append({
                    "total": metrics["loss/total"],
                    "value": metrics["loss/value"],
                    "policy": metrics["loss/policy"],
                })
                if len(self._recent_losses) > self._rolling_window:
                    self._recent_losses.pop(0)

                # Log progress
                if step % self.config.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    # Compute rolling averages
                    n = len(self._recent_losses)
                    avg_total = sum(x["total"] for x in self._recent_losses) / n
                    avg_value = sum(x["value"] for x in self._recent_losses) / n
                    avg_policy = sum(x["policy"] for x in self._recent_losses) / n
                    logger.info(
                        f"Step {global_step} ({step}/{self.config.total_steps}): "
                        f"loss={metrics['loss/total']:.4f} "
                        f"(v={metrics['loss/value']:.4f}, p={metrics['loss/policy']:.4f}) "
                        f"avg100={avg_total:.4f} (v={avg_value:.4f}, p={avg_policy:.4f}) "
                        f"lr={lr:.2e}"
                    )

                # Save stats (uses bounded append)
                if step % self.config.stats_interval == 0:
                    self.stats.append_history(
                        {
                            "step": global_step,
                            "total_loss": metrics["loss/total"],
                            "value_loss": metrics["loss/value"],
                            "policy_loss": metrics["loss/policy"],
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                    )
                    self._write_stats()

                # Save checkpoint
                if step % self.config.checkpoint_interval == 0:
                    checkpoint_path = self._save_checkpoint(global_step)
                    self.stats.last_checkpoint = str(checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")

                # Run evaluation
                if (
                    self.config.eval_interval > 0
                    and step % self.config.eval_interval == 0
                ):
                    self._evaluate_checkpoint(checkpoint_path, global_step)
                    self._write_stats()

            # Final checkpoint
            final_global_step = start_step + self.config.total_steps
            self._save_checkpoint(final_global_step, is_final=True)
            self._write_stats()

        logger.info("Training complete")
        return self.stats

    def _train_step(
        self,
        observations: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
    ) -> dict[str, float]:
        """Perform a single training step.

        Args:
            observations: Game observations (batch, obs_size)
            policy_targets: MCTS policy distributions (batch, action_size)
            value_targets: Value targets (batch,)
        """
        self.network.train()
        self.optimizer.zero_grad()

        # Convert to tensors
        obs_t = torch.from_numpy(observations).to(self.device)
        policy_targets_t = torch.from_numpy(policy_targets).to(self.device)
        value_targets_t = torch.from_numpy(value_targets).to(self.device)

        # Extract legal mask from observations using game-specific offsets
        mask_start = self.game_config.legal_mask_offset
        mask_end = mask_start + self.game_config.num_actions
        legal_mask = obs_t[:, mask_start:mask_end]

        # Forward pass
        policy_logits, value_pred = self.network(obs_t)

        # Compute loss with soft policy targets
        loss, metrics = self.loss_fn(
            policy_logits, value_pred, policy_targets_t, value_targets_t, legal_mask
        )

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        if self.config.grad_clip_norm > 0:
            grad_norm = nn_utils.clip_grad_norm_(
                self.network.parameters(), self.config.grad_clip_norm
            )
            metrics["grad_norm"] = grad_norm.item()

        self.optimizer.step()

        return metrics

    def _evaluate_checkpoint(self, checkpoint_path: Path, step: int) -> None:
        """Run evaluation on a checkpoint and record results.

        Args:
            checkpoint_path: Path to the ONNX checkpoint to evaluate.
            step: Current training step for recording.
        """
        logger.info(f"Running evaluation at step {step} ({self.config.eval_games} games)...")

        try:
            model_policy = OnnxPolicy(str(checkpoint_path), temperature=0.0)
            random_policy = RandomPolicy()

            results = evaluate(
                player1=model_policy,
                player2=random_policy,
                env_id=self.config.env_id,
                config=self.game_config,
                num_games=self.config.eval_games,
                verbose=False,
            )

            eval_stats = EvalStats(
                step=step,
                win_rate=results.player1_win_rate,
                draw_rate=results.draw_rate,
                loss_rate=results.player2_win_rate,
                games_played=results.games_played,
                avg_game_length=results.avg_game_length,
                timestamp=time.time(),
            )

            self.stats.append_eval(eval_stats)

            logger.info(
                f"Evaluation complete: win_rate={eval_stats.win_rate:.1%}, "
                f"draw_rate={eval_stats.draw_rate:.1%}, "
                f"avg_length={eval_stats.avg_game_length:.1f}"
            )

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

    def _save_checkpoint(self, step: int, is_final: bool = False) -> Path:
        """Save model checkpoint with atomic write-then-rename.

        Exports ONNX once, then copies to latest.onnx to avoid duplicate work.

        Returns the path to the saved checkpoint.
        """
        model_dir = Path(self.config.model_dir)

        # Export to ONNX
        self.network.eval()

        # Create deterministic dummy input for ONNX export
        # Shape matches the network's expected observation size
        dummy_input = torch.zeros(1, self.network.obs_size, device=self.device)

        # Write to temp file first, then rename (atomic on most filesystems)
        temp_fd, temp_path = tempfile.mkstemp(suffix=".onnx", dir=model_dir)
        os.close(temp_fd)

        try:
            torch.onnx.export(
                self.network,
                dummy_input,
                temp_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["observation"],
                output_names=["policy_logits", "value"],
                dynamic_axes={
                    "observation": {0: "batch_size"},
                    "policy_logits": {0: "batch_size"},
                    "value": {0: "batch_size"},
                },
            )

            # If PyTorch emitted external data, inline it so the checkpoint is
            # self-contained and survives renames.
            data_sidecar = f"{temp_path}.data"
            if os.path.exists(data_sidecar):
                model = onnx.load(temp_path, load_external_data=True)
                onnx.save_model(model, temp_path, save_as_external_data=False)
                os.unlink(data_sidecar)

            # Atomic rename to final path
            checkpoint_path = model_dir / f"model_step_{step:06d}.onnx"
            os.replace(temp_path, checkpoint_path)

            # Copy to latest.onnx (instead of exporting twice)
            # Use atomic copy: copy to temp, then rename
            latest_path = model_dir / "latest.onnx"
            temp_fd2, temp_path2 = tempfile.mkstemp(suffix=".onnx", dir=model_dir)
            os.close(temp_fd2)
            try:
                shutil.copy2(checkpoint_path, temp_path2)
                os.replace(temp_path2, latest_path)
            except Exception:
                if os.path.exists(temp_path2):
                    os.unlink(temp_path2)
                raise

            # Track checkpoints for cleanup
            self.checkpoints.append(checkpoint_path)
            self._cleanup_old_checkpoints()

            # Clean up orphaned .onnx.data files from PyTorch exporter
            self._cleanup_temp_onnx_data(model_dir)

            return checkpoint_path

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def _cleanup_temp_onnx_data(self, model_dir: Path) -> None:
        """Remove orphaned tmp*.onnx.data files created by PyTorch ONNX exporter."""
        pattern = str(model_dir / "tmp*.onnx.data")
        for data_file in glob.glob(pattern):
            try:
                os.unlink(data_file)
                logger.debug(f"Removed orphaned ONNX data file: {data_file}")
            except OSError as e:
                logger.warning(f"Failed to remove {data_file}: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save disk space."""
        while len(self.checkpoints) > self.config.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    def _write_stats(self) -> None:
        """Write stats.json for web polling (atomic write)."""
        stats_path = Path(self.config.stats_path)

        # Write to temp file then rename
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".json", dir=stats_path.parent
        )
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            os.replace(temp_path, stats_path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
