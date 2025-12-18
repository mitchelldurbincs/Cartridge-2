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

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .backoff import LOG_EVERY_N_WAITS, WaitTimeout, wait_with_backoff
from .checkpoint import (
    cleanup_old_checkpoints,
    cleanup_temp_onnx_data,
    load_pytorch_checkpoint,
    save_onnx_checkpoint,
    save_pytorch_checkpoint,
)
from .config import TrainerConfig
from .evaluator import OnnxPolicy, RandomPolicy, evaluate
from .game_config import GameConfig, get_config
from .network import AlphaZeroLoss, create_network
from .replay import ReplayBuffer
from .stats import EvalStats, TrainerStats, load_stats, write_stats

# Re-export for convenience (used by tests and external callers)
__all__ = ["Trainer", "TrainerConfig", "EvalStats", "TrainerStats", "WaitTimeout"]

logger = logging.getLogger(__name__)


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

        # LR warmup configuration
        self.warmup_steps = config.lr_warmup_steps
        self.warmup_start_lr = config.learning_rate * config.lr_warmup_start_ratio
        self.target_lr = config.learning_rate

        # Initialize optimizer (start at warmup LR if warmup enabled)
        initial_lr = (
            self.warmup_start_lr if self.warmup_steps > 0 else config.learning_rate
        )
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=initial_lr,
            weight_decay=config.weight_decay,
        )

        # Initialize LR scheduler (cosine annealing with optional warmup)
        self.scheduler = None

        if config.use_lr_scheduler:
            # Cosine annealing after warmup (T_max excludes warmup steps)
            effective_steps = max(1, config.total_steps - self.warmup_steps)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=effective_steps,
                eta_min=config.learning_rate * config.lr_min_ratio,
            )

        # Try to load existing checkpoint (critical for training continuity!)
        self._checkpoint_loaded = False
        loaded_step = load_pytorch_checkpoint(
            self.network,
            self.optimizer,
            Path(config.model_dir),
            self.device,
        )
        if loaded_step is not None:
            self._checkpoint_loaded = True
            logger.info(f"Resuming training from checkpoint (step {loaded_step})")

        # Initialize loss function
        self.loss_fn = AlphaZeroLoss(
            value_weight=config.value_loss_weight,
            policy_weight=config.policy_loss_weight,
        )

        # Stats tracking - load existing stats to preserve eval history
        self.stats = load_stats(config.stats_path)
        self.stats.total_steps = config.total_steps
        self.stats.env_id = config.env_id
        self.stats._max_history = config.max_history_length
        self.samples_seen = 0

        # Replay maintenance
        self._replay_cleanup_every = (
            config.replay_cleanup_interval
            if config.replay_cleanup_interval > 0
            else config.stats_interval
        )

        # Rolling window for averaging (last 100 steps)
        self._recent_losses: list[dict[str, float]] = []
        self._rolling_window = 100

        # Buffer size caching (avoid expensive count() calls every step)
        self._buffer_size_cache: int = 0
        self._buffer_size_update_interval: int = 100  # Update every 100 steps

        # Checkpoint tracking
        self.checkpoints: list[Path] = []
        self.latest_checkpoint: Path | None = None

    def _wait_with_backoff(
        self, condition_fn, description: str, check_interval: float | None = None
    ) -> None:
        """Wait for a condition with periodic checks and timeout.

        Args:
            condition_fn: Callable returning True when condition is met.
            description: Human-readable description for logging.
            check_interval: Override default wait interval.

        Raises:
            WaitTimeout: If max_wait is exceeded (and max_wait > 0).
        """
        interval = check_interval or self.config.wait_interval
        wait_with_backoff(
            condition_fn=condition_fn,
            description=description,
            interval=interval,
            max_wait=self.config.max_wait,
            logger=logger,
        )

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
            logger.info(
                f"Gradient clipping enabled: max_norm={self.config.grad_clip_norm}"
            )
        if self.scheduler:
            min_lr = self.config.learning_rate * self.config.lr_min_ratio
            if self.warmup_steps > 0:
                logger.info(
                    f"LR schedule: {self.warmup_steps}-step warmup "
                    f"({self.warmup_start_lr:.2e} -> {self.target_lr:.2e}), "
                    f"then cosine annealing to {min_lr:.2e}"
                )
            else:
                logger.info(
                    f"LR scheduler enabled: cosine annealing "
                    f"{self.config.learning_rate} -> {min_lr}"
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

            if self.config.clear_replay_on_start:
                deleted = replay.clear_transitions()
                logger.info(
                    f"Cleared {deleted} transitions from replay buffer before training"
                )

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
            logger.info(
                f"Replay buffer contains {buffer_size} transitions for {env_id}"
            )

            # Wait for enough data with proper backoff
            if buffer_size < self.config.batch_size:
                self._wait_with_backoff(
                    lambda: replay.count(env_id=env_id) >= self.config.batch_size,
                    f"sufficient data ({self.config.batch_size} samples for {env_id})",
                )
                buffer_size = replay.count(env_id=env_id)
                logger.info(
                    f"Replay buffer now has {buffer_size} transitions for {env_id}"
                )

            # Initialize buffer size cache
            self._buffer_size_cache = buffer_size
            self.stats.replay_buffer_size = buffer_size

            # Training loop
            consecutive_skips = 0
            start_step = self.config.start_step
            for step in range(1, self.config.total_steps + 1):
                global_step = start_step + step
                self.stats.step = global_step
                checkpoint_path: Path | None = None

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

                # Update learning rate (warmup then cosine annealing)
                self._update_learning_rate(step)

                # Update stats
                self.stats.total_loss = metrics["loss/total"]
                self.stats.value_loss = metrics["loss/value"]
                self.stats.policy_loss = metrics["loss/policy"]
                self.stats.learning_rate = self.optimizer.param_groups[0]["lr"]
                self.stats.samples_seen = self.samples_seen
                self.stats.timestamp = time.time()

                # Use cached buffer size (update periodically to avoid expensive count() every step)
                if step % self._buffer_size_update_interval == 0:
                    self._buffer_size_cache = replay.count(env_id=env_id)
                self.stats.replay_buffer_size = self._buffer_size_cache

                # Track recent losses for rolling average
                self._recent_losses.append(
                    {
                        "total": metrics["loss/total"],
                        "value": metrics["loss/value"],
                        "policy": metrics["loss/policy"],
                    }
                )
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

                if (
                    self.config.replay_window > 0
                    and global_step % self._replay_cleanup_every == 0
                ):
                    deleted = replay.cleanup(self.config.replay_window)
                    if deleted > 0:
                        logger.info(
                            f"Replay cleanup removed {deleted} old transitions "
                            f"(window={self.config.replay_window})"
                        )
                    # Update cache after cleanup (buffer size changed)
                    self._buffer_size_cache = replay.count(env_id=env_id)
                    self.stats.replay_buffer_size = self._buffer_size_cache

                # Save checkpoint
                if step % self.config.checkpoint_interval == 0:
                    checkpoint_path = self._save_checkpoint(global_step)
                    self.stats.last_checkpoint = str(checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")

                # Run evaluation (based on global step count)
                if (
                    self.config.eval_interval > 0
                    and global_step % self.config.eval_interval == 0
                ):
                    if checkpoint_path is None:
                        checkpoint_path = self._save_checkpoint(global_step)
                        self.stats.last_checkpoint = str(checkpoint_path)
                        logger.info(
                            f"Saved checkpoint for evaluation: {checkpoint_path}"
                        )

                    self._evaluate_checkpoint(checkpoint_path, global_step)
                    self._write_stats()

            # Final checkpoint
            final_global_step = start_step + self.config.total_steps
            self._save_checkpoint(final_global_step, is_final=True)
            self._write_stats()

        logger.info("Training complete")
        return self.stats

    def _update_learning_rate(self, step: int) -> None:
        """Update learning rate with warmup and cosine annealing.

        Args:
            step: Current training step (1-indexed within this training run).
        """
        if step <= self.warmup_steps:
            # Linear warmup: interpolate from warmup_start_lr to target_lr
            warmup_progress = step / self.warmup_steps
            lr = self.warmup_start_lr + warmup_progress * (
                self.target_lr - self.warmup_start_lr
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        elif self.scheduler:
            # After warmup, step the cosine scheduler
            self.scheduler.step()

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
        logger.info(
            f"Running evaluation at step {step} ({self.config.eval_games} games)..."
        )

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

        Saves both ONNX (for actor inference) and PyTorch (for training continuity).

        Args:
            step: Current training step.
            is_final: Whether this is the final checkpoint (unused, for future use).

        Returns:
            Path to the saved ONNX checkpoint.
        """
        model_dir = Path(self.config.model_dir)

        # Save ONNX checkpoint (for actor inference)
        checkpoint_path = save_onnx_checkpoint(
            network=self.network,
            obs_size=self.network.obs_size,
            step=step,
            model_dir=model_dir,
            device=self.device,
        )

        # Save PyTorch checkpoint (for training continuity across iterations)
        save_pytorch_checkpoint(
            network=self.network,
            optimizer=self.optimizer,
            step=step,
            model_dir=model_dir,
        )

        # Track checkpoints for cleanup
        self.checkpoints.append(checkpoint_path)
        self.checkpoints = cleanup_old_checkpoints(
            self.checkpoints, self.config.max_checkpoints
        )
        self.latest_checkpoint = checkpoint_path

        # Clean up orphaned .onnx.data files from PyTorch exporter
        cleanup_temp_onnx_data(model_dir)

        return checkpoint_path

    def _write_stats(self) -> None:
        """Write stats.json for web polling (atomic write)."""
        write_stats(self.stats, self.config.stats_path)
