"""CLI entrypoint for the Cartridge2 trainer.

Usage:
    python -m trainer --db data/replay.db --model-dir data/models/ --steps 1000
"""

import argparse
import logging
import sys

from .trainer import Trainer, TrainerConfig, WaitTimeout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cartridge2 AlphaZero-style Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required paths
    parser.add_argument(
        "--db",
        type=str,
        default="./data/replay.db",
        help="Path to SQLite replay database",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./data/models",
        help="Directory to save ONNX model checkpoints",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="./data/stats.json",
        help="Path to write stats.json for web polling",
    )

    # Training parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (0 to disable)",
    )

    # LR scheduler
    parser.add_argument(
        "--no-lr-schedule",
        action="store_true",
        help="Disable cosine annealing LR scheduler",
    )
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.1,
        help="Final LR as ratio of initial LR (for cosine schedule)",
    )

    # Schedule parameters
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Steps between checkpoint saves",
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=10,
        help="Steps between stats updates",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between log messages",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=10,
        help="Maximum number of checkpoints to keep",
    )

    # Wait/timeout settings
    parser.add_argument(
        "--wait-interval",
        type=float,
        default=2.0,
        help="Seconds between checks when waiting for data",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=300.0,
        help="Max seconds to wait for DB/data (0 = wait forever)",
    )

    # Environment
    parser.add_argument(
        "--env-id",
        type=str,
        default="tictactoe",
        help="Environment ID (game to train)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to train on",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info("Cartridge2 Trainer starting...")
    logger.info(f"Config: db={args.db}, model_dir={args.model_dir}, steps={args.steps}")

    # Create config
    config = TrainerConfig(
        db_path=args.db,
        model_dir=args.model_dir,
        stats_path=args.stats,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        use_lr_scheduler=not args.no_lr_schedule,
        lr_min_ratio=args.lr_min_ratio,
        total_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        stats_interval=args.stats_interval,
        log_interval=args.log_interval,
        max_checkpoints=args.max_checkpoints,
        wait_interval=args.wait_interval,
        max_wait=args.max_wait,
        env_id=args.env_id,
        device=args.device,
    )

    # Create and run trainer
    try:
        trainer = Trainer(config)
        stats = trainer.train()
        logger.info(f"Training complete! Final loss: {stats.total_loss:.4f}")
        logger.info(f"Last checkpoint: {stats.last_checkpoint}")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except WaitTimeout as e:
        logger.error(f"Timeout: {e}")
        return 2
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
