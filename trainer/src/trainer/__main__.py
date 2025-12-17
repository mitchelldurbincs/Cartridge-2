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

    TrainerConfig.configure_parser(parser)

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

    config = TrainerConfig.from_args(args)

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
