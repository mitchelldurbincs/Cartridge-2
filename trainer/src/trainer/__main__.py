"""CLI entrypoint for the Cartridge2 trainer.

Provides subcommands for different operations:
    python -m trainer train     - Run training loop on replay buffer
    python -m trainer evaluate  - Evaluate model against random baseline
    python -m trainer loop      - Run synchronized AlphaZero training

For backwards compatibility, running without a subcommand defaults to 'train':
    python -m trainer --db data/replay.db --steps 1000

Entry points after pip install:
    trainer           - Same as 'python -m trainer train'
    trainer-loop      - Same as 'python -m trainer loop'
    trainer-evaluate  - Same as 'python -m trainer evaluate'
"""

import argparse
import logging
import sys


def cmd_train(args: argparse.Namespace) -> int:
    """Run the training loop."""
    from .trainer import Trainer, TrainerConfig, WaitTimeout

    logger = logging.getLogger(__name__)
    logger.info("Cartridge2 Trainer starting...")
    logger.info(f"Config: db={args.db}, model_dir={args.model_dir}, steps={args.steps}")

    config = TrainerConfig.from_args(args)

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


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run model evaluation."""
    from pathlib import Path
    from .evaluator import (
        OnnxPolicy, RandomPolicy, evaluate,
        get_game_metadata_or_config
    )

    logger = logging.getLogger(__name__)

    # Load game configuration
    config = get_game_metadata_or_config(args.db, args.env_id)
    logger.info(
        f"Game config for {args.env_id}: "
        f"board={config.board_width}x{config.board_height}, "
        f"actions={config.num_actions}, obs_size={config.obs_size}"
    )

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    logger.info(f"Loading model: {model_path}")

    try:
        model_policy = OnnxPolicy(str(model_path), temperature=args.temperature)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    random_policy = RandomPolicy()

    logger.info(f"Running {args.games} games: {model_policy.name} vs {random_policy.name}")

    results = evaluate(
        player1=model_policy,
        player2=random_policy,
        env_id=args.env_id,
        config=config,
        num_games=args.games,
        verbose=args.verbose,
    )

    print(results.summary())

    if results.player1_win_rate > 0.7:
        print("\nModel is significantly better than random play!")
    elif results.player1_win_rate > 0.5:
        print("\nModel is slightly better than random play.")
    else:
        print("\nModel needs more training.")

    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    """Run synchronized AlphaZero training loop."""
    from .orchestrator import main as orchestrator_main
    # orchestrator has its own arg parsing, so we need to strip the 'loop' subcommand
    # from sys.argv before calling it
    if len(sys.argv) >= 2 and sys.argv[1] == "loop":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    return orchestrator_main()


def setup_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the train subcommand parser."""
    from .trainer import TrainerConfig

    parser = subparsers.add_parser(
        "train",
        help="Train on replay buffer data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    TrainerConfig.configure_parser(parser)
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.set_defaults(func=cmd_train)


def setup_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the evaluate subcommand parser."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model against random baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="./data/models/latest.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--db", type=str, default="./data/replay.db",
        help="Path to replay database (for game metadata)",
    )
    parser.add_argument(
        "--env-id", type=str, default="tictactoe",
        choices=["tictactoe", "connect4"],
        help="Game environment",
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games to play",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print individual game moves",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.set_defaults(func=cmd_evaluate)


def setup_loop_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the loop subcommand parser."""
    parser = subparsers.add_parser(
        "loop",
        help="Run synchronized AlphaZero training (actor + trainer + eval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Synchronized AlphaZero training loop that coordinates:
1. Self-play episode generation (actor)
2. Neural network training
3. Model evaluation against random baseline

Each iteration clears the replay buffer to ensure training data
comes only from the current model version.
        """,
    )
    # The loop command re-parses sys.argv via orchestrator.main()
    # so we don't need to add arguments here
    parser.set_defaults(func=cmd_loop)


def main() -> int:
    """Main entry point with subcommand support."""
    # Check if we're being called with a subcommand
    # For backwards compatibility, default to 'train' if no subcommand given
    if len(sys.argv) >= 2 and sys.argv[1] in ("train", "evaluate", "loop", "-h", "--help"):
        # Subcommand mode
        parser = argparse.ArgumentParser(
            description="Cartridge2 AlphaZero Trainer",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparsers = parser.add_subparsers(
            title="commands",
            description="Available commands",
            dest="command",
        )

        setup_train_parser(subparsers)
        setup_evaluate_parser(subparsers)
        setup_loop_parser(subparsers)

        args = parser.parse_args()

        if args.command is None:
            parser.print_help()
            return 0

        # Configure logging
        log_level = getattr(args, "log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        return args.func(args)

    else:
        # Backwards compatibility: treat as 'train' command
        from .trainer import TrainerConfig

        parser = argparse.ArgumentParser(
            description="Cartridge2 AlphaZero-style Trainer",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        TrainerConfig.configure_parser(parser)
        parser.add_argument(
            "--log-level", type=str, default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level",
        )

        args = parser.parse_args()

        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        return cmd_train(args)


if __name__ == "__main__":
    sys.exit(main())
