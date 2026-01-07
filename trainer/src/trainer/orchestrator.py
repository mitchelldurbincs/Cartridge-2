"""Backwards-compatible wrapper for the orchestrator module.

This module re-exports from the new trainer.orchestrator package for
backwards compatibility. New code should import directly from
trainer.orchestrator instead.

Example migration:
    # Old (still works but deprecated)
    from trainer.orchestrator import Orchestrator, LoopConfig, main

    # New (preferred)
    from trainer.orchestrator import Orchestrator, LoopConfig, main
"""

# Re-export everything from the new package
from .orchestrator import (
    ActorRunner,
    EvalRunner,
    IterationStats,
    LoopConfig,
    Orchestrator,
    StatsManager,
    main,
    parse_args,
)

__all__ = [
    "Orchestrator",
    "LoopConfig",
    "IterationStats",
    "main",
    "parse_args",
    "ActorRunner",
    "EvalRunner",
    "StatsManager",
]

# Note: We don't emit a deprecation warning here because the import path
# is technically the same (trainer.orchestrator). The package structure
# has changed but the public API remains compatible.
