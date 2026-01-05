"""Storage abstraction layer for K8s-ready backends.

This package provides pluggable storage backends that work both locally
(SQLite, filesystem) and in Kubernetes (PostgreSQL, S3).

Usage:
    from trainer.storage import create_replay_buffer, create_model_store

    # Uses config.toml [storage] section to select backend
    buffer = create_replay_buffer()
    model_store = create_model_store()

    # Or explicitly specify backend
    buffer = create_replay_buffer(backend="sqlite", path="./data/replay.db")
    buffer = create_replay_buffer(backend="postgres", connection_string="...")
"""

from trainer.storage.base import (
    GameMetadata,
    ModelStore,
    ReplayBufferBase,
    Transition,
)
from trainer.storage.factory import create_model_store, create_replay_buffer

__all__ = [
    # Base classes
    "ReplayBufferBase",
    "ModelStore",
    "Transition",
    "GameMetadata",
    # Factory functions
    "create_replay_buffer",
    "create_model_store",
]
