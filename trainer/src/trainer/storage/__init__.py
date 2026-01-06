"""Storage abstraction layer for replay buffer and model storage.

PostgreSQL is the only replay buffer backend (local via Docker or cloud).
Model storage supports filesystem (local) and S3 (cloud/MinIO).

Usage:
    from trainer.storage import create_replay_buffer, create_model_store

    # Reads CARTRIDGE_STORAGE_POSTGRES_URL from environment
    buffer = create_replay_buffer()

    # Or explicitly specify connection string
    buffer = create_replay_buffer(
        connection_string="postgresql://user:pass@localhost:5432/cartridge"
    )

    # Model store defaults to filesystem, or use S3 for cloud
    model_store = create_model_store()  # filesystem
    model_store = create_model_store(backend="s3", bucket="my-bucket")
"""

from trainer.storage.base import (
    GameMetadata,
    ModelStore,
    ReplayBufferBase,
    Transition,
)
from trainer.storage.factory import create_model_store, create_replay_buffer
from trainer.storage.postgres import PostgresReplayBuffer

__all__ = [
    # Base classes
    "ReplayBufferBase",
    "ModelStore",
    "Transition",
    "GameMetadata",
    # Implementation
    "PostgresReplayBuffer",
    # Factory functions
    "create_replay_buffer",
    "create_model_store",
]
