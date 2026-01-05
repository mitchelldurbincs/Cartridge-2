"""Factory functions for creating storage backends.

These functions select the appropriate backend based on configuration,
allowing the same code to work locally (SQLite/filesystem) and in
Kubernetes (PostgreSQL/S3).
"""

import logging
import os
from pathlib import Path

from trainer.storage.base import ModelStore, ReplayBufferBase

logger = logging.getLogger(__name__)


def create_replay_buffer(
    backend: str | None = None,
    path: str | Path | None = None,
    connection_string: str | None = None,
    validate_schema: bool = True,
    **kwargs,
) -> ReplayBufferBase:
    """Create a replay buffer with the specified backend.

    Args:
        backend: Backend type ("sqlite" or "postgres"). If None, auto-detects
                 from environment or defaults to "sqlite".
        path: Path to SQLite database (for sqlite backend).
        connection_string: PostgreSQL connection string (for postgres backend).
        validate_schema: Whether to validate schema on connect.
        **kwargs: Additional backend-specific options.

    Returns:
        A ReplayBufferBase implementation.

    Raises:
        ValueError: If backend is unknown or required parameters are missing.
        FileNotFoundError: If SQLite database doesn't exist.
        ConnectionError: If PostgreSQL connection fails.

    Environment variables:
        CARTRIDGE_STORAGE_BACKEND: Default backend type
        CARTRIDGE_REPLAY_DB_PATH: Default SQLite path
        CARTRIDGE_POSTGRES_URL: Default PostgreSQL connection string
    """
    # Auto-detect backend from environment
    if backend is None:
        backend = os.environ.get("CARTRIDGE_STORAGE_BACKEND", "sqlite").lower()

    if backend == "sqlite":
        # Get path from argument, env, or default
        if path is None:
            path = os.environ.get("CARTRIDGE_REPLAY_DB_PATH", "./data/replay.db")

        from trainer.storage.sqlite import SqliteReplayBuffer

        logger.info(f"Using SQLite replay buffer: {path}")
        return SqliteReplayBuffer(path, validate_schema=validate_schema)

    elif backend == "postgres":
        # Get connection string from argument or env
        if connection_string is None:
            connection_string = os.environ.get("CARTRIDGE_POSTGRES_URL")

        if connection_string is None:
            raise ValueError(
                "PostgreSQL connection string required. "
                "Set CARTRIDGE_POSTGRES_URL or pass connection_string parameter."
            )

        from trainer.storage.postgres import PostgresReplayBuffer

        logger.info("Using PostgreSQL replay buffer")
        return PostgresReplayBuffer(
            connection_string, validate_schema=validate_schema, **kwargs
        )

    else:
        raise ValueError(
            f"Unknown replay buffer backend: {backend}. "
            "Supported backends: sqlite, postgres"
        )


def create_model_store(
    backend: str | None = None,
    path: str | Path | None = None,
    bucket: str | None = None,
    endpoint: str | None = None,
    **kwargs,
) -> ModelStore:
    """Create a model store with the specified backend.

    Args:
        backend: Backend type ("filesystem" or "s3"). If None, auto-detects
                 from environment or defaults to "filesystem".
        path: Path to model directory (for filesystem backend).
        bucket: S3 bucket name (for s3 backend).
        endpoint: S3-compatible endpoint URL (for s3 backend, e.g., MinIO).
        **kwargs: Additional backend-specific options.

    Returns:
        A ModelStore implementation.

    Raises:
        ValueError: If backend is unknown or required parameters are missing.

    Environment variables:
        CARTRIDGE_MODEL_BACKEND: Default backend type
        CARTRIDGE_MODEL_DIR: Default filesystem path
        CARTRIDGE_S3_BUCKET: Default S3 bucket
        CARTRIDGE_S3_ENDPOINT: S3-compatible endpoint (for MinIO)
    """
    # Auto-detect backend from environment
    if backend is None:
        backend = os.environ.get("CARTRIDGE_MODEL_BACKEND", "filesystem").lower()

    if backend == "filesystem":
        # Get path from argument, env, or default
        if path is None:
            path = os.environ.get("CARTRIDGE_MODEL_DIR", "./data/models")

        from trainer.storage.filesystem import FilesystemModelStore

        logger.info(f"Using filesystem model store: {path}")
        return FilesystemModelStore(path)

    elif backend == "s3":
        # Get bucket from argument or env
        if bucket is None:
            bucket = os.environ.get("CARTRIDGE_S3_BUCKET")

        if bucket is None:
            raise ValueError(
                "S3 bucket required. "
                "Set CARTRIDGE_S3_BUCKET or pass bucket parameter."
            )

        if endpoint is None:
            endpoint = os.environ.get("CARTRIDGE_S3_ENDPOINT")

        from trainer.storage.s3 import S3ModelStore

        logger.info(f"Using S3 model store: {bucket}")
        return S3ModelStore(bucket, endpoint=endpoint, **kwargs)

    else:
        raise ValueError(
            f"Unknown model store backend: {backend}. "
            "Supported backends: filesystem, s3"
        )


def get_storage_config_from_toml() -> dict:
    """Load storage configuration from config.toml.

    Returns:
        Dictionary with storage settings.
    """
    try:
        from trainer.central_config import get_config

        config = get_config()
        return config.get("storage", {})
    except Exception:
        return {}
