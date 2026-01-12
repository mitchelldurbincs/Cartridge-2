"""Structured logging configuration for cloud deployments.

This module provides JSON-formatted logging for Google Cloud Logging integration.
It can be toggled via the CARTRIDGE_LOGGING_FORMAT environment variable or
config.toml [logging] section.

Usage:
    from trainer.structured_logging import setup_logging

    # In your main() function:
    setup_logging(level="INFO", component="trainer")
"""

import logging
import os
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


class CloudJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter optimized for Google Cloud Logging.

    Outputs logs in a format that Google Cloud Logging can parse and index,
    with proper severity mapping and structured fields.
    """

    # Map Python log levels to Google Cloud Logging severity
    SEVERITY_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def __init__(self, component: str = "trainer", include_timestamp: bool = True):
        """Initialize the formatter.

        Args:
            component: Component name to include in every log (trainer, orchestrator, etc.)
            include_timestamp: Whether to include timestamp (set False if cloud adds it)
        """
        self.component = component
        self.include_timestamp = include_timestamp
        # Define format with timestamp if needed
        fmt = "%(message)s"
        if include_timestamp:
            fmt = "%(asctime)s " + fmt
        super().__init__(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S%z")

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        """Add structured fields to the log record."""
        super().add_fields(log_record, record, message_dict)

        # Add severity for Google Cloud Logging
        log_record["severity"] = self.SEVERITY_MAP.get(record.levelname, "DEFAULT")

        # Add component identifier
        log_record["component"] = self.component

        # Add module/target for context
        log_record["target"] = record.name

        # Remove redundant fields
        if "levelname" in log_record:
            del log_record["levelname"]

        # Add timestamp in ISO format if enabled
        if self.include_timestamp and "asctime" in log_record:
            log_record["timestamp"] = log_record.pop("asctime")


def get_logging_format() -> str:
    """Determine logging format from environment or config.

    Returns:
        "json" or "text"
    """
    # Environment variable takes precedence
    env_format = os.environ.get("CARTRIDGE_LOGGING_FORMAT", "").lower()
    if env_format in ("json", "text"):
        return env_format

    # Try to load from config.toml
    try:
        from .central_config import load_central_config

        config = load_central_config()
        return config.get("logging", {}).get("format", "text").lower()
    except Exception:
        return "text"


def setup_logging(
    level: str = "INFO",
    component: str = "trainer",
    silence_noisy: bool = True,
) -> None:
    """Configure logging with optional JSON format for cloud deployments.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        component: Component name to include in logs
        silence_noisy: Whether to silence noisy third-party loggers
    """
    log_format = get_logging_format()
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level.upper())

    if log_format == "json":
        # JSON format for cloud logging
        include_timestamp = os.environ.get(
            "CARTRIDGE_LOGGING_INCLUDE_TIMESTAMPS", "true"
        ).lower() in ("true", "1", "yes")
        formatter = CloudJsonFormatter(
            component=component,
            include_timestamp=include_timestamp,
        )
    else:
        # Human-readable format for local development
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Silence noisy loggers
    if silence_noisy:
        from .logging_utils import silence_noisy_loggers

        silence_noisy_loggers()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds structured fields to log messages.

    This adapter allows adding extra fields to log messages that will be
    properly formatted as JSON fields when using the JSON formatter.

    Usage:
        logger = StructuredLoggerAdapter(
            logging.getLogger(__name__),
            {"component": "trainer", "env_id": "tictactoe"}
        )
        logger.info("Training started", extra={"step": 100, "loss": 0.5})
    """

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Process the logging message and keyword arguments."""
        extra = kwargs.get("extra", {})
        # Merge adapter extra with call extra
        merged_extra = {**self.extra, **extra}
        kwargs["extra"] = merged_extra
        return msg, kwargs
