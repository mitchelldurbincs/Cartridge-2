"""Wait and retry utilities with exponential backoff.

This module provides generic wait/retry functionality for conditions
that may take time to become true (e.g., waiting for a database file
to appear or for sufficient data to accumulate).
"""

import logging
import time
from typing import Callable

# Constants for backoff/wait behavior
DEFAULT_WAIT_INTERVAL = 2.0  # seconds between checks
DEFAULT_MAX_WAIT = 300.0  # 5 minutes max wait
LOG_EVERY_N_WAITS = 5  # Log waiting message every N intervals


class WaitTimeout(Exception):
    """Raised when waiting for a condition exceeds max_wait."""

    pass


def wait_with_backoff(
    condition_fn: Callable[[], bool],
    description: str,
    interval: float = DEFAULT_WAIT_INTERVAL,
    max_wait: float = DEFAULT_MAX_WAIT,
    logger: logging.Logger | None = None,
) -> None:
    """Wait for a condition with periodic checks and timeout.

    Args:
        condition_fn: Callable returning True when condition is met.
        description: Human-readable description for logging.
        interval: Seconds between condition checks.
        max_wait: Maximum seconds to wait (0 = wait forever).
        logger: Logger instance for status messages. If None, no logging.

    Raises:
        WaitTimeout: If max_wait is exceeded (and max_wait > 0).
    """
    start_time = time.time()
    wait_count = 0

    while not condition_fn():
        elapsed = time.time() - start_time
        if max_wait > 0 and elapsed >= max_wait:
            raise WaitTimeout(
                f"Timed out waiting for {description} after {elapsed:.1f}s"
            )

        wait_count += 1
        if logger and wait_count % LOG_EVERY_N_WAITS == 1:
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
