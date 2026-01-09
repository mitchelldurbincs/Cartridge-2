"""Logging utilities for the trainer.

This module provides shared logging configuration helpers used
across multiple entry points.
"""

import logging

# Loggers that are verbose during normal operation
NOISY_LOGGERS = [
    "onnx_ir.passes.common.unused_removal",
    "onnx_ir.passes.common.initializer_deduplication",
    "onnxscript.optimizer._constant_folding",
    "onnxscript.rewriter.rules.common._collapse_slices",
]


def silence_noisy_loggers() -> None:
    """Silence verbose third-party loggers (ONNX optimization, etc.)."""
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
