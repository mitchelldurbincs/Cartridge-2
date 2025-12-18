"""Model checkpointing with atomic writes and cleanup.

This module handles ONNX model export and checkpoint management:
- Atomic write-then-rename for safe checkpointing
- Checkpoint rotation to limit disk usage
- Cleanup of orphaned temporary files from PyTorch ONNX exporter
"""

import glob
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import onnx
import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


def save_onnx_checkpoint(
    network: "nn.Module",
    obs_size: int,
    step: int,
    model_dir: Path,
    device: torch.device,
) -> Path:
    """Export network to ONNX checkpoint with atomic write-then-rename.

    Exports ONNX once, then copies to latest.onnx to avoid duplicate export work.

    Args:
        network: The neural network to export.
        obs_size: Size of the observation input.
        step: Current training step (used for checkpoint naming).
        model_dir: Directory to save checkpoints.
        device: Device the network is on.

    Returns:
        Path to the saved checkpoint file.

    Raises:
        Exception: If ONNX export fails.
    """
    network.eval()

    # Create deterministic dummy input for ONNX export
    dummy_input = torch.zeros(1, obs_size, device=device)

    # Write to temp file first, then rename (atomic on most filesystems)
    temp_fd, temp_path = tempfile.mkstemp(suffix=".onnx", dir=model_dir)
    os.close(temp_fd)

    try:
        torch.onnx.export(
            network,
            dummy_input,
            temp_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "policy_logits": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
        )

        # If PyTorch emitted external data, inline it so the checkpoint is
        # self-contained and survives renames.
        data_sidecar = f"{temp_path}.data"
        if os.path.exists(data_sidecar):
            model = onnx.load(temp_path, load_external_data=True)
            onnx.save_model(model, temp_path, save_as_external_data=False)
            os.unlink(data_sidecar)

        # Atomic rename to final path
        checkpoint_path = model_dir / f"model_step_{step:06d}.onnx"
        os.replace(temp_path, checkpoint_path)

        # Copy to latest.onnx (instead of exporting twice)
        # Use atomic copy: copy to temp, then rename
        latest_path = model_dir / "latest.onnx"
        temp_fd2, temp_path2 = tempfile.mkstemp(suffix=".onnx", dir=model_dir)
        os.close(temp_fd2)
        try:
            shutil.copy2(checkpoint_path, temp_path2)
            os.replace(temp_path2, latest_path)
        except Exception:
            if os.path.exists(temp_path2):
                os.unlink(temp_path2)
            raise

        return checkpoint_path

    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e


def cleanup_old_checkpoints(
    checkpoints: list[Path],
    max_keep: int,
) -> list[Path]:
    """Remove old checkpoints to save disk space.

    Removes checkpoints from the front of the list (oldest first) until
    the list is at most max_keep items long.

    Args:
        checkpoints: List of checkpoint paths (oldest first).
        max_keep: Maximum number of checkpoints to retain.

    Returns:
        Updated list of checkpoints after cleanup.
    """
    while len(checkpoints) > max_keep:
        old_checkpoint = checkpoints.pop(0)
        if old_checkpoint.exists():
            old_checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    return checkpoints


def cleanup_temp_onnx_data(model_dir: Path) -> None:
    """Remove orphaned tmp*.onnx.data files created by PyTorch ONNX exporter.

    These files can accumulate when ONNX export creates external data files
    that are later inlined or when exports fail partway through.

    Args:
        model_dir: Directory to clean up.
    """
    pattern = str(model_dir / "tmp*.onnx.data")
    for data_file in glob.glob(pattern):
        try:
            os.unlink(data_file)
            logger.debug(f"Removed orphaned ONNX data file: {data_file}")
        except OSError as e:
            logger.warning(f"Failed to remove {data_file}: {e}")
