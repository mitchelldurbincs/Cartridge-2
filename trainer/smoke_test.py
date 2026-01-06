#!/usr/bin/env python3
"""Standalone smoke test script for the trainer.

This script runs a minimal test suite without requiring pytest.
It tests the network and verifies basic functionality.

Note: Integration tests requiring PostgreSQL are skipped unless
CARTRIDGE_STORAGE_POSTGRES_URL is set.

Usage:
    python smoke_test.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trainer.network import PolicyValueNetwork, create_network
from trainer.trainer import Trainer, TrainerConfig


def test_network():
    """Test that the network produces valid outputs."""
    import torch

    print("Testing network...")
    net = create_network("tictactoe")

    # Test forward pass
    batch = torch.randn(4, 29)
    policy_logits, value = net(batch)

    assert policy_logits.shape == (4, 9), f"Expected (4, 9), got {policy_logits.shape}"
    assert value.shape == (4, 1), f"Expected (4, 1), got {value.shape}"
    assert (value >= -1).all() and (value <= 1).all(), "Value should be in [-1, 1]"

    print("  Network forward pass: OK")

    # Test with legal mask
    legal_mask = torch.ones(4, 9)
    legal_mask[:, :3] = 0  # Mark first 3 positions as illegal

    policy_probs, _ = net.predict(batch, legal_mask)
    assert (policy_probs[:, :3] == 0).all(), "Illegal moves should have 0 probability"

    print("  Network with legal mask: OK")
    print("Network tests passed!")


def test_storage_factory():
    """Test that storage factory works correctly."""
    print("\nTesting storage factory...")

    from trainer.storage import create_replay_buffer

    # Test that it requires PostgreSQL URL
    with patch.dict(os.environ, {}, clear=True):
        try:
            create_replay_buffer()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "PostgreSQL connection string required" in str(e)
            print("  Factory requires PostgreSQL URL: OK")

    print("Storage factory tests passed!")


def test_trainer_creation():
    """Test that trainer can be created with mocked replay buffer."""
    print("\nTesting trainer creation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        stats_path = Path(tmpdir) / "stats.json"

        config = TrainerConfig(
            model_dir=str(model_dir),
            stats_path=str(stats_path),
            total_steps=10,
        )

        # Mock the replay buffer to avoid needing PostgreSQL
        with patch("trainer.trainer.create_replay_buffer") as mock_factory:
            mock_replay = MagicMock()
            mock_replay.get_metadata.return_value = None
            mock_replay.count.return_value = 0
            mock_factory.return_value = mock_replay

            trainer = Trainer(config)
            assert trainer.network is not None
            assert model_dir.exists()
            print("  Trainer creation: OK")

    print("Trainer creation tests passed!")


def test_training_with_postgres():
    """Run a minimal training loop with PostgreSQL (if configured)."""
    postgres_url = os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL")
    if not postgres_url:
        print("\nSkipping PostgreSQL integration test (CARTRIDGE_STORAGE_POSTGRES_URL not set)")
        return

    print("\nTesting training loop with PostgreSQL...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        stats_path = Path(tmpdir) / "stats.json"

        config = TrainerConfig(
            model_dir=str(model_dir),
            stats_path=str(stats_path),
            total_steps=10,
            batch_size=32,
            checkpoint_interval=5,
            max_wait=10.0,
        )

        trainer = Trainer(config)

        # Check if there's data in the database
        from trainer.storage import create_replay_buffer
        replay = create_replay_buffer()
        count = replay.count()
        replay.close()

        if count < 32:
            print(f"  Not enough data in database ({count} transitions), skipping training")
            return

        stats = trainer.train()
        print(f"  Training completed: {stats.step} steps, loss={stats.total_loss:.4f}")

    print("PostgreSQL integration tests passed!")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Cartridge2 Trainer Smoke Test")
    print("=" * 60)

    try:
        test_network()
        test_storage_factory()
        test_trainer_creation()
        test_training_with_postgres()

        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
