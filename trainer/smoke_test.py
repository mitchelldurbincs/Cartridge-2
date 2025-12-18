#!/usr/bin/env python3
"""Standalone smoke test script for the trainer.

This script runs a minimal training loop without requiring pytest.
It creates synthetic data, trains for a few steps, and verifies outputs.

Usage:
    python smoke_test.py
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trainer.network import PolicyValueNetwork, create_network
from trainer.replay import create_empty_db, insert_test_transitions, ReplayBuffer
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


def test_replay():
    """Test replay buffer operations."""
    print("\nTesting replay buffer...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Create empty database
        create_empty_db(db_path)
        assert db_path.exists(), "Database should exist"
        print("  Create database: OK")

        # Insert test data
        insert_test_transitions(db_path, count=100)
        print("  Insert test transitions: OK")

        # Open and verify
        with ReplayBuffer(db_path) as replay:
            count = replay.count()
            assert count == 100, f"Expected 100 transitions, got {count}"
            print(f"  Count transitions ({count}): OK")

            # Sample batch (num_actions=9 for TicTacToe)
            batch = replay.sample_batch_tensors(32, num_actions=9)
            assert batch is not None, "Should return batch"
            obs, policy_targets, value_targets = batch
            assert obs.shape == (32, 29), f"Expected (32, 29), got {obs.shape}"
            assert policy_targets.shape == (32, 9), f"Expected (32, 9), got {policy_targets.shape}"
            assert value_targets.shape == (32,), f"Expected (32,), got {value_targets.shape}"
            print(f"  Sample batch: OK")

    print("Replay buffer tests passed!")


def test_training_loop():
    """Run a minimal training loop."""
    print("\nTesting training loop...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "replay.db"
        model_dir = Path(tmpdir) / "models"
        stats_path = Path(tmpdir) / "stats.json"

        # Create test data
        create_empty_db(db_path)
        insert_test_transitions(db_path, count=200)
        print("  Created test database with 200 transitions")

        # Configure minimal training
        config = TrainerConfig(
            db_path=str(db_path),
            model_dir=str(model_dir),
            stats_path=str(stats_path),
            total_steps=20,
            batch_size=32,
            checkpoint_interval=10,
            stats_interval=5,
            log_interval=10,
        )

        # Train
        trainer = Trainer(config)
        stats = trainer.train()

        # Verify training completed
        assert stats.step == 20, f"Expected step 20, got {stats.step}"
        print(f"  Training completed: {stats.step} steps, loss={stats.total_loss:.4f}")

        # Verify checkpoints
        latest_onnx = model_dir / "latest.onnx"
        assert latest_onnx.exists(), "latest.onnx should exist"
        print(f"  Checkpoint saved: {latest_onnx}")

        # Verify ONNX is valid
        import onnx
        model = onnx.load(str(latest_onnx))
        onnx.checker.check_model(model)
        print("  ONNX model validation: OK")

        # Verify stats.json
        assert stats_path.exists(), "stats.json should exist"
        with open(stats_path) as f:
            loaded_stats = json.load(f)
        assert loaded_stats["step"] == 20
        print(f"  Stats written: step={loaded_stats['step']}, loss={loaded_stats['total_loss']:.4f}")

    print("Training loop tests passed!")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Cartridge2 Trainer Smoke Test")
    print("=" * 60)

    try:
        test_network()
        test_replay()
        test_training_loop()

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
