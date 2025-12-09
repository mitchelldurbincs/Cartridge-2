"""Smoke tests for the trainer.

Run with: pytest tests/test_smoke.py -v
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from trainer.network import AlphaZeroLoss, PolicyValueNetwork, create_network
from trainer.replay import ReplayBuffer, SchemaError, create_empty_db, insert_test_transitions
from trainer.trainer import Trainer, TrainerConfig, WaitTimeout


class TestNetwork:
    """Tests for the neural network."""

    def test_network_creation(self):
        net = create_network("tictactoe")
        assert net.obs_size == 29
        assert net.action_size == 9

    def test_network_forward(self):
        net = create_network("tictactoe")
        batch = torch.randn(8, 29)

        policy_logits, value = net(batch)

        assert policy_logits.shape == (8, 9)
        assert value.shape == (8, 1)
        # Value should be in [-1, 1] due to tanh
        assert (value >= -1).all() and (value <= 1).all()

    def test_network_predict_with_mask(self):
        net = create_network("tictactoe")
        batch = torch.randn(4, 29)
        # Mask out positions 0, 1, 2 as illegal
        legal_mask = torch.ones(4, 9)
        legal_mask[:, :3] = 0

        policy_probs, value = net.predict(batch, legal_mask)

        assert policy_probs.shape == (4, 9)
        # Illegal moves should have zero probability
        assert (policy_probs[:, :3] == 0).all()
        # Probabilities should sum to 1
        assert torch.allclose(policy_probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_create_network(self):
        net = create_network("tictactoe")
        assert isinstance(net, PolicyValueNetwork)

        with pytest.raises(ValueError):
            create_network("unknown_game")


class TestLoss:
    """Tests for the loss function."""

    def test_loss_computation(self):
        loss_fn = AlphaZeroLoss()

        batch_size = 8
        policy_logits = torch.randn(batch_size, 9)
        value = torch.randn(batch_size, 1)
        # Create soft policy targets (probability distributions)
        target_policy = torch.softmax(torch.randn(batch_size, 9), dim=-1)
        target_values = torch.randn(batch_size)

        loss, metrics = loss_fn(policy_logits, value, target_policy, target_values)

        assert loss.item() > 0
        assert "loss/total" in metrics
        assert "loss/value" in metrics
        assert "loss/policy" in metrics


class TestReplay:
    """Tests for the replay buffer."""

    def test_create_empty_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            create_empty_db(db_path)
            assert db_path.exists()

            replay = ReplayBuffer(db_path)
            assert replay.count() == 0
            replay.close()

    def test_insert_and_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            create_empty_db(db_path)
            insert_test_transitions(db_path, count=100)

            with ReplayBuffer(db_path) as replay:
                assert replay.count() == 100

                # Sample batch
                transitions = replay.sample(10)
                assert len(transitions) == 10

                # All transitions should be from tictactoe
                assert all(t.env_id == "tictactoe" for t in transitions)

    def test_sample_batch_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            create_empty_db(db_path)
            insert_test_transitions(db_path, count=100)

            with ReplayBuffer(db_path) as replay:
                batch = replay.sample_batch_tensors(32)
                assert batch is not None

                obs, policy_targets, value_targets = batch
                assert obs.shape == (32, 29)
                assert policy_targets.shape == (32, 9)
                assert value_targets.shape == (32,)
                assert obs.dtype == np.float32
                assert policy_targets.dtype == np.float32
                assert value_targets.dtype == np.float32

    def test_sample_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            create_empty_db(db_path)
            insert_test_transitions(db_path, count=10)

            with ReplayBuffer(db_path) as replay:
                # Try to sample more than available
                batch = replay.sample_batch_tensors(100)
                assert batch is None

    def test_schema_validation_missing_table(self):
        """Test that missing transitions table raises SchemaError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "empty.db"
            # Create empty database without transitions table
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE other_table (id INTEGER)")
            conn.close()

            with pytest.raises(SchemaError, match="Table 'transitions' not found"):
                ReplayBuffer(db_path)

    def test_schema_validation_missing_column(self):
        """Test that missing columns raise SchemaError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "incomplete.db"
            # Create transitions table with missing columns
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE transitions (
                    id TEXT PRIMARY KEY,
                    env_id TEXT NOT NULL
                )
                """
            )
            conn.close()

            with pytest.raises(SchemaError, match="Missing columns"):
                ReplayBuffer(db_path)

    def test_schema_validation_disabled(self):
        """Test that schema validation can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "incomplete.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE transitions (id TEXT PRIMARY KEY)")
            conn.close()

            # Should not raise with validation disabled
            replay = ReplayBuffer(db_path, validate_schema=False)
            replay.close()


class TestTrainer:
    """Tests for the trainer."""

    def test_trainer_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "replay.db"
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            create_empty_db(db_path)

            config = TrainerConfig(
                db_path=str(db_path),
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=10,
            )

            trainer = Trainer(config)
            assert trainer.network is not None
            assert model_dir.exists()

    def test_trainer_short_run(self):
        """Run a tiny training loop to verify everything works end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "replay.db"
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            # Create test data
            create_empty_db(db_path)
            insert_test_transitions(db_path, count=200)

            config = TrainerConfig(
                db_path=str(db_path),
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=20,
                batch_size=32,
                checkpoint_interval=10,
                stats_interval=5,
                log_interval=5,
            )

            trainer = Trainer(config)
            stats = trainer.train()

            # Verify training completed
            assert stats.step == 20
            assert stats.total_loss > 0

            # Verify checkpoints were saved
            assert (model_dir / "latest.onnx").exists()
            assert (model_dir / "model_step_000010.onnx").exists()
            assert (model_dir / "model_step_000020.onnx").exists()

            # Verify stats.json was written
            assert stats_path.exists()
            with open(stats_path) as f:
                loaded_stats = json.load(f)
            assert loaded_stats["step"] == 20
            assert "history" in loaded_stats


class TestONNXExport:
    """Tests for ONNX export."""

    def test_onnx_export_valid(self):
        """Verify exported ONNX model is valid."""
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "replay.db"
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            create_empty_db(db_path)
            insert_test_transitions(db_path, count=100)

            config = TrainerConfig(
                db_path=str(db_path),
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=10,
                checkpoint_interval=10,
            )

            trainer = Trainer(config)
            trainer.train()

            # Load and validate ONNX model
            onnx_path = model_dir / "latest.onnx"
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)

            # Verify input/output names
            assert model.graph.input[0].name == "observation"
            assert model.graph.output[0].name == "policy_logits"
            assert model.graph.output[1].name == "value"


class TestCheckpointAtomicity:
    """Tests for atomic checkpoint saving."""

    def test_checkpoint_atomicity_no_temp_files_left(self):
        """Verify no temp files are left after successful checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "replay.db"
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            create_empty_db(db_path)
            insert_test_transitions(db_path, count=100)

            config = TrainerConfig(
                db_path=str(db_path),
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=10,
                checkpoint_interval=10,
            )

            trainer = Trainer(config)
            trainer.train()

            # Check no temp files remain in model_dir
            files = list(model_dir.iterdir())
            temp_files = [f for f in files if f.name.startswith("tmp") or ".tmp" in f.name]
            assert len(temp_files) == 0, f"Temp files found: {temp_files}"

            # Should have checkpoint and latest files
            assert (model_dir / "latest.onnx").exists()
            assert (model_dir / "model_step_000010.onnx").exists()

    def test_checkpoint_atomicity_cleanup_on_failure(self):
        """Verify temp files are cleaned up on export failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "replay.db"
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            create_empty_db(db_path)
            insert_test_transitions(db_path, count=100)

            config = TrainerConfig(
                db_path=str(db_path),
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=5,
                checkpoint_interval=5,
            )

            trainer = Trainer(config)

            # Mock torch.onnx.export to raise an exception
            with patch("torch.onnx.export", side_effect=RuntimeError("Mock export failure")):
                with pytest.raises(RuntimeError, match="Mock export failure"):
                    trainer._save_checkpoint(5)

            # Check no temp files remain after failure
            files = list(model_dir.iterdir()) if model_dir.exists() else []
            temp_files = [f for f in files if f.suffix == ".onnx"]
            assert len(temp_files) == 0, f"Temp files found after failure: {temp_files}"

    def test_latest_onnx_matches_checkpoint(self):
        """Verify latest.onnx has same content as step checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "replay.db"
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            create_empty_db(db_path)
            insert_test_transitions(db_path, count=100)

            config = TrainerConfig(
                db_path=str(db_path),
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=10,
                checkpoint_interval=10,
            )

            trainer = Trainer(config)
            trainer.train()

            # Check file sizes match (simple way to verify copy worked)
            latest_size = (model_dir / "latest.onnx").stat().st_size
            checkpoint_size = (model_dir / "model_step_000010.onnx").stat().st_size
            assert latest_size == checkpoint_size


class TestWaitTimeout:
    """Tests for wait/timeout behavior."""

    def test_wait_timeout_raises(self):
        """Test that WaitTimeout is raised after max_wait."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nonexistent.db"
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            config = TrainerConfig(
                db_path=str(db_path),
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=10,
                wait_interval=0.1,
                max_wait=0.3,  # Very short timeout
            )

            trainer = Trainer(config)

            with pytest.raises(WaitTimeout, match="Timed out"):
                trainer.train()


class TestHistoryBounding:
    """Tests for stats history bounding."""

    def test_history_bounded_on_append(self):
        """Verify history stays bounded after many appends."""
        from trainer.trainer import TrainerStats

        stats = TrainerStats()
        stats._max_history = 10

        # Append more than max
        for i in range(25):
            stats.append_history({"step": i, "loss": float(i)})

        # Should be bounded
        assert len(stats.history) == 10
        # Should have the most recent entries
        assert stats.history[0]["step"] == 15
        assert stats.history[-1]["step"] == 24


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_grad_clip_in_metrics(self):
        """Verify grad_norm is included in metrics when clipping enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "replay.db"
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            create_empty_db(db_path)
            insert_test_transitions(db_path, count=100)

            config = TrainerConfig(
                db_path=str(db_path),
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=1,
                grad_clip_norm=1.0,
            )

            trainer = Trainer(config)

            # Run a single train step manually
            with ReplayBuffer(db_path) as replay:
                batch = replay.sample_batch_tensors(32)
                obs, policy_targets, value_targets = batch
                metrics = trainer._train_step(obs, policy_targets, value_targets)

            assert "grad_norm" in metrics
            assert metrics["grad_norm"] >= 0


class TestLRScheduler:
    """Tests for learning rate scheduler."""

    def test_lr_decreases_with_scheduler(self):
        """Verify LR decreases over training with scheduler enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "replay.db"
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            create_empty_db(db_path)
            insert_test_transitions(db_path, count=200)

            config = TrainerConfig(
                db_path=str(db_path),
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=50,
                learning_rate=1e-3,
                lr_min_ratio=0.1,
                use_lr_scheduler=True,
                checkpoint_interval=100,  # No checkpoints
                stats_interval=10,
            )

            trainer = Trainer(config)
            initial_lr = trainer.optimizer.param_groups[0]["lr"]

            stats = trainer.train()

            final_lr = trainer.optimizer.param_groups[0]["lr"]

            # LR should have decreased
            assert final_lr < initial_lr
            # Final LR should be approximately lr * lr_min_ratio
            expected_min = config.learning_rate * config.lr_min_ratio
            assert abs(final_lr - expected_min) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
