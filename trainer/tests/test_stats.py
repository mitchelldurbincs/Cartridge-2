"""Tests for stats module.

Tests cover:
- History bounds (TrainerStats.append_history, append_eval)
- Serialization round-trip (to_dict/from_dict)
- Atomic writes (write_stats)
- File loading (load_stats)
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from trainer.stats import (
    DEFAULT_MAX_EVAL_HISTORY,
    DEFAULT_MAX_HISTORY,
    EvalStats,
    TrainerStats,
    load_stats,
    write_stats,
)


class TestEvalStats:
    """Tests for EvalStats dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        stats = EvalStats()
        assert stats.step == 0
        assert stats.win_rate == 0.0
        assert stats.draw_rate == 0.0
        assert stats.loss_rate == 0.0
        assert stats.games_played == 0
        assert stats.avg_game_length == 0.0
        assert stats.timestamp > 0  # Should have current time

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = EvalStats(
            step=100,
            win_rate=0.75,
            draw_rate=0.15,
            loss_rate=0.10,
            games_played=50,
            avg_game_length=8.5,
            timestamp=1234567890.0,
        )

        d = stats.to_dict()

        assert d["step"] == 100
        assert d["win_rate"] == 0.75
        assert d["draw_rate"] == 0.15
        assert d["loss_rate"] == 0.10
        assert d["games_played"] == 50
        assert d["avg_game_length"] == 8.5
        assert d["timestamp"] == 1234567890.0

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "step": 200,
            "win_rate": 0.80,
            "draw_rate": 0.10,
            "loss_rate": 0.10,
            "games_played": 100,
            "avg_game_length": 9.2,
            "timestamp": 9876543210.0,
        }

        stats = EvalStats.from_dict(data)

        assert stats.step == 200
        assert stats.win_rate == 0.80
        assert stats.draw_rate == 0.10
        assert stats.loss_rate == 0.10
        assert stats.games_played == 100
        assert stats.avg_game_length == 9.2
        assert stats.timestamp == 9876543210.0

    def test_from_dict_missing_fields(self):
        """Test deserialization with missing fields uses defaults."""
        stats = EvalStats.from_dict({})

        assert stats.step == 0
        assert stats.win_rate == 0.0
        assert stats.games_played == 0

    def test_round_trip(self):
        """Test serialization round-trip preserves data."""
        original = EvalStats(
            step=500,
            win_rate=0.65,
            draw_rate=0.20,
            loss_rate=0.15,
            games_played=200,
            avg_game_length=7.3,
            timestamp=1111111111.0,
        )

        restored = EvalStats.from_dict(original.to_dict())

        assert restored.step == original.step
        assert restored.win_rate == original.win_rate
        assert restored.draw_rate == original.draw_rate
        assert restored.loss_rate == original.loss_rate
        assert restored.games_played == original.games_played
        assert restored.avg_game_length == original.avg_game_length
        assert restored.timestamp == original.timestamp


class TestTrainerStats:
    """Tests for TrainerStats dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        stats = TrainerStats()
        assert stats.step == 0
        assert stats.total_steps == 0
        assert stats.total_loss == 0.0
        assert stats.value_loss == 0.0
        assert stats.policy_loss == 0.0
        assert stats.learning_rate == 0.0
        assert stats.samples_seen == 0
        assert stats.replay_buffer_size == 0
        assert stats.last_checkpoint == ""
        assert stats.timestamp > 0
        assert stats.history == []
        assert stats.env_id == ""
        assert stats.last_eval is None
        assert stats.eval_history == []
        assert stats._max_history == DEFAULT_MAX_HISTORY
        assert stats._max_eval_history == DEFAULT_MAX_EVAL_HISTORY

    def test_append_history(self):
        """Test appending to history."""
        stats = TrainerStats()
        stats.append_history({"step": 1, "loss": 0.5})
        stats.append_history({"step": 2, "loss": 0.4})

        assert len(stats.history) == 2
        assert stats.history[0]["step"] == 1
        assert stats.history[1]["step"] == 2

    def test_history_bounded_on_append(self):
        """Test that history stays bounded after many appends."""
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

    def test_history_bounded_with_default_max(self):
        """Test history is bounded at default max."""
        stats = TrainerStats()

        # Append more than default max
        for i in range(DEFAULT_MAX_HISTORY + 50):
            stats.append_history({"step": i})

        assert len(stats.history) == DEFAULT_MAX_HISTORY
        assert stats.history[-1]["step"] == DEFAULT_MAX_HISTORY + 49

    def test_append_eval(self):
        """Test appending evaluation results."""
        stats = TrainerStats()
        eval1 = EvalStats(step=100, win_rate=0.5)
        eval2 = EvalStats(step=200, win_rate=0.6)

        stats.append_eval(eval1)
        assert stats.last_eval == eval1
        assert len(stats.eval_history) == 1

        stats.append_eval(eval2)
        assert stats.last_eval == eval2
        assert len(stats.eval_history) == 2

    def test_eval_history_bounded(self):
        """Test that eval_history stays bounded."""
        stats = TrainerStats()
        stats._max_eval_history = 5

        for i in range(10):
            stats.append_eval(EvalStats(step=i * 100, win_rate=i * 0.1))

        assert len(stats.eval_history) == 5
        # Should have most recent entries
        assert stats.eval_history[0]["step"] == 500
        assert stats.eval_history[-1]["step"] == 900

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = TrainerStats(
            step=100,
            total_steps=1000,
            total_loss=0.5,
            value_loss=0.2,
            policy_loss=0.3,
            learning_rate=0.001,
            samples_seen=5000,
            replay_buffer_size=10000,
            last_checkpoint="model_100.onnx",
            timestamp=1234567890.0,
            env_id="tictactoe",
        )
        stats.append_history({"step": 100, "loss": 0.5})
        stats.append_eval(EvalStats(step=100, win_rate=0.7, timestamp=1234567890.0))

        d = stats.to_dict()

        assert d["step"] == 100
        assert d["total_steps"] == 1000
        assert d["total_loss"] == 0.5
        assert d["value_loss"] == 0.2
        assert d["policy_loss"] == 0.3
        assert d["learning_rate"] == 0.001
        assert d["samples_seen"] == 5000
        assert d["replay_buffer_size"] == 10000
        assert d["last_checkpoint"] == "model_100.onnx"
        assert d["timestamp"] == 1234567890.0
        assert d["env_id"] == "tictactoe"
        assert len(d["history"]) == 1
        assert d["last_eval"]["step"] == 100
        assert d["last_eval"]["win_rate"] == 0.7
        assert len(d["eval_history"]) == 1

    def test_to_dict_no_eval(self):
        """Test serialization with no eval data."""
        stats = TrainerStats()
        d = stats.to_dict()

        assert d["last_eval"] is None
        assert d["eval_history"] == []

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "step": 200,
            "total_steps": 500,
            "total_loss": 0.3,
            "value_loss": 0.1,
            "policy_loss": 0.2,
            "learning_rate": 0.0005,
            "samples_seen": 10000,
            "replay_buffer_size": 20000,
            "last_checkpoint": "model_200.onnx",
            "timestamp": 9876543210.0,
            "history": [{"step": 100}, {"step": 200}],
            "env_id": "connect4",
            "last_eval": {"step": 200, "win_rate": 0.8, "timestamp": 9876543210.0},
            "eval_history": [{"step": 100, "win_rate": 0.6}, {"step": 200, "win_rate": 0.8}],
        }

        stats = TrainerStats.from_dict(data)

        assert stats.step == 200
        assert stats.total_steps == 500
        assert stats.total_loss == 0.3
        assert stats.value_loss == 0.1
        assert stats.policy_loss == 0.2
        assert stats.learning_rate == 0.0005
        assert stats.samples_seen == 10000
        assert stats.replay_buffer_size == 20000
        assert stats.last_checkpoint == "model_200.onnx"
        assert stats.timestamp == 9876543210.0
        assert len(stats.history) == 2
        assert stats.env_id == "connect4"
        assert stats.last_eval is not None
        assert stats.last_eval.step == 200
        assert stats.last_eval.win_rate == 0.8
        assert len(stats.eval_history) == 2

    def test_from_dict_missing_fields(self):
        """Test deserialization with missing fields uses defaults."""
        stats = TrainerStats.from_dict({})

        assert stats.step == 0
        assert stats.total_steps == 0
        assert stats.history == []
        assert stats.last_eval is None

    def test_round_trip(self):
        """Test serialization round-trip preserves data."""
        original = TrainerStats(
            step=300,
            total_steps=1000,
            total_loss=0.4,
            value_loss=0.15,
            policy_loss=0.25,
            learning_rate=0.001,
            samples_seen=15000,
            replay_buffer_size=30000,
            last_checkpoint="model_300.onnx",
            timestamp=5555555555.0,
            env_id="tictactoe",
        )
        original.append_history({"step": 300, "loss": 0.4})
        original.append_eval(EvalStats(step=300, win_rate=0.75, timestamp=5555555555.0))

        restored = TrainerStats.from_dict(original.to_dict())

        assert restored.step == original.step
        assert restored.total_steps == original.total_steps
        assert restored.total_loss == original.total_loss
        assert restored.value_loss == original.value_loss
        assert restored.policy_loss == original.policy_loss
        assert restored.learning_rate == original.learning_rate
        assert restored.samples_seen == original.samples_seen
        assert restored.replay_buffer_size == original.replay_buffer_size
        assert restored.last_checkpoint == original.last_checkpoint
        assert restored.timestamp == original.timestamp
        assert restored.env_id == original.env_id
        assert len(restored.history) == len(original.history)
        assert restored.last_eval.step == original.last_eval.step
        assert restored.last_eval.win_rate == original.last_eval.win_rate


class TestLoadStats:
    """Tests for load_stats function."""

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file returns empty stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            stats = load_stats(path)

            assert isinstance(stats, TrainerStats)
            assert stats.step == 0
            assert stats.history == []

    def test_load_existing_file(self):
        """Test loading from existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            data = {
                "step": 100,
                "total_steps": 1000,
                "total_loss": 0.5,
                "history": [{"step": 50}, {"step": 100}],
                "env_id": "tictactoe",
                "last_eval": {"step": 100, "win_rate": 0.7},
                "eval_history": [{"step": 100, "win_rate": 0.7}],
            }
            with open(path, "w") as f:
                json.dump(data, f)

            stats = load_stats(path)

            assert stats.step == 100
            assert stats.total_steps == 1000
            assert len(stats.history) == 2
            assert stats.env_id == "tictactoe"
            assert stats.last_eval is not None
            assert stats.last_eval.win_rate == 0.7

    def test_load_invalid_json(self):
        """Test loading invalid JSON returns empty stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            with open(path, "w") as f:
                f.write("not valid json {{{")

            stats = load_stats(path)

            assert isinstance(stats, TrainerStats)
            assert stats.step == 0

    def test_load_accepts_string_path(self):
        """Test load_stats accepts string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "stats.json")
            stats = load_stats(path)
            assert isinstance(stats, TrainerStats)


class TestWriteStats:
    """Tests for write_stats function."""

    def test_write_creates_file(self):
        """Test writing stats creates the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            stats = TrainerStats(step=100, total_loss=0.5)

            write_stats(stats, path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["step"] == 100
            assert data["total_loss"] == 0.5

    def test_write_creates_parent_directories(self):
        """Test writing stats creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "stats.json"
            stats = TrainerStats(step=50)

            write_stats(stats, path)

            assert path.exists()

    def test_write_overwrites_existing(self):
        """Test writing stats overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"

            # Write first version
            stats1 = TrainerStats(step=100)
            write_stats(stats1, path)

            # Write second version
            stats2 = TrainerStats(step=200)
            write_stats(stats2, path)

            with open(path) as f:
                data = json.load(f)
            assert data["step"] == 200

    def test_write_is_atomic_no_temp_files(self):
        """Test that no temp files are left after successful write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            stats = TrainerStats(step=100)

            write_stats(stats, path)

            # Check no temp files remain
            files = list(Path(tmpdir).iterdir())
            temp_files = [f for f in files if f.name.startswith("tmp") or ".tmp" in f.name]
            assert len(temp_files) == 0

    def test_write_cleans_up_on_failure(self):
        """Test that temp files are cleaned up on write failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            stats = TrainerStats(step=100)

            # Mock json.dump to fail
            with patch("json.dump", side_effect=RuntimeError("Mock dump failure")):
                with pytest.raises(RuntimeError, match="Mock dump failure"):
                    write_stats(stats, path)

            # Check no temp files remain
            files = list(Path(tmpdir).iterdir())
            assert len(files) == 0

    def test_write_accepts_string_path(self):
        """Test write_stats accepts string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "stats.json")
            stats = TrainerStats(step=100)

            write_stats(stats, path)

            assert Path(path).exists()

    def test_write_round_trip(self):
        """Test full write and load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"

            original = TrainerStats(
                step=500,
                total_steps=1000,
                total_loss=0.3,
                value_loss=0.1,
                policy_loss=0.2,
                learning_rate=0.001,
                samples_seen=25000,
                replay_buffer_size=50000,
                last_checkpoint="model_500.onnx",
                env_id="connect4",
            )
            original.append_history({"step": 500, "loss": 0.3})
            original.append_eval(EvalStats(step=500, win_rate=0.85))

            write_stats(original, path)
            loaded = load_stats(path)

            assert loaded.step == original.step
            assert loaded.total_steps == original.total_steps
            assert loaded.total_loss == original.total_loss
            assert loaded.env_id == original.env_id
            assert len(loaded.history) == len(original.history)
            assert loaded.last_eval.win_rate == original.last_eval.win_rate


class TestAtomicWriteSimulation:
    """Tests simulating concurrent access to verify atomicity."""

    def test_concurrent_reads_during_write(self):
        """Test that reads don't see partial writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"

            # Write initial version
            initial = TrainerStats(step=0)
            write_stats(initial, path)

            read_results = []
            errors = []

            def reader():
                """Reader thread that continuously reads the file."""
                for _ in range(50):
                    try:
                        if path.exists():
                            with open(path) as f:
                                data = json.load(f)
                                read_results.append(data["step"])
                    except json.JSONDecodeError as e:
                        errors.append(str(e))
                    time.sleep(0.001)

            def writer():
                """Writer thread that continuously updates the file."""
                for i in range(1, 51):
                    stats = TrainerStats(step=i * 10)
                    write_stats(stats, path)
                    time.sleep(0.001)

            reader_thread = threading.Thread(target=reader)
            writer_thread = threading.Thread(target=writer)

            reader_thread.start()
            writer_thread.start()

            reader_thread.join()
            writer_thread.join()

            # Should have no JSON decode errors (would indicate partial writes)
            assert len(errors) == 0, f"JSON errors during concurrent access: {errors}"

            # All read values should be valid step values (multiples of 10 or 0)
            for step in read_results:
                assert step % 10 == 0, f"Invalid step value: {step}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
