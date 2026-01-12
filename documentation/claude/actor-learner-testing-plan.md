# Actor & Trainer Testing Plan

This document outlines gaps in test coverage and proposes high-value tests for the actor and trainer components.

## Current State

### Actor (Rust)
- **33 total tests** (29 passing, 4 ignored due to database requirements)
- Good coverage: config validation, game config, health state
- Weak coverage: storage backends, MCTS policy with models, core episode execution

### Trainer (Python)
- **~15 tests** across 4 test files
- Good coverage: stats serialization, network forward pass, loss function
- Weak coverage: training loop, evaluator, checkpoints, LR scheduler, orchestrator

---

## Actor Test Proposals

### 1. PostgreSQL Storage Backend Tests
**File:** `actor/src/storage/postgres.rs`
**Priority:** High
**Current Coverage:** 0%

The PostgreSQL storage backend handles all transition persistence and has zero test coverage.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_insert_generates_correct_sql() {
        // Test that batch_insert creates valid SQL for various batch sizes
        // Mock the connection pool and verify the generated query
    }

    #[tokio::test]
    async fn test_store_metadata_upserts_correctly() {
        // Test metadata storage with insert and update scenarios
        // Verify ON CONFLICT behavior
    }

    #[tokio::test]
    async fn test_get_transition_count_returns_correct_value() {
        // Insert known number of transitions, verify count
    }

    #[tokio::test]
    async fn test_clear_removes_all_transitions() {
        // Insert transitions, clear, verify count is zero
    }

    #[tokio::test]
    async fn test_concurrent_inserts_do_not_deadlock() {
        // Spawn multiple tasks inserting simultaneously
        // Verify all complete without deadlock
    }
}
```

**Why:** Storage is critical path. Bugs here corrupt training data silently.

---

### 2. MCTS Policy with Model Evaluation
**File:** `actor/src/mcts_policy.rs`
**Priority:** High
**Current Coverage:** ~40% (only random fallback tested)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_action_with_mock_evaluator() {
        // Create MctsPolicy with mock evaluator that returns known priors
        // Verify action selection follows MCTS with those priors
    }

    #[test]
    fn test_temperature_scheduling_before_threshold() {
        // Move number < temperature_threshold
        // Verify temperature equals configured training temperature
    }

    #[test]
    fn test_temperature_scheduling_after_threshold() {
        // Move number >= temperature_threshold
        // Verify temperature drops to 0.0 (greedy)
    }

    #[test]
    fn test_action_byte_encoding_tictactoe() {
        // Verify action 4 encodes to [4, 0, 0, 0] (little-endian u32)
    }

    #[test]
    fn test_policy_distribution_sums_to_one() {
        // Run MCTS, verify policy distribution sums to ~1.0
    }

    #[test]
    fn test_mcts_stats_recorded() {
        // Run select_action, verify stats.mcts_searches and mcts_avg_inference updated
    }
}
```

**Why:** MCTS policy is core to self-play quality. Temperature bugs cause exploration/exploitation imbalance.

---

### 3. Actor Episode Execution (Mock Storage)
**File:** `actor/src/actor.rs`
**Priority:** High
**Current Coverage:** 4 tests (all ignored)

Create a mock `ReplayStore` to test without database:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct MockReplayStore {
        transitions: Arc<Mutex<Vec<Transition>>>,
    }

    #[async_trait]
    impl ReplayStore for MockReplayStore {
        async fn store_batch(&self, batch: Vec<Transition>) -> Result<()> {
            self.transitions.lock().unwrap().extend(batch);
            Ok(())
        }
        // ... implement other methods
    }

    #[tokio::test]
    async fn test_episode_completes_and_stores_transitions() {
        let store = MockReplayStore::new();
        let actor = Actor::new(config, store, policy);

        actor.run_single_episode().await.unwrap();

        let transitions = store.transitions.lock().unwrap();
        assert!(!transitions.is_empty());
        // Verify all transitions have valid state/action/reward
    }

    #[tokio::test]
    async fn test_episode_respects_max_steps() {
        // Configure max_steps = 10
        // Run episode on game that could go longer
        // Verify episode terminates at max_steps
    }

    #[tokio::test]
    async fn test_game_outcome_backfilled_correctly() {
        let store = MockReplayStore::new();
        // Run complete episode
        // Verify all transitions have reward set based on game outcome
        // Check alternating sign for two-player games
    }

    #[tokio::test]
    async fn test_legal_mask_propagated_through_steps() {
        // Run episode, verify each transition has valid legal_mask
        // Check mask matches actual legal moves at that state
    }

    #[tokio::test]
    async fn test_episode_timeout_triggers_cleanup() {
        // Configure short episode_timeout
        // Mock policy that delays
        // Verify episode terminates and logs timeout
    }
}
```

**Why:** Episode execution is the actor's primary function. Testing without DB enables faster CI.

---

### 4. Stats Module Edge Cases
**File:** `actor/src/stats.rs`
**Priority:** Medium
**Current Coverage:** ~30%

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_average_with_zero_episodes() {
        let stats = Stats::new();
        // Verify averages don't divide by zero, return 0.0 or NaN handled
    }

    #[test]
    fn test_rate_with_zero_runtime() {
        let stats = Stats::new();
        // Record episode with zero elapsed time
        // Verify rate calculation handles gracefully
    }

    #[test]
    fn test_outcome_categorization_boundaries() {
        // reward = 0.0 -> draw
        // reward = 0.5 -> win (boundary)
        // reward = -0.5 -> loss (boundary)
        // Verify categorization at exact boundaries
    }

    #[test]
    fn test_mcts_average_with_zero_searches() {
        let stats = Stats::new();
        // Verify mcts_avg_inference returns 0.0 when no searches recorded
    }
}
```

---

### 5. Health Module Integration
**File:** `actor/src/health.rs`
**Priority:** Medium
**Current Coverage:** ~60%

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_progress_timeout_detection() {
        let health = HealthState::new();
        health.record_progress();

        // Simulate time passage beyond timeout
        std::thread::sleep(Duration::from_secs(PROGRESS_TIMEOUT + 1));

        assert!(health.is_progress_stalled());
    }

    #[tokio::test]
    async fn test_health_handler_returns_unhealthy_on_stall() {
        // Set up health state with stalled progress
        // Call health handler
        // Verify returns 503 or appropriate error status
    }
}
```

---

## Trainer Test Proposals

### 1. Training Loop Integration
**File:** `trainer/tests/test_trainer.py`
**Priority:** Critical
**Current Coverage:** 0%

```python
import pytest
from unittest.mock import Mock, patch
from trainer.trainer import Trainer
from trainer.config import TrainerConfig

class TestTrainer:
    @pytest.fixture
    def mock_replay_buffer(self):
        """Create mock replay buffer with sample transitions."""
        buffer = Mock()
        buffer.sample_batch.return_value = (
            torch.randn(32, 27),  # states (batch_size, input_dim)
            torch.randn(32, 9),   # policies
            torch.randn(32, 1),   # values
            torch.ones(32, 9),    # legal_masks
        )
        buffer.__len__ = Mock(return_value=1000)
        return buffer

    def test_train_step_updates_weights(self, mock_replay_buffer):
        """Verify a single training step updates model weights."""
        config = TrainerConfig(batch_size=32, learning_rate=0.001)
        trainer = Trainer(config, mock_replay_buffer)

        initial_weights = trainer.network.fc1.weight.clone()
        trainer._train_step()

        assert not torch.equal(initial_weights, trainer.network.fc1.weight)

    def test_train_records_loss_history(self, mock_replay_buffer):
        """Verify training records loss values."""
        trainer = Trainer(config, mock_replay_buffer)
        trainer.train(steps=10)

        assert len(trainer.stats.loss_history) == 10
        assert all(loss > 0 for loss in trainer.stats.loss_history)

    def test_train_respects_step_limit(self, mock_replay_buffer):
        """Verify training stops at specified step count."""
        trainer = Trainer(config, mock_replay_buffer)
        trainer.train(steps=50)

        assert trainer.current_step == 50

    def test_train_with_empty_buffer_waits(self, mock_replay_buffer):
        """Verify trainer waits when buffer is empty."""
        mock_replay_buffer.__len__ = Mock(return_value=0)

        with patch('time.sleep') as mock_sleep:
            trainer = Trainer(config, mock_replay_buffer)
            trainer.train(steps=1, timeout=5)
            mock_sleep.assert_called()
```

**Why:** The training loop is the trainer's core function. Zero test coverage is unacceptable.

---

### 2. Checkpoint Save/Load
**File:** `trainer/tests/test_checkpoint.py`
**Priority:** Critical
**Current Coverage:** 0%

```python
import pytest
import torch
from pathlib import Path
from trainer.checkpoint import (
    save_onnx_checkpoint,
    save_pytorch_checkpoint,
    load_pytorch_checkpoint,
    cleanup_old_checkpoints,
)

class TestCheckpoint:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path / "checkpoints"

    def test_onnx_export_creates_valid_file(self, temp_dir, sample_network):
        """Verify ONNX export creates loadable model."""
        path = temp_dir / "model.onnx"
        save_onnx_checkpoint(sample_network, path, input_dim=27)

        assert path.exists()
        # Verify ONNX model is loadable
        import onnxruntime
        session = onnxruntime.InferenceSession(str(path))
        assert session is not None

    def test_onnx_export_atomic_write(self, temp_dir, sample_network):
        """Verify ONNX export uses atomic write-then-rename."""
        path = temp_dir / "model.onnx"
        # Write initial model
        save_onnx_checkpoint(sample_network, path, input_dim=27)
        initial_content = path.read_bytes()

        # Modify network and re-export
        sample_network.fc1.weight.data.fill_(0)
        save_onnx_checkpoint(sample_network, path, input_dim=27)

        # Verify file changed (atomic update worked)
        assert path.read_bytes() != initial_content

    def test_pytorch_checkpoint_roundtrip(self, temp_dir, sample_network):
        """Verify PyTorch checkpoint save/load preserves state."""
        path = temp_dir / "checkpoint.pt"
        optimizer = torch.optim.Adam(sample_network.parameters())

        # Modify optimizer state
        for _ in range(10):
            loss = sample_network(torch.randn(1, 27)).sum()
            loss.backward()
            optimizer.step()

        save_pytorch_checkpoint(sample_network, optimizer, path, step=100)

        # Create fresh network and optimizer
        new_network = Network(input_dim=27, hidden_dim=128, output_dim=9)
        new_optimizer = torch.optim.Adam(new_network.parameters())

        step = load_pytorch_checkpoint(new_network, new_optimizer, path)

        assert step == 100
        assert torch.equal(
            sample_network.fc1.weight,
            new_network.fc1.weight
        )

    def test_cleanup_keeps_n_most_recent(self, temp_dir):
        """Verify cleanup removes old checkpoints, keeps N newest."""
        # Create 5 checkpoints with different timestamps
        for i in range(5):
            (temp_dir / f"checkpoint_{i}.pt").touch()

        cleanup_old_checkpoints(temp_dir, max_keep=2)

        remaining = list(temp_dir.glob("*.pt"))
        assert len(remaining) == 2

    def test_load_corrupted_checkpoint_raises(self, temp_dir):
        """Verify loading corrupted checkpoint raises clear error."""
        path = temp_dir / "corrupted.pt"
        path.write_bytes(b"not a valid checkpoint")

        with pytest.raises(Exception):
            load_pytorch_checkpoint(Network(), None, path)
```

**Why:** Checkpoint corruption can lose days of training. Must be bulletproof.

---

### 3. Evaluator Tests
**File:** `trainer/tests/test_evaluator.py`
**Priority:** High
**Current Coverage:** 0%

```python
import pytest
from unittest.mock import Mock
from trainer.evaluator import evaluate, EvalResults

class TestEvaluator:
    @pytest.fixture
    def mock_game(self):
        """Create mock game that plays deterministically."""
        game = Mock()
        game.reset.return_value = (initial_state, initial_obs)
        game.step.side_effect = self._deterministic_game_steps
        return game

    def test_evaluate_returns_valid_results(self, mock_game):
        """Verify evaluation returns sensible win/draw/loss rates."""
        policy = Mock()
        policy.select_action.return_value = 0

        results = evaluate(mock_game, policy, games=100)

        assert isinstance(results, EvalResults)
        assert results.win_rate + results.draw_rate + results.loss_rate == 1.0
        assert results.games == 100

    def test_evaluate_swaps_colors(self, mock_game):
        """Verify evaluation plays as both first and second player."""
        policy = Mock()
        call_count = {'first': 0, 'second': 0}

        results = evaluate(mock_game, policy, games=10)

        # Should play ~5 games as each color
        assert 3 <= call_count['first'] <= 7
        assert 3 <= call_count['second'] <= 7

    def test_eval_results_properties(self):
        """Verify EvalResults computed properties."""
        results = EvalResults(wins=30, draws=20, losses=50)

        assert results.win_rate == 0.30
        assert results.draw_rate == 0.20
        assert results.loss_rate == 0.50
        assert results.games == 100
```

---

### 4. Learning Rate Scheduler
**File:** `trainer/tests/test_lr_scheduler.py`
**Priority:** High
**Current Coverage:** 0%

```python
import pytest
import torch
from trainer.lr_scheduler import WarmupCosineScheduler

class TestLRScheduler:
    def test_warmup_phase_linear_increase(self):
        """Verify LR increases linearly during warmup."""
        optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.001)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0001
        )

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should increase linearly
        assert lrs[0] < lrs[50] < lrs[99]
        # Final warmup LR should be near base LR
        assert abs(lrs[99] - 0.001) < 0.0001

    def test_cosine_phase_decreases(self):
        """Verify LR decreases via cosine annealing after warmup."""
        optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.001)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=0,
            total_steps=100,
            min_lr=0.0001
        )

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should decrease
        assert lrs[0] > lrs[50] > lrs[99]
        # Final LR should be near min_lr
        assert abs(lrs[99] - 0.0001) < 0.0001

    def test_state_dict_roundtrip(self):
        """Verify scheduler state saves and loads correctly."""
        optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.001)
        scheduler = WarmupCosineScheduler(optimizer, 100, 1000, 0.0001)

        # Advance scheduler
        for _ in range(250):
            scheduler.step()

        state = scheduler.state_dict()

        # Create new scheduler and load state
        new_scheduler = WarmupCosineScheduler(optimizer, 100, 1000, 0.0001)
        new_scheduler.load_state_dict(state)

        assert new_scheduler.last_epoch == scheduler.last_epoch
        assert new_scheduler.get_last_lr() == scheduler.get_last_lr()
```

---

### 5. Backoff Utility Tests
**File:** `trainer/tests/test_backoff.py`
**Priority:** Medium
**Current Coverage:** 0%

```python
import pytest
from unittest.mock import patch
from trainer.backoff import wait_with_backoff, WaitTimeout

class TestBackoff:
    def test_immediate_success(self):
        """Verify returns immediately when condition is true."""
        result = wait_with_backoff(
            condition=lambda: True,
            max_wait=10
        )
        assert result is True

    def test_timeout_raises_exception(self):
        """Verify WaitTimeout raised when max_wait exceeded."""
        with pytest.raises(WaitTimeout):
            wait_with_backoff(
                condition=lambda: False,
                max_wait=0.1  # 100ms timeout
            )

    def test_exponential_backoff_intervals(self):
        """Verify sleep intervals increase exponentially."""
        sleep_times = []

        with patch('time.sleep', side_effect=lambda t: sleep_times.append(t)):
            call_count = [0]
            def condition():
                call_count[0] += 1
                return call_count[0] >= 5

            wait_with_backoff(condition, max_wait=60)

        # Each interval should be roughly double the previous
        for i in range(1, len(sleep_times)):
            assert sleep_times[i] >= sleep_times[i-1] * 1.5

    def test_backoff_caps_at_max_interval(self):
        """Verify backoff doesn't exceed MAX_BACKOFF_INTERVAL."""
        sleep_times = []

        with patch('time.sleep', side_effect=lambda t: sleep_times.append(t)):
            wait_with_backoff(
                condition=lambda: len(sleep_times) >= 10,
                max_wait=300
            )

        # No sleep should exceed max (typically 30-60s)
        assert all(t <= 60 for t in sleep_times)
```

---

### 6. Storage Backend Tests
**File:** `trainer/tests/test_storage.py`
**Priority:** Medium
**Current Coverage:** ~10% (factory only)

```python
import pytest
from pathlib import Path
from trainer.storage.filesystem import FilesystemModelStore
from trainer.storage.postgres import PostgresReplayBuffer

class TestFilesystemModelStore:
    def test_save_and_load_model(self, tmp_path):
        """Verify model save/load roundtrip."""
        store = FilesystemModelStore(tmp_path)
        model_bytes = b"fake onnx model data"

        store.save("latest.onnx", model_bytes)
        loaded = store.load("latest.onnx")

        assert loaded == model_bytes

    def test_list_models(self, tmp_path):
        """Verify listing available models."""
        store = FilesystemModelStore(tmp_path)
        store.save("model_1.onnx", b"data1")
        store.save("model_2.onnx", b"data2")

        models = store.list_models()

        assert "model_1.onnx" in models
        assert "model_2.onnx" in models

    def test_load_nonexistent_raises(self, tmp_path):
        """Verify loading missing model raises FileNotFoundError."""
        store = FilesystemModelStore(tmp_path)

        with pytest.raises(FileNotFoundError):
            store.load("nonexistent.onnx")


class TestPostgresReplayBuffer:
    @pytest.fixture
    def db_url(self):
        """Return test database URL (skip if not available)."""
        import os
        url = os.getenv("TEST_POSTGRES_URL")
        if not url:
            pytest.skip("TEST_POSTGRES_URL not set")
        return url

    def test_store_and_sample_transitions(self, db_url):
        """Verify storing and sampling transitions."""
        buffer = PostgresReplayBuffer(db_url)

        # Store transitions
        transitions = [
            {"state": b"s1", "policy": [0.5]*9, "value": 1.0},
            {"state": b"s2", "policy": [0.5]*9, "value": -1.0},
        ]
        buffer.store_batch(transitions)

        # Sample should return data
        batch = buffer.sample_batch(batch_size=2)
        assert len(batch) == 2

    def test_clear_removes_all(self, db_url):
        """Verify clear empties the buffer."""
        buffer = PostgresReplayBuffer(db_url)
        buffer.store_batch([{"state": b"s1", "policy": [0.5]*9, "value": 1.0}])

        buffer.clear()

        assert len(buffer) == 0
```

---

## Implementation Recommendations

### Phase 1: Critical Path (Week 1)
1. Trainer training loop tests (mock replay buffer)
2. Checkpoint save/load tests
3. Actor episode execution with mock storage

### Phase 2: Core Components (Week 2)
4. Evaluator tests
5. LR scheduler tests
6. MCTS policy tests with mock evaluator
7. PostgreSQL storage tests (requires test DB)

### Phase 3: Edge Cases (Week 3)
8. Stats edge case tests (both actor and trainer)
9. Backoff utility tests
10. Health module integration tests

### Test Infrastructure Needed
- **Actor:** Mock `ReplayStore` trait implementation
- **Actor:** Mock `Evaluator` for MCTS tests
- **Trainer:** Pytest fixtures for common setups
- **Trainer:** Test database for PostgreSQL tests (can use testcontainers)
- **CI:** Separate test stage for database-dependent tests

### Running Tests

```bash
# Actor (Rust)
cd actor && cargo test

# Trainer (Python)
cd trainer && python -m pytest tests/ -v

# With database tests (requires PostgreSQL)
TEST_POSTGRES_URL=postgresql://test:test@localhost/test python -m pytest tests/ -v
```
