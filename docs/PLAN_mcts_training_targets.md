# Implementation Plan: MCTS-Based Training Targets

## Summary

Upgrade the trainer to use proper AlphaZero supervision:
- **Policy targets**: MCTS visit count distributions (already stored by actor)
- **Value targets**: Final game outcomes propagated to all positions (needs implementation)

## Current State

### Actor (already done)
The actor has been updated to:
1. Run MCTS at each position (`mcts_policy.rs`)
2. Store `policy_probs` (MCTS visit distribution) in transitions
3. Store `mcts_value` (MCTS value estimate) in transitions

Schema in `actor/src/replay.rs`:
```sql
policy_probs BLOB,           -- f32[9] MCTS visit distribution (already populated)
mcts_value REAL DEFAULT 0.0, -- MCTS value estimate (already populated)
```

### Trainer (needs update)
The trainer currently:
1. Ignores `policy_probs` column - uses one-hot of action taken
2. Ignores `mcts_value` column - uses per-step `reward`
3. Has outdated schema validation (missing new columns)

## Implementation Tasks

### Phase 1: Trainer Updates (Python)

#### Task 1.1: Update schema validation
**File**: `trainer/src/trainer/replay.py`

Add new columns to `EXPECTED_COLUMNS`:
```python
EXPECTED_COLUMNS = [
    # ... existing columns ...
    ("policy_probs", "BLOB"),
    ("mcts_value", "REAL"),
]
```

#### Task 1.2: Update Transition dataclass
**File**: `trainer/src/trainer/replay.py`

```python
@dataclass
class Transition:
    # ... existing fields ...
    policy_probs: bytes    # f32[num_actions] MCTS visit distribution
    mcts_value: float      # MCTS value estimate at this position
```

#### Task 1.3: Update sample query
**File**: `trainer/src/trainer/replay.py`

Update `sample()` to fetch new columns:
```python
SELECT id, env_id, episode_id, step_number, state, action, next_state,
       observation, next_observation, reward, done, timestamp,
       policy_probs, mcts_value
FROM transitions
ORDER BY RANDOM()
LIMIT ?
```

#### Task 1.4: Update sample_batch_tensors
**File**: `trainer/src/trainer/replay.py`

Change from one-hot actions to MCTS policy:
```python
def sample_batch_tensors(self, batch_size: int) -> tuple[...] | None:
    # ...
    for t in transitions:
        # Parse observation (unchanged)
        obs = np.frombuffer(t.observation, dtype=np.float32)
        observations.append(obs)

        # NEW: Use MCTS policy distribution as target
        if t.policy_probs:
            policy = np.frombuffer(t.policy_probs, dtype=np.float32)
        else:
            # Fallback to one-hot if no MCTS data (backward compat)
            action_idx = int.from_bytes(t.action, byteorder="little")
            policy = np.zeros(9, dtype=np.float32)
            policy[action_idx] = 1.0
        policies.append(policy)

        # Value target (see Phase 2 for game outcome propagation)
        values.append(t.mcts_value)

    return (
        np.array(observations, dtype=np.float32),
        np.array(policies, dtype=np.float32),  # Changed from int64 actions
        np.array(values, dtype=np.float32),
    )
```

#### Task 1.5: Update loss function
**File**: `trainer/src/trainer/network.py`

Change policy loss from cross-entropy with class labels to KL divergence with soft targets:
```python
class AlphaZeroLoss(nn.Module):
    def forward(
        self,
        policy_logits: torch.Tensor,   # (batch, num_actions)
        value_pred: torch.Tensor,      # (batch, 1)
        policy_target: torch.Tensor,   # (batch, num_actions) - MCTS distribution
        value_target: torch.Tensor,    # (batch,) - game outcome
        legal_mask: torch.Tensor,      # (batch, num_actions)
    ):
        # Mask illegal actions
        masked_logits = policy_logits.masked_fill(legal_mask == 0, float('-inf'))

        # Policy loss: cross-entropy with soft targets
        # = -sum(target * log_softmax(logits))
        log_probs = F.log_softmax(masked_logits, dim=-1)
        policy_loss = -torch.sum(policy_target * log_probs, dim=-1).mean()

        # Value loss: MSE (unchanged)
        value_loss = F.mse_loss(value_pred.squeeze(-1), value_target)

        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss
        return total_loss, {"loss/policy": policy_loss.item(), ...}
```

#### Task 1.6: Update trainer._train_step
**File**: `trainer/src/trainer/trainer.py`

```python
def _train_step(self, observations, policies, values):  # policies instead of actions
    # ...
    policies_t = torch.from_numpy(policies).to(self.device)  # float tensor now
    # ...
```

### Phase 2: Game Outcome Propagation

The current `mcts_value` is the MCTS estimate at each position, not the final game outcome.
For proper AlphaZero training, we need to propagate the final outcome back to all positions.

#### Option A: Post-episode backfill in actor (recommended)

**File**: `actor/src/actor.rs`

After episode completes:
1. Determine final outcome from terminal reward
2. Update all transitions in that episode with `game_outcome`

```rust
// After episode loop ends:
let game_outcome = if total_reward > 0.0 { 1.0 }
                   else if total_reward < 0.0 { -1.0 }
                   else { 0.0 };

// Backfill all transitions in this episode
// Note: Need to track transition IDs or use episode_id for UPDATE
{
    let replay = self.replay.lock().unwrap();
    replay.update_game_outcome(&episode_id, game_outcome)?;
}
```

**File**: `actor/src/replay.rs`

Add schema column and update method:
```rust
// In CREATE TABLE:
game_outcome REAL,  -- Final outcome: +1 (win), -1 (loss), 0 (draw)

// New method:
pub fn update_game_outcome(&self, episode_id: &str, outcome: f32) -> Result<()> {
    // Propagate with alternating sign for two-player games
    self.conn.execute(
        "UPDATE transitions
         SET game_outcome = CASE
             WHEN step_number % 2 = 0 THEN ?1
             ELSE -?1
         END
         WHERE episode_id = ?2",
        params![outcome, episode_id],
    )?;
    Ok(())
}
```

#### Option B: Compute in trainer during sampling

**File**: `trainer/src/trainer/replay.py`

Sample entire episodes and propagate outcome:
```python
def sample_episodes(self, num_episodes: int) -> list[list[Transition]]:
    """Sample complete episodes for proper value target computation."""
    # Get random episode IDs
    cursor = self._conn.execute(
        "SELECT DISTINCT episode_id FROM transitions ORDER BY RANDOM() LIMIT ?",
        (num_episodes,)
    )
    episode_ids = [row[0] for row in cursor.fetchall()]

    episodes = []
    for ep_id in episode_ids:
        cursor = self._conn.execute(
            "SELECT * FROM transitions WHERE episode_id = ? ORDER BY step_number",
            (ep_id,)
        )
        episodes.append([Transition(...) for row in cursor.fetchall()])

    return episodes

def sample_batch_with_outcomes(self, batch_size: int) -> tuple[...]:
    """Sample transitions with proper game outcome propagation."""
    # Sample episodes
    episodes = self.sample_episodes(batch_size // 5)  # ~5 steps per game

    all_obs, all_policies, all_values = [], [], []
    for episode in episodes:
        # Get final outcome from terminal transition
        terminal = episode[-1]
        final_outcome = terminal.reward  # +1, -1, or 0

        # Propagate to all positions with alternating sign
        for i, t in enumerate(episode):
            all_obs.append(np.frombuffer(t.observation, dtype=np.float32))
            all_policies.append(np.frombuffer(t.policy_probs, dtype=np.float32))
            # Alternate sign based on whose turn it was
            player_outcome = final_outcome * ((-1) ** i)
            all_values.append(player_outcome)

    # Shuffle and truncate to batch_size
    # ...
```

**Recommendation**: Option A is cleaner - compute once at generation time rather than repeatedly at training time.

### Phase 3: Update documentation

#### Task 3.1: Update module docstrings
Remove "MVP simplification" notes from:
- `trainer/src/trainer/trainer.py`
- `trainer/src/trainer/replay.py`

#### Task 3.2: Update README
**File**: `trainer/README.md`

Document the training target semantics.

## Schema Migration

For existing replay databases, add migration:

```sql
-- Add new columns if they don't exist
ALTER TABLE transitions ADD COLUMN policy_probs BLOB;
ALTER TABLE transitions ADD COLUMN mcts_value REAL DEFAULT 0.0;
ALTER TABLE transitions ADD COLUMN game_outcome REAL;
```

The trainer should handle missing columns gracefully (fall back to one-hot/reward).

## Testing

### Unit tests
1. Test policy target parsing (f32 array from bytes)
2. Test soft cross-entropy loss computation
3. Test game outcome propagation logic

### Integration tests
1. Run actor to generate data with MCTS policy
2. Train on that data
3. Verify loss decreases appropriately

## Rollout Plan

1. **Phase 1** (trainer reads MCTS policy): Can deploy immediately since actor already writes the data
2. **Phase 2** (game outcome propagation): Requires actor update, then trainer update
3. **Phase 3** (documentation): After both phases complete

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Backward compatibility | Low | Fallback to one-hot if policy_probs is NULL |
| Schema migration | Low | SQLite ALTER TABLE is non-destructive |
| Training instability | Medium | Compare loss curves before/after |
| Performance | Low | Same batch size, slightly more data per row |
