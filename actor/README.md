# Actor

Self-play episode runner for Cartridge2. Generates game experience data by running continuous episodes and storing transitions in SQLite for training.

## Overview

The Actor is a long-running Rust binary that:

1. Runs game episodes using the engine-core library directly (no gRPC)
2. Selects actions using MCTS with ONNX neural network evaluation
3. Hot-reloads the model when `latest.onnx` changes
4. Stores transitions with MCTS policy distributions and game outcomes in SQLite
5. Supports graceful shutdown via Ctrl+C
6. Can run multiple instances in parallel for faster data generation

## Quick Start

```bash
# Run with defaults (tictactoe, unlimited episodes)
cargo run

# Run 100 episodes with debug logging
cargo run -- --max-episodes 100 --log-level debug

# Run multiple actors in parallel
cargo run -- --actor-id actor-1 &
cargo run -- --actor-id actor-2 &
cargo run -- --actor-id actor-3 &
```

## CLI Arguments

| Argument | Env Variable | Default | Description |
|----------|--------------|---------|-------------|
| `--actor-id` | `ACTOR_ACTOR_ID` | `actor-1` | Unique identifier for this actor |
| `--env-id` | `ACTOR_ENV_ID` | `tictactoe` | Game environment to run |
| `--max-episodes` | `ACTOR_MAX_EPISODES` | `-1` | Max episodes (-1 = unlimited) |
| `--episode-timeout-secs` | `ACTOR_EPISODE_TIMEOUT` | `30` | Per-episode timeout |
| `--flush-interval-secs` | `ACTOR_FLUSH_INTERVAL` | `5` | SQLite flush interval |
| `--log-level` | `ACTOR_LOG_LEVEL` | `info` | Logging level |
| `--replay-db-path` | `ACTOR_REPLAY_DB_PATH` | `./data/replay.db` | SQLite database path |
| `--data-dir` | `ACTOR_DATA_DIR` | `./data` | Data directory for models |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        ACTOR                             │
├─────────────────────────────────────────────────────────┤
│  main()                                                  │
│    ├─ Parse CLI args (clap)                             │
│    ├─ Validate config                                   │
│    ├─ Init tracing                                      │
│    └─ Run actor.run() async loop                        │
│                                                          │
│  run() async loop                                        │
│    ├─ Check shutdown flag                               │
│    ├─ Check episode limit                               │
│    └─ Run episode:                                      │
│        ├─ engine.reset()                                │
│        └─ loop:                                         │
│            ├─ policy.select_action(obs)                 │
│            ├─ engine.step(state, action)                │
│            ├─ replay.store(transition)                  │
│            └─ break if done                             │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
              ./data/replay.db (SQLite)
```

## Components

### Actor (`src/actor.rs`)

Orchestrates the self-play loop:

```rust
pub struct Actor {
    config: Config,
    engine: Mutex<EngineContext>,      // Game simulation
    policy: Mutex<Box<dyn Policy>>,    // Action selection
    replay: Mutex<ReplayBuffer>,       // SQLite storage
    episode_count: AtomicU32,
    shutdown_signal: AtomicBool,
}

impl Actor {
    pub fn new(config: Config) -> Result<Self>;
    pub async fn run(&self) -> Result<()>;
    pub fn shutdown(&self);
    pub fn episode_count(&self) -> u32;
}
```

### Policy (`src/policy.rs`, `src/mcts_policy.rs`)

Trait for action selection strategies:

```rust
pub trait Policy: Send + Sync {
    fn select_action(&mut self, observation: &[u8]) -> Result<Vec<u8>>;
}
```

Implements:
- **MctsPolicy** - MCTS with ONNX neural network evaluation (default)
- **RandomPolicy** - Uniform random for testing/fallback

The MCTS policy:
- Watches `./data/models/latest.onnx` for updates
- Falls back to random if no model is available
- Returns visit count distributions as policy targets

Supported action spaces:
- **Discrete(n)** - Single integer 0..n (encoded as 4-byte u32)
- **MultiDiscrete(nvec)** - Multiple discrete dimensions
- **Continuous { low, high }** - Real-valued actions (encoded as f32s)

### Replay Buffer (`src/replay.rs`)

SQLite-backed storage for transitions:

```rust
pub struct ReplayBuffer {
    conn: Connection,
}

impl ReplayBuffer {
    pub fn new(db_path: &str) -> Result<Self>;
    pub fn store(&self, transition: &Transition) -> Result<()>;
    pub fn store_batch(&self, transitions: &[Transition]) -> Result<()>;
    pub fn sample(&self, batch_size: usize) -> Result<Vec<Transition>>;
    pub fn count(&self) -> Result<usize>;
    pub fn cleanup(&self, window_size: usize) -> Result<usize>;
}
```

### Transition

Data structure for a single step:

```rust
pub struct Transition {
    pub id: String,              // Unique ID
    pub env_id: String,          // Game environment
    pub episode_id: String,      // Episode grouping
    pub step_number: u32,        // Step within episode
    pub state: Vec<u8>,          // Serialized state
    pub action: Vec<u8>,         // Serialized action
    pub next_state: Vec<u8>,     // Next state
    pub observation: Vec<u8>,    // Current observation
    pub next_observation: Vec<u8>,
    pub reward: f32,
    pub done: bool,
    pub timestamp: u64,
    pub policy_probs: Vec<u8>,   // MCTS visit distribution (f32 array)
    pub mcts_value: f32,         // MCTS value estimate
    pub game_outcome: Option<f32>, // Final outcome (+1/-1/0), backfilled
}
```

## Database Schema

```sql
CREATE TABLE transitions (
    id TEXT PRIMARY KEY,
    env_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    state BLOB NOT NULL,
    action BLOB NOT NULL,
    next_state BLOB NOT NULL,
    observation BLOB NOT NULL,
    next_observation BLOB NOT NULL,
    reward REAL NOT NULL,
    done INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    policy_probs BLOB,           -- f32[num_actions] MCTS visit distribution
    mcts_value REAL DEFAULT 0.0, -- MCTS value estimate
    game_outcome REAL            -- Final outcome, backfilled after episode
);

CREATE INDEX idx_transitions_timestamp ON transitions(timestamp);
CREATE INDEX idx_transitions_episode ON transitions(episode_id);
```

## Examples

### Basic Usage

```bash
# Default configuration
cargo run

# Specific actor ID and episode limit
cargo run -- --actor-id actor-1 --max-episodes 1000

# Custom database location
cargo run -- --replay-db-path /data/replay.db --data-dir /data
```

### Environment Variables

```bash
export ACTOR_ACTOR_ID=distributed-1
export ACTOR_ENV_ID=tictactoe
export ACTOR_MAX_EPISODES=10000
export ACTOR_LOG_LEVEL=info
cargo run
```

### Multiple Actors

```bash
# Run 4 actors in parallel
for i in {1..4}; do
  cargo run -- --actor-id "actor-$i" &
done
wait
```

### Integration with Trainer

```bash
# Terminal 1: Actor generates data
cargo run -- --actor-id actor-1 --replay-db-path ./data/replay.db

# Terminal 2: Python trainer consumes data
cd ../trainer
python -m trainer --db ../data/replay.db
```

## Testing

```bash
# Run all tests (30 tests)
cargo test

# Run specific test with output
cargo test test_actor_run_single_episode -- --nocapture

# Run benchmarks
cargo bench
```

### Test Coverage

- **Config tests (10)** - Validation, defaults, duration conversion
- **Policy tests (9)** - Action space handling, determinism, edge cases
- **Replay tests (8)** - Store, sample, cleanup, batch operations
- **Actor tests (3)** - Creation, episode execution, error handling

## Dependencies

| Crate | Purpose |
|-------|---------|
| `engine-core` | Game simulation API |
| `games-tictactoe` | TicTacToe game |
| `mcts` | Monte Carlo Tree Search |
| `ort` | ONNX Runtime bindings |
| `tokio` | Async runtime |
| `clap` | CLI argument parsing |
| `rusqlite` | SQLite interface |
| `rand_chacha` | Deterministic RNG |
| `tracing` | Structured logging |
| `notify` | File watching for model hot-reload |

## Performance

- **Zero-copy buffers** - Efficient handling of state/action/observation data
- **Async I/O** - Non-blocking episode execution
- **Deterministic RNG** - ChaCha20Rng for reproducible randomness
- **SQLite bundled** - No external database server required

## Troubleshooting

### Common Issues

1. **Unknown env_id** - Ensure game is registered with engine-core
2. **Database locked** - Only one writer at a time; use separate DB files for parallel actors
3. **Permission denied** - Check write permissions on data directory

### Debug Mode

```bash
RUST_LOG=debug cargo run -- --log-level debug
```

## Future Work

- [x] MCTS policy implementation
- [x] ONNX neural network policy
- [x] Model hot-reload via file watching
- [ ] Distributed actor coordination
- [ ] Priority experience replay
