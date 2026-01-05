# Cartridge2

A simplified AlphaZero training and visualization platform for board games. Train neural network agents via self-play and play against them through a web interface.

**Games:** TicTacToe (complete), Connect 4 (complete), Othello (planned)

## Architecture

```
+---------------------------------------------------------------+
|                      Shared Filesystem                        |
|   ./data/replay.db      - SQLite replay buffer                |
|   ./data/models/        - ONNX model files                    |
|   ./data/stats.json     - Training telemetry                  |
+---------------------------------------------------------------+
         |                       |                       |
         v                       v                       v
+-----------------+    +-----------------+    +------------------+
|   Web Server    |    | Python Trainer  |    | Svelte Frontend  |
|   (Axum :8080)  |    | (Learner)       |    | (Vite :5173)     |
|   - Engine lib  |    | - PyTorch       |    | - Play UI        |
|   - Game API    |    | - SQLite        |    | - Stats display  |
|   - Stats API   |    | - ONNX export   |    |                  |
+-----------------+    +-----------------+    +------------------+
```

By default, this is a monolithic, filesystem-based approach--just shared filesystem
and local processes. For horizontal scaling and cloud deployments, optional backends
support PostgreSQL (replay buffer) and S3/MinIO (model storage).

## Quick Start

### Play TicTacToe in Browser

**Terminal 1** - Start the Rust backend:
```bash
cd web
cargo run
# Server starts on http://localhost:8080
```

**Terminal 2** - Start the Svelte frontend:
```bash
cd web/frontend
npm install
npm run dev
# Dev server starts on http://localhost:5173
```

Open http://localhost:5173 to play!

### Run Self-Play

```bash
cd actor
cargo run -- --env-id tictactoe --max-episodes 1000
# Generates training data in ./data/replay.db
```

### Train a Model

```bash
cd trainer
pip install -e .

# Train on existing replay data
trainer train --db ../data/replay.db --steps 1000
# Or: python -m trainer train --db ../data/replay.db --steps 1000

# Exports model to ./data/models/latest.onnx
```

### Evaluate the Model

```bash
# Evaluate model against random baseline
trainer evaluate --model ../data/models/latest.onnx --games 100
# Or: python -m trainer evaluate --model ../data/models/latest.onnx --games 100

# Plays 100 games against random, reports win rate
```

## Project Structure

```
cartridge2/
|-- actor/                     # Self-play episode runner
|   |-- src/
|   |   |-- main.rs            # Entry point
|   |   |-- actor.rs           # Episode loop
|   |   |-- config.rs          # CLI configuration
|   |   |-- game_config.rs     # Game-specific config from metadata
|   |   |-- mcts_policy.rs     # MCTS policy implementation
|   |   |-- model_watcher.rs   # ONNX model hot-reload
|   |   |-- policy.rs          # Policy trait + random policy
|   |   +-- replay.rs          # SQLite interface
|   +-- tests/
|
|-- engine/                    # Rust workspace
|   |-- engine-core/           # Game trait + registry
|   |   +-- src/
|   |       |-- typed.rs       # Game trait definition
|   |       |-- adapter.rs     # Type erasure adapter
|   |       |-- erased.rs      # Erased game interface
|   |       |-- context.rs     # EngineContext API
|   |       |-- metadata.rs    # GameMetadata for game configuration
|   |       +-- registry.rs    # Static game registry
|   |-- games-tictactoe/       # TicTacToe implementation
|   |-- games-connect4/        # Connect 4 implementation
|   +-- mcts/                  # Monte Carlo Tree Search
|
|-- web/                       # HTTP server + frontend
|   |-- src/
|   |   |-- main.rs            # Axum endpoints
|   |   |-- game.rs            # Session management
|   |   +-- model_watcher.rs   # ONNX model hot-reload
|   +-- frontend/              # Svelte application
|       +-- src/
|           |-- App.svelte
|           |-- Board.svelte         # TicTacToe board
|           |-- Connect4Board.svelte # Connect 4 board
|           +-- Stats.svelte
|
|-- trainer/                   # Python training
|   |-- pyproject.toml         # Package configuration
|   +-- src/trainer/
|       |-- __main__.py        # CLI entrypoint
|       |-- trainer.py         # Training loop
|       |-- network.py         # Neural network
|       |-- replay.py          # SQLite interface
|       |-- evaluator.py       # Model evaluation
|       +-- game_config.py     # Game-specific configurations
|
|-- data/                      # Runtime data (gitignored)
|   |-- replay.db              # SQLite replay buffer
|   |-- models/                # ONNX checkpoints
|   +-- stats.json             # Training telemetry
|
+-- docs/
    +-- MVP.md                 # Design document
```

## Components

### Engine Core (`engine/engine-core/`)

Pure Rust game logic library:

- **Game Trait** - Type-safe interface for implementing games
- **Type Erasure** - Runtime polymorphism via trait objects
- **Registry** - Static game registration system
- **EngineContext** - High-level API for game simulation

```rust
use engine_core::EngineContext;
use games_tictactoe::register_tictactoe;

// Register games at startup
register_tictactoe();

// Create context and play
let mut ctx = EngineContext::new("tictactoe").expect("game registered");
let reset = ctx.reset(42, &[]).unwrap();

// Take action (position 4 = center square)
let action = 4u32.to_le_bytes().to_vec();
let step = ctx.step(&reset.state, &action).unwrap();
```

### Actor (`actor/`)

Self-play episode generator:

- Runs game simulations using `EngineContext`
- MCTS with ONNX neural network evaluation
- Hot-reloads model when `latest.onnx` changes
- Stores transitions in SQLite replay buffer
- MCTS visit distributions saved as policy targets
- Game outcomes backfilled to all positions

```bash
cargo run -- \
    --env-id tictactoe \
    --max-episodes 10000 \
    --replay-db-path ./data/replay.db
```

### Web Server (`web/`)

Axum HTTP server with endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/game/state` | GET | Get current board state |
| `/game/new` | POST | Start a new game |
| `/move` | POST | Make player move + get bot response |
| `/stats` | GET | Read training telemetry |
| `/selfplay` | POST | Start/stop self-play |

### Python Trainer (`trainer/`)

PyTorch training loop:

- Reads transitions from SQLite replay buffer
- AlphaZero-style loss (policy cross-entropy + value MSE)
- MCTS visit distributions as soft policy targets
- Game outcomes propagated as value targets
- Exports ONNX models with atomic write-then-rename
- Cosine annealing LR schedule
- Gradient clipping for stability
- Checkpoint management
- Model evaluation against random baseline

#### Synchronized AlphaZero training loop

The Python package now includes an orchestrated, synchronous AlphaZero workflow
that coordinates the actor, trainer, and post-iteration evaluation. This
pipeline clears the replay buffer each iteration, generates fresh self-play
episodes, trains on that data, and then benchmarks the resulting model against
the random baseline.

Run locally (with defaults targeting TicTacToe):

```bash
# Using the subcommand interface
trainer loop --iterations 5 --episodes 200 --steps 500
# Or: python -m trainer loop --iterations 5 --episodes 200 --steps 500

# Legacy entry point also works
trainer-loop --iterations 5 --episodes 200 --steps 500
```

Configuration can be supplied via flags or environment variables (prefixed with
`ALPHAZERO_` or `CARTRIDGE_`). For example, to train Connect4 with GPU acceleration
and disable evaluation for speed:

```bash
ALPHAZERO_ENV_ID=connect4 ALPHAZERO_DEVICE=cuda ALPHAZERO_EVAL_INTERVAL=0 \
    trainer loop --iterations 20 --episodes 300 --steps 1000
```

Docker usage mirrors the same interface. The root-level `docker-compose.yml`
defines an `alphazero` service that runs the synchronized loop and mounts the
shared `./data` volume for replay, models, and stats:

```bash
docker compose up alphazero
# Override parameters as needed
ALPHAZERO_ENV_ID=connect4 ALPHAZERO_EVAL_GAMES=100 docker compose up alphazero
```

#### K8s-Style Backends (PostgreSQL + MinIO)

For horizontal scaling and cloud deployments, Cartridge2 supports PostgreSQL for
the replay buffer and S3-compatible storage (MinIO) for models. This allows
multiple actors to run in parallel, sharing a centralized replay buffer.

Test the K8s backends locally with Docker Compose:

```bash
# Start PostgreSQL, MinIO, and the training pipeline with K8s backends
docker compose -f docker-compose.yml -f docker-compose.k8s.yml --profile k8s up

# Scale actors horizontally (4 parallel self-play workers)
docker compose -f docker-compose.yml -f docker-compose.k8s.yml --profile k8s up --scale actor-k8s=4

# Access MinIO web console at http://localhost:9001
# Credentials: minioadmin / minioadmin
```

Configuration is controlled via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CARTRIDGE_STORAGE_REPLAY_BACKEND` | `sqlite` or `postgres` | `sqlite` |
| `CARTRIDGE_STORAGE_MODEL_BACKEND` | `filesystem` or `s3` | `filesystem` |
| `CARTRIDGE_STORAGE_POSTGRES_URL` | PostgreSQL connection string | - |
| `CARTRIDGE_STORAGE_S3_BUCKET` | S3 bucket for models | - |
| `CARTRIDGE_STORAGE_S3_ENDPOINT` | S3-compatible endpoint (MinIO) | - |

## Adding a New Game

1. Create a new crate in `engine/games-{name}/`
2. Implement the `Game` trait:

```rust
pub trait Game {
    type State;   // Game state (must be Copy-friendly)
    type Action;  // Action space
    type Obs;     // Observation for neural networks

    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8]) -> (State, Obs);
    fn step(&mut self, state: &mut State, action: Action, rng: &mut ChaCha20Rng)
        -> (Obs, f32, bool, u64);  // obs, reward, done, info_bits

    fn encode_state(state: &State, buf: &mut Vec<u8>) -> Result<(), Error>;
    fn decode_state(buf: &[u8]) -> Result<State, Error>;
    // ... similar for Action and Obs
}
```

3. Register the game:

```rust
use engine_core::{register_game, GameAdapter};

pub fn register_connect4() {
    register_game("connect4".to_string(), || {
        Box::new(GameAdapter::new(Connect4::new()))
    });
}
```

4. Add tests for game logic and encoding round-trips

## Development

### Build

```bash
# Build all Rust components
cd engine && cargo build --release
cd ../actor && cargo build --release
cd ../web && cargo build --release

# Install Python dependencies
cd trainer && pip install -e .
```

### Test

```bash
cd engine && cargo test    # 119 tests
cd actor && cargo test     # 46 tests
cd web && cargo test       # 4 tests
cd trainer && pytest       # Python tests
```

### Format & Lint

```bash
cd engine && cargo fmt && cargo clippy
cd actor && cargo fmt && cargo clippy
cd web && cargo fmt && cargo clippy
```

## Current Status

- [x] Engine core abstractions (Game trait, adapter, registry, metadata)
- [x] EngineContext high-level API
- [x] TicTacToe game implementation
- [x] Connect 4 game implementation
- [x] Actor (episode runner, SQLite replay, MCTS + ONNX)
- [x] MCTS implementation
- [x] Model hot-reload via file watching
- [x] Auto-derived game configuration from GameMetadata
- [x] Web server (Axum, game API)
- [x] Web frontend (Svelte, play UI, stats, Connect4 board)
- [x] Python trainer (PyTorch, ONNX export)
- [x] MCTS policy targets + game outcome propagation
- [x] Model evaluation against random baseline
- [x] K8s backends (PostgreSQL replay, S3 model storage)
- [ ] Othello game

## Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Architecture | Monolith + Python | Simplicity for MVP, easy local development |
| IPC | Filesystem (local), Postgres+S3 (cloud) | No network complexity locally, scales horizontally |
| Language | Rust + Python | Type safety + ML ecosystem |
| Game Interface | Typed trait + erasure | Compile-time safety + runtime flexibility |
| Replay Storage | SQLite (local), PostgreSQL (K8s) | Feature-gated for deployment flexibility |
| Model Storage | Filesystem (local), S3/MinIO (K8s) | Hot-reload via file watch or ETag polling |
| Model Format | ONNX | Framework-agnostic, production-ready |
| RNG | ChaCha20 | Deterministic, reproducible simulations |

## License

MIT
