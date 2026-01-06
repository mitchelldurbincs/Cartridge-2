<p align="center">
  <img src="./logo.png" alt="Cartridge2 Logo" width="500">
</p>

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

### Option 1: Docker (Easiest)

```bash
# Train a model using synchronized AlphaZero loop
docker compose --profile local up alphazero

# Train Connect4 instead of TicTacToe
CARTRIDGE_COMMON_ENV_ID=connect4 docker compose --profile local up alphazero

# Play against the trained model
docker compose --profile local up web frontend
# Open http://localhost in browser
```

### Option 2: Local Development

**Terminal 1** - Start the Rust backend:
```bash
cd web && cargo run
# Server starts on http://localhost:8080
```

**Terminal 2** - Start the Svelte frontend:
```bash
cd web/frontend && npm install && npm run dev
# Dev server starts on http://localhost:5173
```

**Terminal 3** - Train a model:
```bash
cd trainer && pip install -e .
python -m trainer loop --iterations 50 --episodes 200 --steps 500
```

Open http://localhost:5173 to play!

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
|   |   |-- replay.rs          # SQLite interface
|   |   +-- storage/           # Storage backends (SQLite, PostgreSQL)
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
|   |-- mcts/                  # Monte Carlo Tree Search
|   +-- model-watcher/         # Shared model hot-reload library
|
|-- web/                       # HTTP server + frontend
|   |-- src/
|   |   |-- main.rs            # Axum endpoints
|   |   |-- game.rs            # Session management
|   |   |-- central_config.rs  # Central config.toml loading
|   |   +-- model_watcher.rs   # ONNX model hot-reload
|   +-- frontend/              # Svelte application
|       +-- src/
|           |-- App.svelte
|           |-- GenericBoard.svelte     # Game board component
|           |-- LossChart.svelte        # Loss visualization chart
|           |-- LossOverTimePage.svelte # Training progress page
|           +-- Stats.svelte
|
|-- trainer/                   # Python training
|   |-- pyproject.toml         # Package configuration
|   +-- src/trainer/
|       |-- __main__.py        # CLI entrypoint
|       |-- trainer.py         # Training loop
|       |-- orchestrator.py    # Synchronized AlphaZero orchestrator
|       |-- network.py         # Neural network (MLP)
|       |-- resnet.py          # ResNet architecture
|       |-- replay.py          # Replay buffer interface
|       |-- evaluator.py       # Model evaluation
|       |-- game_config.py     # Game-specific configurations
|       |-- stats.py           # Training statistics
|       |-- config.py          # TrainerConfig dataclass
|       |-- checkpoint.py      # Checkpoint utilities
|       |-- central_config.py  # Central config.toml loading
|       +-- storage/           # Storage backends (SQLite, PostgreSQL, S3)
|
|-- data/                      # Runtime data (gitignored)
|   |-- replay.db              # SQLite replay buffer
|   |-- models/                # ONNX checkpoints
|   +-- stats.json             # Training telemetry
|
|-- docs/
|   |-- MVP.md                 # Design document
|   |-- claude-k8s-ideas.md    # K8s migration roadmap
|   +-- codex-k8s-ideas.md     # K8s adoption checklist
|
|-- config.toml                # Central configuration
|-- docker-compose.yml         # Local mode services
+-- docker-compose.k8s.yml     # K8s mode services (PostgreSQL + MinIO)
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

Docker usage mirrors the same interface. Use the `--profile local` flag to start
the alphazero service:

```bash
docker compose --profile local up alphazero
# Override parameters as needed
CARTRIDGE_COMMON_ENV_ID=connect4 docker compose --profile local up alphazero
```

See [Deployment Modes](#deployment-modes) for K8s-style backends with PostgreSQL and MinIO.

## Deployment Modes

Cartridge2 supports two deployment modes via Docker Compose profiles:

### Local Mode (Default)

Uses SQLite for replay buffer and filesystem for model storage. Simple, no external dependencies.

```bash
# Start synchronized training
docker compose --profile local up alphazero

# Start web UI to play against trained model
docker compose --profile local up web frontend
```

### K8s Mode (Horizontal Scaling)

Uses PostgreSQL for replay buffer and MinIO (S3-compatible) for model storage. Enables multiple actors running in parallel.

```bash
# Start infrastructure (PostgreSQL + MinIO) and training services
docker compose -f docker-compose.yml -f docker-compose.k8s.yml --profile k8s up

# Scale actors horizontally (4 parallel self-play workers)
docker compose -f docker-compose.yml -f docker-compose.k8s.yml --profile k8s up --scale actor-k8s=4

# Access MinIO web console at http://localhost:9001
# Credentials: aspect / password123
```

**K8s Mode Services:**

| Service | Description |
|---------|-------------|
| `postgres` | PostgreSQL 16 for replay buffer storage |
| `minio` | S3-compatible object storage for models |
| `actor-k8s` | Self-play actor (scalable with `--scale`) |
| `trainer-k8s` | Continuous trainer consuming replay data |
| `web-k8s` | Web backend with S3 model loading |

**K8s Mode Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `CARTRIDGE_STORAGE_REPLAY_BACKEND` | `sqlite` or `postgres` | `sqlite` |
| `CARTRIDGE_STORAGE_MODEL_BACKEND` | `filesystem` or `s3` | `filesystem` |
| `CARTRIDGE_STORAGE_POSTGRES_URL` | PostgreSQL connection string | - |
| `CARTRIDGE_STORAGE_S3_BUCKET` | S3 bucket for models | - |
| `CARTRIDGE_STORAGE_S3_ENDPOINT` | S3-compatible endpoint (MinIO) | - |

### Orchestrator Mode

For synchronized AlphaZero training with K8s backends (single container running actor+trainer loop):

```bash
docker compose -f docker-compose.yml -f docker-compose.k8s.yml --profile orchestrator up alphazero-k8s
```

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
cd engine && cargo test    # 134 tests
cd actor && cargo test     # 46 tests
cd web && cargo test       # 22 tests
cd trainer && pytest       # Python tests
```

### Format & Lint

```bash
cd engine && cargo fmt && cargo clippy
cd actor && cargo fmt && cargo clippy
cd web && cargo fmt && cargo clippy
```

## Current Status

**Core:**
- [x] Engine core abstractions (Game trait, adapter, registry, metadata)
- [x] EngineContext high-level API
- [x] TicTacToe game implementation
- [x] Connect 4 game implementation
- [x] MCTS implementation with ONNX evaluation

**Training:**
- [x] Actor (episode runner, MCTS + ONNX, model hot-reload)
- [x] Python trainer (PyTorch, ONNX export, cosine LR)
- [x] Synchronized AlphaZero training loop (orchestrator)
- [x] MCTS policy targets + game outcome propagation
- [x] Model evaluation against random baseline

**Storage Backends:**
- [x] SQLite replay buffer (local mode)
- [x] PostgreSQL replay buffer (K8s mode)
- [x] Filesystem model storage (local mode)
- [x] S3/MinIO model storage (K8s mode)

**Web:**
- [x] Web server (Axum, game API)
- [x] Web frontend (Svelte, play UI, stats)
- [x] Loss visualization chart

**Deployment:**
- [x] Docker Compose with profiles (local, k8s, orchestrator)
- [x] Horizontal actor scaling via `--scale`

**Planned:**
- [ ] Othello game
- [ ] Kubernetes manifests (Helm/Kustomize)

## Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Architecture | Monolith + Python | Simplicity for MVP, easy local development |
| Deployment | Docker Compose profiles | `local` for simple setups, `k8s` for horizontal scaling |
| Language | Rust + Python | Type safety + ML ecosystem |
| Game Interface | Typed trait + erasure | Compile-time safety + runtime flexibility |
| Replay Storage | SQLite / PostgreSQL | SQLite for local, PostgreSQL for multi-actor scaling |
| Model Storage | Filesystem / S3 | Filesystem for local, S3/MinIO for distributed |
| Model Format | ONNX | Framework-agnostic, production-ready |
| RNG | ChaCha20 | Deterministic, reproducible simulations |

## License

MIT
