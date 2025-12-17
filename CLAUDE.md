# Cartridge2 - Claude Code Guide

## Project Overview

Cartridge2 is a simplified AlphaZero training and visualization platform. It enables training neural network game agents via self-play and lets users play against trained models through a web interface.

**Target Games:** TicTacToe (complete), Connect 4 (complete), Othello (planned)

**Key Difference from Cartridge1:** This is a monolithic/filesystem approach vs. Cartridge1's microservices architecture. No Kubernetes, no gRPC between services—just shared filesystem and local processes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Filesystem                         │
│  ./data/replay.db     - SQLite replay buffer                │
│  ./data/models/       - ONNX model files                    │
│  ./data/stats.json    - Training telemetry                  │
└─────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
┌────────┴────────┐  ┌───────┴───────┐  ┌────────┴────────┐
│  Web Server     │  │ Python Trainer│  │  Svelte Frontend│
│  (Axum :8080)   │  │ (Learner)     │  │  (Vite :5173)   │
│  - Engine lib   │  │ - PyTorch     │  │  - Play UI      │
│  - Game API     │  │ - SQLite      │  │  - Stats display│
│  - Stats API    │  │ - ONNX export │  │                 │
└─────────────────┘  └───────────────┘  └─────────────────┘
```

## Components

### Engine (Rust Library) - `engine/`
**Status: COMPLETE**

Pure game logic library. No network I/O. Library-only design (no gRPC).

- `engine-core/` - Game trait, erased adapter, registry, EngineContext API, GameMetadata (48 tests)
- `games-tictactoe/` - TicTacToe implementation (26 tests)
- `games-connect4/` - Connect 4 implementation (20 tests)
- `mcts/` - Monte Carlo Tree Search implementation (25 tests)

### Actor (Rust Binary) - `actor/`
**Status: COMPLETE (46 tests)**

Self-play episode runner using engine-core directly:
- Uses `EngineContext` for game simulation (no gRPC)
- Stores transitions in SQLite (`./data/replay.db`)
- MCTS policy with ONNX neural network evaluation
- Hot-reloads model when `latest.onnx` changes (via model_watcher)
- Stores MCTS visit distributions as policy targets
- Game outcome backfill for value targets
- Auto-derives game configuration from GameMetadata

### Web Server (Rust Binary) - `web/`
**Status: COMPLETE (4 tests)**

Axum HTTP server for frontend interaction:
- `/health` - Health check
- `/game/new` - Start new game
- `/game/state` - Get current board state
- `/move` - Make player move + bot response
- `/stats` - Read training stats from stats.json
- `/selfplay` - Start/stop self-play (placeholder)

### Web Frontend (Svelte + TypeScript) - `web/frontend/`
**Status: COMPLETE**

Svelte 5 frontend with Vite:
- TicTacToe board display
- Play against bot (random moves for now)
- Live training stats polling
- Responsive dark-mode UI

### Trainer (Python) - `trainer/`
**Status: COMPLETE**

PyTorch training loop with AlphaZero-style learning:
- Reads transitions from SQLite replay buffer
- MCTS policy distributions as soft targets
- Game outcome propagation for value targets
- Exports ONNX models with atomic write-then-rename
- Writes `stats.json` telemetry
- Cosine annealing LR schedule
- Gradient clipping for stability
- Model evaluation against random baseline

## Directory Structure

```
cartridge2/
├── actor/                  # Rust actor binary
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Entry point
│       ├── actor.rs        # Episode runner using EngineContext
│       ├── config.rs       # CLI configuration
│       ├── game_config.rs  # Game-specific config derived from metadata
│       ├── mcts_policy.rs  # MCTS policy implementation
│       ├── model_watcher.rs # ONNX model hot-reload via file watching
│       ├── policy.rs       # Policy trait and random policy
│       └── replay.rs       # SQLite replay buffer
├── engine/                 # Rust workspace
│   ├── Cargo.toml         # Workspace config
│   ├── engine-core/       # Core Game trait + EngineContext API
│   │   └── src/
│   │       ├── adapter.rs  # GameAdapter (typed -> erased)
│   │       ├── context.rs  # EngineContext high-level API
│   │       ├── erased.rs   # ErasedGame trait
│   │       ├── metadata.rs # GameMetadata for game configuration
│   │       ├── registry.rs # Static game registration
│   │       └── typed.rs    # Game trait definition
│   ├── games-tictactoe/   # TicTacToe implementation
│   ├── games-connect4/    # Connect 4 implementation
│   └── mcts/              # Monte Carlo Tree Search
│       └── src/
│           ├── config.rs   # MctsConfig
│           ├── evaluator.rs # Evaluator trait + UniformEvaluator
│           ├── node.rs     # MctsNode
│           ├── onnx.rs     # OnnxEvaluator (feature-gated)
│           ├── search.rs   # MCTS search algorithm
│           └── tree.rs     # MctsTree with arena allocation
├── web/                    # Web server + frontend
│   ├── Cargo.toml         # Axum server
│   ├── src/
│   │   ├── main.rs        # HTTP endpoints
│   │   ├── game.rs        # Game session management
│   │   └── model_watcher.rs # Model hot-reload for web
│   ├── frontend/          # Svelte frontend
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── App.svelte
│   │   │   ├── Board.svelte      # TicTacToe board
│   │   │   ├── Connect4Board.svelte # Connect 4 board
│   │   │   └── Stats.svelte
│   │   └── vite.config.ts
│   └── README.md          # Run commands
├── trainer/               # Python training
│   ├── pyproject.toml     # Package configuration
│   └── src/trainer/
│       ├── __main__.py    # CLI entrypoint
│       ├── trainer.py     # Training loop
│       ├── network.py     # Neural network
│       ├── replay.py      # SQLite interface
│       ├── evaluator.py   # Model evaluation
│       └── game_config.py # Game-specific configurations
├── scripts/               # Training scripts
│   ├── train_loop.py      # Synchronized AlphaZero training loop (local)
│   ├── docker_train.sh    # Docker entrypoint for synchronized training
│   └── Dockerfile         # Combined actor+trainer image for Docker
├── docs/
│   └── MVP.md             # Design document
├── data/                  # Runtime data (gitignored)
│   ├── replay.db          # SQLite replay buffer
│   ├── models/            # ONNX model files
│   └── stats.json         # Training telemetry
└── CLAUDE.md              # This file
```

## Quick Start

### Play TicTacToe in Browser

Terminal 1 - Start Rust backend:
```bash
cd web
cargo run
# Server starts on http://localhost:8080
```

Terminal 2 - Start Svelte frontend:
```bash
cd web/frontend
npm install
npm run dev
# Dev server starts on http://localhost:5173
```

Open http://localhost:5173 in your browser!

### Train with Docker (Easiest)

```bash
# Train TicTacToe (default)
docker compose up alphazero

# Train Connect4
ALPHAZERO_ENV_ID=connect4 docker compose up alphazero

# Customize training parameters
ALPHAZERO_ITERATIONS=50 ALPHAZERO_EPISODES=200 ALPHAZERO_STEPS=500 docker compose up alphazero

# Run in background
docker compose up alphazero -d
docker compose logs -f alphazero  # Watch progress

# Play against trained model (in another terminal)
docker compose up web frontend
# Open http://localhost in browser
```

## Commands

```bash
# Build engine
cd engine && cargo build --release

# Build actor
cd actor && cargo build --release

# Build web server
cd web && cargo build --release

# Run all tests
cd engine && cargo test   # 119 tests (48 + 26 + 20 + 25)
cd actor && cargo test    # 46 tests
cd web && cargo test      # 4 tests

# Format and lint
cd engine && cargo fmt && cargo clippy
cd actor && cargo fmt && cargo clippy
cd web && cargo fmt && cargo clippy

# Start web server
cd web && cargo run

# Start frontend dev server
cd web/frontend && npm run dev

# ======= RECOMMENDED: Synchronized AlphaZero Training =======
# Each iteration: clear buffer -> generate episodes -> train -> repeat
# This ensures training data comes from the current model only

# Basic synchronized training (TicTacToe)
python3 scripts/train_loop.py --iterations 50 --episodes 200 --steps 500

# Connect4 with more data per iteration
python3 scripts/train_loop.py --env-id connect4 --iterations 100 --episodes 500 --steps 1000

# With GPU and evaluation
python3 scripts/train_loop.py --device cuda --eval-interval 100 --eval-games 50

# Resume from a specific iteration
python3 scripts/train_loop.py --iterations 100 --start-iteration 25

# ======= Alternative: Continuous (non-synchronized) training =======
# Actor and trainer run concurrently - mixes data from multiple model versions
# Less correct for AlphaZero but simpler for quick experiments

# Run self-play to generate training data
cd actor && cargo run -- --env-id tictactoe --max-episodes 1000

# Train the model (in separate terminal)
cd trainer && python3 -m trainer --db ../data/replay.db --steps 1000

# Evaluate model against random play
cd trainer && python3 -m trainer.evaluator --model ../data/models/latest.onnx --games 100
```

## Current Status

- [x] Engine core abstractions (Game trait, adapter, registry, metadata) - 48 tests
- [x] EngineContext high-level API
- [x] TicTacToe game implementation - 26 tests
- [x] Connect 4 game implementation - 20 tests
- [x] Removed gRPC/proto dependencies (library-only)
- [x] Actor core (episode runner, SQLite replay) - 46 tests
- [x] MCTS integration in actor with ONNX evaluation
- [x] Model hot-reload via file watching
- [x] Auto-derived game configuration from GameMetadata
- [x] Web server (Axum, game API) - 4 tests
- [x] Web frontend (Svelte, play UI, stats, Connect4 board)
- [x] MCTS implementation - 25 tests
- [x] Python trainer (PyTorch, ONNX export, evaluator)
- [x] MCTS policy targets + game outcome propagation
- [ ] Othello game

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/game/state` | GET | Get current board state |
| `/game/new` | POST | Start a new game |
| `/move` | POST | Make a move (player + bot) |
| `/stats` | GET | Read training stats |
| `/selfplay` | POST | Start/stop self-play |

## Using the Engine

```rust
use engine_core::EngineContext;
use games_tictactoe::register_tictactoe;

// Register games at startup
register_tictactoe();

// Create a context for TicTacToe
let mut ctx = EngineContext::new("tictactoe").expect("game registered");

// Reset to initial state
let reset = ctx.reset(42, &[]).unwrap();
println!("Initial state: {} bytes", reset.state.len());

// Take a step (action = position 4 = center)
let action = 4u32.to_le_bytes().to_vec();
let step = ctx.step(&reset.state, &action).unwrap();
println!("Reward: {}, Done: {}", step.reward, step.done);
```

## Game Trait Pattern

Games implement a typed trait that gets erased for runtime dispatch:

```rust
pub trait Game {
    type State;
    type Action;
    type Obs;

    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8]) -> (State, Obs);
    fn step(&mut self, state: &mut State, action: Action, rng: &mut ChaCha20Rng)
        -> (Obs, f32, bool, u64);
    fn encode_state(state: &State, buf: &mut Vec<u8>) -> Result<(), Error>;
    fn decode_state(buf: &[u8]) -> Result<State, Error>;
    // ... similar for Action and Obs
}
```

## Adding a New Game

1. Create crate in `engine/games-{name}/`
2. Implement `Game` trait with State/Action/Obs types
3. Implement encode/decode for each type
4. Add a `register_{name}()` function that calls `register_game()`
5. Add tests for game logic + encoding round-trips

Example registration:
```rust
use engine_core::{register_game, GameAdapter};

pub fn register_connect4() {
    register_game("connect4".to_string(), || {
        Box::new(GameAdapter::new(Connect4::new()))
    });
}
```

## Differences from Cartridge1

| Aspect | Cartridge1 | Cartridge2 |
|--------|------------|------------|
| Architecture | 7 microservices | Monolith + Python |
| Communication | gRPC everywhere | Filesystem + HTTP |
| Replay Buffer | Go service + Redis | SQLite file |
| Model Storage | Go service + MinIO | Single ONNX file |
| Orchestration | K8s/Docker Compose | Shell script |
| Complexity | Production-grade | MVP-focused |

## Using MCTS

```rust
use mcts::{MctsConfig, UniformEvaluator, run_mcts};
use engine_core::EngineContext;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

// Register game and create context
games_tictactoe::register_tictactoe();
let mut ctx = EngineContext::new("tictactoe").unwrap();
let reset = ctx.reset(42, &[]).unwrap();

// Set up MCTS with uniform evaluator (for testing)
let evaluator = UniformEvaluator::new();
let config = MctsConfig::for_training()
    .with_simulations(800)
    .with_temperature(1.0);

// Run search
let legal_mask = 0b111111111u64; // All 9 positions legal initially
let mut rng = ChaCha20Rng::seed_from_u64(42);
let result = run_mcts(&mut ctx, &evaluator, config, reset.state, legal_mask, &mut rng).unwrap();

println!("Best action: {}", result.action);
println!("Policy: {:?}", result.policy);
println!("Value: {}", result.value);
```

### MCTS Architecture

```
engine/mcts/src/
├── lib.rs          # Public API exports
├── config.rs       # MctsConfig (num_simulations, c_puct, temperature, etc.)
├── evaluator.rs    # Evaluator trait + UniformEvaluator
├── node.rs         # MctsNode (visit_count, value_sum, prior, children)
├── tree.rs         # MctsTree with arena allocation
└── search.rs       # Select, expand, backpropagate, run_search
```

### Key Types

- `MctsConfig` - Search parameters (simulations, c_puct, dirichlet noise, temperature)
- `Evaluator` trait - Provides policy priors and value estimates
- `UniformEvaluator` - Returns uniform policy (for testing without neural network)
- `SearchResult` - Contains best action, policy distribution, value estimate

## Next Steps

1. **Othello Game** - Add third game implementation

## Reference

- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) - Python AlphaZero reference
- MVP.md in docs/ - Full design document
