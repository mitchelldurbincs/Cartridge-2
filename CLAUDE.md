# Cartridge2 - Claude Code Guide

## Project Overview

Cartridge2 is a simplified AlphaZero training and visualization platform. It enables training neural network game agents via self-play and lets users play against trained models through a web interface.

**Target Games:** TicTacToe, Connect 4, Othello

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

- `engine-core/` - Game trait, erased adapter, registry, EngineContext API (39 tests)
- `games-tictactoe/` - TicTacToe implementation (15 tests)
- `mcts/` - Monte Carlo Tree Search implementation (21 tests)

### Actor (Rust Binary) - `actor/`
**Status: CORE COMPLETE (30 tests)**

Self-play episode runner using engine-core directly:
- Uses `EngineContext` for game simulation (no gRPC)
- Stores transitions in SQLite (`./data/replay.db`)
- Random policy for action selection
- Ready for MCTS and ONNX integration

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

### Learner (Python) - To Be Built
- PyTorch training loop
- Reads from SQLite replay buffer
- Exports ONNX models with atomic write-then-rename
- Writes `stats.json` telemetry

## Directory Structure

```
cartridge2/
├── actor/                  # Rust actor binary
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Entry point
│       ├── actor.rs        # Episode runner using EngineContext
│       ├── config.rs       # CLI configuration
│       ├── policy.rs       # Action selection (random policy)
│       └── replay.rs       # SQLite replay buffer
├── engine/                 # Rust workspace
│   ├── Cargo.toml         # Workspace config
│   ├── engine-core/       # Core Game trait + EngineContext API
│   ├── games-tictactoe/   # Reference game implementation
│   └── mcts/              # Monte Carlo Tree Search
├── web/                    # Web server + frontend
│   ├── Cargo.toml         # Axum server
│   ├── src/
│   │   ├── main.rs        # HTTP endpoints
│   │   └── game.rs        # Game session management
│   ├── frontend/          # Svelte frontend
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── App.svelte
│   │   │   ├── Board.svelte
│   │   │   └── Stats.svelte
│   │   └── vite.config.ts
│   └── README.md          # Run commands
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

## Commands

```bash
# Build engine
cd engine && cargo build --release

# Build actor
cd actor && cargo build --release

# Build web server
cd web && cargo build --release

# Run all tests
cd engine && cargo test   # 75 tests (39 + 15 + 21)
cd actor && cargo test    # 30 tests
cd web && cargo test      # 4 tests

# Format and lint
cd engine && cargo fmt && cargo clippy
cd actor && cargo fmt && cargo clippy
cd web && cargo fmt && cargo clippy

# Start web server
cd web && cargo run

# Start frontend dev server
cd web/frontend && npm run dev
```

## Current Status

- [x] Engine core abstractions (Game trait, adapter, registry) - 39 tests
- [x] EngineContext high-level API
- [x] TicTacToe game implementation - 15 tests
- [x] Removed gRPC/proto dependencies (library-only)
- [x] Actor core (episode runner, SQLite replay) - 30 tests
- [x] Web server (Axum, game API) - 4 tests
- [x] Web frontend (Svelte, play UI, stats)
- [x] MCTS implementation - 21 tests
- [ ] ONNX model integration
- [ ] Python trainer
- [ ] Connect 4 game
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

1. **ONNX Integration** - Load and run neural network policies
2. **Python Learner** - Training script that reads from SQLite
3. **Connect 4 Game** - Add second game implementation
4. **Othello Game** - Add third game implementation

## Reference

- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) - Python AlphaZero reference
- MVP.md in docs/ - Full design document
