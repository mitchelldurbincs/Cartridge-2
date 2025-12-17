# Engine

Rust workspace containing the core game engine, game implementations, and MCTS search.

## Crates

| Crate | Description |
|-------|-------------|
| `engine-core` | Game trait, type erasure, registry, EngineContext API, GameMetadata |
| `games-tictactoe` | TicTacToe reference implementation (26 tests) |
| `games-connect4` | Connect 4 implementation (20 tests) |
| `mcts` | Monte Carlo Tree Search for AlphaZero-style play (25 tests) |

## Quick Start

```bash
# Build all crates
cargo build --release

# Run all tests
cargo test

# Run with ONNX support
cargo build --release --features mcts/onnx
```

## Architecture

```
+------------------+     +-------------------+
|  games-tictactoe |---->|    engine-core    |
+------------------+     +-------------------+
                               ^
+------------------+           |
|  games-connect4  |-----------|
+------------------+           |
                               |
+------------------+           |
|       mcts       |-----------+
+------------------+
```

All game implementations depend on `engine-core` for the Game trait.
MCTS uses `engine-core` for game simulation via EngineContext.

## Workspace Dependencies

Shared dependencies are defined in the root `Cargo.toml`:

- `rand_chacha` / `rand` - Deterministic randomness
- `thiserror` / `anyhow` - Error handling
- `once_cell` - Lazy static initialization
- `serde` - Serialization
- `tracing` - Logging
- `criterion` / `proptest` - Testing and benchmarks

## Adding a New Game

1. Create crate: `cargo new games-{name} --lib`
2. Add to workspace members in `Cargo.toml`
3. Implement the `Game` trait from `engine-core`
4. Add a `register_{name}()` function
5. Write tests for game logic and encoding

See `games-tictactoe` or `games-connect4` for reference implementations.

## Testing

```bash
# All tests (119 total)
cargo test

# Specific crate
cargo test -p engine-core      # 48 tests
cargo test -p games-tictactoe  # 26 tests
cargo test -p games-connect4   # 20 tests
cargo test -p mcts             # 25 tests

# With output
cargo test -- --nocapture
```

## Benchmarks

```bash
# Run benchmarks
cargo bench -p games-tictactoe
```
