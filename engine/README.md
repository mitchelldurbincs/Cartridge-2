# Engine

Rust workspace containing the core game engine, game implementations, and MCTS search.

## Crates

| Crate | Description |
|-------|-------------|
| `engine-core` | Game trait, type erasure, registry, EngineContext API |
| `games-tictactoe` | TicTacToe reference implementation |
| `mcts` | Monte Carlo Tree Search for AlphaZero-style play |

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

See `games-tictactoe` for a reference implementation.

## Testing

```bash
# All tests
cargo test

# Specific crate
cargo test -p engine-core
cargo test -p games-tictactoe
cargo test -p mcts

# With output
cargo test -- --nocapture
```

## Benchmarks

```bash
# Run benchmarks
cargo bench -p games-tictactoe
```
