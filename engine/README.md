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

# MCTS microbenchmarks (Criterion)
cargo bench -p mcts --bench mcts
```

Recent run (devcontainer, plotters backend) highlights:

- `mcts_search_simulations` (Uniform policy): `50` sims ~126 µs; `100` sims ~295 µs; `200` sims ~579 µs; `400` sims ~1.11 ms; `800` sims ~2.21 ms.
- `mcts_game_phases`: opening ~558 µs; midgame ~68 µs; near-terminal ~31 µs.
- `mcts_tree_ops`: allocate node ~14.9 µs; select child ~57 ns; backpropagate depth 5 ~56 ns; root policy ~107 ns; root policy (τ=0.5) ~222 ns.
- `mcts_configs`: training config ~593 µs; evaluation config ~567 µs; `c_puct` 0.5/1.25/2.5/4.0 around 559/572/546/554 µs respectively.
