# games-tictactoe

TicTacToe implementation for Cartridge2. Reference game demonstrating the engine-core Game trait pattern.

## Overview

A complete TicTacToe game with:
- 3x3 board, two players (X and O)
- Win detection (rows, columns, diagonals)
- Draw detection (full board)
- Neural network-friendly observation encoding

## Usage

```rust
use engine_core::EngineContext;
use games_tictactoe::register_tictactoe;

// Register the game
register_tictactoe();

// Create context
let mut ctx = EngineContext::new("tictactoe").unwrap();

// Start a new game
let reset = ctx.reset(42, &[]).unwrap();

// Make a move (center square = position 4)
let action = 4u32.to_le_bytes().to_vec();
let step = ctx.step(&reset.state, &action).unwrap();

// Check result
println!("Done: {}, Reward: {}", step.done, step.reward);
```

## Game Specification

### State

- **Board**: 9 cells, each 0 (empty), 1 (X), or 2 (O)
- **Current player**: 1 (X) or 2 (O)
- **Winner**: 0 (none), 1 (X wins), 2 (O wins), 3 (draw)

### Actions

Integer 0-8 representing board position:

```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

### Observation

27 floats for neural network input:

| Indices | Description |
|---------|-------------|
| 0-8 | Current player's pieces (1.0 where present) |
| 9-17 | Opponent's pieces (1.0 where present) |
| 18-26 | Legal moves mask (1.0 where legal) |

### Rewards

- **+1.0**: Win
- **-1.0**: Loss
- **0.0**: Draw or game continues

### Info Bits

The `info` field (u64) encodes the legal move mask in the lower 9 bits:
- Bit N is set if position N is a legal move
- Example: `0b111111111` = all positions legal (empty board)

## Encoding

State is encoded as 11 bytes:
- Bytes 0-8: Board cells
- Byte 9: Current player
- Byte 10: Winner

Action is encoded as 4 bytes (little-endian u32).

## Testing

```bash
cargo test
```

## Benchmarks

```bash
cargo bench
```

Benchmarks include:
- Reset performance
- Step performance
- Full episode simulation
