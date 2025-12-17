# Trainer

Python training loop for Cartridge2. Implements AlphaZero-style learning from self-play data.

## Overview

The trainer:
1. Reads transitions from SQLite replay buffer
2. Trains a PyTorch neural network
3. Exports ONNX models for the Rust actor
4. Writes stats.json for the web frontend

## Quick Start

```bash
# Install
pip install -e .

# Run training (defaults assume running from trainer/ directory)
python -m trainer --steps 1000

# With custom settings
python -m trainer \
    --steps 5000 \
    --batch-size 128 \
    --lr 0.001
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--db` | `../data/replay.db` | SQLite replay database path |
| `--model-dir` | `../data/models` | Directory for ONNX checkpoints |
| `--stats` | `../data/stats.json` | Stats file for web polling |
| `--steps` | 1000 | Total training steps |
| `--batch-size` | 64 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--weight-decay` | 0.0001 | L2 regularization |
| `--grad-clip` | 1.0 | Gradient clipping norm |
| `--checkpoint-interval` | 100 | Steps between saves |
| `--device` | cpu | Training device (cpu/cuda/mps) |

### LR Schedule

```bash
# Disable cosine annealing
python -m trainer --no-lr-schedule

# Custom min LR ratio
python -m trainer --lr-min-ratio 0.01
```

### Wait Settings

The trainer waits for the replay database to exist and contain data:

```bash
# Custom wait interval and timeout
python -m trainer --wait-interval 5.0 --max-wait 600
```

## Architecture

```
+-------------------+     +------------------+     +------------------+
|  SQLite Replay    |---->|  PyTorch Model   |---->|  ONNX Export     |
|  (transitions)    |     |  (policy+value)  |     |  (model.onnx)    |
+-------------------+     +------------------+     +------------------+
                                  |
                                  v
                          +------------------+
                          |   stats.json     |
                          |   (telemetry)    |
                          +------------------+
```

## Loss Function

AlphaZero-style combined loss:

```
L = L_policy + L_value

L_policy = -sum(pi * log(p))    # Cross-entropy with MCTS policy
L_value  = (z - v)^2            # MSE with game outcome
```

## Model Export

Models are exported with atomic write-then-rename:

1. Train PyTorch model
2. Export to `temp_model.onnx`
3. Rename to `model.onnx`

This prevents the Rust actor from loading a partially-written file.

## Stats Output

The trainer writes `stats.json` for the web frontend:

```json
{
  "step": 1000,
  "total_loss": 0.523,
  "policy_loss": 0.412,
  "value_loss": 0.111,
  "learning_rate": 0.0001,
  "timestamp": 1699999999
}
```

## Module Structure

```
src/trainer/
|-- __init__.py
|-- __main__.py    # CLI entrypoint
|-- trainer.py     # Training loop, TrainerConfig
|-- network.py     # Neural network architecture
|-- replay.py      # SQLite replay buffer interface
|-- evaluator.py   # Model evaluation against baselines
+-- game_config.py # Game-specific configurations (TicTacToe, Connect4)
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

## Dependencies

- `torch>=2.0.0` - Neural network training
- `numpy>=1.24.0` - Numerical operations
- `onnx>=1.14.0` - Model export
- `onnxruntime>=1.15.0` - Model inference for evaluation

## Evaluator

The evaluator measures how well a trained model plays against random opponents.

### Quick Start

```bash
# Basic evaluation (100 games)
python -m trainer.evaluator --model ../data/models/latest.onnx --games 100

# More games for statistical confidence
python -m trainer.evaluator --model ../data/models/latest.onnx --games 500

# Verbose mode to see individual game moves
python -m trainer.evaluator --model ../data/models/latest.onnx --games 10 --verbose

# Compare different checkpoints
python -m trainer.evaluator --model ../data/models/model_step_000100.onnx --games 100
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `../data/models/latest.onnx` | Path to ONNX model file |
| `--db` | `../data/replay.db` | Replay database for loading metadata (self-describing configs) |
| `--env-id` | `tictactoe` | Game environment to evaluate |
| `--games` | 100 | Number of games to play |
| `--temperature` | 0.0 | Sampling temperature (0 = greedy/argmax) |
| `--verbose` | false | Print individual game moves |
| `--log-level` | INFO | Logging level |

### Output Example

```
==================================================
Evaluation Results: ONNX(latest.onnx) vs Random
==================================================
Games played: 100

ONNX(latest.onnx):
  Wins: 72 (72.0%)
    As X (first): 45
    As O (second): 27

Random:
  Wins: 12 (12.0%)
    As X (first): 8
    As O (second): 4

Draws: 16 (16.0%)
Average game length: 6.8 moves
==================================================

✓ Model is significantly better than random play!
```

### Interpreting Results

| Win Rate | Interpretation |
|----------|----------------|
| >70% | Model is significantly better than random |
| 50-70% | Model is slightly better than random |
| 30-50% | Model is roughly equivalent to random |
| <30% | Model is worse than random |

For TicTacToe specifically:
- A well-trained model should achieve **70%+ win rate** vs random
- High draw rates (>50%) indicate strong defensive play
- First-player (X) advantage is expected - watch for parity between X and O performance

### Evaluating Training Progress

Compare checkpoints to see learning progress:

```bash
# Early checkpoint
python -m trainer.evaluator --model ./data/models/model_step_000100.onnx --games 100

# Mid checkpoint
python -m trainer.evaluator --model ./data/models/model_step_000500.onnx --games 100

# Final model
python -m trainer.evaluator --model ./data/models/latest.onnx --games 100
```

## Integration with Actor

```bash
# Terminal 1: Actor generates data
cd actor
cargo run -- --env-id tictactoe --replay-db-path ../data/replay.db

# Terminal 2: Trainer consumes data (defaults work from trainer/ directory)
cd trainer
python -m trainer
```

The actor will hot-reload `model.onnx` when it changes.
