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

# Run training
python -m trainer --db ../data/replay.db --steps 1000

# With custom settings
python -m trainer \
    --db ../data/replay.db \
    --model-dir ../data/models \
    --steps 5000 \
    --batch-size 128 \
    --lr 0.001
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--db` | `./data/replay.db` | SQLite replay database path |
| `--model-dir` | `./data/models` | Directory for ONNX checkpoints |
| `--stats` | `./data/stats.json` | Stats file for web polling |
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
+-- replay.py      # SQLite replay buffer interface
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

## Integration with Actor

```bash
# Terminal 1: Actor generates data
cd actor
cargo run -- --env-id tictactoe --replay-db-path ../data/replay.db

# Terminal 2: Trainer consumes data
cd trainer
python -m trainer --db ../data/replay.db --model-dir ../data/models
```

The actor will hot-reload `model.onnx` when it changes.
