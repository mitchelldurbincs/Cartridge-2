# Enhanced Network Architecture Plan

## Problem Statement

The current `PolicyValueNetwork` in `trainer/src/trainer/network.py` uses a simple 3-layer MLP architecture. While adequate for TicTacToe (3x3 board), this architecture has significant limitations for spatially-structured games like Connect4 (7x6) and Othello (8x8):

1. **No spatial awareness**: Flattened input loses 2D structure
2. **No weight sharing**: Cannot generalize patterns across board positions
3. **Limited capacity**: Cannot efficiently learn "4-in-a-row" or flipping patterns
4. **No batch normalization**: Reduces training stability

## Proposed Solution

Implement a **configurable dual-architecture system** that selects the appropriate network based on game characteristics:

- **MLP Network**: For simple/small games (TicTacToe)
- **Convolutional ResNet**: For spatially-structured games (Connect4, Othello)

## Architecture Design

### 1. Convolutional ResNet (AlphaZero-style)

```
Input: (batch, channels, height, width)
  ↓
Initial Conv Block:
  Conv2d(in_channels, 128, 3x3, padding=1)
  BatchNorm2d(128)
  ReLU
  ↓
Residual Tower (N blocks):
  ┌─────────────────────────────┐
  │ Conv2d(128, 128, 3x3, pad=1)│
  │ BatchNorm2d(128)            │
  │ ReLU                        │
  │ Conv2d(128, 128, 3x3, pad=1)│
  │ BatchNorm2d(128)            │
  │ + skip connection           │
  │ ReLU                        │
  └─────────────────────────────┘
  ↓
Policy Head:
  Conv2d(128, 2, 1x1)
  BatchNorm2d(2)
  ReLU
  Flatten
  Linear → num_actions
  ↓
Value Head:
  Conv2d(128, 1, 1x1)
  BatchNorm2d(1)
  ReLU
  Flatten
  Linear(board_size, 256)
  ReLU
  Linear(256, 1)
  Tanh
```

### 2. Input Encoding Changes

Current flat encoding must be reshaped to spatial format:

**Connect4 (7x6):**
- Input channels: 3 (Red positions, Yellow positions, legal column broadcast)
- Shape: `(batch, 3, 6, 7)` - 3 channels, 6 rows, 7 columns

**Othello (8x8):**
- Input channels: 3 (Black positions, White positions, legal moves)
- Shape: `(batch, 3, 8, 8)`

**TicTacToe (3x3):**
- Can use either MLP or CNN (MLP preferred for simplicity)
- CNN shape: `(batch, 3, 3, 3)`

### 3. Configuration Schema

Extend `GameConfig` in `game_config.py`:

```python
@dataclass
class GameConfig:
    # ... existing fields ...

    # Network architecture selection
    network_type: str = "mlp"  # "mlp" or "resnet"

    # CNN-specific settings
    num_res_blocks: int = 4      # Number of residual blocks
    num_filters: int = 128       # Filters per conv layer
    input_channels: int = 2      # Channels for board encoding
```

**Recommended defaults:**
| Game      | network_type | num_res_blocks | num_filters |
|-----------|--------------|----------------|-------------|
| TicTacToe | mlp          | N/A            | N/A         |
| Connect4  | resnet       | 4              | 128         |
| Othello   | resnet       | 6              | 256         |

## Implementation Steps

### Step 1: Create ResNet Module

Add new file: `trainer/src/trainer/resnet.py`

```python
class ResidualBlock(nn.Module):
    """Single residual block with skip connection."""

class ConvPolicyValueNetwork(nn.Module):
    """AlphaZero-style convolutional network."""
```

Key components:
- `ResidualBlock`: Conv → BN → ReLU → Conv → BN → Skip → ReLU
- `ConvPolicyValueNetwork`: Initial conv + residual tower + dual heads
- Proper weight initialization (He initialization for conv layers)

### Step 2: Add Observation Reshaping

Create utility functions to reshape flat observations to spatial format:

```python
def reshape_observation(obs: torch.Tensor, config: GameConfig) -> torch.Tensor:
    """Reshape flat observation to (batch, channels, height, width)."""
```

This requires:
1. Parse board planes from observation vector
2. Reshape to `(batch, channels, height, width)`
3. Optionally broadcast legal moves to spatial format

### Step 3: Extend GameConfig

Update `game_config.py` with:
- `network_type` field
- CNN-specific hyperparameters
- Input channel count per game

### Step 4: Create Network Factory

Update `network.py`:

```python
def create_network(env_id: str) -> nn.Module:
    config = get_config(env_id)
    if config.network_type == "resnet":
        return ConvPolicyValueNetwork(config)
    else:
        return PolicyValueNetwork(config)
```

### Step 5: Update Training Loop

Modify `trainer.py` to:
1. Detect network type from config
2. Apply appropriate observation preprocessing
3. Handle different input shapes during training

### Step 6: Update ONNX Export

Ensure ONNX export handles both architectures:
- MLP: Input shape `(batch, obs_size)`
- CNN: Input shape `(batch, channels, height, width)`

The Rust ONNX evaluator in `mcts/src/onnx.rs` will need updates to handle CNN input format.

### Step 7: Update Rust ONNX Evaluator

Modify `engine/mcts/src/onnx.rs`:
1. Detect model input shape from ONNX metadata
2. Reshape observations appropriately before inference
3. Support both flat and spatial input formats

## Testing Strategy

### Unit Tests
1. `ResidualBlock` forward pass shape validation
2. `ConvPolicyValueNetwork` output shapes (policy, value)
3. Observation reshaping correctness
4. Network factory returns correct type

### Integration Tests
1. Training loop with CNN architecture completes without error
2. ONNX export/import roundtrip for CNN models
3. Rust evaluator correctly loads and runs CNN models

### Benchmarks
1. Compare MLP vs CNN training curves on Connect4
2. Measure inference latency (CNN will be slower but more accurate)
3. Evaluate against random baseline after N iterations

## Migration Path

### Phase 1: Add CNN (Non-Breaking)
- Add `resnet.py` with CNN implementation
- Add `network_type` config with default "mlp"
- Existing behavior unchanged

### Phase 2: Enable for Connect4
- Set Connect4 default to `network_type="resnet"`
- Update Rust evaluator for spatial input
- Validate training improvement

### Phase 3: Enable for Othello
- Configure Othello with appropriate CNN settings
- Test end-to-end training

## Performance Considerations

### Training Speed
CNN will be slower per step but should converge faster:
- MLP: ~10,000 steps/sec on CPU
- CNN: ~2,000-5,000 steps/sec on CPU
- CNN with GPU: ~20,000+ steps/sec

### Memory Usage
CNN uses more memory due to intermediate activations:
- MLP (Connect4): ~1MB per model
- CNN (Connect4, 4 blocks): ~5-10MB per model

### Inference Latency (MCTS)
Critical for self-play performance:
- MLP: ~0.1ms per inference
- CNN: ~0.5-1ms per inference

Mitigation: Use batched MCTS evaluation (already supported in `onnx.rs`).

## Alternative Approaches Considered

### 1. Attention/Transformer Architecture
- Pros: Can capture global patterns, flexible
- Cons: More complex, higher memory, slower inference
- Decision: **Defer** - ResNet is proven for board games

### 2. Deeper MLP
- Pros: Simple, fast inference
- Cons: Cannot learn spatial patterns efficiently
- Decision: **Rejected** - Fundamental limitation

### 3. Hybrid MLP + Position Encoding
- Pros: Some spatial awareness without full CNN
- Cons: Still inferior to CNN for spatial patterns
- Decision: **Rejected** - CNN is standard approach

## Success Metrics

1. **Training Efficiency**: CNN reaches 80% win rate vs random in 50% fewer iterations than MLP on Connect4
2. **Final Performance**: CNN achieves >90% win rate vs random on Connect4 (MLP plateaus lower)
3. **No Regression**: TicTacToe performance unchanged with MLP

## Files to Modify

| File | Changes |
|------|---------|
| `trainer/src/trainer/resnet.py` | **New** - CNN architecture |
| `trainer/src/trainer/network.py` | Add factory, keep MLP |
| `trainer/src/trainer/game_config.py` | Add CNN config fields |
| `trainer/src/trainer/trainer.py` | Handle CNN input shape |
| `engine/mcts/src/onnx.rs` | Support spatial input |
| `actor/src/mcts_policy.rs` | Update observation handling if needed |

## Timeline Estimate

Not provided - implementation complexity is moderate, approximately 5-7 focused development sessions.

## References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Original architecture
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) - Python reference implementation
- [Leela Chess Zero](https://lczero.org/) - Production AlphaZero implementation
