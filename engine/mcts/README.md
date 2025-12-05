# mcts

Monte Carlo Tree Search implementation for AlphaZero-style game playing. Works with any game implementing the engine-core Game trait.

## Overview

MCTS builds a search tree by running simulations. Each simulation has four phases:

1. **Selection** - Traverse tree using PUCT (Polynomial UCT) formula
2. **Expansion** - Add child nodes for each legal action at leaf
3. **Evaluation** - Get policy/value estimate from evaluator
4. **Backpropagation** - Update visit counts and values up to root

## Usage

```rust
use mcts::{MctsConfig, UniformEvaluator, run_mcts};
use engine_core::EngineContext;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

// Register game
games_tictactoe::register_tictactoe();

// Create game context
let mut ctx = EngineContext::new("tictactoe").unwrap();
let reset = ctx.reset(42, &[]).unwrap();

// Set up MCTS
let evaluator = UniformEvaluator::new();
let config = MctsConfig::for_training()
    .with_simulations(800)
    .with_temperature(1.0);

// Run search
let legal_mask = reset.info & 0x1FF;  // Lower 9 bits
let mut rng = ChaCha20Rng::seed_from_u64(42);
let result = run_mcts(&mut ctx, &evaluator, config, reset.state, legal_mask, &mut rng).unwrap();

println!("Best action: {}", result.action);
println!("Policy: {:?}", result.policy);
println!("Value: {}", result.value);
```

## Configuration

```rust
let config = MctsConfig {
    num_simulations: 800,    // Simulations per search
    c_puct: 1.25,            // Exploration constant
    dirichlet_alpha: 0.3,    // Root noise for exploration
    dirichlet_epsilon: 0.25, // Weight of noise vs prior
    temperature: 1.0,        // Action selection temperature
};

// Presets
let training = MctsConfig::for_training();  // Exploratory
let testing = MctsConfig::for_testing();    // Fewer sims
let playing = MctsConfig::for_playing();    // Greedy (temp=0)
```

## Evaluators

The `Evaluator` trait provides policy priors and value estimates:

```rust
pub trait Evaluator: Send + Sync {
    fn evaluate(
        &self,
        state: &[u8],
        legal_mask: u64,
        action_space_size: usize,
    ) -> Result<EvalResult, EvaluatorError>;
}

pub struct EvalResult {
    pub policy: Vec<f32>,  // Prior probabilities
    pub value: f32,        // State value estimate [-1, 1]
}
```

### Built-in Evaluators

- **UniformEvaluator** - Returns uniform policy over legal moves (for testing)

### ONNX Evaluator (optional)

Enable the `onnx` feature for neural network inference:

```bash
cargo build --features onnx
```

```rust
use mcts::OnnxEvaluator;

let evaluator = OnnxEvaluator::new("model.onnx")?;
```

## Module Structure

```
src/
|-- lib.rs        # Public exports
|-- config.rs     # MctsConfig
|-- evaluator.rs  # Evaluator trait, UniformEvaluator
|-- node.rs       # MctsNode (visit_count, value_sum, prior, children)
|-- tree.rs       # MctsTree with arena allocation
|-- search.rs     # Selection, expansion, backprop, run_mcts
+-- onnx.rs       # OnnxEvaluator (feature-gated)
```

## Search Result

```rust
pub struct SearchResult {
    pub action: u32,        // Best action index
    pub policy: Vec<f32>,   // Visit count distribution
    pub value: f32,         // Root value estimate
    pub visits: u32,        // Total simulations run
}
```

## Testing

```bash
# All tests
cargo test

# With ONNX support
cargo test --features onnx
```

## Dependencies

- `engine-core` - Game abstraction
- `rand` / `rand_chacha` - Deterministic randomness
- `rand_distr` - Dirichlet distribution for root noise
- `ort` / `ndarray` - ONNX Runtime (optional)
