# engine-core

Core abstractions for the Cartridge2 game engine. Provides the Game trait, type erasure, registry system, and high-level EngineContext API.

## Overview

This crate defines the fundamental interfaces that all games must implement:

- **Game trait** - Strongly-typed interface for game logic
- **ErasedGame** - Runtime interface working with raw bytes
- **GameAdapter** - Automatic conversion from typed to erased
- **Registry** - Static game registration and lookup
- **EngineContext** - High-level API for game simulation

## Usage

### Implementing a Game

```rust
use engine_core::{Game, ActionSpace, EngineId, Capabilities, Encoding};
use engine_core::typed::{EncodeError, DecodeError};
use rand_chacha::ChaCha20Rng;

#[derive(Debug)]
struct MyGame;

impl Game for MyGame {
    type State = MyState;
    type Action = MyAction;
    type Obs = MyObs;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "mygame".to_string(),
            build_id: "0.1.0".to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding { /* ... */ },
            max_horizon: 100,
            action_space: ActionSpace::Discrete(9),
            preferred_batch: 32,
        }
    }

    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8]) -> (Self::State, Self::Obs) {
        // Initialize game state
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64) {
        // Execute action, return (obs, reward, done, info)
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> { /* ... */ }
    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> { /* ... */ }
    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> { /* ... */ }
    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> { /* ... */ }
    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> { /* ... */ }
}
```

### Registering a Game

```rust
use engine_core::{register_game, GameAdapter};

pub fn register_mygame() {
    register_game("mygame".to_string(), || {
        Box::new(GameAdapter::new(MyGame))
    });
}
```

### Using EngineContext

```rust
use engine_core::EngineContext;

// Register game first
register_mygame();

// Create context
let mut ctx = EngineContext::new("mygame").expect("game registered");

// Reset to initial state
let reset = ctx.reset(42, &[]).unwrap();
println!("Initial state: {} bytes", reset.state.len());

// Take a step
let action = 4u32.to_le_bytes().to_vec();
let step = ctx.step(&reset.state, &action).unwrap();
println!("Reward: {}, Done: {}", step.reward, step.done);
```

## Action Spaces

Three action space types are supported:

```rust
// Single discrete action (0..n)
ActionSpace::Discrete(9)

// Multiple discrete dimensions
ActionSpace::MultiDiscrete(vec![3, 4, 5])

// Continuous actions with bounds
ActionSpace::Continuous {
    low: vec![-1.0, -1.0],
    high: vec![1.0, 1.0],
    shape: vec![2],
}
```

## Module Structure

```
src/
|-- lib.rs        # Public exports
|-- typed.rs      # Game trait definition
|-- adapter.rs    # GameAdapter (typed -> erased)
|-- erased.rs     # ErasedGame trait
|-- context.rs    # EngineContext high-level API
+-- registry.rs   # Static game registration
```

## Testing

```bash
cargo test
```

Tests cover:
- Game trait basic functionality
- State/action/observation encoding roundtrips
- Multi-discrete and continuous action spaces
- Registry operations
- EngineContext API
