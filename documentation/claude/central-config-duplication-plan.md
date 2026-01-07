# Central Configuration Duplication Removal Plan

## Problem Statement

The central configuration loading code is duplicated across three files totaling ~1,324 lines:

| File | Lines | Scope |
|------|-------|-------|
| `actor/src/central_config.rs` | 687 | Full config (all sections) |
| `web/src/central_config.rs` | 325 | Subset (common, web, storage) |
| `trainer/src/trainer/central_config.py` | 312 | Full config (Python) |

### Duplicated Components (Rust)

1. **Default Constants Module** (~30 lines each)
   - `defaults::DATA_DIR`, `ENV_ID`, `LOG_LEVEL`, `HOST`, `PORT`, etc.

2. **Config Structs** (~200 lines each)
   - `CentralConfig`, `CommonConfig`, `WebConfig`, `StorageConfig`
   - `TrainingConfig`, `EvaluationConfig`, `ActorConfig`, `MctsConfig` (actor only)
   - Each struct has serde default functions and `impl Default`

3. **Env Override Macro** (~25 lines each)
   - `env_override!` macro with string, parse, and optional variants

4. **Config Loading Functions** (~50 lines each)
   - `CONFIG_SEARCH_PATHS` constant
   - `load_config()`, `load_from_path()`, `apply_env_overrides()`

5. **Tests** (~100 lines each)
   - Near-identical test cases for defaults, TOML parsing, env overrides

---

## Solution: Create `engine/engine-config` Crate

Create a new shared crate `engine/engine-config/` that contains all configuration structs, default values, and loading logic. Both `actor` and `web` crates will depend on this shared crate.

### Why This Approach?

1. **Single Source of Truth** - One place for all config structs and defaults
2. **Type Safety** - Shared types guarantee consistency between components
3. **Follows Existing Pattern** - The `engine/` workspace already has shared crates
4. **Minimal Migration** - Actor and web just change imports, no API changes

---

## Implementation Plan

### Phase 1: Create `engine/engine-config` Crate

#### Step 1.1: Create Crate Structure

```
engine/engine-config/
├── Cargo.toml
└── src/
    └── lib.rs
```

#### Step 1.2: Cargo.toml

```toml
[package]
name = "engine-config"
version = "0.1.0"
edition = "2021"
description = "Centralized configuration for Cartridge2"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
tracing = "0.1"
```

#### Step 1.3: lib.rs Structure

```rust
//! Centralized configuration loading from config.toml.
//!
//! This crate provides configuration structs and loading logic shared
//! across all Rust components (actor, web).

mod defaults;
mod structs;
mod loader;

pub use structs::*;
pub use loader::{load_config, load_from_path, apply_env_overrides};
```

#### Step 1.4: src/defaults.rs

Move all default constants here (merged from both files):

```rust
//! Default configuration values.

pub const DATA_DIR: &str = "./data";
pub const ENV_ID: &str = "tictactoe";
pub const LOG_LEVEL: &str = "info";

// Training defaults
pub const ITERATIONS: i32 = 100;
pub const START_ITERATION: i32 = 1;
pub const EPISODES_PER_ITERATION: i32 = 500;
pub const STEPS_PER_ITERATION: i32 = 1000;
pub const BATCH_SIZE: i32 = 64;
pub const LEARNING_RATE: f64 = 0.001;
pub const WEIGHT_DECAY: f64 = 0.0001;
pub const GRAD_CLIP_NORM: f64 = 1.0;
pub const DEVICE: &str = "cpu";
pub const CHECKPOINT_INTERVAL: i32 = 100;
pub const MAX_CHECKPOINTS: i32 = 10;

// Evaluation defaults
pub const EVAL_INTERVAL: i32 = 1;
pub const EVAL_GAMES: i32 = 50;

// Actor defaults
pub const ACTOR_ID: &str = "actor-1";
pub const MAX_EPISODES: i32 = -1;
pub const EPISODE_TIMEOUT_SECS: u64 = 30;
pub const FLUSH_INTERVAL_SECS: u64 = 5;
pub const LOG_INTERVAL: u32 = 50;

// Web defaults
pub const HOST: &str = "0.0.0.0";
pub const PORT: u16 = 8080;

// MCTS defaults
pub const NUM_SIMULATIONS: u32 = 800;
pub const C_PUCT: f64 = 1.4;
pub const TEMPERATURE: f64 = 1.0;
pub const DIRICHLET_ALPHA: f64 = 0.3;
pub const DIRICHLET_WEIGHT: f64 = 0.25;

// Storage defaults
pub const MODEL_BACKEND: &str = "filesystem";
pub const POSTGRES_URL: &str = "postgresql://cartridge:cartridge@localhost:5432/cartridge";
```

#### Step 1.5: src/structs.rs

All config structs with serde defaults (merged superset):

```rust
use serde::Deserialize;
use crate::defaults;

// ... serde default functions ...

#[derive(Debug, Deserialize, Default, Clone)]
pub struct CentralConfig {
    #[serde(default)]
    pub common: CommonConfig,
    #[serde(default)]
    pub training: TrainingConfig,
    #[serde(default)]
    pub evaluation: EvaluationConfig,
    #[serde(default)]
    pub actor: ActorConfig,
    #[serde(default)]
    pub web: WebConfig,
    #[serde(default)]
    pub mcts: MctsConfig,
    #[serde(default)]
    pub storage: StorageConfig,
}

// CommonConfig, TrainingConfig, EvaluationConfig, ActorConfig,
// WebConfig, MctsConfig, StorageConfig - all with impl Default
```

#### Step 1.6: src/loader.rs

Config loading logic with env_override macro:

```rust
use std::path::PathBuf;
use tracing::{debug, info, warn};
use crate::CentralConfig;

pub const CONFIG_SEARCH_PATHS: &[&str] = &[
    "config.toml",
    "../config.toml",
    "/app/config.toml",
];

macro_rules! env_override {
    // ... macro implementation ...
}

pub fn load_config() -> CentralConfig { ... }
pub fn load_from_path(path: &PathBuf) -> CentralConfig { ... }
pub fn apply_env_overrides(config: CentralConfig) -> CentralConfig { ... }
```

### Phase 2: Update Engine Workspace

#### Step 2.1: Add to engine/Cargo.toml

```toml
[workspace]
members = [
    "engine-core",
    "engine-config",  # ADD THIS
    "engine-games",
    "games-tictactoe",
    "games-connect4",
    "mcts",
    "model-watcher"
]

[workspace.dependencies]
# ... existing deps ...
toml = "0.8"  # ADD THIS
```

### Phase 3: Migrate Actor Crate

#### Step 3.1: Update actor/Cargo.toml

```toml
[dependencies]
engine-config = { path = "../engine/engine-config" }  # ADD THIS
# Remove toml if not used elsewhere
```

#### Step 3.2: Update actor/src/central_config.rs

Replace entire file with re-export:

```rust
//! Re-export centralized configuration from engine-config.

pub use engine_config::*;
```

Or delete the file entirely and update imports in other actor modules.

#### Step 3.3: Update Actor Imports

In `actor/src/main.rs`, `actor/src/config.rs`, etc.:

```rust
// Before
use crate::central_config::{load_config, CentralConfig};

// After
use engine_config::{load_config, CentralConfig};
```

### Phase 4: Migrate Web Crate

#### Step 4.1: Update web/Cargo.toml

```toml
[dependencies]
engine-config = { path = "../engine/engine-config" }  # ADD THIS
# toml already present, keep if used elsewhere
```

#### Step 4.2: Delete web/src/central_config.rs

The entire file can be deleted as all functionality moves to engine-config.

#### Step 4.3: Update Web Imports

In `web/src/main.rs`, `web/src/model_watcher.rs`, etc.:

```rust
// Before
use crate::central_config::{load_config, CentralConfig};

// After
use engine_config::{load_config, CentralConfig};
```

### Phase 5: Python Trainer Alignment (Documentation Only)

The Python trainer must maintain its own implementation (different language), but should follow the same schema. Document the schema contract:

#### Step 5.1: Add Schema Documentation

Create `engine/engine-config/SCHEMA.md`:

```markdown
# Config.toml Schema

This document defines the canonical config.toml schema used by all components.
The Python trainer (trainer/src/trainer/central_config.py) must stay aligned
with this schema.

## Sections

### [common]
- data_dir: string = "./data"
- env_id: string = "tictactoe"
- log_level: string = "info"

### [training]
- iterations: i32 = 100
- start_iteration: i32 = 1
- episodes_per_iteration: i32 = 500
- steps_per_iteration: i32 = 1000
- batch_size: i32 = 64
- learning_rate: f64 = 0.001
- weight_decay: f64 = 0.0001
- grad_clip_norm: f64 = 1.0
- device: string = "cpu"
- checkpoint_interval: i32 = 100
- max_checkpoints: i32 = 10

### [evaluation]
- interval: i32 = 1
- games: i32 = 50

### [actor]
- actor_id: string = "actor-1"
- max_episodes: i32 = -1
- episode_timeout_secs: u64 = 30
- flush_interval_secs: u64 = 5
- log_interval: u32 = 50

### [web]
- host: string = "0.0.0.0"
- port: u16 = 8080

### [mcts]
- num_simulations: u32 = 800
- c_puct: f64 = 1.4
- temperature: f64 = 1.0
- dirichlet_alpha: f64 = 0.3
- dirichlet_weight: f64 = 0.25

### [storage]
- model_backend: string = "filesystem"
- postgres_url: string? = "postgresql://cartridge:cartridge@localhost:5432/cartridge"
- s3_bucket: string? = None
- s3_endpoint: string? = None

## Environment Variable Overrides

Pattern: CARTRIDGE_<SECTION>_<KEY>=value

Examples:
- CARTRIDGE_COMMON_ENV_ID=connect4
- CARTRIDGE_TRAINING_ITERATIONS=50
- CARTRIDGE_WEB_PORT=3000
```

### Phase 6: Update Tests

#### Step 6.1: Consolidate Tests in engine-config

Move tests from both actor and web into `engine/engine-config/src/lib.rs` or a separate `tests/` directory.

#### Step 6.2: Run Full Test Suite

```bash
cd engine && cargo test
cd actor && cargo test
cd web && cargo test
```

### Phase 7: Update Documentation

#### Step 7.1: Update CLAUDE.md

Add engine-config to the directory structure and component list.

#### Step 7.2: Update README references

Ensure any docs referencing configuration point to the shared crate.

---

## File Changes Summary

### Files to Create

| File | Description |
|------|-------------|
| `engine/engine-config/Cargo.toml` | Crate configuration |
| `engine/engine-config/src/lib.rs` | Main module |
| `engine/engine-config/src/defaults.rs` | Default constants |
| `engine/engine-config/src/structs.rs` | Config struct definitions |
| `engine/engine-config/src/loader.rs` | Loading logic |
| `engine/engine-config/SCHEMA.md` | Schema documentation |

### Files to Modify

| File | Change |
|------|--------|
| `engine/Cargo.toml` | Add engine-config to workspace members |
| `actor/Cargo.toml` | Add engine-config dependency |
| `web/Cargo.toml` | Add engine-config dependency |
| `actor/src/*.rs` | Update imports |
| `web/src/*.rs` | Update imports |
| `CLAUDE.md` | Add engine-config documentation |

### Files to Delete

| File | Reason |
|------|--------|
| `actor/src/central_config.rs` | Moved to engine-config |
| `web/src/central_config.rs` | Moved to engine-config |

---

## Lines of Code Impact

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| actor/central_config.rs | 687 | 0 | -687 |
| web/central_config.rs | 325 | 0 | -325 |
| engine-config (new) | 0 | ~400 | +400 |
| **Total Rust LOC** | 1,012 | ~400 | **-612 (60%)** |

The Python trainer keeps its ~312 lines (different language constraint).

---

## Verification Checklist

- [ ] All tests pass: `cargo test` in engine/, actor/, web/
- [ ] Docker build succeeds: `docker compose build`
- [ ] Actor runs with config.toml: `cargo run -p actor`
- [ ] Web runs with config.toml: `cargo run -p web`
- [ ] Python trainer reads same config.toml correctly
- [ ] Environment variable overrides work for all components
- [ ] CI pipeline passes

---

## Rollback Plan

If issues arise, the migration can be rolled back by:

1. Reverting Cargo.toml changes in actor and web
2. Restoring deleted central_config.rs files from git
3. Removing engine-config from workspace

The git history preserves all original files.

---

## Future Improvements

1. **JSON Schema Generation** - Generate JSON Schema from Rust structs for validation
2. **Config Validation** - Add validation functions (e.g., port in range, valid device)
3. **Hot Reload** - Support config hot-reload without restart
4. **Python Codegen** - Consider generating Python dataclasses from Rust structs
