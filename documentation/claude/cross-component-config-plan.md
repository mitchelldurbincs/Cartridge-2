# Cross-Component Configuration Duplication Mitigation Plan

## Executive Summary

Configuration loading is duplicated across three files totaling ~1,400 lines with 80%+ semantic overlap. This creates maintenance burden and risks configuration drift between components.

**Current State:**
| File | Lines | Language | Scope |
|------|-------|----------|-------|
| `actor/src/central_config.rs` | 751 | Rust | Full config (all sections) |
| `web/src/central_config.rs` | 326 | Rust | Subset (common, web, storage) |
| `trainer/src/trainer/central_config.py` | 315 | Python | Full config (all sections) |

**Core Problem:** Same defaults, same environment override logic, same TOML structure defined 3 times. Adding a config field requires changes in up to 3 places.

---

## Analysis of Duplication

### What's Actually Duplicated

| Component | Actor | Web | Python | Notes |
|-----------|-------|-----|--------|-------|
| Default constants | 35 | 8 | N/A (inline) | Web is subset of actor |
| Config structs | 7 | 3 | 6 | Python uses dataclasses |
| Serde/default functions | 30 | 7 | N/A | Python infers types |
| env_override! macro | 1 | 1 | 1 func | Same pattern 3x |
| load_config() | 1 | 1 | 1 | Nearly identical |
| CONFIG_SEARCH_PATHS | 3 paths | 3 paths | 3 paths | Identical |
| apply_env_overrides() | 50 calls | 12 calls | loop | Actor has most |
| Tests | 8 | 6 | 0 | Similar patterns |

### Semantic Drift Risk

Currently these defaults **differ** between implementations:

| Field | Actor (Rust) | Python | Issue |
|-------|--------------|--------|-------|
| `mcts.eval_batch_size` | 32 | 1 | Different defaults! |
| `mcts.start_sims` | N/A | 50 | Python-only field |
| `mcts.max_sims` | N/A | 400 | Python-only field |
| `mcts.sim_ramp_rate` | N/A | 20 | Python-only field |
| `evaluation.win_threshold` | N/A | 0.55 | Python-only field |
| `evaluation.eval_vs_random` | N/A | true | Python-only field |
| `training.num_actors` | N/A | 1 | Python-only field |
| `storage.pool_*` | present | N/A | Rust-only fields |

This drift is the real problem - the files look similar but have diverged.

---

## Solution Options

### Option A: Shared Rust Crate (Existing Plan)

**See:** `documentation/claude/central-config-duplication-plan.md`

Creates `engine/engine-config` crate shared by actor and web.

**Pros:**
- Eliminates Rust duplication (612 lines saved)
- Type-safe sharing between Rust components
- Follows existing workspace pattern

**Cons:**
- Doesn't solve Python duplication
- Still need schema documentation for Python alignment
- Two sources of truth remain (Rust crate + Python)

**Recommendation:** Do this regardless of other options.

---

### Option B: Schema-First with Code Generation

Define configuration schema once, generate code for both languages.

#### Implementation

1. **Create JSON Schema** (`config.schema.json`):
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "common": {
      "type": "object",
      "properties": {
        "data_dir": { "type": "string", "default": "./data" },
        "env_id": { "type": "string", "default": "tictactoe" },
        "log_level": { "type": "string", "default": "info" }
      }
    }
    // ... all sections
  }
}
```

2. **Generate Rust code** using `schemafy` or custom build script
3. **Generate Python code** using `datamodel-code-generator`

**Pros:**
- True single source of truth
- Automatic validation
- IDE support via JSON Schema

**Cons:**
- Added build complexity
- Generated code can be harder to debug
- Schema format is verbose

**Recommendation:** Good for large projects, overkill for Cartridge2's current size.

---

### Option C: TOML Defaults File (Recommended Complement)

Create a canonical defaults file that both languages read at build/import time.

#### Implementation

1. **Create `config.defaults.toml`** at project root:

```toml
# Canonical default values for all components.
# This file is the single source of truth for default configuration.
# Both Rust and Python read these defaults.

[common]
data_dir = "./data"
env_id = "tictactoe"
log_level = "info"

[training]
iterations = 100
start_iteration = 1
episodes_per_iteration = 500
steps_per_iteration = 1000
batch_size = 64
learning_rate = 0.001
weight_decay = 0.0001
grad_clip_norm = 1.0
device = "cpu"
checkpoint_interval = 100
max_checkpoints = 10
num_actors = 1

[evaluation]
interval = 1
games = 50
win_threshold = 0.55
eval_vs_random = true

[actor]
actor_id = "actor-1"
max_episodes = -1
episode_timeout_secs = 30
flush_interval_secs = 5
log_interval = 50

[web]
host = "0.0.0.0"
port = 8080

[mcts]
num_simulations = 800
c_puct = 1.4
temperature = 1.0
temp_threshold = 0
dirichlet_alpha = 0.3
dirichlet_weight = 0.25
start_sims = 50
max_sims = 400
sim_ramp_rate = 20
eval_batch_size = 32

[storage]
model_backend = "filesystem"
postgres_url = "postgresql://cartridge:cartridge@localhost:5432/cartridge"
pool_max_size = 16
pool_connect_timeout = 30
pool_idle_timeout = 300
```

2. **Rust: Embed at compile time**

```rust
// engine/engine-config/src/defaults.rs
const DEFAULTS_TOML: &str = include_str!("../../../config.defaults.toml");

lazy_static! {
    pub static ref DEFAULTS: CentralConfig =
        toml::from_str(DEFAULTS_TOML).expect("defaults.toml must be valid");
}
```

3. **Python: Load at import time**

```python
# trainer/src/trainer/central_config.py
from pathlib import Path
import tomllib

_DEFAULTS_PATH = Path(__file__).parent.parent.parent.parent / "config.defaults.toml"

def _load_defaults() -> dict:
    with open(_DEFAULTS_PATH, "rb") as f:
        return tomllib.load(f)

_DEFAULTS = _load_defaults()
```

4. **Merge defaults with user config**

Both languages load `config.defaults.toml` first, then overlay `config.toml`.

**Pros:**
- True single source of truth for defaults
- Human-readable/editable format
- No code generation needed
- Easy to add new fields

**Cons:**
- Runtime file dependency (mitigated by include_str! in Rust)
- Struct definitions still duplicated (just not defaults)

---

### Option D: Simplified Config Hierarchy

Reduce scope of what's configurable to minimize duplication impact.

#### Current: 7 sections, 40+ fields

Many fields are rarely changed. Consider:

1. **Move MCTS params to per-game metadata** - already derived from GameMetadata
2. **Remove training/evaluation from actor/web** - they don't need these
3. **Hardcode storage pool settings** - rarely customized

#### Reduced Config

```toml
[common]
data_dir = "./data"
env_id = "tictactoe"
log_level = "info"

[web]
host = "0.0.0.0"
port = 8080

[storage]
backend = "filesystem"
postgres_url = "..."  # only if backend = postgres

# Everything else uses sensible defaults or derives from env_id
```

**Pros:**
- Less to duplicate
- Simpler user experience
- Fewer knobs = fewer bugs

**Cons:**
- Less flexible for advanced users
- Breaks existing workflows using these fields

**Recommendation:** Consider for v2, but too disruptive now.

---

## Recommended Approach: A + C Combined

### Phase 1: Create Shared Rust Crate (Option A)

Follow existing plan in `central-config-duplication-plan.md`:

1. Create `engine/engine-config` crate
2. Move all Rust config code there
3. Update actor and web to import from shared crate
4. Delete duplicate `central_config.rs` files

**Timeline:** 2-4 hours
**Lines Saved:** ~612 Rust lines

### Phase 2: Create Defaults File (Option C)

1. Create `config.defaults.toml` with canonical defaults
2. Update `engine/engine-config` to load defaults from this file
3. Update Python `central_config.py` to load defaults from this file
4. Remove hardcoded defaults from both implementations

**Timeline:** 1-2 hours
**Benefit:** Single source of truth for all default values

### Phase 3: Align Python Struct Fields

1. Review Python dataclasses against Rust structs
2. Add missing fields to both (see drift table above)
3. Ensure types match (i32 vs int, f64 vs float, etc.)
4. Add CI check that validates schema alignment

**Timeline:** 1 hour
**Benefit:** Prevents future drift

### Phase 4: Add Schema Validation (Optional)

1. Generate JSON Schema from `config.defaults.toml`
2. Use for IDE autocomplete and validation
3. Add to CI to catch config.toml errors

**Timeline:** 2 hours
**Benefit:** Better developer experience

---

## Implementation Checklist

### Phase 1: Rust Crate
- [ ] Create `engine/engine-config/Cargo.toml`
- [ ] Create `engine/engine-config/src/lib.rs` with structs
- [ ] Add to `engine/Cargo.toml` workspace members
- [ ] Update `actor/Cargo.toml` dependency
- [ ] Update `web/Cargo.toml` dependency
- [ ] Update imports in `actor/src/*.rs`
- [ ] Update imports in `web/src/*.rs`
- [ ] Delete `actor/src/central_config.rs`
- [ ] Delete `web/src/central_config.rs`
- [ ] Run `cargo test` in all crates
- [ ] Run `cargo clippy` with no warnings

### Phase 2: Defaults File
- [ ] Create `config.defaults.toml` at project root
- [ ] Update `engine-config` to use `include_str!`
- [ ] Update Python to load defaults file
- [ ] Remove hardcoded defaults from Python dataclasses
- [ ] Verify `python -m trainer loop --help` shows correct defaults
- [ ] Verify `cargo run -p actor -- --help` shows correct defaults

### Phase 3: Schema Alignment
- [ ] Add `num_actors` to Rust TrainingConfig
- [ ] Add `win_threshold`, `eval_vs_random` to Rust EvaluationConfig
- [ ] Add `temp_threshold`, `start_sims`, `max_sims`, `sim_ramp_rate` to Rust MctsConfig
- [ ] Add `pool_*` fields to Python StorageConfig (or document as Rust-only)
- [ ] Reconcile `eval_batch_size` default (32 vs 1)
- [ ] Update tests for new fields

### Phase 4: Documentation
- [ ] Update `CLAUDE.md` with new crate
- [ ] Create `engine/engine-config/README.md`
- [ ] Document environment variable patterns
- [ ] Add config troubleshooting guide

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing scripts | Medium | High | Test all CLI commands before merge |
| Python import failure | Low | High | Add fallback to inline defaults |
| Drift after migration | Medium | Medium | Add CI schema check |
| Build time increase | Low | Low | include_str! is fast |

---

## Success Metrics

1. **Lines of Code:** Reduce config-related code by 50%+ (1400 → <700)
2. **Single Source of Truth:** All defaults in one file
3. **Zero Drift:** Automated check prevents divergence
4. **Same Behavior:** All existing config.toml files work unchanged

---

## Future Considerations

1. **Config Versioning:** Add `version` field for migration support
2. **Config Validation:** Validate at load time (port ranges, valid devices, etc.)
3. **Hot Reload:** Support config changes without restart
4. **Web UI:** Expose config editing via web interface
