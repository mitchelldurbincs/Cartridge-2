# Orchestrator Refactoring Plan

## Problem

`trainer/src/trainer/orchestrator.py` is 1257 lines - too large for a single module. It handles multiple concerns that should be separated for better maintainability, testability, and readability.

## Current Structure Analysis

| Section | Lines | Description |
|---------|-------|-------------|
| Dataclasses | 49-144 (~95 lines) | `IterationStats`, `LoopConfig` |
| Orchestrator.__init__ + signals | 146-169 (~25 lines) | Initialization |
| Best model tracking | 171-208 (~40 lines) | Load/save/promote best model |
| Auto-resume | 210-281 (~70 lines) | Resume from loop_stats.json |
| Actor binary finder | 283-333 (~50 lines) | Find actor binary |
| Directory/buffer ops | 335-360 (~25 lines) | Ensure dirs, clear buffer |
| Actor runner | 362-501 (~140 lines) | Spawn and manage actor processes |
| Trainer runner | 503-546 (~45 lines) | Run training |
| Evaluation runner | 548-675 (~130 lines) | Run evaluation vs best/random |
| Stats management | 677-783 (~105 lines) | Save loop/eval stats, update stats.json |
| run_iteration | 785-887 (~100 lines) | Single iteration logic |
| run | 889-1008 (~120 lines) | Main loop |
| CLI parse_args | 1010-1226 (~215 lines) | Argument parsing |
| main | 1229-1256 (~30 lines) | Entry point |

## Proposed Module Structure

```
trainer/src/trainer/
├── orchestrator/
│   ├── __init__.py          # Re-exports Orchestrator, LoopConfig, main()
│   ├── config.py            # LoopConfig, IterationStats dataclasses
│   ├── actor_runner.py      # ActorRunner class (find binary, spawn actors)
│   ├── eval_runner.py       # EvalRunner class (evaluation, best model tracking)
│   ├── stats_manager.py     # StatsManager class (save/load stats)
│   ├── cli.py               # parse_args(), main()
│   └── orchestrator.py      # Orchestrator class (uses above components)
├── orchestrator.py          # Thin wrapper for backwards compatibility (deprecated)
```

## Module Breakdown

### 1. `config.py` (~100 lines)

```python
# Contents:
@dataclass
class IterationStats: ...

@dataclass
class LoopConfig: ...
```

This is a clean separation - configuration dataclasses with no dependencies on other orchestrator modules.

### 2. `actor_runner.py` (~200 lines)

```python
class ActorRunner:
    def __init__(self, config: LoopConfig): ...
    def find_binary(self) -> Path: ...
    def run(self, num_episodes: int, iteration: int) -> tuple[bool, float]: ...
```

Encapsulates:
- Binary discovery logic
- Process spawning
- Output streaming with threads
- Graceful shutdown handling

### 3. `eval_runner.py` (~200 lines)

```python
class EvalRunner:
    def __init__(self, config: LoopConfig): ...
    def load_best_model_info(self) -> None: ...
    def promote_to_best(self, iteration: int, win_rate: float) -> None: ...
    def run(self, iteration: int) -> tuple[float | None, float | None, float]: ...

    @property
    def best_model_iteration(self) -> int | None: ...
```

Encapsulates:
- Best model (gatekeeper) tracking
- Model vs best evaluation
- Model vs random evaluation
- Best model promotion logic

### 4. `stats_manager.py` (~150 lines)

```python
class StatsManager:
    def __init__(self, config: LoopConfig): ...
    def save_loop_stats(self, history: list[IterationStats]) -> None: ...
    def save_eval_stats(self, eval_history: list[dict]) -> None: ...
    def update_stats_with_eval(self, eval_history: list[dict], best_iteration: int | None) -> None: ...
    def load_previous_state(self) -> tuple[list[IterationStats], list[dict], int]: ...
```

Encapsulates:
- loop_stats.json management
- eval_stats.json management
- stats.json updating for frontend
- Auto-resume state loading

### 5. `cli.py` (~250 lines)

```python
def parse_args() -> LoopConfig: ...
def main() -> int: ...
```

Encapsulates:
- Argument parser setup
- Central config integration
- Logging setup
- Entry point

### 6. `orchestrator.py` (~300 lines)

```python
class Orchestrator:
    def __init__(self, config: LoopConfig):
        self.config = config
        self.actor_runner = ActorRunner(config)
        self.eval_runner = EvalRunner(config)
        self.stats_manager = StatsManager(config)
        self._setup_signals()
        self._auto_resume()

    def run_iteration(self, iteration: int) -> IterationStats | None: ...
    def run(self) -> None: ...
```

The main orchestrator becomes a thin coordinator that delegates to specialized components.

## Implementation Steps

1. **Create `orchestrator/` package directory**
   - Create `trainer/src/trainer/orchestrator/` directory

2. **Extract `config.py`**
   - Move `IterationStats` and `LoopConfig` dataclasses
   - No other changes needed (no internal dependencies)

3. **Extract `stats_manager.py`**
   - Move `_save_loop_stats()`, `_save_eval_stats()`, `_update_stats_with_eval()`
   - Move state loading from `_auto_resume_if_needed()`
   - Create `StatsManager` class

4. **Extract `actor_runner.py`**
   - Move `_find_actor_binary()` and `_run_actor()`
   - Create `ActorRunner` class
   - Handle shutdown callback via dependency injection

5. **Extract `eval_runner.py`**
   - Move `_run_evaluation()`, `_load_best_model_info()`, `_save_best_model_info()`, `_promote_to_best()`
   - Create `EvalRunner` class

6. **Extract `cli.py`**
   - Move `parse_args()` and `main()`
   - Keep imports minimal

7. **Refactor main `orchestrator.py`**
   - Use composition with extracted classes
   - Keep `run()` and `run_iteration()` as coordination logic
   - Inject dependencies for testability

8. **Create `__init__.py`**
   - Re-export public API: `Orchestrator`, `LoopConfig`, `IterationStats`, `main`

9. **Create backwards-compatible wrapper**
   - Keep `trainer/src/trainer/orchestrator.py` as thin import wrapper
   - Add deprecation warning
   - Update `__main__.py` to use new location

10. **Update tests**
    - Ensure existing imports still work
    - Add unit tests for extracted components

## Benefits

1. **Testability**: Each component can be unit tested in isolation
2. **Readability**: ~200-300 lines per file vs 1257 in one file
3. **Maintainability**: Changes to actor logic don't touch eval logic
4. **Reusability**: `ActorRunner` or `EvalRunner` could be used independently
5. **Single Responsibility**: Each module has one clear purpose

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking imports | Backwards-compatible wrapper with deprecation warning |
| Circular imports | Config module has no deps; others import only config |
| Over-engineering | Keep it to 6 modules max; don't split further |
| Losing cohesion | Orchestrator class remains the coordinator |

## Estimated Line Counts After Refactoring

| Module | Lines |
|--------|-------|
| `config.py` | ~100 |
| `actor_runner.py` | ~200 |
| `eval_runner.py` | ~200 |
| `stats_manager.py` | ~150 |
| `cli.py` | ~250 |
| `orchestrator.py` | ~300 |
| `__init__.py` | ~20 |
| **Total** | ~1220 |

Line count is similar but now distributed across focused, testable modules.

## Alternative Considered: Fewer Modules

Could consolidate to just 3 modules:
- `config.py` - dataclasses
- `orchestrator.py` - main class
- `cli.py` - argument parsing

This keeps orchestrator at ~700 lines. Acceptable but less testable. The 6-module approach is preferred for better separation of concerns.
