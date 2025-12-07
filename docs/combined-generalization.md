# Combined Game Generalization Plan

This document combines high-level design principles with concrete implementation details for generalizing Cartridge2 to support multiple games (TicTacToe, Connect 4, Othello).

---

## Part 1: Design Principles

### Core Abstractions

| Principle | Current State | Action Needed |
|-----------|---------------|---------------|
| Common `Game` interface (`initial_state`, `legal_actions`, `apply_action`, `is_terminal`, `utility`) | **Done** - `Game` trait in `engine-core/src/typed.rs` | None |
| Generic player representation (not X/O) | Partial - engine is generic, but web/frontend hardcode X/O | Expose player info in metadata |
| Generic `GameState` with board dimensions and metadata | Partial - state is opaque bytes, no exposed dimensions | Add `GameMetadata` struct |
| Pluggable `Renderer` decoupled from logic | **Not done** - frontend has hardcoded 3x3 grid | Make board rendering dynamic |
| Registration/discovery mechanism | **Done** - `register_game()` + `EngineContext::new(env_id)` | None |

### Configuration and Metadata

| Principle | Current State | Action Needed |
|-----------|---------------|---------------|
| Game constants in config, not hardcoded | **Not done** - constants scattered in actor/web/trainer | Create `GameMetadata` and config registry |
| Per-game action encoding mapped to generic IDs | Partial - actions are bytes, but parsing is game-specific | Document action formats per game |
| Standardized coordinate systems | Not standardized - each game does its own thing | Define conventions |

### Training and Evaluation

| Principle | Current State | Action Needed |
|-----------|---------------|---------------|
| MCTS reads branching factor from game interface | **Not done** - hardcoded `TICTACTOE_NUM_ACTIONS = 9` | Query from capabilities |
| Network architecture parameterized by dimensions | **Not done** - `TicTacToeNet(obs_size=29, action_size=9)` | Create `PolicyValueNetwork` with required params |
| Replay buffer stores game ID | **Done** - `env_id` field per transition | None |

### Serialization and Persistence

| Principle | Current State | Action Needed |
|-----------|---------------|---------------|
| Save/load includes game ID and dimensions | Partial - replay has `env_id`, but web sessions don't | Add game metadata to session state |
| Versioned action schema | Not done | Define action format versioning |

### Testing and Tooling

| Principle | Current State | Action Needed |
|-----------|---------------|---------------|
| Property-based tests for generic guarantees | Not done | Add proptest for all registered games |
| Small-board variant fixtures | Not done | Create 4x4 Connect-3 for fast testing |
| Golden game transcripts | Not done | Add replay validation tests |

### UI/UX

| Principle | Current State | Action Needed |
|-----------|---------------|---------------|
| Layout derived from dimensions | **Not done** - CSS is `repeat(3, 1fr)` | Calculate from metadata |
| Per-game descriptions/tooltips | Not done | Add to `GameMetadata` |
| Highlight last move and valid moves | Partial - legal moves shown, no last-move highlight | Add last_action to game state |

---

## Part 2: Current Codebase Analysis

### What's Already Generalized

1. **Engine Core** - `Game` trait, `EngineContext`, registry pattern
2. **Replay Buffer** - Stores `env_id` per transition, fully game-agnostic
3. **MCTS Core** - Uses `Evaluator` trait, action count from config

### What's Hardcoded (Must Fix)

#### Actor (`actor/src/`)

| Location | Hardcoded Value | Fix |
|----------|-----------------|-----|
| `actor.rs:19-24` | `TICTACTOE_NUM_ACTIONS = 9`, `TICTACTOE_OBS_SIZE = 29` | Query from game capabilities |
| `actor.rs:35-65` | `extract_legal_mask_from_obs()` - TicTacToe observation offsets | Make offset configurable per game |
| `actor.rs:97-103` | Match on env_id with TicTacToe fallback | Fail explicitly, use config registry |
| `actor.rs:330` | `step_result.info & 0x1FF` - 9-bit mask | Use `num_actions` bits |
| `config.rs:22` | Default `env_id = "tictactoe"` | Keep default, but support all games |

#### Web Server (`web/src/`)

| Location | Hardcoded Value | Fix |
|----------|-----------------|-----|
| `main.rs:53-55` | `TICTACTOE_OBS_SIZE = 29` | Query from game metadata |
| `main.rs:93` | Only registers TicTacToe | Register all games |
| `main.rs:171,259` | `"tictactoe"` in session creation | Accept from request/config |
| `game.rs:34-38` | `board: [u8; 9]` | Use `Vec<u8>` |
| `game.rs:81-82` | 11-byte state parser | Delegate to engine |
| `game.rs:197-199` | `(0..9u8)` legal moves | Query from game state |
| `game.rs:323-336` | "X" vs "O" messages | Use player names from metadata |

#### Web Frontend (`web/frontend/src/`)

| Location | Hardcoded Value | Fix |
|----------|-----------------|-----|
| `App.svelte` | Title "Cartridge2 TicTacToe" | Dynamic from game metadata |
| `Board.svelte` | `getCellSymbol()` returns X/O | Use player symbols from metadata |
| `Board.svelte` | `repeat(3, 1fr)` CSS | `repeat(${width}, 1fr)` |
| `Board.svelte` | 80x80px cells | Scale by board size |
| `lib/api.ts` | No `game_id` in requests | Add to all endpoints |

#### Trainer (`trainer/src/trainer/`)

| Location | Hardcoded Value | Fix |
|----------|-----------------|-----|
| `network.py` | `TicTacToeNet(obs_size=29, action_size=9)` | `PolicyValueNetwork` with required params |
| `network.py` | `create_network()` only handles tictactoe | Add game config registry |
| `evaluator.py` | Entire TicTacToe reimplementation | Call Rust engine or create per-game evaluators |

---

## Part 3: Implementation Plan

### Phase 1: Infrastructure (Do First)

**Goal:** Expose game metadata so other components can query it.

#### 1.1 Add `GameMetadata` to Engine

```rust
// engine-core/src/metadata.rs
pub struct GameMetadata {
    pub env_id: String,
    pub display_name: String,
    pub board_width: usize,
    pub board_height: usize,
    pub num_actions: usize,
    pub obs_size: usize,
    pub legal_mask_offset: usize,      // Where legal moves start in observation
    pub player_count: usize,
    pub player_names: Vec<String>,     // ["X", "O"] or ["Black", "White"]
    pub player_symbols: Vec<char>,     // ['X', 'O'] or ['B', 'W']
    pub description: String,           // Rules summary for UI
}
```

Each game implements a `metadata()` method returning this struct.

#### 1.2 Add `/game-info` and `/games` Endpoints

```
GET /games
Response: ["tictactoe", "connect4", "othello"]

GET /game-info?env_id=connect4
Response: {
    "env_id": "connect4",
    "display_name": "Connect 4",
    "board_width": 7,
    "board_height": 6,
    "num_actions": 7,
    "player_names": ["Red", "Yellow"],
    ...
}
```

#### 1.3 Create Game Config Registry in Actor/Trainer

```rust
// actor/src/game_config.rs
pub struct GameConfig {
    pub num_actions: usize,
    pub obs_size: usize,
    pub legal_mask_offset: usize,
}

pub fn get_config(env_id: &str) -> Result<GameConfig, Error> {
    match env_id {
        "tictactoe" => Ok(GameConfig { num_actions: 9, obs_size: 29, legal_mask_offset: 18 }),
        "connect4" => Ok(GameConfig { num_actions: 7, obs_size: 100, legal_mask_offset: 86 }),
        _ => Err(Error::UnknownGame(env_id.to_string())),
    }
}
```

```python
# trainer/src/trainer/game_config.py
GAME_CONFIGS = {
    "tictactoe": {"obs_size": 29, "action_size": 9, "hidden": 128},
    "connect4": {"obs_size": 100, "action_size": 7, "hidden": 256},
    "othello": {"obs_size": 195, "action_size": 64, "hidden": 256},
}
```

---

### Phase 2: Core Generalization (Required for Connect 4)

#### 2.1 Actor: Remove Hardcoded Constants

- Replace `TICTACTOE_*` constants with `get_config(env_id)`
- Generalize `extract_legal_mask_from_obs()` to use `legal_mask_offset`
- Remove fallback to TicTacToe - fail explicitly

#### 2.2 Web Server: Generic Game Session

- Change `board: [u8; 9]` to `board: Vec<u8>`
- Query board dimensions from `/game-info` or engine
- Accept `game_id` in `/game/new` request
- Delegate state parsing to game-specific logic

#### 2.3 Trainer: Parameterized Network

- Rename `TicTacToeNet` to `PolicyValueNetwork`
- Remove default values for `obs_size` and `action_size`
- Update `create_network()` to use `GAME_CONFIGS`

---

### Phase 3: Frontend (Required for Playable Connect 4)

#### 3.1 Dynamic Board Component

```svelte
<script>
    export let board: number[];
    export let width: number;
    export let height: number;
    export let playerSymbols: string[];

    $: gridStyle = `grid-template-columns: repeat(${width}, 1fr)`;
    $: cellSize = Math.min(400 / width, 400 / height);
</script>

<div class="board" style={gridStyle}>
    {#each board as cell, i}
        <button
            class="cell"
            style="width: {cellSize}px; height: {cellSize}px"
            on:click={() => onCellClick(i)}
        >
            {cell > 0 ? playerSymbols[cell - 1] : ''}
        </button>
    {/each}
</div>
```

#### 3.2 Game Selector

```svelte
<select bind:value={selectedGame} on:change={loadGameInfo}>
    {#each availableGames as game}
        <option value={game}>{game}</option>
    {/each}
</select>
```

#### 3.3 API Updates

Add `game_id` to all requests and responses in `lib/api.ts`.

---

### Phase 4: Testing and Polish (Optional)

- Property-based tests for all registered games
- Golden transcript validation
- Small-board test fixtures (4x4 Connect-3)
- Last-move highlighting in UI
- Game rules tooltip

---

## Part 4: Quick Reference

### Game Parameters

| Game | Board | Actions | Obs Size | Legal Mask Offset |
|------|-------|---------|----------|-------------------|
| TicTacToe | 3x3 | 9 | 29 | 18 |
| Connect 4 | 7x6 | 7 | ~100 | TBD |
| Othello | 8x8 | 64 | ~195 | TBD |

### Observation Format Convention

```
[board_view: W*H*2 floats] [legal_moves: num_actions floats] [current_player: 2 floats]
```

- `board_view`: One-hot encoding per cell (2 channels for 2 players)
- `legal_moves`: 1.0 if action is legal, 0.0 otherwise
- `current_player`: One-hot encoding of current player

### Action Format Convention

- TicTacToe: Position index 0-8 (row-major)
- Connect 4: Column index 0-6
- Othello: Position index 0-63 (row-major)

---

## Part 5: Decision Points

### Q1: Single dynamic board component vs game-specific components?

**Recommendation:** Single dynamic component (Option B)

- Less code duplication
- Easier to add new games
- Connect 4's "gravity" and Othello's "flipping" can be handled in game logic, not UI

### Q2: How to handle the evaluator?

**Recommendation:** Call Rust engine via subprocess (Option B)

- Avoids reimplementing game logic in Python
- Single source of truth for rules
- Slightly slower but more maintainable

### Q3: Where should game metadata live?

**Recommendation:** In the engine, exposed via `Game::metadata()`

- Single source of truth
- Queryable via API
- Other components derive from it

---

## Summary Checklist

### Must Have (Before Connect 4)
- [ ] `GameMetadata` struct in engine
- [ ] Game config registry in actor
- [ ] Game config registry in trainer
- [ ] Remove hardcoded constants in actor
- [ ] Generic board storage in web server
- [ ] Dynamic board rendering in frontend

### Nice to Have
- [ ] `/games` and `/game-info` endpoints
- [ ] Game selector UI
- [ ] Property-based tests
- [ ] Rust engine evaluator
- [ ] Last-move highlighting
