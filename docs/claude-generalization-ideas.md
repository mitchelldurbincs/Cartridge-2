# Game Generalization Plan

This document outlines what needs to be generalized in Cartridge2 to support multiple games (TicTacToe, Connect 4, Othello) instead of just TicTacToe.

## Executive Summary

The engine is already well-designed for multiple games, but the web server, frontend, actor, and trainer have significant TicTacToe-specific hardcoding that needs generalization.

---

## 1. Engine (Low Effort - Already Generalized)

**Status:** The engine core is well-designed for multiple games.

**What's Working:**
- `Game` trait is generic and game-agnostic
- `EngineContext` accepts `env_id` to select game
- Registry pattern allows registering multiple games

**What's Missing - Add GameMetadata:**

Extend capabilities to expose more game-specific metadata:

```rust
pub struct GameMetadata {
    pub board_width: usize,
    pub board_height: usize,
    pub num_actions: usize,
    pub obs_size: usize,
    pub player_names: Vec<String>,      // ["X", "O"] or ["Black", "White"]
    pub player_symbols: Vec<char>,      // ['X', 'O'] or ['B', 'W']
    pub observation_format: String,     // Documentation of obs layout
}
```

---

## 2. Web Server (`web/src/`)

### Hardcoded Issues

| File | Line | Issue |
|------|------|-------|
| `main.rs` | 53-55 | `const TICTACTOE_OBS_SIZE: usize = 29` |
| `main.rs` | 93 | Only `games_tictactoe::register_tictactoe()` called |
| `main.rs` | 171, 259 | `"tictactoe"` hardcoded in game session creation |
| `game.rs` | 24 | `const TICTACTOE_OBS_SIZE: usize = 29` |
| `game.rs` | 34-38 | `board: [u8; 9]` - fixed 9-cell board |
| `game.rs` | 81-82 | State parser assumes 11-byte TicTacToe format |
| `game.rs` | 197-199 | `(0..9u8)` legal move iteration |
| `game.rs` | 323-336 | Message generation assumes "X" vs "O" |

### Recommended Changes

1. **Add game config/selection:**
   - Accept `game_id` from environment variable or request parameter
   - Register all games at startup, not just TicTacToe

2. **Make GameSession generic:**
   - Store board as `Vec<u8>` instead of `[u8; 9]`
   - Query board dimensions and player info from game metadata
   - Delegate state parsing to game-specific logic or engine

3. **Add `/game-info` endpoint:**
   ```
   GET /game-info?env_id=connect4
   Response: { board_width: 7, board_height: 6, num_actions: 7, player_names: [...] }
   ```

---

## 3. Web Frontend (`web/frontend/src/`)

### Hardcoded Issues

| File | Issue |
|------|-------|
| `App.svelte` | Title says "Cartridge2 TicTacToe" |
| `Board.svelte` | `getCellSymbol()` returns X/O only |
| `Board.svelte` | CSS grid is `repeat(3, 1fr)` - 3 columns |
| `Board.svelte` | Cell size is 80x80px |
| `Board.svelte` | `.cell.x` and `.cell.o` styles |
| `lib/api.ts` | No `game_id` in requests |

### Recommended Changes

1. **Add game selector UI:**
   - Dropdown to choose game before starting
   - Fetch available games from `/games` endpoint

2. **Dynamic board rendering:**
   - Query `/game-info` to get board dimensions
   - Calculate grid columns: `repeat(${width}, 1fr)`
   - Scale cell size based on board dimensions
   - Use game metadata for player symbols/colors

3. **Create game-specific board components OR one dynamic component:**
   - Option A: `TicTacToeBoard.svelte`, `Connect4Board.svelte`, `OthelloBoard.svelte`
   - Option B: Single `Board.svelte` that adapts to game metadata (preferred)

---

## 4. Actor (`actor/src/`)

### Hardcoded Issues

| File | Line | Issue |
|------|------|-------|
| `actor.rs` | 19-24 | Constants: `TICTACTOE_NUM_ACTIONS = 9`, `TICTACTOE_OBS_SIZE = 29` |
| `actor.rs` | 35-65 | `extract_legal_mask_from_obs()` parses TicTacToe observation format |
| `actor.rs` | 97-103 | Match on env_id with TicTacToe fallback |
| `actor.rs` | 330 | Legal mask extraction: `step_result.info & 0x1FF` (9 bits) |
| `actor.rs` | 503 | Policy size assertion assumes 9 actions |
| `config.rs` | 22 | Default env_id is "tictactoe" |

### Recommended Changes

1. **Query game capabilities dynamically:**
   ```rust
   let caps = ctx.capabilities();
   let num_actions = caps.action_space.size();
   let obs_size = caps.obs_size;
   ```

2. **Generalize legal mask extraction:**
   - Option A: Games return legal mask in `step_result.info` with consistent format
   - Option B: Observation includes legal moves at a known offset (per-game config)
   - Store legal_moves_offset and num_actions in game metadata

3. **Remove hardcoded fallbacks:**
   - Fail explicitly if game is not registered
   - Add game parameter lookup table or query engine

4. **Support multiple games in config:**
   ```rust
   struct GameParams {
       num_actions: usize,
       obs_size: usize,
       legal_mask_bits: usize,
   }

   fn get_game_params(env_id: &str) -> Result<GameParams, Error>
   ```

---

## 5. Trainer (`trainer/src/trainer/`)

### Hardcoded Issues

| File | Issue |
|------|-------|
| `network.py` | Class `TicTacToeNet` with defaults `obs_size=29, action_size=9` |
| `network.py` | `create_network()` only handles "tictactoe" |
| `trainer.py` | Default `env_id="tictactoe"` |
| `evaluator.py` | Entire file reimplements TicTacToe game logic in Python |

### Recommended Changes

1. **Rename and generalize network:**
   ```python
   class PolicyValueNetwork(nn.Module):
       def __init__(self, obs_size: int, action_size: int, hidden_size: int = 128):
           # No defaults for obs_size/action_size - must be provided
   ```

2. **Add game config registry:**
   ```python
   GAME_CONFIGS = {
       "tictactoe": {"obs_size": 29, "action_size": 9, "hidden": 128},
       "connect4": {"obs_size": TBD, "action_size": 7, "hidden": 256},
       "othello": {"obs_size": TBD, "action_size": 64, "hidden": 256},
   }

   def create_network(env_id: str) -> PolicyValueNetwork:
       config = GAME_CONFIGS[env_id]
       return PolicyValueNetwork(**config)
   ```

3. **Fix evaluator - two options:**
   - **Option A (Quick):** Create game-specific evaluator classes
   - **Option B (Better):** Call Rust engine via subprocess or FFI for evaluation

---

## 6. Replay Buffer

**Status:** Already generalized! Stores `env_id` per transition. No changes needed.

---

## Implementation Priority

### Phase 1: Infrastructure (Do First)
- [ ] Add `GameMetadata` to engine capabilities
- [ ] Create game params lookup in actor
- [ ] Add `/game-info` endpoint to web server

### Phase 2: Core Generalization (Required for Connect 4)
- [ ] Web server: Make game session generic (store board as Vec, query metadata)
- [ ] Actor: Remove hardcoded constants, query capabilities
- [ ] Trainer: Rename network class, add game config registry

### Phase 3: Frontend (Required for Playable Connect 4)
- [ ] Dynamic board rendering based on game metadata
- [ ] Game selector UI
- [ ] Generalize cell symbols/colors

### Phase 4: Evaluator (Optional, can skip initially)
- [ ] Either create game-specific evaluator or call Rust engine

---

## Quick Win: Minimal Changes for Connect 4

If you want to add Connect 4 with minimal generalization:

1. **Engine:** Just implement `Game` trait for Connect4 (already supported)
2. **Actor:** Add Connect4 to the match statement with its constants
3. **Trainer:** Add Connect4 to `create_network()` with its params
4. **Web Server:** Hardcode Connect4 as alternative (ugly but works)
5. **Frontend:** Create separate `Connect4Board.svelte`

This gets Connect 4 working but accumulates tech debt. The proper generalization is recommended.

---

## Game-Specific Parameters Reference

| Game | Board | Actions | Obs Size (estimate) |
|------|-------|---------|---------------------|
| TicTacToe | 3x3 | 9 | 29 |
| Connect 4 | 7x6 | 7 (columns) | ~100 |
| Othello | 8x8 | 64 | ~200 |

---

## Summary

The main work is:
1. **Engine:** Add metadata structure (small change)
2. **Actor:** Query capabilities instead of hardcoding (medium change)
3. **Trainer:** Parameterize network creation (medium change)
4. **Web Server:** Make game session flexible (medium change)
5. **Frontend:** Dynamic board rendering (medium-large change)

The replay buffer is already game-agnostic. The engine architecture is solid - just needs metadata exposure.
