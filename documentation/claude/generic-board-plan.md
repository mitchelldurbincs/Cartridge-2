# Generic Two-Player Board Game Abstraction Plan

## Executive Summary

TicTacToe and Connect4 share approximately **300 lines of nearly identical code**. This document outlines a comprehensive plan to extract common patterns into a generic `TwoPlayerBoardGame` trait and helper utilities, reducing duplication while maintaining the flexibility needed for game-specific logic.

---

## Current State Analysis

### Files Affected
- `engine/games-tictactoe/src/lib.rs` (354 lines)
- `engine/games-connect4/src/lib.rs` (442 lines)
- `engine/engine-core/src/board_game.rs` (135 lines - already has `TwoPlayerObs`)
- `engine/engine-core/src/game_utils.rs` (243 lines - already has utilities)

### Duplicated Patterns

#### 1. State Structure (~30 lines each)
Both games have nearly identical state fields:

```rust
// TicTacToe (lines 41-49)
pub struct State {
    board: [u8; 9],
    current_player: u8,  // 1 or 2
    winner: u8,          // 0=none, 1=p1, 2=p2, 3=draw
}

// Connect4 (lines 60-71)
pub struct State {
    board: [u8; 42],
    current_player: u8,
    winner: u8,
    column_heights: [u8; 7],  // Game-specific extra field
}
```

**Shared**: `board`, `current_player`, `winner`
**Different**: Connect4 has `column_heights` for drop optimization

#### 2. State Methods (~50 lines each)
```rust
// Both implement identically:
fn new() -> Self
fn is_done(&self) -> bool { self.winner != 0 }
fn legal_moves(&self) -> Vec<u8>
fn legal_moves_mask(&self) -> u8/u16

// Different implementations:
fn make_move() / fn drop_piece()  // Game-specific move logic
fn check_winner()                  // Game-specific win detection
```

#### 3. Game Implementation (~20 lines each)
```rust
// TicTacToe (lines 170-195)
pub struct TicTacToe;
impl TicTacToe {
    pub fn new() -> Self { Self }
    fn compute_info_bits(state: &State) -> u64 { ... }
}

// Connect4 (lines 236-262) - identical structure
pub struct Connect4;
impl Connect4 {
    pub fn new() -> Self { Self }
    fn compute_info_bits(state: &State) -> u64 { ... }
}
```

#### 4. Observation Creation (~7 lines each)
```rust
// TicTacToe (lines 161-167)
pub fn observation_from_state(state: &State) -> Observation {
    TwoPlayerObs::from_board(
        &state.board,
        state.legal_moves_mask() as u64,
        state.current_player,
    )
}

// Connect4 (lines 228-234) - IDENTICAL except type
```

#### 5. Game Trait impl - reset() (~5 lines each)
```rust
// Both games (TicTacToe 239-243, Connect4 314-318)
fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
    let state = State::new();
    let obs = observation_from_state(&state);
    (state, obs)
}
```

#### 6. Game Trait impl - step() (~15 lines each)
```rust
// TicTacToe (lines 245-260)
fn step(&mut self, state: &mut Self::State, action: Self::Action, _rng: &mut ChaCha20Rng)
    -> (Self::Obs, f32, bool, u64)
{
    let previous_player = state.current_player;
    *state = state.make_move(action);         // <-- Only difference: make_move vs drop_piece

    let obs = observation_from_state(state);
    let reward = calculate_reward(state.winner, previous_player);
    let done = state.is_done();
    let info = Self::compute_info_bits(state);

    (obs, reward, done, info)
}

// Connect4 (lines 320-335) - IDENTICAL except make_move -> drop_piece
```

#### 7. State Encoding (~20 lines each)
```rust
// TicTacToe (lines 262-268)
fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    out.extend_from_slice(&state.board);
    out.push(state.current_player);
    out.push(state.winner);
    Ok(())
}

// Connect4 (lines 337-344) - IDENTICAL pattern, different board size
```

#### 8. State Decoding with Validation (~45 lines each)
```rust
// Both games have IDENTICAL validation:
fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
    // 1. Length check (different expected length)
    if buf.len() != EXPECTED_LEN { return Err(...); }

    // 2. Copy board
    let mut board = [0u8; BOARD_SIZE];
    board.copy_from_slice(&buf[0..BOARD_SIZE]);

    // 3. Extract fields
    let current_player = buf[BOARD_SIZE];
    let winner = buf[BOARD_SIZE + 1];

    // 4. Validate current_player (IDENTICAL in both)
    if current_player != 1 && current_player != 2 {
        return Err(DecodeError::CorruptedData(format!(
            "Invalid current_player: {}", current_player
        )));
    }

    // 5. Validate winner (IDENTICAL in both)
    if winner > 3 {
        return Err(DecodeError::CorruptedData(format!(
            "Invalid winner: {}", winner
        )));
    }

    // 6. Validate board cells (IDENTICAL in both)
    for &cell in &board {
        if cell > 2 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid board cell: {}", cell
            )));
        }
    }

    // 7. Reconstruct state (game-specific for Connect4 column_heights)
    Ok(State { ... })
}
```

#### 9. Action Encoding/Decoding (~35 lines each)
```rust
// Both games (TicTacToe 315-344, Connect4 403-432)
fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    if *action >= NUM_ACTIONS {
        return Err(EncodeError::InvalidData(format!(...)));
    }
    out.extend_from_slice(&(*action as u32).to_le_bytes());
    Ok(())
}

fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
    if buf.len() != 4 {
        return Err(DecodeError::InvalidLength { expected: 4, actual: buf.len() });
    }
    let value = u32::from_le_bytes(buf.try_into().unwrap());
    if value >= NUM_ACTIONS {
        return Err(DecodeError::CorruptedData(format!(...)));
    }
    Ok(value as u8)
}
```

#### 10. Observation Encoding (~5 lines each)
```rust
// Both games - IDENTICAL
fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    obs.encode(out);
    Ok(())
}
```

---

## Proposed Solution

### Approach: Trait + Default Implementations + Macros

We'll use a combination of:
1. **`BoardGameState` trait** - For state operations
2. **`TwoPlayerBoardGame` trait** - Extends `Game` with defaults
3. **Helper functions** - For encoding/decoding with validation
4. **Optional macro** - For remaining boilerplate

### New Files

```
engine/engine-core/src/
├── board_game.rs           # EXTEND: Add BoardGameState + TwoPlayerBoardGame
├── board_game_encoding.rs  # NEW: Shared encoding/decoding utilities
└── lib.rs                  # UPDATE: Re-export new types
```

---

## Detailed Design

### Phase 1: State Trait

Add to `board_game.rs`:

```rust
/// Trait for two-player board game states.
///
/// Implement this for your game's state type to get automatic
/// encoding/decoding and shared game logic.
pub trait BoardGameState: Clone + Send + Sync + 'static {
    /// Board size in cells
    const BOARD_SIZE: usize;
    /// Number of legal actions
    const NUM_ACTIONS: usize;
    /// Encoded state size in bytes (board + current_player + winner + extras)
    const ENCODED_SIZE: usize;

    /// Create initial state
    fn new() -> Self;

    /// Get the board as a slice
    fn board(&self) -> &[u8];

    /// Get current player (1 or 2)
    fn current_player(&self) -> u8;

    /// Get winner (0=none, 1=p1, 2=p2, 3=draw)
    fn winner(&self) -> u8;

    /// Check if game is over
    fn is_done(&self) -> bool {
        self.winner() != 0
    }

    /// Get legal moves as bitmask
    fn legal_moves_mask(&self) -> u64;

    /// Apply an action and return the new state
    fn apply_action(&self, action: u8) -> Self;

    /// Count moves played (for info bits)
    fn moves_played(&self) -> u64;

    // === Encoding hooks (with defaults) ===

    /// Encode game-specific extra fields (e.g., column_heights)
    /// Default: no extra fields
    fn encode_extra(&self, _out: &mut Vec<u8>) {}

    /// Decode game-specific extra fields
    /// Default: no extra fields
    fn decode_extra(_buf: &[u8], _offset: usize) -> Result<Self, DecodeError>
    where
        Self: Sized;
}
```

### Phase 2: Encoding Utilities

Create `board_game_encoding.rs`:

```rust
use crate::typed::{DecodeError, EncodeError};
use crate::board_game::BoardGameState;

/// Encode a board game state to bytes.
///
/// Format: [board...][current_player][winner][extra...]
pub fn encode_board_state<S: BoardGameState>(state: &S, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    out.extend_from_slice(state.board());
    out.push(state.current_player());
    out.push(state.winner());
    state.encode_extra(out);
    Ok(())
}

/// Decode a board game state from bytes with full validation.
pub fn decode_board_state<S: BoardGameState>(buf: &[u8]) -> Result<S, DecodeError> {
    // Length validation
    if buf.len() < S::BOARD_SIZE + 2 {
        return Err(DecodeError::InvalidLength {
            expected: S::ENCODED_SIZE,
            actual: buf.len(),
        });
    }

    let board = &buf[0..S::BOARD_SIZE];
    let current_player = buf[S::BOARD_SIZE];
    let winner = buf[S::BOARD_SIZE + 1];

    // Validate current_player
    validate_current_player(current_player)?;

    // Validate winner
    validate_winner(winner)?;

    // Validate board cells
    validate_board_cells(board)?;

    // Delegate to game-specific decoder for extra fields
    S::decode_extra(buf, S::BOARD_SIZE + 2)
}

/// Validate current_player field
fn validate_current_player(player: u8) -> Result<(), DecodeError> {
    if player != 1 && player != 2 {
        return Err(DecodeError::CorruptedData(format!(
            "Invalid current_player: {}", player
        )));
    }
    Ok(())
}

/// Validate winner field
fn validate_winner(winner: u8) -> Result<(), DecodeError> {
    if winner > 3 {
        return Err(DecodeError::CorruptedData(format!(
            "Invalid winner: {}", winner
        )));
    }
    Ok(())
}

/// Validate all board cells contain valid values (0, 1, or 2)
fn validate_board_cells(board: &[u8]) -> Result<(), DecodeError> {
    for &cell in board {
        if cell > 2 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid board cell: {}", cell
            )));
        }
    }
    Ok(())
}

/// Encode a discrete action as u32 little-endian.
pub fn encode_discrete_action(action: u8, num_actions: usize, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    if action as usize >= num_actions {
        return Err(EncodeError::InvalidData(format!(
            "Invalid action: {} (max {})", action, num_actions - 1
        )));
    }
    out.extend_from_slice(&(action as u32).to_le_bytes());
    Ok(())
}

/// Decode a discrete action from u32 little-endian.
pub fn decode_discrete_action(buf: &[u8], num_actions: usize) -> Result<u8, DecodeError> {
    if buf.len() != 4 {
        return Err(DecodeError::InvalidLength {
            expected: 4,
            actual: buf.len(),
        });
    }
    let value = u32::from_le_bytes(buf.try_into().unwrap());
    if value as usize >= num_actions {
        return Err(DecodeError::CorruptedData(format!(
            "Invalid action: {} (max {})", value, num_actions - 1
        )));
    }
    Ok(value as u8)
}
```

### Phase 3: Game Trait with Defaults

Extend `board_game.rs`:

```rust
use crate::typed::{Game, EncodeError, DecodeError, EngineId, Capabilities, Encoding, ActionSpace};
use crate::game_utils::{calculate_reward, info_bits};
use crate::metadata::GameMetadata;
use crate::board_game_encoding::{encode_board_state, decode_board_state, encode_discrete_action, decode_discrete_action};
use rand_chacha::ChaCha20Rng;

/// Trait for two-player board games with sensible defaults.
///
/// Implementing this trait automatically provides implementations for
/// most of the `Game` trait methods, reducing boilerplate significantly.
pub trait TwoPlayerBoardGame: Send + Sync + std::fmt::Debug + 'static {
    /// The state type (must implement BoardGameState)
    type State: BoardGameState;

    /// The observation type (typically TwoPlayerObs<BOARD_VIEW, ACTIONS>)
    type Obs: Send + Sync + 'static;

    // === Required: Game identification ===

    /// Game identifier (e.g., "tictactoe", "connect4")
    fn env_id(&self) -> &'static str;

    /// Human-readable name (e.g., "Tic-Tac-Toe", "Connect 4")
    fn display_name(&self) -> &'static str;

    /// Game description
    fn description(&self) -> &'static str;

    /// Build/version identifier
    fn build_id(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    // === Required: Game configuration ===

    /// Board dimensions (width, height)
    fn board_dimensions(&self) -> (usize, usize);

    /// Player names (e.g., ["X", "O"])
    fn player_names(&self) -> Vec<String>;

    /// Player symbols (e.g., ['X', 'O'])
    fn player_symbols(&self) -> Vec<char>;

    /// Observation encoding version string
    fn obs_encoding(&self) -> String;

    // === Required: Observation creation ===

    /// Create observation from state
    fn make_observation(state: &Self::State) -> Self::Obs;

    /// Encode observation to bytes
    fn encode_observation(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError>;

    // === Optional: Board type (for UI) ===

    /// Board type hint for frontend rendering
    /// Default: "grid" (standard click-on-cell)
    /// Override with "drop_column" for Connect4-style
    fn board_type(&self) -> Option<&'static str> {
        None
    }

    // === Derived implementations ===

    /// Get engine ID (derived from env_id and build_id)
    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: self.env_id().to_string(),
            build_id: self.build_id(),
        }
    }

    /// Get capabilities (derived from State constants and config)
    fn capabilities(&self) -> Capabilities {
        let (width, height) = self.board_dimensions();
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: format!("{}_state:v1", self.env_id()),
                action: "discrete:v1".to_string(),
                obs: self.obs_encoding(),
                schema_version: 1,
            },
            max_horizon: (width * height) as u32,
            action_space: ActionSpace::Discrete(Self::State::NUM_ACTIONS as u32),
            preferred_batch: 64,
        }
    }

    /// Get metadata (derived from configuration)
    fn metadata(&self) -> GameMetadata {
        let (width, height) = self.board_dimensions();
        let obs_size = Self::State::BOARD_SIZE * 2 + Self::State::NUM_ACTIONS + 2;
        let legal_mask_offset = Self::State::BOARD_SIZE * 2;

        let mut meta = GameMetadata::new(self.env_id(), self.display_name())
            .with_board(width, height)
            .with_actions(Self::State::NUM_ACTIONS)
            .with_observation(obs_size, legal_mask_offset)
            .with_players(2, self.player_names(), self.player_symbols())
            .with_description(self.description());

        if let Some(board_type) = self.board_type() {
            meta = meta.with_board_type(board_type);
        }

        meta
    }

    /// Compute info bits from state (derived from state methods)
    fn compute_info_bits(state: &Self::State) -> u64 {
        info_bits::compute_info_bits(
            state.legal_moves_mask(),
            state.current_player(),
            state.winner(),
            state.moves_played(),
        )
    }
}

/// Blanket implementation of Game for TwoPlayerBoardGame implementors.
///
/// This provides all the boilerplate Game trait methods automatically.
impl<T> Game for T
where
    T: TwoPlayerBoardGame,
{
    type State = <T as TwoPlayerBoardGame>::State;
    type Action = u8;
    type Obs = <T as TwoPlayerBoardGame>::Obs;

    fn engine_id(&self) -> EngineId {
        TwoPlayerBoardGame::engine_id(self)
    }

    fn capabilities(&self) -> Capabilities {
        TwoPlayerBoardGame::capabilities(self)
    }

    fn metadata(&self) -> GameMetadata {
        TwoPlayerBoardGame::metadata(self)
    }

    fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
        let state = Self::State::new();
        let obs = T::make_observation(&state);
        (state, obs)
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64) {
        let previous_player = state.current_player();
        *state = state.apply_action(action);

        let obs = T::make_observation(state);
        let reward = calculate_reward(state.winner(), previous_player);
        let done = state.is_done();
        let info = T::compute_info_bits(state);

        (obs, reward, done, info)
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        encode_board_state(state, out)
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        decode_board_state::<Self::State>(buf)
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        encode_discrete_action(*action, Self::State::NUM_ACTIONS, out)
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        decode_discrete_action(buf, Self::State::NUM_ACTIONS)
    }

    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        T::encode_observation(obs, out)
    }
}
```

---

## Refactored Game Implementations

### TicTacToe After Refactoring

```rust
// games-tictactoe/src/lib.rs (~120 lines, down from 354)

use engine_core::{
    BoardGameState, TwoPlayerBoardGame, TwoPlayerObs,
    register_game, GameAdapter,
    typed::{DecodeError, EncodeError},
};

// === State (game-specific logic only) ===

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    board: [u8; 9],
    current_player: u8,
    winner: u8,
}

impl BoardGameState for State {
    const BOARD_SIZE: usize = 9;
    const NUM_ACTIONS: usize = 9;
    const ENCODED_SIZE: usize = 11; // 9 + 1 + 1

    fn new() -> Self {
        Self { board: [0; 9], current_player: 1, winner: 0 }
    }

    fn board(&self) -> &[u8] { &self.board }
    fn current_player(&self) -> u8 { self.current_player }
    fn winner(&self) -> u8 { self.winner }

    fn legal_moves_mask(&self) -> u64 {
        if self.is_done() { return 0; }
        self.board.iter().enumerate()
            .fold(0u64, |m, (i, &c)| if c == 0 { m | (1 << i) } else { m })
    }

    fn apply_action(&self, action: u8) -> Self {
        if self.is_done() || action >= 9 || self.board[action as usize] != 0 {
            return *self;
        }
        let mut new = *self;
        new.board[action as usize] = self.current_player;
        new.winner = Self::check_winner(&new.board);
        if new.winner == 0 {
            new.current_player = if self.current_player == 1 { 2 } else { 1 };
        }
        new
    }

    fn moves_played(&self) -> u64 {
        self.board.iter().filter(|&&c| c != 0).count() as u64
    }

    fn decode_extra(buf: &[u8], _offset: usize) -> Result<Self, DecodeError> {
        let mut board = [0u8; 9];
        board.copy_from_slice(&buf[0..9]);
        Ok(Self { board, current_player: buf[9], winner: buf[10] })
    }
}

impl State {
    fn check_winner(board: &[u8; 9]) -> u8 {
        const LINES: [[usize; 3]; 8] = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6],
        ];
        for [a,b,c] in LINES {
            if board[a] != 0 && board[a] == board[b] && board[b] == board[c] {
                return board[a];
            }
        }
        if board.iter().all(|&c| c != 0) { 3 } else { 0 }
    }
}

// === Game Implementation ===

pub type Observation = TwoPlayerObs<18, 9>;

#[derive(Debug)]
pub struct TicTacToe;

impl TwoPlayerBoardGame for TicTacToe {
    type State = State;
    type Obs = Observation;

    fn env_id(&self) -> &'static str { "tictactoe" }
    fn display_name(&self) -> &'static str { "Tic-Tac-Toe" }
    fn description(&self) -> &'static str { "Get three in a row to win!" }
    fn board_dimensions(&self) -> (usize, usize) { (3, 3) }
    fn player_names(&self) -> Vec<String> { vec!["X".into(), "O".into()] }
    fn player_symbols(&self) -> Vec<char> { vec!['X', 'O'] }
    fn obs_encoding(&self) -> String { "f32x29:v1".into() }

    fn make_observation(state: &State) -> Observation {
        TwoPlayerObs::from_board(&state.board, state.legal_moves_mask(), state.current_player)
    }

    fn encode_observation(obs: &Observation, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        obs.encode(out);
        Ok(())
    }
}

// === Registration ===

pub fn register_tictactoe() {
    register_game("tictactoe".to_string(), || {
        Box::new(GameAdapter::new(TicTacToe))
    });
}
```

### Connect4 After Refactoring

```rust
// games-connect4/src/lib.rs (~150 lines, down from 442)

use engine_core::{
    BoardGameState, TwoPlayerBoardGame, TwoPlayerObs,
    register_game, GameAdapter,
    typed::{DecodeError, EncodeError},
};

pub const COLS: usize = 7;
pub const ROWS: usize = 6;
pub const BOARD_SIZE: usize = 42;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    board: [u8; BOARD_SIZE],
    current_player: u8,
    winner: u8,
    column_heights: [u8; COLS],
}

impl BoardGameState for State {
    const BOARD_SIZE: usize = BOARD_SIZE;
    const NUM_ACTIONS: usize = COLS;
    const ENCODED_SIZE: usize = BOARD_SIZE + 2; // column_heights reconstructed

    fn new() -> Self {
        Self { board: [0; BOARD_SIZE], current_player: 1, winner: 0, column_heights: [0; COLS] }
    }

    fn board(&self) -> &[u8] { &self.board }
    fn current_player(&self) -> u8 { self.current_player }
    fn winner(&self) -> u8 { self.winner }

    fn legal_moves_mask(&self) -> u64 {
        if self.is_done() { return 0; }
        self.column_heights.iter().enumerate()
            .fold(0u64, |m, (i, &h)| if h < ROWS as u8 { m | (1 << i) } else { m })
    }

    fn apply_action(&self, action: u8) -> Self {
        let col = action as usize;
        if self.is_done() || col >= COLS || self.column_heights[col] >= ROWS as u8 {
            return self.clone();
        }
        let mut new = self.clone();
        let row = self.column_heights[col] as usize;
        let pos = row * COLS + col;
        new.board[pos] = self.current_player;
        new.column_heights[col] += 1;
        new.winner = new.check_winner_at(col, row);
        if new.winner == 0 {
            new.current_player = if self.current_player == 1 { 2 } else { 1 };
        }
        new
    }

    fn moves_played(&self) -> u64 {
        self.column_heights.iter().map(|&h| h as u64).sum()
    }

    fn decode_extra(buf: &[u8], _offset: usize) -> Result<Self, DecodeError> {
        let mut board = [0u8; BOARD_SIZE];
        board.copy_from_slice(&buf[0..BOARD_SIZE]);
        // Reconstruct column heights
        let mut column_heights = [0u8; COLS];
        for col in 0..COLS {
            for row in 0..ROWS {
                if board[row * COLS + col] != 0 {
                    column_heights[col] = (row + 1) as u8;
                }
            }
        }
        Ok(Self { board, current_player: buf[BOARD_SIZE], winner: buf[BOARD_SIZE+1], column_heights })
    }
}

impl State {
    fn check_winner_at(&self, col: usize, row: usize) -> u8 {
        // ... (game-specific win detection, ~40 lines)
    }
}

// === Game Implementation ===

pub type Observation = TwoPlayerObs<84, 7>;

#[derive(Debug)]
pub struct Connect4;

impl TwoPlayerBoardGame for Connect4 {
    type State = State;
    type Obs = Observation;

    fn env_id(&self) -> &'static str { "connect4" }
    fn display_name(&self) -> &'static str { "Connect 4" }
    fn description(&self) -> &'static str { "Drop discs to connect four in a row!" }
    fn board_dimensions(&self) -> (usize, usize) { (COLS, ROWS) }
    fn player_names(&self) -> Vec<String> { vec!["Red".into(), "Yellow".into()] }
    fn player_symbols(&self) -> Vec<char> { vec!['\u{1F534}', '\u{1F7E1}'] }
    fn obs_encoding(&self) -> String { "f32x93:v1".into() }
    fn board_type(&self) -> Option<&'static str> { Some("drop_column") }

    fn make_observation(state: &State) -> Observation {
        TwoPlayerObs::from_board(&state.board, state.legal_moves_mask(), state.current_player)
    }

    fn encode_observation(obs: &Observation, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        obs.encode(out);
        Ok(())
    }
}

pub fn register_connect4() {
    register_game("connect4".to_string(), || {
        Box::new(GameAdapter::new(Connect4))
    });
}
```

---

## Line Count Comparison

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| TicTacToe | 354 | ~120 | **66%** |
| Connect4 | 442 | ~150 | **66%** |
| engine-core board_game.rs | 135 | ~300 | +165 (shared) |
| engine-core board_game_encoding.rs | 0 | ~100 | +100 (shared) |
| **Total** | 931 | ~670 | **28%** |

**Net reduction**: ~260 lines
**Duplication eliminated**: ~200 lines of identical validation/encoding code
**Better**: New games now require only ~100-150 lines instead of ~400

---

## Implementation Plan

### Step 1: Add BoardGameState trait (engine-core)
- Add trait definition to `board_game.rs`
- Add encoding utilities to new `board_game_encoding.rs`
- Update `lib.rs` exports
- Add tests for encoding utilities

### Step 2: Add TwoPlayerBoardGame trait (engine-core)
- Add trait with default implementations
- Add blanket `impl Game for T where T: TwoPlayerBoardGame`
- Test that blanket impl compiles and works

### Step 3: Refactor TicTacToe
- Implement `BoardGameState` for `State`
- Implement `TwoPlayerBoardGame` for `TicTacToe`
- Remove manual `Game` implementation
- Verify all tests pass

### Step 4: Refactor Connect4
- Implement `BoardGameState` for `State`
- Implement `TwoPlayerBoardGame` for `Connect4`
- Remove manual `Game` implementation
- Verify all tests pass

### Step 5: Documentation & Cleanup
- Add documentation for new traits
- Add example in CLAUDE.md for adding new games
- Remove any dead code

---

## Testing Strategy

1. **Unit tests for encoding utilities**
   - Round-trip encoding for various board sizes
   - Error cases (invalid length, corrupted data)

2. **Integration tests**
   - Existing game tests should pass unchanged
   - Actor integration tests
   - MCTS integration tests

3. **Property tests**
   - Arbitrary board states encode/decode correctly
   - All valid actions encode/decode correctly

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Blanket impl conflicts with other Game impls | Use marker trait or feature flag |
| Performance regression from virtual dispatch | Trait methods are inlined; benchmark MCTS |
| Breaking change for game implementations | Provide both old and new patterns during transition |
| Complexity increase in engine-core | Clear documentation and examples |

---

## Future Extensions

Once this refactoring is complete, adding new games becomes much simpler:

1. **Othello**: ~150 lines (was estimated ~400)
2. **Chess**: Would need variant handling but core structure applies
3. **Go**: Larger boards work automatically via const generics

The `BoardGameState` trait also opens possibilities for:
- Automatic game tree visualization
- Generic MCTS improvements
- Board state serialization for replay databases
