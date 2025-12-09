//! Connect 4 game implementation for the Cartridge engine
//!
//! Connect 4 is a two-player connection game where players drop colored discs
//! into a 7-column, 6-row vertically suspended grid. The objective is to be
//! the first to form a horizontal, vertical, or diagonal line of four discs.
//!
//! # Board Layout
//!
//! The board is stored in row-major order, with row 0 at the bottom:
//! ```text
//! Row 5: [35][36][37][38][39][40][41]  <- Top
//! Row 4: [28][29][30][31][32][33][34]
//! Row 3: [21][22][23][24][25][26][27]
//! Row 2: [14][15][16][17][18][19][20]
//! Row 1: [ 7][ 8][ 9][10][11][12][13]
//! Row 0: [ 0][ 1][ 2][ 3][ 4][ 5][ 6]  <- Bottom
//!         Col 0  1  2  3  4  5  6
//! ```
//!
//! # Usage
//!
//! ```rust
//! use games_connect4::{Connect4, register_connect4};
//! use engine_core::EngineContext;
//!
//! // Register the game with the global registry
//! register_connect4();
//!
//! // Create a context to play
//! let mut ctx = EngineContext::new("connect4").expect("connect4 should be registered");
//! let reset = ctx.reset(42, &[]).unwrap();
//! ```

use engine_core::typed::{
    ActionSpace, Capabilities, DecodeError, EncodeError, Encoding, EngineId, Game,
};
use engine_core::{register_game, GameAdapter, GameMetadata};
use rand_chacha::ChaCha20Rng;

/// Board dimensions
pub const COLS: usize = 7;
pub const ROWS: usize = 6;
pub const BOARD_SIZE: usize = COLS * ROWS; // 42

/// Register Connect4 with the global game registry
///
/// Call this function once at startup to make Connect4 available
/// via `EngineContext::new("connect4")`.
pub fn register_connect4() {
    register_game("connect4".to_string(), || {
        Box::new(GameAdapter::new(Connect4::new()))
    });
}

/// Connect4 game state
///
/// Represents the complete state of a Connect4 game including the board,
/// current player, and winner information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Board representation: 0=empty, 1=Red (player 1), 2=Yellow (player 2)
    /// Stored in row-major order with row 0 at the bottom
    board: [u8; BOARD_SIZE],
    /// Current player: 1=Red, 2=Yellow
    current_player: u8,
    /// Winner: 0=none/ongoing, 1=Red, 2=Yellow, 3=draw
    winner: u8,
    /// Height of each column (0-6 means number of pieces in column)
    column_heights: [u8; COLS],
}

impl State {
    /// Create a new initial game state
    pub fn new() -> Self {
        Self {
            board: [0; BOARD_SIZE],
            current_player: 1, // Red goes first
            winner: 0,
            column_heights: [0; COLS],
        }
    }

    /// Check if the game is over
    pub fn is_done(&self) -> bool {
        self.winner != 0
    }

    /// Get legal moves (columns that are not full)
    pub fn legal_moves(&self) -> Vec<u8> {
        if self.is_done() {
            return Vec::new();
        }

        (0..COLS as u8)
            .filter(|&col| self.column_heights[col as usize] < ROWS as u8)
            .collect()
    }

    /// Bit-mask representation of legal moves.
    ///
    /// Bits 0-6 correspond to columns 0-6. A bit set to 1 indicates the
    /// column is not full and a piece can be dropped there.
    pub fn legal_moves_mask(&self) -> u8 {
        if self.is_done() {
            return 0;
        }

        self.column_heights
            .iter()
            .enumerate()
            .fold(0u8, |mask, (col, &height)| {
                if height < ROWS as u8 {
                    mask | (1u8 << col)
                } else {
                    mask
                }
            })
    }

    /// Convert column and row to board index
    #[inline]
    fn pos(col: usize, row: usize) -> usize {
        row * COLS + col
    }

    /// Drop a piece in the given column and return the new state
    pub fn drop_piece(&self, column: u8) -> State {
        let col = column as usize;

        // Check if move is valid
        if self.is_done() || col >= COLS || self.column_heights[col] >= ROWS as u8 {
            return self.clone(); // Invalid move, return unchanged state
        }

        let mut new_state = self.clone();
        let row = self.column_heights[col] as usize;
        let pos = Self::pos(col, row);

        // Place the piece
        new_state.board[pos] = self.current_player;
        new_state.column_heights[col] += 1;

        // Check for winner
        new_state.winner = new_state.check_winner_at(col, row);

        // Switch player if game not over
        if new_state.winner == 0 {
            new_state.current_player = if self.current_player == 1 { 2 } else { 1 };
        }

        new_state
    }

    /// Check if the piece at (col, row) creates a winning line
    fn check_winner_at(&self, col: usize, row: usize) -> u8 {
        let player = self.board[Self::pos(col, row)];
        if player == 0 {
            return 0;
        }

        // Direction vectors: horizontal, vertical, diagonal /, diagonal \
        let directions: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];

        for (dc, dr) in directions {
            let mut count = 1; // Count the piece we just placed

            // Count in positive direction
            let (mut c, mut r) = (col as i32 + dc, row as i32 + dr);
            while c >= 0 && c < COLS as i32 && r >= 0 && r < ROWS as i32 {
                if self.board[Self::pos(c as usize, r as usize)] == player {
                    count += 1;
                    c += dc;
                    r += dr;
                } else {
                    break;
                }
            }

            // Count in negative direction
            let (mut c, mut r) = (col as i32 - dc, row as i32 - dr);
            while c >= 0 && c < COLS as i32 && r >= 0 && r < ROWS as i32 {
                if self.board[Self::pos(c as usize, r as usize)] == player {
                    count += 1;
                    c -= dc;
                    r -= dr;
                } else {
                    break;
                }
            }

            if count >= 4 {
                return player;
            }
        }

        // Check for draw (board full but no winner)
        if self.column_heights.iter().all(|&h| h >= ROWS as u8) {
            return 3; // Draw
        }

        0 // Game ongoing
    }

    /// Get the row where the last piece was placed in a column
    pub fn last_row_in_column(&self, col: usize) -> Option<usize> {
        if self.column_heights[col] == 0 {
            None
        } else {
            Some((self.column_heights[col] - 1) as usize)
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

/// Connect4 action - drop a piece in a column
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Drop a piece in the given column (0-6)
    Drop(u8),
}

impl Action {
    /// Get the column for this action
    pub fn column(&self) -> u8 {
        match self {
            Action::Drop(col) => *col,
        }
    }
}

/// Connect4 observation
///
/// Provides a neural network-friendly representation of the game state.
/// Format: [Red_positions (42), Yellow_positions (42), legal_moves (7), current_player (2)]
/// Total: 93 floats
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// One-hot encoding of board: [Red_positions (42), Yellow_positions (42)]
    pub board_view: [f32; BOARD_SIZE * 2],
    /// Legal moves mask (7 values: 1.0 = legal, 0.0 = illegal)
    pub legal_moves: [f32; COLS],
    /// Current player indicator: [is_Red, is_Yellow] (2 values)
    pub current_player: [f32; 2],
}

impl Observation {
    /// Create observation from game state
    pub fn from_state(state: &State) -> Self {
        let mut board_view = [0.0; BOARD_SIZE * 2];
        let mut legal_moves = [0.0; COLS];
        let mut current_player = [0.0; 2];

        // Encode board state (one-hot for Red and Yellow)
        for (i, &cell) in state.board.iter().enumerate() {
            if cell == 1 {
                board_view[i] = 1.0; // Red positions
            } else if cell == 2 {
                board_view[i + BOARD_SIZE] = 1.0; // Yellow positions
            }
        }

        // Encode legal moves
        let mask = state.legal_moves_mask();
        for (col, slot) in legal_moves.iter_mut().enumerate() {
            if (mask & (1u8 << col)) != 0 {
                *slot = 1.0;
            }
        }

        // Encode current player
        if state.current_player == 1 {
            current_player[0] = 1.0; // Red
        } else {
            current_player[1] = 1.0; // Yellow
        }

        Self {
            board_view,
            legal_moves,
            current_player,
        }
    }
}

/// Connect4 game implementation
#[derive(Debug)]
pub struct Connect4;

impl Connect4 {
    /// Create a new Connect4 game
    pub fn new() -> Self {
        Self
    }

    /// Calculate reward for the current state
    fn calculate_reward(state: &State, previous_player: u8) -> f32 {
        match state.winner {
            0 => 0.0, // Game ongoing
            1 => {
                if previous_player == 1 {
                    1.0
                } else {
                    -1.0
                }
            } // Red wins
            2 => {
                if previous_player == 2 {
                    1.0
                } else {
                    -1.0
                }
            } // Yellow wins
            3 => 0.0, // Draw
            _ => 0.0, // Shouldn't happen
        }
    }

    /// Pack auxiliary information about the state into a u64 bit-field.
    ///
    /// Layout (little endian bit numbering):
    /// * Bits 0-6  : Legal move mask (7 columns)
    /// * Bits 16-19: Current player (1 = Red, 2 = Yellow)
    /// * Bits 20-23: Winner (0 = none, 1 = Red, 2 = Yellow, 3 = draw)
    /// * Bits 24-31: Moves played so far (0-42)
    fn compute_info_bits(state: &State) -> u64 {
        const CURRENT_PLAYER_SHIFT: u32 = 16;
        const WINNER_SHIFT: u32 = 20;
        const MOVES_PLAYED_SHIFT: u32 = 24;

        let mut info = state.legal_moves_mask() as u64;
        info |= (state.current_player as u64) << CURRENT_PLAYER_SHIFT;
        info |= (state.winner as u64) << WINNER_SHIFT;

        let moves_played: u64 = state.column_heights.iter().map(|&h| h as u64).sum();
        info |= moves_played << MOVES_PLAYED_SHIFT;

        info
    }
}

impl Default for Connect4 {
    fn default() -> Self {
        Self::new()
    }
}

/// Observation size: 42 (Red) + 42 (Yellow) + 7 (legal) + 2 (player) = 93
const OBS_SIZE: usize = BOARD_SIZE * 2 + COLS + 2;

impl Game for Connect4 {
    type State = State;
    type Action = Action;
    type Obs = Observation;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "connect4".to_string(),
            build_id: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: "connect4_state:v1".to_string(),
                action: "discrete_column:v1".to_string(),
                obs: format!("f32x{}:v1", OBS_SIZE), // 93 floats
                schema_version: 1,
            },
            max_horizon: BOARD_SIZE as u32, // Maximum 42 moves
            action_space: ActionSpace::Discrete(COLS as u32), // 7 possible columns
            preferred_batch: 64,
        }
    }

    fn metadata(&self) -> GameMetadata {
        GameMetadata::new("connect4", "Connect 4")
            .with_board(COLS, ROWS)
            .with_actions(COLS)
            .with_observation(OBS_SIZE, BOARD_SIZE * 2) // legal mask starts after board views
            .with_players(
                2,
                vec!["Red".to_string(), "Yellow".to_string()],
                vec!['\u{1F534}', '\u{1F7E1}'], // Red circle, Yellow circle emoji
            )
            .with_description("Drop discs to connect four in a row!")
    }

    fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
        let state = State::new();
        let obs = Observation::from_state(&state);
        (state, obs)
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64) {
        let previous_player = state.current_player;
        *state = state.drop_piece(action.column());

        let obs = Observation::from_state(state);
        let reward = Self::calculate_reward(state, previous_player);
        let done = state.is_done();
        let info = Self::compute_info_bits(state);

        (obs, reward, done, info)
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        // Binary encoding: board (42 bytes) + current_player (1 byte) + winner (1 byte)
        // Note: column_heights can be reconstructed from board
        out.extend_from_slice(&state.board);
        out.push(state.current_player);
        out.push(state.winner);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        let expected_len = BOARD_SIZE + 2; // board + current_player + winner
        if buf.len() != expected_len {
            return Err(DecodeError::InvalidLength {
                expected: expected_len,
                actual: buf.len(),
            });
        }

        let mut board = [0u8; BOARD_SIZE];
        board.copy_from_slice(&buf[0..BOARD_SIZE]);

        let current_player = buf[BOARD_SIZE];
        let winner = buf[BOARD_SIZE + 1];

        // Validate the state
        if current_player != 1 && current_player != 2 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid current_player: {}",
                current_player
            )));
        }

        if winner > 3 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid winner: {}",
                winner
            )));
        }

        for &cell in &board {
            if cell > 2 {
                return Err(DecodeError::CorruptedData(format!(
                    "Invalid board cell: {}",
                    cell
                )));
            }
        }

        // Reconstruct column heights from board
        let mut column_heights = [0u8; COLS];
        for col in 0..COLS {
            for row in 0..ROWS {
                if board[State::pos(col, row)] != 0 {
                    column_heights[col] = (row + 1) as u8;
                }
            }
        }

        Ok(State {
            board,
            current_player,
            winner,
            column_heights,
        })
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        let column = action.column();
        if column as usize >= COLS {
            return Err(EncodeError::InvalidData(format!(
                "Invalid action column: {}",
                column
            )));
        }
        // Encode as u32 in little-endian format (4 bytes)
        out.extend_from_slice(&(column as u32).to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 4 {
            return Err(DecodeError::InvalidLength {
                expected: 4,
                actual: buf.len(),
            });
        }

        let column = u32::from_le_bytes(buf.try_into().unwrap());
        if column as usize >= COLS {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid action column: {}",
                column
            )));
        }

        Ok(Action::Drop(column as u8))
    }

    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        // Encode as OBS_SIZE f32 values in little-endian format
        for &value in &obs.board_view {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for &value in &obs.legal_moves {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for &value in &obs.current_player {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_initial_state() {
        let state = State::new();
        assert_eq!(state.board, [0; BOARD_SIZE]);
        assert_eq!(state.current_player, 1);
        assert_eq!(state.winner, 0);
        assert_eq!(state.column_heights, [0; COLS]);
        assert!(!state.is_done());
    }

    #[test]
    fn test_legal_moves() {
        let state = State::new();
        let legal = state.legal_moves();
        assert_eq!(legal, (0..COLS as u8).collect::<Vec<_>>());
        assert_eq!(state.legal_moves_mask(), 0x7Fu8); // All 7 columns

        // After one move
        let state = state.drop_piece(3); // Center column
        let legal = state.legal_moves();
        assert_eq!(legal.len(), 7); // All columns still available
        assert!(legal.contains(&3));
    }

    #[test]
    fn test_drop_piece() {
        let state = State::new();
        let new_state = state.drop_piece(3); // Red drops in center

        // Piece should be at bottom of column 3
        assert_eq!(new_state.board[State::pos(3, 0)], 1);
        assert_eq!(new_state.column_heights[3], 1);
        assert_eq!(new_state.current_player, 2); // Now Yellow's turn
        assert!(!new_state.is_done());
    }

    #[test]
    fn test_stacking_pieces() {
        let mut state = State::new();

        // Stack pieces in column 0
        for i in 0..ROWS {
            state = state.drop_piece(0);
            assert_eq!(state.column_heights[0], (i + 1) as u8);
        }

        // Column 0 is now full
        assert!(!state.legal_moves().contains(&0));
        assert_eq!(state.legal_moves_mask() & 1, 0);
    }

    #[test]
    fn test_invalid_move_full_column() {
        let mut state = State::new();

        // Fill column 0
        for _ in 0..ROWS {
            state = state.drop_piece(0);
        }

        let before = state.clone();
        let after = state.drop_piece(0); // Try to drop in full column

        // State should be unchanged
        assert_eq!(before, after);
    }

    #[test]
    fn test_horizontal_win() {
        let mut state = State::new();

        // Red: 0, 1, 2, 3 (bottom row)
        // Yellow: 0, 1, 2 (second row)
        state = state.drop_piece(0); // Red at (0,0)
        state = state.drop_piece(0); // Yellow at (0,1)
        state = state.drop_piece(1); // Red at (1,0)
        state = state.drop_piece(1); // Yellow at (1,1)
        state = state.drop_piece(2); // Red at (2,0)
        state = state.drop_piece(2); // Yellow at (2,1)
        state = state.drop_piece(3); // Red at (3,0) - WINS

        assert_eq!(state.winner, 1); // Red wins
        assert!(state.is_done());
        assert!(state.legal_moves().is_empty());
    }

    #[test]
    fn test_vertical_win() {
        let mut state = State::new();

        // Red stacks in column 0, Yellow in column 1
        state = state.drop_piece(0); // Red
        state = state.drop_piece(1); // Yellow
        state = state.drop_piece(0); // Red
        state = state.drop_piece(1); // Yellow
        state = state.drop_piece(0); // Red
        state = state.drop_piece(1); // Yellow
        state = state.drop_piece(0); // Red - WINS

        assert_eq!(state.winner, 1); // Red wins
        assert!(state.is_done());
    }

    #[test]
    fn test_diagonal_win_ascending() {
        let mut state = State::new();

        // Build ascending diagonal for Red: (0,0), (1,1), (2,2), (3,3)
        // Yellow plays defensively in columns 5 and 6 to avoid horizontal wins
        state = state.drop_piece(0); // Red at (0,0)    - Red's turn
        state = state.drop_piece(5); // Yellow at (5,0) - Yellow's turn
        state = state.drop_piece(1); // Red at (1,0)    - Red's turn (need base for col 1)
        state = state.drop_piece(6); // Yellow at (6,0) - Yellow's turn
        state = state.drop_piece(1); // Red at (1,1)    - Red's turn
        state = state.drop_piece(5); // Yellow at (5,1) - Yellow's turn
        state = state.drop_piece(2); // Red at (2,0)    - Red's turn (need base for col 2)
        state = state.drop_piece(6); // Yellow at (6,1) - Yellow's turn
        state = state.drop_piece(2); // Red at (2,1)    - Red's turn
        state = state.drop_piece(5); // Yellow at (5,2) - Yellow's turn
        state = state.drop_piece(2); // Red at (2,2)    - Red's turn
        state = state.drop_piece(6); // Yellow at (6,2) - Yellow's turn
        state = state.drop_piece(3); // Red at (3,0)    - Red's turn (need base for col 3)
        state = state.drop_piece(5); // Yellow at (5,3) - Yellow's turn
        state = state.drop_piece(3); // Red at (3,1)    - Red's turn
        state = state.drop_piece(6); // Yellow at (6,3) - Yellow's turn
        state = state.drop_piece(3); // Red at (3,2)    - Red's turn
        state = state.drop_piece(5); // Yellow at (5,4) - Yellow's turn
        state = state.drop_piece(3); // Red at (3,3)    - Red's turn - WINS

        assert_eq!(state.winner, 1); // Red wins
    }

    #[test]
    fn test_diagonal_win_descending() {
        let mut state = State::new();

        // Build descending diagonal for Red
        // Red at: (3,0), (2,1), (1,2), (0,3)
        state = state.drop_piece(3); // Red at (3,0)
        state = state.drop_piece(2); // Yellow at (2,0)
        state = state.drop_piece(2); // Red at (2,1)
        state = state.drop_piece(1); // Yellow at (1,0)
        state = state.drop_piece(1); // Red at (1,1)
        state = state.drop_piece(0); // Yellow at (0,0)
        state = state.drop_piece(1); // Red at (1,2)
        state = state.drop_piece(0); // Yellow at (0,1)
        state = state.drop_piece(0); // Red at (0,2)
        state = state.drop_piece(4); // Yellow at (4,0)
        state = state.drop_piece(0); // Red at (0,3) - WINS

        assert_eq!(state.winner, 1); // Red wins
    }

    #[test]
    fn test_draw_game() {
        // Creating a draw requires filling all 42 positions without a win
        // This is tricky to construct manually, so we'll test the draw detection logic
        let mut state = State::new();

        // Fill the board in a pattern that creates a draw
        // Pattern that avoids 4-in-a-row:
        // Row 0: R Y R Y R Y R
        // Row 1: R Y R Y R Y R
        // Row 2: Y R Y R Y R Y
        // Row 3: Y R Y R Y R Y
        // Row 4: R Y R Y R Y R
        // Row 5: R Y R Y R Y R

        let pattern = [
            // Column 0: R R Y Y R R
            [1, 1, 2, 2, 1, 1],
            // Column 1: Y Y R R Y Y
            [2, 2, 1, 1, 2, 2],
            // Column 2: R R Y Y R R
            [1, 1, 2, 2, 1, 1],
            // Column 3: Y Y R R Y Y
            [2, 2, 1, 1, 2, 2],
            // Column 4: R R Y Y R R
            [1, 1, 2, 2, 1, 1],
            // Column 5: Y Y R R Y Y
            [2, 2, 1, 1, 2, 2],
            // Column 6: R R Y Y R R
            [1, 1, 2, 2, 1, 1],
        ];

        // Build the board directly
        let mut board = [0u8; BOARD_SIZE];
        for col in 0..COLS {
            for row in 0..ROWS {
                board[State::pos(col, row)] = pattern[col][row];
            }
        }

        state.board = board;
        state.column_heights = [6; COLS];
        state.winner = state.check_winner_at(0, 0); // Check any position

        // Verify it's a draw (no winner but board is full)
        assert_eq!(state.winner, 3);
        assert!(state.is_done());
    }

    #[test]
    fn test_observation_encoding() {
        let state = State::new();
        let obs = Observation::from_state(&state);

        // All board positions should be 0 initially
        assert_eq!(obs.board_view, [0.0; BOARD_SIZE * 2]);
        // All columns should be legal
        assert_eq!(obs.legal_moves, [1.0; COLS]);
        // Red should be current player
        assert_eq!(obs.current_player, [1.0, 0.0]);
    }

    #[test]
    fn test_game_trait_implementation() {
        let mut game = Connect4::new();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let (state, _obs) = game.reset(&mut rng, &[]);
        assert_eq!(state, State::new());

        let action = Action::Drop(3);
        let (_new_obs, reward, done, info) = game.step(&mut state.clone(), action, &mut rng);

        // Should not be done after one move
        assert!(!done);
        // Reward should be 0 for ongoing game
        assert_eq!(reward, 0.0);

        // All columns should still be legal
        assert_eq!(info & 0x7F, 0x7Fu64);
        // Next player should be Yellow (value 2)
        assert_eq!((info >> 16) & 0xF, 2);
    }

    #[test]
    fn test_state_encoding_roundtrip() {
        let mut original_state = State::new();
        original_state = original_state.drop_piece(3);
        original_state = original_state.drop_piece(3);
        original_state = original_state.drop_piece(4);

        let mut buf = Vec::new();
        Connect4::encode_state(&original_state, &mut buf).unwrap();
        let decoded_state = Connect4::decode_state(&buf).unwrap();

        assert_eq!(original_state, decoded_state);
    }

    #[test]
    fn test_action_encoding_roundtrip() {
        for col in 0..COLS as u8 {
            let action = Action::Drop(col);

            let mut buf = Vec::new();
            Connect4::encode_action(&action, &mut buf).unwrap();
            let decoded_action = Connect4::decode_action(&buf).unwrap();

            assert_eq!(action, decoded_action);
        }
    }

    #[test]
    fn test_observation_byte_encoding() {
        let mut state = State::new();
        state = state.drop_piece(3);

        let obs = Observation::from_state(&state);

        let mut buf = Vec::new();
        Connect4::encode_obs(&obs, &mut buf).unwrap();

        // Should be OBS_SIZE * 4 bytes (OBS_SIZE f32 values)
        assert_eq!(buf.len(), OBS_SIZE * 4);
    }

    #[test]
    fn test_engine_capabilities() {
        let game = Connect4::new();
        let caps = game.capabilities();

        assert_eq!(caps.id.env_id, "connect4");
        assert_eq!(caps.max_horizon, BOARD_SIZE as u32);

        match caps.action_space {
            ActionSpace::Discrete(n) => assert_eq!(n, COLS as u32),
            ref other => {
                panic!("Expected discrete action space, but got {:?}", other);
            }
        }
    }

    #[test]
    fn test_invalid_state_decoding() {
        // Test wrong length
        let buf = vec![1, 2, 3]; // Too short
        let result = Connect4::decode_state(&buf);
        assert!(result.is_err());

        // Test invalid current_player
        let mut buf = vec![0; BOARD_SIZE + 2];
        buf[BOARD_SIZE] = 5; // Invalid player
        let result = Connect4::decode_state(&buf);
        assert!(result.is_err());

        // Test invalid winner
        let mut buf = vec![0; BOARD_SIZE + 2];
        buf[BOARD_SIZE] = 1; // Valid player
        buf[BOARD_SIZE + 1] = 5; // Invalid winner
        let result = Connect4::decode_state(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_action_decoding() {
        // Test wrong length
        let buf = vec![1, 2]; // Too short
        let result = Connect4::decode_action(&buf);
        assert!(result.is_err());

        // Test invalid column
        let buf = (7u32).to_le_bytes().to_vec(); // Column out of bounds
        let result = Connect4::decode_action(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata() {
        let game = Connect4::new();
        let meta = game.metadata();

        assert_eq!(meta.env_id, "connect4");
        assert_eq!(meta.display_name, "Connect 4");
        assert_eq!(meta.board_width, COLS);
        assert_eq!(meta.board_height, ROWS);
        assert_eq!(meta.num_actions, COLS);
        assert_eq!(meta.player_count, 2);
    }

    #[test]
    fn test_random_games_invariants() {
        use rand::Rng;

        for seed in 0..20 {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let mut game = Connect4::new();
            let (mut state, _) = game.reset(&mut rng, &[]);

            let mut move_count = 0;
            let max_moves = BOARD_SIZE;

            while !state.is_done() && move_count < max_moves {
                let legal = state.legal_moves();
                assert!(
                    !legal.is_empty(),
                    "Non-done game must have legal moves (seed={}, moves={})",
                    seed,
                    move_count
                );

                // Pick random legal move
                let action_col = legal[rng.gen_range(0..legal.len())];
                let action = Action::Drop(action_col);

                let prev_player = state.current_player;
                let (_, reward, done, info) = game.step(&mut state, action, &mut rng);

                move_count += 1;

                // Invariants
                if done {
                    assert!(
                        state.winner != 0,
                        "Done game must have winner (seed={})",
                        seed
                    );
                    assert!(
                        state.legal_moves().is_empty(),
                        "Done game must have no legal moves (seed={})",
                        seed
                    );
                    // Winner should match reward
                    if state.winner == prev_player {
                        assert_eq!(reward, 1.0, "Winner should get +1 reward (seed={})", seed);
                    } else if state.winner == 3 {
                        assert_eq!(reward, 0.0, "Draw should give 0 reward (seed={})", seed);
                    }
                } else {
                    assert_eq!(
                        reward, 0.0,
                        "Ongoing game should have 0 reward (seed={})",
                        seed
                    );
                    // Player should have switched
                    assert_ne!(
                        state.current_player, prev_player,
                        "Player should switch after move (seed={})",
                        seed
                    );
                }

                // Info bits should match state
                let mask_from_info = (info & 0x7F) as u8;
                assert_eq!(
                    mask_from_info,
                    state.legal_moves_mask(),
                    "Info mask should match state (seed={})",
                    seed
                );
            }

            // Game should finish within max_moves
            assert!(
                state.is_done() || move_count == max_moves,
                "Game should finish within {} moves (seed={})",
                max_moves,
                seed
            );
        }
    }
}
