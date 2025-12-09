//! TicTacToe game implementation for the Cartridge engine
//!
//! This crate provides a complete reference implementation of TicTacToe
//! demonstrating how to implement the Game trait for the engine framework.
//!
//! # Usage
//!
//! ```rust
//! use games_tictactoe::{TicTacToe, register_tictactoe};
//! use engine_core::EngineContext;
//!
//! // Register the game with the global registry
//! register_tictactoe();
//!
//! // Create a context to play
//! let mut ctx = EngineContext::new("tictactoe").expect("tictactoe should be registered");
//! let reset = ctx.reset(42, &[]).unwrap();
//! ```

use engine_core::typed::{
    ActionSpace, Capabilities, DecodeError, EncodeError, Encoding, EngineId, Game,
};
use engine_core::{register_game, GameAdapter, GameMetadata};
use rand_chacha::ChaCha20Rng;

/// Register TicTacToe with the global game registry
///
/// Call this function once at startup to make TicTacToe available
/// via `EngineContext::new("tictactoe")`.
pub fn register_tictactoe() {
    register_game("tictactoe".to_string(), || {
        Box::new(GameAdapter::new(TicTacToe::new()))
    });
}

/// TicTacToe game state
///
/// Represents the complete state of a TicTacToe game including the board,
/// current player, and winner information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    /// Board representation: 0=empty, 1=X, 2=O
    board: [u8; 9],
    /// Current player: 1=X, 2=O
    current_player: u8,
    /// Winner: 0=none/ongoing, 1=X, 2=O, 3=draw
    winner: u8,
}

impl State {
    /// Create a new initial game state
    pub fn new() -> Self {
        Self {
            board: [0; 9],
            current_player: 1, // X goes first
            winner: 0,
        }
    }

    /// Check if the game is over
    pub fn is_done(&self) -> bool {
        self.winner != 0
    }

    /// Get legal moves (empty positions)
    pub fn legal_moves(&self) -> Vec<u8> {
        if self.is_done() {
            return Vec::new();
        }

        (0..9u8)
            .filter(|&pos| self.board[pos as usize] == 0)
            .collect()
    }

    /// Bit-mask representation of legal moves.
    ///
    /// Bits 0-8 correspond to board positions 0-8. A bit set to 1 indicates the
    /// position is currently legal. When the game is finished the mask is zeroed.
    pub fn legal_moves_mask(&self) -> u16 {
        if self.is_done() {
            return 0;
        }

        self.board
            .iter()
            .enumerate()
            .fold(0u16, |mask, (idx, cell)| {
                if *cell == 0 {
                    mask | (1u16 << idx)
                } else {
                    mask
                }
            })
    }

    /// Make a move and return the new state
    pub fn make_move(&self, position: u8) -> State {
        if self.is_done() || position >= 9 || self.board[position as usize] != 0 {
            return *self; // Invalid move, return unchanged state
        }

        let mut new_state = *self;
        new_state.board[position as usize] = self.current_player;

        // Check for winner
        new_state.winner = Self::check_winner(&new_state.board);

        // Switch player if game not over
        if new_state.winner == 0 {
            new_state.current_player = if self.current_player == 1 { 2 } else { 1 };
        }

        new_state
    }

    /// Check for winner on the board
    fn check_winner(board: &[u8; 9]) -> u8 {
        // Winning positions (rows, columns, diagonals)
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8], // rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8], // columns
            [0, 4, 8],
            [2, 4, 6], // diagonals
        ];

        for line in &LINES {
            let [a, b, c] = *line;
            if board[a] != 0 && board[a] == board[b] && board[b] == board[c] {
                return board[a]; // Return the winning player
            }
        }

        // Check for draw (board full but no winner)
        if board.iter().all(|&cell| cell != 0) {
            return 3; // Draw
        }

        0 // Game ongoing
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

/// TicTacToe action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Place a piece at the given position (0-8)
    Place(u8),
}

impl Action {
    /// Get the position for this action
    pub fn position(&self) -> u8 {
        match self {
            Action::Place(pos) => *pos,
        }
    }
}

/// TicTacToe observation
///
/// Provides a neural network-friendly representation of the game state
/// including board state and legal move mask.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// One-hot encoding of board: [X_positions, O_positions] (18 values)
    pub board_view: [f32; 18],
    /// Legal moves mask (9 values: 1.0 = legal, 0.0 = illegal)
    pub legal_moves: [f32; 9],
    /// Current player indicator: [is_X, is_O] (2 values)
    pub current_player: [f32; 2],
}

impl Observation {
    /// Create observation from game state
    pub fn from_state(state: &State) -> Self {
        let mut board_view = [0.0; 18];
        let mut legal_moves = [0.0; 9];
        let mut current_player = [0.0; 2];

        // Encode board state (one-hot for X and O)
        for (i, &cell) in state.board.iter().enumerate() {
            if cell == 1 {
                board_view[i] = 1.0; // X positions
            } else if cell == 2 {
                board_view[i + 9] = 1.0; // O positions
            }
        }

        // Encode legal moves
        let mask = state.legal_moves_mask();
        for (pos, slot) in legal_moves.iter_mut().enumerate() {
            if (mask & (1u16 << pos)) != 0 {
                *slot = 1.0;
            }
        }

        // Encode current player
        if state.current_player == 1 {
            current_player[0] = 1.0; // X
        } else {
            current_player[1] = 1.0; // O
        }

        Self {
            board_view,
            legal_moves,
            current_player,
        }
    }
}

/// TicTacToe game implementation
#[derive(Debug)]
pub struct TicTacToe;

impl TicTacToe {
    /// Create a new TicTacToe game
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
            } // X wins
            2 => {
                if previous_player == 2 {
                    1.0
                } else {
                    -1.0
                }
            } // O wins
            3 => 0.0, // Draw
            _ => 0.0, // Shouldn't happen
        }
    }

    /// Pack auxiliary information about the state into a u64 bit-field.
    ///
    /// Layout (little endian bit numbering):
    /// * Bits 0-8  : Legal move mask
    /// * Bits 16-19: Current player (1 = X, 2 = O)
    /// * Bits 20-23: Winner (0 = none, 1 = X, 2 = O, 3 = draw)
    /// * Bits 24-27: Moves played so far
    fn compute_info_bits(state: &State) -> u64 {
        const CURRENT_PLAYER_SHIFT: u32 = 16;
        const WINNER_SHIFT: u32 = 20;
        const MOVES_PLAYED_SHIFT: u32 = 24;

        let mut info = state.legal_moves_mask() as u64;
        info |= (state.current_player as u64) << CURRENT_PLAYER_SHIFT;
        info |= (state.winner as u64) << WINNER_SHIFT;

        let moves_played = state.board.iter().filter(|&&cell| cell != 0).count() as u64;
        info |= moves_played << MOVES_PLAYED_SHIFT;

        info
    }
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for TicTacToe {
    type State = State;
    type Action = Action;
    type Obs = Observation;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "tictactoe".to_string(),
            build_id: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: "tictactoe_state:v1".to_string(),
                action: "discrete_position:v1".to_string(),
                obs: "f32x29:v1".to_string(), // 18 + 9 + 2 = 29 floats
                schema_version: 1,
            },
            max_horizon: 9,                         // Maximum 9 moves in TicTacToe
            action_space: ActionSpace::Discrete(9), // 9 possible positions
            preferred_batch: 64,
        }
    }

    fn metadata(&self) -> GameMetadata {
        GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_board(3, 3)
            .with_actions(9)
            .with_observation(29, 18) // 29 floats, legal mask starts at index 18
            .with_players(
                2,
                vec!["X".to_string(), "O".to_string()],
                vec!['X', 'O'],
            )
            .with_description("Get three in a row to win!")
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
        *state = state.make_move(action.position());

        let obs = Observation::from_state(state);
        let reward = Self::calculate_reward(state, previous_player);
        let done = state.is_done();
        let info = Self::compute_info_bits(state);

        (obs, reward, done, info)
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        // Simple binary encoding: board (9 bytes) + current_player (1 byte) + winner (1 byte)
        out.extend_from_slice(&state.board);
        out.push(state.current_player);
        out.push(state.winner);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        if buf.len() != 11 {
            return Err(DecodeError::InvalidLength {
                expected: 11,
                actual: buf.len(),
            });
        }

        let mut board = [0u8; 9];
        board.copy_from_slice(&buf[0..9]);

        let current_player = buf[9];
        let winner = buf[10];

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

        Ok(State {
            board,
            current_player,
            winner,
        })
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        let position = action.position();
        if position >= 9 {
            return Err(EncodeError::InvalidData(format!(
                "Invalid action position: {}",
                position
            )));
        }
        // Encode as u32 in little-endian format (4 bytes)
        out.extend_from_slice(&(position as u32).to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 4 {
            return Err(DecodeError::InvalidLength {
                expected: 4,
                actual: buf.len(),
            });
        }

        let position = u32::from_le_bytes(buf.try_into().unwrap());
        if position >= 9 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid action position: {}",
                position
            )));
        }

        Ok(Action::Place(position as u8))
    }

    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        // Encode as 29 f32 values in little-endian format
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
        assert_eq!(state.board, [0; 9]);
        assert_eq!(state.current_player, 1);
        assert_eq!(state.winner, 0);
        assert!(!state.is_done());
    }

    #[test]
    fn test_legal_moves() {
        let state = State::new();
        let legal = state.legal_moves();
        assert_eq!(legal, (0..9).collect::<Vec<_>>());
        assert_eq!(state.legal_moves_mask(), 0x1FFu16);

        // After one move
        let state = state.make_move(4); // Center
        let legal = state.legal_moves();
        assert_eq!(legal.len(), 8);
        assert!(!legal.contains(&4));
        assert_eq!(state.legal_moves_mask(), 0x1FFu16 & !(1u16 << 4));
    }

    #[test]
    fn test_make_move() {
        let state = State::new();
        let new_state = state.make_move(4); // X places in center

        assert_eq!(new_state.board[4], 1);
        assert_eq!(new_state.current_player, 2); // Now O's turn
        assert!(!new_state.is_done());
    }

    #[test]
    fn test_invalid_move() {
        let state = State::new();
        let state_with_move = state.make_move(4);

        // Try to place in same position
        let invalid_state = state_with_move.make_move(4);
        assert_eq!(invalid_state, state_with_move); // Should be unchanged
    }

    #[test]
    fn test_winning_game() {
        let mut state = State::new();

        // X wins with top row
        state = state.make_move(0); // X
        state = state.make_move(3); // O
        state = state.make_move(1); // X
        state = state.make_move(4); // O
        state = state.make_move(2); // X wins

        assert_eq!(state.winner, 1);
        assert!(state.is_done());
        assert!(state.legal_moves().is_empty());
    }

    #[test]
    fn test_draw_game() {
        // Create a draw state manually since getting the exact move sequence is tricky
        // Board: X O X / O X O / O X O
        let state = State {
            board: [1, 2, 1, 2, 1, 2, 2, 1, 2], // X=1, O=2
            current_player: 1,                  // Doesn't matter since game is over
            winner: 3,                          // This should be detected as a draw
        };

        // Verify this is actually a draw by checking the game logic
        let detected_winner = State::check_winner(&state.board);
        assert_eq!(detected_winner, 3); // Should be draw
        assert!(state.is_done());
    }

    #[test]
    fn test_observation_encoding() {
        let state = State::new();
        let obs = Observation::from_state(&state);

        // All board positions should be 0 initially
        assert_eq!(obs.board_view, [0.0; 18]);
        // All moves should be legal
        assert_eq!(obs.legal_moves, [1.0; 9]);
        // X should be current player
        assert_eq!(obs.current_player, [1.0, 0.0]);
    }

    #[test]
    fn test_game_trait_implementation() {
        let mut game = TicTacToe::new();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let (state, _obs) = game.reset(&mut rng, &[]);
        assert_eq!(state, State::new());

        let action = Action::Place(4);
        let (_new_obs, reward, done, info) = game.step(&mut state.clone(), action, &mut rng);

        // Should not be done after one move
        assert!(!done);
        // Reward should be 0 for ongoing game
        assert_eq!(reward, 0.0);

        // Mask should no longer include the center position
        assert_eq!(info & 0x1FF, 0x1FFu64 & !(1u64 << 4));
        // Next player should be O (value 2)
        assert_eq!((info >> 16) & 0xF, 2);
    }

    #[test]
    fn test_state_encoding_roundtrip() {
        let original_state = State {
            board: [1, 0, 2, 0, 1, 0, 2, 0, 0],
            current_player: 2,
            winner: 0,
        };

        let mut buf = Vec::new();
        TicTacToe::encode_state(&original_state, &mut buf).unwrap();
        let decoded_state = TicTacToe::decode_state(&buf).unwrap();

        assert_eq!(original_state, decoded_state);
    }

    #[test]
    fn test_action_encoding_roundtrip() {
        let action = Action::Place(5);

        let mut buf = Vec::new();
        TicTacToe::encode_action(&action, &mut buf).unwrap();
        let decoded_action = TicTacToe::decode_action(&buf).unwrap();

        assert_eq!(action, decoded_action);
    }

    #[test]
    fn test_observation_byte_encoding() {
        let state = State {
            board: [1, 0, 2, 0, 0, 0, 0, 0, 0],
            current_player: 2,
            winner: 0,
        };
        let obs = Observation::from_state(&state);

        let mut buf = Vec::new();
        TicTacToe::encode_obs(&obs, &mut buf).unwrap();

        // Should be 29 * 4 = 116 bytes (29 f32 values)
        assert_eq!(buf.len(), 116);
    }

    #[test]
    fn test_engine_capabilities() {
        let game = TicTacToe::new();
        let caps = game.capabilities();

        assert_eq!(caps.id.env_id, "tictactoe");
        assert_eq!(caps.max_horizon, 9);

        match caps.action_space {
            ActionSpace::Discrete(n) => assert_eq!(n, 9),
            ref other => {
                panic!("Expected discrete action space, but got {:?}", other);
            }
        }
    }

    #[test]
    fn test_invalid_state_decoding() {
        // Test wrong length
        let buf = vec![1, 2, 3]; // Too short
        let result = TicTacToe::decode_state(&buf);
        assert!(result.is_err());

        // Test invalid current_player
        let mut buf = vec![0; 11];
        buf[9] = 5; // Invalid player
        let result = TicTacToe::decode_state(&buf);
        assert!(result.is_err());

        // Test invalid winner
        let mut buf = vec![0; 11];
        buf[9] = 1; // Valid player
        buf[10] = 5; // Invalid winner
        let result = TicTacToe::decode_state(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_action_decoding() {
        // Test wrong length
        let buf = vec![1, 2]; // Too short
        let result = TicTacToe::decode_action(&buf);
        assert!(result.is_err());

        // Test invalid position
        let buf = 9u32.to_le_bytes().to_vec(); // Position out of bounds
        let result = TicTacToe::decode_action(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_info_bits_encoding() {
        let state = State {
            board: [1, 2, 1, 0, 2, 0, 0, 0, 0],
            current_player: 1,
            winner: 0,
        };

        let info = TicTacToe::compute_info_bits(&state);

        assert_eq!(info & 0x1FF, state.legal_moves_mask() as u64);
        assert_eq!((info >> 16) & 0xF, state.current_player as u64);
        assert_eq!((info >> 20) & 0xF, state.winner as u64);
        // Four occupied squares
        assert_eq!((info >> 24) & 0xF, 4);
    }

    // =========================================================================
    // Property tests for TicTacToe rules
    // =========================================================================

    /// All 8 winning lines should be detected correctly
    #[test]
    fn test_all_winning_lines() {
        // Rows
        let row_lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8]];
        // Columns
        let col_lines = [[0, 3, 6], [1, 4, 7], [2, 5, 8]];
        // Diagonals
        let diag_lines = [[0, 4, 8], [2, 4, 6]];

        let all_lines: Vec<[usize; 3]> = row_lines
            .iter()
            .chain(col_lines.iter())
            .chain(diag_lines.iter())
            .copied()
            .collect();

        assert_eq!(all_lines.len(), 8, "Should have 8 winning lines");

        for (line_idx, line) in all_lines.iter().enumerate() {
            // Test X wins on this line
            let mut board_x = [0u8; 9];
            for &pos in line {
                board_x[pos] = 1; // X
            }
            let winner = State::check_winner(&board_x);
            assert_eq!(winner, 1, "X should win on line {}: {:?}", line_idx, line);

            // Test O wins on this line
            let mut board_o = [0u8; 9];
            for &pos in line {
                board_o[pos] = 2; // O
            }
            let winner = State::check_winner(&board_o);
            assert_eq!(winner, 2, "O should win on line {}: {:?}", line_idx, line);
        }
    }

    /// Legal move mask should match legal_moves vector
    #[test]
    fn test_legal_moves_mask_consistency() {
        // Test various board configurations
        let boards = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0], // Empty
            [1, 0, 0, 0, 0, 0, 0, 0, 0], // One move
            [1, 2, 1, 2, 0, 0, 0, 0, 0], // Four moves
            [1, 2, 1, 2, 1, 2, 0, 0, 0], // Six moves
            [1, 2, 1, 2, 1, 2, 2, 1, 0], // Eight moves
        ];

        for board in &boards {
            let state = State {
                board: *board,
                current_player: 1,
                winner: 0,
            };

            let legal_vec = state.legal_moves();
            let legal_mask = state.legal_moves_mask();

            // Verify mask has correct bits set
            for pos in 0..9u8 {
                let is_in_vec = legal_vec.contains(&pos);
                let is_in_mask = (legal_mask & (1u16 << pos)) != 0;
                assert_eq!(
                    is_in_vec, is_in_mask,
                    "Mismatch at pos {} for board {:?}",
                    pos, board
                );
            }

            // Verify counts match
            assert_eq!(
                legal_vec.len(),
                legal_mask.count_ones() as usize,
                "Count mismatch for board {:?}",
                board
            );
        }
    }

    /// Draw detection: full board with no winner
    #[test]
    fn test_draw_detection_comprehensive() {
        // Several known draw configurations
        let draw_boards = [
            [1, 2, 1, 1, 2, 2, 2, 1, 1], // X O X / X O O / O X X
            [1, 2, 1, 2, 1, 1, 2, 1, 2], // X O X / O X X / O X O
            [2, 1, 2, 2, 1, 1, 1, 2, 2], // O X O / O X X / X O O
        ];

        for board in &draw_boards {
            let winner = State::check_winner(board);
            assert_eq!(winner, 3, "Should detect draw for board {:?}", board);

            let state = State {
                board: *board,
                current_player: 1,
                winner,
            };
            assert!(state.is_done());
            assert!(state.legal_moves().is_empty());
            assert_eq!(state.legal_moves_mask(), 0);
        }
    }

    /// Reward symmetry: X winning gives +1 to X, -1 to O (and vice versa)
    #[test]
    fn test_reward_symmetry() {
        let mut game = TicTacToe::new();
        let mut rng = ChaCha20Rng::seed_from_u64(123);

        // Play game where X wins (top row)
        let (mut state, _) = game.reset(&mut rng, &[]);

        // X at 0
        let (_, r0, _, _) = game.step(&mut state, Action::Place(0), &mut rng);
        assert_eq!(r0, 0.0); // No winner yet

        // O at 3
        let (_, r1, _, _) = game.step(&mut state, Action::Place(3), &mut rng);
        assert_eq!(r1, 0.0);

        // X at 1
        let (_, r2, _, _) = game.step(&mut state, Action::Place(1), &mut rng);
        assert_eq!(r2, 0.0);

        // O at 4
        let (_, r3, _, _) = game.step(&mut state, Action::Place(4), &mut rng);
        assert_eq!(r3, 0.0);

        // X at 2 - X wins!
        let (_, r4, done, _) = game.step(&mut state, Action::Place(2), &mut rng);
        assert!(done);
        assert_eq!(r4, 1.0, "X (previous player) should get +1 for winning");
    }

    /// No moves allowed on finished game
    #[test]
    fn test_no_moves_after_game_over() {
        // Create a state where X has won
        let state = State {
            board: [1, 1, 1, 2, 2, 0, 0, 0, 0],
            current_player: 2, // Doesn't matter
            winner: 1,         // X won
        };

        assert!(state.is_done());
        assert!(state.legal_moves().is_empty());
        assert_eq!(state.legal_moves_mask(), 0);

        // Attempting a move should return unchanged state
        let new_state = state.make_move(5);
        assert_eq!(new_state, state);
    }

    // =========================================================================
    // Encode/decode roundtrip tests
    // =========================================================================

    /// Test state encoding roundtrip for all valid states
    #[test]
    fn test_state_encoding_roundtrip_comprehensive() {
        let test_states = [
            State::new(),
            State {
                board: [1, 0, 0, 0, 0, 0, 0, 0, 0],
                current_player: 2,
                winner: 0,
            },
            State {
                board: [1, 2, 1, 2, 1, 2, 0, 0, 0],
                current_player: 1,
                winner: 0,
            },
            State {
                board: [1, 1, 1, 2, 2, 0, 0, 0, 0],
                current_player: 2,
                winner: 1,
            },
            State {
                board: [1, 2, 1, 1, 2, 2, 2, 1, 1],
                current_player: 1,
                winner: 3,
            },
        ];

        for original in &test_states {
            let mut buf = Vec::new();
            TicTacToe::encode_state(original, &mut buf).expect("encode should succeed");
            assert_eq!(buf.len(), 11, "State should encode to 11 bytes");

            let decoded = TicTacToe::decode_state(&buf).expect("decode should succeed");
            assert_eq!(original, &decoded, "Roundtrip should preserve state");
        }
    }

    /// Test action encoding roundtrip for all 9 positions
    #[test]
    fn test_action_encoding_roundtrip_all_positions() {
        for pos in 0..9u8 {
            let action = Action::Place(pos);
            let mut buf = Vec::new();
            TicTacToe::encode_action(&action, &mut buf).expect("encode should succeed");
            assert_eq!(buf.len(), 4, "Action should encode to 4 bytes (u32)");

            let decoded = TicTacToe::decode_action(&buf).expect("decode should succeed");
            assert_eq!(
                action, decoded,
                "Roundtrip should preserve action at pos {}",
                pos
            );
        }
    }

    /// Test observation encoding roundtrip
    #[test]
    fn test_observation_encoding_roundtrip() {
        let states = [
            State::new(),
            State {
                board: [1, 2, 0, 0, 1, 0, 0, 0, 2],
                current_player: 1,
                winner: 0,
            },
        ];

        for state in &states {
            let obs = Observation::from_state(state);
            let mut buf = Vec::new();
            TicTacToe::encode_obs(&obs, &mut buf).expect("encode should succeed");
            assert_eq!(
                buf.len(),
                116,
                "Observation should encode to 116 bytes (29 * 4)"
            );

            // Decode manually and verify
            let decoded_floats: Vec<f32> = buf
                .chunks(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            assert_eq!(decoded_floats.len(), 29);

            // Verify board_view
            for (i, &decoded_val) in decoded_floats.iter().enumerate().take(18) {
                assert_eq!(
                    decoded_val, obs.board_view[i],
                    "board_view[{}] mismatch",
                    i
                );
            }
            // Verify legal_moves
            for i in 0..9 {
                assert_eq!(
                    decoded_floats[18 + i],
                    obs.legal_moves[i],
                    "legal_moves[{}] mismatch",
                    i
                );
            }
            // Verify current_player
            assert_eq!(decoded_floats[27], obs.current_player[0]);
            assert_eq!(decoded_floats[28], obs.current_player[1]);
        }
    }

    // =========================================================================
    // Fuzz-style tests with random seeds/actions
    // =========================================================================

    /// Play many random games and verify invariants hold
    #[test]
    fn test_random_games_invariants() {
        use rand::Rng;

        for seed in 0..50 {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let mut game = TicTacToe::new();
            let (mut state, _) = game.reset(&mut rng, &[]);

            let mut move_count = 0;
            let max_moves = 9; // TicTacToe has at most 9 moves

            while !state.is_done() && move_count < max_moves {
                let legal = state.legal_moves();
                assert!(
                    !legal.is_empty(),
                    "Non-done game must have legal moves (seed={}, moves={})",
                    seed,
                    move_count
                );

                // Pick random legal move
                let action_pos = legal[rng.gen_range(0..legal.len())];
                let action = Action::Place(action_pos);

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
                let mask_from_info = (info & 0x1FF) as u16;
                assert_eq!(
                    mask_from_info,
                    state.legal_moves_mask(),
                    "Info mask should match state (seed={})",
                    seed
                );
            }

            // Game should finish within 9 moves
            assert!(
                state.is_done(),
                "Game should finish within 9 moves (seed={})",
                seed
            );
        }
    }

    /// Verify encode/decode doesn't panic for random valid states
    #[test]
    fn test_encoding_no_panic_random_states() {
        use rand::Rng;

        for seed in 0..100 {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);

            // Generate random board (not necessarily valid game state)
            let mut board = [0u8; 9];
            for cell in board.iter_mut() {
                *cell = rng.gen_range(0..=2);
            }
            let current_player = rng.gen_range(1..=2);
            let winner = State::check_winner(&board);

            let state = State {
                board,
                current_player,
                winner,
            };

            // Encode should not panic
            let mut buf = Vec::new();
            let encode_result = TicTacToe::encode_state(&state, &mut buf);
            assert!(
                encode_result.is_ok(),
                "Encode should succeed for seed {}",
                seed
            );

            // Decode should roundtrip
            let decode_result = TicTacToe::decode_state(&buf);
            assert!(
                decode_result.is_ok(),
                "Decode should succeed for seed {}",
                seed
            );
            assert_eq!(state, decode_result.unwrap());
        }
    }

    /// Test that invalid action positions are rejected
    #[test]
    fn test_invalid_action_positions() {
        for invalid_pos in [9, 10, 100, 255] {
            let buf = (invalid_pos as u32).to_le_bytes().to_vec();
            let result = TicTacToe::decode_action(&buf);
            assert!(
                result.is_err(),
                "Position {} should be invalid",
                invalid_pos
            );
        }
    }
}
