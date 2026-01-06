//! Shared types for two-player board games.
//!
//! This module provides reusable observation types that eliminate code
//! duplication between similar board games like TicTacToe and Connect4.

use crate::game_utils::encode_f32_slices;

/// Neural network observation for two-player board games.
///
/// Generic over board view size and number of actions to support different board sizes.
/// - `BOARD_VIEW_SIZE`: Total size of one-hot board encoding (board_size * 2 for two players)
/// - `NUM_ACTIONS`: Number of possible actions (board positions or columns)
#[derive(Debug, Clone, PartialEq)]
pub struct TwoPlayerObs<const BOARD_VIEW_SIZE: usize, const NUM_ACTIONS: usize> {
    /// One-hot encoding of board: [player1_positions, player2_positions]
    pub board_view: [f32; BOARD_VIEW_SIZE],
    /// Legal moves mask (1.0 = legal, 0.0 = illegal)
    pub legal_moves: [f32; NUM_ACTIONS],
    /// Current player indicator: [is_player1, is_player2]
    pub current_player: [f32; 2],
}

impl<const BOARD_VIEW_SIZE: usize, const NUM_ACTIONS: usize>
    TwoPlayerObs<BOARD_VIEW_SIZE, NUM_ACTIONS>
{
    /// Create a new empty observation.
    pub fn new() -> Self {
        Self {
            board_view: [0.0; BOARD_VIEW_SIZE],
            legal_moves: [0.0; NUM_ACTIONS],
            current_player: [0.0; 2],
        }
    }

    /// Create observation from board state.
    ///
    /// - `board`: Slice of cell values (0=empty, 1=player1, 2=player2)
    /// - `legal_mask`: Bitmask of legal moves
    /// - `current_player`: Current player (1 or 2)
    pub fn from_board(board: &[u8], legal_mask: u64, current_player: u8) -> Self {
        let mut obs = Self::new();
        let board_size = BOARD_VIEW_SIZE / 2;

        // Encode board state (one-hot for each player)
        for (i, &cell) in board.iter().enumerate() {
            if cell == 1 {
                obs.board_view[i] = 1.0;
            } else if cell == 2 {
                obs.board_view[i + board_size] = 1.0;
            }
        }

        // Encode legal moves
        for (pos, slot) in obs.legal_moves.iter_mut().enumerate() {
            if (legal_mask & (1u64 << pos)) != 0 {
                *slot = 1.0;
            }
        }

        // Encode current player
        if current_player == 1 {
            obs.current_player[0] = 1.0;
        } else {
            obs.current_player[1] = 1.0;
        }

        obs
    }

    /// Encode observation as bytes for neural network input.
    pub fn encode(&self, out: &mut Vec<u8>) {
        encode_f32_slices(
            out,
            [
                &self.board_view[..],
                &self.legal_moves[..],
                &self.current_player[..],
            ],
        );
    }

    /// Total observation size in floats.
    pub const fn obs_size() -> usize {
        BOARD_VIEW_SIZE + NUM_ACTIONS + 2
    }
}

impl<const BOARD_VIEW_SIZE: usize, const NUM_ACTIONS: usize> Default
    for TwoPlayerObs<BOARD_VIEW_SIZE, NUM_ACTIONS>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tictactoe_obs() {
        // TicTacToe: 9 positions * 2 players = 18, 9 actions
        type TicTacToeObs = TwoPlayerObs<18, 9>;

        let board = [1, 0, 2, 0, 1, 0, 0, 0, 0u8];
        let legal_mask = 0b111101010u64; // positions 1, 3, 5, 6, 7, 8
        let obs = TicTacToeObs::from_board(&board, legal_mask, 2);

        // Player 1 at positions 0, 4
        assert_eq!(obs.board_view[0], 1.0);
        assert_eq!(obs.board_view[4], 1.0);
        // Player 2 at position 2
        assert_eq!(obs.board_view[9 + 2], 1.0);
        // Current player is 2
        assert_eq!(obs.current_player, [0.0, 1.0]);
        // Check obs size
        assert_eq!(TicTacToeObs::obs_size(), 29);
    }

    #[test]
    fn test_connect4_obs() {
        // Connect4: 42 positions * 2 players = 84, 7 actions
        type Connect4Obs = TwoPlayerObs<84, 7>;

        let mut board = [0u8; 42];
        board[3] = 1; // Red at column 3, row 0
        let legal_mask = 0b1111111u64; // All 7 columns
        let obs = Connect4Obs::from_board(&board, legal_mask, 1);

        assert_eq!(obs.board_view[3], 1.0);
        assert_eq!(obs.current_player, [1.0, 0.0]);
        assert_eq!(obs.legal_moves, [1.0; 7]);
        assert_eq!(Connect4Obs::obs_size(), 93);
    }
}
