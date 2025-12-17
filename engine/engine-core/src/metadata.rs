//! Game metadata for UI and configuration
//!
//! This module provides display-oriented metadata about games that can be
//! used by frontends, actors, and trainers to configure themselves dynamically.

use serde::{Deserialize, Serialize};

/// Metadata about a game for UI display and configuration
///
/// This struct contains all the information needed to:
/// - Display the game in a UI (board dimensions, player symbols)
/// - Configure actors and trainers (obs_size, num_actions)
/// - Parse observations correctly (legal_mask_offset)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameMetadata {
    /// Environment identifier (e.g., "tictactoe", "connect4")
    pub env_id: String,

    /// Human-readable display name (e.g., "Tic-Tac-Toe", "Connect 4")
    pub display_name: String,

    /// Board width in cells
    pub board_width: usize,

    /// Board height in cells
    pub board_height: usize,

    /// Number of possible actions
    pub num_actions: usize,

    /// Size of observation vector (number of f32 values)
    pub obs_size: usize,

    /// Offset in observation where legal moves mask starts
    /// (index of first legal move indicator in the obs array)
    pub legal_mask_offset: usize,

    /// Number of players (typically 2)
    pub player_count: usize,

    /// Display names for each player (e.g., ["X", "O"] or ["Black", "White"])
    pub player_names: Vec<String>,

    /// Single-character symbols for each player (e.g., ['X', 'O'] or ['B', 'W'])
    pub player_symbols: Vec<char>,

    /// Brief description of the game rules for UI tooltips
    pub description: String,

    /// Board rendering type for the frontend
    /// - "grid": Simple grid where clicks place pieces directly (TicTacToe, Othello)
    /// - "drop_column": Column-based where pieces drop to bottom (Connect 4)
    pub board_type: String,
}

impl GameMetadata {
    /// Create a new GameMetadata with required fields
    pub fn new(env_id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            env_id: env_id.into(),
            display_name: display_name.into(),
            board_width: 0,
            board_height: 0,
            num_actions: 0,
            obs_size: 0,
            legal_mask_offset: 0,
            player_count: 2,
            player_names: vec!["Player 1".to_string(), "Player 2".to_string()],
            player_symbols: vec!['1', '2'],
            description: String::new(),
            board_type: "grid".to_string(),
        }
    }

    /// Builder method for board dimensions
    pub fn with_board(mut self, width: usize, height: usize) -> Self {
        self.board_width = width;
        self.board_height = height;
        self
    }

    /// Builder method for action count
    pub fn with_actions(mut self, num_actions: usize) -> Self {
        self.num_actions = num_actions;
        self
    }

    /// Builder method for observation size and legal mask offset
    pub fn with_observation(mut self, obs_size: usize, legal_mask_offset: usize) -> Self {
        self.obs_size = obs_size;
        self.legal_mask_offset = legal_mask_offset;
        self
    }

    /// Builder method for player information
    pub fn with_players(mut self, count: usize, names: Vec<String>, symbols: Vec<char>) -> Self {
        self.player_count = count;
        self.player_names = names;
        self.player_symbols = symbols;
        self
    }

    /// Builder method for description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Builder method for board type
    /// - "grid": Simple grid where clicks place pieces directly (TicTacToe, Othello)
    /// - "drop_column": Column-based where pieces drop to bottom (Connect 4)
    pub fn with_board_type(mut self, board_type: impl Into<String>) -> Self {
        self.board_type = board_type.into();
        self
    }

    /// Get the total number of board cells
    pub fn board_size(&self) -> usize {
        self.board_width * self.board_height
    }

    /// Create a bitmask for extracting legal moves from info bits
    /// based on num_actions
    pub fn legal_mask_bits(&self) -> u64 {
        (1u64 << self.num_actions) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_builder() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_board(3, 3)
            .with_actions(9)
            .with_observation(29, 18)
            .with_players(2, vec!["X".to_string(), "O".to_string()], vec!['X', 'O'])
            .with_description("Get three in a row to win!");

        assert_eq!(meta.env_id, "tictactoe");
        assert_eq!(meta.display_name, "Tic-Tac-Toe");
        assert_eq!(meta.board_width, 3);
        assert_eq!(meta.board_height, 3);
        assert_eq!(meta.num_actions, 9);
        assert_eq!(meta.obs_size, 29);
        assert_eq!(meta.legal_mask_offset, 18);
        assert_eq!(meta.player_count, 2);
        assert_eq!(meta.player_names, vec!["X", "O"]);
        assert_eq!(meta.player_symbols, vec!['X', 'O']);
        assert_eq!(meta.description, "Get three in a row to win!");
    }

    #[test]
    fn test_board_size() {
        let meta = GameMetadata::new("test", "Test").with_board(7, 6);
        assert_eq!(meta.board_size(), 42);
    }

    #[test]
    fn test_legal_mask_bits() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe").with_actions(9);
        assert_eq!(meta.legal_mask_bits(), 0x1FF);

        let meta = GameMetadata::new("connect4", "Connect 4").with_actions(7);
        assert_eq!(meta.legal_mask_bits(), 0x7F);
    }

    #[test]
    fn test_serialization() {
        let meta = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_board(3, 3)
            .with_actions(9);

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: GameMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(meta, parsed);
    }
}
