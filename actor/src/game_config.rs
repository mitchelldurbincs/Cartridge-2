//! Game configuration registry for the actor
//!
//! This module provides game-specific configuration that the actor needs
//! for MCTS and observation parsing. The configuration is derived from
//! the engine's GameMetadata.

use anyhow::{anyhow, Result};

/// Game configuration for the actor
#[derive(Debug, Clone)]
pub struct GameConfig {
    /// Number of possible actions
    pub num_actions: usize,
    /// Size of observation vector (number of f32 values)
    pub obs_size: usize,
    /// Offset in observation where legal moves mask starts
    pub legal_mask_offset: usize,
}

impl GameConfig {
    /// Create a new GameConfig
    pub fn new(num_actions: usize, obs_size: usize, legal_mask_offset: usize) -> Self {
        Self {
            num_actions,
            obs_size,
            legal_mask_offset,
        }
    }

    /// Create a bitmask for extracting legal moves from info bits
    pub fn legal_mask_bits(&self) -> u64 {
        (1u64 << self.num_actions) - 1
    }
}

/// Get the game configuration for a given environment ID
///
/// # Arguments
///
/// * `env_id` - Environment identifier (e.g., "tictactoe", "connect4")
///
/// # Returns
///
/// Returns the GameConfig for the specified game, or an error if unknown.
pub fn get_config(env_id: &str) -> Result<GameConfig> {
    match env_id {
        "tictactoe" => Ok(GameConfig::new(9, 29, 18)),
        "connect4" => Ok(GameConfig::new(7, 100, 86)),  // Placeholder values
        "othello" => Ok(GameConfig::new(64, 195, 129)), // Placeholder values
        _ => Err(anyhow!("Unknown game: {}. Please add configuration for this game.", env_id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_config_tictactoe() {
        let config = get_config("tictactoe").unwrap();
        assert_eq!(config.num_actions, 9);
        assert_eq!(config.obs_size, 29);
        assert_eq!(config.legal_mask_offset, 18);
    }

    #[test]
    fn test_legal_mask_bits() {
        let config = get_config("tictactoe").unwrap();
        assert_eq!(config.legal_mask_bits(), 0x1FF); // 9 bits set

        let connect4 = get_config("connect4").unwrap();
        assert_eq!(connect4.legal_mask_bits(), 0x7F); // 7 bits set
    }

    #[test]
    fn test_unknown_game() {
        let result = get_config("unknown_game");
        assert!(result.is_err());
    }
}
