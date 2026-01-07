//! Game configuration for the actor
//!
//! This module re-exports `GameMetadata` from engine-core and provides
//! a convenience function to get game configuration by environment ID.
//!
//! The `GameConfig` type alias and `get_config` function provide backward
//! compatibility while using the shared implementation from engine-core.

use anyhow::{anyhow, Result};
use engine_core::{EngineContext, GameMetadata};

/// Game configuration for the actor.
///
/// This is a type alias for `GameMetadata` from engine-core, which contains
/// all configuration values needed by the actor for MCTS and observation parsing.
pub type GameConfig = GameMetadata;

/// Get the game configuration for a given environment ID.
///
/// This function creates an `EngineContext` for the specified game and
/// returns its metadata. This ensures the configuration always matches
/// the actual game implementation.
///
/// # Arguments
///
/// * `env_id` - Environment identifier (e.g., "tictactoe", "connect4")
///
/// # Returns
///
/// Returns the GameConfig (GameMetadata) for the specified game, or an error
/// if the game is not registered.
///
/// # Note
///
/// The game must be registered before calling this function. Call the
/// appropriate `register_*` function first (e.g., via `engine_games::register_all_games()`).
pub fn get_config(env_id: &str) -> Result<GameConfig> {
    let ctx = EngineContext::new(env_id).ok_or_else(|| {
        anyhow!(
            "Unknown game: '{}'. Ensure the game is registered before calling get_config.",
            env_id
        )
    })?;

    Ok(ctx.metadata())
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    /// Mutex to serialize registry-dependent tests
    static REGISTRY_TEST_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn setup_games() {
        engine_games::register_all_games();
    }

    #[test]
    fn test_get_config_tictactoe() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_games();

        let config = get_config("tictactoe").unwrap();
        assert_eq!(config.num_actions, 9);
        assert_eq!(config.obs_size, 29);
        assert_eq!(config.legal_mask_offset, 18);
    }

    #[test]
    fn test_get_config_connect4() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_games();

        let config = get_config("connect4").unwrap();
        assert_eq!(config.num_actions, 7);
        assert_eq!(config.obs_size, 93);
        assert_eq!(config.legal_mask_offset, 84);
    }

    #[test]
    fn test_legal_mask_bits() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_games();

        let config = get_config("tictactoe").unwrap();
        assert_eq!(config.legal_mask_bits(), 0x1FF); // 9 bits set

        let connect4 = get_config("connect4").unwrap();
        assert_eq!(connect4.legal_mask_bits(), 0x7F); // 7 bits set
    }

    #[test]
    fn test_unknown_game() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        let result = get_config("unknown_game");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_legal_mask() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_games();

        let config = get_config("tictactoe").unwrap();

        // Create observation with all moves legal
        let mut obs = vec![0u8; 29 * 4];
        for i in 0..9 {
            let byte_offset = (18 + i) * 4;
            let bytes = 1.0f32.to_le_bytes();
            obs[byte_offset..byte_offset + 4].copy_from_slice(&bytes);
        }

        let mask = config.extract_legal_mask(&obs);
        assert_eq!(mask, 0x1FF);
    }
}
