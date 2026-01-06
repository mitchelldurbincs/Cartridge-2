//! Game configuration registry for the actor
//!
//! This module provides game-specific configuration that the actor needs
//! for MCTS and observation parsing. The configuration is **automatically derived**
//! from the engine's GameMetadata, eliminating hardcoded values.

use anyhow::{anyhow, Result};
use engine_core::{EngineContext, GameMetadata};

/// Game configuration for the actor
///
/// This struct contains the configuration values needed by the actor
/// for MCTS policy and observation parsing. Values are derived from
/// the game's `GameMetadata`.
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
    /// Create a new GameConfig with explicit values
    ///
    /// Prefer using `from_metadata` or `from_context` to auto-derive values.
    /// This constructor is kept for testing and edge cases where metadata
    /// is not available.
    #[allow(dead_code)]
    pub fn new(num_actions: usize, obs_size: usize, legal_mask_offset: usize) -> Self {
        Self {
            num_actions,
            obs_size,
            legal_mask_offset,
        }
    }

    /// Create a GameConfig from GameMetadata
    ///
    /// This automatically extracts the required configuration values from
    /// the game's metadata, ensuring consistency with the game implementation.
    pub fn from_metadata(metadata: &GameMetadata) -> Self {
        Self {
            num_actions: metadata.num_actions,
            obs_size: metadata.obs_size,
            legal_mask_offset: metadata.legal_mask_offset,
        }
    }

    /// Create a GameConfig from an EngineContext
    ///
    /// This is a convenience method that extracts metadata from the context.
    pub fn from_context(ctx: &EngineContext) -> Self {
        Self::from_metadata(&ctx.metadata())
    }

    /// Create a bitmask for extracting legal moves from info bits
    pub fn legal_mask_bits(&self) -> u64 {
        (1u64 << self.num_actions) - 1
    }

    /// Extract a legal moves mask from the observation bytes.
    ///
    /// The observation contains f32 values where indices starting at `legal_mask_offset`
    /// are the legal moves (1.0 = legal, 0.0 = illegal). This function extracts
    /// those values and packs them into a u64 bitmask.
    pub fn extract_legal_mask(&self, obs: &[u8]) -> u64 {
        let legal_start_byte = self.legal_mask_offset * 4;
        let legal_end_byte = legal_start_byte + self.num_actions * 4;

        if obs.len() < legal_end_byte {
            tracing::warn!(
                "Observation too short ({} bytes), expected at least {}. Using fallback mask.",
                obs.len(),
                legal_end_byte
            );
            return self.legal_mask_bits();
        }

        let mut mask = 0u64;
        for i in 0..self.num_actions {
            let byte_offset = legal_start_byte + i * 4;
            let value = f32::from_le_bytes([
                obs[byte_offset],
                obs[byte_offset + 1],
                obs[byte_offset + 2],
                obs[byte_offset + 3],
            ]);
            if value > 0.5 {
                mask |= 1u64 << i;
            }
        }
        mask
    }
}

impl From<&GameMetadata> for GameConfig {
    fn from(metadata: &GameMetadata) -> Self {
        Self::from_metadata(metadata)
    }
}

impl From<GameMetadata> for GameConfig {
    fn from(metadata: GameMetadata) -> Self {
        Self::from_metadata(&metadata)
    }
}

/// Get the game configuration for a given environment ID
///
/// This function creates an `EngineContext` for the specified game and
/// derives the configuration from the game's metadata. This ensures
/// the configuration always matches the actual game implementation.
///
/// # Arguments
///
/// * `env_id` - Environment identifier (e.g., "tictactoe", "connect4")
///
/// # Returns
///
/// Returns the GameConfig for the specified game, or an error if the game
/// is not registered.
///
/// # Note
///
/// The game must be registered before calling this function. Call the
/// appropriate `register_*` function first (e.g., `register_tictactoe()`).
pub fn get_config(env_id: &str) -> Result<GameConfig> {
    let ctx = EngineContext::new(env_id).ok_or_else(|| {
        anyhow!(
            "Unknown game: '{}'. Ensure the game is registered before calling get_config.",
            env_id
        )
    })?;

    Ok(GameConfig::from_context(&ctx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    /// Mutex to serialize registry-dependent tests
    static REGISTRY_TEST_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn setup_games() {
        games_tictactoe::register_tictactoe();
        games_connect4::register_connect4();
    }

    #[test]
    fn test_get_config_tictactoe() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_games();

        let config = get_config("tictactoe").unwrap();
        // Values derived from GameMetadata in games-tictactoe
        // TicTacToe: 9 actions, obs = 18 (board) + 9 (legal) + 2 (player) = 29
        assert_eq!(config.num_actions, 9);
        assert_eq!(config.obs_size, 29);
        assert_eq!(config.legal_mask_offset, 18);
    }

    #[test]
    fn test_get_config_connect4() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_games();

        let config = get_config("connect4").unwrap();
        // Values derived from GameMetadata in games-connect4
        // Connect4: 7 actions, obs = 84 (board) + 7 (legal) + 2 (player) = 93
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
    fn test_from_metadata() {
        // Test that GameConfig::from_metadata works correctly
        let metadata = GameMetadata::new("test", "Test Game")
            .with_actions(10)
            .with_observation(50, 30);

        let config = GameConfig::from_metadata(&metadata);
        assert_eq!(config.num_actions, 10);
        assert_eq!(config.obs_size, 50);
        assert_eq!(config.legal_mask_offset, 30);
    }

    #[test]
    fn test_from_trait_impls() {
        // Test From<&GameMetadata>
        let metadata = GameMetadata::new("test", "Test Game")
            .with_actions(5)
            .with_observation(20, 10);

        let config: GameConfig = (&metadata).into();
        assert_eq!(config.num_actions, 5);

        // Test From<GameMetadata>
        let config2: GameConfig = metadata.into();
        assert_eq!(config2.num_actions, 5);
    }

    #[test]
    fn test_unknown_game() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        // Don't register games - test that unknown games fail
        let result = get_config("unknown_game");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_matches_game_metadata() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_games();

        // Verify that get_config returns values that match the game's metadata
        let ttt_ctx = EngineContext::new("tictactoe").unwrap();
        let ttt_meta = ttt_ctx.metadata();
        let ttt_config = get_config("tictactoe").unwrap();

        assert_eq!(ttt_config.num_actions, ttt_meta.num_actions);
        assert_eq!(ttt_config.obs_size, ttt_meta.obs_size);
        assert_eq!(ttt_config.legal_mask_offset, ttt_meta.legal_mask_offset);

        let c4_ctx = EngineContext::new("connect4").unwrap();
        let c4_meta = c4_ctx.metadata();
        let c4_config = get_config("connect4").unwrap();

        assert_eq!(c4_config.num_actions, c4_meta.num_actions);
        assert_eq!(c4_config.obs_size, c4_meta.obs_size);
        assert_eq!(c4_config.legal_mask_offset, c4_meta.legal_mask_offset);
    }
}
