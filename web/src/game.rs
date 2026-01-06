//! Game session management
//!
//! Wraps the EngineContext to provide a convenient API for the web server.

use anyhow::{anyhow, Result};
use engine_core::{EngineContext, GameMetadata};
#[cfg(feature = "onnx")]
use mcts::{run_mcts, MctsConfig, OnnxEvaluator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "onnx")]
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(feature = "onnx")]
use tracing::{debug, info};

use crate::types::GameStateResponse;
#[cfg(not(feature = "onnx"))]
use crate::OnnxEvaluator;

// =============================================================================
// Configuration Constants
// =============================================================================

/// Number of MCTS simulations for web play (less than training, more than eval)
#[cfg(feature = "onnx")]
const MCTS_SIMULATIONS: u32 = 200;

/// MCTS temperature for web play (some randomness but not too much)
#[cfg(feature = "onnx")]
const MCTS_TEMPERATURE: f32 = 0.5;

/// Threshold for determining if a move is legal from observation floats
const LEGAL_MOVE_THRESHOLD: f32 = 0.5;

/// Default human player number (1 = goes first)
const DEFAULT_HUMAN_PLAYER: u8 = 1;

// =============================================================================
// Game Session
// =============================================================================

/// A game session tracking current state
pub struct GameSession {
    ctx: EngineContext,
    /// Game metadata (board size, num_actions, etc.)
    metadata: GameMetadata,
    /// Current encoded state
    state: Vec<u8>,
    /// Current observation
    obs: Vec<u8>,
    /// Decoded board for easy access (length = board_width * board_height)
    board: Vec<u8>,
    /// Current player (1=X, 2=O)
    current_player: u8,
    /// Winner (0=ongoing, 1=X, 2=O, 3=draw)
    winner: u8,
    /// Which player the human is (1 or 2). Set when game starts based on who goes first.
    human_player: u8,
    /// RNG for bot moves
    rng: ChaCha20Rng,
    /// Shared evaluator for MCTS (loaded from model file)
    #[cfg(feature = "onnx")]
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// Stub evaluator when ONNX is disabled
    #[cfg(not(feature = "onnx"))]
    #[allow(dead_code)]
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// MCTS configuration for bot play
    #[cfg(feature = "onnx")]
    mcts_config: MctsConfig,
    /// Reusable simulation context for MCTS (avoids repeated registry lookups)
    /// Separate from `ctx` because MCTS needs its own context for simulations
    #[cfg(feature = "onnx")]
    mcts_sim_ctx: Option<EngineContext>,
}

impl GameSession {
    /// Create a new game session with default (empty) evaluator.
    /// Used in tests; production code uses `with_evaluator` for hot-reloading.
    #[cfg(test)]
    pub fn new(env_id: &str) -> Result<Self> {
        Self::with_evaluator(env_id, Arc::new(RwLock::new(None)))
    }

    /// Create a new game session with a shared evaluator (for hot-reloading)
    #[cfg(feature = "onnx")]
    pub fn with_evaluator(
        env_id: &str,
        evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    ) -> Result<Self> {
        let mut ctx = EngineContext::new(env_id)
            .ok_or_else(|| anyhow!("Game '{}' not registered", env_id))?;

        // Get game metadata
        let metadata = ctx.metadata();
        let board_size = metadata.board_size();

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let reset = ctx.reset(seed, &[])?;

        // Parse the state (board_size + current_player + winner bytes)
        let (board, current_player, winner) = Self::parse_state(&reset.state, board_size)?;

        // Configure MCTS for playing (less exploration than training)
        let mcts_config = MctsConfig::for_evaluation()
            .with_simulations(MCTS_SIMULATIONS)
            .with_temperature(MCTS_TEMPERATURE);

        // Pre-create simulation context for MCTS (avoids repeated registry lookups)
        let mcts_sim_ctx = EngineContext::new(env_id);

        Ok(Self {
            ctx,
            metadata,
            state: reset.state,
            obs: reset.obs,
            board,
            current_player,
            winner,
            human_player: DEFAULT_HUMAN_PLAYER,
            rng: ChaCha20Rng::seed_from_u64(seed),
            evaluator,
            mcts_config,
            mcts_sim_ctx,
        })
    }

    /// Create a new game session with a shared evaluator (stub when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn with_evaluator(
        env_id: &str,
        evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    ) -> Result<Self> {
        let mut ctx = EngineContext::new(env_id)
            .ok_or_else(|| anyhow!("Game '{}' not registered", env_id))?;

        // Get game metadata
        let metadata = ctx.metadata();
        let board_size = metadata.board_size();

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let reset = ctx.reset(seed, &[])?;

        // Parse the state (board_size + current_player + winner bytes)
        let (board, current_player, winner) = Self::parse_state(&reset.state, board_size)?;

        Ok(Self {
            ctx,
            metadata,
            state: reset.state,
            obs: reset.obs,
            board,
            current_player,
            winner,
            human_player: DEFAULT_HUMAN_PLAYER,
            rng: ChaCha20Rng::seed_from_u64(seed),
            evaluator,
        })
    }

    /// Load an ONNX model for the bot (for future model hot-reloading)
    #[cfg(feature = "onnx")]
    #[allow(dead_code)]
    pub fn load_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        info!("Loading ONNX model from {:?}", path_ref);

        let new_evaluator = OnnxEvaluator::load(path_ref, self.metadata.obs_size)
            .map_err(|e| anyhow!("Failed to load model: {}", e))?;

        let mut guard = self
            .evaluator
            .write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        *guard = Some(new_evaluator);

        info!("Model loaded successfully");
        Ok(())
    }

    /// Check if a model is loaded (for future use)
    #[cfg(feature = "onnx")]
    #[allow(dead_code)]
    pub fn has_model(&self) -> bool {
        self.evaluator
            .read()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Get the shared evaluator reference (for future model hot-reloading)
    #[cfg(feature = "onnx")]
    #[allow(dead_code)]
    pub fn evaluator_ref(&self) -> Arc<RwLock<Option<OnnxEvaluator>>> {
        Arc::clone(&self.evaluator)
    }

    /// Parse state bytes into board, current_player, winner
    fn parse_state(state: &[u8], board_size: usize) -> Result<(Vec<u8>, u8, u8)> {
        let expected_len = board_size + 2; // board + current_player + winner
        if state.len() != expected_len {
            return Err(anyhow!(
                "Invalid state length: expected {}, got {}",
                expected_len,
                state.len()
            ));
        }

        let board = state[0..board_size].to_vec();
        let current_player = state[board_size];
        let winner = state[board_size + 1];

        Ok((board, current_player, winner))
    }

    /// Get legal moves by extracting from observation using metadata
    pub fn legal_moves(&self) -> Vec<u8> {
        if self.winner != 0 {
            return Vec::new();
        }

        let Some(mask_bytes) = self.legal_mask_bytes() else {
            return Vec::new();
        };

        mask_bytes
            .chunks_exact(4)
            .enumerate()
            .filter_map(|(action, bytes)| {
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                (value > LEGAL_MOVE_THRESHOLD).then_some(action as u8)
            })
            .collect()
    }

    /// Check if a move is legal by extracting from observation using metadata
    pub fn is_legal_move(&self, position: u8) -> bool {
        if self.winner != 0 {
            return false;
        }

        let action = position as usize;
        if action >= self.metadata.num_actions {
            return false;
        }

        let Some(mask_bytes) = self.legal_mask_bytes() else {
            return false;
        };

        let start = action * 4;
        let bytes: [u8; 4] = mask_bytes[start..start + 4].try_into().unwrap();
        let value = f32::from_le_bytes(bytes);
        value > LEGAL_MOVE_THRESHOLD
    }

    /// Extract the legal moves slice from the observation buffer
    fn legal_mask_bytes(&self) -> Option<&[u8]> {
        let start = self.metadata.legal_mask_offset.checked_mul(4)?;
        let len = self.metadata.num_actions.checked_mul(4)?;
        let end = start.checked_add(len)?;

        if end <= self.obs.len() {
            Some(&self.obs[start..end])
        } else {
            None
        }
    }

    /// Check if game is over
    pub fn is_game_over(&self) -> bool {
        self.winner != 0
    }

    /// Get current player
    #[allow(dead_code)]
    pub fn current_player(&self) -> u8 {
        self.current_player
    }

    /// Get which player the human is (1 or 2)
    #[allow(dead_code)]
    pub fn human_player(&self) -> u8 {
        self.human_player
    }

    /// Set which player the human is (called when game starts)
    pub fn set_human_player(&mut self, player: u8) {
        self.human_player = player;
    }

    /// Check if it's the human's turn
    pub fn is_human_turn(&self) -> bool {
        self.current_player == self.human_player
    }

    /// Make a player move
    pub fn player_move(&mut self, position: u8) -> Result<()> {
        self.make_move(position)
    }

    /// Make a bot move using MCTS if model is available, otherwise random
    #[cfg(feature = "onnx")]
    pub fn bot_move(&mut self) -> Result<u8> {
        let legal = self.legal_moves();
        if legal.is_empty() {
            return Err(anyhow!("No legal moves available"));
        }

        // Build legal moves mask
        let legal_mask: u64 = legal.iter().fold(0u64, |acc, &pos| acc | (1u64 << pos));

        // Check if we have a model
        let has_model = {
            let guard = self
                .evaluator
                .read()
                .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
            guard.is_some()
        };

        let position = if has_model {
            // Try to use MCTS with neural network
            debug!("Attempting MCTS for bot move");

            let mcts_result = (|| -> Result<u8> {
                let guard = self
                    .evaluator
                    .read()
                    .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
                let evaluator = guard.as_ref().unwrap();

                // Use pre-created simulation context (avoids repeated registry lookups)
                let sim_ctx = self
                    .mcts_sim_ctx
                    .as_mut()
                    .ok_or_else(|| anyhow!("Simulation context not available"))?;

                let result = run_mcts(
                    sim_ctx,
                    evaluator,
                    self.mcts_config.clone(),
                    self.state.clone(),
                    self.obs.clone(),
                    legal_mask,
                    &mut self.rng,
                )?;

                debug!(
                    action = result.action,
                    value = result.value,
                    simulations = result.simulations,
                    "MCTS selected move"
                );

                Ok(result.action as u8)
            })();

            match mcts_result {
                Ok(action) => action,
                Err(e) => {
                    // MCTS failed (e.g., model incompatible with current game)
                    // Fall back to random move
                    debug!("MCTS failed ({}), falling back to random move", e);
                    use rand::seq::SliceRandom;
                    *legal.choose(&mut self.rng).unwrap()
                }
            }
        } else {
            // Fall back to random move
            debug!("No model loaded, using random move");
            use rand::seq::SliceRandom;
            *legal.choose(&mut self.rng).unwrap()
        };

        self.make_move(position)?;
        Ok(position)
    }

    /// Make a bot move using random selection (when ONNX is disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn bot_move(&mut self) -> Result<u8> {
        let legal = self.legal_moves();
        if legal.is_empty() {
            return Err(anyhow!("No legal moves available"));
        }

        // Random move selection
        use rand::seq::SliceRandom;
        let position = *legal.choose(&mut self.rng).unwrap();

        self.make_move(position)?;
        Ok(position)
    }

    /// Internal move execution
    fn make_move(&mut self, position: u8) -> Result<()> {
        // Encode action as u32 little-endian
        let action = (position as u32).to_le_bytes().to_vec();

        let step = self.ctx.step(&self.state, &action)?;

        // Update state and observation
        self.state = step.state;
        self.obs = step.obs;
        let board_size = self.metadata.board_size();
        let (board, current_player, winner) = Self::parse_state(&self.state, board_size)?;
        self.board = board;
        self.current_player = current_player;
        self.winner = winner;

        Ok(())
    }

    /// Convert to API response
    pub fn to_response(&self) -> GameStateResponse {
        let human_symbol = self.metadata.player_symbols[(self.human_player - 1) as usize];
        let bot_symbol = self.metadata.player_symbols[(2 - self.human_player) as usize];

        let message = match self.winner {
            0 => {
                if self.is_human_turn() {
                    format!("Your turn ({})", human_symbol)
                } else {
                    format!("Bot's turn ({})", bot_symbol)
                }
            }
            w if w == self.human_player => "You win!".to_string(),
            3 => "It's a draw!".to_string(),
            _ => "Bot wins!".to_string(),
        };

        GameStateResponse {
            board: self.board.clone(),
            current_player: self.current_player,
            human_player: self.human_player,
            winner: self.winner,
            game_over: self.is_game_over(),
            legal_moves: self.legal_moves(),
            message,
        }
    }

    /// Get game metadata
    #[allow(dead_code)]
    pub fn metadata(&self) -> &GameMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game_session() {
        games_tictactoe::register_tictactoe();

        let session = GameSession::new("tictactoe").unwrap();

        assert_eq!(session.board, vec![0u8; 9]);
        assert_eq!(session.current_player, 1);
        assert_eq!(session.winner, 0);
        assert_eq!(session.legal_moves().len(), 9);
    }

    #[test]
    fn test_player_move() {
        games_tictactoe::register_tictactoe();

        let mut session = GameSession::new("tictactoe").unwrap();
        session.player_move(4).unwrap(); // Center

        assert_eq!(session.board[4], 1); // X placed
        assert_eq!(session.current_player, 2); // Now O's turn
        assert!(!session.legal_moves().contains(&4));
    }

    #[test]
    fn test_bot_move() {
        games_tictactoe::register_tictactoe();

        let mut session = GameSession::new("tictactoe").unwrap();
        session.player_move(4).unwrap();

        let bot_pos = session.bot_move().unwrap();

        assert!(bot_pos < 9);
        assert_ne!(bot_pos, 4);
        assert_eq!(session.board[bot_pos as usize], 2); // O placed
        assert_eq!(session.current_player, 1); // Back to X
    }

    #[test]
    fn test_illegal_move() {
        games_tictactoe::register_tictactoe();

        let mut session = GameSession::new("tictactoe").unwrap();
        session.player_move(4).unwrap();

        // Position 4 is now occupied
        assert!(!session.is_legal_move(4));
    }

    #[test]
    fn test_legal_moves_handles_short_obs() {
        games_tictactoe::register_tictactoe();

        let mut session = GameSession::new("tictactoe").unwrap();
        // Corrupt the observation buffer to simulate a mismatch with metadata
        session.obs.truncate(4);

        assert_eq!(session.legal_moves(), Vec::<u8>::new());
        assert!(!session.is_legal_move(0));
    }
}
