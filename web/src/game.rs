//! Game session management for TicTacToe
//!
//! Wraps the EngineContext to provide a convenient API for the web server.

use anyhow::{anyhow, Result};
use engine_core::EngineContext;
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

use crate::GameStateResponse;
#[cfg(not(feature = "onnx"))]
use crate::OnnxEvaluator;

/// TicTacToe constants (obs_size used when loading ONNX models)
#[cfg(feature = "onnx")]
const TICTACTOE_OBS_SIZE: usize = 29;

/// A game session tracking current state
pub struct GameSession {
    ctx: EngineContext,
    /// Current encoded state
    state: Vec<u8>,
    /// Current observation
    obs: Vec<u8>,
    /// Decoded board for easy access
    board: [u8; 9],
    /// Current player (1=X, 2=O)
    current_player: u8,
    /// Winner (0=ongoing, 1=X, 2=O, 3=draw)
    winner: u8,
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

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let reset = ctx.reset(seed, &[])?;

        // Parse the state (11 bytes: 9 board + current_player + winner)
        let (board, current_player, winner) = Self::parse_state(&reset.state)?;

        // Configure MCTS for playing (less exploration than training)
        let mcts_config = MctsConfig::for_evaluation()
            .with_simulations(200) // More simulations for better play
            .with_temperature(0.5); // Some randomness but not too much

        // Pre-create simulation context for MCTS (avoids repeated registry lookups)
        let mcts_sim_ctx = EngineContext::new(env_id);

        Ok(Self {
            ctx,
            state: reset.state,
            obs: reset.obs,
            board,
            current_player,
            winner,
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

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let reset = ctx.reset(seed, &[])?;

        // Parse the state (11 bytes: 9 board + current_player + winner)
        let (board, current_player, winner) = Self::parse_state(&reset.state)?;

        Ok(Self {
            ctx,
            state: reset.state,
            obs: reset.obs,
            board,
            current_player,
            winner,
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

        let new_evaluator = OnnxEvaluator::load(path_ref, TICTACTOE_OBS_SIZE)
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
    fn parse_state(state: &[u8]) -> Result<([u8; 9], u8, u8)> {
        if state.len() != 11 {
            return Err(anyhow!(
                "Invalid state length: expected 11, got {}",
                state.len()
            ));
        }

        let mut board = [0u8; 9];
        board.copy_from_slice(&state[0..9]);
        let current_player = state[9];
        let winner = state[10];

        Ok((board, current_player, winner))
    }

    /// Get legal moves
    pub fn legal_moves(&self) -> Vec<u8> {
        if self.winner != 0 {
            return Vec::new();
        }

        (0..9u8)
            .filter(|&pos| self.board[pos as usize] == 0)
            .collect()
    }

    /// Check if a move is legal
    pub fn is_legal_move(&self, position: u8) -> bool {
        position < 9 && self.board[position as usize] == 0 && self.winner == 0
    }

    /// Check if game is over
    pub fn is_game_over(&self) -> bool {
        self.winner != 0
    }

    /// Get current player
    pub fn current_player(&self) -> u8 {
        self.current_player
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
            // Use MCTS with neural network
            debug!("Using MCTS for bot move");

            let guard = self
                .evaluator
                .read()
                .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
            let evaluator = guard.as_ref().unwrap();

            // Use pre-created simulation context (avoids repeated registry lookups)
            let sim_ctx = self
                .mcts_sim_ctx
                .as_mut()
                .ok_or_else(|| anyhow!("TicTacToe not registered"))?;

            let result = run_mcts(
                sim_ctx,
                evaluator,
                self.mcts_config.clone(),
                self.state.clone(),
                self.obs.clone(),
                legal_mask,
                &mut self.rng,
            )
            .map_err(|e| anyhow!("MCTS failed: {}", e))?;

            debug!(
                action = result.action,
                value = result.value,
                simulations = result.simulations,
                "MCTS selected move"
            );

            result.action as u8
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
        let (board, current_player, winner) = Self::parse_state(&self.state)?;
        self.board = board;
        self.current_player = current_player;
        self.winner = winner;

        Ok(())
    }

    /// Convert to API response
    pub fn to_response(&self) -> GameStateResponse {
        let message = match self.winner {
            0 => {
                if self.current_player == 1 {
                    "Your turn (X)".to_string()
                } else {
                    "Bot's turn (O)".to_string()
                }
            }
            1 => "You win!".to_string(),
            2 => "Bot wins!".to_string(),
            3 => "It's a draw!".to_string(),
            _ => "Unknown state".to_string(),
        };

        GameStateResponse {
            board: self.board,
            current_player: self.current_player,
            winner: self.winner,
            game_over: self.is_game_over(),
            legal_moves: self.legal_moves(),
            message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game_session() {
        games_tictactoe::register_tictactoe();

        let session = GameSession::new("tictactoe").unwrap();

        assert_eq!(session.board, [0; 9]);
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
}
