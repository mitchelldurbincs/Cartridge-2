//! MCTS-based policy using neural network evaluation
//!
//! This module provides a policy that uses Monte Carlo Tree Search with
//! an ONNX neural network evaluator to select actions.

use anyhow::{anyhow, Result};
use engine_core::EngineContext;
use mcts::{run_mcts, MctsConfig, OnnxEvaluator, SearchResult};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tracing::debug;

/// Result from MCTS policy selection, including training data
pub struct MctsPolicyResult {
    /// Selected action as bytes
    pub action: Vec<u8>,
    /// Policy distribution from MCTS (for training)
    pub policy: Vec<f32>,
    /// Value estimate from MCTS root
    pub value: f32,
}

/// MCTS-based policy that uses neural network for evaluation
pub struct MctsPolicy {
    /// Environment ID for creating simulation contexts
    env_id: String,
    /// MCTS configuration
    config: MctsConfig,
    /// Base temperature (used for early moves)
    base_temperature: f32,
    /// Temperature for late-game moves (after threshold)
    late_temperature: f32,
    /// Move number after which to use late_temperature
    temp_threshold: u32,
    /// Number of actions in the game
    num_actions: usize,
    /// Observation size for the neural network
    obs_size: usize,
    /// Shared evaluator that can be hot-swapped
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// RNG for action sampling
    rng: ChaCha20Rng,
    /// Reusable simulation context for MCTS (avoids repeated registry lookups)
    sim_ctx: Option<EngineContext>,
}

impl std::fmt::Debug for MctsPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MctsPolicy")
            .field("env_id", &self.env_id)
            .field("num_actions", &self.num_actions)
            .field("obs_size", &self.obs_size)
            .field("temp_threshold", &self.temp_threshold)
            .field(
                "has_model",
                &self.evaluator.read().map(|e| e.is_some()).unwrap_or(false),
            )
            .finish()
    }
}

impl MctsPolicy {
    /// Create a new MCTS policy without a model loaded
    pub fn new(env_id: String, num_actions: usize, obs_size: usize) -> Self {
        // Pre-create the simulation context to avoid repeated registry lookups
        let sim_ctx = EngineContext::new(&env_id);
        let config = MctsConfig::for_training();
        let base_temp = config.temperature;
        Self {
            env_id,
            base_temperature: base_temp,
            late_temperature: 0.1, // More deterministic in late game
            temp_threshold: 0,     // Disabled by default (0 = no threshold)
            config,
            num_actions,
            obs_size,
            evaluator: Arc::new(RwLock::new(None)),
            rng: ChaCha20Rng::from_entropy(),
            sim_ctx,
        }
    }

    /// Create with a specific seed for determinism (used in tests)
    #[allow(dead_code)]
    pub fn with_seed(env_id: String, num_actions: usize, obs_size: usize, seed: u64) -> Self {
        let sim_ctx = EngineContext::new(&env_id);
        let config = MctsConfig::for_training();
        let base_temp = config.temperature;
        Self {
            env_id,
            base_temperature: base_temp,
            late_temperature: 0.1,
            temp_threshold: 0,
            config,
            num_actions,
            obs_size,
            evaluator: Arc::new(RwLock::new(None)),
            rng: ChaCha20Rng::seed_from_u64(seed),
            sim_ctx,
        }
    }

    /// Set the MCTS configuration
    pub fn with_config(mut self, config: MctsConfig) -> Self {
        self.base_temperature = config.temperature;
        self.config = config;
        self
    }

    /// Set the temperature schedule for move-dependent exploration
    ///
    /// After `threshold` moves, temperature drops from base to `late_temp`.
    /// This encourages exploration early and exploitation late in games.
    ///
    /// Set threshold to 0 to disable (always use base temperature).
    pub fn with_temp_schedule(mut self, threshold: u32, late_temp: f32) -> Self {
        self.temp_threshold = threshold;
        self.late_temperature = late_temp;
        self
    }

    /// Update MCTS simulation count (for ramping during training)
    #[allow(dead_code)]
    pub fn set_simulations(&mut self, num_simulations: u32) {
        self.config.num_simulations = num_simulations;
    }

    /// Check if a model is loaded (used for debugging/logging)
    #[allow(dead_code)]
    pub fn has_model(&self) -> bool {
        self.evaluator
            .read()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Get shared evaluator reference for hot-reloading
    pub fn evaluator_ref(&self) -> Arc<RwLock<Option<OnnxEvaluator>>> {
        Arc::clone(&self.evaluator)
    }

    /// Select an action using MCTS
    ///
    /// # Arguments
    /// * `state` - Current game state bytes
    /// * `obs` - Current observation bytes
    /// * `legal_moves_mask` - Bit mask of legal actions
    /// * `move_number` - Current move number in the game (0-indexed)
    ///
    /// # Returns
    /// `MctsPolicyResult` with action, policy distribution, and value estimate
    pub fn select_action(
        &mut self,
        state: &[u8],
        obs: &[u8],
        legal_moves_mask: u64,
        move_number: u32,
    ) -> Result<MctsPolicyResult> {
        // Acquire read lock once and hold it through the operation to avoid TOCTOU race.
        // This prevents the model from being swapped out between checking and using it.
        let guard = self
            .evaluator
            .read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        let evaluator = match guard.as_ref() {
            Some(eval) => eval,
            None => {
                // No model loaded - fall back to random legal action
                // Note: This is expected during early training before first model is exported
                // Drop the lock before calling random_action since it needs &mut self
                drop(guard);
                debug!("No model loaded, using random policy");
                return self.random_action(legal_moves_mask);
            }
        };

        // Use the pre-created simulation context (avoids repeated registry lookups)
        let sim_ctx = self
            .sim_ctx
            .as_mut()
            .ok_or_else(|| anyhow!("Game '{}' not registered", self.env_id))?;

        // Apply temperature schedule: use lower temperature for late-game moves
        let mut config = self.config.clone();
        if self.temp_threshold > 0 && move_number >= self.temp_threshold {
            config.temperature = self.late_temperature;
        }

        // Run MCTS search with timing
        let mcts_start = Instant::now();
        let result: SearchResult = run_mcts(
            sim_ctx,
            evaluator,
            config,
            state.to_vec(),
            obs.to_vec(),
            legal_moves_mask,
            &mut self.rng,
        )
        .map_err(|e| anyhow!("MCTS search failed: {}", e))?;
        let mcts_elapsed_ms = mcts_start.elapsed().as_millis();

        debug!(
            action = result.action,
            value = result.value,
            simulations = result.simulations,
            move_number = move_number,
            mcts_ms = mcts_elapsed_ms,
            "MCTS selected action"
        );

        // Convert action index to bytes (u32 little-endian)
        let action_bytes = (result.action as u32).to_le_bytes().to_vec();

        Ok(MctsPolicyResult {
            action: action_bytes,
            policy: result.policy,
            value: result.value,
        })
    }

    /// Fall back to random action selection when no model is available
    fn random_action(&mut self, legal_moves_mask: u64) -> Result<MctsPolicyResult> {
        use rand::Rng;

        if legal_moves_mask == 0 {
            return Err(anyhow!("No legal moves available"));
        }

        // Count legal moves and build uniform policy
        let num_legal = legal_moves_mask.count_ones() as f32;
        let mut policy = vec![0.0; self.num_actions];
        let mut legal_actions = Vec::new();

        for (i, p) in policy.iter_mut().enumerate().take(self.num_actions) {
            if (legal_moves_mask >> i) & 1 == 1 {
                *p = 1.0 / num_legal;
                legal_actions.push(i);
            }
        }

        // Sample random legal action
        let idx = self.rng.gen_range(0..legal_actions.len());
        let action = legal_actions[idx];
        let action_bytes = (action as u32).to_le_bytes().to_vec();

        Ok(MctsPolicyResult {
            action: action_bytes,
            policy,
            value: 0.0, // No value estimate without model
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        engine_games::register_all_games();
    }

    #[test]
    fn test_mcts_policy_creation() {
        let policy = MctsPolicy::new("tictactoe".into(), 9, 29);
        assert!(!policy.has_model());
        assert_eq!(policy.num_actions, 9);
    }

    #[test]
    fn test_mcts_policy_without_model_uses_random() {
        setup();

        let mut policy = MctsPolicy::with_seed("tictactoe".into(), 9, 29, 42);

        // Create a dummy state/obs - in practice these come from the game
        let state = vec![0u8; 10]; // TicTacToe state is about 10 bytes
        let obs = vec![0u8; 29 * 4]; // 29 floats = 116 bytes
        let legal_mask = 0b111111111u64; // All 9 positions legal

        // Without a model, should return random action
        let result = policy.select_action(&state, &obs, legal_mask, 0);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.action.len(), 4); // u32

        // Action should be in valid range
        let action = u32::from_le_bytes(result.action.try_into().unwrap());
        assert!(action < 9);

        // Policy should be uniform
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_random_action_with_partial_legal_mask() {
        let mut policy = MctsPolicy::with_seed("tictactoe".into(), 9, 29, 42);

        // Only positions 0, 2, 4 are legal
        let legal_mask = 0b000010101u64;

        // Run multiple times to verify only legal actions selected
        for _ in 0..20 {
            let result = policy.random_action(legal_mask).unwrap();
            let action = u32::from_le_bytes(result.action.try_into().unwrap());
            assert!(
                action == 0 || action == 2 || action == 4,
                "Action {} should be 0, 2, or 4",
                action
            );

            // Illegal actions should have 0 probability
            assert_eq!(result.policy[1], 0.0);
            assert_eq!(result.policy[3], 0.0);
        }
    }

    #[test]
    fn test_random_action_no_legal_moves_fails() {
        let mut policy = MctsPolicy::new("tictactoe".into(), 9, 29);
        let result = policy.random_action(0);
        assert!(result.is_err());
    }
}
