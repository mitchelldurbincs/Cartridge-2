//! Engine context providing a high-level API for game simulation
//!
//! This module provides `EngineContext`, a convenient wrapper for running
//! game simulations without dealing with raw bytes and registry lookups.

use crate::erased::{ErasedGame, ErasedGameError};
use crate::registry::create_game;
use crate::typed::{ActionSpace, Capabilities, EngineId};

/// High-level context for running game simulations
///
/// `EngineContext` wraps an `ErasedGame` instance and provides convenient
/// methods for reset/step operations while managing internal buffers.
///
/// # Example
///
/// ```rust,ignore
/// use engine_core::context::EngineContext;
///
/// // Create context for TicTacToe
/// let mut ctx = EngineContext::new("tictactoe").expect("game not found");
///
/// // Reset and get initial state
/// let (state, obs) = ctx.reset(42, &[]).unwrap();
///
/// // Take a step with an action
/// let action = 4u32.to_le_bytes().to_vec(); // Center position
/// let result = ctx.step(&state, &action).unwrap();
/// println!("Reward: {}, Done: {}", result.reward, result.done);
/// ```
#[derive(Debug)]
pub struct EngineContext {
    game: Box<dyn ErasedGame>,
    state_buf: Vec<u8>,
    obs_buf: Vec<u8>,
}

/// Result of a step operation
#[derive(Debug, Clone)]
pub struct StepResult {
    /// New state after the action
    pub state: Vec<u8>,
    /// Observation for the new state
    pub obs: Vec<u8>,
    /// Reward received from this step
    pub reward: f32,
    /// Whether the episode has terminated
    pub done: bool,
    /// Additional packed info bits
    pub info: u64,
}

/// Result of a reset operation
#[derive(Debug, Clone)]
pub struct ResetResult {
    /// Initial state
    pub state: Vec<u8>,
    /// Initial observation
    pub obs: Vec<u8>,
}

impl EngineContext {
    /// Create a new engine context for the specified game
    ///
    /// # Arguments
    ///
    /// * `env_id` - Environment identifier (e.g., "tictactoe")
    ///
    /// # Returns
    ///
    /// Returns `Some(context)` if the game is registered, `None` otherwise.
    pub fn new(env_id: &str) -> Option<Self> {
        let game = create_game(env_id)?;
        Some(Self {
            game,
            state_buf: Vec::with_capacity(256),
            obs_buf: Vec::with_capacity(512),
        })
    }

    /// Create a new engine context from an existing game instance
    pub fn from_game(game: Box<dyn ErasedGame>) -> Self {
        Self {
            game,
            state_buf: Vec::with_capacity(256),
            obs_buf: Vec::with_capacity(512),
        }
    }

    /// Get the engine ID for this game
    pub fn engine_id(&self) -> EngineId {
        self.game.engine_id()
    }

    /// Get the capabilities of this game
    pub fn capabilities(&self) -> Capabilities {
        self.game.capabilities()
    }

    /// Get the action space for this game
    pub fn action_space(&self) -> ActionSpace {
        self.game.capabilities().action_space
    }

    /// Reset the game to initial state
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for deterministic reset
    /// * `hint` - Optional hint data for environment setup
    ///
    /// # Returns
    ///
    /// Returns `ResetResult` containing initial state and observation.
    pub fn reset(&mut self, seed: u64, hint: &[u8]) -> Result<ResetResult, ErasedGameError> {
        self.state_buf.clear();
        self.obs_buf.clear();

        self.game
            .reset(seed, hint, &mut self.state_buf, &mut self.obs_buf)?;

        Ok(ResetResult {
            state: self.state_buf.clone(),
            obs: self.obs_buf.clone(),
        })
    }

    /// Perform one simulation step
    ///
    /// # Arguments
    ///
    /// * `state` - Current state encoded as bytes
    /// * `action` - Action to take encoded as bytes
    ///
    /// # Returns
    ///
    /// Returns `StepResult` containing new state, observation, reward, done flag, and info.
    pub fn step(&mut self, state: &[u8], action: &[u8]) -> Result<StepResult, ErasedGameError> {
        self.state_buf.clear();
        self.obs_buf.clear();

        let (reward, done, info) =
            self.game
                .step(state, action, &mut self.state_buf, &mut self.obs_buf)?;

        Ok(StepResult {
            state: self.state_buf.clone(),
            obs: self.obs_buf.clone(),
            reward,
            done,
            info,
        })
    }

    /// Sample a random action from the action space
    ///
    /// Uses the provided RNG to sample a valid action for this game's action space.
    pub fn sample_random_action(&self, rng: &mut rand_chacha::ChaCha20Rng) -> Vec<u8> {
        use rand::Rng;

        let mut out = Vec::new();
        let caps = self.game.capabilities();

        match &caps.action_space {
            ActionSpace::Discrete(n) => {
                let action = rng.gen_range(0..*n);
                out.extend_from_slice(&action.to_le_bytes());
            }
            ActionSpace::MultiDiscrete(nvec) => {
                for &n in nvec {
                    let action = rng.gen_range(0..n);
                    out.extend_from_slice(&action.to_le_bytes());
                }
            }
            ActionSpace::Continuous { low, high, .. } => {
                for i in 0..low.len() {
                    let value = rng.gen_range(low[i]..high[i]);
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
        }

        out
    }

    /// Get access to the underlying erased game
    pub fn game(&self) -> &dyn ErasedGame {
        self.game.as_ref()
    }

    /// Get mutable access to the underlying erased game
    pub fn game_mut(&mut self) -> &mut dyn ErasedGame {
        self.game.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::GameAdapter;
    use crate::registry::{clear_registry, register_game};
    use crate::test_utils::REGISTRY_TEST_MUTEX;
    use crate::typed::{DecodeError, EncodeError, Encoding, Game};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[derive(Debug, Default)]
    struct SimpleGame;

    impl Game for SimpleGame {
        type State = u32;
        type Action = u32; // Changed from u8 to match sample_random_action's u32 output
        type Obs = f32;

        fn engine_id(&self) -> EngineId {
            EngineId {
                env_id: "simple".into(),
                build_id: "test".into(),
            }
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities {
                id: self.engine_id(),
                encoding: Encoding {
                    state: "u32".into(),
                    action: "u32".into(),
                    obs: "f32".into(),
                    schema_version: 1,
                },
                max_horizon: 10,
                action_space: ActionSpace::Discrete(4),
                preferred_batch: 1,
            }
        }

        fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
            (0, 0.0)
        }

        fn step(
            &mut self,
            state: &mut Self::State,
            action: Self::Action,
            _rng: &mut ChaCha20Rng,
        ) -> (Self::Obs, f32, bool, u64) {
            *state += action;
            let done = *state >= 10;
            (*state as f32, 1.0, done, *state as u64)
        }

        fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
            out.extend_from_slice(&state.to_le_bytes());
            Ok(())
        }

        fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
            if buf.len() != 4 {
                return Err(DecodeError::InvalidLength {
                    expected: 4,
                    actual: buf.len(),
                });
            }
            Ok(u32::from_le_bytes(buf.try_into().unwrap()))
        }

        fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
            out.extend_from_slice(&action.to_le_bytes());
            Ok(())
        }

        fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
            if buf.len() != 4 {
                return Err(DecodeError::InvalidLength {
                    expected: 4,
                    actual: buf.len(),
                });
            }
            Ok(u32::from_le_bytes(buf.try_into().unwrap()))
        }

        fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
            out.extend_from_slice(&obs.to_le_bytes());
            Ok(())
        }
    }

    fn setup_registry() {
        clear_registry();
        register_game("simple".to_string(), || {
            Box::new(GameAdapter::new(SimpleGame))
        });
    }

    #[test]
    fn test_context_creation() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let ctx = EngineContext::new("simple");
        assert!(ctx.is_some());

        let ctx = ctx.unwrap();
        assert_eq!(ctx.engine_id().env_id, "simple");
    }

    #[test]
    fn test_context_nonexistent_game() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let ctx = EngineContext::new("nonexistent");
        assert!(ctx.is_none());
    }

    #[test]
    fn test_context_reset() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let mut ctx = EngineContext::new("simple").unwrap();
        let result = ctx.reset(42, &[]).unwrap();

        assert_eq!(result.state.len(), 4);
        assert_eq!(result.obs.len(), 4);

        let state = u32::from_le_bytes(result.state.try_into().unwrap());
        assert_eq!(state, 0);
    }

    #[test]
    fn test_context_step() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let mut ctx = EngineContext::new("simple").unwrap();
        let reset_result = ctx.reset(42, &[]).unwrap();

        let action = 3u32.to_le_bytes().to_vec();
        let step_result = ctx.step(&reset_result.state, &action).unwrap();

        assert_eq!(step_result.reward, 1.0);
        assert!(!step_result.done);

        let new_state = u32::from_le_bytes(step_result.state.try_into().unwrap());
        assert_eq!(new_state, 3);
    }

    #[test]
    fn test_context_sample_random_action() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let ctx = EngineContext::new("simple").unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let action = ctx.sample_random_action(&mut rng);
        assert!(!action.is_empty());
    }

    #[test]
    fn test_context_full_episode() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let ctx_opt = EngineContext::new("simple");
        assert!(ctx_opt.is_some(), "simple game should be registered");
        let mut ctx = ctx_opt.unwrap();

        let reset_result = ctx.reset(42, &[]).unwrap();
        let mut state = reset_result.state;
        let mut done = false;
        let mut steps = 0;

        while !done && steps < 20 {
            let action = 2u32.to_le_bytes().to_vec(); // Always add 2
            let step_result = ctx.step(&state, &action).unwrap();
            state = step_result.state;
            done = step_result.done;
            steps += 1;
        }

        assert!(done);
        assert!(steps <= 10); // Should terminate when state >= 10
    }

    // =========================================================================
    // Fuzz-style tests for EngineContext
    // =========================================================================

    #[test]
    fn test_context_random_seeds_deterministic() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        // Same seed should produce same results
        for seed in [0u64, 42, 12345, u64::MAX] {
            let mut ctx1 = EngineContext::new("simple").unwrap();
            let mut ctx2 = EngineContext::new("simple").unwrap();

            let result1 = ctx1.reset(seed, &[]).unwrap();
            let result2 = ctx2.reset(seed, &[]).unwrap();

            assert_eq!(
                result1.state, result2.state,
                "Same seed should produce same state"
            );
            assert_eq!(
                result1.obs, result2.obs,
                "Same seed should produce same obs"
            );
        }
    }

    #[test]
    fn test_context_many_random_episodes() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let mut ctx = EngineContext::new("simple").unwrap();

        for episode_seed in 0..20u64 {
            let mut rng = ChaCha20Rng::seed_from_u64(episode_seed);

            let reset_result = ctx.reset(episode_seed, &[]).unwrap();
            let mut state = reset_result.state;
            let mut done = false;
            let mut steps = 0;
            const MAX_STEPS: usize = 100;

            while !done && steps < MAX_STEPS {
                // Sample random action
                let action = ctx.sample_random_action(&mut rng);

                let step_result = ctx.step(&state, &action);
                assert!(
                    step_result.is_ok(),
                    "Step should not error (episode={}, step={})",
                    episode_seed,
                    steps
                );

                let step_result = step_result.unwrap();
                state = step_result.state;
                done = step_result.done;
                steps += 1;
            }

            // Episode should terminate (SimpleGame terminates when state >= 10)
            assert!(
                done || steps == MAX_STEPS,
                "Episode {} should terminate",
                episode_seed
            );
        }
    }

    #[test]
    fn test_context_invalid_action_bytes() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let mut ctx = EngineContext::new("simple").unwrap();
        let reset_result = ctx.reset(42, &[]).unwrap();

        // SimpleGame expects 4 bytes for action (u32), test with wrong sizes
        let wrong_sizes = [vec![], vec![1], vec![1, 2], vec![1, 2, 3, 4, 5]];

        for bad_action in &wrong_sizes {
            let result = ctx.step(&reset_result.state, bad_action);
            assert!(
                result.is_err(),
                "Should reject action of size {}",
                bad_action.len()
            );
        }
    }

    #[test]
    fn test_context_invalid_state_bytes() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let mut ctx = EngineContext::new("simple").unwrap();

        // SimpleGame expects 4 bytes for state (u32)
        let wrong_state_sizes = [vec![], vec![1], vec![1, 2], vec![1, 2, 3, 4, 5]];
        let valid_action = 1u32.to_le_bytes().to_vec();

        for bad_state in &wrong_state_sizes {
            let result = ctx.step(bad_state, &valid_action);
            assert!(
                result.is_err(),
                "Should reject state of size {}",
                bad_state.len()
            );
        }
    }

    #[test]
    fn test_context_step_roundtrip_encoding() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        setup_registry();

        let mut ctx = EngineContext::new("simple").unwrap();

        // Reset and verify state decodes correctly
        let reset = ctx.reset(123, &[]).unwrap();
        let initial_state = u32::from_le_bytes(reset.state.clone().try_into().unwrap());
        assert_eq!(initial_state, 0);

        // Step and verify output state decodes correctly
        let action = 5u32.to_le_bytes().to_vec();
        let step = ctx.step(&reset.state, &action).unwrap();
        let new_state = u32::from_le_bytes(step.state.clone().try_into().unwrap());
        assert_eq!(new_state, 5);

        // Step again to verify chaining works
        let action2 = 3u32.to_le_bytes().to_vec();
        let step2 = ctx.step(&step.state, &action2).unwrap();
        let final_state = u32::from_le_bytes(step2.state.try_into().unwrap());
        assert_eq!(final_state, 8);
    }
}
