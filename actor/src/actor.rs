//! Actor implementation using engine-core library directly

use anyhow::{anyhow, Result};
use engine_core::EngineContext;
use mcts::MctsConfig;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Mutex, MutexGuard,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Get the current resident set size (RSS) in MB from /proc/self/status.
/// Returns None if unable to read (e.g., on non-Linux systems).
fn get_rss_mb() -> Option<f64> {
    use std::fs;
    // Read /proc/self/status and find VmRSS line
    if let Ok(contents) = fs::read_to_string("/proc/self/status") {
        for line in contents.lines() {
            if line.starts_with("VmRSS:") {
                // Format: "VmRSS:    12345 kB"
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<f64>() {
                        return Some(kb / 1024.0); // Convert to MB
                    }
                }
            }
        }
    }
    None
}

use crate::config::Config;
use crate::game_config::{get_config, GameConfig};
use crate::mcts_policy::MctsPolicy;
use crate::model_watcher::ModelWatcher;
use crate::storage::{create_replay_store, ReplayStore, StorageBackend, StorageConfig, Transition};
use std::sync::Arc;

/// Extract a legal moves mask from the observation bytes.
///
/// The observation contains f32 values where indices starting at `legal_mask_offset`
/// are the legal moves (1.0 = legal, 0.0 = illegal). This function extracts
/// those values and packs them into a u64 bitmask.
///
/// This is more robust than hardcoding the initial mask, as it works for:
/// - Any initial state (including non-empty boards if reset from saved state)
/// - Any game that encodes legal moves in the observation
fn extract_legal_mask_from_obs(obs: &[u8], config: &GameConfig) -> u64 {
    // Each f32 is 4 bytes
    let legal_start_byte = config.legal_mask_offset * 4;
    let legal_end_byte = legal_start_byte + config.num_actions * 4;

    if obs.len() < legal_end_byte {
        // Fallback: if observation is too short, assume all legal (shouldn't happen)
        warn!(
            "Observation too short ({} bytes), expected at least {}. Using fallback mask.",
            obs.len(),
            legal_end_byte
        );
        return config.legal_mask_bits();
    }

    let mut mask = 0u64;
    for i in 0..config.num_actions {
        let byte_offset = legal_start_byte + i * 4;
        let value = f32::from_le_bytes([
            obs[byte_offset],
            obs[byte_offset + 1],
            obs[byte_offset + 2],
            obs[byte_offset + 3],
        ]);
        // If value > 0.5, consider it legal (should be 1.0 for legal, 0.0 for illegal)
        if value > 0.5 {
            mask |= 1u64 << i;
        }
    }
    mask
}

pub struct Actor {
    config: Config,
    game_config: GameConfig,
    engine: Mutex<EngineContext>,
    mcts_policy: Mutex<MctsPolicy>,
    replay: Arc<dyn ReplayStore>,
    episode_count: AtomicU32,
    shutdown_signal: AtomicBool,
    model_watcher: ModelWatcher,
}

impl Actor {
    pub async fn new(config: Config) -> Result<Self> {
        // Register games
        games_tictactoe::register_tictactoe();
        games_connect4::register_connect4();

        // Get game configuration from registry
        let game_config = get_config(&config.env_id)?;
        info!(
            "Loaded game config for {}: {} actions, {} obs size",
            config.env_id, game_config.num_actions, game_config.obs_size
        );

        // Create engine context for the specified game
        let engine = EngineContext::new(&config.env_id)
            .ok_or_else(|| anyhow!("Game '{}' not registered", config.env_id))?;

        let caps = engine.capabilities();
        info!(
            "Actor {} initialized for environment {}",
            config.actor_id, caps.id.env_id
        );
        info!(
            "Game capabilities: max_horizon={}, preferred_batch={}",
            caps.max_horizon, caps.preferred_batch
        );

        let num_actions = game_config.num_actions;
        let obs_size = game_config.obs_size;

        // Create MCTS policy with training configuration
        // num_simulations and temp_threshold are configurable via CLI/env for orchestrator control
        let mcts_config = MctsConfig::for_training()
            .with_simulations(config.num_simulations)
            .with_temperature(1.0); // Base exploration temperature

        let mcts_policy = MctsPolicy::new(config.env_id.clone(), num_actions, obs_size)
            .with_config(mcts_config)
            .with_temp_schedule(config.temp_threshold, 0.1); // Late-game temp

        info!(
            "MCTS config: {} simulations, temp_threshold={} (0=disabled)",
            config.num_simulations, config.temp_threshold
        );

        // Create model watcher
        let model_dir = format!("{}/models", config.data_dir);
        let model_watcher = ModelWatcher::new(
            &model_dir,
            "latest.onnx",
            obs_size,
            mcts_policy.evaluator_ref(),
        );

        // Try to load existing model
        match model_watcher.try_load_existing() {
            Ok(true) => info!("Loaded existing model"),
            Ok(false) => {
                info!("No existing model found, will use random policy until model available")
            }
            Err(e) => warn!("Failed to load existing model: {}", e),
        }

        // Initialize replay buffer based on configured backend
        let backend: StorageBackend = config.replay_backend.parse()?;
        let storage_config = StorageConfig {
            backend,
            sqlite_path: Some(config.replay_db_path.clone()),
            #[cfg(feature = "postgres")]
            postgres_url: config.postgres_url.clone(),
        };

        let replay = create_replay_store(&storage_config).await?;
        info!(
            "Replay buffer initialized with {:?} backend",
            config.replay_backend
        );

        // Store game metadata in database (makes it self-describing for trainer)
        let metadata = engine.metadata();
        replay.store_metadata(&metadata).await?;
        info!(
            "Stored game metadata: {} actions, {} obs_size, legal_mask_offset={}",
            metadata.num_actions, metadata.obs_size, metadata.legal_mask_offset
        );

        Ok(Self {
            config,
            game_config,
            engine: Mutex::new(engine),
            mcts_policy: Mutex::new(mcts_policy),
            replay: Arc::from(replay),
            episode_count: AtomicU32::new(0),
            shutdown_signal: AtomicBool::new(false),
            model_watcher,
        })
    }

    pub async fn run(&self) -> Result<()> {
        let initial_rss = get_rss_mb().unwrap_or(0.0);
        info!(
            actor_id = %self.config.actor_id,
            max_episodes = self.config.max_episodes,
            initial_rss_mb = format!("{:.1}", initial_rss),
            "Actor starting main loop"
        );

        // Start model watcher
        let mut model_updates = self.model_watcher.start_watching().await?;

        // Setup flush timer for periodic database commits
        let mut flush_timer = interval(self.config.flush_interval());

        info!("Entering main event loop");

        loop {
            // Check shutdown signal
            if self.shutdown_signal.load(Ordering::Relaxed) {
                info!("Shutdown signal received, stopping actor");
                break;
            }

            tokio::select! {
                // Handle model updates
                Some(()) = model_updates.recv() => {
                    info!("Model updated, next episode will use new model");
                }

                _ = flush_timer.tick() => {
                    // Periodic flush is handled by SQLite transactions
                    debug!("Periodic flush tick");
                }

                _ = tokio::time::sleep(Duration::from_millis(1)) => {
                    // Check episode limit
                    let current_episode_count = self.episode_count.load(Ordering::Relaxed);
                    debug!(
                        current_episodes = current_episode_count,
                        max_episodes = self.config.max_episodes,
                        "Checking episode limit"
                    );
                    if self.config.max_episodes > 0 && current_episode_count >= self.config.max_episodes as u32 {
                        info!("Reached maximum episodes ({}), stopping", self.config.max_episodes);
                        break;
                    }

                    // Run an episode
                    let episode_start = Instant::now();
                    match self.run_episode().await {
                        Ok((steps, total_reward)) => {
                            let new_count = self.episode_count.fetch_add(1, Ordering::Relaxed) + 1;
                            let duration = episode_start.elapsed().as_secs_f64();
                            debug!(
                                episode = new_count,
                                steps,
                                total_reward,
                                duration,
                                "Episode completed"
                            );
                            #[allow(clippy::manual_is_multiple_of)] // is_multiple_of is unstable
                            if self.config.log_interval > 0 && new_count % self.config.log_interval == 0 {
                                // Include memory diagnostics in periodic logging
                                let rss_info = get_rss_mb()
                                    .map(|mb| format!(", RSS: {:.1} MB", mb))
                                    .unwrap_or_default();
                                let avg_duration = duration; // Most recent episode duration
                                info!(
                                    "Completed {} episodes (last: {:.2}s{})",
                                    new_count, avg_duration, rss_info
                                );
                            }
                        }
                        Err(e) => {
                            let count = self.episode_count.load(Ordering::Relaxed);
                            error!("Episode {} failed: {}", count + 1, e);
                            // Continue with next episode rather than stopping
                        }
                    }
                }
            }
        }

        // Report final memory usage
        let final_rss = get_rss_mb().unwrap_or(0.0);
        let rss_growth = final_rss - initial_rss;
        info!(
            "Actor stopped gracefully (final RSS: {:.1} MB, growth: {:.1} MB)",
            final_rss, rss_growth
        );
        Ok(())
    }

    pub fn shutdown(&self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);
        info!("Shutdown signal set");
    }

    /// Acquire engine lock with consistent error handling
    fn lock_engine(&self) -> Result<MutexGuard<'_, EngineContext>> {
        self.engine
            .lock()
            .map_err(|e| anyhow!("Engine lock poisoned: {}", e))
    }

    /// Acquire MCTS policy lock with consistent error handling
    fn lock_mcts_policy(&self) -> Result<MutexGuard<'_, MctsPolicy>> {
        self.mcts_policy
            .lock()
            .map_err(|e| anyhow!("MCTS policy lock poisoned: {}", e))
    }

    async fn run_episode(&self) -> Result<(u32, f32)> {
        let episode_count = self.episode_count.load(Ordering::Relaxed);
        let episode_id = format!(
            "{}-ep-{}-{}",
            self.config.actor_id,
            episode_count,
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
        );

        debug!(
            episode = episode_count + 1,
            env_id = %self.config.env_id,
            "Starting new episode"
        );

        // Generate a seed for this episode
        let seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;

        // Reset the game and get max_horizon for step limit
        let (reset_result, max_horizon) = {
            let mut engine = self.lock_engine()?;
            let caps = engine.capabilities();
            let reset = engine.reset(seed, &[])?;
            (reset, caps.max_horizon)
        };

        let mut current_state = reset_result.state;
        let mut current_obs = reset_result.obs;
        let mut step_number = 0u32;
        let mut steps_taken = 0u32;
        let mut total_reward = 0.0f32;

        // Pre-allocate transition buffer for batched insert (reduces fsync overhead)
        // TicTacToe games are typically 5-9 moves
        let mut transitions: Vec<Transition> = Vec::with_capacity(12);

        // Extract initial legal moves mask from the observation.
        // The observation format is game-specific:
        //   - board_view: N floats (indices 0 to legal_mask_offset-1)
        //   - legal_moves: num_actions floats (1.0 = legal, 0.0 = illegal)
        //   - other game-specific data
        // This approach works for any game that encodes legal moves in the observation.
        let mut current_legal_mask = extract_legal_mask_from_obs(&current_obs, &self.game_config);

        // Timeout tracking
        let episode_start = Instant::now();
        let timeout = Duration::from_secs(self.config.episode_timeout_secs);
        // Step limit: use 10x max_horizon as a generous upper bound
        // This protects against infinite loops in buggy game implementations
        let max_steps = max_horizon.saturating_mul(10).max(1000);

        debug!(
            "Started episode {} (timeout={}s, max_steps={})",
            episode_id,
            timeout.as_secs(),
            max_steps
        );

        loop {
            // Check timeout
            if episode_start.elapsed() > timeout {
                warn!(
                    "Episode {} timed out after {:?} ({} steps taken)",
                    episode_id,
                    episode_start.elapsed(),
                    steps_taken
                );
                return Err(anyhow!(
                    "Episode timed out after {} seconds",
                    timeout.as_secs()
                ));
            }

            // Check step limit
            if steps_taken >= max_steps {
                warn!(
                    "Episode {} exceeded max steps ({}) without terminating",
                    episode_id, max_steps
                );
                return Err(anyhow!(
                    "Episode exceeded {} steps without terminating",
                    max_steps
                ));
            }

            // Select action using MCTS policy
            // Pass step_number for temperature schedule (lower temp in late game)
            let policy_result = {
                let mut policy = self.lock_mcts_policy()?;
                policy.select_action(
                    &current_state,
                    &current_obs,
                    current_legal_mask,
                    step_number,
                )?
            };

            // Take step in environment
            let step_result = {
                let mut engine = self.lock_engine()?;
                engine.step(&current_state, &policy_result.action)?
            };

            total_reward += step_result.reward;
            steps_taken += 1;

            // Extract legal moves mask from info for next step
            // Games pack legal_moves_mask in lower N bits of info (N = num_actions)
            let next_legal_mask = step_result.info & self.game_config.legal_mask_bits();

            // Convert policy to bytes for storage
            let policy_bytes: Vec<u8> = policy_result
                .policy
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

            // Create transition - we'll batch insert at episode end
            // game_outcome is initially None - we backfill it after the episode ends
            //
            // Note: We move current_state/current_obs into the transition and then
            // update them from step_result. This avoids cloning on each step.
            let transition = Transition {
                id: format!("{}-step-{}", episode_id, step_number),
                env_id: self.config.env_id.clone(),
                episode_id: episode_id.clone(),
                step_number,
                state: std::mem::take(&mut current_state),
                action: policy_result.action,
                next_state: step_result.state.clone(),
                observation: std::mem::take(&mut current_obs),
                next_observation: step_result.obs.clone(),
                reward: step_result.reward,
                done: step_result.done,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                policy_probs: policy_bytes,
                mcts_value: policy_result.value,
                game_outcome: None,
            };

            // Add to batch (we'll store all at once at episode end)
            transitions.push(transition);

            // Check if episode is done
            if step_result.done {
                let total_steps = step_number + 1;
                debug!(
                    "Episode {} completed in {} steps, total reward: {:.2}",
                    episode_id, total_steps, total_reward
                );

                // Backfill game outcomes for all transitions before storing
                // The final reward indicates the outcome from the last mover's perspective:
                // +1 = win, -1 = loss, 0 = draw
                let final_outcome = step_result.reward;
                for t in &mut transitions {
                    let steps_from_end =
                        total_steps.saturating_sub(1).saturating_sub(t.step_number);
                    let sign = if steps_from_end % 2 == 0 { 1.0 } else { -1.0 };
                    t.game_outcome = Some(final_outcome * sign);
                }

                // Batch store all transitions in a single transaction
                // This dramatically reduces fsync overhead (1 fsync vs N fsyncs)
                if let Err(e) = self.replay.store_batch(&transitions).await {
                    error!(
                        "Failed to store transitions for episode {}: {}",
                        episode_id, e
                    );
                    return Err(e);
                }
                debug!(
                    "Stored {} transitions with game_outcome={} for episode {}",
                    transitions.len(),
                    final_outcome,
                    episode_id
                );
                break;
            }

            // Update state for next step
            current_state = step_result.state;
            current_obs = step_result.obs;
            current_legal_mask = next_legal_mask;
            step_number += 1;
        }

        Ok((steps_taken, total_reward))
    }

    /// Get current episode count (for testing)
    #[allow(dead_code)]
    pub fn episode_count(&self) -> u32 {
        self.episode_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_config(db_path: &str) -> Config {
        Config {
            actor_id: "test-actor".into(),
            env_id: "tictactoe".into(),
            max_episodes: 1,
            episode_timeout_secs: 30,
            flush_interval_secs: 5,
            log_level: "info".into(),
            log_interval: 10,
            replay_db_path: db_path.into(),
            data_dir: "./data".into(),
            num_simulations: 50, // Fewer for tests
            temp_threshold: 0,   // Disabled for tests
            replay_backend: "sqlite".into(),
            postgres_url: None,
        }
    }

    #[tokio::test]
    async fn test_actor_creation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_replay.db");
        let config = test_config(db_path.to_str().unwrap());

        let actor = Actor::new(config).await;
        assert!(actor.is_ok());
    }

    #[tokio::test]
    async fn test_actor_run_single_episode() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_replay.db");
        let config = test_config(db_path.to_str().unwrap());

        let actor = Actor::new(config).await.unwrap();

        // Run a single episode
        let result = actor.run_episode().await;
        assert!(result.is_ok());

        let (steps, reward) = result.unwrap();
        assert!(steps > 0, "Episode should have at least one step");
        // TicTacToe gives reward at end of game
        // Steps and reward are validated by assertions above
        debug!(steps, reward, "Episode completed");
    }

    #[tokio::test]
    async fn test_actor_nonexistent_game() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_replay.db");
        let mut config = test_config(db_path.to_str().unwrap());
        config.env_id = "nonexistent_game".into();

        let result = Actor::new(config).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        // Could fail at game_config lookup or engine context creation
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Unknown game") || err_msg.contains("not registered"),
            "Expected error about unknown/unregistered game, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_actor_stores_transitions() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_replay.db");
        let config = test_config(db_path.to_str().unwrap());

        let actor = Actor::new(config).await.unwrap();

        // Run an episode
        actor.run_episode().await.unwrap();

        // Check that transitions were stored
        let count = actor.replay.count().await.unwrap();
        assert!(count > 0, "Should have stored some transitions");
    }
}
