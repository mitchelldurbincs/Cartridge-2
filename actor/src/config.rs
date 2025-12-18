//! Configuration for the Actor service
//!
//! Configuration is loaded from config.toml with environment variable overrides.
//! CLI arguments take highest priority, followed by env vars, then config.toml.

use anyhow::{anyhow, Result};
use clap::Parser;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::level_filters::LevelFilter;

use crate::central_config::{load_config, CentralConfig};

// Load central config once at startup
static CENTRAL_CONFIG: Lazy<CentralConfig> = Lazy::new(load_config);

// Default value functions that read from central config
fn default_actor_id() -> String {
    std::env::var("ACTOR_ACTOR_ID").unwrap_or_else(|_| CENTRAL_CONFIG.actor.actor_id.clone())
}

fn default_env_id() -> String {
    std::env::var("ACTOR_ENV_ID").unwrap_or_else(|_| CENTRAL_CONFIG.common.env_id.clone())
}

fn default_max_episodes() -> i32 {
    std::env::var("ACTOR_MAX_EPISODES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CENTRAL_CONFIG.actor.max_episodes)
}

fn default_episode_timeout() -> u64 {
    std::env::var("ACTOR_EPISODE_TIMEOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CENTRAL_CONFIG.actor.episode_timeout_secs)
}

fn default_flush_interval() -> u64 {
    std::env::var("ACTOR_FLUSH_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CENTRAL_CONFIG.actor.flush_interval_secs)
}

fn default_log_level() -> String {
    std::env::var("ACTOR_LOG_LEVEL").unwrap_or_else(|_| CENTRAL_CONFIG.common.log_level.clone())
}

fn default_log_interval() -> u32 {
    std::env::var("ACTOR_LOG_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CENTRAL_CONFIG.actor.log_interval)
}

fn default_replay_db_path() -> String {
    std::env::var("ACTOR_REPLAY_DB_PATH")
        .unwrap_or_else(|_| format!("{}/replay.db", CENTRAL_CONFIG.common.data_dir))
}

fn default_data_dir() -> String {
    std::env::var("ACTOR_DATA_DIR").unwrap_or_else(|_| CENTRAL_CONFIG.common.data_dir.clone())
}

fn default_num_simulations() -> u32 {
    std::env::var("ACTOR_NUM_SIMULATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CENTRAL_CONFIG.mcts.num_simulations)
}

fn default_temp_threshold() -> u32 {
    std::env::var("ACTOR_TEMP_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0) // Disabled by default
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(name = "actor")]
#[command(about = "Cartridge2 Actor - Self-play episode runner")]
#[command(
    long_about = "Actor that runs game episodes using the engine library and stores
transitions in the SQLite replay buffer for training.

Configuration is loaded from config.toml with environment variable overrides.
CLI arguments take highest priority."
)]
pub struct Config {
    /// Unique actor identifier
    #[arg(long, default_value_t = default_actor_id())]
    pub actor_id: String,

    /// Environment ID to run (e.g., tictactoe)
    #[arg(long, default_value_t = default_env_id())]
    pub env_id: String,

    /// Maximum episodes to run (-1 for unlimited)
    #[arg(long, default_value_t = default_max_episodes())]
    pub max_episodes: i32,

    /// Timeout per episode in seconds
    #[arg(long, default_value_t = default_episode_timeout())]
    pub episode_timeout_secs: u64,

    /// Interval to flush data in seconds
    #[arg(long, default_value_t = default_flush_interval())]
    pub flush_interval_secs: u64,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value_t = default_log_level())]
    pub log_level: String,

    /// Log progress every N episodes (0 to disable)
    #[arg(long, default_value_t = default_log_interval())]
    pub log_interval: u32,

    /// Path to SQLite replay database
    #[arg(long, default_value_t = default_replay_db_path())]
    pub replay_db_path: String,

    /// Data directory for models and other files
    #[arg(long, default_value_t = default_data_dir())]
    pub data_dir: String,

    /// Number of MCTS simulations per move
    #[arg(long, default_value_t = default_num_simulations())]
    pub num_simulations: u32,

    /// Move number after which to use lower temperature (0 to disable)
    /// When enabled, temperature drops to 0.1 after this many moves for more deterministic late-game play
    #[arg(long, default_value_t = default_temp_threshold())]
    pub temp_threshold: u32,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.actor_id.is_empty() {
            return Err(anyhow!("actor_id cannot be empty"));
        }

        if self.env_id.is_empty() {
            return Err(anyhow!("env_id cannot be empty"));
        }

        if self.episode_timeout_secs == 0 {
            return Err(anyhow!("episode_timeout_secs must be greater than 0"));
        }

        if self.flush_interval_secs == 0 {
            return Err(anyhow!("flush_interval_secs must be greater than 0"));
        }

        if self.log_level.parse::<LevelFilter>().is_err() {
            return Err(anyhow!(
                "invalid log level '{}', expected one of trace, debug, info, warn, error",
                self.log_level
            ));
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn episode_timeout(&self) -> Duration {
        Duration::from_secs(self.episode_timeout_secs)
    }

    pub fn flush_interval(&self) -> Duration {
        Duration::from_secs(self.flush_interval_secs)
    }

    /// Path to the latest model file
    #[allow(dead_code)]
    pub fn model_path(&self) -> String {
        format!("{}/models/latest.onnx", self.data_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> Config {
        Config {
            actor_id: "actor".into(),
            env_id: "tictactoe".into(),
            max_episodes: 1,
            episode_timeout_secs: 30,
            flush_interval_secs: 5,
            log_level: "info".into(),
            log_interval: 10,
            replay_db_path: "../data/replay.db".into(),
            data_dir: "../data".into(),
            num_simulations: 100,
            temp_threshold: 0,
        }
    }

    #[test]
    fn validate_accepts_valid_configuration() {
        let cfg = base_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_actor_id() {
        let mut cfg = base_config();
        cfg.actor_id.clear();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("actor_id"));
    }

    #[test]
    fn validate_rejects_empty_env_id() {
        let mut cfg = base_config();
        cfg.env_id.clear();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("env_id"));
    }

    #[test]
    fn validate_rejects_invalid_log_level() {
        let mut cfg = base_config();
        cfg.log_level = "nope".into();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("invalid log level"));
    }

    #[test]
    fn validate_rejects_zero_episode_timeout() {
        let mut cfg = base_config();
        cfg.episode_timeout_secs = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("episode_timeout_secs"));
    }

    #[test]
    fn validate_rejects_zero_flush_interval() {
        let mut cfg = base_config();
        cfg.flush_interval_secs = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("flush_interval_secs"));
    }

    #[test]
    fn validate_accepts_negative_max_episodes() {
        let mut cfg = base_config();
        cfg.max_episodes = -1; // Unlimited mode
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn episode_timeout_returns_correct_duration() {
        let cfg = base_config();
        assert_eq!(cfg.episode_timeout(), Duration::from_secs(30));
    }

    #[test]
    fn flush_interval_returns_correct_duration() {
        let cfg = base_config();
        assert_eq!(cfg.flush_interval(), Duration::from_secs(5));
    }

    #[test]
    fn model_path_constructs_correctly() {
        let cfg = base_config();
        assert_eq!(cfg.model_path(), "../data/models/latest.onnx");
    }
}
