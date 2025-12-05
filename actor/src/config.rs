//! Configuration for the Actor service

use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::level_filters::LevelFilter;

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(name = "actor")]
#[command(about = "Cartridge2 Actor - Self-play episode runner")]
#[command(
    long_about = "Actor that runs game episodes using the engine library and stores
transitions in the SQLite replay buffer for training."
)]
pub struct Config {
    /// Unique actor identifier
    #[arg(long, env = "ACTOR_ACTOR_ID", default_value = "actor-1")]
    pub actor_id: String,

    /// Environment ID to run (e.g., tictactoe)
    #[arg(long, env = "ACTOR_ENV_ID", default_value = "tictactoe")]
    pub env_id: String,

    /// Maximum episodes to run (-1 for unlimited)
    #[arg(long, env = "ACTOR_MAX_EPISODES", default_value = "-1")]
    pub max_episodes: i32,

    /// Timeout per episode in seconds
    #[arg(long, env = "ACTOR_EPISODE_TIMEOUT", default_value = "30")]
    pub episode_timeout_secs: u64,

    /// Interval to flush data in seconds
    #[arg(long, env = "ACTOR_FLUSH_INTERVAL", default_value = "5")]
    pub flush_interval_secs: u64,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "ACTOR_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Log progress every N episodes (0 to disable)
    #[arg(long, env = "ACTOR_LOG_INTERVAL", default_value = "10")]
    pub log_interval: u32,

    /// Path to SQLite replay database
    #[arg(long, env = "ACTOR_REPLAY_DB_PATH", default_value = "./data/replay.db")]
    pub replay_db_path: String,

    /// Data directory for models and other files
    #[arg(long, env = "ACTOR_DATA_DIR", default_value = "./data")]
    pub data_dir: String,
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
            replay_db_path: "./data/replay.db".into(),
            data_dir: "./data".into(),
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
        assert_eq!(cfg.model_path(), "./data/models/latest.onnx");
    }
}
