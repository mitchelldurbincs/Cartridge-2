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

// Default value functions - env var -> central config fallback
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
        .unwrap_or(0)
}

fn default_replay_backend() -> String {
    std::env::var("ACTOR_REPLAY_BACKEND")
        .unwrap_or_else(|_| CENTRAL_CONFIG.storage.replay_backend.clone())
}

fn default_postgres_url() -> Option<String> {
    std::env::var("ACTOR_POSTGRES_URL")
        .ok()
        .or_else(|| CENTRAL_CONFIG.storage.postgres_url.clone())
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(name = "actor")]
#[command(about = "Cartridge2 Actor - Self-play episode runner")]
pub struct Config {
    #[arg(long, default_value_t = default_actor_id())]
    pub actor_id: String,

    #[arg(long, default_value_t = default_env_id())]
    pub env_id: String,

    #[arg(long, default_value_t = default_max_episodes())]
    pub max_episodes: i32,

    #[arg(long, default_value_t = default_episode_timeout())]
    pub episode_timeout_secs: u64,

    #[arg(long, default_value_t = default_flush_interval())]
    pub flush_interval_secs: u64,

    #[arg(long, default_value_t = default_log_level())]
    pub log_level: String,

    #[arg(long, default_value_t = default_log_interval())]
    pub log_interval: u32,

    #[arg(long, default_value_t = default_replay_db_path())]
    pub replay_db_path: String,

    #[arg(long, default_value_t = default_data_dir())]
    pub data_dir: String,

    #[arg(long, default_value_t = default_num_simulations())]
    pub num_simulations: u32,

    #[arg(long, default_value_t = default_temp_threshold())]
    pub temp_threshold: u32,

    #[arg(long, default_value_t = default_replay_backend())]
    pub replay_backend: String,

    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postgres_url: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            actor_id: default_actor_id(),
            env_id: default_env_id(),
            max_episodes: default_max_episodes(),
            episode_timeout_secs: default_episode_timeout(),
            flush_interval_secs: default_flush_interval(),
            log_level: default_log_level(),
            log_interval: default_log_interval(),
            replay_db_path: default_replay_db_path(),
            data_dir: default_data_dir(),
            num_simulations: default_num_simulations(),
            temp_threshold: default_temp_threshold(),
            replay_backend: default_replay_backend(),
            postgres_url: default_postgres_url(),
        }
    }
}

impl Config {
    pub fn with_defaults(mut self) -> Self {
        if self.postgres_url.is_none() {
            self.postgres_url = default_postgres_url();
        }
        self
    }

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

        match self.replay_backend.as_str() {
            "sqlite" => {}
            "postgres" | "postgresql" => {
                if self.postgres_url.is_none() {
                    return Err(anyhow!(
                        "postgres_url is required when replay_backend is '{}'",
                        self.replay_backend
                    ));
                }
            }
            _ => {
                return Err(anyhow!(
                    "invalid replay_backend '{}', expected 'sqlite' or 'postgres'",
                    self.replay_backend
                ));
            }
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
            replay_backend: "sqlite".into(),
            postgres_url: None,
        }
    }

    #[test]
    fn validate_accepts_valid_configuration() {
        assert!(base_config().validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_actor_id() {
        let mut cfg = base_config();
        cfg.actor_id.clear();
        assert!(cfg.validate().unwrap_err().to_string().contains("actor_id"));
    }

    #[test]
    fn validate_rejects_empty_env_id() {
        let mut cfg = base_config();
        cfg.env_id.clear();
        assert!(cfg.validate().unwrap_err().to_string().contains("env_id"));
    }

    #[test]
    fn validate_rejects_invalid_log_level() {
        let mut cfg = base_config();
        cfg.log_level = "nope".into();
        assert!(cfg.validate().unwrap_err().to_string().contains("invalid log level"));
    }

    #[test]
    fn validate_rejects_zero_episode_timeout() {
        let mut cfg = base_config();
        cfg.episode_timeout_secs = 0;
        assert!(cfg.validate().unwrap_err().to_string().contains("episode_timeout_secs"));
    }

    #[test]
    fn validate_rejects_zero_flush_interval() {
        let mut cfg = base_config();
        cfg.flush_interval_secs = 0;
        assert!(cfg.validate().unwrap_err().to_string().contains("flush_interval_secs"));
    }

    #[test]
    fn validate_accepts_negative_max_episodes() {
        let mut cfg = base_config();
        cfg.max_episodes = -1;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn episode_timeout_returns_correct_duration() {
        assert_eq!(base_config().episode_timeout(), Duration::from_secs(30));
    }

    #[test]
    fn flush_interval_returns_correct_duration() {
        assert_eq!(base_config().flush_interval(), Duration::from_secs(5));
    }

    #[test]
    fn model_path_constructs_correctly() {
        assert_eq!(base_config().model_path(), "../data/models/latest.onnx");
    }
}
