//! Centralized configuration loading from config.toml.
//!
//! This module provides a single source of truth for configuration values,
//! loaded from config.toml at the project root with support for environment
//! variable overrides.

use serde::Deserialize;
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Root configuration structure matching config.toml
#[derive(Debug, Deserialize, Default)]
pub struct CentralConfig {
    #[serde(default)]
    pub common: CommonConfig,
    #[serde(default)]
    pub training: TrainingConfig,
    #[serde(default)]
    pub evaluation: EvaluationConfig,
    #[serde(default)]
    pub actor: ActorConfig,
    #[serde(default)]
    pub web: WebConfig,
    #[serde(default)]
    pub mcts: MctsConfig,
}

#[derive(Debug, Deserialize)]
pub struct CommonConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    #[serde(default = "default_env_id")]
    pub env_id: String,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            env_id: default_env_id(),
            log_level: default_log_level(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    #[serde(default = "default_iterations")]
    pub iterations: i32,
    #[serde(default = "default_one")]
    pub start_iteration: i32,
    #[serde(default = "default_episodes")]
    pub episodes_per_iteration: i32,
    #[serde(default = "default_steps")]
    pub steps_per_iteration: i32,
    #[serde(default = "default_batch_size")]
    pub batch_size: i32,
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_grad_clip_norm")]
    pub grad_clip_norm: f64,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: i32,
    #[serde(default = "default_max_checkpoints")]
    pub max_checkpoints: i32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            iterations: default_iterations(),
            start_iteration: default_one(),
            episodes_per_iteration: default_episodes(),
            steps_per_iteration: default_steps(),
            batch_size: default_batch_size(),
            learning_rate: default_lr(),
            weight_decay: default_weight_decay(),
            grad_clip_norm: default_grad_clip_norm(),
            device: default_device(),
            checkpoint_interval: default_checkpoint_interval(),
            max_checkpoints: default_max_checkpoints(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EvaluationConfig {
    #[serde(default = "default_one")]
    pub interval: i32,
    #[serde(default = "default_eval_games")]
    pub games: i32,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            interval: default_one(),
            games: default_eval_games(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ActorConfig {
    #[serde(default = "default_actor_id")]
    pub actor_id: String,
    #[serde(default = "default_max_episodes")]
    pub max_episodes: i32,
    #[serde(default = "default_episode_timeout")]
    pub episode_timeout_secs: u64,
    #[serde(default = "default_flush_interval")]
    pub flush_interval_secs: u64,
    #[serde(default = "default_log_interval")]
    pub log_interval: u32,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            actor_id: default_actor_id(),
            max_episodes: default_max_episodes(),
            episode_timeout_secs: default_episode_timeout(),
            flush_interval_secs: default_flush_interval(),
            log_interval: default_log_interval(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct WebConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct MctsConfig {
    #[serde(default = "default_num_simulations")]
    pub num_simulations: u32,
    #[serde(default = "default_c_puct")]
    pub c_puct: f64,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_dirichlet_alpha")]
    pub dirichlet_alpha: f64,
    #[serde(default = "default_dirichlet_weight")]
    pub dirichlet_weight: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: default_num_simulations(),
            c_puct: default_c_puct(),
            temperature: default_temperature(),
            dirichlet_alpha: default_dirichlet_alpha(),
            dirichlet_weight: default_dirichlet_weight(),
        }
    }
}

// Default value functions
fn default_data_dir() -> String {
    "./data".to_string()
}
fn default_env_id() -> String {
    "tictactoe".to_string()
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_iterations() -> i32 {
    100
}
fn default_one() -> i32 {
    1
}
fn default_episodes() -> i32 {
    500
}
fn default_steps() -> i32 {
    1000
}
fn default_batch_size() -> i32 {
    64
}
fn default_lr() -> f64 {
    0.001
}
fn default_weight_decay() -> f64 {
    0.0001
}
fn default_grad_clip_norm() -> f64 {
    1.0
}
fn default_device() -> String {
    "cpu".to_string()
}
fn default_checkpoint_interval() -> i32 {
    100
}
fn default_max_checkpoints() -> i32 {
    10
}
fn default_eval_games() -> i32 {
    50
}
fn default_actor_id() -> String {
    "actor-1".to_string()
}
fn default_max_episodes() -> i32 {
    -1
}
fn default_episode_timeout() -> u64 {
    30
}
fn default_flush_interval() -> u64 {
    5
}
fn default_log_interval() -> u32 {
    50
}
fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_num_simulations() -> u32 {
    800
}
fn default_c_puct() -> f64 {
    1.4
}
fn default_temperature() -> f64 {
    1.0
}
fn default_dirichlet_alpha() -> f64 {
    0.3
}
fn default_dirichlet_weight() -> f64 {
    0.25
}

/// Standard locations to search for config.toml
const CONFIG_SEARCH_PATHS: &[&str] = &[
    "config.toml",      // Current directory
    "../config.toml",   // Parent directory (when running from actor/)
    "/app/config.toml", // Docker container
];

/// Load the central configuration from config.toml.
///
/// Searches for config.toml in standard locations and falls back to defaults
/// if not found. Environment variable CARTRIDGE_CONFIG can override the path.
pub fn load_config() -> CentralConfig {
    // Check for explicit config path
    if let Ok(path) = std::env::var("CARTRIDGE_CONFIG") {
        let path = PathBuf::from(&path);
        if path.exists() {
            info!("Loading config from CARTRIDGE_CONFIG: {}", path.display());
            return load_from_path(&path);
        }
        warn!(
            "CARTRIDGE_CONFIG={} not found, searching defaults",
            path.display()
        );
    }

    // Search default locations
    for path_str in CONFIG_SEARCH_PATHS {
        let path = PathBuf::from(path_str);
        if path.exists() {
            info!("Loading config from {}", path.display());
            return load_from_path(&path);
        }
    }

    // Fall back to defaults
    debug!("No config.toml found, using built-in defaults");
    apply_env_overrides(CentralConfig::default())
}

fn load_from_path(path: &PathBuf) -> CentralConfig {
    match std::fs::read_to_string(path) {
        Ok(content) => match toml::from_str(&content) {
            Ok(config) => apply_env_overrides(config),
            Err(e) => {
                warn!("Failed to parse {}: {}, using defaults", path.display(), e);
                apply_env_overrides(CentralConfig::default())
            }
        },
        Err(e) => {
            warn!("Failed to read {}: {}, using defaults", path.display(), e);
            apply_env_overrides(CentralConfig::default())
        }
    }
}

fn apply_env_overrides(mut config: CentralConfig) -> CentralConfig {
    for (key, value) in std::env::vars() {
        let Some(stripped) = key.strip_prefix("CARTRIDGE_") else {
            continue;
        };

        let (section, field) = match stripped.split_once('_') {
            Some(parts) => parts,
            None => continue,
        };

        match (section, field) {
            ("COMMON", "ENV_ID") => config.common.env_id = value,
            ("COMMON", "DATA_DIR") => config.common.data_dir = value,
            ("COMMON", "LOG_LEVEL") => config.common.log_level = value,

            ("TRAINING", "ITERATIONS") => {
                if let Ok(v) = value.parse() {
                    config.training.iterations = v
                }
            }
            ("TRAINING", "START_ITERATION") => {
                if let Ok(v) = value.parse() {
                    config.training.start_iteration = v
                }
            }
            ("TRAINING", "EPISODES_PER_ITERATION") => {
                if let Ok(v) = value.parse() {
                    config.training.episodes_per_iteration = v
                }
            }
            ("TRAINING", "STEPS_PER_ITERATION") => {
                if let Ok(v) = value.parse() {
                    config.training.steps_per_iteration = v
                }
            }
            ("TRAINING", "BATCH_SIZE") => {
                if let Ok(v) = value.parse() {
                    config.training.batch_size = v
                }
            }
            ("TRAINING", "LEARNING_RATE") => {
                if let Ok(v) = value.parse() {
                    config.training.learning_rate = v
                }
            }
            ("TRAINING", "WEIGHT_DECAY") => {
                if let Ok(v) = value.parse() {
                    config.training.weight_decay = v
                }
            }
            ("TRAINING", "GRAD_CLIP_NORM") => {
                if let Ok(v) = value.parse() {
                    config.training.grad_clip_norm = v
                }
            }
            ("TRAINING", "DEVICE") => config.training.device = value,
            ("TRAINING", "CHECKPOINT_INTERVAL") => {
                if let Ok(v) = value.parse() {
                    config.training.checkpoint_interval = v
                }
            }
            ("TRAINING", "MAX_CHECKPOINTS") => {
                if let Ok(v) = value.parse() {
                    config.training.max_checkpoints = v
                }
            }

            ("EVALUATION", "INTERVAL") => {
                if let Ok(v) = value.parse() {
                    config.evaluation.interval = v
                }
            }
            ("EVALUATION", "GAMES") => {
                if let Ok(v) = value.parse() {
                    config.evaluation.games = v
                }
            }

            ("ACTOR", "ACTOR_ID") => config.actor.actor_id = value,
            ("ACTOR", "MAX_EPISODES") => {
                if let Ok(v) = value.parse() {
                    config.actor.max_episodes = v
                }
            }
            ("ACTOR", "EPISODE_TIMEOUT_SECS") => {
                if let Ok(v) = value.parse() {
                    config.actor.episode_timeout_secs = v
                }
            }
            ("ACTOR", "FLUSH_INTERVAL_SECS") => {
                if let Ok(v) = value.parse() {
                    config.actor.flush_interval_secs = v
                }
            }
            ("ACTOR", "LOG_INTERVAL") => {
                if let Ok(v) = value.parse() {
                    config.actor.log_interval = v
                }
            }

            ("WEB", "HOST") => config.web.host = value,
            ("WEB", "PORT") => {
                if let Ok(v) = value.parse() {
                    config.web.port = v
                }
            }

            ("MCTS", "NUM_SIMULATIONS") => {
                if let Ok(v) = value.parse() {
                    config.mcts.num_simulations = v
                }
            }
            ("MCTS", "C_PUCT") => {
                if let Ok(v) = value.parse() {
                    config.mcts.c_puct = v
                }
            }
            ("MCTS", "TEMPERATURE") => {
                if let Ok(v) = value.parse() {
                    config.mcts.temperature = v
                }
            }
            ("MCTS", "DIRICHLET_ALPHA") => {
                if let Ok(v) = value.parse() {
                    config.mcts.dirichlet_alpha = v
                }
            }
            ("MCTS", "DIRICHLET_WEIGHT") => {
                if let Ok(v) = value.parse() {
                    config.mcts.dirichlet_weight = v
                }
            }
            _ => {}
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CentralConfig::default();
        assert_eq!(config.common.env_id, "tictactoe");
        assert_eq!(config.common.data_dir, "./data");
        assert_eq!(config.actor.actor_id, "actor-1");
        assert_eq!(config.actor.max_episodes, -1);
    }

    #[test]
    fn test_cartridge_env_overrides() {
        // Ensure overrides are applied from environment variables
        std::env::set_var("CARTRIDGE_COMMON_ENV_ID", "connect4");
        std::env::set_var("CARTRIDGE_ACTOR_MAX_EPISODES", "7");
        std::env::set_var("CARTRIDGE_TRAINING_WEIGHT_DECAY", "0.5");

        let config = load_config();
        assert_eq!(config.common.env_id, "connect4");
        assert_eq!(config.actor.max_episodes, 7);
        assert!((config.training.weight_decay - 0.5).abs() < f64::EPSILON);

        std::env::remove_var("CARTRIDGE_COMMON_ENV_ID");
        std::env::remove_var("CARTRIDGE_ACTOR_MAX_EPISODES");
        std::env::remove_var("CARTRIDGE_TRAINING_WEIGHT_DECAY");
    }

    #[test]
    fn test_parse_config_toml() {
        let toml_content = r#"
[common]
env_id = "connect4"
data_dir = "/custom/data"

[actor]
actor_id = "my-actor"
max_episodes = 100
"#;
        let config: CentralConfig = toml::from_str(toml_content).unwrap();
        assert_eq!(config.common.env_id, "connect4");
        assert_eq!(config.common.data_dir, "/custom/data");
        assert_eq!(config.actor.actor_id, "my-actor");
        assert_eq!(config.actor.max_episodes, 100);
    }

    #[test]
    fn test_partial_config() {
        // Test that partial config uses defaults for missing fields
        let toml_content = r#"
[common]
env_id = "connect4"
"#;
        let config: CentralConfig = toml::from_str(toml_content).unwrap();
        assert_eq!(config.common.env_id, "connect4");
        assert_eq!(config.common.data_dir, "./data"); // default
        assert_eq!(config.actor.actor_id, "actor-1"); // default
    }
}
