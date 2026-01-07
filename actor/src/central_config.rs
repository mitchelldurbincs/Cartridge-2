//! Centralized configuration loading from config.toml.
//!
//! This module provides a single source of truth for configuration values,
//! loaded from config.toml at the project root with support for environment
//! variable overrides.

use serde::Deserialize;
use std::path::PathBuf;
use tracing::{debug, info, warn};

// Default constants (much cleaner than 26 separate functions)
mod defaults {
    pub const DATA_DIR: &str = "./data";
    pub const ENV_ID: &str = "tictactoe";
    pub const LOG_LEVEL: &str = "info";
    pub const ITERATIONS: i32 = 100;
    pub const START_ITERATION: i32 = 1;
    pub const EPISODES_PER_ITERATION: i32 = 500;
    pub const STEPS_PER_ITERATION: i32 = 1000;
    pub const BATCH_SIZE: i32 = 64;
    pub const LEARNING_RATE: f64 = 0.001;
    pub const WEIGHT_DECAY: f64 = 0.0001;
    pub const GRAD_CLIP_NORM: f64 = 1.0;
    pub const DEVICE: &str = "cpu";
    pub const CHECKPOINT_INTERVAL: i32 = 100;
    pub const MAX_CHECKPOINTS: i32 = 10;
    pub const EVAL_INTERVAL: i32 = 1;
    pub const EVAL_GAMES: i32 = 50;
    pub const ACTOR_ID: &str = "actor-1";
    pub const MAX_EPISODES: i32 = -1;
    pub const EPISODE_TIMEOUT_SECS: u64 = 30;
    pub const FLUSH_INTERVAL_SECS: u64 = 5;
    pub const LOG_INTERVAL: u32 = 50;
    pub const HOST: &str = "0.0.0.0";
    pub const PORT: u16 = 8080;
    pub const NUM_SIMULATIONS: u32 = 800;
    pub const C_PUCT: f64 = 1.4;
    pub const TEMPERATURE: f64 = 1.0;
    pub const DIRICHLET_ALPHA: f64 = 0.3;
    pub const DIRICHLET_WEIGHT: f64 = 0.25;
    pub const EVAL_BATCH_SIZE: usize = 32;
    pub const MODEL_BACKEND: &str = "filesystem";
    pub const POSTGRES_URL: &str = "postgresql://cartridge:cartridge@localhost:5432/cartridge";
    pub const POOL_MAX_SIZE: usize = 16;
    pub const POOL_CONNECT_TIMEOUT: u64 = 30;
    pub const POOL_IDLE_TIMEOUT: u64 = 300;
}

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
    #[serde(default)]
    pub storage: StorageConfig,
}

// Serde default functions (required for #[serde(default = "...")])
// These are thin wrappers around constants
fn d_data_dir() -> String {
    defaults::DATA_DIR.into()
}
fn d_env_id() -> String {
    defaults::ENV_ID.into()
}
fn d_log_level() -> String {
    defaults::LOG_LEVEL.into()
}
fn d_iterations() -> i32 {
    defaults::ITERATIONS
}
fn d_start_iteration() -> i32 {
    defaults::START_ITERATION
}
fn d_episodes() -> i32 {
    defaults::EPISODES_PER_ITERATION
}
fn d_steps() -> i32 {
    defaults::STEPS_PER_ITERATION
}
fn d_batch_size() -> i32 {
    defaults::BATCH_SIZE
}
fn d_lr() -> f64 {
    defaults::LEARNING_RATE
}
fn d_weight_decay() -> f64 {
    defaults::WEIGHT_DECAY
}
fn d_grad_clip() -> f64 {
    defaults::GRAD_CLIP_NORM
}
fn d_device() -> String {
    defaults::DEVICE.into()
}
fn d_ckpt_interval() -> i32 {
    defaults::CHECKPOINT_INTERVAL
}
fn d_max_ckpts() -> i32 {
    defaults::MAX_CHECKPOINTS
}
fn d_eval_interval() -> i32 {
    defaults::EVAL_INTERVAL
}
fn d_eval_games() -> i32 {
    defaults::EVAL_GAMES
}
fn d_actor_id() -> String {
    defaults::ACTOR_ID.into()
}
fn d_max_episodes() -> i32 {
    defaults::MAX_EPISODES
}
fn d_episode_timeout() -> u64 {
    defaults::EPISODE_TIMEOUT_SECS
}
fn d_flush_interval() -> u64 {
    defaults::FLUSH_INTERVAL_SECS
}
fn d_log_interval() -> u32 {
    defaults::LOG_INTERVAL
}
fn d_host() -> String {
    defaults::HOST.into()
}
fn d_port() -> u16 {
    defaults::PORT
}
fn d_num_sims() -> u32 {
    defaults::NUM_SIMULATIONS
}
fn d_c_puct() -> f64 {
    defaults::C_PUCT
}
fn d_temperature() -> f64 {
    defaults::TEMPERATURE
}
fn d_dirichlet_alpha() -> f64 {
    defaults::DIRICHLET_ALPHA
}
fn d_dirichlet_weight() -> f64 {
    defaults::DIRICHLET_WEIGHT
}
fn d_eval_batch_size() -> usize {
    defaults::EVAL_BATCH_SIZE
}
fn d_model_backend() -> String {
    defaults::MODEL_BACKEND.into()
}
fn d_postgres_url() -> Option<String> {
    Some(defaults::POSTGRES_URL.into())
}
fn d_pool_max_size() -> usize {
    defaults::POOL_MAX_SIZE
}
fn d_pool_connect_timeout() -> u64 {
    defaults::POOL_CONNECT_TIMEOUT
}
fn d_pool_idle_timeout() -> Option<u64> {
    Some(defaults::POOL_IDLE_TIMEOUT)
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct CommonConfig {
    #[serde(default = "d_data_dir")]
    pub data_dir: String,
    #[serde(default = "d_env_id")]
    pub env_id: String,
    #[serde(default = "d_log_level")]
    pub log_level: String,
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            data_dir: defaults::DATA_DIR.into(),
            env_id: defaults::ENV_ID.into(),
            log_level: defaults::LOG_LEVEL.into(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    #[serde(default = "d_iterations")]
    pub iterations: i32,
    #[serde(default = "d_start_iteration")]
    pub start_iteration: i32,
    #[serde(default = "d_episodes")]
    pub episodes_per_iteration: i32,
    #[serde(default = "d_steps")]
    pub steps_per_iteration: i32,
    #[serde(default = "d_batch_size")]
    pub batch_size: i32,
    #[serde(default = "d_lr")]
    pub learning_rate: f64,
    #[serde(default = "d_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "d_grad_clip")]
    pub grad_clip_norm: f64,
    #[serde(default = "d_device")]
    pub device: String,
    #[serde(default = "d_ckpt_interval")]
    pub checkpoint_interval: i32,
    #[serde(default = "d_max_ckpts")]
    pub max_checkpoints: i32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            iterations: defaults::ITERATIONS,
            start_iteration: defaults::START_ITERATION,
            episodes_per_iteration: defaults::EPISODES_PER_ITERATION,
            steps_per_iteration: defaults::STEPS_PER_ITERATION,
            batch_size: defaults::BATCH_SIZE,
            learning_rate: defaults::LEARNING_RATE,
            weight_decay: defaults::WEIGHT_DECAY,
            grad_clip_norm: defaults::GRAD_CLIP_NORM,
            device: defaults::DEVICE.into(),
            checkpoint_interval: defaults::CHECKPOINT_INTERVAL,
            max_checkpoints: defaults::MAX_CHECKPOINTS,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct EvaluationConfig {
    #[serde(default = "d_eval_interval")]
    pub interval: i32,
    #[serde(default = "d_eval_games")]
    pub games: i32,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            interval: defaults::EVAL_INTERVAL,
            games: defaults::EVAL_GAMES,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ActorConfig {
    #[serde(default = "d_actor_id")]
    pub actor_id: String,
    #[serde(default = "d_max_episodes")]
    pub max_episodes: i32,
    #[serde(default = "d_episode_timeout")]
    pub episode_timeout_secs: u64,
    #[serde(default = "d_flush_interval")]
    pub flush_interval_secs: u64,
    #[serde(default = "d_log_interval")]
    pub log_interval: u32,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            actor_id: defaults::ACTOR_ID.into(),
            max_episodes: defaults::MAX_EPISODES,
            episode_timeout_secs: defaults::EPISODE_TIMEOUT_SECS,
            flush_interval_secs: defaults::FLUSH_INTERVAL_SECS,
            log_interval: defaults::LOG_INTERVAL,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct WebConfig {
    #[serde(default = "d_host")]
    pub host: String,
    #[serde(default = "d_port")]
    pub port: u16,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            host: defaults::HOST.into(),
            port: defaults::PORT,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MctsConfig {
    #[serde(default = "d_num_sims")]
    pub num_simulations: u32,
    #[serde(default = "d_c_puct")]
    pub c_puct: f64,
    #[serde(default = "d_temperature")]
    pub temperature: f64,
    #[serde(default = "d_dirichlet_alpha")]
    pub dirichlet_alpha: f64,
    #[serde(default = "d_dirichlet_weight")]
    pub dirichlet_weight: f64,
    #[serde(default = "d_eval_batch_size")]
    pub eval_batch_size: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: defaults::NUM_SIMULATIONS,
            c_puct: defaults::C_PUCT,
            temperature: defaults::TEMPERATURE,
            dirichlet_alpha: defaults::DIRICHLET_ALPHA,
            dirichlet_weight: defaults::DIRICHLET_WEIGHT,
            eval_batch_size: defaults::EVAL_BATCH_SIZE,
        }
    }
}

/// Storage backend configuration
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    #[serde(default = "d_model_backend")]
    pub model_backend: String,
    #[serde(default = "d_postgres_url")]
    pub postgres_url: Option<String>,
    #[serde(default)]
    pub s3_bucket: Option<String>,
    #[serde(default)]
    pub s3_endpoint: Option<String>,
    /// Maximum number of connections in the PostgreSQL pool
    #[serde(default = "d_pool_max_size")]
    pub pool_max_size: usize,
    /// Timeout in seconds to wait for a connection from the pool
    #[serde(default = "d_pool_connect_timeout")]
    pub pool_connect_timeout: u64,
    /// Idle timeout for connections in seconds (None = no timeout)
    #[serde(default = "d_pool_idle_timeout")]
    pub pool_idle_timeout: Option<u64>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            model_backend: defaults::MODEL_BACKEND.into(),
            postgres_url: Some(defaults::POSTGRES_URL.into()),
            s3_bucket: None,
            s3_endpoint: None,
            pool_max_size: defaults::POOL_MAX_SIZE,
            pool_connect_timeout: defaults::POOL_CONNECT_TIMEOUT,
            pool_idle_timeout: Some(defaults::POOL_IDLE_TIMEOUT),
        }
    }
}

/// Standard locations to search for config.toml
const CONFIG_SEARCH_PATHS: &[&str] = &["config.toml", "../config.toml", "/app/config.toml"];

/// Load the central configuration from config.toml.
pub fn load_config() -> CentralConfig {
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

    for path_str in CONFIG_SEARCH_PATHS {
        let path = PathBuf::from(path_str);
        if path.exists() {
            info!("Loading config from {}", path.display());
            return load_from_path(&path);
        }
    }

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

/// Macro to reduce env override boilerplate
macro_rules! env_override {
    // String field
    ($config:expr, $section:ident . $field:ident, $key:expr) => {
        if let Ok(v) = std::env::var($key) {
            $config.$section.$field = v;
        }
    };
    // Parseable field (i32, u64, f64, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, parse) => {
        if let Ok(v) =
            std::env::var($key).and_then(|s| s.parse().map_err(|_| std::env::VarError::NotPresent))
        {
            $config.$section.$field = v;
        }
    };
    // Optional string field
    ($config:expr, $section:ident . $field:ident, $key:expr, optional) => {
        if let Ok(v) = std::env::var($key) {
            $config.$section.$field = Some(v);
        }
    };
    // Optional parseable field (Option<i32>, Option<u64>, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, optional_parse) => {
        if let Ok(v) =
            std::env::var($key).and_then(|s| s.parse().map_err(|_| std::env::VarError::NotPresent))
        {
            $config.$section.$field = Some(v);
        }
    };
}

fn apply_env_overrides(mut config: CentralConfig) -> CentralConfig {
    // Common
    env_override!(config, common.env_id, "CARTRIDGE_COMMON_ENV_ID");
    env_override!(config, common.data_dir, "CARTRIDGE_COMMON_DATA_DIR");
    env_override!(config, common.log_level, "CARTRIDGE_COMMON_LOG_LEVEL");

    // Training
    env_override!(
        config,
        training.iterations,
        "CARTRIDGE_TRAINING_ITERATIONS",
        parse
    );
    env_override!(
        config,
        training.start_iteration,
        "CARTRIDGE_TRAINING_START_ITERATION",
        parse
    );
    env_override!(
        config,
        training.episodes_per_iteration,
        "CARTRIDGE_TRAINING_EPISODES_PER_ITERATION",
        parse
    );
    env_override!(
        config,
        training.steps_per_iteration,
        "CARTRIDGE_TRAINING_STEPS_PER_ITERATION",
        parse
    );
    env_override!(
        config,
        training.batch_size,
        "CARTRIDGE_TRAINING_BATCH_SIZE",
        parse
    );
    env_override!(
        config,
        training.learning_rate,
        "CARTRIDGE_TRAINING_LEARNING_RATE",
        parse
    );
    env_override!(
        config,
        training.weight_decay,
        "CARTRIDGE_TRAINING_WEIGHT_DECAY",
        parse
    );
    env_override!(
        config,
        training.grad_clip_norm,
        "CARTRIDGE_TRAINING_GRAD_CLIP_NORM",
        parse
    );
    env_override!(config, training.device, "CARTRIDGE_TRAINING_DEVICE");
    env_override!(
        config,
        training.checkpoint_interval,
        "CARTRIDGE_TRAINING_CHECKPOINT_INTERVAL",
        parse
    );
    env_override!(
        config,
        training.max_checkpoints,
        "CARTRIDGE_TRAINING_MAX_CHECKPOINTS",
        parse
    );

    // Evaluation
    env_override!(
        config,
        evaluation.interval,
        "CARTRIDGE_EVALUATION_INTERVAL",
        parse
    );
    env_override!(
        config,
        evaluation.games,
        "CARTRIDGE_EVALUATION_GAMES",
        parse
    );

    // Actor
    env_override!(config, actor.actor_id, "CARTRIDGE_ACTOR_ACTOR_ID");
    env_override!(
        config,
        actor.max_episodes,
        "CARTRIDGE_ACTOR_MAX_EPISODES",
        parse
    );
    env_override!(
        config,
        actor.episode_timeout_secs,
        "CARTRIDGE_ACTOR_EPISODE_TIMEOUT_SECS",
        parse
    );
    env_override!(
        config,
        actor.flush_interval_secs,
        "CARTRIDGE_ACTOR_FLUSH_INTERVAL_SECS",
        parse
    );
    env_override!(
        config,
        actor.log_interval,
        "CARTRIDGE_ACTOR_LOG_INTERVAL",
        parse
    );

    // Web
    env_override!(config, web.host, "CARTRIDGE_WEB_HOST");
    env_override!(config, web.port, "CARTRIDGE_WEB_PORT", parse);

    // MCTS
    env_override!(
        config,
        mcts.num_simulations,
        "CARTRIDGE_MCTS_NUM_SIMULATIONS",
        parse
    );
    env_override!(config, mcts.c_puct, "CARTRIDGE_MCTS_C_PUCT", parse);
    env_override!(
        config,
        mcts.temperature,
        "CARTRIDGE_MCTS_TEMPERATURE",
        parse
    );
    env_override!(
        config,
        mcts.dirichlet_alpha,
        "CARTRIDGE_MCTS_DIRICHLET_ALPHA",
        parse
    );
    env_override!(
        config,
        mcts.dirichlet_weight,
        "CARTRIDGE_MCTS_DIRICHLET_WEIGHT",
        parse
    );
    env_override!(
        config,
        mcts.eval_batch_size,
        "CARTRIDGE_MCTS_EVAL_BATCH_SIZE",
        parse
    );

    // Storage
    env_override!(
        config,
        storage.model_backend,
        "CARTRIDGE_STORAGE_MODEL_BACKEND"
    );
    env_override!(
        config,
        storage.postgres_url,
        "CARTRIDGE_STORAGE_POSTGRES_URL",
        optional
    );
    env_override!(
        config,
        storage.s3_bucket,
        "CARTRIDGE_STORAGE_S3_BUCKET",
        optional
    );
    env_override!(
        config,
        storage.s3_endpoint,
        "CARTRIDGE_STORAGE_S3_ENDPOINT",
        optional
    );
    env_override!(
        config,
        storage.pool_max_size,
        "CARTRIDGE_STORAGE_POOL_MAX_SIZE",
        parse
    );
    env_override!(
        config,
        storage.pool_connect_timeout,
        "CARTRIDGE_STORAGE_POOL_CONNECT_TIMEOUT",
        parse
    );
    env_override!(
        config,
        storage.pool_idle_timeout,
        "CARTRIDGE_STORAGE_POOL_IDLE_TIMEOUT",
        optional_parse
    );

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
        let toml_content = r#"
[common]
env_id = "connect4"
"#;
        let config: CentralConfig = toml::from_str(toml_content).unwrap();
        assert_eq!(config.common.env_id, "connect4");
        assert_eq!(config.common.data_dir, "./data");
        assert_eq!(config.actor.actor_id, "actor-1");
    }

    #[test]
    fn test_storage_config_defaults() {
        let config = CentralConfig::default();
        assert_eq!(config.storage.model_backend, "filesystem");
        assert_eq!(
            config.storage.postgres_url,
            Some("postgresql://cartridge:cartridge@localhost:5432/cartridge".to_string())
        );
        assert!(config.storage.s3_bucket.is_none());
        assert!(config.storage.s3_endpoint.is_none());
    }

    #[test]
    fn test_storage_config_from_toml() {
        let toml_content = r#"
[storage]
model_backend = "s3"
postgres_url = "postgresql://user:pass@localhost:5432/cartridge"
s3_bucket = "my-bucket"
s3_endpoint = "http://minio:9000"
"#;
        let config: CentralConfig = toml::from_str(toml_content).unwrap();
        assert_eq!(config.storage.model_backend, "s3");
        assert_eq!(
            config.storage.postgres_url,
            Some("postgresql://user:pass@localhost:5432/cartridge".to_string())
        );
        assert_eq!(config.storage.s3_bucket, Some("my-bucket".to_string()));
        assert_eq!(
            config.storage.s3_endpoint,
            Some("http://minio:9000".to_string())
        );
    }

    #[test]
    fn test_storage_env_overrides() {
        std::env::set_var("CARTRIDGE_STORAGE_MODEL_BACKEND", "s3");
        std::env::set_var(
            "CARTRIDGE_STORAGE_POSTGRES_URL",
            "postgresql://test@localhost/db",
        );

        let config = load_config();
        assert_eq!(config.storage.model_backend, "s3");
        assert_eq!(
            config.storage.postgres_url,
            Some("postgresql://test@localhost/db".to_string())
        );

        std::env::remove_var("CARTRIDGE_STORAGE_MODEL_BACKEND");
        std::env::remove_var("CARTRIDGE_STORAGE_POSTGRES_URL");
    }
}
