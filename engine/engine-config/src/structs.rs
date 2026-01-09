//! Configuration struct definitions.
//!
//! All config structs with serde deserialization support and default values.

use crate::defaults;
use serde::Deserialize;

// ============================================================================
// Serde default functions (required for #[serde(default = "...")])
// These are thin wrappers around constants
// ============================================================================

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
fn d_onnx_intra_threads() -> usize {
    defaults::ONNX_INTRA_THREADS
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

// ============================================================================
// Configuration Structs
// ============================================================================

/// Root configuration structure matching config.toml
#[derive(Debug, Deserialize, Default, Clone)]
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

/// Common configuration shared by all components
#[derive(Debug, Deserialize, Clone)]
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

/// Training configuration for the trainer
#[derive(Debug, Deserialize, Clone)]
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

/// Evaluation configuration
#[derive(Debug, Deserialize, Clone)]
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

/// Actor (self-play) configuration
#[derive(Debug, Deserialize, Clone)]
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

/// Web server configuration
#[derive(Debug, Deserialize, Clone)]
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

/// MCTS (Monte Carlo Tree Search) configuration
#[derive(Debug, Deserialize, Clone)]
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
    #[serde(default = "d_onnx_intra_threads")]
    pub onnx_intra_threads: usize,
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
            onnx_intra_threads: defaults::ONNX_INTRA_THREADS,
        }
    }
}

/// Storage backend configuration
#[derive(Debug, Deserialize, Clone)]
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
