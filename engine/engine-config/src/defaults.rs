//! Default configuration values.
//!
//! This module provides a single source of truth for all default configuration
//! values used across the Cartridge2 system.

// Common defaults
pub const DATA_DIR: &str = "./data";
pub const ENV_ID: &str = "tictactoe";
pub const LOG_LEVEL: &str = "info";

// Training defaults
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

// Evaluation defaults
pub const EVAL_INTERVAL: i32 = 1;
pub const EVAL_GAMES: i32 = 50;

// Actor defaults
pub const ACTOR_ID: &str = "actor-1";
pub const MAX_EPISODES: i32 = -1;
pub const EPISODE_TIMEOUT_SECS: u64 = 30;
pub const FLUSH_INTERVAL_SECS: u64 = 5;
pub const LOG_INTERVAL: u32 = 50;

// Web defaults
pub const HOST: &str = "0.0.0.0";
pub const PORT: u16 = 8080;

// MCTS defaults
pub const NUM_SIMULATIONS: u32 = 800;
pub const C_PUCT: f64 = 1.4;
pub const TEMPERATURE: f64 = 1.0;
pub const DIRICHLET_ALPHA: f64 = 0.3;
pub const DIRICHLET_WEIGHT: f64 = 0.25;
pub const EVAL_BATCH_SIZE: usize = 32;
pub const ONNX_INTRA_THREADS: usize = 1;

// Storage defaults
pub const MODEL_BACKEND: &str = "filesystem";
pub const POSTGRES_URL: &str = "postgresql://cartridge:cartridge@localhost:5432/cartridge";
pub const POOL_MAX_SIZE: usize = 16;
pub const POOL_CONNECT_TIMEOUT: u64 = 30;
pub const POOL_IDLE_TIMEOUT: u64 = 300;
