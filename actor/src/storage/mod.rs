//! Storage backend for replay buffer (PostgreSQL).
//!
//! This module provides the PostgreSQL storage backend for the replay buffer.
//!
//! # Usage
//!
//! ```rust,ignore
//! use actor::storage::{create_replay_store, ReplayStore};
//!
//! let store = create_replay_store(&config).await?;
//!
//! // Store transitions
//! store.store_batch(&transitions).await?;
//! ```

mod postgres;

pub use postgres::{PoolConfig, PostgresReplayStore};

use anyhow::Result;
use async_trait::async_trait;
use engine_core::GameMetadata;

/// A single transition from one game state to the next
#[derive(Debug, Clone)]
pub struct Transition {
    pub id: String,
    pub env_id: String,
    pub episode_id: String,
    pub step_number: u32,
    pub state: Vec<u8>,
    pub action: Vec<u8>,
    pub next_state: Vec<u8>,
    pub observation: Vec<u8>,
    pub next_observation: Vec<u8>,
    pub reward: f32,
    pub done: bool,
    pub timestamp: u64,
    /// MCTS policy probabilities for training (stored as f32 bytes)
    pub policy_probs: Vec<u8>,
    /// MCTS value estimate from root
    pub mcts_value: f32,
    /// Final game outcome from this player's perspective (+1 win, -1 loss, 0 draw)
    /// This is backfilled after the episode ends
    pub game_outcome: Option<f32>,
}

/// Abstract interface for replay buffer storage.
///
/// Implementations must be thread-safe and support concurrent writes
/// from multiple actor instances.
#[async_trait]
#[allow(dead_code)]
pub trait ReplayStore: Send + Sync {
    /// Store a single transition in the replay buffer
    async fn store(&self, transition: &Transition) -> Result<()>;

    /// Store multiple transitions in a batch (more efficient)
    async fn store_batch(&self, transitions: &[Transition]) -> Result<()>;

    /// Get the total number of transitions in the buffer
    async fn count(&self) -> Result<usize>;

    /// Store or update game metadata (upsert)
    async fn store_metadata(&self, metadata: &GameMetadata) -> Result<()>;

    /// Clear all transitions (preserves metadata)
    async fn clear(&self) -> Result<()>;
}

/// Configuration for creating a replay store
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// PostgreSQL connection string
    pub postgres_url: String,
    /// Connection pool configuration
    pub pool_config: PoolConfig,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            postgres_url: std::env::var("CARTRIDGE_STORAGE_POSTGRES_URL").unwrap_or_else(|_| {
                "postgresql://cartridge:cartridge@localhost:5432/cartridge".to_string()
            }),
            pool_config: PoolConfig::default(),
        }
    }
}

/// Create a replay store based on configuration
pub async fn create_replay_store(config: &StorageConfig) -> Result<Box<dyn ReplayStore>> {
    let store =
        PostgresReplayStore::with_pool_config(&config.postgres_url, config.pool_config.clone())
            .await?;
    Ok(Box::new(store))
}
