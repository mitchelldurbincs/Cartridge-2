//! Storage backend abstraction for replay buffer.
//!
//! This module provides pluggable storage backends that work both locally
//! (SQLite) and in Kubernetes (PostgreSQL).
//!
//! # Usage
//!
//! ```rust,ignore
//! use actor::storage::{create_replay_store, ReplayStore};
//!
//! // Uses config to select backend (sqlite or postgres)
//! let store = create_replay_store(&config).await?;
//!
//! // Store transitions
//! store.store_batch(&transitions).await?;
//! ```

mod sqlite;

#[cfg(feature = "postgres")]
mod postgres;

pub use sqlite::SqliteReplayStore;

#[cfg(feature = "postgres")]
pub use postgres::PostgresReplayStore;

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
/// from multiple actor instances (for PostgreSQL backend).
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

/// Storage backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StorageBackend {
    #[default]
    Sqlite,
    #[cfg(feature = "postgres")]
    Postgres,
}

impl std::str::FromStr for StorageBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sqlite" => Ok(StorageBackend::Sqlite),
            #[cfg(feature = "postgres")]
            "postgres" | "postgresql" => Ok(StorageBackend::Postgres),
            #[cfg(not(feature = "postgres"))]
            "postgres" | "postgresql" => {
                anyhow::bail!("PostgreSQL support not compiled. Rebuild with --features postgres")
            }
            _ => anyhow::bail!(
                "Unknown storage backend: {}. Supported: sqlite, postgres",
                s
            ),
        }
    }
}

/// Configuration for creating a replay store
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub backend: StorageBackend,
    /// Path to SQLite database (for sqlite backend)
    pub sqlite_path: Option<String>,
    /// PostgreSQL connection string (for postgres backend)
    #[cfg(feature = "postgres")]
    pub postgres_url: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::default(),
            sqlite_path: Some("./data/replay.db".to_string()),
            #[cfg(feature = "postgres")]
            postgres_url: None,
        }
    }
}

/// Create a replay store based on configuration
pub async fn create_replay_store(config: &StorageConfig) -> Result<Box<dyn ReplayStore>> {
    match config.backend {
        StorageBackend::Sqlite => {
            let path = config
                .sqlite_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("SQLite path required for sqlite backend"))?;
            let store = SqliteReplayStore::new(path)?;
            Ok(Box::new(store))
        }
        #[cfg(feature = "postgres")]
        StorageBackend::Postgres => {
            let url = config
                .postgres_url
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("PostgreSQL URL required for postgres backend"))?;
            let store = PostgresReplayStore::new(url).await?;
            Ok(Box::new(store))
        }
    }
}
