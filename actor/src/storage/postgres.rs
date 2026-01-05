//! PostgreSQL backend for replay buffer storage.
//!
//! This backend is designed for Kubernetes deployments where multiple
//! actors need concurrent access to the replay buffer.
//!
//! Requires the `postgres` feature to be enabled.

use anyhow::Result;
use async_trait::async_trait;
use engine_core::GameMetadata;
use tokio::sync::Mutex;
use tokio_postgres::{Client, NoTls};

use super::{ReplayStore, StoredGameMetadata, Transition};

/// PostgreSQL-backed replay buffer implementation.
///
/// This backend supports:
/// - Concurrent writes from multiple actors
/// - Connection pooling (via external connection manager)
/// - Efficient batch inserts with ON CONFLICT handling
///
/// The client is wrapped in a Mutex to allow transactions which require
/// mutable access while maintaining the `&self` trait signature.
pub struct PostgresReplayStore {
    client: Mutex<Client>,
    // Keep the connection task alive
    _connection_handle: tokio::task::JoinHandle<()>,
}

impl PostgresReplayStore {
    /// Create a new PostgreSQL replay store.
    ///
    /// # Arguments
    /// * `connection_string` - PostgreSQL connection URL
    ///   Format: `postgresql://user:password@host:port/database`
    pub async fn new(connection_string: &str) -> Result<Self> {
        let (client, connection) = tokio_postgres::connect(connection_string, NoTls).await?;

        // Spawn the connection handler
        let handle = tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::error!("PostgreSQL connection error: {}", e);
            }
        });

        let store = Self {
            client: Mutex::new(client),
            _connection_handle: handle,
        };

        // Ensure schema exists
        store.ensure_schema().await?;

        Ok(store)
    }

    async fn ensure_schema(&self) -> Result<()> {
        let client = self.client.lock().await;

        // Create transitions table (PostgreSQL syntax)
        client
            .execute(
                "CREATE TABLE IF NOT EXISTS transitions (
                    id TEXT PRIMARY KEY,
                    env_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    state BYTEA NOT NULL,
                    action BYTEA NOT NULL,
                    next_state BYTEA NOT NULL,
                    observation BYTEA NOT NULL,
                    next_observation BYTEA NOT NULL,
                    reward REAL NOT NULL,
                    done BOOLEAN NOT NULL,
                    timestamp BIGINT NOT NULL,
                    policy_probs BYTEA,
                    mcts_value REAL DEFAULT 0.0,
                    game_outcome REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )",
                &[],
            )
            .await?;

        // Create indices
        client
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON transitions(timestamp)",
                &[],
            )
            .await?;

        client
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_transitions_episode ON transitions(episode_id)",
                &[],
            )
            .await?;

        client
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_transitions_env_id ON transitions(env_id)",
                &[],
            )
            .await?;

        // Create game_metadata table
        client
            .execute(
                "CREATE TABLE IF NOT EXISTS game_metadata (
                    env_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    board_width INTEGER NOT NULL,
                    board_height INTEGER NOT NULL,
                    num_actions INTEGER NOT NULL,
                    obs_size INTEGER NOT NULL,
                    legal_mask_offset INTEGER NOT NULL,
                    player_count INTEGER NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )",
                &[],
            )
            .await?;

        tracing::info!("PostgreSQL schema validated/created");
        Ok(())
    }
}

#[async_trait]
impl ReplayStore for PostgresReplayStore {
    async fn store(&self, transition: &Transition) -> Result<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "INSERT INTO transitions
                 (id, env_id, episode_id, step_number, state, action, next_state,
                  observation, next_observation, reward, done, timestamp,
                  policy_probs, mcts_value, game_outcome)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                 ON CONFLICT (id) DO UPDATE SET
                     game_outcome = EXCLUDED.game_outcome,
                     mcts_value = EXCLUDED.mcts_value",
                &[
                    &transition.id,
                    &transition.env_id,
                    &transition.episode_id,
                    &(transition.step_number as i32),
                    &transition.state,
                    &transition.action,
                    &transition.next_state,
                    &transition.observation,
                    &transition.next_observation,
                    &transition.reward,
                    &transition.done,
                    &(transition.timestamp as i64),
                    &transition.policy_probs,
                    &transition.mcts_value,
                    &transition.game_outcome,
                ],
            )
            .await?;
        Ok(())
    }

    async fn store_batch(&self, transitions: &[Transition]) -> Result<()> {
        // Use a transaction for batch insert
        let mut client = self.client.lock().await;
        let transaction = client.transaction().await?;

        // Prepare statement once
        let stmt = transaction
            .prepare(
                "INSERT INTO transitions
                 (id, env_id, episode_id, step_number, state, action, next_state,
                  observation, next_observation, reward, done, timestamp,
                  policy_probs, mcts_value, game_outcome)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                 ON CONFLICT (id) DO UPDATE SET
                     game_outcome = EXCLUDED.game_outcome,
                     mcts_value = EXCLUDED.mcts_value",
            )
            .await?;

        for transition in transitions {
            transaction
                .execute(
                    &stmt,
                    &[
                        &transition.id,
                        &transition.env_id,
                        &transition.episode_id,
                        &(transition.step_number as i32),
                        &transition.state,
                        &transition.action,
                        &transition.next_state,
                        &transition.observation,
                        &transition.next_observation,
                        &transition.reward,
                        &transition.done,
                        &(transition.timestamp as i64),
                        &transition.policy_probs,
                        &transition.mcts_value,
                        &transition.game_outcome,
                    ],
                )
                .await?;
        }

        transaction.commit().await?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let client = self.client.lock().await;
        let row = client
            .query_one("SELECT COUNT(*) FROM transitions", &[])
            .await?;
        let count: i64 = row.get(0);
        Ok(count as usize)
    }

    async fn store_metadata(&self, metadata: &GameMetadata) -> Result<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "INSERT INTO game_metadata
                 (env_id, display_name, board_width, board_height, num_actions,
                  obs_size, legal_mask_offset, player_count, updated_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                 ON CONFLICT (env_id) DO UPDATE SET
                     display_name = EXCLUDED.display_name,
                     board_width = EXCLUDED.board_width,
                     board_height = EXCLUDED.board_height,
                     num_actions = EXCLUDED.num_actions,
                     obs_size = EXCLUDED.obs_size,
                     legal_mask_offset = EXCLUDED.legal_mask_offset,
                     player_count = EXCLUDED.player_count,
                     updated_at = CURRENT_TIMESTAMP",
                &[
                    &metadata.env_id,
                    &metadata.display_name,
                    &(metadata.board_width as i32),
                    &(metadata.board_height as i32),
                    &(metadata.num_actions as i32),
                    &(metadata.obs_size as i32),
                    &(metadata.legal_mask_offset as i32),
                    &(metadata.player_count as i32),
                ],
            )
            .await?;
        Ok(())
    }

    async fn get_metadata(&self, env_id: &str) -> Result<Option<StoredGameMetadata>> {
        let client = self.client.lock().await;
        let rows = client
            .query(
                "SELECT env_id, display_name, board_width, board_height, num_actions,
                        obs_size, legal_mask_offset, player_count
                 FROM game_metadata WHERE env_id = $1",
                &[&env_id],
            )
            .await?;

        if rows.is_empty() {
            return Ok(None);
        }

        let row = &rows[0];
        Ok(Some(StoredGameMetadata {
            env_id: row.get(0),
            display_name: row.get(1),
            board_width: row.get::<_, i32>(2) as usize,
            board_height: row.get::<_, i32>(3) as usize,
            num_actions: row.get::<_, i32>(4) as usize,
            obs_size: row.get::<_, i32>(5) as usize,
            legal_mask_offset: row.get::<_, i32>(6) as usize,
            player_count: row.get::<_, i32>(7) as usize,
        }))
    }

    async fn clear(&self) -> Result<()> {
        let client = self.client.lock().await;
        client.execute("DELETE FROM transitions", &[]).await?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        // Connection is closed when the client is dropped
        // The connection handle will complete
        Ok(())
    }
}
