//! PostgreSQL backend for replay buffer storage.
//!
//! This is the only storage backend for the actor.
//! Supports concurrent writes from multiple actor instances.

use anyhow::Result;
use async_trait::async_trait;
use engine_core::GameMetadata;
use tokio::sync::Mutex;
use tokio_postgres::{Client, NoTls};

use super::{ReplayStore, Transition};

/// SQL schema embedded at compile time from the shared schema file.
const SCHEMA_SQL: &str = include_str!("../../../sql/schema.sql");

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

        // Execute each statement from the shared schema file
        // Split on semicolons and execute non-empty statements
        for statement in SCHEMA_SQL.split(';') {
            let stmt = statement.trim();
            // Skip empty statements and comments
            if stmt.is_empty() || stmt.starts_with("--") {
                continue;
            }
            client.execute(stmt, &[]).await?;
        }

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

    async fn clear(&self) -> Result<()> {
        let client = self.client.lock().await;
        client.execute("DELETE FROM transitions", &[]).await?;
        Ok(())
    }
}
