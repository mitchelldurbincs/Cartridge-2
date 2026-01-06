//! SQLite backend for replay buffer storage.
//!
//! This is the default backend for local development and single-machine training.

use anyhow::Result;
use async_trait::async_trait;
use engine_core::GameMetadata;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Mutex;

use super::{ReplayStore, Transition};

/// SQLite-based replay buffer implementation.
///
/// Uses a Mutex for thread-safety since rusqlite Connection is not Sync.
/// For high-throughput scenarios, consider the PostgreSQL backend.
pub struct SqliteReplayStore {
    conn: Mutex<Connection>,
}

impl SqliteReplayStore {
    /// Create a new SQLite replay store, initializing the database if needed.
    pub fn new(db_path: &str) -> Result<Self> {
        // Create parent directories if they don't exist
        if let Some(parent) = Path::new(db_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;

        // Create the transitions table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS transitions (
                id TEXT PRIMARY KEY,
                env_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                state BLOB NOT NULL,
                action BLOB NOT NULL,
                next_state BLOB NOT NULL,
                observation BLOB NOT NULL,
                next_observation BLOB NOT NULL,
                reward REAL NOT NULL,
                done INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                policy_probs BLOB,
                mcts_value REAL DEFAULT 0.0,
                game_outcome REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Create index for efficient sampling
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON transitions(timestamp)",
            [],
        )?;

        // Create index for episode queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_transitions_episode ON transitions(episode_id)",
            [],
        )?;

        // Create game_metadata table to make database self-describing
        conn.execute(
            "CREATE TABLE IF NOT EXISTS game_metadata (
                env_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                board_width INTEGER NOT NULL,
                board_height INTEGER NOT NULL,
                num_actions INTEGER NOT NULL,
                obs_size INTEGER NOT NULL,
                legal_mask_offset INTEGER NOT NULL,
                player_count INTEGER NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

#[async_trait]
impl ReplayStore for SqliteReplayStore {
    async fn store(&self, transition: &Transition) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        conn.execute(
            "INSERT OR REPLACE INTO transitions
             (id, env_id, episode_id, step_number, state, action, next_state,
              observation, next_observation, reward, done, timestamp, policy_probs, mcts_value, game_outcome)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                transition.id,
                transition.env_id,
                transition.episode_id,
                transition.step_number,
                transition.state,
                transition.action,
                transition.next_state,
                transition.observation,
                transition.next_observation,
                transition.reward,
                transition.done as i32,
                transition.timestamp as i64,
                transition.policy_probs,
                transition.mcts_value,
                transition.game_outcome,
            ],
        )?;
        Ok(())
    }

    async fn store_batch(&self, transitions: &[Transition]) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        let tx = conn.unchecked_transaction()?;

        // Prepare the INSERT statement once and reuse it for the whole batch
        let mut stmt = tx.prepare_cached(
            "INSERT OR REPLACE INTO transitions
             (id, env_id, episode_id, step_number, state, action, next_state,
              observation, next_observation, reward, done, timestamp, policy_probs, mcts_value, game_outcome)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        )?;

        for transition in transitions {
            stmt.execute(params![
                transition.id,
                transition.env_id,
                transition.episode_id,
                transition.step_number,
                transition.state,
                transition.action,
                transition.next_state,
                transition.observation,
                transition.next_observation,
                transition.reward,
                transition.done as i32,
                transition.timestamp as i64,
                transition.policy_probs,
                transition.mcts_value,
                transition.game_outcome,
            ])?;
        }

        // Drop stmt before commit to release borrow on tx
        drop(stmt);
        tx.commit()?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM transitions", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    async fn store_metadata(&self, metadata: &GameMetadata) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        conn.execute(
            "INSERT OR REPLACE INTO game_metadata
             (env_id, display_name, board_width, board_height, num_actions,
              obs_size, legal_mask_offset, player_count, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, CURRENT_TIMESTAMP)",
            params![
                metadata.env_id,
                metadata.display_name,
                metadata.board_width,
                metadata.board_height,
                metadata.num_actions,
                metadata.obs_size,
                metadata.legal_mask_offset,
                metadata.player_count,
            ],
        )?;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        conn.execute("DELETE FROM transitions", [])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_transition(id: &str, step: u32) -> Transition {
        Transition {
            id: id.to_string(),
            env_id: "test_env".to_string(),
            episode_id: "test_episode".to_string(),
            step_number: step,
            state: vec![1, 2, 3, 4],
            action: vec![0],
            next_state: vec![5, 6, 7, 8],
            observation: vec![1, 2, 3],
            next_observation: vec![4, 5, 6],
            reward: 1.0,
            done: false,
            timestamp: 1234567890,
            policy_probs: vec![],
            mcts_value: 0.0,
            game_outcome: None,
        }
    }

    #[tokio::test]
    async fn test_create_store() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteReplayStore::new(db_path.to_str().unwrap());
        assert!(store.is_ok());
    }

    #[tokio::test]
    async fn test_store_and_count() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteReplayStore::new(db_path.to_str().unwrap()).unwrap();

        // Store a transition
        let transition = create_test_transition("t1", 0);
        store.store(&transition).await.unwrap();

        // Check count
        assert_eq!(store.count().await.unwrap(), 1);

        // Store another
        let transition2 = create_test_transition("t2", 1);
        store.store(&transition2).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_store_batch() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteReplayStore::new(db_path.to_str().unwrap()).unwrap();

        let transitions: Vec<Transition> = (0..10)
            .map(|i| create_test_transition(&format!("t{}", i), i as u32))
            .collect();

        store.store_batch(&transitions).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 10);
    }

    #[tokio::test]
    async fn test_clear() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteReplayStore::new(db_path.to_str().unwrap()).unwrap();

        let transitions: Vec<Transition> = (0..10)
            .map(|i| create_test_transition(&format!("t{}", i), i as u32))
            .collect();
        store.store_batch(&transitions).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 10);

        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }
}
