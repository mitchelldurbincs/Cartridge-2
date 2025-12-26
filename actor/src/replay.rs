//! SQLite-based replay buffer for storing transitions
//!
//! This module provides a SQLite-backed replay buffer for storing
//! game transitions. It replaces the gRPC replay service with a
//! simple file-based approach for the MVP.
//!
//! Also stores game metadata to make the database self-describing,
//! allowing the trainer to read configuration without hardcoding.

use anyhow::Result;
use engine_core::GameMetadata;
use rusqlite::{params, Connection};
use std::path::Path;

/// Stored game metadata for self-describing replay databases
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields are part of the public API and may be read by consumers
pub struct StoredGameMetadata {
    pub env_id: String,
    pub display_name: String,
    pub board_width: usize,
    pub board_height: usize,
    pub num_actions: usize,
    pub obs_size: usize,
    pub legal_mask_offset: usize,
    pub player_count: usize,
}

impl From<&GameMetadata> for StoredGameMetadata {
    fn from(meta: &GameMetadata) -> Self {
        Self {
            env_id: meta.env_id.clone(),
            display_name: meta.display_name.clone(),
            board_width: meta.board_width,
            board_height: meta.board_height,
            num_actions: meta.num_actions,
            obs_size: meta.obs_size,
            legal_mask_offset: meta.legal_mask_offset,
            player_count: meta.player_count,
        }
    }
}

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

/// SQLite-based replay buffer
pub struct ReplayBuffer {
    conn: Connection,
}

impl ReplayBuffer {
    /// Create a new replay buffer, initializing the database if needed
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

        Ok(Self { conn })
    }

    /// Store a transition in the replay buffer
    #[allow(dead_code)] // Used in tests
    pub fn store(&self, transition: &Transition) -> Result<()> {
        self.conn.execute(
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

    /// Store multiple transitions in a batch (single transaction for efficiency)
    pub fn store_batch(&self, transitions: &[Transition]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;

        // Prepare the INSERT statement once and reuse it for the whole batch.
        // Preparing per-row was adding measurable overhead as the replay buffer grows.
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

    /// Sample random transitions for training.
    ///
    /// Uses efficient reservoir sampling instead of ORDER BY RANDOM() which
    /// would require sorting the entire table (O(n log n)). This approach:
    /// 1. Gets the total count and max rowid
    /// 2. Generates random rowids and fetches those rows
    /// 3. Falls back gracefully if rows were deleted
    #[allow(dead_code)]
    pub fn sample(&self, batch_size: usize) -> Result<Vec<Transition>> {
        use rand::Rng;

        // Get count and max rowid for efficient random sampling
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM transitions", [], |row| row.get(0))?;

        if count == 0 {
            return Ok(Vec::new());
        }

        // If requesting more than available, just return all
        if batch_size >= count as usize {
            return self.sample_all();
        }

        // Get min and max rowid for random generation
        let (min_rowid, max_rowid): (i64, i64) = self.conn.query_row(
            "SELECT MIN(rowid), MAX(rowid) FROM transitions",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        let mut rng = rand::thread_rng();
        let mut results = Vec::with_capacity(batch_size);
        let mut attempts = 0;
        let max_attempts = batch_size * 10; // Avoid infinite loop if table is sparse

        let mut stmt = self.conn.prepare(
            "SELECT id, env_id, episode_id, step_number, state, action, next_state,
                    observation, next_observation, reward, done, timestamp, policy_probs, mcts_value, game_outcome
             FROM transitions
             WHERE rowid = ?1",
        )?;

        // Track seen rowids to avoid duplicates
        let mut seen = std::collections::HashSet::with_capacity(batch_size);

        while results.len() < batch_size && attempts < max_attempts {
            attempts += 1;
            let rowid = rng.gen_range(min_rowid..=max_rowid);

            if seen.contains(&rowid) {
                continue;
            }
            seen.insert(rowid);

            // Try to fetch this rowid (may not exist if deleted)
            if let Ok(transition) = stmt.query_row([rowid], |row| {
                Ok(Transition {
                    id: row.get(0)?,
                    env_id: row.get(1)?,
                    episode_id: row.get(2)?,
                    step_number: row.get(3)?,
                    state: row.get(4)?,
                    action: row.get(5)?,
                    next_state: row.get(6)?,
                    observation: row.get(7)?,
                    next_observation: row.get(8)?,
                    reward: row.get(9)?,
                    done: row.get::<_, i32>(10)? != 0,
                    timestamp: row.get::<_, i64>(11)? as u64,
                    policy_probs: row.get::<_, Option<Vec<u8>>>(12)?.unwrap_or_default(),
                    mcts_value: row.get::<_, Option<f64>>(13)?.unwrap_or(0.0) as f32,
                    game_outcome: row.get::<_, Option<f64>>(14)?.map(|v| v as f32),
                })
            }) {
                results.push(transition);
            }
        }

        Ok(results)
    }

    /// Helper to fetch all transitions (used when batch_size >= count)
    fn sample_all(&self) -> Result<Vec<Transition>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, env_id, episode_id, step_number, state, action, next_state,
                    observation, next_observation, reward, done, timestamp, policy_probs, mcts_value, game_outcome
             FROM transitions",
        )?;

        let transitions = stmt
            .query_map([], |row| {
                Ok(Transition {
                    id: row.get(0)?,
                    env_id: row.get(1)?,
                    episode_id: row.get(2)?,
                    step_number: row.get(3)?,
                    state: row.get(4)?,
                    action: row.get(5)?,
                    next_state: row.get(6)?,
                    observation: row.get(7)?,
                    next_observation: row.get(8)?,
                    reward: row.get(9)?,
                    done: row.get::<_, i32>(10)? != 0,
                    timestamp: row.get::<_, i64>(11)? as u64,
                    policy_probs: row.get::<_, Option<Vec<u8>>>(12)?.unwrap_or_default(),
                    mcts_value: row.get::<_, Option<f64>>(13)?.unwrap_or(0.0) as f32,
                    game_outcome: row.get::<_, Option<f64>>(14)?.map(|v| v as f32),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(transitions)
    }

    /// Get the total number of transitions in the buffer
    #[allow(dead_code)]
    pub fn count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM transitions", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Delete old transitions to maintain a sliding window
    #[allow(dead_code)]
    pub fn cleanup(&self, window_size: usize) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM transitions
             WHERE id NOT IN (
                 SELECT id FROM transitions
                 ORDER BY created_at DESC
                 LIMIT ?1
             )",
            [window_size],
        )?;
        Ok(deleted)
    }

    /// Clear all transitions (for testing)
    #[allow(dead_code)]
    pub fn clear(&self) -> Result<()> {
        self.conn.execute("DELETE FROM transitions", [])?;
        Ok(())
    }

    /// Store or update game metadata (upsert)
    ///
    /// This makes the replay database self-describing, allowing the trainer
    /// to read configuration without hardcoding values.
    pub fn store_metadata(&self, metadata: &GameMetadata) -> Result<()> {
        self.conn.execute(
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

    /// Get metadata for a specific game
    #[allow(dead_code)]
    pub fn get_metadata(&self, env_id: &str) -> Result<Option<StoredGameMetadata>> {
        let mut stmt = self.conn.prepare(
            "SELECT env_id, display_name, board_width, board_height, num_actions,
                    obs_size, legal_mask_offset, player_count
             FROM game_metadata WHERE env_id = ?1",
        )?;

        let result = stmt.query_row([env_id], |row| {
            Ok(StoredGameMetadata {
                env_id: row.get(0)?,
                display_name: row.get(1)?,
                board_width: row.get::<_, i64>(2)? as usize,
                board_height: row.get::<_, i64>(3)? as usize,
                num_actions: row.get::<_, i64>(4)? as usize,
                obs_size: row.get::<_, i64>(5)? as usize,
                legal_mask_offset: row.get::<_, i64>(6)? as usize,
                player_count: row.get::<_, i64>(7)? as usize,
            })
        });

        match result {
            Ok(meta) => Ok(Some(meta)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get all stored game metadata
    #[allow(dead_code)]
    pub fn list_metadata(&self) -> Result<Vec<StoredGameMetadata>> {
        let mut stmt = self.conn.prepare(
            "SELECT env_id, display_name, board_width, board_height, num_actions,
                    obs_size, legal_mask_offset, player_count
             FROM game_metadata",
        )?;

        let metadata = stmt
            .query_map([], |row| {
                Ok(StoredGameMetadata {
                    env_id: row.get(0)?,
                    display_name: row.get(1)?,
                    board_width: row.get::<_, i64>(2)? as usize,
                    board_height: row.get::<_, i64>(3)? as usize,
                    num_actions: row.get::<_, i64>(4)? as usize,
                    obs_size: row.get::<_, i64>(5)? as usize,
                    legal_mask_offset: row.get::<_, i64>(6)? as usize,
                    player_count: row.get::<_, i64>(7)? as usize,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(metadata)
    }

    /// Update game_outcome for all transitions in an episode.
    ///
    /// After an episode ends, we know the final outcome (win/loss/draw).
    /// This method backfills the outcome to all transitions, alternating
    /// the sign for each step to represent the outcome from each player's
    /// perspective (since players alternate turns).
    ///
    /// Note: The actor now backfills outcomes in-memory before store_batch(),
    /// so this method is primarily for debugging/reprocessing existing data.
    ///
    /// # Arguments
    /// * `episode_id` - The episode to update
    /// * `final_outcome` - The outcome from the perspective of the player who moved last
    ///   (+1.0 for win, -1.0 for loss, 0.0 for draw)
    /// * `total_steps` - Total number of steps in the episode (used to compute perspective)
    #[allow(dead_code)]
    pub fn update_episode_outcomes(
        &self,
        episode_id: &str,
        final_outcome: f32,
        total_steps: u32,
    ) -> Result<usize> {
        // For two-player alternating games:
        // - The last player to move gets the final_outcome directly at step (total_steps - 1)
        // - The previous player gets -final_outcome at step (total_steps - 2)
        // - And so on, alternating signs
        //
        // If total_steps is odd, the last mover is player 1 (started first)
        // If total_steps is even, the last mover is player 2
        //
        // Formula: outcome_at_step = final_outcome * (-1)^(total_steps - 1 - step_number)
        //
        // We'll update each step with the appropriate outcome based on whose perspective
        // that transition was recorded from.

        let mut updated = 0;

        // Get all transitions for this episode ordered by step
        let mut stmt = self.conn.prepare(
            "SELECT id, step_number FROM transitions WHERE episode_id = ? ORDER BY step_number",
        )?;

        let transitions: Vec<(String, u32)> = stmt
            .query_map([episode_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Update each transition with the outcome from that player's perspective
        for (id, step_number) in transitions {
            // Calculate the outcome from this step's player's perspective
            // At step N, the player who moved sees the game from their viewpoint
            // If they eventually won (final_outcome from winner's view), we propagate that back
            let steps_from_end = total_steps.saturating_sub(1).saturating_sub(step_number);
            let sign = if steps_from_end % 2 == 0 { 1.0 } else { -1.0 };
            let outcome = final_outcome * sign;

            self.conn.execute(
                "UPDATE transitions SET game_outcome = ? WHERE id = ?",
                params![outcome, id],
            )?;
            updated += 1;
        }

        Ok(updated)
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

    #[test]
    fn test_create_replay_buffer() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let buffer = ReplayBuffer::new(db_path.to_str().unwrap());
        assert!(buffer.is_ok());
    }

    #[test]
    fn test_store_and_count() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let buffer = ReplayBuffer::new(db_path.to_str().unwrap()).unwrap();

        // Store a transition
        let transition = create_test_transition("t1", 0);
        buffer.store(&transition).unwrap();

        // Check count
        assert_eq!(buffer.count().unwrap(), 1);

        // Store another
        let transition2 = create_test_transition("t2", 1);
        buffer.store(&transition2).unwrap();

        assert_eq!(buffer.count().unwrap(), 2);
    }

    #[test]
    fn test_store_batch() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let buffer = ReplayBuffer::new(db_path.to_str().unwrap()).unwrap();

        let transitions: Vec<Transition> = (0..10)
            .map(|i| create_test_transition(&format!("t{}", i), i as u32))
            .collect();

        buffer.store_batch(&transitions).unwrap();

        assert_eq!(buffer.count().unwrap(), 10);
    }

    #[test]
    fn test_sample() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let buffer = ReplayBuffer::new(db_path.to_str().unwrap()).unwrap();

        // Store 100 transitions
        let transitions: Vec<Transition> = (0..100)
            .map(|i| create_test_transition(&format!("t{}", i), i as u32))
            .collect();
        buffer.store_batch(&transitions).unwrap();

        // Sample 10
        let sampled = buffer.sample(10).unwrap();
        assert_eq!(sampled.len(), 10);

        // Verify each sampled transition has valid data
        for t in &sampled {
            assert!(!t.id.is_empty());
            assert_eq!(t.env_id, "test_env");
        }
    }

    #[test]
    fn test_sample_more_than_available() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let buffer = ReplayBuffer::new(db_path.to_str().unwrap()).unwrap();

        // Store 5 transitions
        let transitions: Vec<Transition> = (0..5)
            .map(|i| create_test_transition(&format!("t{}", i), i as u32))
            .collect();
        buffer.store_batch(&transitions).unwrap();

        // Try to sample 10 - should return only 5
        let sampled = buffer.sample(10).unwrap();
        assert_eq!(sampled.len(), 5);
    }

    #[test]
    fn test_cleanup() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let buffer = ReplayBuffer::new(db_path.to_str().unwrap()).unwrap();

        // Store 100 transitions
        let transitions: Vec<Transition> = (0..100)
            .map(|i| create_test_transition(&format!("t{}", i), i as u32))
            .collect();
        buffer.store_batch(&transitions).unwrap();

        // Cleanup to keep only 50
        let deleted = buffer.cleanup(50).unwrap();
        assert_eq!(deleted, 50);
        assert_eq!(buffer.count().unwrap(), 50);
    }

    #[test]
    fn test_clear() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let buffer = ReplayBuffer::new(db_path.to_str().unwrap()).unwrap();

        // Store some transitions
        let transitions: Vec<Transition> = (0..10)
            .map(|i| create_test_transition(&format!("t{}", i), i as u32))
            .collect();
        buffer.store_batch(&transitions).unwrap();

        assert_eq!(buffer.count().unwrap(), 10);

        // Clear all
        buffer.clear().unwrap();
        assert_eq!(buffer.count().unwrap(), 0);
    }

    #[test]
    fn test_transition_roundtrip() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let buffer = ReplayBuffer::new(db_path.to_str().unwrap()).unwrap();

        // Create policy probs as f32 bytes
        let policy: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let policy_bytes: Vec<u8> = policy.iter().flat_map(|f| f.to_le_bytes()).collect();

        let original = Transition {
            id: "unique_id".to_string(),
            env_id: "tictactoe".to_string(),
            episode_id: "ep123".to_string(),
            step_number: 5,
            state: vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            action: vec![4, 0, 0, 0],
            next_state: vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            observation: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            next_observation: vec![1, 1, 2, 0, 1, 2, 0, 1, 2],
            reward: 0.5,
            done: true,
            timestamp: 9999999,
            policy_probs: policy_bytes,
            mcts_value: 0.75,
            game_outcome: Some(1.0),
        };

        buffer.store(&original).unwrap();

        let sampled = buffer.sample(1).unwrap();
        assert_eq!(sampled.len(), 1);

        let retrieved = &sampled[0];
        assert_eq!(retrieved.id, original.id);
        assert_eq!(retrieved.env_id, original.env_id);
        assert_eq!(retrieved.episode_id, original.episode_id);
        assert_eq!(retrieved.step_number, original.step_number);
        assert_eq!(retrieved.state, original.state);
        assert_eq!(retrieved.action, original.action);
        assert_eq!(retrieved.next_state, original.next_state);
        assert_eq!(retrieved.observation, original.observation);
        assert_eq!(retrieved.next_observation, original.next_observation);
        assert!((retrieved.reward - original.reward).abs() < 0.001);
        assert_eq!(retrieved.done, original.done);
        assert_eq!(retrieved.timestamp, original.timestamp);
        assert_eq!(retrieved.policy_probs, original.policy_probs);
        assert!((retrieved.mcts_value - original.mcts_value).abs() < 0.001);
        assert_eq!(retrieved.game_outcome, original.game_outcome);
    }

    #[test]
    fn test_update_episode_outcomes() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let buffer = ReplayBuffer::new(db_path.to_str().unwrap()).unwrap();

        // Create an episode with 5 steps (player 1 wins)
        // Steps: 0 (P1), 1 (P2), 2 (P1), 3 (P2), 4 (P1 wins)
        let episode_id = "test_episode_outcome";
        for step in 0..5u32 {
            let mut t = create_test_transition(&format!("t-{}", step), step);
            t.episode_id = episode_id.to_string();
            t.done = step == 4;
            buffer.store(&t).unwrap();
        }

        // Player 1 won (final_outcome = 1.0 from winner's perspective)
        // The last mover (step 4) was Player 1, so they get +1
        let updated = buffer.update_episode_outcomes(episode_id, 1.0, 5).unwrap();
        assert_eq!(updated, 5);

        // Verify outcomes
        // Step 4: P1 moved, won -> +1.0 (steps_from_end=0, sign=+1)
        // Step 3: P2 moved, lost -> -1.0 (steps_from_end=1, sign=-1)
        // Step 2: P1 moved, won -> +1.0 (steps_from_end=2, sign=+1)
        // Step 1: P2 moved, lost -> -1.0 (steps_from_end=3, sign=-1)
        // Step 0: P1 moved, won -> +1.0 (steps_from_end=4, sign=+1)
        let transitions = buffer.sample(10).unwrap();
        for t in transitions {
            if t.episode_id == episode_id {
                let expected = if t.step_number % 2 == 0 { 1.0 } else { -1.0 };
                assert!(
                    (t.game_outcome.unwrap() - expected).abs() < 0.001,
                    "Step {} expected outcome {}, got {:?}",
                    t.step_number,
                    expected,
                    t.game_outcome
                );
            }
        }
    }

    #[test]
    fn test_update_episode_outcomes_draw() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let buffer = ReplayBuffer::new(db_path.to_str().unwrap()).unwrap();

        // Create a 9-step draw episode (TicTacToe full board)
        let episode_id = "draw_episode";
        for step in 0..9u32 {
            let mut t = create_test_transition(&format!("draw-{}", step), step);
            t.episode_id = episode_id.to_string();
            t.done = step == 8;
            buffer.store(&t).unwrap();
        }

        // Draw game (outcome = 0.0)
        let updated = buffer.update_episode_outcomes(episode_id, 0.0, 9).unwrap();
        assert_eq!(updated, 9);

        // All steps should have outcome 0.0 (draw)
        let transitions = buffer.sample(20).unwrap();
        for t in transitions {
            if t.episode_id == episode_id {
                assert!(
                    (t.game_outcome.unwrap() - 0.0).abs() < 0.001,
                    "Draw game should have outcome 0.0, got {:?}",
                    t.game_outcome
                );
            }
        }
    }
}
