//! Actor statistics tracking and persistence.
//!
//! This module provides statistics tracking for the actor, including:
//! - Episode counts and outcomes
//! - MCTS performance metrics
//! - Episode timing information
//!
//! Stats are written to a JSON file for the web frontend to display.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, warn};

/// Aggregated actor statistics, designed for lock-free updates.
#[derive(Debug)]
pub struct ActorStats {
    /// Number of episodes completed
    episodes_completed: AtomicU32,
    /// Total game steps across all episodes
    total_steps: AtomicU64,
    /// Episodes that ended in player 1 win (reward > 0)
    player1_wins: AtomicU32,
    /// Episodes that ended in player 2 win (reward < 0)
    player2_wins: AtomicU32,
    /// Episodes that ended in draw (reward == 0)
    draws: AtomicU32,
    /// Sum of episode lengths for average calculation
    total_episode_length: AtomicU64,
    /// Start time for rate calculations
    start_time: Instant,
    /// Path to write stats file
    stats_path: String,
    /// Environment ID
    env_id: String,
    /// MCTS stats: total inference time (microseconds)
    mcts_inference_us: AtomicU64,
    /// MCTS stats: total searches performed
    mcts_searches: AtomicU64,
}

/// Serializable stats for JSON output.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActorStatsSnapshot {
    pub env_id: String,
    pub episodes_completed: u32,
    pub total_steps: u64,
    pub player1_wins: u32,
    pub player2_wins: u32,
    pub draws: u32,
    pub avg_episode_length: f64,
    pub episodes_per_second: f64,
    pub runtime_seconds: f64,
    pub mcts_avg_inference_us: f64,
    pub timestamp: u64,
}

impl ActorStats {
    /// Create new stats tracker.
    pub fn new(data_dir: &str, env_id: &str) -> Self {
        let stats_path = format!("{}/actor_stats.json", data_dir);

        // Ensure data directory exists
        if let Err(e) = fs::create_dir_all(data_dir) {
            warn!("Failed to create data directory: {}", e);
        }

        Self {
            episodes_completed: AtomicU32::new(0),
            total_steps: AtomicU64::new(0),
            player1_wins: AtomicU32::new(0),
            player2_wins: AtomicU32::new(0),
            draws: AtomicU32::new(0),
            total_episode_length: AtomicU64::new(0),
            start_time: Instant::now(),
            stats_path,
            env_id: env_id.to_string(),
            mcts_inference_us: AtomicU64::new(0),
            mcts_searches: AtomicU64::new(0),
        }
    }

    /// Record a completed episode.
    pub fn record_episode(&self, steps: u32, final_reward: f32) {
        self.episodes_completed.fetch_add(1, Ordering::Relaxed);
        self.total_steps.fetch_add(steps as u64, Ordering::Relaxed);
        self.total_episode_length
            .fetch_add(steps as u64, Ordering::Relaxed);

        // Categorize outcome based on final reward
        // Positive = player 1 wins, negative = player 2 wins, zero = draw
        if final_reward > 0.0 {
            self.player1_wins.fetch_add(1, Ordering::Relaxed);
        } else if final_reward < 0.0 {
            self.player2_wins.fetch_add(1, Ordering::Relaxed);
        } else {
            self.draws.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record MCTS performance for an episode.
    pub fn record_mcts_stats(&self, searches: u32, inference_us: u64) {
        self.mcts_searches
            .fetch_add(searches as u64, Ordering::Relaxed);
        self.mcts_inference_us
            .fetch_add(inference_us, Ordering::Relaxed);
    }

    /// Get a snapshot of current stats.
    pub fn snapshot(&self) -> ActorStatsSnapshot {
        let episodes = self.episodes_completed.load(Ordering::Relaxed);
        let total_length = self.total_episode_length.load(Ordering::Relaxed);
        let runtime = self.start_time.elapsed().as_secs_f64();
        let searches = self.mcts_searches.load(Ordering::Relaxed);
        let inference_us = self.mcts_inference_us.load(Ordering::Relaxed);

        let avg_episode_length = if episodes > 0 {
            total_length as f64 / episodes as f64
        } else {
            0.0
        };

        let episodes_per_second = if runtime > 0.0 {
            episodes as f64 / runtime
        } else {
            0.0
        };

        let mcts_avg_inference_us = if searches > 0 {
            inference_us as f64 / searches as f64
        } else {
            0.0
        };

        ActorStatsSnapshot {
            env_id: self.env_id.clone(),
            episodes_completed: episodes,
            total_steps: self.total_steps.load(Ordering::Relaxed),
            player1_wins: self.player1_wins.load(Ordering::Relaxed),
            player2_wins: self.player2_wins.load(Ordering::Relaxed),
            draws: self.draws.load(Ordering::Relaxed),
            avg_episode_length,
            episodes_per_second,
            runtime_seconds: runtime,
            mcts_avg_inference_us,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Write stats to JSON file (atomic write-then-rename).
    pub fn write_stats(&self) {
        let snapshot = self.snapshot();

        // Serialize to JSON
        let json = match serde_json::to_string_pretty(&snapshot) {
            Ok(j) => j,
            Err(e) => {
                warn!("Failed to serialize actor stats: {}", e);
                return;
            }
        };

        // Write to temp file then rename (atomic on most filesystems)
        let temp_path = format!("{}.tmp", self.stats_path);
        match fs::File::create(&temp_path) {
            Ok(mut file) => {
                if let Err(e) = file.write_all(json.as_bytes()) {
                    warn!("Failed to write actor stats: {}", e);
                    return;
                }
            }
            Err(e) => {
                warn!("Failed to create temp stats file: {}", e);
                return;
            }
        }

        if let Err(e) = fs::rename(&temp_path, &self.stats_path) {
            warn!("Failed to rename stats file: {}", e);
            // Try to clean up temp file
            let _ = fs::remove_file(&temp_path);
            return;
        }

        debug!("Wrote actor stats to {}", self.stats_path);
    }

    /// Check if stats file path exists.
    pub fn stats_path(&self) -> &str {
        &self.stats_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn test_record_episode() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Record some episodes
        stats.record_episode(9, 1.0); // P1 win
        stats.record_episode(8, -1.0); // P2 win
        stats.record_episode(9, 0.0); // Draw

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.episodes_completed, 3);
        assert_eq!(snapshot.player1_wins, 1);
        assert_eq!(snapshot.player2_wins, 1);
        assert_eq!(snapshot.draws, 1);
        assert_eq!(snapshot.total_steps, 26);
    }

    #[test]
    fn test_write_stats() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        stats.record_episode(9, 1.0);
        stats.write_stats();

        // Verify file exists and is valid JSON
        let path = Path::new(stats.stats_path());
        assert!(path.exists());

        let content = fs::read_to_string(path).unwrap();
        let parsed: ActorStatsSnapshot = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.episodes_completed, 1);
    }
}
