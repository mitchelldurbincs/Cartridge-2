//! Model file watcher for hot-reloading ONNX models
//!
//! Watches a directory for new ONNX model files and reloads them
//! into the evaluator when they appear.

use anyhow::{anyhow, Result};
use mcts::OnnxEvaluator;
use notify::{recommended_watcher, Event, EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Watches for new ONNX model files and hot-reloads them
pub struct ModelWatcher {
    /// Path to the models directory
    model_dir: PathBuf,
    /// Expected model filename
    model_filename: String,
    /// Observation size for the model
    obs_size: usize,
    /// Shared evaluator to update
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
}

impl ModelWatcher {
    /// Create a new model watcher
    ///
    /// # Arguments
    /// * `model_dir` - Directory to watch for model files
    /// * `model_filename` - Name of the model file to watch (e.g., "latest.onnx")
    /// * `obs_size` - Observation size expected by the model
    /// * `evaluator` - Shared evaluator reference to update on reload
    pub fn new(
        model_dir: impl AsRef<Path>,
        model_filename: impl Into<String>,
        obs_size: usize,
        evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    ) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            model_filename: model_filename.into(),
            obs_size,
            evaluator,
        }
    }

    /// Get the full path to the model file
    pub fn model_path(&self) -> PathBuf {
        self.model_dir.join(&self.model_filename)
    }

    /// Try to load the model if it exists
    pub fn try_load_existing(&self) -> Result<bool> {
        let path = self.model_path();
        if path.exists() {
            info!("Found existing model at {:?}", path);
            self.load_model(&path)?;
            Ok(true)
        } else {
            debug!("No existing model at {:?}", path);
            Ok(false)
        }
    }

    /// Load a model from the given path
    fn load_model(&self, path: &Path) -> Result<()> {
        info!("Loading model from {:?}", path);

        // Load the new model
        let new_evaluator = OnnxEvaluator::load(path, self.obs_size)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;

        // Swap in the new model
        let mut guard = self
            .evaluator
            .write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        *guard = Some(new_evaluator);

        info!("Model loaded successfully from {:?}", path);
        Ok(())
    }

    /// Start watching for model changes
    ///
    /// This spawns a background task that watches the model directory
    /// and reloads the model when it changes.
    ///
    /// Returns a channel that receives () when a new model is loaded.
    pub async fn start_watching(&self) -> Result<mpsc::Receiver<()>> {
        let (tx, rx) = mpsc::channel(16);
        let model_dir = self.model_dir.clone();
        let model_filename = self.model_filename.clone();
        let obs_size = self.obs_size;
        let evaluator = Arc::clone(&self.evaluator);

        // Create channel for file system events
        let (fs_tx, mut fs_rx) = mpsc::channel(100);

        // Set up the file watcher
        let mut watcher = recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    // Non-blocking send - drop events if channel is full
                    let _ = fs_tx.blocking_send(event);
                }
                Err(e) => {
                    warn!("File watcher error: {}", e);
                }
            }
        })
        .map_err(|e| anyhow!("Failed to create file watcher: {}", e))?;

        // Create directory if it doesn't exist
        if !model_dir.exists() {
            std::fs::create_dir_all(&model_dir)
                .map_err(|e| anyhow!("Failed to create model directory: {}", e))?;
        }

        // Start watching the directory
        watcher
            .watch(&model_dir, RecursiveMode::NonRecursive)
            .map_err(|e| anyhow!("Failed to watch directory: {}", e))?;

        info!("Started watching {:?} for model updates", model_dir);

        // Spawn task to handle file events
        tokio::spawn(async move {
            // Keep watcher alive in this task
            let _watcher = watcher;

            // Debounce timer to avoid rapid reloads
            let mut last_reload = std::time::Instant::now();
            let debounce_duration = Duration::from_millis(500);

            while let Some(event) = fs_rx.recv().await {
                // Only care about create/modify events
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {}
                    _ => continue,
                }

                // Check if this affects our model file
                let model_path = model_dir.join(&model_filename);
                let is_our_file = event
                    .paths
                    .iter()
                    .any(|p| p.file_name() == model_path.file_name());

                if !is_our_file {
                    continue;
                }

                // Debounce rapid events
                if last_reload.elapsed() < debounce_duration {
                    debug!("Debouncing model reload event");
                    continue;
                }

                // Small delay to ensure file is fully written
                tokio::time::sleep(Duration::from_millis(100)).await;

                // Verify file exists and is readable
                if !model_path.exists() {
                    debug!("Model file doesn't exist yet");
                    continue;
                }

                // Try to load the model
                info!("Model file changed, reloading {:?}", model_path);
                match Self::load_model_static(&model_path, obs_size, &evaluator) {
                    Ok(()) => {
                        last_reload = std::time::Instant::now();
                        // Notify listeners
                        let _ = tx.send(()).await;
                    }
                    Err(e) => {
                        error!("Failed to reload model: {}", e);
                    }
                }
            }
        });

        Ok(rx)
    }

    /// Static method to load model (for use in spawned task)
    fn load_model_static(
        path: &Path,
        obs_size: usize,
        evaluator: &Arc<RwLock<Option<OnnxEvaluator>>>,
    ) -> Result<()> {
        let new_evaluator = OnnxEvaluator::load(path, obs_size)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;

        let mut guard = evaluator
            .write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        *guard = Some(new_evaluator);

        info!("Model reloaded successfully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_model_watcher_creation() {
        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new("/tmp/models", "latest.onnx", 29, evaluator);

        assert_eq!(
            watcher.model_path(),
            PathBuf::from("/tmp/models/latest.onnx")
        );
    }

    #[test]
    fn test_try_load_nonexistent() {
        let temp_dir = tempdir().unwrap();
        let evaluator = Arc::new(RwLock::new(None));
        let watcher = ModelWatcher::new(temp_dir.path(), "nonexistent.onnx", 29, evaluator.clone());

        let result = watcher.try_load_existing();
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should return false - no file

        // Evaluator should still be None
        let guard = evaluator.read().unwrap();
        assert!(guard.is_none());
    }
}
