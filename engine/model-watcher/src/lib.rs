//! Model file watcher for hot-reloading ONNX models
//!
//! This crate provides a shared implementation for watching and hot-reloading
//! ONNX model files used by both the actor and web server components.
//!
//! # Features
//!
//! - `metadata`: Enable additional metadata tracking (load time, file modification time,
//!   training step extraction). Useful for web server display.
//!
//! # Example
//!
//! ```ignore
//! use model_watcher::ModelWatcher;
//! use std::sync::{Arc, RwLock};
//!
//! let evaluator = Arc::new(RwLock::new(None));
//! let watcher = ModelWatcher::new("./data/models", "latest.onnx", 29, evaluator);
//!
//! // Try to load existing model
//! watcher.try_load_existing()?;
//!
//! // Start watching for changes
//! let mut rx = watcher.start_watching().await?;
//! while let Some(()) = rx.recv().await {
//!     println!("Model reloaded!");
//! }
//! ```

use anyhow::{anyhow, Result};
use mcts::OnnxEvaluator;
use notify::{recommended_watcher, Event, EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

#[cfg(feature = "metadata")]
use std::time::{SystemTime, UNIX_EPOCH};

/// Information about the currently loaded model.
///
/// Only available with the `metadata` feature enabled.
#[cfg(feature = "metadata")]
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// Whether a model is currently loaded
    pub loaded: bool,
    /// Path to the loaded model file
    pub path: Option<String>,
    /// When the model file was last modified (Unix timestamp)
    pub file_modified: Option<u64>,
    /// When the model was loaded into memory (Unix timestamp)
    pub loaded_at: Option<u64>,
    /// Training step from filename (if parseable, e.g., "model_step_000100.onnx")
    pub training_step: Option<u32>,
}

/// Watches for new ONNX model files and hot-reloads them.
///
/// This is the shared implementation used by both actor and web server components.
pub struct ModelWatcher {
    /// Path to the models directory
    model_dir: PathBuf,
    /// Expected model filename
    model_filename: String,
    /// Observation size for the model
    obs_size: usize,
    /// Shared evaluator to update
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// Current model info (only with metadata feature)
    #[cfg(feature = "metadata")]
    model_info: Arc<RwLock<ModelInfo>>,
}

impl ModelWatcher {
    /// Create a new model watcher.
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
            #[cfg(feature = "metadata")]
            model_info: Arc::new(RwLock::new(ModelInfo::default())),
        }
    }

    /// Get the full path to the model file.
    pub fn model_path(&self) -> PathBuf {
        self.model_dir.join(&self.model_filename)
    }

    /// Get the current model info.
    ///
    /// Only available with the `metadata` feature enabled.
    #[cfg(feature = "metadata")]
    pub fn model_info(&self) -> Arc<RwLock<ModelInfo>> {
        Arc::clone(&self.model_info)
    }

    /// Try to load the model if it exists.
    ///
    /// Returns `Ok(true)` if a model was loaded, `Ok(false)` if no model exists.
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

    /// Load a model from the given path.
    fn load_model(&self, path: &Path) -> Result<()> {
        info!("Loading model from {:?}", path);

        #[cfg(feature = "metadata")]
        let (file_modified, training_step) = Self::extract_metadata(path);

        // Load the new model
        let new_evaluator = OnnxEvaluator::load(path, self.obs_size)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;

        // Swap in the new model
        {
            let mut guard = self
                .evaluator
                .write()
                .map_err(|e| anyhow!("Failed to acquire evaluator write lock: {}", e))?;
            *guard = Some(new_evaluator);
        }

        #[cfg(feature = "metadata")]
        {
            let mut info = self
                .model_info
                .write()
                .map_err(|e| anyhow!("Failed to acquire model_info write lock: {}", e))?;
            *info = ModelInfo {
                loaded: true,
                path: Some(path.to_string_lossy().to_string()),
                file_modified,
                loaded_at: Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                ),
                training_step,
            };
            info!(
                "Model loaded successfully from {:?} (modified: {:?}, step: {:?})",
                path, file_modified, training_step
            );
        }

        #[cfg(not(feature = "metadata"))]
        info!("Model loaded successfully from {:?}", path);

        Ok(())
    }

    /// Extract metadata from a model file path.
    #[cfg(feature = "metadata")]
    fn extract_metadata(path: &Path) -> (Option<u64>, Option<u32>) {
        // Get file modification time
        let file_modified = path
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        // Try to parse training step from filename (e.g., "model_step_000100.onnx")
        let training_step = path.file_name().and_then(|n| n.to_str()).and_then(|name| {
            name.split('_')
                .filter_map(|part| part.trim_end_matches(".onnx").parse::<u32>().ok())
                .next_back()
        });

        (file_modified, training_step)
    }

    /// Start watching for model changes.
    ///
    /// This spawns a background task that watches the model directory
    /// and reloads the model when it changes.
    ///
    /// Returns a channel that receives `()` when a new model is loaded.
    pub async fn start_watching(&self) -> Result<mpsc::Receiver<()>> {
        let (tx, rx) = mpsc::channel(16);
        let model_dir = self.model_dir.clone();
        let model_filename = self.model_filename.clone();
        let obs_size = self.obs_size;
        let evaluator = Arc::clone(&self.evaluator);

        #[cfg(feature = "metadata")]
        let model_info = Arc::clone(&self.model_info);

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

                #[cfg(feature = "metadata")]
                let result =
                    Self::load_model_static(&model_path, obs_size, &evaluator, &model_info);

                #[cfg(not(feature = "metadata"))]
                let result = Self::load_model_static(&model_path, obs_size, &evaluator);

                match result {
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

    /// Static method to load model (for use in spawned task).
    #[cfg(feature = "metadata")]
    fn load_model_static(
        path: &Path,
        obs_size: usize,
        evaluator: &Arc<RwLock<Option<OnnxEvaluator>>>,
        model_info: &Arc<RwLock<ModelInfo>>,
    ) -> Result<()> {
        let (file_modified, training_step) = Self::extract_metadata(path);

        let new_evaluator = OnnxEvaluator::load(path, obs_size)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;

        {
            let mut guard = evaluator
                .write()
                .map_err(|e| anyhow!("Failed to acquire evaluator write lock: {}", e))?;
            *guard = Some(new_evaluator);
        }

        {
            let mut info = model_info
                .write()
                .map_err(|e| anyhow!("Failed to acquire model_info write lock: {}", e))?;
            *info = ModelInfo {
                loaded: true,
                path: Some(path.to_string_lossy().to_string()),
                file_modified,
                loaded_at: Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                ),
                training_step,
            };
        }

        info!("Model reloaded successfully (step: {:?})", training_step);
        Ok(())
    }

    /// Static method to load model (for use in spawned task) - without metadata.
    #[cfg(not(feature = "metadata"))]
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

    #[cfg(feature = "metadata")]
    #[test]
    fn test_model_info_default() {
        let info = ModelInfo::default();
        assert!(!info.loaded);
        assert!(info.path.is_none());
        assert!(info.file_modified.is_none());
        assert!(info.loaded_at.is_none());
        assert!(info.training_step.is_none());
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn test_extract_metadata_training_step() {
        use std::path::Path;

        // Test with step in filename
        let path = Path::new("/tmp/models/model_step_000100.onnx");
        let (_, training_step) = ModelWatcher::extract_metadata(path);
        assert_eq!(training_step, Some(100));

        // Test without step
        let path = Path::new("/tmp/models/latest.onnx");
        let (_, training_step) = ModelWatcher::extract_metadata(path);
        assert!(training_step.is_none());
    }
}
