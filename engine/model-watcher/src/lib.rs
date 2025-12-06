//! Shared ONNX model watcher used by the actor and web server.
//!
//! Watches a directory for model updates and hot-reloads them into an
//! [`mcts::OnnxEvaluator`]. Metadata about the currently loaded model is
//! tracked so UIs can display it.

use anyhow::{anyhow, Result};
use mcts::OnnxEvaluator;
use notify::{recommended_watcher, Event, EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Information about the currently loaded model.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// Whether a model is currently loaded.
    pub loaded: bool,
    /// Path to the loaded model file.
    pub path: Option<String>,
    /// When the model file was last modified (Unix timestamp).
    pub file_modified: Option<u64>,
    /// When the model was loaded into memory (Unix timestamp).
    pub loaded_at: Option<u64>,
    /// Training step from filename (if parseable, e.g., "model_step_000100.onnx").
    pub training_step: Option<u32>,
}

/// Watches for new ONNX model files and hot-reloads them.
pub struct ModelWatcher {
    /// Path to the models directory.
    model_dir: PathBuf,
    /// Expected model filename.
    model_filename: String,
    /// Observation size for the model.
    obs_size: usize,
    /// Shared evaluator to update.
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// Current model info.
    model_info: Arc<RwLock<ModelInfo>>,
}

impl ModelWatcher {
    /// Create a new model watcher.
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
            model_info: Arc::new(RwLock::new(ModelInfo::default())),
        }
    }

    /// Get the full path to the model file.
    pub fn model_path(&self) -> PathBuf {
        self.model_dir.join(&self.model_filename)
    }

    /// Get the current model info.
    pub fn model_info(&self) -> Arc<RwLock<ModelInfo>> {
        Arc::clone(&self.model_info)
    }

    /// Try to load the model if it exists.
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

    /// Load a model from the given path and update metadata.
    fn load_model(&self, path: &Path) -> Result<()> {
        Self::load_model_static(path, self.obs_size, &self.evaluator, &self.model_info)
    }

    /// Start watching for model changes.
    ///
    /// Returns a channel that receives () when a new model is loaded.
    pub async fn start_watching(&self) -> Result<mpsc::Receiver<()>> {
        let (tx, rx) = mpsc::channel(16);
        let model_dir = self.model_dir.clone();
        let model_filename = self.model_filename.clone();
        let obs_size = self.obs_size;
        let evaluator = Arc::clone(&self.evaluator);
        let model_info = Arc::clone(&self.model_info);

        // Create channel for file system events.
        let (fs_tx, mut fs_rx) = mpsc::channel(100);

        // Set up the file watcher.
        let mut watcher = recommended_watcher(move |res: Result<Event, notify::Error>| match res {
            Ok(event) => {
                let _ = fs_tx.blocking_send(event);
            }
            Err(e) => {
                warn!("File watcher error: {}", e);
            }
        })
        .map_err(|e| anyhow!("Failed to create file watcher: {}", e))?;

        // Create directory if it doesn't exist.
        if !model_dir.exists() {
            std::fs::create_dir_all(&model_dir)
                .map_err(|e| anyhow!("Failed to create model directory: {}", e))?;
        }

        // Start watching the directory.
        watcher
            .watch(&model_dir, RecursiveMode::NonRecursive)
            .map_err(|e| anyhow!("Failed to watch directory: {}", e))?;

        info!("Started watching {:?} for model updates", model_dir);

        // Spawn task to handle file events.
        tokio::spawn(async move {
            let _watcher = watcher; // Keep watcher alive

            let mut last_reload = std::time::Instant::now();
            let debounce_duration = Duration::from_millis(500);

            while let Some(event) = fs_rx.recv().await {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {}
                    _ => continue,
                }

                let model_path = model_dir.join(&model_filename);
                let is_our_file = event
                    .paths
                    .iter()
                    .any(|p| p.file_name() == model_path.file_name());

                if !is_our_file {
                    continue;
                }

                if last_reload.elapsed() < debounce_duration {
                    debug!("Debouncing model reload event");
                    continue;
                }

                // Small delay to ensure file is fully written.
                tokio::time::sleep(Duration::from_millis(100)).await;

                if !model_path.exists() {
                    debug!("Model file doesn't exist yet");
                    continue;
                }

                info!("Model file changed, reloading {:?}", model_path);
                match Self::load_model_static(&model_path, obs_size, &evaluator, &model_info) {
                    Ok(()) => {
                        last_reload = std::time::Instant::now();
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
    fn load_model_static(
        path: &Path,
        obs_size: usize,
        evaluator: &Arc<RwLock<Option<OnnxEvaluator>>>,
        model_info: &Arc<RwLock<ModelInfo>>,
    ) -> Result<()> {
        // Get file modification time.
        let file_modified = path
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        // Try to parse training step from filename.
        let training_step = path.file_name().and_then(|n| n.to_str()).and_then(|name| {
            name.split('_')
                .filter_map(|part| part.trim_end_matches(".onnx").parse::<u32>().ok())
                .next_back()
        });

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
        assert!(!result.unwrap());

        let guard = evaluator.read().unwrap();
        assert!(guard.is_none());
    }
}
