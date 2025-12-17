//! Model file watcher for hot-reloading ONNX models in the web server
//!
//! This module re-exports the shared `model_watcher` crate with the
//! `metadata` feature enabled for tracking model info.
//! See that crate for full documentation.

pub use model_watcher::{ModelInfo, ModelWatcher};
