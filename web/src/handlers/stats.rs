//! Stats and model info handlers.

use axum::{extract::State, Json};
use std::sync::Arc;

use crate::types::{ModelInfoResponse, TrainingStats};
use crate::AppState;

/// Get training stats from stats.json.
pub async fn get_stats(State(state): State<Arc<AppState>>) -> Json<TrainingStats> {
    let stats_path = format!("{}/stats.json", state.data_dir);

    match tokio::fs::read_to_string(&stats_path).await {
        Ok(content) => match serde_json::from_str::<TrainingStats>(&content) {
            Ok(stats) => Json(stats),
            Err(e) => {
                tracing::warn!("Failed to parse stats.json: {}", e);
                Json(TrainingStats::default())
            }
        },
        Err(_) => {
            // Return empty stats if file doesn't exist
            Json(TrainingStats::default())
        }
    }
}

/// Get info about the currently loaded model.
pub async fn get_model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfoResponse> {
    // model_info uses std::sync::RwLock (shared with model_watcher crate).
    // This is safe because the lock is held briefly and not across await points.
    let info = state
        .model_info
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default();

    let status = if info.loaded {
        match info.training_step {
            Some(step) => format!("Model loaded (step {})", step),
            None => "Model loaded".to_string(),
        }
    } else {
        "No model loaded - bot plays randomly".to_string()
    };

    Json(ModelInfoResponse {
        loaded: info.loaded,
        path: info.path,
        file_modified: info.file_modified,
        loaded_at: info.loaded_at,
        training_step: info.training_step,
        status,
    })
}
