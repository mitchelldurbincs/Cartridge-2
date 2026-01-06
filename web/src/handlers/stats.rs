//! Stats and model info handlers.

use axum::{extract::State, Json};
use std::sync::Arc;

use crate::types::{ModelInfoResponse, SelfPlayRequest, SelfPlayResponse, TrainingStats};
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

/// Control self-play actor (placeholder).
pub async fn selfplay(Json(req): Json<SelfPlayRequest>) -> Json<SelfPlayResponse> {
    // Placeholder - actual implementation would control the actor process
    let message = match req.action.as_str() {
        "start" => "Self-play start requested (not implemented yet)",
        "stop" => "Self-play stop requested (not implemented yet)",
        _ => "Unknown action",
    };

    Json(SelfPlayResponse {
        status: "placeholder".to_string(),
        message: message.to_string(),
    })
}
