//! Health check endpoint.

use axum::Json;

use crate::types::HealthResponse;

/// Health check handler.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}
