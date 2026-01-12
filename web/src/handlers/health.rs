//! Health check and metrics endpoints.

use axum::{
    http::{header, StatusCode},
    Json,
};

use crate::metrics;
use crate::types::HealthResponse;

/// Health check handler.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Prometheus metrics handler.
pub async fn metrics_handler() -> (StatusCode, [(header::HeaderName, &'static str); 1], String) {
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        metrics::encode_metrics(),
    )
}
