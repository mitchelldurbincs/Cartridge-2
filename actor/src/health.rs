//! Health check HTTP server for Kubernetes probes.
//!
//! Provides liveness and readiness endpoints for container orchestration.

use axum::{routing::get, Router};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{error, info};

/// Shared health state between main actor loop and health server.
#[derive(Debug, Clone)]
pub struct HealthState {
    /// Set to true once the actor has completed initialization.
    ready: Arc<AtomicBool>,
    /// Set to false if the actor encounters a fatal error.
    healthy: Arc<AtomicBool>,
    /// Timestamp of last successful episode completion (Unix seconds).
    last_episode_time: Arc<AtomicU64>,
}

impl HealthState {
    pub fn new() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(false)),
            healthy: Arc::new(AtomicBool::new(true)),
            last_episode_time: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Mark the actor as ready to receive traffic.
    pub fn set_ready(&self) {
        self.ready.store(true, Ordering::SeqCst);
        info!("Actor marked as ready");
    }

    /// Mark the actor as unhealthy (will trigger restart).
    /// Called when the actor encounters a fatal error that requires restart.
    #[allow(dead_code)]
    pub fn set_unhealthy(&self) {
        self.healthy.store(false, Ordering::SeqCst);
        error!("Actor marked as unhealthy");
    }

    /// Update the last episode completion time.
    pub fn record_episode_complete(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_episode_time.store(now, Ordering::SeqCst);
    }

    /// Check if the actor is ready.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Check if the actor is healthy.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::SeqCst)
    }

    /// Check if the actor has completed an episode recently (within timeout).
    pub fn is_making_progress(&self, timeout_secs: u64) -> bool {
        let last = self.last_episode_time.load(Ordering::SeqCst);
        if last == 0 {
            // No episodes completed yet, but that's ok during startup
            return true;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        (now - last) < timeout_secs
    }
}

impl Default for HealthState {
    fn default() -> Self {
        Self::new()
    }
}

/// Start the health check HTTP server on the given port.
pub async fn start_health_server(
    port: u16,
    state: HealthState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = Router::new()
        .route(
            "/health",
            get({
                let state = state.clone();
                move || health_handler(state.clone())
            }),
        )
        .route(
            "/ready",
            get({
                let state = state.clone();
                move || ready_handler(state.clone())
            }),
        )
        .route("/metrics", get(metrics_handler));

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Health server listening on {}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}

async fn health_handler(state: HealthState) -> axum::http::StatusCode {
    // Liveness: is the process fundamentally healthy?
    // Check both health flag and progress (no stuck episodes)
    // 5 minute timeout for progress
    if state.is_healthy() && state.is_making_progress(300) {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn ready_handler(state: HealthState) -> axum::http::StatusCode {
    // Readiness: is the actor ready to generate episodes?
    if state.is_ready() && state.is_healthy() {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn metrics_handler() -> String {
    // Placeholder for Prometheus metrics
    "# HELP actor_up Actor is up\n# TYPE actor_up gauge\nactor_up 1\n".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_state_initial() {
        let state = HealthState::new();
        assert!(!state.is_ready());
        assert!(state.is_healthy());
    }

    #[test]
    fn test_health_state_ready() {
        let state = HealthState::new();
        state.set_ready();
        assert!(state.is_ready());
    }

    #[test]
    fn test_health_state_unhealthy() {
        let state = HealthState::new();
        state.set_unhealthy();
        assert!(!state.is_healthy());
    }

    #[test]
    fn test_progress_tracking() {
        let state = HealthState::new();
        // Initially, no episodes completed but should report as making progress
        assert!(state.is_making_progress(300));

        // Record an episode completion
        state.record_episode_complete();
        assert!(state.is_making_progress(300));
    }

    #[test]
    fn test_progress_timeout() {
        let state = HealthState::new();
        // Manually set an old timestamp (1 second in the past)
        state.last_episode_time.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                - 1,
            Ordering::SeqCst,
        );

        // With a 2 second timeout, should still be making progress
        assert!(state.is_making_progress(2));

        // With a 0 second timeout, should NOT be making progress
        assert!(!state.is_making_progress(0));
    }
}
