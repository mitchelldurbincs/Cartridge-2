//! Prometheus metrics for the web server component.
//!
//! This module provides metrics for monitoring web server performance,
//! game session tracking, and bot move latency.

use lazy_static::lazy_static;
use prometheus::{
    Encoder, Histogram, HistogramOpts, HistogramVec, IntCounter, IntGauge, Opts, Registry,
    TextEncoder,
};
use std::sync::Once;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // ========== Game Session Metrics ==========

    /// Total game sessions created
    pub static ref GAMES_CREATED: IntCounter = IntCounter::with_opts(
        Opts::new("web_games_created_total", "Total game sessions created")
    ).unwrap();

    /// Currently active game sessions
    pub static ref GAMES_ACTIVE: IntGauge = IntGauge::with_opts(
        Opts::new("web_games_active", "Currently active game sessions")
    ).unwrap();

    /// Total moves played across all games
    pub static ref MOVES_PLAYED: IntCounter = IntCounter::with_opts(
        Opts::new("web_moves_played_total", "Total moves played across all games")
    ).unwrap();

    /// Games completed with each outcome
    pub static ref GAMES_COMPLETED: IntCounter = IntCounter::with_opts(
        Opts::new("web_games_completed_total", "Total games completed")
    ).unwrap();

    // ========== Request Latency ==========

    /// HTTP request latency by endpoint and method
    pub static ref REQUEST_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new("web_request_duration_seconds", "HTTP request latency")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
        &["endpoint", "method"]
    ).unwrap();

    // ========== Bot Metrics ==========

    /// Time for bot to compute move
    pub static ref BOT_MOVE_SECONDS: Histogram = Histogram::with_opts(
        HistogramOpts::new("web_bot_move_seconds", "Time for bot to compute move")
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    ).unwrap();

    // ========== Model Status ==========

    /// Whether a valid model is loaded (0/1)
    pub static ref MODEL_LOADED: IntGauge = IntGauge::with_opts(
        Opts::new("web_model_loaded", "Whether a valid model is loaded (0/1)")
    ).unwrap();

    /// Number of model reloads
    pub static ref MODEL_RELOADS: IntCounter = IntCounter::with_opts(
        Opts::new("web_model_reloads_total", "Number of model hot-reload events")
    ).unwrap();
}

static INIT: Once = Once::new();

/// Initialize and register all metrics with the registry.
/// Safe to call multiple times - only initializes once.
pub fn init_metrics() {
    INIT.call_once(|| {
        REGISTRY.register(Box::new(GAMES_CREATED.clone())).unwrap();
        REGISTRY.register(Box::new(GAMES_ACTIVE.clone())).unwrap();
        REGISTRY.register(Box::new(MOVES_PLAYED.clone())).unwrap();
        REGISTRY
            .register(Box::new(GAMES_COMPLETED.clone()))
            .unwrap();
        REGISTRY
            .register(Box::new(REQUEST_LATENCY.clone()))
            .unwrap();
        REGISTRY
            .register(Box::new(BOT_MOVE_SECONDS.clone()))
            .unwrap();
        REGISTRY.register(Box::new(MODEL_LOADED.clone())).unwrap();
        REGISTRY.register(Box::new(MODEL_RELOADS.clone())).unwrap();
    });
}

/// Encode all metrics to Prometheus text format
pub fn encode_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_init() {
        init_metrics();
    }

    #[test]
    fn test_encode_metrics() {
        init_metrics();
        let output = encode_metrics();
        assert!(output.contains("web_games_created_total"));
        assert!(output.contains("web_moves_played_total"));
    }
}
