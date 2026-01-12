# Structured Logging Implementation Plan

## Executive Summary

This document outlines a comprehensive plan for implementing structured logging across Cartridge-2. Structured logging transforms human-readable log messages into machine-parseable JSON, enabling powerful querying, alerting, and correlation in Google Cloud Logging.

**Current State:** Basic `tracing` (Rust) and `logging` (Python) with unstructured text output to stdout.

**Target State:** JSON-structured logs with correlation IDs, automatic Cloud Logging integration, and queryable fields across all components.

**Relationship to Other Plans:**
- **Prometheus Metrics (`prometheus.md`):** Metrics answer "how much/how often?" Logs answer "what happened and why?"
- **Deployment Plans (`deployment-plan-part1/2.md`):** This plan expands on the brief structured logging section in part 2.

---

## Why Structured Logging Matters for GCP

| Scenario | Unstructured Logs | Structured Logs |
|----------|-------------------|-----------------|
| Find errors | `grep ERROR` across files | `severity>=ERROR` in Cloud Logging Console |
| Debug slow episode | Scroll through thousands of lines | `jsonPayload.episode_id="ep-12345" AND jsonPayload.duration_ms>1000` |
| Correlate actor→trainer | Manually match timestamps | `jsonPayload.trace_id="abc123"` |
| Alert on training stall | Custom parsing scripts | Log-based metric on `jsonPayload.event="training_step"` |
| Cost analysis | Impossible | Query by `jsonPayload.gpu_utilization` |

### Cost Benefits

1. **Faster debugging** - Query logs instead of SSH into pods
2. **Proactive alerting** - Log-based metrics trigger alerts before users notice
3. **Audit trail** - Structured logs feed BigQuery for ML experiment tracking
4. **Reduced noise** - Filter by severity/service without regex

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Google Cloud Logging                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Log Router                                                         │ │
│  │  ├── _Default sink → Cloud Logging Storage (30-day retention)      │ │
│  │  ├── errors-sink → BigQuery (long-term analysis)                   │ │
│  │  └── metrics-sink → Log-based Metrics → Cloud Monitoring           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                     │
│                                    │ JSON logs (auto-collected)          │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────┐
│                    GKE Cluster (stdout → Fluent Bit → Cloud Logging)    │
│                                    │                                     │
│  ┌─────────────┐    ┌─────────────┴─┐    ┌─────────────┐                │
│  │   Actor     │    │   Trainer     │    │ Web Server  │                │
│  │  (Rust)     │    │  (Python)     │    │  (Rust)     │                │
│  │             │    │               │    │             │                │
│  │ tracing     │    │ structlog     │    │ tracing     │                │
│  │ + JSON fmt  │    │ + JSON fmt    │    │ + JSON fmt  │                │
│  └─────────────┘    └───────────────┘    └─────────────┘                │
│        │                   │                    │                        │
│        └───────────────────┼────────────────────┘                        │
│                            │                                             │
│                   Shared correlation: trace_id                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Log Flow

1. **Application** → Writes JSON to stdout
2. **Container runtime** → Captures stdout
3. **Fluent Bit (GKE)** → Collects and forwards to Cloud Logging
4. **Cloud Logging** → Parses JSON into `jsonPayload`, routes to sinks
5. **BigQuery/Metrics** → Long-term storage and alerting

---

## Phase 1: Rust Components (Actor & Web Server)

### 1.1 Dependencies

**Files:** `actor/Cargo.toml`, `web/Cargo.toml`

```toml
[dependencies]
# Existing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }

# New
uuid = { version = "1.0", features = ["v4"] }  # For trace IDs
serde_json = "1.0"  # JSON serialization

# Future (OpenTelemetry integration)
# tracing-opentelemetry = "0.22"
# opentelemetry = { version = "0.21", features = ["rt-tokio"] }
# opentelemetry-otlp = "0.14"
```

### 1.2 Logging Configuration Module

**File:** `actor/src/logging.rs` (new), `web/src/logging.rs` (new)

```rust
//! Structured logging configuration for cloud deployments.
//!
//! Supports both human-readable text (local development) and JSON (production).

use tracing_subscriber::{
    fmt::{self, format::FmtSpan, time::UtcTime},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};
use std::io;

/// Log output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable text with colors (local development)
    Text,
    /// JSON with Cloud Logging-compatible fields (production)
    Json,
    /// Compact text without colors (CI/testing)
    Compact,
}

impl LogFormat {
    /// Detect format from environment.
    ///
    /// Priority:
    /// 1. LOG_FORMAT env var (json, text, compact)
    /// 2. Auto-detect: JSON if KUBERNETES_SERVICE_HOST is set
    /// 3. Default: Text
    pub fn from_env() -> Self {
        match std::env::var("LOG_FORMAT").as_deref() {
            Ok("json") => LogFormat::Json,
            Ok("text") => LogFormat::Text,
            Ok("compact") => LogFormat::Compact,
            _ => {
                if std::env::var("KUBERNETES_SERVICE_HOST").is_ok() {
                    LogFormat::Json
                } else {
                    LogFormat::Text
                }
            }
        }
    }
}

/// GCP Cloud Logging severity levels.
///
/// Maps tracing levels to GCP-recognized severity strings.
/// See: https://cloud.google.com/logging/docs/reference/v2/rest/v2/LogEntry#LogSeverity
fn gcp_severity(level: &tracing::Level) -> &'static str {
    match *level {
        tracing::Level::ERROR => "ERROR",
        tracing::Level::WARN => "WARNING",
        tracing::Level::INFO => "INFO",
        tracing::Level::DEBUG => "DEBUG",
        tracing::Level::TRACE => "DEBUG",  // GCP doesn't have TRACE
    }
}

/// Custom JSON formatter for Google Cloud Logging compatibility.
///
/// Outputs logs in a format that Cloud Logging automatically parses:
/// - `severity` field for log level
/// - `message` field for the log message
/// - `timestamp` in RFC3339 format
/// - All span fields flattened into the JSON object
pub struct GcpJsonFormatter;

impl<S, N> fmt::FormatEvent<S, N> for GcpJsonFormatter
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &fmt::FmtContext<'_, S, N>,
        mut writer: fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        use std::fmt::Write;
        use tracing_subscriber::field::Visit;

        // Collect event fields
        struct FieldCollector {
            fields: serde_json::Map<String, serde_json::Value>,
            message: Option<String>,
        }

        impl Visit for FieldCollector {
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                if field.name() == "message" {
                    self.message = Some(value.to_string());
                } else {
                    self.fields.insert(
                        field.name().to_string(),
                        serde_json::Value::String(value.to_string()),
                    );
                }
            }

            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                let value_str = format!("{:?}", value);
                if field.name() == "message" {
                    self.message = Some(value_str);
                } else {
                    self.fields.insert(
                        field.name().to_string(),
                        serde_json::Value::String(value_str),
                    );
                }
            }

            fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
                self.fields.insert(
                    field.name().to_string(),
                    serde_json::Value::Number(value.into()),
                );
            }

            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                self.fields.insert(
                    field.name().to_string(),
                    serde_json::Value::Number(value.into()),
                );
            }

            fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
                if let Some(n) = serde_json::Number::from_f64(value) {
                    self.fields.insert(field.name().to_string(), serde_json::Value::Number(n));
                }
            }

            fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
                self.fields.insert(field.name().to_string(), serde_json::Value::Bool(value));
            }
        }

        let mut collector = FieldCollector {
            fields: serde_json::Map::new(),
            message: None,
        };
        event.record(&mut collector);

        // Add span context fields
        if let Some(scope) = ctx.event_scope() {
            for span in scope.from_root() {
                let extensions = span.extensions();
                if let Some(fields) = extensions.get::<fmt::FormattedFields<N>>() {
                    // Parse span fields (simplified - in production use proper parsing)
                    let fields_str = fields.to_string();
                    if !fields_str.is_empty() {
                        collector.fields.insert(
                            format!("span.{}", span.name()),
                            serde_json::Value::String(fields_str),
                        );
                    }
                }
            }
        }

        // Build the log entry
        let mut log_entry = serde_json::Map::new();

        // Required fields for Cloud Logging
        log_entry.insert(
            "severity".to_string(),
            serde_json::Value::String(gcp_severity(event.metadata().level()).to_string()),
        );
        log_entry.insert(
            "message".to_string(),
            serde_json::Value::String(
                collector.message.unwrap_or_else(|| "".to_string())
            ),
        );
        log_entry.insert(
            "timestamp".to_string(),
            serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
        );

        // Add target (module path)
        log_entry.insert(
            "logging.googleapis.com/sourceLocation".to_string(),
            serde_json::json!({
                "file": event.metadata().file().unwrap_or("unknown"),
                "line": event.metadata().line().unwrap_or(0),
                "function": event.metadata().target(),
            }),
        );

        // Add all collected fields
        for (key, value) in collector.fields {
            log_entry.insert(key, value);
        }

        // Write JSON
        let json = serde_json::Value::Object(log_entry);
        writeln!(writer, "{}", json)
    }
}

/// Initialize the logging system.
///
/// Call this early in main() before any logging occurs.
///
/// # Arguments
///
/// * `service_name` - The name of this service (e.g., "actor", "web")
/// * `default_level` - Default log level if RUST_LOG is not set
pub fn init_logging(service_name: &str, default_level: &str) {
    let format = LogFormat::from_env();

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            // Apply default level with reduced noise from dependencies
            EnvFilter::new(format!("{},ort=warn,h2=warn,hyper=warn", default_level))
        });

    match format {
        LogFormat::Text => {
            tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .with_target(true)
                        .with_thread_ids(false)
                        .with_file(true)
                        .with_line_number(true)
                        .with_ansi(true)
                )
                .init();
        }
        LogFormat::Json => {
            tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .event_format(GcpJsonFormatter)
                        .with_ansi(false)
                )
                .init();

            // Log service metadata once at startup
            tracing::info!(
                service = service_name,
                version = env!("CARGO_PKG_VERSION"),
                log_format = "json",
                "Service starting with structured logging"
            );
        }
        LogFormat::Compact => {
            tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .with_target(true)
                        .with_thread_ids(false)
                        .with_ansi(false)
                        .compact()
                )
                .init();
        }
    }
}

/// Generate a new trace ID for request correlation.
///
/// Uses UUID v4 format compatible with Cloud Trace.
pub fn generate_trace_id() -> String {
    uuid::Uuid::new_v4().to_string().replace("-", "")
}

/// Parse Cloud Trace context header.
///
/// Format: `TRACE_ID/SPAN_ID;o=TRACE_TRUE`
pub fn parse_cloud_trace_context(header: Option<&str>) -> (String, Option<String>) {
    match header {
        Some(ctx) => {
            let parts: Vec<&str> = ctx.split('/').collect();
            let trace_id = parts.get(0).unwrap_or(&"").to_string();
            let span_id = parts.get(1).map(|s| s.split(';').next().unwrap_or("").to_string());

            if trace_id.is_empty() {
                (generate_trace_id(), None)
            } else {
                (trace_id, span_id)
            }
        }
        None => (generate_trace_id(), None),
    }
}
```

### 1.3 Update Actor Main

**File:** `actor/src/main.rs`

```rust
mod logging;

use tracing::{info, warn, error, debug, instrument, Span};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging first (before any other initialization)
    let log_level = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| "info".to_string());
    logging::init_logging("actor", &log_level);

    info!(
        actor_id = %config.actor_id,
        env_id = %config.env_id,
        max_episodes = config.max_episodes,
        mcts_simulations = config.mcts.num_simulations,
        "Actor configuration loaded"
    );

    // ... rest of main
}

// Add #[instrument] to key functions for automatic span creation
#[instrument(
    skip(ctx, storage, evaluator, rng),
    fields(
        episode_num = %episode_num,
        game = %game_name,
    )
)]
async fn run_episode(
    episode_num: u64,
    game_name: &str,
    ctx: &mut EngineContext,
    storage: &dyn Storage,
    evaluator: &dyn Evaluator,
    rng: &mut ChaCha20Rng,
) -> Result<EpisodeResult, EpisodeError> {
    let start = std::time::Instant::now();

    // Log episode start
    debug!("Episode starting");

    // ... episode logic ...

    let duration_ms = start.elapsed().as_millis();

    match &result {
        Ok(episode) => {
            info!(
                moves = episode.moves.len(),
                outcome = ?episode.outcome,
                duration_ms = duration_ms,
                searches = episode.search_count,
                "Episode completed"
            );
        }
        Err(e) => {
            error!(
                error = %e,
                duration_ms = duration_ms,
                "Episode failed"
            );
        }
    }

    result
}
```

### 1.4 Update Web Server with Request Tracing

**File:** `web/src/main.rs`

```rust
mod logging;

use axum::{
    extract::Request,
    middleware::{self, Next},
    response::Response,
};
use tracing::{info, warn, error, Span, info_span};

/// Middleware to add request tracing with Cloud Trace integration.
async fn trace_request(request: Request, next: Next) -> Response {
    let start = std::time::Instant::now();

    // Extract or generate trace ID
    let (trace_id, span_id) = logging::parse_cloud_trace_context(
        request.headers()
            .get("x-cloud-trace-context")
            .and_then(|v| v.to_str().ok())
    );

    // Create request span
    let span = info_span!(
        "http_request",
        method = %request.method(),
        uri = %request.uri().path(),
        trace_id = %trace_id,
        span_id = ?span_id,
    );

    let _guard = span.enter();

    // Process request
    let response = next.run(request).await;

    // Log request completion
    let duration_ms = start.elapsed().as_millis();
    let status = response.status().as_u16();

    if status >= 500 {
        error!(
            status = status,
            duration_ms = duration_ms,
            trace_id = %trace_id,
            "Request failed"
        );
    } else if status >= 400 {
        warn!(
            status = status,
            duration_ms = duration_ms,
            trace_id = %trace_id,
            "Request error"
        );
    } else {
        info!(
            status = status,
            duration_ms = duration_ms,
            trace_id = %trace_id,
            "Request completed"
        );
    }

    response
}

// In router setup:
let app = Router::new()
    .route("/health", get(health_handler))
    .route("/game/new", post(new_game_handler))
    .route("/game/state", get(get_state_handler))
    .route("/move", post(move_handler))
    .route("/stats", get(stats_handler))
    .route("/metrics", get(metrics_handler))
    .layer(middleware::from_fn(trace_request));
```

### 1.5 Structured Fields Reference (Rust)

| Context | Field | Type | Example |
|---------|-------|------|---------|
| **All** | `severity` | string | "INFO", "ERROR" |
| **All** | `timestamp` | string | "2024-01-15T10:30:00Z" |
| **All** | `service` | string | "actor", "web" |
| **Episode** | `episode_num` | u64 | 12345 |
| **Episode** | `game` | string | "tictactoe" |
| **Episode** | `moves` | usize | 9 |
| **Episode** | `outcome` | string | "Player1Win" |
| **Episode** | `duration_ms` | u128 | 150 |
| **MCTS** | `searches` | u64 | 800 |
| **MCTS** | `inference_ms` | f64 | 5.2 |
| **HTTP** | `method` | string | "POST" |
| **HTTP** | `uri` | string | "/move" |
| **HTTP** | `status` | u16 | 200 |
| **HTTP** | `trace_id` | string | "abc123..." |
| **Storage** | `transitions` | usize | 50 |
| **Storage** | `db_write_ms` | f64 | 12.5 |
| **Model** | `model_path` | string | "/data/models/latest.onnx" |
| **Model** | `reload_ms` | f64 | 250.0 |

---

## Phase 2: Python Trainer

### 2.1 Dependencies

**File:** `trainer/pyproject.toml`

```toml
dependencies = [
    # ... existing deps ...
    "structlog>=24.1.0",  # Structured logging
    "python-json-logger>=2.0.0",  # JSON formatting fallback
]
```

### 2.2 Logging Configuration Module

**File:** `trainer/src/trainer/logging_config.py` (new)

```python
"""Structured logging configuration for cloud deployments.

Supports both human-readable text (local development) and JSON (production).
"""

import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog

# Context variable for trace ID propagation
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_span_id: ContextVar[str] = ContextVar("span_id", default="")


def get_trace_id() -> str:
    """Get current trace ID from context."""
    return _trace_id.get()


def set_trace_id(trace_id: str) -> None:
    """Set trace ID in context."""
    _trace_id.set(trace_id)


def generate_trace_id() -> str:
    """Generate a new trace ID (UUID without dashes)."""
    import uuid
    return uuid.uuid4().hex


class CloudLoggingProcessor:
    """Structlog processor that formats output for Google Cloud Logging.

    Produces JSON with fields that Cloud Logging automatically parses:
    - severity: Log level (INFO, WARNING, ERROR, DEBUG)
    - message: The log message
    - timestamp: RFC3339 format
    - Additional fields as jsonPayload
    """

    SEVERITY_MAP = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "warn": "WARNING",
        "error": "ERROR",
        "critical": "CRITICAL",
        "exception": "ERROR",
    }

    def __call__(
        self,
        logger: logging.Logger,
        method_name: str,
        event_dict: Dict[str, Any],
    ) -> str:
        import json

        # Extract standard fields
        timestamp = datetime.now(timezone.utc).isoformat()
        severity = self.SEVERITY_MAP.get(method_name, "DEFAULT")
        message = event_dict.pop("event", "")

        # Build Cloud Logging-compatible structure
        log_entry = {
            "severity": severity,
            "message": message,
            "timestamp": timestamp,
        }

        # Add trace context if available
        trace_id = get_trace_id()
        if trace_id:
            log_entry["logging.googleapis.com/trace"] = (
                f"projects/{os.environ.get('GCP_PROJECT', 'unknown')}/traces/{trace_id}"
            )

        # Add source location
        if "filename" in event_dict and "lineno" in event_dict:
            log_entry["logging.googleapis.com/sourceLocation"] = {
                "file": event_dict.pop("filename"),
                "line": str(event_dict.pop("lineno")),
                "function": event_dict.pop("func_name", ""),
            }

        # Remaining fields go into the main object (Cloud Logging parses as jsonPayload)
        log_entry.update(event_dict)

        return json.dumps(log_entry, default=str)


class ConsoleProcessor:
    """Structlog processor for human-readable console output."""

    COLORS = {
        "debug": "\033[36m",     # Cyan
        "info": "\033[32m",      # Green
        "warning": "\033[33m",   # Yellow
        "error": "\033[31m",     # Red
        "critical": "\033[35m",  # Magenta
        "reset": "\033[0m",
    }

    def __call__(
        self,
        logger: logging.Logger,
        method_name: str,
        event_dict: Dict[str, Any],
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = method_name.upper()
        message = event_dict.pop("event", "")

        # Color the level
        color = self.COLORS.get(method_name, "")
        reset = self.COLORS["reset"]

        # Format additional fields
        extra = ""
        if event_dict:
            extra_items = [f"{k}={v}" for k, v in event_dict.items()
                         if k not in ("filename", "lineno", "func_name")]
            if extra_items:
                extra = f" [{', '.join(extra_items)}]"

        return f"{timestamp} {color}[{level:8}]{reset} {message}{extra}"


def setup_logging(
    service_name: str = "trainer",
    level: str = "INFO",
    force_format: Optional[str] = None,
) -> None:
    """Initialize structured logging for the trainer.

    Args:
        service_name: Name of this service (for log metadata)
        level: Default log level (DEBUG, INFO, WARNING, ERROR)
        force_format: Override format detection ("json" or "text")

    Format Detection:
        1. force_format parameter (if provided)
        2. LOG_FORMAT environment variable
        3. Auto-detect: JSON if KUBERNETES_SERVICE_HOST is set
        4. Default: text
    """
    # Determine format
    if force_format:
        use_json = force_format == "json"
    elif os.environ.get("LOG_FORMAT"):
        use_json = os.environ.get("LOG_FORMAT") == "json"
    elif os.environ.get("KUBERNETES_SERVICE_HOST"):
        use_json = True
    else:
        use_json = False

    # Configure structlog
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if use_json:
        processors.append(CloudLoggingProcessor())
    else:
        processors.append(ConsoleProcessor())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Silence noisy third-party loggers
    for noisy_logger in [
        "onnx_ir.passes.common.unused_removal",
        "onnx_ir.passes.common.initializer_deduplication",
        "onnxscript.optimizer._constant_folding",
        "onnxscript.rewriter.rules.common._collapse_slices",
        "urllib3.connectionpool",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Log startup
    logger = get_logger(__name__)
    logger.info(
        "Logging initialized",
        service=service_name,
        level=level,
        format="json" if use_json else "text",
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Training started", iteration=5, learning_rate=0.001)
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager to add fields to all logs within a block.

    Usage:
        with LogContext(iteration=5, env_id="tictactoe"):
            logger.info("Starting iteration")  # Includes iteration=5, env_id=tictactoe
    """

    def __init__(self, **kwargs):
        self.context = kwargs
        self.token = None

    def __enter__(self):
        structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args):
        structlog.contextvars.unbind_contextvars(*self.context.keys())
```

### 2.3 Update Trainer Main

**File:** `trainer/src/trainer/__main__.py`

```python
from trainer.logging_config import setup_logging, get_logger, LogContext

def main():
    # Parse args first to get log level
    args = parse_args()

    # Initialize structured logging
    setup_logging(
        service_name="trainer",
        level=args.log_level or "INFO",
    )

    logger = get_logger(__name__)

    logger.info(
        "Trainer starting",
        version="1.0.0",
        command=args.command,
        env_id=args.env_id,
        device=str(args.device),
    )

    if args.command == "loop":
        run_training_loop(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "evaluate":
        run_evaluation(args)


def run_training_loop(args):
    logger = get_logger(__name__)

    for iteration in range(args.start_iteration, args.iterations):
        with LogContext(iteration=iteration, phase="training"):
            logger.info("Starting iteration")

            # Self-play phase
            with LogContext(sub_phase="self_play"):
                episodes = run_self_play(args)
                logger.info(
                    "Self-play complete",
                    episodes_generated=episodes,
                )

            # Training phase
            with LogContext(sub_phase="train"):
                for step in range(args.steps_per_iteration):
                    loss_dict = train_step(...)

                    if step % 100 == 0:
                        logger.info(
                            "Training progress",
                            step=step,
                            total_loss=float(loss_dict["loss/total"]),
                            value_loss=float(loss_dict["loss/value"]),
                            policy_loss=float(loss_dict["loss/policy"]),
                            learning_rate=float(lr),
                        )

                logger.info(
                    "Training phase complete",
                    final_loss=float(loss_dict["loss/total"]),
                )

            # Evaluation phase
            if args.eval_interval > 0 and iteration % args.eval_interval == 0:
                with LogContext(sub_phase="evaluate"):
                    stats = run_evaluation(...)
                    logger.info(
                        "Evaluation complete",
                        win_rate=stats.win_rate,
                        draw_rate=stats.draw_rate,
                        loss_rate=stats.loss_rate,
                        games_played=stats.total_games,
                    )

            logger.info("Iteration complete")
```

### 2.4 Structured Fields Reference (Python)

| Context | Field | Type | Example |
|---------|-------|------|---------|
| **All** | `severity` | string | "INFO", "ERROR" |
| **All** | `timestamp` | string | "2024-01-15T10:30:00Z" |
| **All** | `service` | string | "trainer" |
| **Iteration** | `iteration` | int | 5 |
| **Iteration** | `phase` | string | "training" |
| **Iteration** | `sub_phase` | string | "self_play", "train", "evaluate" |
| **Training** | `step` | int | 500 |
| **Training** | `total_loss` | float | 0.543 |
| **Training** | `value_loss` | float | 0.123 |
| **Training** | `policy_loss` | float | 0.420 |
| **Training** | `learning_rate` | float | 0.0001 |
| **Training** | `gradient_norm` | float | 1.234 |
| **Evaluation** | `win_rate` | float | 0.85 |
| **Evaluation** | `draw_rate` | float | 0.10 |
| **Evaluation** | `loss_rate` | float | 0.05 |
| **Checkpoint** | `checkpoint_path` | string | "/data/checkpoints/iter_5.pt" |
| **Checkpoint** | `onnx_path` | string | "/data/models/latest.onnx" |
| **Checkpoint** | `save_duration_ms` | float | 1250.5 |

---

## Phase 3: Configuration & Environment

### 3.1 Update config.toml Schema

**File:** `config.toml`

```toml
[common]
data_dir = "./data"
env_id = "tictactoe"
log_level = "info"  # trace, debug, info, warn, error

# NEW: Logging configuration
[logging]
# Output format: "auto", "json", "text", "compact"
# "auto" = JSON if KUBERNETES_SERVICE_HOST is set, else text
format = "auto"

# Include source location in logs (file:line)
include_source_location = true

# Sampling rate for high-frequency logs (0.0-1.0, 1.0 = log all)
# Applies to DEBUG level logs only
debug_sample_rate = 1.0
```

### 3.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_FORMAT` | Override format: `json`, `text`, `compact` | auto |
| `RUST_LOG` | Rust log filter (e.g., `info,actor=debug`) | `info` |
| `LOG_LEVEL` | Python log level | `INFO` |
| `GCP_PROJECT` | GCP project ID (for trace correlation) | `unknown` |

### 3.3 Kubernetes Deployment Updates

**File:** `k8s/base/actor/deployment.yaml`

```yaml
spec:
  template:
    spec:
      containers:
        - name: actor
          env:
            - name: LOG_FORMAT
              value: "json"
            - name: RUST_LOG
              value: "info,actor=debug,mcts=info,ort=warn"
            - name: GCP_PROJECT
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['gcp-project']
```

**File:** `k8s/base/trainer/deployment.yaml`

```yaml
spec:
  template:
    spec:
      containers:
        - name: trainer
          env:
            - name: LOG_FORMAT
              value: "json"
            - name: LOG_LEVEL
              value: "INFO"
            - name: GCP_PROJECT
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['gcp-project']
```

---

## Phase 4: Google Cloud Integration

### 4.1 Cloud Logging Setup

No additional setup required! GKE automatically:
1. Runs Fluent Bit on each node
2. Collects container stdout/stderr
3. Parses JSON logs into `jsonPayload`
4. Ships to Cloud Logging

### 4.2 Log-Based Metrics

Create metrics from logs for alerting without Prometheus.

**File:** `terraform/logging_metrics.tf`

```hcl
# Training stall detection
resource "google_logging_metric" "training_steps" {
  name        = "trainer/steps_total"
  description = "Count of training steps completed"
  filter      = <<-EOT
    resource.type="k8s_container"
    resource.labels.namespace_name="cartridge"
    resource.labels.container_name="trainer"
    jsonPayload.message="Training progress"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

# Error rate by service
resource "google_logging_metric" "errors_by_service" {
  name        = "cartridge/errors"
  description = "Error count by service"
  filter      = <<-EOT
    resource.type="k8s_container"
    resource.labels.namespace_name="cartridge"
    severity>=ERROR
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
    labels {
      key         = "service"
      value_type  = "STRING"
      description = "Service name"
    }
  }

  label_extractors = {
    "service" = "EXTRACT(resource.labels.container_name)"
  }
}

# Episode completion rate
resource "google_logging_metric" "episodes_completed" {
  name        = "actor/episodes_completed"
  description = "Count of episodes completed"
  filter      = <<-EOT
    resource.type="k8s_container"
    resource.labels.container_name="actor"
    jsonPayload.message="Episode completed"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

# Slow requests
resource "google_logging_metric" "slow_requests" {
  name        = "web/slow_requests"
  description = "HTTP requests taking >1s"
  filter      = <<-EOT
    resource.type="k8s_container"
    resource.labels.container_name="web"
    jsonPayload.message="Request completed"
    jsonPayload.duration_ms > 1000
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}
```

### 4.3 Alerts from Log-Based Metrics

**File:** `terraform/alerts.tf`

```hcl
# Alert: Training stalled
resource "google_monitoring_alert_policy" "training_stalled" {
  display_name = "Training Stalled"
  combiner     = "OR"

  conditions {
    display_name = "No training steps in 15 minutes"
    condition_absent {
      filter   = "resource.type=\"k8s_container\" AND metric.type=\"logging.googleapis.com/user/trainer/steps_total\""
      duration = "900s"

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]
}

# Alert: High error rate
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "High Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "Error rate > 10 per minute"
    condition_threshold {
      filter          = "resource.type=\"k8s_container\" AND metric.type=\"logging.googleapis.com/user/cartridge/errors\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 10

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]
}
```

### 4.4 Cloud Logging Queries

Save these in the Cloud Logging console as "Saved Queries":

```
# All errors across services
resource.type="k8s_container"
resource.labels.namespace_name="cartridge"
severity>=ERROR

# Actor episode completions with outcome
resource.labels.container_name="actor"
jsonPayload.message="Episode completed"
| jsonPayload.outcome jsonPayload.duration_ms jsonPayload.moves

# Training progress over time
resource.labels.container_name="trainer"
jsonPayload.message="Training progress"
| jsonPayload.step jsonPayload.total_loss jsonPayload.learning_rate

# Slow episodes (>1 second)
resource.labels.container_name="actor"
jsonPayload.message="Episode completed"
jsonPayload.duration_ms > 1000

# HTTP requests by status
resource.labels.container_name="web"
jsonPayload.message="Request completed"
| jsonPayload.status jsonPayload.uri jsonPayload.duration_ms

# Trace a specific request
jsonPayload.trace_id="TRACE_ID_HERE"

# Model reload events
jsonPayload.message=~"model.*reload"

# High-loss training steps
resource.labels.container_name="trainer"
jsonPayload.message="Training progress"
jsonPayload.total_loss > 1.0
```

### 4.5 Log Export to BigQuery

For long-term analysis and ML experiment tracking:

**File:** `terraform/log_sink.tf`

```hcl
# Create BigQuery dataset for logs
resource "google_bigquery_dataset" "logs" {
  dataset_id  = "cartridge_logs"
  location    = "US"
  description = "Cartridge training logs for analysis"

  default_table_expiration_ms = 7776000000  # 90 days
}

# Create log sink to BigQuery
resource "google_logging_project_sink" "bigquery" {
  name        = "cartridge-logs-to-bigquery"
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.logs.dataset_id}"

  filter = <<-EOT
    resource.type="k8s_container"
    resource.labels.namespace_name="cartridge"
    (
      jsonPayload.message="Episode completed" OR
      jsonPayload.message="Training progress" OR
      jsonPayload.message="Evaluation complete" OR
      jsonPayload.message="Iteration complete"
    )
  EOT

  unique_writer_identity = true
  bigquery_options {
    use_partitioned_tables = true
  }
}

# Grant sink writer access
resource "google_bigquery_dataset_iam_member" "logs_writer" {
  dataset_id = google_bigquery_dataset.logs.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.bigquery.writer_identity
}
```

**Example BigQuery Analysis:**

```sql
-- Training loss over time
SELECT
  TIMESTAMP_TRUNC(timestamp, HOUR) as hour,
  AVG(CAST(jsonPayload.total_loss AS FLOAT64)) as avg_loss,
  MIN(CAST(jsonPayload.total_loss AS FLOAT64)) as min_loss,
  MAX(CAST(jsonPayload.total_loss AS FLOAT64)) as max_loss
FROM `project.cartridge_logs.stdout_*`
WHERE jsonPayload.message = "Training progress"
GROUP BY hour
ORDER BY hour;

-- Episode outcomes by game
SELECT
  jsonPayload.game,
  jsonPayload.outcome,
  COUNT(*) as count,
  AVG(CAST(jsonPayload.duration_ms AS FLOAT64)) as avg_duration_ms
FROM `project.cartridge_logs.stdout_*`
WHERE jsonPayload.message = "Episode completed"
GROUP BY jsonPayload.game, jsonPayload.outcome;

-- Evaluation win rate trend
SELECT
  TIMESTAMP_TRUNC(timestamp, DAY) as day,
  AVG(CAST(jsonPayload.win_rate AS FLOAT64)) as avg_win_rate
FROM `project.cartridge_logs.stdout_*`
WHERE jsonPayload.message = "Evaluation complete"
GROUP BY day
ORDER BY day;
```

---

## Phase 5: Future - OpenTelemetry (Optional)

For full distributed tracing with Cloud Trace integration:

### 5.1 Rust OpenTelemetry Integration

```toml
# Cargo.toml
[dependencies]
tracing-opentelemetry = "0.22"
opentelemetry = { version = "0.21", features = ["rt-tokio"] }
opentelemetry-otlp = "0.14"
opentelemetry_sdk = { version = "0.21", features = ["rt-tokio"] }
```

```rust
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use tracing_subscriber::layer::SubscriberExt;

fn init_tracing_with_otel(service_name: &str) {
    // Set up OTLP exporter (works with Cloud Trace)
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint("http://localhost:4317"),
        )
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .expect("Failed to install tracer");

    // Create OpenTelemetry layer
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Combine with existing logging
    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(otel_layer)
        .with(fmt::layer().json())
        .init();
}
```

### 5.2 Python OpenTelemetry Integration

```python
# pyproject.toml
dependencies = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp>=1.20.0",
]
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def init_tracing(service_name: str):
    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

# Usage
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("training_iteration") as span:
    span.set_attribute("iteration", 5)
    # ... training code ...
```

---

## Implementation Order

| Phase | Description | Effort | Impact |
|-------|-------------|--------|--------|
| 1 | Rust logging module (actor + web) | 3-4 hours | HIGH - Enables debugging |
| 2 | Python structlog setup | 2-3 hours | HIGH - Training visibility |
| 3 | Config & environment | 1 hour | MEDIUM - Consistency |
| 4 | Cloud Logging integration | 2-3 hours | HIGH - Production ready |
| 5 | OpenTelemetry (future) | 4-6 hours | MEDIUM - Full tracing |

**Total Estimated Effort:** 8-11 hours (Phases 1-4)

---

## Testing

### Local Testing

```bash
# Test JSON output (Rust)
LOG_FORMAT=json cargo run --manifest-path actor/Cargo.toml 2>&1 | jq .

# Test JSON output (Python)
LOG_FORMAT=json python -m trainer train --steps 10 2>&1 | jq .

# Validate JSON structure
LOG_FORMAT=json cargo run 2>&1 | head -1 | jq 'has("severity", "message", "timestamp")'
```

### Cloud Logging Verification

```bash
# After deploying to GKE, verify logs appear
gcloud logging read \
  'resource.type="k8s_container" AND resource.labels.namespace_name="cartridge"' \
  --limit=10 \
  --format=json | jq '.[] | .jsonPayload'
```

---

## Validation Checklist

- [ ] Rust services output JSON when `LOG_FORMAT=json`
- [ ] Python trainer outputs JSON when `LOG_FORMAT=json`
- [ ] JSON includes `severity`, `message`, `timestamp` fields
- [ ] Source location included in logs
- [ ] Trace ID propagated through HTTP headers
- [ ] Local development uses readable text format
- [ ] Noisy third-party loggers are silenced
- [ ] Logs appear in Cloud Logging console
- [ ] Log severity correctly mapped
- [ ] Custom fields are queryable in Cloud Logging
- [ ] Log-based metrics created
- [ ] Alerts configured for critical conditions
- [ ] BigQuery export working (if enabled)

---

## Files Summary

### New Files

| File | Description |
|------|-------------|
| `actor/src/logging.rs` | Rust structured logging module |
| `web/src/logging.rs` | Web server logging (similar to actor) |
| `trainer/src/trainer/logging_config.py` | Python structlog configuration |
| `terraform/logging_metrics.tf` | Log-based metrics definitions |
| `terraform/alerts.tf` | Alerting policies |
| `terraform/log_sink.tf` | BigQuery export configuration |

### Modified Files

| File | Changes |
|------|---------|
| `actor/Cargo.toml` | Add `uuid`, `serde_json`, `chrono` |
| `actor/src/main.rs` | Use new logging module |
| `actor/src/lib.rs` | Add `mod logging;` |
| `web/Cargo.toml` | Add `uuid`, `serde_json`, `chrono` |
| `web/src/main.rs` | Use new logging module + request tracing |
| `trainer/pyproject.toml` | Add `structlog` |
| `trainer/src/trainer/__main__.py` | Use new logging config |
| `config.toml` | Add `[logging]` section |
| `k8s/base/*/deployment.yaml` | Add `LOG_FORMAT=json` env var |

---

## Relationship to Prometheus Metrics

| Aspect | Structured Logging | Prometheus Metrics |
|--------|-------------------|-------------------|
| **Purpose** | Debug, audit, trace | Monitor, alert, dashboard |
| **Query** | "What happened to episode X?" | "What's the episode rate?" |
| **Cardinality** | Unlimited (per-event) | Limited (aggregated) |
| **Retention** | 30 days (Cloud Logging) | 15 days (Prometheus) |
| **Cost** | Higher (more data) | Lower (time series only) |
| **Real-time** | Seconds delay | Near real-time |

**Use Logging For:**
- Debugging specific failures
- Audit trails
- Detailed error context
- Request tracing

**Use Metrics For:**
- Dashboards
- Alerting on thresholds
- Capacity planning
- SLO tracking

Both are complementary and should be implemented together for full observability.
