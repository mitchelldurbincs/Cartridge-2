# Structured Logging Implementation Plan

## Executive Summary

This document outlines a comprehensive plan for implementing structured logging across all Cartridge-2 components (Actor, Web, Trainer) to enable effective observability when deployed to Google Cloud. The plan builds upon the existing Prometheus metrics infrastructure and transforms text-based logs into JSON-structured logs compatible with Google Cloud Logging.

**Current State:**
- Rust components use `tracing` with text output
- Python components use standard `logging` with string formatting
- No correlation IDs for distributed tracing
- No JSON output for cloud log aggregation

**Target State:**
- All components emit JSON-structured logs
- Correlation IDs link related operations across components
- Google Cloud Logging severity levels properly mapped
- Logs complement existing Prometheus metrics

---

## Table of Contents

1. [Google Cloud Logging Requirements](#1-google-cloud-logging-requirements)
2. [Current Logging Analysis](#2-current-logging-analysis)
3. [Implementation Strategy](#3-implementation-strategy)
4. [Phase 1: Rust Components](#4-phase-1-rust-components)
5. [Phase 2: Python Components](#5-phase-2-python-components)
6. [Phase 3: Correlation & Tracing](#6-phase-3-correlation--tracing)
7. [Phase 4: Google Cloud Integration](#7-phase-4-google-cloud-integration)
8. [Log Schema Standards](#8-log-schema-standards)
9. [Testing & Validation](#9-testing--validation)
10. [Migration Checklist](#10-migration-checklist)

---

## 1. Google Cloud Logging Requirements

### 1.1 Structured Log Format

Google Cloud Logging automatically parses JSON logs and extracts fields for querying. The expected format:

```json
{
  "severity": "INFO",
  "message": "Episode completed",
  "timestamp": "2025-01-12T14:30:45.123456789Z",
  "logging.googleapis.com/labels": {
    "component": "actor",
    "env_id": "connect4",
    "actor_id": "actor-1"
  },
  "logging.googleapis.com/trace": "projects/PROJECT_ID/traces/TRACE_ID",
  "logging.googleapis.com/spanId": "SPAN_ID",
  "episode": 1234,
  "duration_ms": 45.2,
  "outcome": "player1_win"
}
```

### 1.2 Severity Level Mapping

| Cartridge Level | Google Cloud Severity | Numeric Value |
|-----------------|----------------------|---------------|
| `trace`         | DEBUG                | 100           |
| `debug`         | DEBUG                | 100           |
| `info`          | INFO                 | 200           |
| `warn`          | WARNING              | 400           |
| `error`         | ERROR                | 500           |

### 1.3 Required Fields for Cloud Logging

| Field | Description | Source |
|-------|-------------|--------|
| `severity` | Log level (must be uppercase) | Log event level |
| `message` | Human-readable message | Log message |
| `timestamp` | RFC 3339 format | System time |
| `logging.googleapis.com/labels` | Indexed labels for filtering | Component metadata |
| `logging.googleapis.com/trace` | Trace ID for correlation | Request/operation context |

### 1.4 Cost Considerations

- Google Cloud Logging charges per log volume ingested
- Structured logs enable efficient filtering (reduce query costs)
- Recommend: Keep `debug`/`trace` logs disabled in production
- Use log exclusion filters for high-volume, low-value logs

---

## 2. Current Logging Analysis

### 2.1 Rust Components (Actor, Web)

**Current Implementation:**
```rust
// actor/src/main.rs
fn init_tracing(level: &str) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level))
        .add_directive("ort=warn".parse().unwrap());
    tracing_subscriber::registry()
        .with(fmt::layer())  // Text output
        .with(filter)
        .init();
    Ok(())
}
```

**Existing Structured Fields (Good Foundation):**
```rust
info!(
    episode = episode_num,
    searches = self.search_count,
    total_ms = format!("{:.1}", total_ms),
    "MCTS episode stats"
);
```

**Gap:** Uses text format, not JSON. Structured fields exist but aren't exported in machine-readable format.

### 2.2 Python Components (Trainer)

**Current Implementation:**
```python
# trainer/__main__.py
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

**Gap:** Uses string formatting with no structured fields. All context is embedded in message strings:
```python
logger.info(f"Training complete! Final loss: {stats.total_loss:.4f}")
# Should be:
logger.info("Training complete", extra={"total_loss": stats.total_loss})
```

### 2.3 Prometheus Metrics Integration

Existing metrics provide quantitative data. Structured logs should provide qualitative context:

| Metric | Complementary Log Event |
|--------|------------------------|
| `actor_episodes_total` | Episode lifecycle events with context |
| `trainer_loss_total` | Training step with loss breakdown |
| `web_request_duration_seconds` | Request with user/game context |
| `actor_model_reloads_total` | Model reload with version/path info |

---

## 3. Implementation Strategy

### 3.1 Phased Approach

```
Phase 1: Rust JSON Output (Actor, Web)     [Week 1]
    └── Switch tracing to JSON format
    └── Add component metadata
    └── Environment-based format switching

Phase 2: Python Structured Logging         [Week 1-2]
    └── Implement python-json-logger
    └── Refactor log calls to use extra fields
    └── Add component metadata

Phase 3: Correlation & Tracing             [Week 2]
    └── Add trace/span ID generation
    └── Propagate IDs across components
    └── Link related operations

Phase 4: Google Cloud Integration          [Week 2-3]
    └── Deploy to GKE/Cloud Run
    └── Configure log sinks and filters
    └── Create log-based metrics
    └── Set up alerts
```

### 3.2 Design Principles

1. **Backward Compatibility:** Text format for local dev, JSON for cloud
2. **Minimal Code Changes:** Leverage existing structured fields
3. **Complement Metrics:** Logs for context, metrics for aggregation
4. **Cost Awareness:** Configurable log levels per component

### 3.3 Configuration Extension

Add to `config.toml`:

```toml
[logging]
format = "text"           # "text" or "json"
include_timestamps = true # Include timestamps (false if cloud adds them)
include_location = false  # Include file:line (useful for debugging)
```

Environment override: `CARTRIDGE_LOGGING_FORMAT=json`

---

## 4. Phase 1: Rust Components

### 4.1 Dependencies

Add to `actor/Cargo.toml` and `web/Cargo.toml`:

```toml
[dependencies]
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
```

### 4.2 Actor Logging Changes

**File:** `actor/src/main.rs`

```rust
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn init_tracing(level: &str, json_format: bool) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level))
        .add_directive("ort=warn".parse().unwrap())
        .add_directive("h2=warn".parse().unwrap())
        .add_directive("hyper=warn".parse().unwrap());

    let registry = tracing_subscriber::registry().with(filter);

    if json_format {
        // JSON format for Google Cloud
        registry
            .with(
                fmt::layer()
                    .json()
                    .with_current_span(true)
                    .with_span_list(false)
                    .with_file(false)
                    .with_line_number(false)
                    .flatten_event(true)
                    .with_target(true)
            )
            .init();
    } else {
        // Human-readable for local development
        registry.with(fmt::layer()).init();
    }

    Ok(())
}
```

**JSON Output Example:**
```json
{
  "timestamp": "2025-01-12T14:30:45.123456Z",
  "level": "INFO",
  "target": "actor::actor",
  "message": "MCTS episode stats",
  "episode": 1234,
  "searches": 800,
  "total_ms": "45.2",
  "inference_pct": "32.1%",
  "expansion_pct": "15.3%"
}
```

### 4.3 Add Google Cloud Severity Mapping

Create `actor/src/logging.rs`:

```rust
use serde::Serialize;
use tracing::Level;
use tracing_subscriber::fmt::format::JsonFields;

/// Maps tracing levels to Google Cloud Logging severity
pub fn level_to_severity(level: &Level) -> &'static str {
    match *level {
        Level::ERROR => "ERROR",
        Level::WARN => "WARNING",
        Level::INFO => "INFO",
        Level::DEBUG => "DEBUG",
        Level::TRACE => "DEBUG",
    }
}

/// Custom JSON formatter that outputs Google Cloud compatible format
pub struct GcpJsonLayer;

// Implementation details for custom layer that:
// 1. Renames "level" to "severity"
// 2. Formats timestamp as RFC 3339
// 3. Adds component labels
```

### 4.4 Component Metadata

Add component identification to all logs:

```rust
// In actor startup
let actor_span = tracing::info_span!(
    "actor",
    component = "actor",
    actor_id = %config.actor_id,
    env_id = %config.env_id,
    version = env!("CARGO_PKG_VERSION"),
);
let _guard = actor_span.enter();

// All subsequent logs will include these fields
info!("Actor service starting");
```

### 4.5 Existing Log Statement Updates

Most existing logs already use structured fields. Key additions:

```rust
// Before (actor/src/actor.rs:111-124)
info!(
    episode = episode_num,
    searches = self.search_count,
    total_ms = format!("{:.1}", total_ms),
    "MCTS episode stats"
);

// After - add event type for filtering
info!(
    event = "episode_stats",
    episode = episode_num,
    searches = self.search_count,
    total_ms = total_ms,  // Use raw number, not formatted string
    inference_ms = inference_ms,
    "MCTS episode stats"
);
```

### 4.6 Web Server Updates

**File:** `web/src/main.rs`

Apply same JSON formatting pattern. Add request-scoped spans:

```rust
// In request handler
async fn handle_move(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MoveRequest>,
) -> Result<Json<MoveResponse>, AppError> {
    let span = tracing::info_span!(
        "handle_move",
        event = "http_request",
        endpoint = "/move",
        method = "POST",
        game_id = ?state.current_game_id(),
    );
    let _guard = span.enter();

    // ... handler logic
    info!(
        event = "move_completed",
        player_move = request.position,
        bot_move = bot_response.position,
        game_over = response.game_over,
        "Move processed"
    );
}
```

---

## 5. Phase 2: Python Components

### 5.1 Dependencies

Add to `trainer/pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing deps
    "python-json-logger>=2.0.0",
]
```

### 5.2 Create Logging Module

**File:** `trainer/src/trainer/structured_logging.py`

```python
"""Structured logging configuration for Google Cloud compatibility."""

import logging
import sys
from datetime import datetime, timezone
from typing import Any

from pythonjsonlogger import jsonlogger


class GoogleCloudJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter compatible with Google Cloud Logging."""

    LEVEL_TO_SEVERITY = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def __init__(self, *args, **kwargs):
        # Include these fields in every log
        kwargs.setdefault("reserved_attrs", [])
        super().__init__(*args, **kwargs)
        self.component = kwargs.get("component", "trainer")
        self.env_id = kwargs.get("env_id", "unknown")

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Google Cloud severity (required)
        log_record["severity"] = self.LEVEL_TO_SEVERITY.get(
            record.levelname, "DEFAULT"
        )

        # RFC 3339 timestamp
        log_record["timestamp"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        # Component labels for filtering
        log_record["logging.googleapis.com/labels"] = {
            "component": self.component,
            "env_id": self.env_id,
        }

        # Source location (useful for debugging)
        log_record["logging.googleapis.com/sourceLocation"] = {
            "file": record.pathname,
            "line": str(record.lineno),
            "function": record.funcName,
        }

        # Remove redundant fields
        log_record.pop("levelname", None)
        log_record.pop("name", None)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    component: str = "trainer",
    env_id: str = "unknown",
) -> None:
    """Configure logging for the trainer component.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, output JSON; otherwise human-readable
        component: Component name for labeling
        env_id: Game environment ID
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        formatter = GoogleCloudJsonFormatter(
            component=component,
            env_id=env_id,
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Silence noisy loggers
    for noisy in [
        "onnx_ir.passes.common.unused_removal",
        "onnxscript.optimizer._constant_folding",
        "urllib3.connectionpool",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
```

### 5.3 Structured Log Helper

**File:** `trainer/src/trainer/structured_logging.py` (continued)

```python
class StructuredLogger:
    """Logger wrapper that encourages structured logging."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def info(self, message: str, **fields) -> None:
        self.logger.info(message, extra=fields)

    def warning(self, message: str, **fields) -> None:
        self.logger.warning(message, extra=fields)

    def error(self, message: str, **fields) -> None:
        self.logger.error(message, extra=fields)

    def debug(self, message: str, **fields) -> None:
        self.logger.debug(message, extra=fields)

    def exception(self, message: str, **fields) -> None:
        self.logger.exception(message, extra=fields)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)
```

### 5.4 Trainer Log Statement Refactoring

**File:** `trainer/src/trainer/trainer.py`

```python
# Before
logger.info(f"Training complete! Final loss: {stats.total_loss:.4f}")

# After
from trainer.structured_logging import get_logger
logger = get_logger(__name__)

logger.info(
    "Training complete",
    event="training_complete",
    total_loss=round(stats.total_loss, 4),
    value_loss=round(stats.value_loss, 4),
    policy_loss=round(stats.policy_loss, 4),
    total_steps=stats.steps,
    duration_seconds=elapsed_time,
)
```

**JSON Output:**
```json
{
  "severity": "INFO",
  "timestamp": "2025-01-12T14:30:45.123456Z",
  "message": "Training complete",
  "event": "training_complete",
  "total_loss": 0.1234,
  "value_loss": 0.0567,
  "policy_loss": 0.0667,
  "total_steps": 10000,
  "duration_seconds": 3600.5,
  "logging.googleapis.com/labels": {
    "component": "trainer",
    "env_id": "connect4"
  }
}
```

### 5.5 Key Log Statements to Refactor

| Location | Current | Structured Version |
|----------|---------|-------------------|
| `trainer.py:221` | `f"Starting training for {steps} steps"` | `event="training_start", total_steps=steps` |
| `trainer.py:340` | Training step metrics | `event="training_step", step=n, loss=x` |
| `trainer.py:418` | Checkpoint save | `event="checkpoint_saved", path=p, size_mb=s` |
| `trainer.py:535` | Evaluation results | `event="evaluation_complete", win_rate=w` |
| `orchestrator.py:165` | Iteration start | `event="iteration_start", iteration=n` |
| `orchestrator.py:210` | Actor completed | `event="actor_complete", transitions=n` |

---

## 6. Phase 3: Correlation & Tracing

### 6.1 Trace ID Generation

Each logical operation gets a unique trace ID that propagates across components.

**Rust Implementation:**

```rust
use uuid::Uuid;

/// Generate a trace ID for correlation
pub fn generate_trace_id() -> String {
    Uuid::new_v4().to_string().replace("-", "")
}

/// Create a span with trace context
pub fn create_traced_span(name: &str, trace_id: Option<&str>) -> tracing::Span {
    let trace_id = trace_id.unwrap_or_else(|| {
        Box::leak(generate_trace_id().into_boxed_str())
    });

    tracing::info_span!(
        "",
        name = name,
        trace_id = trace_id,
        span_id = %generate_span_id(),
    )
}
```

**Python Implementation:**

```python
import uuid
from contextvars import ContextVar

# Thread-safe trace context
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return uuid.uuid4().hex


def set_trace_context(trace_id: str, span_id: str | None = None) -> None:
    """Set the current trace context."""
    _trace_id.set(trace_id)
    _span_id.set(span_id or uuid.uuid4().hex[:16])


def get_trace_context() -> dict[str, str | None]:
    """Get current trace context for logging."""
    return {
        "trace_id": _trace_id.get(),
        "span_id": _span_id.get(),
    }
```

### 6.2 Cross-Component Correlation

For AlphaZero training loop, correlate operations within an iteration:

```
Iteration 42:
├── trace_id: "abc123..."
├── Actor Episodes (span: "actor_episodes")
│   └── Episode 1001 (span: "episode_1001")
│   └── Episode 1002 (span: "episode_1002")
├── Trainer Steps (span: "trainer_steps")
│   └── Step 5000 (span: "step_5000")
│   └── Step 5001 (span: "step_5001")
└── Evaluation (span: "evaluation")
    └── Game 1 (span: "eval_game_1")
```

**Orchestrator Implementation:**

```python
# orchestrator/orchestrator.py
def run_iteration(self, iteration: int) -> IterationStats:
    trace_id = generate_trace_id()
    set_trace_context(trace_id)

    logger.info(
        "Starting iteration",
        event="iteration_start",
        iteration=iteration,
        trace_id=trace_id,
    )

    # Pass trace_id to actor via environment
    actor_env = {
        "CARTRIDGE_TRACE_ID": trace_id,
        "CARTRIDGE_TRACE_PARENT": f"iteration_{iteration}",
    }

    # Actor process inherits trace context
    self.actor_runner.run(env=actor_env)
```

### 6.3 Google Cloud Trace Format

For full integration with Google Cloud Trace:

```python
def format_gcp_trace(project_id: str, trace_id: str) -> str:
    """Format trace ID for Google Cloud Logging."""
    return f"projects/{project_id}/traces/{trace_id}"
```

Include in logs:
```json
{
  "logging.googleapis.com/trace": "projects/cartridge-prod/traces/abc123...",
  "logging.googleapis.com/spanId": "def456..."
}
```

---

## 7. Phase 4: Google Cloud Integration

### 7.1 Deployment Configuration

**Cloud Run / GKE Pod Configuration:**

```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: cartridge-actor
spec:
  template:
    spec:
      containers:
        - image: gcr.io/PROJECT_ID/cartridge-actor
          env:
            - name: CARTRIDGE_LOGGING_FORMAT
              value: "json"
            - name: CARTRIDGE_COMMON_LOG_LEVEL
              value: "info"
            - name: GOOGLE_CLOUD_PROJECT
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['run.googleapis.com/project-id']
```

### 7.2 Log Router Configuration

Create log sinks for different purposes:

```hcl
# terraform/logging.tf

# Sink for error logs to BigQuery for analysis
resource "google_logging_project_sink" "errors_to_bigquery" {
  name        = "cartridge-errors"
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/cartridge_logs"
  filter      = <<-EOT
    resource.type="cloud_run_revision"
    labels."component"="actor" OR labels."component"="trainer"
    severity >= WARNING
  EOT

  unique_writer_identity = true
}

# Sink for training metrics to Cloud Storage for long-term retention
resource "google_logging_project_sink" "training_to_gcs" {
  name        = "cartridge-training-logs"
  destination = "storage.googleapis.com/${google_storage_bucket.logs.name}"
  filter      = <<-EOT
    jsonPayload.event=~"^training_"
  EOT
}
```

### 7.3 Log-Based Metrics

Create metrics from logs to complement Prometheus:

```hcl
# Log-based metric for error rate
resource "google_logging_metric" "error_count" {
  name   = "cartridge/error_count"
  filter = <<-EOT
    resource.type="cloud_run_revision"
    severity >= ERROR
    jsonPayload.component=~"actor|trainer|web"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    labels {
      key         = "component"
      value_type  = "STRING"
    }
  }

  label_extractors = {
    "component" = "EXTRACT(jsonPayload.component)"
  }
}

# Distribution metric for episode duration
resource "google_logging_metric" "episode_duration" {
  name   = "cartridge/episode_duration_from_logs"
  filter = <<-EOT
    jsonPayload.event="episode_complete"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"
  }

  value_extractor = "EXTRACT(jsonPayload.duration_ms)"
  bucket_options {
    explicit_buckets {
      bounds = [10, 50, 100, 250, 500, 1000, 2500, 5000]
    }
  }
}
```

### 7.4 Alert Policies

```hcl
# Alert on high error rate
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "Cartridge High Error Rate"

  conditions {
    display_name = "Error rate > 1% for 5 minutes"
    condition_threshold {
      filter = <<-EOT
        metric.type="logging.googleapis.com/user/cartridge/error_count"
      EOT
      comparison      = "COMPARISON_GT"
      threshold_value = 10
      duration        = "300s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}
```

### 7.5 Log Explorer Queries

Common queries for debugging:

```
# All errors from actor in last hour
resource.type="cloud_run_revision"
jsonPayload.component="actor"
severity >= ERROR
timestamp >= "2025-01-12T13:00:00Z"

# Training loss trending up
jsonPayload.event="training_step"
jsonPayload.total_loss > 0.5

# Slow episodes (> 1 second)
jsonPayload.event="episode_complete"
jsonPayload.duration_ms > 1000

# Correlation: all logs for a specific trace
jsonPayload.trace_id="abc123..."

# Model reload events
jsonPayload.event="model_reload"
```

---

## 8. Log Schema Standards

### 8.1 Common Fields (All Components)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `severity` | string | Google Cloud severity | `"INFO"` |
| `timestamp` | string | RFC 3339 UTC | `"2025-01-12T14:30:45.123Z"` |
| `message` | string | Human-readable message | `"Episode completed"` |
| `event` | string | Machine-readable event type | `"episode_complete"` |
| `component` | string | Component name | `"actor"`, `"trainer"`, `"web"` |
| `env_id` | string | Game environment | `"connect4"` |
| `trace_id` | string | Correlation ID (optional) | `"abc123..."` |
| `span_id` | string | Operation ID (optional) | `"def456..."` |

### 8.2 Actor-Specific Fields

| Event | Fields |
|-------|--------|
| `actor_start` | `actor_id`, `max_episodes`, `mcts_simulations` |
| `episode_start` | `episode`, `model_version` |
| `episode_complete` | `episode`, `duration_ms`, `steps`, `outcome`, `searches` |
| `mcts_stats` | `episode`, `searches`, `inference_ms`, `expansion_ms` |
| `model_reload` | `model_path`, `load_time_ms`, `model_version` |
| `storage_write` | `transitions`, `write_time_ms`, `buffer_size` |

### 8.3 Trainer-Specific Fields

| Event | Fields |
|-------|--------|
| `training_start` | `total_steps`, `batch_size`, `learning_rate`, `device` |
| `training_step` | `step`, `total_loss`, `value_loss`, `policy_loss`, `lr`, `samples_per_sec` |
| `training_complete` | `total_steps`, `final_loss`, `duration_seconds` |
| `checkpoint_saved` | `path`, `step`, `size_bytes`, `save_time_ms` |
| `evaluation_start` | `games`, `model_path` |
| `evaluation_complete` | `win_rate`, `draw_rate`, `games_played`, `duration_seconds` |

### 8.4 Web-Specific Fields

| Event | Fields |
|-------|--------|
| `request_start` | `endpoint`, `method`, `game_id` |
| `request_complete` | `endpoint`, `method`, `status`, `duration_ms` |
| `game_created` | `game_id`, `env_id`, `player_first` |
| `move_made` | `game_id`, `player`, `position`, `game_over` |
| `bot_move` | `game_id`, `position`, `think_time_ms`, `mcts_simulations` |

### 8.5 Event Naming Convention

- Use `snake_case` for event names
- Format: `{noun}_{verb}` (e.g., `episode_complete`, `model_reload`)
- Keep names concise but descriptive
- Avoid abbreviations except well-known ones (`ms`, `id`)

---

## 9. Testing & Validation

### 9.1 Local Testing

```bash
# Test JSON output locally
export CARTRIDGE_LOGGING_FORMAT=json
cargo run --manifest-path actor/Cargo.toml 2>&1 | jq .

# Validate JSON structure
cargo run --manifest-path actor/Cargo.toml 2>&1 | \
  jq -e '.severity and .timestamp and .message' > /dev/null && \
  echo "Valid JSON logs"

# Test Python JSON output
cd trainer
python -c "
from trainer.structured_logging import configure_logging, get_logger
configure_logging(json_format=True, component='trainer', env_id='test')
logger = get_logger('test')
logger.info('Test message', event='test_event', value=42)
"
```

### 9.2 Log Validation Script

**File:** `scripts/validate_logs.py`

```python
#!/usr/bin/env python3
"""Validate structured log output against schema."""

import json
import sys
from typing import Any

REQUIRED_FIELDS = {"severity", "timestamp", "message"}
VALID_SEVERITIES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_COMPONENTS = {"actor", "trainer", "web"}


def validate_log_line(line: str) -> list[str]:
    """Validate a single log line. Returns list of errors."""
    errors = []

    try:
        log = json.loads(line)
    except json.JSONDecodeError:
        return [f"Invalid JSON: {line[:50]}..."]

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in log:
            errors.append(f"Missing required field: {field}")

    # Validate severity
    if "severity" in log and log["severity"] not in VALID_SEVERITIES:
        errors.append(f"Invalid severity: {log['severity']}")

    # Validate component if present
    labels = log.get("logging.googleapis.com/labels", {})
    if "component" in labels and labels["component"] not in VALID_COMPONENTS:
        errors.append(f"Invalid component: {labels['component']}")

    return errors


def main():
    total_lines = 0
    error_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        total_lines += 1
        errors = validate_log_line(line)

        if errors:
            error_count += 1
            print(f"Line {total_lines}: {errors}", file=sys.stderr)

    print(f"Validated {total_lines} lines, {error_count} errors")
    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
```

### 9.3 Integration Test

```bash
# Run actor with JSON logging and validate output
export CARTRIDGE_LOGGING_FORMAT=json
export CARTRIDGE_ACTOR_MAX_EPISODES=10

cargo run --manifest-path actor/Cargo.toml 2>&1 | \
  python scripts/validate_logs.py
```

### 9.4 Google Cloud Validation

After deployment, verify logs appear correctly:

```bash
# Check logs in Cloud Logging
gcloud logging read \
  'jsonPayload.component="actor"' \
  --limit=10 \
  --format="json(jsonPayload)"

# Verify structured fields are indexed
gcloud logging read \
  'jsonPayload.event="episode_complete" AND jsonPayload.duration_ms>100' \
  --limit=5
```

---

## 10. Migration Checklist

### 10.1 Pre-Implementation

- [ ] Review current log output samples from each component
- [ ] Identify high-volume log statements for potential filtering
- [ ] Estimate log volume for cost planning
- [ ] Set up development environment for testing

### 10.2 Phase 1: Rust Components

- [ ] Add `json` feature to `tracing-subscriber` dependency
- [ ] Implement `init_tracing()` with format switching
- [ ] Add component metadata spans (actor_id, env_id)
- [ ] Update existing log statements with `event` field
- [ ] Test JSON output locally
- [ ] Verify existing functionality unchanged

### 10.3 Phase 2: Python Components

- [ ] Add `python-json-logger` dependency
- [ ] Create `structured_logging.py` module
- [ ] Update `__main__.py` logging initialization
- [ ] Refactor `trainer.py` log statements
- [ ] Refactor `orchestrator.py` log statements
- [ ] Refactor `evaluator.py` log statements
- [ ] Test JSON output locally
- [ ] Run full training loop and validate logs

### 10.4 Phase 3: Correlation

- [ ] Implement trace ID generation
- [ ] Add trace context to orchestrator
- [ ] Pass trace ID to actor subprocess
- [ ] Test correlation across components
- [ ] Verify trace IDs appear in logs

### 10.5 Phase 4: Google Cloud

- [ ] Deploy to staging environment
- [ ] Verify logs appear in Cloud Logging
- [ ] Test log queries work as expected
- [ ] Create log-based metrics
- [ ] Set up alert policies
- [ ] Configure log retention/archival
- [ ] Document operational procedures

### 10.6 Post-Implementation

- [ ] Update CLAUDE.md with logging configuration
- [ ] Create runbook for log analysis
- [ ] Train team on new log structure
- [ ] Monitor log volume and costs
- [ ] Iterate on log content based on operational experience

---

## Appendix A: File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `actor/Cargo.toml` | Modify | Add json feature to tracing-subscriber |
| `actor/src/main.rs` | Modify | Update init_tracing with format switching |
| `actor/src/actor.rs` | Modify | Add event fields to log statements |
| `actor/src/logging.rs` | Create | GCP-compatible JSON formatter |
| `web/Cargo.toml` | Modify | Add json feature to tracing-subscriber |
| `web/src/main.rs` | Modify | Update tracing initialization |
| `web/src/handlers/*.rs` | Modify | Add structured fields to logs |
| `trainer/pyproject.toml` | Modify | Add python-json-logger dependency |
| `trainer/src/trainer/structured_logging.py` | Create | Structured logging module |
| `trainer/src/trainer/__main__.py` | Modify | Use new logging configuration |
| `trainer/src/trainer/trainer.py` | Modify | Refactor to structured logs |
| `trainer/src/trainer/orchestrator/*.py` | Modify | Add trace context, structured logs |
| `config.toml` | Modify | Add [logging] section |
| `engine/engine-config/src/structs.rs` | Modify | Add LoggingConfig struct |
| `scripts/validate_logs.py` | Create | Log validation script |

---

## Appendix B: Example Log Output

### Actor (JSON Format)

```json
{"timestamp":"2025-01-12T14:30:45.123456Z","level":"INFO","target":"actor","message":"Actor service starting","component":"actor","actor_id":"actor-1","env_id":"connect4","version":"0.1.0"}
{"timestamp":"2025-01-12T14:30:45.234567Z","level":"INFO","target":"actor::actor","message":"Model loaded","event":"model_load","model_path":"./data/models/latest.onnx","load_time_ms":123.4,"component":"actor"}
{"timestamp":"2025-01-12T14:30:46.345678Z","level":"INFO","target":"actor::actor","message":"Episode completed","event":"episode_complete","episode":1,"duration_ms":45.2,"steps":9,"outcome":"player1_win","searches":800,"component":"actor"}
```

### Trainer (JSON Format)

```json
{"severity":"INFO","timestamp":"2025-01-12T14:31:00.000000Z","message":"Starting training","event":"training_start","total_steps":1000,"batch_size":64,"learning_rate":0.001,"device":"cuda","logging.googleapis.com/labels":{"component":"trainer","env_id":"connect4"}}
{"severity":"INFO","timestamp":"2025-01-12T14:31:01.123456Z","message":"Training step","event":"training_step","step":100,"total_loss":0.8234,"value_loss":0.4123,"policy_loss":0.4111,"lr":0.001,"samples_per_sec":1234.5,"logging.googleapis.com/labels":{"component":"trainer","env_id":"connect4"}}
```

### Web (JSON Format)

```json
{"timestamp":"2025-01-12T14:32:00.000000Z","level":"INFO","target":"web::handlers::game","message":"Move processed","event":"move_made","game_id":"abc123","player":"human","position":4,"game_over":false,"component":"web"}
{"timestamp":"2025-01-12T14:32:00.123456Z","level":"INFO","target":"web::handlers::game","message":"Bot move computed","event":"bot_move","game_id":"abc123","position":0,"think_time_ms":52.3,"mcts_simulations":800,"component":"web"}
```

---

## Appendix C: Prometheus Metrics vs. Structured Logs

| Use Case | Prometheus Metrics | Structured Logs |
|----------|-------------------|-----------------|
| Real-time dashboards | Yes | No |
| Alerting on thresholds | Yes | Via log-based metrics |
| Long-term trend analysis | Yes (with retention) | Yes (BigQuery) |
| Debugging specific issues | No | Yes |
| Correlation across services | No | Yes (trace ID) |
| Cost at scale | Low | Higher (volume-based) |
| Historical investigation | Limited | Full context |

**Recommendation:** Use both. Metrics for monitoring, logs for debugging.
