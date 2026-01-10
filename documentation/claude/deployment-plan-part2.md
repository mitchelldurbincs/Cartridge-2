# Deployment Plan Part 2: High Priority Items

This document outlines the implementation plan for **HIGH** priority items that should be addressed before production traffic.

## Overview

| # | Issue | Severity | Estimated Effort |
|---|-------|----------|------------------|
| 1 | Use Managed Database | HIGH | 3-4 hours |
| 2 | Structured Logging | HIGH | 2-3 hours |
| 3 | Connection Pool Configuration | HIGH | 1-2 hours |
| 4 | Production unwrap()/expect() Calls | HIGH | 2-3 hours |
| 5 | Docker Security Hardening | HIGH | 1-2 hours |

**Total Estimated Effort:** 9-14 hours

---

## 1. Migrate to Managed Database (Cloud SQL)

### Current State

- PostgreSQL runs as Kubernetes StatefulSet (`k8s/base/postgres/statefulset.yaml`)
- Single 10GB PVC with no automated backups
- No point-in-time recovery
- No high availability

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Cloud SQL                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  PostgreSQL 15 (db-custom-2-4096)                   │    │
│  │  - Automatic backups (daily, 7-day retention)       │    │
│  │  - Point-in-time recovery enabled                   │    │
│  │  - Private IP (VPC peering)                         │    │
│  │  - Encrypted at rest (Google-managed keys)          │    │
│  │  - High availability (regional)                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Private IP (10.x.x.x)
                              │ SSL/TLS required
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      GKE Cluster                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Actor 1  │  │ Actor 2  │  │ Trainer  │  │ Web      │    │
│  │ (pool:8) │  │ (pool:8) │  │ (pool:4) │  │ (pool:4) │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│       Total connections: 8+8+4+4 = 24 (within limits)       │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Steps

#### Step 1.1: Create Cloud SQL Instance

```bash
# Enable Cloud SQL API
gcloud services enable sqladmin.googleapis.com

# Create PostgreSQL instance
gcloud sql instances create cartridge-db \
  --database-version=POSTGRES_15 \
  --tier=db-custom-2-4096 \
  --region=us-central1 \
  --availability-type=REGIONAL \
  --storage-type=SSD \
  --storage-size=20GB \
  --storage-auto-increase \
  --backup-start-time=03:00 \
  --enable-point-in-time-recovery \
  --retained-backups-count=7 \
  --network=projects/PROJECT_ID/global/networks/default \
  --no-assign-ip \
  --database-flags=max_connections=100

# Create database
gcloud sql databases create cartridge --instance=cartridge-db

# Create user (password should come from Secret Manager)
gcloud sql users create cartridge \
  --instance=cartridge-db \
  --password="$(gcloud secrets versions access latest --secret=cartridge-db-password)"
```

#### Step 1.2: Configure Private IP Access

```bash
# Reserve IP range for VPC peering
gcloud compute addresses create google-managed-services-default \
  --global \
  --purpose=VPC_PEERING \
  --prefix-length=16 \
  --network=default

# Create private connection
gcloud services vpc-peerings connect \
  --service=servicenetworking.googleapis.com \
  --ranges=google-managed-services-default \
  --network=default

# Get the private IP
gcloud sql instances describe cartridge-db --format='value(ipAddresses[0].ipAddress)'
```

#### Step 1.3: Create Terraform Configuration

Create `terraform/modules/cloudsql/main.tf`:

```hcl
# terraform/modules/cloudsql/main.tf

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "db_password_secret_id" {
  description = "Secret Manager secret ID for database password"
  type        = string
  default     = "cartridge-db-password"
}

# Get password from Secret Manager
data "google_secret_manager_secret_version" "db_password" {
  secret = var.db_password_secret_id
}

# Cloud SQL Instance
resource "google_sql_database_instance" "cartridge" {
  name             = "cartridge-db"
  database_version = "POSTGRES_15"
  region           = var.region
  project          = var.project_id

  settings {
    tier              = "db-custom-2-4096"
    availability_type = "REGIONAL"
    disk_type         = "PD_SSD"
    disk_size         = 20
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 7
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = "projects/${var.project_id}/global/networks/default"
      require_ssl     = true
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }

    maintenance_window {
      day          = 7  # Sunday
      hour         = 3  # 3 AM
      update_track = "stable"
    }

    insights_config {
      query_insights_enabled  = true
      record_client_address   = true
      record_application_tags = true
    }
  }

  deletion_protection = true
}

# Database
resource "google_sql_database" "cartridge" {
  name     = "cartridge"
  instance = google_sql_database_instance.cartridge.name
}

# User
resource "google_sql_user" "cartridge" {
  name     = "cartridge"
  instance = google_sql_database_instance.cartridge.name
  password = data.google_secret_manager_secret_version.db_password.secret_data
}

# Output connection details
output "instance_connection_name" {
  value = google_sql_database_instance.cartridge.connection_name
}

output "private_ip" {
  value = google_sql_database_instance.cartridge.private_ip_address
}
```

#### Step 1.4: Update Kubernetes ConfigMap

Update `k8s/base/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cartridge-config
  namespace: cartridge
data:
  # Database configuration (password from ExternalSecret)
  CARTRIDGE_STORAGE_POSTGRES_HOST: "10.x.x.x"  # Cloud SQL private IP
  CARTRIDGE_STORAGE_POSTGRES_PORT: "5432"
  CARTRIDGE_STORAGE_POSTGRES_USER: "cartridge"
  CARTRIDGE_STORAGE_POSTGRES_DB: "cartridge"
  CARTRIDGE_STORAGE_POSTGRES_SSLMODE: "require"

  # Connection pool settings (see Section 3)
  CARTRIDGE_STORAGE_POOL_MAX_SIZE: "8"
  CARTRIDGE_STORAGE_POOL_CONNECT_TIMEOUT: "30"
  CARTRIDGE_STORAGE_POOL_IDLE_TIMEOUT: "300"
```

#### Step 1.5: Add SSL Certificate for Cloud SQL

```bash
# Download server CA certificate
gcloud sql instances describe cartridge-db \
  --format='value(serverCaCert.cert)' > server-ca.pem

# Create Kubernetes secret with CA cert
kubectl create secret generic cloudsql-ca-cert \
  --from-file=server-ca.pem=server-ca.pem \
  -n cartridge
```

Update deployments to mount the certificate:

```yaml
# In deployment spec:
volumes:
  - name: cloudsql-certs
    secret:
      secretName: cloudsql-ca-cert
containers:
  - name: actor
    volumeMounts:
      - name: cloudsql-certs
        mountPath: /etc/ssl/cloudsql
        readOnly: true
    env:
      - name: CARTRIDGE_STORAGE_POSTGRES_SSLROOTCERT
        value: "/etc/ssl/cloudsql/server-ca.pem"
```

#### Step 1.6: Update Rust PostgreSQL Connection

Update `actor/src/storage/postgres.rs`:

```rust
use deadpool_postgres::{Config, Pool, Runtime, SslMode};
use tokio_postgres_rustls::MakeRustlsConnect;
use rustls::{ClientConfig, RootCertStore};
use std::fs;

pub async fn create_pool(config: &PostgresConfig) -> Result<Pool, PoolError> {
    let mut pool_config = Config::new();
    pool_config.host = Some(config.host.clone());
    pool_config.port = Some(config.port);
    pool_config.user = Some(config.user.clone());
    pool_config.password = Some(config.password.clone());
    pool_config.dbname = Some(config.database.clone());

    // Configure SSL if certificate path is provided
    if let Some(ca_cert_path) = &config.ssl_root_cert {
        let ca_cert = fs::read(ca_cert_path)
            .map_err(|e| PoolError::Config(format!("Failed to read CA cert: {}", e)))?;

        let mut root_store = RootCertStore::empty();
        let certs = rustls_pemfile::certs(&mut ca_cert.as_slice())
            .map_err(|e| PoolError::Config(format!("Failed to parse CA cert: {}", e)))?;

        for cert in certs {
            root_store.add(&rustls::Certificate(cert))
                .map_err(|e| PoolError::Config(format!("Failed to add CA cert: {}", e)))?;
        }

        let tls_config = ClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let tls = MakeRustlsConnect::new(tls_config);

        pool_config.ssl_mode = Some(SslMode::Require);
        pool_config.create_pool(Some(Runtime::Tokio1), tls)
    } else {
        // No SSL (local development)
        pool_config.create_pool(Some(Runtime::Tokio1), tokio_postgres::NoTls)
    }
}
```

#### Step 1.7: Update Python Trainer Connection

Update `trainer/src/trainer/storage/postgres.py`:

```python
import ssl
import os
from typing import Optional
import psycopg2
from psycopg2 import pool

class PostgresStorage:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        ssl_ca_cert: Optional[str] = None,
        pool_min: int = 1,
        pool_max: int = 4,
    ):
        # Build connection kwargs
        conn_kwargs = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }

        # Configure SSL if CA cert provided
        if ssl_ca_cert and os.path.exists(ssl_ca_cert):
            conn_kwargs["sslmode"] = "verify-ca"
            conn_kwargs["sslrootcert"] = ssl_ca_cert
        elif os.environ.get("CARTRIDGE_STORAGE_POSTGRES_SSLMODE") == "require":
            conn_kwargs["sslmode"] = "require"

        self._pool = pool.ThreadedConnectionPool(
            minconn=pool_min,
            maxconn=pool_max,
            **conn_kwargs
        )

    def get_connection(self):
        return self._pool.getconn()

    def release_connection(self, conn):
        self._pool.putconn(conn)

    def close(self):
        self._pool.closeall()
```

#### Step 1.8: Migration Script

Create `scripts/migrate-to-cloudsql.sh`:

```bash
#!/bin/bash
set -euo pipefail

# Configuration
SOURCE_POD="postgres-0"
SOURCE_NAMESPACE="cartridge"
CLOUDSQL_INSTANCE="cartridge-db"
DATABASE="cartridge"

echo "=== Migrating data to Cloud SQL ==="

# 1. Create backup from current StatefulSet
echo "Creating backup from Kubernetes PostgreSQL..."
kubectl exec -n $SOURCE_NAMESPACE $SOURCE_POD -- \
  pg_dump -U cartridge -d $DATABASE -F c -f /tmp/backup.dump

# 2. Copy backup locally
echo "Copying backup locally..."
kubectl cp $SOURCE_NAMESPACE/$SOURCE_POD:/tmp/backup.dump ./backup.dump

# 3. Get Cloud SQL connection info
CLOUDSQL_IP=$(gcloud sql instances describe $CLOUDSQL_INSTANCE \
  --format='value(ipAddresses[0].ipAddress)')

# 4. Restore to Cloud SQL
echo "Restoring to Cloud SQL..."
PGPASSWORD=$(gcloud secrets versions access latest --secret=cartridge-db-password) \
  pg_restore -h $CLOUDSQL_IP -U cartridge -d $DATABASE \
  --no-owner --no-privileges ./backup.dump

echo "=== Migration complete ==="
echo "Verify data integrity before switching traffic."
```

#### Step 1.9: Validation Checklist

- [ ] Cloud SQL instance created with proper settings
- [ ] Private IP connectivity from GKE verified
- [ ] SSL/TLS connection working
- [ ] Data migrated from StatefulSet
- [ ] Automated backups running
- [ ] Connection pool sizes appropriate
- [ ] Query Insights enabled and showing data
- [ ] Old StatefulSet scaled to 0 (keep for rollback)

---

## 2. Structured Logging Implementation

### Current State

- Rust services use `tracing_subscriber` with text format
- Python trainer uses basic `logging` module
- No correlation between services
- Cloud Logging receives unstructured text

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Google Cloud Logging                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Structured JSON logs with:                         │    │
│  │  - severity (INFO, WARN, ERROR)                     │    │
│  │  - timestamp (RFC3339)                              │    │
│  │  - trace_id (for correlation)                       │    │
│  │  - span_id (for nested operations)                  │    │
│  │  - service_name (actor, trainer, web)               │    │
│  │  - custom fields (episode_id, game_id, etc.)        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │ JSON               │ JSON               │ JSON
┌────────┴────────┐  ┌───────┴───────┐  ┌────────┴────────┐
│  Actor          │  │ Trainer       │  │  Web Server     │
│  (tracing-json) │  │ (structlog)   │  │  (tracing-json) │
└─────────────────┘  └───────────────┘  └─────────────────┘
```

### Implementation Steps

#### Step 2.1: Update Rust Dependencies

Update `web/Cargo.toml` and `actor/Cargo.toml`:

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
tracing-opentelemetry = "0.22"  # For future Cloud Trace integration
uuid = { version = "1.0", features = ["v4"] }
```

#### Step 2.2: Create Logging Configuration Module

Create `actor/src/logging.rs` (and similar for `web/src/logging.rs`):

```rust
//! Structured logging configuration for cloud deployments.

use tracing_subscriber::{
    fmt::{self, format::JsonFields},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

/// Log output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable text (for local development)
    Text,
    /// JSON (for cloud deployments)
    Json,
}

impl LogFormat {
    pub fn from_env() -> Self {
        match std::env::var("LOG_FORMAT").as_deref() {
            Ok("json") => LogFormat::Json,
            Ok("text") => LogFormat::Text,
            _ => {
                // Auto-detect: use JSON if running in Kubernetes
                if std::env::var("KUBERNETES_SERVICE_HOST").is_ok() {
                    LogFormat::Json
                } else {
                    LogFormat::Text
                }
            }
        }
    }
}

/// Initialize the logging system.
///
/// Call this early in main() before any logging occurs.
pub fn init_logging(service_name: &str) {
    let format = LogFormat::from_env();
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    match format {
        LogFormat::Text => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().with_target(true))
                .init();
        }
        LogFormat::Json => {
            // GCP Cloud Logging compatible JSON format
            tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .json()
                        .with_current_span(true)
                        .with_span_list(true)
                        .with_file(true)
                        .with_line_number(true)
                        .with_thread_ids(true)
                        .flatten_event(true)
                        .with_target(true)
                        // GCP expects "severity" not "level"
                        .with_level(true)
                )
                .init();
        }
    }

    tracing::info!(
        service = service_name,
        format = ?format,
        "Logging initialized"
    );
}

/// Create a span for an episode with relevant context.
#[macro_export]
macro_rules! episode_span {
    ($episode_id:expr, $game:expr) => {
        tracing::info_span!(
            "episode",
            episode_id = $episode_id,
            game = $game,
        )
    };
}

/// Create a span for a training step.
#[macro_export]
macro_rules! training_span {
    ($iteration:expr, $step:expr) => {
        tracing::info_span!(
            "training",
            iteration = $iteration,
            step = $step,
        )
    };
}
```

#### Step 2.3: Update Actor Main to Use Structured Logging

Update `actor/src/main.rs`:

```rust
mod logging;

use logging::init_logging;
use tracing::{info, warn, error, instrument, Span};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging first
    init_logging("actor");

    info!(
        version = env!("CARGO_PKG_VERSION"),
        "Actor starting"
    );

    // ... rest of initialization ...

    // Example: structured log with context
    info!(
        env_id = %config.env_id,
        max_episodes = config.max_episodes,
        "Configuration loaded"
    );
}

#[instrument(skip(ctx, storage, evaluator), fields(episode_id = %episode_id))]
async fn run_episode(
    episode_id: u64,
    ctx: &mut EngineContext,
    storage: &dyn Storage,
    evaluator: &dyn Evaluator,
) -> Result<EpisodeResult, EpisodeError> {
    info!("Starting episode");

    // ... episode logic ...

    match result {
        Ok(episode) => {
            info!(
                moves = episode.moves.len(),
                outcome = ?episode.outcome,
                "Episode completed"
            );
            Ok(episode)
        }
        Err(e) => {
            error!(
                error = %e,
                "Episode failed"
            );
            Err(e)
        }
    }
}
```

#### Step 2.4: Update Web Server Logging

Update `web/src/main.rs`:

```rust
mod logging;

use axum::extract::Request;
use tower_http::trace::{TraceLayer, MakeSpan, OnRequest, OnResponse};
use tracing::{info_span, Span, Level};

// Custom span maker for HTTP requests
#[derive(Clone)]
struct CustomMakeSpan;

impl<B> MakeSpan<B> for CustomMakeSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // Extract trace ID from header or generate new one
        let trace_id = request
            .headers()
            .get("x-cloud-trace-context")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split('/').next().unwrap_or("unknown"))
            .unwrap_or("none")
            .to_string();

        info_span!(
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            trace_id = %trace_id,
        )
    }
}

#[derive(Clone)]
struct CustomOnResponse;

impl<B> OnResponse<B> for CustomOnResponse {
    fn on_response(self, response: &axum::http::Response<B>, latency: std::time::Duration, _span: &Span) {
        tracing::info!(
            status = response.status().as_u16(),
            latency_ms = latency.as_millis(),
            "Request completed"
        );
    }
}

fn create_trace_layer() -> TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    CustomMakeSpan,
    (),
    CustomOnResponse,
> {
    TraceLayer::new_for_http()
        .make_span_with(CustomMakeSpan)
        .on_request(())
        .on_response(CustomOnResponse)
}

// In router setup:
let app = Router::new()
    // ... routes ...
    .layer(create_trace_layer());
```

#### Step 2.5: Update Python Trainer Logging

Update `trainer/src/trainer/logging_config.py`:

```python
"""Structured logging configuration for cloud deployments."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class CloudLoggingFormatter(logging.Formatter):
    """JSON formatter compatible with Google Cloud Logging."""

    LEVEL_TO_SEVERITY = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": self.LEVEL_TO_SEVERITY.get(record.levelno, "DEFAULT"),
            "message": record.getMessage(),
            "service": self.service_name,
            "logger": record.name,
            "source_location": {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            },
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            ):
                log_entry[key] = value

        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(
    service_name: str = "trainer",
    level: str = "INFO",
    format_override: Optional[str] = None,
) -> None:
    """
    Initialize structured logging.

    Args:
        service_name: Name of the service (appears in logs)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_override: Force "json" or "text" format
    """
    # Determine format
    log_format = format_override or os.environ.get("LOG_FORMAT")
    if log_format is None:
        # Auto-detect: use JSON if running in Kubernetes
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            log_format = "json"
        else:
            log_format = "text"

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        handler.setFormatter(CloudLoggingFormatter(service_name))
    else:
        handler.setFormatter(TextFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers = [handler]

    # Log initialization
    logging.info(
        "Logging initialized",
        extra={
            "format": log_format,
            "level": level,
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


# Context manager for adding fields to all logs in a block
class LogContext:
    """Add context fields to all logs within a block."""

    def __init__(self, **fields):
        self.fields = fields
        self.old_factory = None

    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.fields.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args):
        logging.setLogRecordFactory(self.old_factory)
```

#### Step 2.6: Update Trainer to Use Structured Logging

Update `trainer/src/trainer/__main__.py`:

```python
from trainer.logging_config import setup_logging, get_logger, LogContext

def main():
    # Initialize logging first
    setup_logging(service_name="trainer", level="INFO")
    logger = get_logger(__name__)

    logger.info("Trainer starting", extra={"version": "1.0.0"})

    # ... argument parsing ...

    # Example: add context to training logs
    with LogContext(iteration=iteration, env_id=args.env_id):
        logger.info("Starting training iteration")
        # All logs within this block will have iteration and env_id fields

        for step in range(steps):
            with LogContext(step=step):
                loss = train_step(...)
                logger.info(
                    "Training step complete",
                    extra={
                        "loss": float(loss),
                        "learning_rate": float(lr),
                    }
                )
```

#### Step 2.7: Configure GKE to Forward Logs

Logs are automatically forwarded to Cloud Logging when running on GKE. To enhance this:

```yaml
# k8s/base/actor/deployment.yaml
spec:
  template:
    spec:
      containers:
        - name: actor
          env:
            - name: LOG_FORMAT
              value: "json"
            - name: RUST_LOG
              value: "info,actor=debug"
```

#### Step 2.8: Create Cloud Logging Queries

Save useful queries in `documentation/logging-queries.md`:

```markdown
# Useful Cloud Logging Queries

## All errors across services
```
severity>=ERROR
resource.type="k8s_container"
resource.labels.namespace_name="cartridge"
```

## Actor episode completions
```
jsonPayload.message="Episode completed"
resource.labels.container_name="actor"
```

## Training steps with loss
```
jsonPayload.message="Training step complete"
| jsonPayload.loss < 0.1
```

## Slow HTTP requests (>1s)
```
jsonPayload.message="Request completed"
jsonPayload.latency_ms > 1000
```

## Trace a specific request
```
jsonPayload.trace_id="TRACE_ID_HERE"
```
```

#### Step 2.9: Validation Checklist

- [ ] Rust services output JSON logs when `LOG_FORMAT=json`
- [ ] Python trainer outputs JSON logs when `LOG_FORMAT=json`
- [ ] Logs appear in Cloud Logging console
- [ ] Log severity is correctly mapped
- [ ] Custom fields (episode_id, loss, etc.) are searchable
- [ ] HTTP request tracing includes latency
- [ ] Local development uses readable text format

---

## 3. Connection Pool Configuration

### Current State

- `config.defaults.toml:63-65` - Fixed pool of 16 connections per actor
- Multiple actors can exhaust PostgreSQL connection limit
- No dynamic sizing based on workload

### Analysis

```
PostgreSQL max_connections = 100 (default)
Reserved for admin/monitoring = 10
Available for application = 90

If we have:
- 4 actor pods × 16 connections = 64
- 1 trainer pod × 16 connections = 16
- 1 web pod × 16 connections = 16
Total = 96 connections > 90 available!
```

### Implementation Steps

#### Step 3.1: Calculate Optimal Pool Sizes

Create sizing guidelines:

| Component | Connections | Rationale |
|-----------|-------------|-----------|
| Actor (each) | 8 | Episodes are sequential per actor |
| Trainer | 8 | Batch reads, periodic writes |
| Web | 4 | Stateless API calls |
| Reserved | 10 | Admin, monitoring, migrations |

With 4 actors: `(4 × 8) + 8 + 4 + 10 = 54` connections

#### Step 3.2: Update Configuration Defaults

Update `config.defaults.toml`:

```toml
[storage]
# Connection pool settings
# Size pools based on component role and total expected pods
# Formula: sum of all (pods × pool_size) must be < max_connections - reserved
pool_max_size = 8         # Per-component pool size (was 16)
pool_connect_timeout = 30  # Seconds to wait for connection
pool_idle_timeout = 300    # Seconds before idle connections are closed
pool_min_size = 1          # Minimum connections to maintain
```

#### Step 3.3: Add Pool Size Validation

Update `actor/src/config.rs`:

```rust
/// Validate pool configuration against expected deployment size.
pub fn validate_pool_config(config: &StorageConfig) -> Result<(), ConfigError> {
    // Check that pool size is reasonable
    if config.pool_max_size > 20 {
        tracing::warn!(
            pool_max_size = config.pool_max_size,
            "Large connection pool size. Ensure this won't exhaust database connections."
        );
    }

    if config.pool_max_size < config.pool_min_size {
        return Err(ConfigError::InvalidConfig(
            "pool_max_size must be >= pool_min_size".to_string()
        ));
    }

    Ok(())
}
```

#### Step 3.4: Add Pool Metrics

Update `actor/src/storage/postgres.rs`:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

pub struct PoolMetrics {
    pub connections_acquired: AtomicU64,
    pub connections_released: AtomicU64,
    pub connection_timeouts: AtomicU64,
    pub connection_errors: AtomicU64,
}

impl PoolMetrics {
    pub fn new() -> Self {
        Self {
            connections_acquired: AtomicU64::new(0),
            connections_released: AtomicU64::new(0),
            connection_timeouts: AtomicU64::new(0),
            connection_errors: AtomicU64::new(0),
        }
    }

    pub fn record_acquire(&self) {
        self.connections_acquired.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_release(&self) {
        self.connections_released.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_timeout(&self) {
        self.connection_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn log_stats(&self) {
        tracing::info!(
            acquired = self.connections_acquired.load(Ordering::Relaxed),
            released = self.connections_released.load(Ordering::Relaxed),
            timeouts = self.connection_timeouts.load(Ordering::Relaxed),
            errors = self.connection_errors.load(Ordering::Relaxed),
            "Connection pool stats"
        );
    }
}
```

#### Step 3.5: Environment-Specific Pool Sizing

Update Kubernetes deployments with appropriate sizes:

```yaml
# k8s/base/actor/deployment.yaml
env:
  - name: CARTRIDGE_STORAGE_POOL_MAX_SIZE
    value: "8"  # 4 actors × 8 = 32 connections

# k8s/base/trainer/deployment.yaml
env:
  - name: CARTRIDGE_STORAGE_POOL_MAX_SIZE
    value: "8"  # 1 trainer × 8 = 8 connections

# k8s/base/web/deployment.yaml
env:
  - name: CARTRIDGE_STORAGE_POOL_MAX_SIZE
    value: "4"  # 1 web × 4 = 4 connections
```

#### Step 3.6: Add PgBouncer for Large Deployments

For scaling beyond 10+ actors, add PgBouncer:

Create `k8s/base/pgbouncer/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: cartridge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
        - name: pgbouncer
          image: bitnami/pgbouncer:1.21.0
          ports:
            - containerPort: 6432
          env:
            - name: POSTGRESQL_HOST
              valueFrom:
                configMapKeyRef:
                  name: cartridge-config
                  key: CARTRIDGE_STORAGE_POSTGRES_HOST
            - name: POSTGRESQL_PORT
              value: "5432"
            - name: POSTGRESQL_USERNAME
              value: "cartridge"
            - name: POSTGRESQL_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: cartridge-secrets
                  key: POSTGRES_PASSWORD
            - name: POSTGRESQL_DATABASE
              value: "cartridge"
            - name: PGBOUNCER_POOL_MODE
              value: "transaction"
            - name: PGBOUNCER_MAX_CLIENT_CONN
              value: "200"
            - name: PGBOUNCER_DEFAULT_POOL_SIZE
              value: "20"
          resources:
            requests:
              memory: "64Mi"
              cpu: "100m"
            limits:
              memory: "128Mi"
              cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: pgbouncer
  namespace: cartridge
spec:
  selector:
    app: pgbouncer
  ports:
    - port: 5432
      targetPort: 6432
```

#### Step 3.7: Validation Checklist

- [ ] Pool sizes documented and calculated
- [ ] Sum of all pools < PostgreSQL max_connections
- [ ] Pool metrics being logged
- [ ] Connection timeouts are rare (< 0.1%)
- [ ] PgBouncer deployed if > 10 actors
- [ ] Load tested with expected number of pods

---

## 4. Remove Production unwrap()/expect() Calls

### Current State

Multiple `unwrap()` and `expect()` calls that will panic in production:

| File | Line | Code |
|------|------|------|
| `web/src/main.rs` | 123 | `.expect("Failed to install Ctrl+C handler")` |
| `web/src/main.rs` | 133-134 | `.unwrap()` on parsing |
| `web/src/main.rs` | 170 | `.expect("At least tictactoe should be registered")` |
| `actor/src/actor.rs` | 302 | `.unwrap()` in production path |

### Implementation Steps

#### Step 4.1: Create Custom Error Types

Create `web/src/error.rs`:

```rust
//! Application error types.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Game error: {0}")]
    Game(String),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::Config(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::Game(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::Storage(e) => (StatusCode::SERVICE_UNAVAILABLE, e.to_string()),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };

        tracing::error!(error = %self, "Request failed");

        (status, Json(json!({ "error": message }))).into_response()
    }
}

// Helper trait for converting Results
pub trait ResultExt<T> {
    fn or_internal(self, context: &str) -> Result<T, AppError>;
}

impl<T, E: std::fmt::Display> ResultExt<T> for Result<T, E> {
    fn or_internal(self, context: &str) -> Result<T, AppError> {
        self.map_err(|e| AppError::Internal(format!("{}: {}", context, e)))
    }
}
```

#### Step 4.2: Update Web Server Main

Update `web/src/main.rs`:

```rust
mod error;

use error::{AppError, ResultExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    init_logging("web");

    // Load configuration with proper error handling
    let config = load_config().map_err(|e| {
        tracing::error!(error = %e, "Failed to load configuration");
        e
    })?;

    // Register games - log error but continue if some fail
    register_games();

    // Set up signal handler with fallback
    let shutdown = setup_shutdown_signal();

    // Parse log filter with fallback
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Create engine context with error handling
    let ctx = EngineContext::new(&config.common.env_id).map_err(|e| {
        tracing::error!(
            env_id = %config.common.env_id,
            error = %e,
            "Failed to create engine context. Is the game registered?"
        );
        AppError::Config(format!("Unknown game: {}", config.common.env_id))
    })?;

    // ... rest of server setup ...

    Ok(())
}

fn register_games() {
    // Register all games, logging any failures
    if let Err(e) = games_tictactoe::try_register() {
        tracing::warn!(error = %e, "Failed to register tictactoe");
    }
    if let Err(e) = games_connect4::try_register() {
        tracing::warn!(error = %e, "Failed to register connect4");
    }
}

fn setup_shutdown_signal() -> impl std::future::Future<Output = ()> {
    async {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                tracing::info!("Received Ctrl+C, shutting down");
            },
            _ = terminate => {
                tracing::info!("Received SIGTERM, shutting down");
            },
        }
    }
}
```

#### Step 4.3: Update Actor Error Handling

Update `actor/src/actor.rs`:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EpisodeError {
    #[error("Engine error: {0}")]
    Engine(#[from] engine_core::Error),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("MCTS error: {0}")]
    Mcts(String),

    #[error("Timeout after {0} seconds")]
    Timeout(u64),

    #[error("Model not available")]
    ModelNotAvailable,
}

impl EpisodeError {
    /// Returns true if this error should trigger a retry.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            EpisodeError::Storage(_) | EpisodeError::ModelNotAvailable
        )
    }
}

/// Run an episode with proper error handling.
pub async fn run_episode(
    ctx: &mut EngineContext,
    storage: &dyn Storage,
    evaluator: &dyn Evaluator,
    config: &EpisodeConfig,
) -> Result<Episode, EpisodeError> {
    // Reset game state
    let reset = ctx.reset(config.seed, &[]).map_err(EpisodeError::Engine)?;

    let mut state = reset.state;
    let mut moves = Vec::new();

    for step in 0..config.max_steps {
        // Check timeout
        if config.start_time.elapsed() > config.timeout {
            return Err(EpisodeError::Timeout(config.timeout.as_secs()));
        }

        // Get legal moves
        let legal_mask = ctx.legal_mask(&state).map_err(EpisodeError::Engine)?;

        if legal_mask == 0 {
            // No legal moves - game should be over
            break;
        }

        // Run MCTS - handle errors gracefully
        let mcts_result = run_mcts(ctx, evaluator, &config.mcts, &state, legal_mask)
            .map_err(|e| EpisodeError::Mcts(e.to_string()))?;

        // Execute move
        let step_result = ctx
            .step(&state, &mcts_result.action_bytes)
            .map_err(EpisodeError::Engine)?;

        moves.push(MoveRecord {
            state: state.clone(),
            action: mcts_result.action,
            policy: mcts_result.policy,
            value: mcts_result.value,
        });

        if step_result.done {
            break;
        }

        state = step_result.state;
    }

    Ok(Episode {
        moves,
        outcome: determine_outcome(&state, ctx)?,
    })
}
```

#### Step 4.4: Add Fallback Behavior for Game Registration

Update `engine/engine-core/src/registry.rs`:

```rust
/// Try to register a game, returning an error if it fails.
pub fn try_register_game(
    name: String,
    factory: fn() -> Box<dyn ErasedGame>,
) -> Result<(), RegistryError> {
    let mut registry = REGISTRY.write().map_err(|_| RegistryError::LockPoisoned)?;

    if registry.contains_key(&name) {
        return Err(RegistryError::AlreadyRegistered(name));
    }

    registry.insert(name, factory);
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Registry lock poisoned")]
    LockPoisoned,

    #[error("Game '{0}' is already registered")]
    AlreadyRegistered(String),

    #[error("Game '{0}' not found")]
    NotFound(String),
}
```

#### Step 4.5: Audit and Fix All unwrap() Calls

Run this command to find all production unwrap/expect calls:

```bash
# Find all unwrap/expect in non-test code
rg "\.unwrap\(\)|\.expect\(" --type rust \
  -g '!**/tests/**' \
  -g '!**/test_*.rs' \
  -g '!**/*_test.rs' \
  actor/src web/src engine/*/src
```

For each occurrence, determine:
1. **Can it actually fail?** If not, add a comment explaining why.
2. **What should happen if it fails?** Return error, use default, or log and skip.
3. **Is it in a critical path?** Critical paths need error propagation.

#### Step 4.6: Validation Checklist

- [ ] All `unwrap()` in production code reviewed
- [ ] Critical paths return `Result` types
- [ ] Non-critical paths log and use fallbacks
- [ ] Custom error types provide context
- [ ] Error responses are user-friendly
- [ ] Panics are only in truly impossible cases (with comments)

---

## 5. Docker Security Hardening

### Current State

- Containers run as root user
- Using full `ubuntu:24.04` base image (~77MB)
- No read-only filesystem
- No capability dropping

### Implementation Steps

#### Step 5.1: Update Dockerfile.alphazero

```dockerfile
# Dockerfile.alphazero - Secured version

# ============================================
# Stage 1: Rust builder
# ============================================
FROM rust:1.75-bookworm AS rust-builder

WORKDIR /build

# Copy manifests first for dependency caching
COPY engine/Cargo.toml engine/Cargo.lock ./engine/
COPY actor/Cargo.toml actor/Cargo.lock ./actor/
COPY engine/engine-core/Cargo.toml ./engine/engine-core/
COPY engine/engine-config/Cargo.toml ./engine/engine-config/
COPY engine/games-tictactoe/Cargo.toml ./engine/games-tictactoe/
COPY engine/games-connect4/Cargo.toml ./engine/games-connect4/
COPY engine/mcts/Cargo.toml ./engine/mcts/
COPY engine/model-watcher/Cargo.toml ./engine/model-watcher/

# Create dummy source files for dependency compilation
RUN mkdir -p engine/engine-core/src && echo "fn main() {}" > engine/engine-core/src/lib.rs && \
    mkdir -p engine/engine-config/src && echo "fn main() {}" > engine/engine-config/src/lib.rs && \
    mkdir -p engine/games-tictactoe/src && echo "fn main() {}" > engine/games-tictactoe/src/lib.rs && \
    mkdir -p engine/games-connect4/src && echo "fn main() {}" > engine/games-connect4/src/lib.rs && \
    mkdir -p engine/mcts/src && echo "fn main() {}" > engine/mcts/src/lib.rs && \
    mkdir -p engine/model-watcher/src && echo "fn main() {}" > engine/model-watcher/src/lib.rs && \
    mkdir -p actor/src && echo "fn main() {}" > actor/src/main.rs

# Build dependencies only
RUN cd actor && cargo build --release 2>/dev/null || true

# Copy actual source code
COPY engine/ ./engine/
COPY actor/ ./actor/

# Touch source files to trigger rebuild
RUN find engine actor -name "*.rs" -exec touch {} \;

# Build the actual binary
RUN cd actor && cargo build --release

# ============================================
# Stage 2: Python builder
# ============================================
FROM python:3.11-slim-bookworm AS python-builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python package
COPY trainer/pyproject.toml trainer/setup.py* ./
COPY trainer/src ./src

RUN pip install --no-cache-dir --target=/install .

# ============================================
# Stage 3: Final runtime image
# ============================================
FROM debian:bookworm-slim AS runtime

# Security: Create non-root user
RUN groupadd --gid 1000 cartridge && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home cartridge

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libpq5 \
    python3 \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Rust binary
COPY --from=rust-builder /build/actor/target/release/actor /usr/local/bin/actor

# Copy Python installation
COPY --from=python-builder /install /usr/local/lib/python3.11/dist-packages

# Create data directory with correct permissions
RUN mkdir -p /data/models && chown -R cartridge:cartridge /data

# Set Python path
ENV PYTHONPATH=/usr/local/lib/python3.11/dist-packages
ENV PATH="/usr/local/bin:${PATH}"

# Security: Switch to non-root user
USER cartridge

# Set working directory
WORKDIR /home/cartridge

# Default command
CMD ["actor"]

# Labels
LABEL org.opencontainers.image.source="https://github.com/your-org/cartridge2"
LABEL org.opencontainers.image.description="Cartridge2 AlphaZero training"
```

#### Step 5.2: Update web/Dockerfile

```dockerfile
# web/Dockerfile - Secured version

# ============================================
# Stage 1: Build frontend
# ============================================
FROM node:20-slim AS frontend-builder

WORKDIR /build

# Copy package files for caching
COPY web/frontend/package*.json ./
RUN npm ci --ignore-scripts

# Copy and build frontend
COPY web/frontend/ ./
RUN npm run build

# ============================================
# Stage 2: Build Rust backend
# ============================================
FROM rust:1.75-bookworm AS rust-builder

WORKDIR /build

# Copy manifests for caching
COPY engine/Cargo.toml engine/Cargo.lock ./engine/
COPY web/Cargo.toml web/Cargo.lock ./web/
COPY engine/engine-core/Cargo.toml ./engine/engine-core/
COPY engine/engine-config/Cargo.toml ./engine/engine-config/
COPY engine/games-tictactoe/Cargo.toml ./engine/games-tictactoe/
COPY engine/games-connect4/Cargo.toml ./engine/games-connect4/
COPY engine/mcts/Cargo.toml ./engine/mcts/
COPY engine/model-watcher/Cargo.toml ./engine/model-watcher/

# Create dummy source files
RUN mkdir -p engine/engine-core/src && echo "fn main() {}" > engine/engine-core/src/lib.rs && \
    mkdir -p engine/engine-config/src && echo "fn main() {}" > engine/engine-config/src/lib.rs && \
    mkdir -p engine/games-tictactoe/src && echo "fn main() {}" > engine/games-tictactoe/src/lib.rs && \
    mkdir -p engine/games-connect4/src && echo "fn main() {}" > engine/games-connect4/src/lib.rs && \
    mkdir -p engine/mcts/src && echo "fn main() {}" > engine/mcts/src/lib.rs && \
    mkdir -p engine/model-watcher/src && echo "fn main() {}" > engine/model-watcher/src/lib.rs && \
    mkdir -p web/src && echo "fn main() {}" > web/src/main.rs

# Build dependencies
RUN cd web && cargo build --release 2>/dev/null || true

# Copy actual source
COPY engine/ ./engine/
COPY web/src ./web/src

# Touch to trigger rebuild
RUN find engine web -name "*.rs" -exec touch {} \;

# Build
RUN cd web && cargo build --release

# ============================================
# Stage 3: Runtime
# ============================================
FROM debian:bookworm-slim

# Security: Create non-root user
RUN groupadd --gid 1000 cartridge && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home cartridge

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=rust-builder /build/web/target/release/web /usr/local/bin/web

# Create data directory
RUN mkdir -p /data && chown -R cartridge:cartridge /data

# Security: Switch to non-root user
USER cartridge

WORKDIR /home/cartridge

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/web", "--health-check"] || exit 1

CMD ["/usr/local/bin/web"]
```

#### Step 5.3: Update Kubernetes Security Context

Update `k8s/base/actor/deployment.yaml`:

```yaml
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
        - name: actor
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: data
              mountPath: /data
      volumes:
        - name: tmp
          emptyDir: {}
        - name: data
          persistentVolumeClaim:
            claimName: actor-data
```

#### Step 5.4: Add Image Scanning to CI

Update `.github/workflows/ci.yml`:

```yaml
  security-scan:
    runs-on: ubuntu-latest
    needs: [build-images]
    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'gcr.io/${{ env.PROJECT_ID }}/actor:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Fail on critical vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'gcr.io/${{ env.PROJECT_ID }}/actor:${{ github.sha }}'
          format: 'table'
          exit-code: '1'
          severity: 'CRITICAL'
```

#### Step 5.5: Create .dockerignore

Create `.dockerignore`:

```
# Git
.git
.gitignore

# Build artifacts
**/target/
**/__pycache__/
**/*.pyc
**/node_modules/
**/dist/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Documentation
docs/
*.md
!README.md

# Test files
**/tests/
**/*_test.rs
**/test_*.py

# Local config
.env
.env.*
*.local

# Data directory (mounted at runtime)
data/
```

#### Step 5.6: Validation Checklist

- [ ] Images build successfully with slim base
- [ ] Containers run as non-root user (UID 1000)
- [ ] Read-only root filesystem works
- [ ] Image size reduced (target: <200MB for web, <500MB for alphazero)
- [ ] Trivy scan passes with no CRITICAL vulnerabilities
- [ ] Kubernetes pods start with security context
- [ ] Application functions correctly with dropped capabilities

---

## Summary

| Item | Before | After | Risk Reduction |
|------|--------|-------|----------------|
| **Database** | K8s StatefulSet | Cloud SQL | Automated backups, HA, encryption |
| **Logging** | Text to stdout | Structured JSON | Searchable, alertable logs |
| **Connections** | 16 per component | Sized per role | No connection exhaustion |
| **Error Handling** | Panics | Proper errors | No production crashes |
| **Containers** | Root, full Ubuntu | Non-root, slim | Smaller attack surface |

## Next Steps

After completing Part 2, proceed to:

- **Part 3**: Medium-priority items (monitoring/metrics, IaC, CI/CD improvements)
- **Part 4**: Production hardening (performance tuning, documentation, runbooks)

## Dependencies

Part 2 depends on Part 1 completion:
- Cloud SQL requires secrets in Secret Manager (Part 1, Item 1)
- Structured logging benefits from trace IDs propagated through HTTPS (Part 1, Item 2)
