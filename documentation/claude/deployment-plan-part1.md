# Deployment Plan Part 1: Critical Security Items

This document outlines the implementation plan for all **CRITICAL** security items that must be addressed before deploying Cartridge-2 to Google Cloud (or AWS).

## Overview

| # | Issue | Severity | Estimated Effort |
|---|-------|----------|------------------|
| 1 | Hardcoded Credentials | CRITICAL | 4-6 hours |
| 2 | No HTTPS/TLS | CRITICAL | 2-4 hours |
| 3 | Overly Permissive CORS | CRITICAL | 1-2 hours |
| 4 | No Actor Health Checks | CRITICAL | 2-3 hours |

**Total Estimated Effort:** 9-15 hours

---

## 1. Hardcoded Credentials Remediation

### Current State

Credentials are hardcoded in multiple files:

| File | Line(s) | Issue |
|------|---------|-------|
| `docker-compose.yml` | 27, 34-36, 75-77, 98 | MinIO creds `aspect/password123` |
| `k8s/base/configmap.yaml` | 62-65, 78-81 | PostgreSQL and MinIO secrets in K8s Secret |
| `config.toml` | 170 | Default PostgreSQL URL |
| `config.defaults.toml` | 62 | Default PostgreSQL URL |
| `actor/src/config.rs` | 63-71 | Hardcoded fallback credentials |
| `trainer/src/trainer/central_config.py` | defaults | PostgreSQL URL in defaults |

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Google Cloud Secret Manager                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ db-password │  │ minio-creds │  │ other-secrets       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│              GKE Workload Identity Federation                │
│         (Pods authenticate as GCP Service Account)          │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────────┐
│  Actor Pods    │  │  Trainer Pods  │  │  Web Server Pods   │
│  (env vars)    │  │  (env vars)    │  │  (env vars)        │
└────────────────┘  └────────────────┘  └────────────────────┘
```

### Implementation Steps

#### Step 1.1: Create GCP Secret Manager Secrets

```bash
# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# Create secrets (replace with actual values)
echo -n "STRONG_PASSWORD_HERE" | \
  gcloud secrets create cartridge-db-password \
    --data-file=- \
    --replication-policy="automatic"

echo -n "MINIO_ACCESS_KEY" | \
  gcloud secrets create cartridge-minio-access-key \
    --data-file=- \
    --replication-policy="automatic"

echo -n "MINIO_SECRET_KEY" | \
  gcloud secrets create cartridge-minio-secret-key \
    --data-file=- \
    --replication-policy="automatic"
```

#### Step 1.2: Create GCP Service Account for Workload Identity

```bash
# Create service account
gcloud iam service-accounts create cartridge-workload \
  --display-name="Cartridge Workload Identity"

# Grant secret access
gcloud secrets add-iam-policy-binding cartridge-db-password \
  --member="serviceAccount:cartridge-workload@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding cartridge-minio-access-key \
  --member="serviceAccount:cartridge-workload@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding cartridge-minio-secret-key \
  --member="serviceAccount:cartridge-workload@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

#### Step 1.3: Configure Workload Identity in GKE

```bash
# Bind K8s service account to GCP service account
gcloud iam service-accounts add-iam-policy-binding \
  cartridge-workload@PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:PROJECT_ID.svc.id.goog[cartridge/cartridge-sa]"
```

#### Step 1.4: Create New Kubernetes Manifests

Create `k8s/base/secrets-external.yaml`:

```yaml
# k8s/base/secrets-external.yaml
# Uses External Secrets Operator to sync from GCP Secret Manager
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: gcp-secret-store
  namespace: cartridge
spec:
  provider:
    gcpsm:
      projectID: PROJECT_ID
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: cartridge-secrets
  namespace: cartridge
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: gcp-secret-store
    kind: SecretStore
  target:
    name: cartridge-secrets
    creationPolicy: Owner
  data:
    - secretKey: POSTGRES_PASSWORD
      remoteRef:
        key: cartridge-db-password
    - secretKey: MINIO_ACCESS_KEY
      remoteRef:
        key: cartridge-minio-access-key
    - secretKey: MINIO_SECRET_KEY
      remoteRef:
        key: cartridge-minio-secret-key
```

Create `k8s/base/service-account.yaml`:

```yaml
# k8s/base/service-account.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cartridge-sa
  namespace: cartridge
  annotations:
    iam.gke.io/gcp-service-account: cartridge-workload@PROJECT_ID.iam.gserviceaccount.com
```

#### Step 1.5: Update Kubernetes Deployments

Update `k8s/base/actor/deployment.yaml` to use external secrets:

```yaml
# In spec.template.spec:
serviceAccountName: cartridge-sa
containers:
  - name: actor
    env:
      - name: CARTRIDGE_STORAGE_POSTGRES_PASSWORD
        valueFrom:
          secretKeyRef:
            name: cartridge-secrets
            key: POSTGRES_PASSWORD
      # Construct URL in init container or use separate env vars
      - name: CARTRIDGE_STORAGE_POSTGRES_HOST
        valueFrom:
          configMapKeyRef:
            name: cartridge-config
            key: POSTGRES_HOST
      - name: CARTRIDGE_STORAGE_POSTGRES_USER
        value: "cartridge"
      - name: CARTRIDGE_STORAGE_POSTGRES_DB
        value: "cartridge"
```

#### Step 1.6: Update Rust Code to Support Split Credentials

Modify `actor/src/config.rs` to support building PostgreSQL URL from components:

```rust
// actor/src/config.rs - Add new fields and builder logic

#[derive(Debug, Clone)]
pub struct PostgresConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,  // From secret
    pub database: String,
}

impl PostgresConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        Ok(Self {
            host: std::env::var("CARTRIDGE_STORAGE_POSTGRES_HOST")
                .unwrap_or_else(|_| "localhost".to_string()),
            port: std::env::var("CARTRIDGE_STORAGE_POSTGRES_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(5432),
            user: std::env::var("CARTRIDGE_STORAGE_POSTGRES_USER")
                .unwrap_or_else(|_| "cartridge".to_string()),
            password: std::env::var("CARTRIDGE_STORAGE_POSTGRES_PASSWORD")
                .map_err(|_| ConfigError::MissingSecret("POSTGRES_PASSWORD"))?,
            database: std::env::var("CARTRIDGE_STORAGE_POSTGRES_DB")
                .unwrap_or_else(|_| "cartridge".to_string()),
        })
    }

    pub fn to_url(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.user, self.password, self.host, self.port, self.database
        )
    }
}
```

#### Step 1.7: Update Python Trainer Configuration

Modify `trainer/src/trainer/central_config.py`:

```python
# trainer/src/trainer/central_config.py - Add secret-aware config loading

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    user: str = "cartridge"
    password: Optional[str] = None  # Must come from environment
    database: str = "cartridge"

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        password = os.environ.get("CARTRIDGE_STORAGE_POSTGRES_PASSWORD")
        if not password:
            raise ValueError(
                "CARTRIDGE_STORAGE_POSTGRES_PASSWORD environment variable is required. "
                "Do not hardcode credentials in configuration files."
            )

        return cls(
            host=os.environ.get("CARTRIDGE_STORAGE_POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("CARTRIDGE_STORAGE_POSTGRES_PORT", "5432")),
            user=os.environ.get("CARTRIDGE_STORAGE_POSTGRES_USER", "cartridge"),
            password=password,
            database=os.environ.get("CARTRIDGE_STORAGE_POSTGRES_DB", "cartridge"),
        )

    def to_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
```

#### Step 1.8: Remove Hardcoded Credentials from Files

Files to modify:

1. **`config.toml`** - Remove `postgres_url` default, add placeholder:
   ```toml
   [storage]
   # PostgreSQL credentials must be provided via environment variables:
   # - CARTRIDGE_STORAGE_POSTGRES_HOST
   # - CARTRIDGE_STORAGE_POSTGRES_PORT
   # - CARTRIDGE_STORAGE_POSTGRES_USER
   # - CARTRIDGE_STORAGE_POSTGRES_PASSWORD (required, from Secret Manager)
   # - CARTRIDGE_STORAGE_POSTGRES_DB
   postgres_url = ""  # Constructed from environment variables
   ```

2. **`config.defaults.toml`** - Same changes

3. **`docker-compose.yml`** - Use `.env` file for local development:
   ```yaml
   services:
     minio:
       environment:
         MINIO_ROOT_USER: ${MINIO_ROOT_USER}
         MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
   ```

4. **`k8s/base/configmap.yaml`** - Remove Secret resources, use ExternalSecret

#### Step 1.9: Create Local Development `.env.example`

Create `.env.example` (committed) and `.env` (gitignored):

```bash
# .env.example - Copy to .env and fill in values
# NEVER commit .env to version control

# PostgreSQL
POSTGRES_PASSWORD=change_me_in_local_env

# MinIO (local S3-compatible storage)
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=change_me_in_local_env

# AWS/GCS credentials (for S3 backend)
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
```

Update `.gitignore`:
```
# Secrets
.env
.env.local
.env.*.local
```

#### Step 1.10: Validation Checklist

- [ ] All secrets removed from version-controlled files
- [ ] `git log -p --all -S 'password123'` returns no results
- [ ] `git log -p --all -S 'cartridge:cartridge'` returns no results
- [ ] External Secrets Operator deployed to GKE
- [ ] Workload Identity configured and tested
- [ ] Local development uses `.env` file
- [ ] CI/CD uses environment variables from secret store

---

## 2. HTTPS/TLS Implementation

### Current State

- `web/src/main.rs:232` - Server binds to HTTP only
- `web/frontend/nginx.conf` - No SSL configuration
- No certificate management

### Target Architecture

```
                    ┌─────────────────────────────────┐
                    │   Google Cloud Load Balancer    │
                    │   (Managed SSL Certificate)     │
                    │   https://cartridge.example.com │
                    └─────────────────────────────────┘
                                    │
                                    │ HTTPS terminated
                                    ▼
                    ┌─────────────────────────────────┐
                    │        GKE Ingress              │
                    │   (BackendConfig for health)    │
                    └─────────────────────────────────┘
                           │                │
                           ▼                ▼
                    ┌───────────┐    ┌───────────┐
                    │ Web Svc   │    │ Frontend  │
                    │ :8080     │    │ :80       │
                    └───────────┘    └───────────┘
```

### Implementation Steps

#### Step 2.1: Create Managed Certificate

Create `k8s/base/managed-certificate.yaml`:

```yaml
# k8s/base/managed-certificate.yaml
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: cartridge-certificate
  namespace: cartridge
spec:
  domains:
    - cartridge.example.com  # Replace with your domain
    - www.cartridge.example.com
```

#### Step 2.2: Create GKE Ingress with HTTPS

Create `k8s/base/ingress.yaml`:

```yaml
# k8s/base/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cartridge-ingress
  namespace: cartridge
  annotations:
    # Use GCP managed certificate
    networking.gke.io/managed-certificates: cartridge-certificate
    # Redirect HTTP to HTTPS
    kubernetes.io/ingress.allow-http: "false"
    # Use GCP external HTTP(S) load balancer
    kubernetes.io/ingress.class: "gce"
    # Frontend config for HTTPS redirect
    networking.gke.io/v1beta1.FrontendConfig: cartridge-frontend-config
spec:
  rules:
    - host: cartridge.example.com
      http:
        paths:
          # API routes
          - path: /health
            pathType: Exact
            backend:
              service:
                name: web
                port:
                  number: 8080
          - path: /game/*
            pathType: Prefix
            backend:
              service:
                name: web
                port:
                  number: 8080
          - path: /move
            pathType: Exact
            backend:
              service:
                name: web
                port:
                  number: 8080
          - path: /stats
            pathType: Exact
            backend:
              service:
                name: web
                port:
                  number: 8080
          # Frontend (default)
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend
                port:
                  number: 80
---
apiVersion: networking.gke.io/v1beta1
kind: FrontendConfig
metadata:
  name: cartridge-frontend-config
  namespace: cartridge
spec:
  redirectToHttps:
    enabled: true
    responseCodeName: MOVED_PERMANENTLY_DEFAULT
```

#### Step 2.3: Add Security Headers to Nginx

Update `web/frontend/nginx.conf`:

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;

    # Content Security Policy (adjust as needed for your app)
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' https://cartridge.example.com;" always;

    # HSTS (only enable after confirming HTTPS works)
    # add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy (for local development only; in prod, Ingress routes directly)
    location /api/ {
        proxy_pass http://web:8080/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Step 2.4: Configure DNS

```bash
# Get the external IP of the load balancer
kubectl get ingress cartridge-ingress -n cartridge -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Create DNS A record pointing to this IP:
# cartridge.example.com -> <EXTERNAL_IP>
```

#### Step 2.5: Verify Certificate Provisioning

```bash
# Check certificate status (may take 10-60 minutes)
kubectl describe managedcertificate cartridge-certificate -n cartridge

# Look for:
# Status:
#   CertificateStatus: Active
```

#### Step 2.6: Validation Checklist

- [ ] Managed certificate created and status is "Active"
- [ ] Ingress routes traffic correctly
- [ ] HTTP automatically redirects to HTTPS
- [ ] Security headers present (check with `curl -I`)
- [ ] SSL Labs test passes (https://www.ssllabs.com/ssltest/)

---

## 3. CORS Configuration Fix

### Current State

```rust
// web/src/main.rs:80-84
let cors = CorsLayer::new()
    .allow_origin(Any)      // DANGEROUS: Allows any origin
    .allow_methods(Any)     // DANGEROUS: Allows any method
    .allow_headers(Any);    // DANGEROUS: Allows any header
```

### Target State

Restrict CORS to only allow requests from the legitimate frontend domain.

### Implementation Steps

#### Step 3.1: Add Configuration for Allowed Origins

Update `engine/engine-config/src/structs.rs`:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct WebConfig {
    pub host: String,
    pub port: u16,
    #[serde(default)]
    pub allowed_origins: Vec<String>,  // New field
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            allowed_origins: vec![],  // Empty = development mode (allow all)
        }
    }
}
```

#### Step 3.2: Update Web Server CORS Configuration

Update `web/src/main.rs`:

```rust
use axum::http::{HeaderValue, Method};
use tower_http::cors::{AllowOrigin, CorsLayer};

fn configure_cors(allowed_origins: &[String]) -> CorsLayer {
    if allowed_origins.is_empty() {
        // Development mode: allow all origins with a warning
        tracing::warn!(
            "CORS: No allowed_origins configured. Allowing all origins. \
             This is insecure for production!"
        );
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::ACCEPT,
            ])
    } else {
        // Production mode: restrict to configured origins
        let origins: Vec<HeaderValue> = allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();

        tracing::info!("CORS: Allowing origins: {:?}", allowed_origins);

        CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::ACCEPT,
            ])
            .allow_credentials(true)
    }
}

// In main() or router setup:
let cors = configure_cors(&config.web.allowed_origins);
```

#### Step 3.3: Update Configuration Files

Update `config.toml`:

```toml
[web]
host = "0.0.0.0"
port = 8080
# Production: set to your actual frontend domain(s)
# allowed_origins = ["https://cartridge.example.com", "https://www.cartridge.example.com"]
# Development: leave empty or omit to allow all origins (with warning)
allowed_origins = []
```

#### Step 3.4: Update Kubernetes ConfigMap

Update `k8s/base/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cartridge-config
  namespace: cartridge
data:
  # ... other config ...
  CARTRIDGE_WEB_ALLOWED_ORIGINS: "https://cartridge.example.com,https://www.cartridge.example.com"
```

#### Step 3.5: Add CORS Tests

Add to `web/src/main.rs` (or `web/src/tests/`):

```rust
#[cfg(test)]
mod cors_tests {
    use super::*;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_cors_rejects_unknown_origin_in_production() {
        let allowed = vec!["https://allowed.example.com".to_string()];
        let cors = configure_cors(&allowed);

        // Create app with CORS
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .layer(cors);

        // Request from disallowed origin
        let request = Request::builder()
            .uri("/health")
            .header("Origin", "https://evil.example.com")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        // Should not have Access-Control-Allow-Origin header for disallowed origin
        assert!(response.headers().get("access-control-allow-origin").is_none());
    }

    #[tokio::test]
    async fn test_cors_allows_configured_origin() {
        let allowed = vec!["https://allowed.example.com".to_string()];
        let cors = configure_cors(&allowed);

        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .layer(cors);

        let request = Request::builder()
            .uri("/health")
            .header("Origin", "https://allowed.example.com")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(
            response.headers().get("access-control-allow-origin").unwrap(),
            "https://allowed.example.com"
        );
    }
}
```

#### Step 3.6: Validation Checklist

- [ ] CORS configuration added to config structs
- [ ] Web server uses configurable CORS
- [ ] Production config specifies exact allowed origins
- [ ] Development mode logs warning about permissive CORS
- [ ] Tests verify CORS behavior
- [ ] Preflight (OPTIONS) requests work correctly

---

## 4. Actor Health Checks

### Current State

- Actor binary has no health endpoint
- Kubernetes deployment has no liveness/readiness probes
- Actor can hang without detection

### Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Actor Pod                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Actor Process                       │   │
│  │  ┌─────────────┐  ┌─────────────────────────┐   │   │
│  │  │ Episode     │  │ Health HTTP Server      │   │   │
│  │  │ Runner      │  │ :8081                   │   │   │
│  │  │ (main loop) │  │ - /health (liveness)    │   │   │
│  │  │             │  │ - /ready (readiness)    │   │   │
│  │  └─────────────┘  └─────────────────────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         ▲                       ▲
         │                       │
    Main thread            Health thread
    (episode generation)   (HTTP server)
```

### Implementation Steps

#### Step 4.1: Add Health Server to Actor

Create `actor/src/health.rs`:

```rust
//! Health check HTTP server for Kubernetes probes.

use axum::{routing::get, Router};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, error};

/// Shared health state between main actor loop and health server.
#[derive(Debug, Clone)]
pub struct HealthState {
    /// Set to true once the actor has completed initialization.
    ready: Arc<AtomicBool>,
    /// Set to false if the actor encounters a fatal error.
    healthy: Arc<AtomicBool>,
    /// Timestamp of last successful episode completion.
    last_episode_time: Arc<std::sync::atomic::AtomicU64>,
}

impl HealthState {
    pub fn new() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(false)),
            healthy: Arc::new(AtomicBool::new(true)),
            last_episode_time: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Mark the actor as ready to receive traffic.
    pub fn set_ready(&self) {
        self.ready.store(true, Ordering::SeqCst);
        info!("Actor marked as ready");
    }

    /// Mark the actor as unhealthy (will trigger restart).
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
        .route("/health", get({
            let state = state.clone();
            move || health_handler(state.clone())
        }))
        .route("/ready", get({
            let state = state.clone();
            move || ready_handler(state.clone())
        }))
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
    // TODO: Integrate with metrics crate
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
        state.record_episode_complete();
        assert!(state.is_making_progress(300));
    }
}
```

#### Step 4.2: Integrate Health Server with Actor Main

Update `actor/src/main.rs`:

```rust
mod health;

use health::{HealthState, start_health_server};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... existing initialization ...

    // Create shared health state
    let health_state = HealthState::new();

    // Start health server in background
    let health_port = std::env::var("HEALTH_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8081);

    let health_handle = {
        let state = health_state.clone();
        tokio::spawn(async move {
            if let Err(e) = start_health_server(health_port, state).await {
                tracing::error!("Health server error: {}", e);
            }
        })
    };

    // ... existing model loading, storage connection, etc. ...

    // Mark as ready once initialization is complete
    health_state.set_ready();

    // In episode loop, record completions:
    loop {
        match run_episode(&mut ctx, &storage, &evaluator).await {
            Ok(_) => {
                health_state.record_episode_complete();
            }
            Err(e) => {
                tracing::error!("Episode error: {}", e);
                // Don't mark unhealthy for transient errors
                // Only mark unhealthy for fatal errors
            }
        }

        // ... check shutdown signal, etc. ...
    }
}
```

#### Step 4.3: Add Health Port to Configuration

Update `actor/src/config.rs`:

```rust
#[derive(Debug, Clone)]
pub struct ActorConfig {
    // ... existing fields ...
    pub health_port: u16,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            health_port: 8081,
        }
    }
}
```

#### Step 4.4: Update Kubernetes Deployment

Update `k8s/base/actor/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: actor
  namespace: cartridge
spec:
  replicas: 4
  selector:
    matchLabels:
      app: actor
  template:
    metadata:
      labels:
        app: actor
    spec:
      serviceAccountName: cartridge-sa
      containers:
        - name: actor
          image: actor:latest
          ports:
            - containerPort: 8081
              name: health
          env:
            - name: HEALTH_PORT
              value: "8081"
            # ... other env vars ...

          # Startup probe: give actor time to initialize (load model, connect to DB)
          startupProbe:
            httpGet:
              path: /ready
              port: health
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 30  # 5 * 30 = 150 seconds max startup time

          # Liveness probe: restart if unhealthy
          livenessProbe:
            httpGet:
              path: /health
              port: health
            initialDelaySeconds: 10
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3  # 3 failures = restart

          # Readiness probe: remove from service if not ready
          readinessProbe:
            httpGet:
              path: /ready
              port: health
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
```

#### Step 4.5: Add Dependencies to Cargo.toml

Update `actor/Cargo.toml`:

```toml
[dependencies]
# ... existing dependencies ...
axum = "0.7"
tokio = { version = "1", features = ["full"] }
```

#### Step 4.6: Validation Checklist

- [ ] Health module compiles and tests pass
- [ ] Actor starts health server on configured port
- [ ] `/health` returns 200 when healthy, 503 when not
- [ ] `/ready` returns 200 when ready, 503 when not
- [ ] Progress tracking detects stuck actors
- [ ] Kubernetes probes configured correctly
- [ ] Actor restarts when liveness probe fails
- [ ] Actor removed from load balancing when readiness fails

---

## Validation & Testing Plan

### Pre-Deployment Testing

#### Local Testing

```bash
# 1. Test credentials are not hardcoded
grep -r "password123" . --include="*.yaml" --include="*.toml" --include="*.rs"
grep -r "cartridge:cartridge" . --include="*.yaml" --include="*.toml" --include="*.rs"
# Both should return no results (except in comments/docs)

# 2. Test CORS configuration
cd web && cargo test cors

# 3. Test health endpoints
cd actor && cargo test health

# 4. Test with docker-compose (using .env file)
cp .env.example .env
# Edit .env with test credentials
docker-compose up -d
curl http://localhost:8080/health
curl http://localhost:8081/health  # Actor health
curl http://localhost:8081/ready   # Actor readiness
```

#### Staging Environment Testing

```bash
# 1. Deploy to staging GKE cluster
kubectl apply -k k8s/overlays/staging/

# 2. Verify secrets are loaded from Secret Manager
kubectl exec -it deployment/actor -n cartridge -- env | grep POSTGRES
# Should show values, not "cartridge:cartridge"

# 3. Verify HTTPS works
curl -I https://staging.cartridge.example.com/health
# Should show HTTP/2 200

# 4. Verify CORS
curl -I -H "Origin: https://evil.example.com" https://staging.cartridge.example.com/health
# Should NOT have Access-Control-Allow-Origin header

# 5. Verify health probes work
kubectl describe pod -l app=actor -n cartridge
# Should show successful probe results

# 6. Test actor restart on failure
kubectl exec -it deployment/actor -n cartridge -- kill -9 1
# Pod should restart automatically
```

### Security Scanning

```bash
# 1. Scan for secrets in git history
trufflehog git file://. --since-commit HEAD~100

# 2. Container image scanning
trivy image gcr.io/PROJECT_ID/actor:latest
trivy image gcr.io/PROJECT_ID/web:latest

# 3. Kubernetes manifest scanning
kubesec scan k8s/base/actor/deployment.yaml
```

---

## Rollback Plan

If issues are discovered after deployment:

### Immediate Rollback

```bash
# Rollback Kubernetes deployments
kubectl rollout undo deployment/actor -n cartridge
kubectl rollout undo deployment/web -n cartridge
kubectl rollout undo deployment/trainer -n cartridge

# Verify rollback
kubectl rollout status deployment/actor -n cartridge
```

### Secret Rotation

If credentials are compromised:

```bash
# 1. Rotate in Secret Manager
echo -n "NEW_PASSWORD" | gcloud secrets versions add cartridge-db-password --data-file=-

# 2. Restart pods to pick up new secret
kubectl rollout restart deployment/actor -n cartridge
kubectl rollout restart deployment/trainer -n cartridge

# 3. Update database password
# (Cloud SQL: via Console or gcloud sql users set-password)
```

---

## Timeline Estimate

| Phase | Tasks | Effort |
|-------|-------|--------|
| **Phase 1** | Credentials cleanup (Steps 1.1-1.10) | 4-6 hours |
| **Phase 2** | HTTPS/TLS setup (Steps 2.1-2.6) | 2-4 hours |
| **Phase 3** | CORS fix (Steps 3.1-3.6) | 1-2 hours |
| **Phase 4** | Actor health checks (Steps 4.1-4.6) | 2-3 hours |
| **Phase 5** | Testing & validation | 2-3 hours |
| **Total** | | **11-18 hours** |

---

## Next Steps

After completing Part 1 (Critical items), proceed to:

- **Part 2**: High-priority items (managed database, structured logging, connection pooling)
- **Part 3**: Medium-priority items (monitoring, IaC, CI/CD improvements)
- **Part 4**: Production hardening (performance tuning, documentation, runbooks)
