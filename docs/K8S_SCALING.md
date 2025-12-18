# Kubernetes Scaling Architecture for Cartridge-2

This document outlines the strategy for scaling AlphaZero training on Kubernetes, from local development with Kind to cost-optimized cloud deployment.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Kind + SQLite](#phase-1-kind--sqlite)
3. [Phase 2: PostgreSQL Replay Buffer](#phase-2-postgresql-replay-buffer)
4. [Phase 3: Cloud Optimization](#phase-3-cloud-optimization)
5. [Component Scaling](#component-scaling)
6. [Cost Optimization](#cost-optimization)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    ConfigMap                              │    │
│  │  config.toml (shared settings)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               Orchestrator (singleton)                    │    │
│  │  - Coordinates training iterations                       │    │
│  │  - Scales actor deployment                               │    │
│  │  - Creates trainer Jobs                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Actor Pod 1 │  │ Actor Pod 2 │  │ Actor Pod N │              │
│  │             │  │             │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Replay Buffer (SQLite → PostgreSQL)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ▲                                       │
│                          │                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Trainer Job                             │    │
│  │  (GPU-enabled, runs per iteration)                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Type | Scaling | Purpose |
|-----------|------|---------|---------|
| **Orchestrator** | Deployment (replicas: 1) | Singleton | Coordinates training loop |
| **Actors** | Deployment | Horizontal (1-N pods) | Self-play episode generation |
| **Trainer** | Job | Vertical (GPU) | Neural network training |
| **Replay Buffer** | SQLite/PostgreSQL | Vertical | Transition storage |
| **Model Storage** | PVC (RWX) | N/A | ONNX model sharing |

---

## Phase 1: Kind + SQLite

**Goal**: Get running on Kubernetes locally with minimal changes.

### Limitations
- SQLite single-writer constraint limits to ~3-5 actors
- Local storage (no RWX) requires all pods on same node
- No GPU support in Kind (CPU training only)

### Setup

```bash
# 1. Create Kind cluster
kind create cluster --config k8s/kind-cluster.yaml --name cartridge

# 2. Build and load images
docker build -f actor/Dockerfile -t cartridge/actor:latest .
docker build -f trainer/Dockerfile -t cartridge/trainer:latest .
kind load docker-image cartridge/actor:latest --name cartridge
kind load docker-image cartridge/trainer:latest --name cartridge

# 3. Deploy
kubectl apply -k k8s/base

# 4. Scale actors (start small!)
kubectl scale deployment actor --replicas=2 -n cartridge

# 5. Monitor
kubectl logs -f deployment/orchestrator -n cartridge
kubectl get pods -n cartridge -w
```

### SQLite WAL Mode

To support multiple actors reading while trainer writes, enable WAL mode:

```rust
// In actor/src/replay.rs
conn.execute("PRAGMA journal_mode=WAL", [])?;
conn.execute("PRAGMA busy_timeout=5000", [])?;  // Wait 5s on lock
```

This allows:
- Multiple concurrent readers
- One writer (with readers continuing)
- 5-second timeout on write locks

### Phase 1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PVC (ReadWriteOnce)                          │
│  Single node constraint - all pods must schedule here           │
├─────────────────────────────────────────────────────────────────┤
│  replay.db (WAL mode)                                           │
│  models/latest.onnx                                             │
│  stats.json                                                     │
└─────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
┌────────┴────────┐  ┌───────┴───────┐  ┌────────┴────────┐
│  Actor Pod 1    │  │  Actor Pod 2  │  │  Trainer Pod    │
│  (writer)       │  │  (writer)     │  │  (writer)       │
│                 │  │               │  │                 │
│  Contention!    │  │  Contention!  │  │  Contention!    │
└─────────────────┘  └───────────────┘  └─────────────────┘
```

---

## Phase 2: PostgreSQL Replay Buffer

**Goal**: True horizontal scaling with 10+ actors.

### Why PostgreSQL?
- Native concurrent writes
- ACID transactions
- Connection pooling (PgBouncer)
- Easy migration from SQLite schema
- Managed options in cloud (RDS, Cloud SQL)

### Schema Migration

```sql
-- Same schema as SQLite, PostgreSQL syntax
CREATE TABLE transitions (
    id TEXT PRIMARY KEY,
    env_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    state BYTEA NOT NULL,
    action BYTEA NOT NULL,
    next_state BYTEA NOT NULL,
    observation BYTEA NOT NULL,
    next_observation BYTEA NOT NULL,
    reward REAL NOT NULL,
    done BOOLEAN NOT NULL,
    timestamp BIGINT NOT NULL,
    policy_probs BYTEA,
    mcts_value REAL DEFAULT 0.0,
    game_outcome REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_transitions_timestamp ON transitions(timestamp);
CREATE INDEX idx_transitions_episode ON transitions(episode_id);
CREATE INDEX idx_transitions_env_id ON transitions(env_id);

-- Game metadata table
CREATE TABLE game_metadata (
    env_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    board_width INTEGER NOT NULL,
    board_height INTEGER NOT NULL,
    num_actions INTEGER NOT NULL,
    obs_size INTEGER NOT NULL,
    legal_mask_offset INTEGER NOT NULL,
    player_count INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Implementation Changes

**Actor (Rust)**:
```rust
// Add to Cargo.toml
// tokio-postgres = "0.7"
// deadpool-postgres = "0.10"

// New replay_postgres.rs
pub struct PostgresReplayBuffer {
    pool: Pool,
}

impl PostgresReplayBuffer {
    pub async fn store_batch(&self, transitions: &[Transition]) -> Result<()> {
        let client = self.pool.get().await?;
        let stmt = client.prepare_cached(
            "INSERT INTO transitions (...) VALUES ($1, $2, ...)"
        ).await?;

        for t in transitions {
            client.execute(&stmt, &[&t.id, &t.env_id, ...]).await?;
        }
        Ok(())
    }
}
```

**Trainer (Python)**:
```python
# Add to requirements
# psycopg2-binary or asyncpg

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

class PostgresReplayBuffer:
    def __init__(self, connection_string: str):
        self.pool = ThreadedConnectionPool(1, 10, connection_string)

    def sample_batch_tensors(self, batch_size: int, num_actions: int):
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT observation, policy_probs, game_outcome
                FROM transitions
                ORDER BY RANDOM()
                LIMIT %s
            """, (batch_size,))
            # ... process results
        finally:
            self.pool.putconn(conn)
```

### Phase 2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PostgreSQL StatefulSet                        │
│                    (+ PgBouncer connection pool)                 │
├─────────────────────────────────────────────────────────────────┤
│  transitions table (indexed)                                     │
│  game_metadata table                                             │
└─────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
┌────────┴────────┐  ┌───────┴───────┐  ┌────────┴────────┐
│  Actor Pod 1    │  │  Actor Pod N  │  │  Trainer Pod    │
│  (parallel      │  │  (parallel    │  │  (parallel      │
│   writes)       │  │   writes)     │  │   reads)        │
└─────────────────┘  └───────────────┘  └─────────────────┘
         │                    │                │
         └────────────────────┼────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               PVC: models (ReadWriteMany)                        │
│  latest.onnx (trainer writes, actors read)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 3: Cloud Optimization

### Cost Components

| Component | Cost Driver | Optimization |
|-----------|-------------|--------------|
| **Actors** | CPU hours | Spot/Preemptible instances |
| **Trainer** | GPU hours | Spot GPU, right-size instance |
| **Storage** | PVC size + IOPS | Use S3/GCS for checkpoints |
| **Database** | Instance + storage | Right-size, use serverless |
| **Networking** | Cross-AZ traffic | Single-AZ for training |

### Spot Instance Strategy

Actors are **perfect for spot instances**:
- Stateless (can restart anytime)
- Work is batched (episodes complete independently)
- No critical state to lose

```yaml
# Actor deployment with spot tolerance
spec:
  template:
    spec:
      tolerations:
        - key: "kubernetes.io/spot"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
      nodeSelector:
        node.kubernetes.io/instance-type: spot
```

### GPU Trainer Optimization

```yaml
# Trainer job with GPU spot instance
spec:
  template:
    spec:
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
        - key: "kubernetes.io/spot"
          operator: "Equal"
          value: "true"
      nodeSelector:
        accelerator: nvidia-tesla-t4  # Cost-effective GPU
      containers:
        - resources:
            limits:
              nvidia.com/gpu: 1
```

### Object Storage for Checkpoints

Instead of expensive PVC storage for model checkpoints:

```python
# trainer/checkpoint.py - S3 upload
import boto3

def save_checkpoint_s3(model, step, bucket="cartridge-models"):
    s3 = boto3.client('s3')

    # Save locally first
    local_path = f"/tmp/checkpoint_{step}.onnx"
    torch.onnx.export(model, dummy_input, local_path)

    # Upload to S3
    s3.upload_file(local_path, bucket, f"checkpoints/checkpoint_{step}.onnx")

    # Update latest pointer
    s3.copy_object(
        Bucket=bucket,
        CopySource=f"{bucket}/checkpoints/checkpoint_{step}.onnx",
        Key="latest.onnx"
    )
```

### Auto-scaling Based on Queue Depth

Scale actors based on replay buffer fill rate:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: actor-hpa
  namespace: cartridge
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: actor
  minReplicas: 1
  maxReplicas: 20
  metrics:
    - type: External
      external:
        metric:
          name: replay_buffer_size
          selector:
            matchLabels:
              app: cartridge
        target:
          type: Value
          value: "10000"  # Target buffer size
```

---

## Component Scaling

### Actor Scaling

| Replicas | Use Case | Notes |
|----------|----------|-------|
| 1-2 | Development | SQLite OK |
| 3-5 | Testing | SQLite with WAL |
| 10-50 | Production | Requires PostgreSQL |
| 50+ | Large-scale | Consider Redis Streams |

### Trainer Scaling

The trainer is typically **not horizontally scaled** in AlphaZero:
- Single model must be consistent
- Gradient updates must be sequential
- GPU parallelism is within the batch

**Vertical scaling options**:
- Larger GPU (T4 → A100)
- Multi-GPU (DataParallel)
- Mixed precision training (fp16)

### Orchestrator

**Always singleton**. Uses leader election if HA is needed:

```yaml
# Add leader election to orchestrator
env:
  - name: LEADER_ELECTION_ENABLED
    value: "true"
  - name: LEADER_ELECTION_NAMESPACE
    valueFrom:
      fieldRef:
        fieldPath: metadata.namespace
```

---

## Cost Optimization Strategies

### 1. Iteration-Based Scaling

Scale actors to 0 during training:

```
Iteration timeline:
├── [Actors: N] Generate episodes (5 min)
├── [Actors: 0] Train model (2 min)      ← Save $$$ here
├── [Actors: 0] Evaluate (1 min)
└── Repeat
```

### 2. Spot Instance Mix

```yaml
# 80% spot, 20% on-demand for reliability
spec:
  replicas: 10
  template:
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 20
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    node-type: on-demand
```

### 3. Right-Sizing

**Actors (CPU-bound MCTS)**:
- TicTacToe: 0.5 CPU, 256MB RAM
- Connect4: 1 CPU, 512MB RAM
- Complex games: 2 CPU, 1GB RAM

**Trainer (GPU-bound)**:
- TicTacToe: No GPU needed, CPU is fast enough
- Connect4: T4 GPU (cheapest)
- Complex games: A10/A100 GPU

### 4. Multi-Region Spot

Spread across regions for spot availability:

```yaml
# GKE example
nodeSelector:
  topology.kubernetes.io/zone: us-central1-a
  # With fallback:
  # topology.kubernetes.io/zone in (us-central1-a, us-central1-b, us-central1-c)
```

---

## Quick Reference

### Commands

```bash
# Create Kind cluster
kind create cluster --config k8s/kind-cluster.yaml --name cartridge

# Deploy
kubectl apply -k k8s/base

# Scale actors
kubectl scale deployment actor --replicas=5 -n cartridge

# Watch progress
kubectl logs -f deployment/orchestrator -n cartridge

# Manually trigger training
kubectl create job trainer-manual --from=job/trainer -n cartridge

# Check replay buffer (exec into pod)
kubectl exec -it deployment/actor -n cartridge -- \
  sqlite3 /data/replay.db "SELECT COUNT(*) FROM transitions"

# Port-forward web UI
kubectl port-forward svc/web 8080:8080 -n cartridge
```

### Environment Variables

```bash
# Override game
CARTRIDGE_COMMON_ENV_ID=connect4

# Override training params
CARTRIDGE_TRAINING_ITERATIONS=100
CARTRIDGE_TRAINING_EPISODES_PER_ITERATION=1000
CARTRIDGE_TRAINING_STEPS_PER_ITERATION=500

# GPU training
CARTRIDGE_TRAINING_DEVICE=cuda
```

---

## Roadmap

- [x] Phase 1: Kind + SQLite (this document)
- [ ] Phase 2: PostgreSQL migration
- [ ] Phase 3: Cloud deployment (GKE/EKS)
- [ ] Phase 4: Auto-scaling based on metrics
- [ ] Phase 5: Multi-game parallel training
