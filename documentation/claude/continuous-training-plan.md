# Continuous Training Architecture Plan

## Overview

Migrate from monolithic `alphazero-k8s` orchestrator to a scalable continuous training setup with separate actors and trainer.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   actor-k8s-1   │     │   actor-k8s-2   │     │   actor-k8s-N   │
│  (continuous)   │     │  (continuous)   │     │  (continuous)   │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │ Store transitions
                                 ▼
                    ┌────────────────────────┐
                    │      PostgreSQL        │
                    │   (replay buffer)      │
                    └────────────┬───────────┘
                                 │ Sample batches
                                 ▼
                    ┌────────────────────────┐
                    │     trainer-k8s        │
                    │  (continuous training) │
                    └────────────┬───────────┘
                                 │ Export models
                                 ▼
                    ┌────────────────────────┐
                    │        MinIO           │
                    │   (model storage)      │
                    └────────────────────────┘
                                 │
                                 ▼
                    (actors hot-reload new models)
```

## Components

### 1. PostgreSQL (replay buffer)
- Stores all transitions from all actors
- Supports concurrent writes from multiple actors
- Trainer samples batches for training

### 2. MinIO (model storage)
- S3-compatible object storage
- Stores trained models (ONNX format)
- Actors watch for new models and hot-reload

### 3. actor-k8s (scalable)
- Runs self-play episodes continuously (`max_episodes = -1`)
- Stores transitions to PostgreSQL
- Hot-reloads models from MinIO when updated
- Horizontally scalable with `--scale actor-k8s=N`

### 4. trainer-k8s (continuous)
- Waits for sufficient data in PostgreSQL
- Trains for 1000 steps per cycle
- Exports model to MinIO
- Exits and Docker restarts it (continuous loop)

## Changes Made

### docker-compose.k8s.yml

1. **Moved `alphazero-k8s` to `orchestrator` profile**
   - Won't run with `--profile k8s`
   - Use `--profile orchestrator` if you want the old monolithic approach

2. **Added `CARTRIDGE_COMMON_ENV_ID=connect4` to:**
   - `actor-k8s`
   - `trainer-k8s`
   - `alphazero-k8s`

3. **Updated `trainer-k8s` entrypoint:**
   - Now runs `python -m trainer train --steps 1000`
   - With `restart: unless-stopped`, creates continuous training loop

4. **Updated MinIO credentials:**
   - Username: `aspect`
   - Password: `password123`

### actor/src/storage/ (new module)

1. **PostgreSQL backend support**
   - `storage/postgres.rs` - PostgreSQL implementation
   - `storage/sqlite.rs` - SQLite implementation
   - `storage/mod.rs` - `ReplayStore` trait and factory

2. **Actor migrated to use storage module**
   - `Actor::new()` now async
   - Uses `create_replay_store()` based on config
   - Reads `CARTRIDGE_STORAGE_*` environment variables

### actor/src/config.rs

1. **Added storage config fields:**
   - `replay_backend` - "sqlite" or "postgres"
   - `postgres_url` - PostgreSQL connection string

2. **Added `with_defaults()` method:**
   - Fills `postgres_url` from central config if not provided via CLI

## Remaining Tasks

### Build & Deploy

1. **Rebuild Docker images:**
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.k8s.yml build actor-k8s trainer-k8s
   ```

2. **Start the stack:**
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.k8s.yml down -v
   docker compose -f docker-compose.yml -f docker-compose.k8s.yml --profile k8s up --scale actor-k8s=4
   ```

### Verification

1. **Check actors are storing to PostgreSQL:**
   ```bash
   docker exec -it cartridge-postgres psql -U cartridge -c "SELECT COUNT(*) FROM transitions;"
   ```

2. **Check trainer is training:**
   - Look for training logs in `cartridge-trainer-k8s`
   - Check for model exports to MinIO

3. **Check model hot-reload:**
   - Actors should log "Model updated" when new model appears

## Usage

### Start continuous training (recommended)
```bash
docker compose -f docker-compose.yml -f docker-compose.k8s.yml --profile k8s up --scale actor-k8s=4
```

### Scale actors up/down
```bash
docker compose -f docker-compose.yml -f docker-compose.k8s.yml --profile k8s up --scale actor-k8s=8
```

### View MinIO console
- URL: http://localhost:9001
- Username: aspect
- Password: password123

### View PostgreSQL
```bash
docker exec -it cartridge-postgres psql -U cartridge -d cartridge
```

## Configuration

All settings in `config.toml`:

```toml
[common]
env_id = "connect4"  # Game to train

[actor]
max_episodes = -1    # Run forever

[training]
steps_per_iteration = 1000  # Steps per training cycle

[mcts]
num_simulations = 200
temp_threshold = 15  # Use lower temp after move 15
```

Override via environment variables:
- `CARTRIDGE_COMMON_ENV_ID`
- `CARTRIDGE_STORAGE_REPLAY_BACKEND`
- `CARTRIDGE_STORAGE_POSTGRES_URL`
