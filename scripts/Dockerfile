# Synchronized AlphaZero Training Container
# Contains both the Rust actor and Python trainer for coordinated training
#
# Each iteration:
#   1. Clear replay buffer
#   2. Run actor for N episodes
#   3. Train for M steps
#   4. Repeat

# ============================================
# Stage 1: Build Rust Actor
# ============================================
FROM rust:1.85-bookworm AS rust-builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and build actor
COPY engine/ ./engine/
COPY actor/ ./actor/

WORKDIR /app/actor
RUN cargo build --release

# ============================================
# Stage 2: Python Runtime with Actor Binary
# ============================================
FROM python:3.11-slim-bookworm

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy actor binary from rust builder
COPY --from=rust-builder /app/actor/target/release/actor /app/actor

# Copy trainer source and install
COPY trainer/pyproject.toml trainer/README.md ./trainer/
COPY trainer/src/ ./trainer/src/

# Install CPU-only PyTorch (avoids ~700MB CUDA download)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install trainer package
RUN pip install --no-cache-dir ./trainer

# Copy the synchronized training script
COPY scripts/train_loop.py ./train_loop.py
COPY scripts/docker_train.sh ./docker_train.sh
RUN chmod +x ./docker_train.sh

# Create data directory
RUN mkdir -p /app/data/models

# Environment variables with defaults
ENV ALPHAZERO_ENV_ID="tictactoe"
ENV ALPHAZERO_ITERATIONS="100"
ENV ALPHAZERO_EPISODES="500"
ENV ALPHAZERO_STEPS="1000"
ENV ALPHAZERO_BATCH_SIZE="64"
ENV ALPHAZERO_LR="0.001"
ENV ALPHAZERO_DEVICE="cpu"
ENV ALPHAZERO_EVAL_INTERVAL="0"
ENV ALPHAZERO_EVAL_GAMES="50"
ENV ALPHAZERO_CHECKPOINT_INTERVAL="100"

# Data paths
ENV DATA_DIR="/app/data"
ENV REPLAY_DB_PATH="/app/data/replay.db"
ENV MODEL_DIR="/app/data/models"
ENV STATS_PATH="/app/data/stats.json"

ENTRYPOINT ["/app/docker_train.sh"]
