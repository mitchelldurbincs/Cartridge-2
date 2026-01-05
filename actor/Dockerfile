# Actor Service Dockerfile
# Multi-stage build for Rust binary with ONNX support
#
# Build args:
#   CARGO_FEATURES - Comma-separated cargo features (e.g., "postgres,s3")

# ============================================
# Stage 1: Build
# ============================================
FROM rust:1.88-bookworm AS builder

# Optional features: postgres (PostgreSQL replay), s3 (S3 model storage)
ARG CARGO_FEATURES=""

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire workspace (engine + actor)
COPY engine/ ./engine/
COPY actor/ ./actor/

# Build the actor binary in release mode with optional features
WORKDIR /app/actor
RUN if [ -z "$CARGO_FEATURES" ]; then \
        cargo build --release; \
    else \
        cargo build --release --features "$CARGO_FEATURES"; \
    fi

# ============================================
# Stage 2: Runtime
# ============================================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary from builder
COPY --from=builder /app/actor/target/release/actor /app/actor

# Create data directory
RUN mkdir -p /app/data/models

# Environment variables with defaults
ENV ACTOR_ACTOR_ID="actor-1"
ENV ACTOR_ENV_ID="tictactoe"
ENV ACTOR_MAX_EPISODES="-1"
ENV ACTOR_EPISODE_TIMEOUT="30"
ENV ACTOR_FLUSH_INTERVAL="5"
ENV ACTOR_LOG_LEVEL="info"
ENV ACTOR_LOG_INTERVAL="10"
ENV ACTOR_REPLAY_DB_PATH="/app/data/replay.db"
ENV ACTOR_DATA_DIR="/app/data"

# Run the actor
ENTRYPOINT ["/app/actor"]
