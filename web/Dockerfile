# Web Backend Service Dockerfile
# Multi-stage build for Rust backend with ONNX support
#
# Build args:
#   CARGO_FEATURES - Comma-separated cargo features (e.g., "s3")

# ============================================
# Stage 1: Build Backend (Rust)
# ============================================
FROM rust:1.88-bookworm AS builder

# Optional features: s3 (S3 model storage)
ARG CARGO_FEATURES=""

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire workspace (engine + web)
COPY engine/ ./engine/
COPY web/ ./web/

# Build the web binary in release mode with optional features
WORKDIR /app/web
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
COPY --from=builder /app/web/target/release/web /app/web

# Create data directory
RUN mkdir -p /app/data/models

# Environment variables with defaults
ENV DATA_DIR="/app/data"
ENV DEFAULT_GAME="tictactoe"
ENV RUST_LOG="info"

# Expose the web server port
EXPOSE 8080

# Run the web server
ENTRYPOINT ["/app/web"]
