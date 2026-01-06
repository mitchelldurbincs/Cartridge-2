# Trainer Service Dockerfile
# Python-based AlphaZero trainer with PyTorch and ONNX
#
# This image provides:
#   - trainer train     - Train on replay buffer data
#   - trainer evaluate  - Evaluate model against random baseline
#   - trainer loop      - Orchestrated training (requires ACTOR_BINARY)
#
# For synchronized training, use the root Dockerfile.alphazero instead,
# which bundles both the actor binary and trainer package.

FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash aspect

WORKDIR /app

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install CPU-only PyTorch first (avoids downloading ~700MB of CUDA libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install trainer package (includes trainer, trainer-loop, trainer-evaluate commands)
RUN pip install --no-cache-dir .

# Create data directory with proper ownership
RUN mkdir -p /app/data/models && chown -R aspect:aspect /app

# Switch to non-root user
USER aspect

# Environment variables with defaults
ENV DATA_DIR="/app/data"
ENV TRAINER_DB_PATH="/app/data/replay.db"
ENV TRAINER_MODEL_DIR="/app/data/models"
ENV TRAINER_STATS_PATH="/app/data/stats.json"
ENV TRAINER_ENV_ID="tictactoe"
ENV TRAINER_DEVICE="cpu"
ENV TRAINER_LOG_LEVEL="INFO"

# Default command runs the trainer
# Use 'trainer evaluate' or 'trainer loop' for other modes
ENTRYPOINT ["python", "-m", "trainer"]
CMD ["--help"]
