# Trainer Service Dockerfile
# Python-based AlphaZero trainer with PyTorch and ONNX

FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject.toml first for better caching
COPY pyproject.toml ./

# Install Python dependencies
# Using pip with --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -e .

# Copy the source code
COPY src/ ./src/

# Create data directory
RUN mkdir -p /app/data/models

# Environment variables with defaults
ENV TRAINER_DB_PATH="/app/data/replay.db"
ENV TRAINER_MODEL_DIR="/app/data/models"
ENV TRAINER_STATS_PATH="/app/data/stats.json"
ENV TRAINER_ENV_ID="tictactoe"
ENV TRAINER_DEVICE="cpu"
ENV TRAINER_LOG_LEVEL="INFO"
ENV TRAINER_STEPS="1000"
ENV TRAINER_BATCH_SIZE="64"
ENV TRAINER_LR="0.001"

# Default command runs the trainer
# Override with docker run ... evaluator to run evaluation instead
ENTRYPOINT ["python", "-m", "trainer"]
CMD ["--db", "/app/data/replay.db", \
     "--model-dir", "/app/data/models", \
     "--stats", "/app/data/stats.json"]
