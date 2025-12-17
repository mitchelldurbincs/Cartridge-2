#!/bin/bash
# AlphaZero-style training loop for Docker
# Runs short training iterations continuously, producing models more frequently

set -e

# Configuration from environment (with defaults)
DB_PATH="${TRAINER_DB_PATH:-/app/data/replay.db}"
MODEL_DIR="${TRAINER_MODEL_DIR:-/app/data/models}"
STATS_PATH="${TRAINER_STATS_PATH:-/app/data/stats.json}"
ENV_ID="${TRAINER_ENV_ID:-tictactoe}"
DEVICE="${TRAINER_DEVICE:-cpu}"
STEPS_PER_ITERATION="${TRAINER_STEPS_PER_ITERATION:-50}"
BATCH_SIZE="${TRAINER_BATCH_SIZE:-32}"
LR="${TRAINER_LR:-0.001}"
CHECKPOINT_INTERVAL="${TRAINER_CHECKPOINT_INTERVAL:-25}"
STATS_INTERVAL="${TRAINER_STATS_INTERVAL:-10}"
WAIT_INTERVAL="${TRAINER_WAIT_INTERVAL:-2}"
MAX_WAIT="${TRAINER_MAX_WAIT:-30}"
SLEEP_INTERVAL="${TRAINER_SLEEP_INTERVAL:-5}"

echo "=== Cartridge2 Training Loop ==="
echo "Game: $ENV_ID"
echo "Steps per iteration: $STEPS_PER_ITERATION"
echo "Batch size: $BATCH_SIZE"
echo "================================"

iteration=0
global_step=0

while true; do
    iteration=$((iteration + 1))
    echo ""
    echo "=== Training Iteration $iteration (starting at step $global_step) ==="

    # Run a short training session with start-step offset
    python -m trainer \
        --db "$DB_PATH" \
        --model-dir "$MODEL_DIR" \
        --stats "$STATS_PATH" \
        --env-id "$ENV_ID" \
        --device "$DEVICE" \
        --steps "$STEPS_PER_ITERATION" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --stats-interval "$STATS_INTERVAL" \
        --wait-interval "$WAIT_INTERVAL" \
        --max-wait "$MAX_WAIT" \
        --start-step "$global_step" \
    || {
        echo "Training iteration $iteration failed or no data available, sleeping..."
        sleep "$SLEEP_INTERVAL"
        continue
    }

    # Update global step counter
    global_step=$((global_step + STEPS_PER_ITERATION))

    echo "=== Iteration $iteration complete (total steps: $global_step) ==="

    # Brief pause between iterations
    sleep 1
done
