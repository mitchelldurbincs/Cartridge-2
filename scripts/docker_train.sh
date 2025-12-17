#!/bin/bash
# Synchronized AlphaZero Training Loop for Docker
#
# Each iteration:
#   1. Clear replay buffer (train only on current model's data)
#   2. Run actor for N episodes
#   3. Train for M steps
#   4. Repeat

set -e

# Configuration from environment
ENV_ID="${ALPHAZERO_ENV_ID:-tictactoe}"
ITERATIONS="${ALPHAZERO_ITERATIONS:-100}"
EPISODES="${ALPHAZERO_EPISODES:-500}"
STEPS="${ALPHAZERO_STEPS:-1000}"
BATCH_SIZE="${ALPHAZERO_BATCH_SIZE:-64}"
LR="${ALPHAZERO_LR:-0.001}"
DEVICE="${ALPHAZERO_DEVICE:-cpu}"
EVAL_INTERVAL="${ALPHAZERO_EVAL_INTERVAL:-0}"
EVAL_GAMES="${ALPHAZERO_EVAL_GAMES:-50}"
CHECKPOINT_INTERVAL="${ALPHAZERO_CHECKPOINT_INTERVAL:-100}"
START_ITERATION="${ALPHAZERO_START_ITERATION:-1}"

# Paths
DATA_DIR="${DATA_DIR:-/app/data}"
REPLAY_DB="${REPLAY_DB_PATH:-$DATA_DIR/replay.db}"
MODEL_DIR="${MODEL_DIR:-$DATA_DIR/models}"
STATS_PATH="${STATS_PATH:-$DATA_DIR/stats.json}"

echo "========================================"
echo "  Synchronized AlphaZero Training"
echo "========================================"
echo "Game:        $ENV_ID"
echo "Iterations:  $ITERATIONS"
echo "Episodes:    $EPISODES per iteration"
echo "Steps:       $STEPS per iteration"
echo "Batch size:  $BATCH_SIZE"
echo "Device:      $DEVICE"
echo "========================================"
echo ""

# Ensure directories exist
mkdir -p "$MODEL_DIR"

# Function to clear the replay buffer
clear_buffer() {
    if [ -f "$REPLAY_DB" ]; then
        python3 -c "
import sqlite3
conn = sqlite3.connect('$REPLAY_DB')
cursor = conn.execute('DELETE FROM transitions')
conn.commit()
deleted = cursor.rowcount
print(f'Cleared {deleted} transitions from buffer')
conn.close()
"
    else
        echo "No replay database yet, will be created by actor"
    fi
}

# Function to get transition count
get_transition_count() {
    if [ -f "$REPLAY_DB" ]; then
        python3 -c "
import sqlite3
conn = sqlite3.connect('$REPLAY_DB')
cursor = conn.execute('SELECT COUNT(*) FROM transitions')
print(cursor.fetchone()[0])
conn.close()
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Main training loop
for iteration in $(seq "$START_ITERATION" $((START_ITERATION + ITERATIONS - 1))); do
    echo ""
    echo "============================================================"
    echo "ITERATION $iteration / $((START_ITERATION + ITERATIONS - 1))"
    echo "============================================================"

    # Step 1: Clear replay buffer
    echo ""
    echo "[Step 1] Clearing replay buffer..."
    clear_buffer

    # Step 2: Run actor for N episodes
    echo ""
    echo "[Step 2] Running actor for $EPISODES episodes..."
    /app/actor \
        --env-id "$ENV_ID" \
        --max-episodes "$EPISODES" \
        --replay-db-path "$REPLAY_DB" \
        --data-dir "$DATA_DIR" \
        --log-interval 50 \
        --log-level info

    transitions=$(get_transition_count)
    echo "Actor generated $transitions transitions"

    # Step 3: Train for M steps
    echo ""
    echo "[Step 3] Training for $STEPS steps..."

    # Calculate start step for checkpoint naming continuity
    start_step=$(( (iteration - 1) * STEPS ))

    python3 -m trainer \
        --db "$REPLAY_DB" \
        --model-dir "$MODEL_DIR" \
        --stats "$STATS_PATH" \
        --env-id "$ENV_ID" \
        --device "$DEVICE" \
        --steps "$STEPS" \
        --start-step "$start_step" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --max-wait 60 \
        --eval-interval "$EVAL_INTERVAL" \
        --eval-games "$EVAL_GAMES" \
        --log-level INFO

    echo ""
    echo "Iteration $iteration complete!"
done

echo ""
echo "========================================"
echo "  Training Complete!"
echo "========================================"
echo "Completed $ITERATIONS iterations"
echo "Models saved to: $MODEL_DIR"
