-- Cartridge2 PostgreSQL Schema
-- Replay buffer storage for Kubernetes deployments

-- Transitions table (main replay buffer)
CREATE TABLE IF NOT EXISTS transitions (
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

-- Index for efficient sampling by timestamp
CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON transitions(timestamp);

-- Index for episode queries (for outcome backfill)
CREATE INDEX IF NOT EXISTS idx_transitions_episode ON transitions(episode_id);

-- Index for env_id filtering
CREATE INDEX IF NOT EXISTS idx_transitions_env ON transitions(env_id);

-- Game metadata table (makes database self-describing)
CREATE TABLE IF NOT EXISTS game_metadata (
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

-- Training stats table (for distributed training coordination)
CREATE TABLE IF NOT EXISTS training_stats (
    id SERIAL PRIMARY KEY,
    env_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    policy_loss REAL,
    value_loss REAL,
    total_loss REAL,
    learning_rate REAL,
    episodes_in_buffer INTEGER,
    transitions_trained INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for stats queries
CREATE INDEX IF NOT EXISTS idx_training_stats_env ON training_stats(env_id, iteration);

-- Model versions table (for model storage coordination)
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    env_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    s3_key TEXT NOT NULL,
    is_latest BOOLEAN DEFAULT FALSE,
    is_best BOOLEAN DEFAULT FALSE,
    eval_win_rate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(env_id, version)
);

-- Index for latest model lookup
CREATE INDEX IF NOT EXISTS idx_model_versions_latest ON model_versions(env_id, is_latest) WHERE is_latest = TRUE;

-- Grant permissions (for K8s service accounts)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cartridge;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cartridge;
