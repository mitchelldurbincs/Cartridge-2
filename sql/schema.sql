-- Cartridge2 PostgreSQL Schema
-- This schema is shared by both the Rust actor and Python trainer.
-- Any changes here must be compatible with both components.

-- Transitions table: stores game episode data for training
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

-- Indices for efficient querying
CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON transitions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transitions_episode ON transitions(episode_id);
CREATE INDEX IF NOT EXISTS idx_transitions_env_id ON transitions(env_id);

-- Game metadata table: stores configuration for each game type
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
