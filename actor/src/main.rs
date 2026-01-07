//! Actor - Self-play episode runner for Cartridge2
//!
//! A long-running process that:
//! 1. Watches `./data/models/latest.onnx` for updates
//! 2. Runs MCTS self-play loops using the Engine library
//! 3. Saves completed games to `./data/replay.db` (SQLite)
//! 4. (Future) Exposes an HTTP server for the Web Frontend

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info};

mod actor;
mod central_config;
mod config;
mod game_config;
mod mcts_policy;
mod model_watcher;
mod storage;

use crate::actor::Actor;
use crate::config::Config;

fn init_tracing(level: &str) -> Result<()> {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level))
        .add_directive("ort=warn".parse().unwrap());

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse configuration
    let config = Config::parse();

    // Validate configuration
    config.validate()?;

    // Initialize tracing
    init_tracing(&config.log_level)?;
    info!(log_level = %config.log_level, "Actor service starting");

    // Log the max_episodes setting
    let max_episode_description = if config.max_episodes < 0 {
        "unlimited".to_string()
    } else {
        config.max_episodes.to_string()
    };
    info!(
        max_episodes = config.max_episodes,
        "Actor will run {} episodes", max_episode_description
    );

    info!(
        "Starting actor {} for environment {}",
        config.actor_id, config.env_id
    );

    // Create actor instance
    let actor = Actor::new(config).await?;
    let actor = Arc::new(actor);

    // Setup graceful shutdown
    let shutdown_actor = Arc::clone(&actor);
    let shutdown_handle = tokio::spawn(async move {
        if let Err(e) = signal::ctrl_c().await {
            error!("Failed to listen for ctrl+c signal: {}", e);
            return;
        }
        info!("Shutdown signal received, stopping actor...");
        shutdown_actor.shutdown();
    });

    // Run the actor
    let run_result = actor.run().await;

    // Wait for shutdown to complete
    shutdown_handle.abort();

    match run_result {
        Ok(_) => {
            info!("Actor completed successfully");
            Ok(())
        }
        Err(e) => {
            error!("Actor failed: {}", e);
            Err(e)
        }
    }
}
