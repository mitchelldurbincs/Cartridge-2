//! Cartridge2 Web Server
//!
//! Minimal HTTP server exposing game API for the Svelte frontend.
//! Endpoints:
//! - GET  /health     - Health check
//! - POST /move       - Make a move (player action + bot response)
//! - GET  /stats      - Read training stats from data/stats.json
//! - GET  /model      - Get info about currently loaded model
//! - POST /selfplay   - Start/stop self-play actor (placeholder)
//! - POST /game/new   - Start a new game
//! - GET  /game/state - Get current game state

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use mcts::OnnxEvaluator;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};

mod game;
mod model_watcher;

use game::GameSession;
use model_watcher::{ModelInfo, ModelWatcher};

/// TicTacToe constants
const TICTACTOE_OBS_SIZE: usize = 29;

/// Shared application state
struct AppState {
    /// Current game session
    session: Mutex<GameSession>,
    /// Data directory for stats.json
    data_dir: String,
    /// Shared evaluator for MCTS (hot-reloaded)
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// Model info (updated on reload)
    model_info: Arc<RwLock<ModelInfo>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("web=info".parse().unwrap()),
        )
        .init();

    // Register games
    games_tictactoe::register_tictactoe();
    info!("Registered tictactoe game");

    // Set up shared evaluator for model hot-reloading
    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string());
    let model_dir = format!("{}/models", data_dir);
    let evaluator: Arc<RwLock<Option<OnnxEvaluator>>> = Arc::new(RwLock::new(None));

    // Create model watcher
    let model_watcher = ModelWatcher::new(
        &model_dir,
        "latest.onnx",
        TICTACTOE_OBS_SIZE,
        Arc::clone(&evaluator),
    );

    // Try to load existing model
    match model_watcher.try_load_existing() {
        Ok(true) => info!("Loaded existing model from {}/latest.onnx", model_dir),
        Ok(false) => info!(
            "No model found at {}/latest.onnx - bot will play randomly",
            model_dir
        ),
        Err(e) => warn!(
            "Failed to load existing model: {} - bot will play randomly",
            e
        ),
    }

    // Get model info reference before moving watcher
    let model_info = model_watcher.model_info();

    // Start watching for model updates
    let mut model_rx = model_watcher.start_watching().await?;

    // Spawn task to log model updates
    tokio::spawn(async move {
        while model_rx.recv().await.is_some() {
            info!("Model updated - bot will use new model for future games");
        }
    });

    // Create initial game session with shared evaluator
    let session = GameSession::with_evaluator("tictactoe", Arc::clone(&evaluator))?;

    let state = Arc::new(AppState {
        session: Mutex::new(session),
        data_dir,
        evaluator,
        model_info,
    });

    // CORS layer for development
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/move", post(make_move))
        .route("/stats", get(get_stats))
        .route("/model", get(get_model_info))
        .route("/selfplay", post(selfplay))
        .route("/game/new", post(new_game))
        .route("/game/state", get(get_game_state))
        .layer(cors)
        .with_state(state);

    let addr = "0.0.0.0:8080";
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// Health Check
// ============================================================================

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

// ============================================================================
// Game State
// ============================================================================

#[derive(Serialize)]
struct GameStateResponse {
    /// Board as 9 cells: 0=empty, 1=X (player), 2=O (bot)
    board: [u8; 9],
    /// Current player: 1=X, 2=O
    current_player: u8,
    /// Winner: 0=ongoing, 1=X wins, 2=O wins, 3=draw
    winner: u8,
    /// Is the game over?
    game_over: bool,
    /// Legal moves (positions 0-8)
    legal_moves: Vec<u8>,
    /// Status message
    message: String,
}

async fn get_game_state(
    State(state): State<Arc<AppState>>,
) -> Result<Json<GameStateResponse>, (StatusCode, String)> {
    let session = state.session.lock().await;
    Ok(Json(session.to_response()))
}

// ============================================================================
// New Game
// ============================================================================

#[derive(Deserialize)]
struct NewGameRequest {
    /// Who plays first: "player" or "bot"
    #[serde(default = "default_first")]
    first: String,
}

fn default_first() -> String {
    "player".to_string()
}

async fn new_game(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NewGameRequest>,
) -> Result<Json<GameStateResponse>, (StatusCode, String)> {
    let mut session = state.session.lock().await;

    // Reset the game with shared evaluator (for hot-reloading)
    *session =
        GameSession::with_evaluator("tictactoe", Arc::clone(&state.evaluator)).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create game: {}", e),
            )
        })?;

    // If bot goes first, make a move
    if req.first == "bot" {
        session.bot_move().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Bot move failed: {}", e),
            )
        })?;
    }

    Ok(Json(session.to_response()))
}

// ============================================================================
// Make Move
// ============================================================================

#[derive(Deserialize)]
struct MoveRequest {
    /// Position to place (0-8)
    position: u8,
}

#[derive(Serialize)]
struct MoveResponse {
    /// Updated game state
    #[serde(flatten)]
    state: GameStateResponse,
    /// Bot's move position (if bot moved)
    bot_move: Option<u8>,
}

async fn make_move(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MoveRequest>,
) -> Result<Json<MoveResponse>, (StatusCode, String)> {
    let mut session = state.session.lock().await;

    // Validate position
    if req.position >= 9 {
        return Err((StatusCode::BAD_REQUEST, "Position must be 0-8".to_string()));
    }

    // Check if game is over
    if session.is_game_over() {
        return Err((StatusCode::BAD_REQUEST, "Game is already over".to_string()));
    }

    // Check if it's player's turn (player is always X = 1)
    if session.current_player() != 1 {
        return Err((StatusCode::BAD_REQUEST, "Not your turn".to_string()));
    }

    // Check if move is legal
    if !session.is_legal_move(req.position) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Illegal move: position {} is occupied", req.position),
        ));
    }

    // Make player's move
    session.player_move(req.position).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Move failed: {}", e),
        )
    })?;

    // If game is not over, bot makes a move
    let bot_move = if !session.is_game_over() {
        let pos = session.bot_move().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Bot move failed: {}", e),
            )
        })?;
        Some(pos)
    } else {
        None
    };

    Ok(Json(MoveResponse {
        state: session.to_response(),
        bot_move,
    }))
}

// ============================================================================
// Stats
// ============================================================================

/// Stats format written by Python trainer
#[derive(Deserialize, Default)]
struct TrainerStats {
    #[serde(default)]
    step: u32,
    #[serde(default)]
    total_loss: f64,
    #[serde(default)]
    policy_loss: f64,
    #[serde(default)]
    value_loss: f64,
    #[serde(default)]
    replay_buffer_size: u64,
    #[serde(default)]
    learning_rate: f64,
    #[serde(default)]
    timestamp: f64,
}

/// Stats format sent to frontend
#[derive(Serialize, Default)]
struct TrainingStats {
    epoch: u32,
    loss: f64,
    policy_loss: f64,
    value_loss: f64,
    games_played: u64,
    learning_rate: f64,
    timestamp: f64,
}

impl From<TrainerStats> for TrainingStats {
    fn from(t: TrainerStats) -> Self {
        Self {
            epoch: t.step,
            loss: t.total_loss,
            policy_loss: t.policy_loss,
            value_loss: t.value_loss,
            games_played: t.replay_buffer_size,
            learning_rate: t.learning_rate,
            timestamp: t.timestamp,
        }
    }
}

async fn get_stats(State(state): State<Arc<AppState>>) -> Json<TrainingStats> {
    let stats_path = format!("{}/stats.json", state.data_dir);

    match tokio::fs::read_to_string(&stats_path).await {
        Ok(content) => match serde_json::from_str::<TrainerStats>(&content) {
            Ok(trainer_stats) => Json(trainer_stats.into()),
            Err(e) => {
                tracing::warn!("Failed to parse stats.json: {}", e);
                Json(TrainingStats::default())
            }
        },
        Err(_) => {
            // Return empty stats if file doesn't exist
            Json(TrainingStats::default())
        }
    }
}

// ============================================================================
// Model Info
// ============================================================================

#[derive(Serialize)]
struct ModelInfoResponse {
    /// Whether a model is currently loaded
    loaded: bool,
    /// Path to the loaded model file
    path: Option<String>,
    /// When the model file was last modified (Unix timestamp)
    file_modified: Option<u64>,
    /// When the model was loaded into memory (Unix timestamp)
    loaded_at: Option<u64>,
    /// Training step from filename (if parseable)
    training_step: Option<u32>,
    /// Human-readable status message
    status: String,
}

async fn get_model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfoResponse> {
    let info = state
        .model_info
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default();

    let status = if info.loaded {
        match info.training_step {
            Some(step) => format!("Model loaded (step {})", step),
            None => "Model loaded".to_string(),
        }
    } else {
        "No model loaded - bot plays randomly".to_string()
    };

    Json(ModelInfoResponse {
        loaded: info.loaded,
        path: info.path,
        file_modified: info.file_modified,
        loaded_at: info.loaded_at,
        training_step: info.training_step,
        status,
    })
}

// ============================================================================
// Self-Play Control (Placeholder)
// ============================================================================

#[derive(Deserialize)]
struct SelfPlayRequest {
    action: String, // "start" or "stop"
}

#[derive(Serialize)]
struct SelfPlayResponse {
    status: String,
    message: String,
}

async fn selfplay(Json(req): Json<SelfPlayRequest>) -> Json<SelfPlayResponse> {
    // Placeholder - actual implementation would control the actor process
    let message = match req.action.as_str() {
        "start" => "Self-play start requested (not implemented yet)",
        "stop" => "Self-play stop requested (not implemented yet)",
        _ => "Unknown action",
    };

    Json(SelfPlayResponse {
        status: "placeholder".to_string(),
        message: message.to_string(),
    })
}
