//! Cartridge2 Web Server
//!
//! Minimal HTTP server exposing game API for the Svelte frontend.
//! Endpoints:
//! - GET  /health        - Health check
//! - GET  /games         - List available games
//! - GET  /game-info/:id  - Get metadata for a specific game
//! - POST /move          - Make a move (player action + bot response)
//! - GET  /stats         - Read training stats from data/stats.json
//! - GET  /model         - Get info about currently loaded model
//! - POST /selfplay      - Start/stop self-play actor (placeholder)
//! - POST /game/new      - Start a new game
//! - GET  /game/state    - Get current game state

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use engine_core::{create_game, list_registered_games, GameMetadata};
#[cfg(feature = "onnx")]
use mcts::OnnxEvaluator;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
#[cfg(not(feature = "onnx"))]
use tracing::info;
#[cfg(feature = "onnx")]
use tracing::{info, warn};

mod game;
#[cfg(feature = "onnx")]
mod model_watcher;

use game::GameSession;
#[cfg(feature = "onnx")]
use model_watcher::{ModelInfo, ModelWatcher};

/// Stub evaluator type when ONNX is disabled (for testing)
#[cfg(not(feature = "onnx"))]
pub type OnnxEvaluator = ();

/// Stub ModelInfo when ONNX is disabled
#[cfg(not(feature = "onnx"))]
#[derive(Default, Clone)]
pub struct ModelInfo {
    pub loaded: bool,
    pub path: Option<String>,
    pub file_modified: Option<u64>,
    pub loaded_at: Option<u64>,
    pub training_step: Option<u32>,
}

/// Shared application state
pub struct AppState {
    /// Current game session
    session: Mutex<GameSession>,
    /// Current game ID (for creating new sessions)
    current_game: RwLock<String>,
    /// Data directory for stats.json
    data_dir: String,
    /// Shared evaluator for MCTS (hot-reloaded)
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// Model info (updated on reload)
    model_info: Arc<RwLock<ModelInfo>>,
}

/// Create the application router with the given state.
/// This is separated out for testing purposes.
pub fn create_app(state: Arc<AppState>) -> Router {
    // CORS layer for development
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health))
        .route("/games", get(list_games))
        .route("/game-info/:id", get(get_game_info))
        .route("/move", post(make_move))
        .route("/stats", get(get_stats))
        .route("/model", get(get_model_info))
        .route("/selfplay", post(selfplay))
        .route("/game/new", post(new_game))
        .route("/game/state", get(get_game_state))
        .layer(cors)
        .with_state(state)
}

/// Create application state for testing (no model watcher, no logging)
#[cfg(test)]
pub fn create_test_state() -> Arc<AppState> {
    games_tictactoe::register_tictactoe();
    games_connect4::register_connect4();
    let evaluator: Arc<RwLock<Option<OnnxEvaluator>>> = Arc::new(RwLock::new(None));
    let model_info = Arc::new(RwLock::new(ModelInfo::default()));
    let session = GameSession::with_evaluator("tictactoe", Arc::clone(&evaluator))
        .expect("Failed to create game session");

    Arc::new(AppState {
        session: Mutex::new(session),
        current_game: RwLock::new("tictactoe".to_string()),
        data_dir: "./test_data".to_string(),
        evaluator,
        model_info,
    })
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
    games_connect4::register_connect4();
    info!("Registered tictactoe and connect4 games");

    // Set up shared evaluator for model hot-reloading
    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string());
    let evaluator: Arc<RwLock<Option<OnnxEvaluator>>> = Arc::new(RwLock::new(None));

    #[cfg(feature = "onnx")]
    let model_info = {
        use engine_core::EngineContext;
        use tracing::warn;
        let model_dir = format!("{}/models", data_dir);

        // Get obs_size from the default game's metadata
        // TODO: Support switching games dynamically (would need to reload model watcher)
        let default_game =
            std::env::var("DEFAULT_GAME").unwrap_or_else(|_| "tictactoe".to_string());
        let obs_size = EngineContext::new(&default_game)
            .map(|ctx| ctx.metadata().obs_size)
            .unwrap_or(29); // Fallback to TicTacToe's obs_size if game not found
        info!(
            "Model watcher using obs_size={} from game '{}'",
            obs_size, default_game
        );

        // Create model watcher
        let model_watcher =
            ModelWatcher::new(&model_dir, "latest.onnx", obs_size, Arc::clone(&evaluator));

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

        model_info
    };

    #[cfg(not(feature = "onnx"))]
    let model_info = Arc::new(RwLock::new(ModelInfo::default()));

    // Create initial game session with shared evaluator
    let session = GameSession::with_evaluator("tictactoe", Arc::clone(&evaluator))?;

    let state = Arc::new(AppState {
        session: Mutex::new(session),
        current_game: RwLock::new("tictactoe".to_string()),
        data_dir,
        evaluator,
        model_info,
    });

    // Build router
    let app = create_app(state);

    let addr = "0.0.0.0:8080";
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// Health Check
// ============================================================================

#[derive(Serialize, Deserialize)]
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
// Games List and Info
// ============================================================================

#[derive(Serialize, Deserialize)]
struct GamesListResponse {
    games: Vec<String>,
}

async fn list_games() -> Json<GamesListResponse> {
    Json(GamesListResponse {
        games: list_registered_games(),
    })
}

#[derive(Serialize, Deserialize)]
struct GameInfoResponse {
    env_id: String,
    display_name: String,
    board_width: usize,
    board_height: usize,
    num_actions: usize,
    obs_size: usize,
    legal_mask_offset: usize,
    player_count: usize,
    player_names: Vec<String>,
    player_symbols: Vec<char>,
    description: String,
    board_type: String,
}

impl From<GameMetadata> for GameInfoResponse {
    fn from(meta: GameMetadata) -> Self {
        Self {
            env_id: meta.env_id,
            display_name: meta.display_name,
            board_width: meta.board_width,
            board_height: meta.board_height,
            num_actions: meta.num_actions,
            obs_size: meta.obs_size,
            legal_mask_offset: meta.legal_mask_offset,
            player_count: meta.player_count,
            player_names: meta.player_names,
            player_symbols: meta.player_symbols,
            description: meta.description,
            board_type: meta.board_type,
        }
    }
}

async fn get_game_info(
    Path(id): Path<String>,
) -> Result<Json<GameInfoResponse>, (StatusCode, String)> {
    let game = create_game(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!(
                "Game not found: {}. Use /games to list available games.",
                id
            ),
        )
    })?;

    let metadata = game.metadata();
    Ok(Json(metadata.into()))
}

// ============================================================================
// Game State
// ============================================================================

#[derive(Serialize, Deserialize)]
struct GameStateResponse {
    /// Board cells: 0=empty, 1=X (player), 2=O (bot)
    /// Length depends on game (e.g., 9 for TicTacToe, 42 for Connect 4)
    board: Vec<u8>,
    /// Current player: 1=X, 2=O
    current_player: u8,
    /// Which player the human is: 1 or 2 (depends on who went first)
    human_player: u8,
    /// Winner: 0=ongoing, 1=X wins, 2=O wins, 3=draw
    winner: u8,
    /// Is the game over?
    game_over: bool,
    /// Legal moves (positions)
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
    /// Game to play (e.g., "tictactoe", "connect4")
    #[serde(default)]
    game: Option<String>,
}

fn default_first() -> String {
    "player".to_string()
}

async fn new_game(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NewGameRequest>,
) -> Result<Json<GameStateResponse>, (StatusCode, String)> {
    let mut session = state.session.lock().await;

    // Determine which game to use
    let game_id = if let Some(ref game) = req.game {
        // Update current game if specified
        if let Ok(mut current) = state.current_game.write() {
            *current = game.clone();
        }
        game.clone()
    } else {
        // Use current game
        state
            .current_game
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "tictactoe".to_string())
    };

    // Reset the game with shared evaluator (for hot-reloading)
    *session =
        GameSession::with_evaluator(&game_id, Arc::clone(&state.evaluator)).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create game '{}': {}", game_id, e),
            )
        })?;

    // If bot goes first, bot is player 1, human is player 2
    // If player goes first, human is player 1, bot is player 2
    if req.first == "bot" {
        session.set_human_player(2); // Human plays as O (player 2)
        session.bot_move().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Bot move failed: {}", e),
            )
        })?;
    } else {
        session.set_human_player(1); // Human plays as X (player 1) - default
    }

    Ok(Json(session.to_response()))
}

// ============================================================================
// Make Move
// ============================================================================

#[derive(Deserialize)]
struct MoveRequest {
    /// Position or column to play (game-specific: 0-8 for TicTacToe, 0-6 for Connect4)
    position: u8,
}

#[derive(Serialize, Deserialize)]
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

    // Check if game is over
    if session.is_game_over() {
        return Err((StatusCode::BAD_REQUEST, "Game is already over".to_string()));
    }

    // Check if it's the human's turn
    if !session.is_human_turn() {
        return Err((StatusCode::BAD_REQUEST, "Not your turn".to_string()));
    }

    // Check if move is legal (this handles position validation based on game type)
    if !session.is_legal_move(req.position) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Illegal move: position/column {} is not valid",
                req.position
            ),
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

/// Evaluation stats from a single evaluation run
#[derive(Deserialize, Serialize, Clone, Default)]
struct EvalStats {
    #[serde(default)]
    step: u32,
    #[serde(default)]
    win_rate: f64,
    #[serde(default)]
    draw_rate: f64,
    #[serde(default)]
    loss_rate: f64,
    #[serde(default)]
    games_played: u32,
    #[serde(default)]
    avg_game_length: f64,
    #[serde(default)]
    timestamp: f64,
}

/// Stats format written by Python trainer
#[derive(Deserialize, Default)]
struct TrainerStats {
    #[serde(default)]
    step: u32,
    #[serde(default)]
    total_steps: u32,
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
    #[serde(default)]
    last_eval: Option<EvalStats>,
    #[serde(default)]
    eval_history: Vec<EvalStats>,
}

/// Stats format sent to frontend
#[derive(Serialize, Default)]
struct TrainingStats {
    epoch: u32,
    step: u32,
    total_steps: u32,
    loss: f64,
    total_loss: f64,
    policy_loss: f64,
    value_loss: f64,
    games_played: u64,
    replay_buffer_size: u64,
    learning_rate: f64,
    timestamp: f64,
    last_eval: Option<EvalStats>,
    eval_history: Vec<EvalStats>,
}

impl From<TrainerStats> for TrainingStats {
    fn from(t: TrainerStats) -> Self {
        Self {
            epoch: t.step,
            step: t.step,
            total_steps: t.total_steps,
            loss: t.total_loss,
            total_loss: t.total_loss,
            policy_loss: t.policy_loss,
            value_loss: t.value_loss,
            games_played: t.replay_buffer_size,
            replay_buffer_size: t.replay_buffer_size,
            learning_rate: t.learning_rate,
            timestamp: t.timestamp,
            last_eval: t.last_eval,
            eval_history: t.eval_history,
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

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// Helper to make a GET request and return response body as string
    async fn get(app: Router, uri: &str) -> (StatusCode, String) {
        let response = app
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        (status, body_str)
    }

    /// Helper to make a POST request with JSON body and return response
    async fn post_json(app: Router, uri: &str, json: &str) -> (StatusCode, String) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(json.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        (status, body_str)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = create_test_state();
        let app = create_app(state);

        let (status, body) = get(app, "/health").await;

        assert_eq!(status, StatusCode::OK);
        let response: HealthResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(response.status, "ok");
    }

    #[tokio::test]
    async fn test_game_state_returns_initial_board() {
        let state = create_test_state();
        let app = create_app(state);

        let (status, body) = get(app, "/game/state").await;

        assert_eq!(status, StatusCode::OK);
        let response: GameStateResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(
            response.board,
            vec![0u8; 9],
            "Initial board should be empty"
        );
        assert_eq!(response.current_player, 1, "Player X should go first");
        assert_eq!(response.winner, 0, "No winner yet");
        assert!(!response.game_over);
        assert_eq!(response.legal_moves.len(), 9, "All 9 moves should be legal");
    }

    #[tokio::test]
    async fn test_new_game_player_first() {
        let state = create_test_state();
        let app = create_app(state);

        let (status, body) = post_json(app, "/game/new", r#"{"first": "player"}"#).await;

        assert_eq!(status, StatusCode::OK);
        let response: GameStateResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(
            response.board,
            vec![0u8; 9],
            "Board should be empty when player goes first"
        );
        assert_eq!(response.current_player, 1);
        assert!(!response.game_over);
    }

    #[tokio::test]
    async fn test_new_game_bot_first() {
        let state = create_test_state();
        let app = create_app(state);

        let (status, body) = post_json(app, "/game/new", r#"{"first": "bot"}"#).await;

        assert_eq!(status, StatusCode::OK);
        let response: GameStateResponse = serde_json::from_str(&body).unwrap();
        // Bot should have made one move (one cell is non-zero)
        let moves_made: usize = response.board.iter().filter(|&&x| x != 0).count();
        assert_eq!(moves_made, 1, "Bot should have made exactly one move");
        // When bot goes "first", it plays as X (since X always starts in TicTacToe)
        // After bot's move as X, it's O's turn (current_player = 2)
        assert_eq!(
            response.current_player, 2,
            "Should be O's turn after bot (X) moves first"
        );
    }

    #[tokio::test]
    async fn test_new_game_default_player_first() {
        let state = create_test_state();
        let app = create_app(state);

        // Empty JSON should default to player first
        let (status, body) = post_json(app, "/game/new", r#"{}"#).await;

        assert_eq!(status, StatusCode::OK);
        let response: GameStateResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(
            response.board,
            vec![0u8; 9],
            "Board should be empty with default"
        );
    }

    #[tokio::test]
    async fn test_move_valid() {
        let state = create_test_state();
        let app = create_app(state);

        // Make a move at position 4 (center)
        let (status, body) = post_json(app, "/move", r#"{"position": 4}"#).await;

        assert_eq!(status, StatusCode::OK);
        let response: MoveResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(response.state.board[4], 1, "Player X should be at center");
        // Bot should have made a move (unless game over)
        if !response.state.game_over {
            assert!(response.bot_move.is_some(), "Bot should make a move");
            let bot_pos = response.bot_move.unwrap() as usize;
            assert_eq!(response.state.board[bot_pos], 2, "Bot should have placed O");
        }
    }

    #[tokio::test]
    async fn test_move_invalid_position() {
        let state = create_test_state();
        let app = create_app(state);

        // Position 9 is out of bounds for TicTacToe
        let (status, body) = post_json(app, "/move", r#"{"position": 9}"#).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Illegal move") || body.contains("not valid"));
    }

    #[tokio::test]
    async fn test_move_occupied_position() {
        let state = create_test_state();

        // First move at center
        {
            let app = create_app(Arc::clone(&state));
            let (status, _) = post_json(app, "/move", r#"{"position": 4}"#).await;
            assert_eq!(status, StatusCode::OK);
        }

        // Try to move at center again (occupied by player X)
        {
            let app = create_app(Arc::clone(&state));
            let (status, body) = post_json(app, "/move", r#"{"position": 4}"#).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(body.contains("Illegal move") || body.contains("occupied"));
        }
    }

    #[tokio::test]
    async fn test_game_flow_player_wins() {
        let state = create_test_state();

        // Start fresh game
        {
            let app = create_app(Arc::clone(&state));
            post_json(app, "/game/new", r#"{"first": "player"}"#).await;
        }

        // Try to get player to win with top row (0, 1, 2)
        // This may not always work due to bot blocking, but tests the flow
        let moves = [0, 1, 2]; // Try top row
        let mut player_positions = vec![];

        for pos in moves {
            let app = create_app(Arc::clone(&state));
            let (status, body) =
                post_json(app, "/move", &format!(r#"{{"position": {}}}"#, pos)).await;

            if status == StatusCode::BAD_REQUEST {
                // Position might be taken by bot, skip
                continue;
            }
            assert_eq!(status, StatusCode::OK);

            let response: MoveResponse = serde_json::from_str(&body).unwrap();
            player_positions.push(pos);

            if response.state.game_over {
                // Game ended - verify state is consistent
                assert!(response.state.winner != 0 || response.state.legal_moves.is_empty());
                break;
            }
        }
    }

    #[tokio::test]
    async fn test_move_when_game_over() {
        let state = create_test_state();

        // Play a complete game by making moves until done
        let mut game_over = false;
        let mut position = 0u8;

        while !game_over && position < 9 {
            let app = create_app(Arc::clone(&state));
            let (status, body) =
                post_json(app, "/move", &format!(r#"{{"position": {}}}"#, position)).await;

            if status == StatusCode::OK {
                let response: MoveResponse = serde_json::from_str(&body).unwrap();
                game_over = response.state.game_over;
            }
            position += 1;
        }

        // If game is over, next move should fail
        if game_over {
            // Find an empty position (if any) and try to move
            let app = create_app(Arc::clone(&state));
            let (status, body) = post_json(app, "/move", r#"{"position": 0}"#).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(body.contains("Game is already over") || body.contains("Illegal move"));
        }
    }

    #[tokio::test]
    async fn test_state_updates_after_move() {
        let state = create_test_state();

        // Get initial state
        let (_, initial_body) = {
            let app = create_app(Arc::clone(&state));
            get(app, "/game/state").await
        };
        let initial: GameStateResponse = serde_json::from_str(&initial_body).unwrap();
        assert_eq!(initial.board, vec![0u8; 9]);

        // Make a move
        {
            let app = create_app(Arc::clone(&state));
            let (status, _) = post_json(app, "/move", r#"{"position": 0}"#).await;
            assert_eq!(status, StatusCode::OK);
        }

        // Get updated state
        let (_, updated_body) = {
            let app = create_app(Arc::clone(&state));
            get(app, "/game/state").await
        };
        let updated: GameStateResponse = serde_json::from_str(&updated_body).unwrap();

        // Verify board changed
        assert_ne!(updated.board, vec![0u8; 9], "Board should have changed");
        assert_eq!(updated.board[0], 1, "Player X should be at position 0");
    }

    #[tokio::test]
    async fn test_new_game_resets_state() {
        let state = create_test_state();

        // Make some moves
        {
            let app = create_app(Arc::clone(&state));
            post_json(app, "/move", r#"{"position": 4}"#).await;
        }

        // Verify board is not empty
        let (_, body) = {
            let app = create_app(Arc::clone(&state));
            get(app, "/game/state").await
        };
        let mid_game: GameStateResponse = serde_json::from_str(&body).unwrap();
        assert_ne!(mid_game.board, vec![0u8; 9], "Board should have moves");

        // Start new game
        {
            let app = create_app(Arc::clone(&state));
            post_json(app, "/game/new", r#"{"first": "player"}"#).await;
        }

        // Verify board is reset
        let (_, body) = {
            let app = create_app(Arc::clone(&state));
            get(app, "/game/state").await
        };
        let new_game: GameStateResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(new_game.board, vec![0u8; 9], "Board should be reset");
        assert_eq!(new_game.current_player, 1);
        assert_eq!(new_game.winner, 0);
    }

    #[tokio::test]
    async fn test_legal_moves_update() {
        let state = create_test_state();

        // Initial state has 9 legal moves
        let (_, body) = {
            let app = create_app(Arc::clone(&state));
            get(app, "/game/state").await
        };
        let initial: GameStateResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(initial.legal_moves.len(), 9);

        // Make a move
        {
            let app = create_app(Arc::clone(&state));
            post_json(app, "/move", r#"{"position": 0}"#).await;
        }

        // Should have fewer legal moves (player + bot each made one)
        let (_, body) = {
            let app = create_app(Arc::clone(&state));
            get(app, "/game/state").await
        };
        let after_move: GameStateResponse = serde_json::from_str(&body).unwrap();
        assert!(
            after_move.legal_moves.len() < 9,
            "Should have fewer legal moves"
        );
        assert!(
            !after_move.legal_moves.contains(&0),
            "Position 0 should not be legal"
        );
    }

    #[tokio::test]
    async fn test_list_games() {
        let state = create_test_state();
        let app = create_app(state);

        let (status, body) = get(app, "/games").await;

        assert_eq!(status, StatusCode::OK);
        let response: GamesListResponse = serde_json::from_str(&body).unwrap();
        assert!(response.games.contains(&"tictactoe".to_string()));
    }

    #[tokio::test]
    async fn test_get_game_info_tictactoe() {
        let state = create_test_state();
        let app = create_app(state);

        let (status, body) = get(app, "/game-info/tictactoe").await;

        assert_eq!(status, StatusCode::OK);
        let response: GameInfoResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(response.env_id, "tictactoe");
        assert_eq!(response.display_name, "Tic-Tac-Toe");
        assert_eq!(response.board_width, 3);
        assert_eq!(response.board_height, 3);
        assert_eq!(response.num_actions, 9);
        assert_eq!(response.player_count, 2);
    }

    #[tokio::test]
    async fn test_get_game_info_not_found() {
        let state = create_test_state();
        let app = create_app(state);

        let (status, body) = get(app, "/game-info/nonexistent").await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("Game not found"));
    }
}
