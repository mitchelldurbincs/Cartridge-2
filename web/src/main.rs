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
    routing::{get, post},
    Router,
};
#[cfg(feature = "onnx")]
use mcts::OnnxEvaluator;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

mod central_config;
mod game;
mod handlers;
#[cfg(feature = "onnx")]
mod model_watcher;
mod types;

use central_config::load_config;
use game::GameSession;
use handlers::{
    get_game_info, get_game_state, get_model_info, get_stats, health, list_games, make_move,
    new_game, selfplay,
};
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
    pub session: Mutex<GameSession>,
    /// Current game ID (for creating new sessions)
    pub current_game: RwLock<String>,
    /// Data directory for stats.json
    pub data_dir: String,
    /// Shared evaluator for MCTS (hot-reloaded)
    pub evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// Model info (updated on reload)
    pub model_info: Arc<RwLock<ModelInfo>>,
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
    engine_games::register_all_games();
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

/// Creates a future that completes when a shutdown signal is received.
/// Handles Ctrl+C on all platforms.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    info!("Shutdown signal received, stopping server...");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("web=info".parse().unwrap())
                .add_directive("ort=warn".parse().unwrap()),
        )
        .init();

    // Register all games
    engine_games::register_all_games();
    info!("Registered all games");

    // Load configuration from config.toml with env var overrides
    let config = load_config();
    let data_dir = config.common.data_dir.clone();
    let default_game = config.common.env_id.clone();
    info!(
        "Configuration: data_dir={}, default_game={}",
        data_dir, default_game
    );

    // Set up shared evaluator for model hot-reloading
    let evaluator: Arc<RwLock<Option<OnnxEvaluator>>> = Arc::new(RwLock::new(None));

    #[cfg(feature = "onnx")]
    let model_info = {
        use engine_core::EngineContext;
        use tracing::warn;
        let model_dir = format!("{}/models", data_dir);
        let obs_size = EngineContext::new(&default_game)
            .map(|ctx| ctx.metadata().obs_size)
            .unwrap_or(29); // Fallback to TicTacToe's obs_size if game not found
        info!(
            "Model watcher using obs_size={} from game '{}'",
            obs_size, default_game
        );

        // Create model watcher
        // Use 1 intra-op thread since web server does single-threaded inference for play
        let model_watcher =
            ModelWatcher::new(&model_dir, "latest.onnx", obs_size, 1, Arc::clone(&evaluator));

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
    let session = GameSession::with_evaluator(&default_game, Arc::clone(&evaluator))?;

    let state = Arc::new(AppState {
        session: Mutex::new(session),
        current_game: RwLock::new(default_game),
        data_dir,
        evaluator,
        model_info,
    });

    // Build router
    let app = create_app(state);

    let addr = format!("{}:{}", config.web.host, config.web.port);
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shut down gracefully");
    Ok(())
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        GameInfoResponse, GameStateResponse, GamesListResponse, HealthResponse, MoveResponse,
    };
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
