//! Game-related handlers.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use engine_core::{create_game, list_registered_games};
use std::sync::Arc;

use crate::game::GameSession;
use crate::types::{
    GameInfoResponse, GameStateResponse, GamesListResponse, MoveRequest, MoveResponse,
    NewGameRequest,
};
use crate::AppState;

/// List all available games.
pub async fn list_games() -> Json<GamesListResponse> {
    Json(GamesListResponse {
        games: list_registered_games(),
    })
}

/// Get metadata for a specific game.
pub async fn get_game_info(
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

/// Get current game state.
pub async fn get_game_state(
    State(state): State<Arc<AppState>>,
) -> Result<Json<GameStateResponse>, (StatusCode, String)> {
    let session = state.session.lock().await;
    Ok(Json(session.to_response()))
}

/// Start a new game.
pub async fn new_game(
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

/// Make a move (player + bot response).
pub async fn make_move(
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
