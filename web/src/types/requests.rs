//! Request types for the web API.

use serde::Deserialize;

/// Request to start a new game.
#[derive(Deserialize)]
pub struct NewGameRequest {
    /// Who plays first: "player" or "bot"
    #[serde(default = "default_first")]
    pub first: String,
    /// Game to play (e.g., "tictactoe", "connect4")
    #[serde(default)]
    pub game: Option<String>,
}

fn default_first() -> String {
    "player".to_string()
}

/// Request to make a move.
#[derive(Deserialize)]
pub struct MoveRequest {
    /// Position or column to play (game-specific: 0-8 for TicTacToe, 0-6 for Connect4)
    pub position: u8,
}

/// Request to control self-play.
#[derive(Deserialize)]
pub struct SelfPlayRequest {
    /// Action: "start" or "stop"
    pub action: String,
}
