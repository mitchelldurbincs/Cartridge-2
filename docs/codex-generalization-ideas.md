# Generalizing the game engine beyond Tic Tac Toe

## Core abstractions
- Define a common `Game` interface (e.g., `initial_state`, `legal_actions(state)`, `apply_action(state, action)`, `is_terminal(state)`, `utility(state, player)`) so new games only implement game logic.
- Represent players generically (e.g., enums or identifiers) instead of `X`/`O`, and let games declare turn order and number of participants.
- Use a generic `GameState` data structure that holds board dimensions, per-cell content, current player, and game-specific metadata (gravity, connect-length, etc.).
- Decouple rendering from logic by using a pluggable `Renderer` (text/UI) that consumes the abstract state instead of board-size-specific assumptions.

## Configuration and metadata
- Move game-specific constants (board size, connect target, player count, move ordering) into game descriptors/config files to avoid hard-coded 3x3 logic.
- Provide a registration/discovery mechanism (e.g., `games/registry.py`) so the engine can load any game module by id.
- Allow per-game action encoding (e.g., row/column selection) and map them to generic action ids for shared RL/training pipelines.

## Input and validation
- Implement generic input validation that defers rule checks to the game module (e.g., whether a column is full in Connect 4) while keeping shared error handling.
- Standardize coordinate systems (0-based vs 1-based) and expose helper converters to avoid UI-specific assumptions.

## Training and evaluation
- Ensure MCTS/agent code reads branching factor and terminal detection from the game interface rather than assuming 9 squares.
- Allow value/policy network architectures to be parameterized by board dimensions and action space size instead of fixed 9-output heads.
- Build a replay buffer format that stores game id + game-specific state/action encodings so past games remain usable after adding new games.

## Serialization and persistence
- Make save/load formats include game id, board dimensions, and current player so sessions for different games can be restored reliably.
- Normalize representation of actions (e.g., `(row, col)`, column index) and store them in a versioned schema to handle future games with different action shapes.

## Testing and tooling
- Add property-based tests for generic guarantees (no illegal actions, turn alternation, detecting terminal states) that run against every registered game.
- Create fixtures for small-board variants of Connect 4 (e.g., 4x4 connect-3) to quickly validate rule logic without full-size search costs.
- Provide golden game transcripts to validate serialization, replay, and rendering across multiple games.

## UI/UX considerations
- Use layout helpers that derive board rendering from dimensions (rows/cols) instead of hard-coded 3x3 grids.
- Allow per-game descriptions/tooltips so the UI can display rules and win conditions dynamically.
- Support highlighting last move and valid moves via the generic action mapping to improve debugging across games.
