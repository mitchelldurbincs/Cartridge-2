### Cartridge 2
Be able to visualize training of AlphaZero AND be able to play against it 
I think I can reuse the engine-rust and the actor from Cartridge

Big Game Goals:
* Tic tac toe 
* Connect 4
* Othello

Here is your rewritten design document, streamlined for the **Monolithic/Filesystem MVP** approach (v2).

### Cartridge 2 (Filesystem MVP)

Be able to visualize training of AlphaZero AND be able to play against it (without the DevOps headache).

I will reuse the engine-rust code, but refactor it from a Service into a Library.

**Big Game Goals:**

- Tic tac toe
- Connect 4
- Othello

**I want to:**

- Monitor training locally (e.g., Python writes a `stats.json` that the frontend polls).
- Play the bots via a Web Portal (hosted by the Rust process).
- Have full control over training hyperparameters (Python).
- **Run everything on one machine** first (simplify the loop), then worry about K8s later.

**Components**

- **Shared Filesystem** - The "glue" between processes (`./data/replays/` and `./data/models/`).
- **Rust Monolith** - Handles Game Logic, MCTS, Self-Play, and serving the Web API.
- **Python Trainer** - Handles the Neural Network training loop.

**MVP:**

- TicTacToe implementation (Rust Crate + Python Script).
- Web Interface to play against the current "best" model.
- No K8s yet—just two terminal windows (one for Rust, one for Python).

[https://github.com/suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)

**Later:**

- Tournament of alphazero
- Porting the Python Trainer to a GPU Cloud Node (communicating via S3 instead of local disk).

---

## Actor (The "Player")

Currently in Rust
Needs to import the Engine crate and ort (ONNX Runtime).
TLDR: A long-running process that:

1. Watches `./data/models/latest.onnx` for updates.
2. Runs MCTS self-play loops using the Engine and ONNX model.
3. Saves completed games to `./data/replay.db` (SQLite).
4. Exposes an HTTP server so the **Web Frontend** can request moves.
    

## Engine (The "Rules")

**Status: COMPLETE**

Rust library crate. Refactored from gRPC Service -> pure Rust library.

TLDR: A pure library. Apply action a to state s -> returns new_state. No network I/O. Used directly by the Actor for maximum speed.

**API:**
```rust
use engine_core::EngineContext;
use games_tictactoe::register_tictactoe;

register_tictactoe();
let mut ctx = EngineContext::new("tictactoe").unwrap();
let reset = ctx.reset(seed, &[]).unwrap();
let step = ctx.step(&reset.state, &action).unwrap();
```

## Replay

**Implementation:** SQLite Database (`./data/replay.db`) **TLDR:** A single SQLite database file acting as a concurrent, persistent buffer. This replaces the folder of JSON files to prevent filesystem exhaustion and enable efficient random sampling.

- **Actor (Rust):** Connects to `replay.db` and executes a blocking `INSERT` to add a new row containing the compressed game binary/JSON and a timestamp after every episode.
    
- **Learner (Python):**
    
    - **Sampling:** Executes `SELECT data FROM games ORDER BY RANDOM() LIMIT :batch_size` to generate training data.
        
    - **Window Management:** Periodically runs a cleanup query (e.g., `DELETE FROM games WHERE id NOT IN (SELECT id FROM games ORDER BY created_at DESC LIMIT :window_size)`) to keep the replay buffer from growing infinitely.
## Weights

Currently in Filesystem

TLDR: A single file: ./data/models/model.onnx
- **Learner** exports this after every epoch.
- **Actor** hot-reloads this when the timestamp changes.
    

## Learner (The "Trainer")
**Currently in Python** **TLDR:** A continuous loop that trains the network on data from SQLite and publishes versioned models safely.

**Process:**
1. **Data Loading (SQLite):** Instead of scanning a folder of thousands of files, it runs a SQL query to fetch a randomized batch of games: `SELECT game_data FROM games ORDER BY RANDOM() LIMIT 1000;`
2. **Training:** Updates the PyTorch model parameters θ to minimize the combined AlphaZero loss (Policy Cross-Entropy + Value MSE).
3. **Checkpointing (Garbage Collection):** Saves a historical snapshot every N steps (e.g., `model_step_001000.onnx`).
    - _Logic:_ Keep the last 10 checkpoints, and delete older ones (unless you want to keep "milestones" like every 10k steps for tournaments).
4. **Atomic Publishing:** To prevents the "Half-Written Model" crash in Rust, it uses a **write-then-rename** pattern:
    - Step A: Save to `temp_model.onnx`.
    - Step B: Execute `os.replace('temp_model.onnx', 'latest.onnx')`.
    - _Result:_ The Rust Actor only ever sees a fully written, valid file at `latest.onnx`.
5. **Telemetry:** Writes `stats.json` (loss, win-rate, learning rate) for the Web Portal to poll.

---

# WEB

## Backend

Rust (The Actor).

The Actor process will run a lightweight HTTP server (e.g., Axum or Actix) to handle move requests from the browser.

## Frontend

Svelte + Typescript.
Talks directly to localhost:8080 (The Rust Actor).

## Messages

**None** (Replaced by Filesystem + HTTP).

## Containerization

Docker (Optional for MVP, but good for environment consistency).

## Orchestration

Manual / Shell Script

./start_training.sh -> launches the Rust binary and the Python script in parallel.

## Local K8s

**None** (Skipped for MVP).

## GitHub Actions

- cargo fmt, black, go fmt, golangci-lint

---

# Later
- **Remote Storage:** Swap local filesystem for S3/MinIO when moving to the cloud.
- **NATS:** Introduce NATS only when you need multiple Actor replicas to coordinate.
- **Embed the Actor:** Technically, the Actor already embeds the Engine in this v2 architecture!
