<script lang="ts">
  import { onMount } from 'svelte';
  import Board from './Board.svelte';
  import Stats from './Stats.svelte';
  import { getGameState, newGame, makeMove, getHealth, type GameState, type MoveResponse } from './lib/api';

  let gameState: GameState | null = $state(null);
  let error: string | null = $state(null);
  let loading: boolean = $state(false);
  let serverOnline: boolean = $state(false);
  let lastBotMove: number | null = $state(null);

  onMount(async () => {
    // Check server health
    try {
      await getHealth();
      serverOnline = true;
      // Load initial game state
      gameState = await getGameState();
    } catch (e) {
      serverOnline = false;
      error = 'Cannot connect to server. Is the Rust backend running on :8080?';
    }
  });

  async function handleNewGame(first: 'player' | 'bot') {
    loading = true;
    error = null;
    lastBotMove = null;
    try {
      gameState = await newGame(first);
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }

  async function handleCellClick(position: number) {
    if (loading || !gameState || gameState.game_over) return;
    if (gameState.current_player !== 1) return; // Not player's turn
    if (!gameState.legal_moves.includes(position)) return; // Illegal move

    loading = true;
    error = null;
    try {
      const response: MoveResponse = await makeMove(position);
      gameState = response;
      lastBotMove = response.bot_move;
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }
</script>

<main>
  <h1>Cartridge2 TicTacToe</h1>

  {#if !serverOnline}
    <div class="error">
      <p>Cannot connect to server.</p>
      <p>Make sure the Rust backend is running:</p>
      <code>cd web && cargo run</code>
    </div>
  {:else}
    <div class="game-container">
      <div class="game-section">
        {#if gameState}
          <Board
            board={gameState.board}
            legalMoves={gameState.legal_moves}
            gameOver={gameState.game_over}
            {lastBotMove}
            onCellClick={handleCellClick}
          />

          <div class="status" class:winner={gameState.winner === 1} class:loser={gameState.winner === 2}>
            {gameState.message}
          </div>

          {#if error}
            <div class="error">{error}</div>
          {/if}

          <div class="controls">
            <button onclick={() => handleNewGame('player')} disabled={loading}>
              New Game (You First)
            </button>
            <button onclick={() => handleNewGame('bot')} disabled={loading}>
              New Game (Bot First)
            </button>
          </div>
        {:else}
          <p>Loading game...</p>
        {/if}
      </div>

      <div class="stats-section">
        <Stats />
      </div>
    </div>
  {/if}
</main>

<style>
  main {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
    text-align: center;
  }

  h1 {
    color: #00d9ff;
    margin-bottom: 2rem;
  }

  .game-container {
    display: flex;
    gap: 3rem;
    justify-content: center;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .game-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .stats-section {
    min-width: 250px;
  }

  .status {
    font-size: 1.2rem;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    background: #2a2a4a;
  }

  .status.winner {
    background: #1a4a1a;
    color: #4f4;
  }

  .status.loser {
    background: #4a1a1a;
    color: #f44;
  }

  .error {
    color: #f66;
    background: #4a1a1a;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
  }

  .error code {
    display: block;
    margin-top: 0.5rem;
    background: #333;
    padding: 0.5rem;
    border-radius: 4px;
  }

  .controls {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
  }

  button {
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    background: #00d9ff;
    color: #1a1a2e;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
  }

  button:hover:not(:disabled) {
    background: #00b8dd;
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
