<script lang="ts">
  import { onMount } from 'svelte';
  import Board from './Board.svelte';
  import Connect4Board from './Connect4Board.svelte';
  import Stats from './Stats.svelte';
  import { getGameState, newGame, makeMove, getHealth, getGameInfo, getGames, type GameState, type MoveResponse, type GameInfo } from './lib/api';

  let gameState: GameState | null = $state(null);
  let gameInfo: GameInfo | null = $state(null);
  let availableGames: string[] = $state([]);
  let selectedGame: string = $state('tictactoe');
  let error: string | null = $state(null);
  let loading: boolean = $state(false);
  let serverOnline: boolean = $state(false);
  let lastBotMove: number | null = $state(null);

  onMount(async () => {
    // Check server health
    try {
      await getHealth();
      serverOnline = true;
      // Load available games
      availableGames = await getGames();
      // Load game metadata
      gameInfo = await getGameInfo(selectedGame);
      // Load initial game state
      gameState = await getGameState();
    } catch (e) {
      serverOnline = false;
      error = 'Cannot connect to server. Is the Rust backend running on :8080?';
    }
  });

  async function handleGameChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    selectedGame = target.value;
    loading = true;
    error = null;
    lastBotMove = null;
    try {
      gameInfo = await getGameInfo(selectedGame);
      gameState = await newGame('player', selectedGame);
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }

  async function handleNewGame(first: 'player' | 'bot') {
    loading = true;
    error = null;
    lastBotMove = null;
    try {
      gameState = await newGame(first, selectedGame);
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }

  async function handleCellClick(position: number) {
    if (loading || !gameState || gameState.game_over) return;
    if (gameState.current_player !== gameState.human_player) return; // Not player's turn
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
  <h1>Cartridge2 {gameInfo?.display_name ?? 'Loading...'}</h1>

  {#if serverOnline && availableGames.length > 1}
    <div class="game-selector">
      <label for="game-select">Select Game:</label>
      <select id="game-select" bind:value={selectedGame} onchange={handleGameChange} disabled={loading}>
        {#each availableGames as game}
          <option value={game}>{game}</option>
        {/each}
      </select>
    </div>
  {/if}

  {#if gameInfo?.description}
    <p class="game-description">{gameInfo.description}</p>
  {/if}

  {#if !serverOnline}
    <div class="error">
      <p>Cannot connect to server.</p>
      <p>Make sure the Rust backend is running:</p>
      <code>cd web && cargo run</code>
    </div>
  {:else}
    <div class="game-container">
      <div class="game-section">
        {#if gameState && gameInfo}
          {#if selectedGame === 'connect4'}
            <Connect4Board
              board={gameState.board}
              legalMoves={gameState.legal_moves}
              gameOver={gameState.game_over}
              {lastBotMove}
              onColumnClick={handleCellClick}
            />
          {:else}
            <Board
              board={gameState.board}
              legalMoves={gameState.legal_moves}
              gameOver={gameState.game_over}
              {lastBotMove}
              onCellClick={handleCellClick}
              width={gameInfo.board_width}
              height={gameInfo.board_height}
              playerSymbols={gameInfo.player_symbols}
            />
          {/if}

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
    margin-bottom: 1rem;
  }

  .game-selector {
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
  }

  .game-selector label {
    color: #aaa;
  }

  .game-selector select {
    padding: 0.5rem 1rem;
    font-size: 1rem;
    background: #2a2a4a;
    color: #fff;
    border: 1px solid #4a4a6a;
    border-radius: 8px;
    cursor: pointer;
  }

  .game-selector select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .game-description {
    color: #888;
    font-style: italic;
    margin-bottom: 1.5rem;
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
