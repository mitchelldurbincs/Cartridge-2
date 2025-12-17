<script lang="ts">
  import type { GameInfo } from './lib/api';

  interface Props {
    board: number[];
    legalMoves: number[];
    gameOver: boolean;
    lastBotMove: number | null;
    gameInfo: GameInfo;
    currentPlayer: number;
    onCellClick: (position: number) => void;
  }

  let {
    board,
    legalMoves,
    gameOver,
    lastBotMove,
    gameInfo,
    currentPlayer,
    onCellClick
  }: Props = $props();

  // Extract dimensions from metadata
  let width = $derived(gameInfo.board_width);
  let height = $derived(gameInfo.board_height);
  let boardType = $derived(gameInfo.board_type);
  let playerSymbols = $derived(gameInfo.player_symbols);

  // ============================================================================
  // Grid Board (TicTacToe, Othello style)
  // ============================================================================

  // Calculate cell size based on board dimensions (max container size 400px)
  const MAX_BOARD_SIZE = 400;
  let gridCellSize = $derived(Math.floor(Math.min(MAX_BOARD_SIZE / width, MAX_BOARD_SIZE / height)));
  let gridStyle = $derived(`grid-template-columns: repeat(${width}, ${gridCellSize}px)`);
  let gridFontSize = $derived(Math.max(1, Math.floor(gridCellSize / 32)));

  function getGridCellSymbol(value: number): string {
    if (value === 0) return '';
    const playerIndex = value - 1;
    return playerSymbols[playerIndex] || String(value);
  }

  function getGridCellClass(index: number, value: number): string {
    let classes = 'grid-cell';
    if (value === 1) classes += ' player1';
    if (value === 2) classes += ' player2';
    if (value === 0 && legalMoves.includes(index) && !gameOver) classes += ' clickable';
    if (index === lastBotMove) classes += ' last-bot-move';
    return classes;
  }

  // ============================================================================
  // Drop Column Board (Connect 4 style)
  // ============================================================================

  // Calculate sizes based on board dimensions
  const DROP_MAX_SIZE = 420;
  let dropCellSize = $derived(Math.floor(Math.min(DROP_MAX_SIZE / width, DROP_MAX_SIZE / height)));
  let dropHoleSize = $derived(Math.floor(dropCellSize * 0.8));
  let dropPieceSize = $derived(Math.floor(dropHoleSize * 0.92));
  const GAP = 4;
  const BOARD_FRAME_PADDING = 12;
  const BOARD_GRID_PADDING = 8;

  // Track dropping pieces for animation
  let droppingPiece: { column: number; row: number; player: number } | null = $state(null);
  let animatingCells: Set<number> = $state(new Set());

  // Convert board array to 2D grid (row 0 at bottom for drop_column)
  function getDropCell(col: number, row: number): number {
    const index = row * width + col;
    return board[index] || 0;
  }

  // Find the row where a piece would land in a column
  function findLandingRow(col: number): number {
    for (let row = 0; row < height; row++) {
      if (getDropCell(col, row) === 0) {
        return row;
      }
    }
    return -1; // Column is full
  }

  // Handle column click with animation
  function handleColumnClick(col: number) {
    if (gameOver || !legalMoves.includes(col)) return;

    const landingRow = findLandingRow(col);
    if (landingRow === -1) return;

    // Start drop animation
    droppingPiece = { column: col, row: landingRow, player: currentPlayer };

    // Trigger the actual move after a brief delay to show animation
    setTimeout(() => {
      droppingPiece = null;
      onCellClick(col);
    }, 400);
  }

  // Check if a cell should show animation (for bot moves)
  $effect(() => {
    if (boardType === 'drop_column' && lastBotMove !== null && lastBotMove >= 0 && lastBotMove < width) {
      const col = lastBotMove;
      for (let row = height - 1; row >= 0; row--) {
        const index = row * width + col;
        if (board[index] === 2) {
          animatingCells.add(index);
          setTimeout(() => {
            animatingCells = new Set([...animatingCells].filter(i => i !== index));
          }, 500);
          break;
        }
      }
    }
  });

  function getDropCellClass(col: number, row: number): string {
    const value = getDropCell(col, row);
    let classes = 'drop-cell';
    if (value === 1) classes += ' player1';
    if (value === 2) classes += ' player2';

    const index = row * width + col;
    if (animatingCells.has(index)) {
      classes += ' just-dropped';
    }
    return classes;
  }

  function isColumnClickable(col: number): boolean {
    return !gameOver && legalMoves.includes(col) && !droppingPiece;
  }

  // Calculate the Y position for the dropping animation
  function getDropStartY(): number {
    return -dropCellSize - 20;
  }

  function getDropEndY(row: number): number {
    const visualRow = height - 1 - row;
    const paddingOffset = BOARD_FRAME_PADDING + BOARD_GRID_PADDING;
    return paddingOffset + visualRow * (dropCellSize + GAP) + (dropCellSize - dropHoleSize) / 2;
  }

  function getDropX(col: number): number {
    const paddingOffset = BOARD_FRAME_PADDING + BOARD_GRID_PADDING;
    return paddingOffset + col * (dropCellSize + GAP) + (dropCellSize - dropHoleSize) / 2;
  }
</script>

{#if boardType === 'grid'}
  <!-- Grid-style board (TicTacToe, Othello) -->
  <div class="grid-board" style={gridStyle}>
    {#each board as cell, i}
      <button
        class={getGridCellClass(i, cell)}
        style="width: {gridCellSize}px; height: {gridCellSize}px; font-size: {gridFontSize}rem;"
        onclick={() => onCellClick(i)}
        disabled={cell !== 0 || gameOver || !legalMoves.includes(i)}
      >
        {getGridCellSymbol(cell)}
      </button>
    {/each}
  </div>
{:else if boardType === 'drop_column'}
  <!-- Drop-column style board (Connect 4) -->
  <div class="drop-container">
    <!-- Hover indicators for columns -->
    <div class="column-indicators" style="padding-left: {BOARD_FRAME_PADDING + BOARD_GRID_PADDING}px; padding-right: {BOARD_FRAME_PADDING + BOARD_GRID_PADDING}px;">
      {#each Array(width) as _, col}
        <button
          class="column-indicator"
          class:clickable={isColumnClickable(col)}
          style="width: {dropCellSize}px; height: 50px;"
          onclick={() => handleColumnClick(col)}
          disabled={!isColumnClickable(col)}
          aria-label={`Drop piece in column ${col + 1}`}
        >
          {#if isColumnClickable(col)}
            <div
              class={`hover-piece player${currentPlayer}-preview`}
              style="width: {dropHoleSize}px; height: {dropHoleSize}px;"
            ></div>
          {/if}
        </button>
      {/each}
    </div>

    <!-- Main board -->
    <div class="board-frame">
      <!-- Dropping piece animation -->
      {#if droppingPiece}
        <div
          class={`dropping-piece player${droppingPiece.player}`}
          style="
            left: {getDropX(droppingPiece.column)}px;
            width: {dropPieceSize}px;
            height: {dropPieceSize}px;
            --drop-start: {getDropStartY()}px;
            --drop-end: {getDropEndY(droppingPiece.row)}px;
          "
        ></div>
      {/if}

      <!-- Board grid (blue frame with holes) -->
      <div class="board-grid" style="gap: {GAP}px;">
        {#each Array(height) as _, visualRow}
          {@const row = height - 1 - visualRow}
          <div class="board-row" style="gap: {GAP}px;">
            {#each Array(width) as _, col}
              <div class={getDropCellClass(col, row)} style="width: {dropCellSize}px; height: {dropCellSize}px;">
                <div class="hole" style="width: {dropHoleSize}px; height: {dropHoleSize}px;">
                  {#if getDropCell(col, row) !== 0}
                    <div
                      class="piece"
                      class:player1={getDropCell(col, row) === 1}
                      class:player2={getDropCell(col, row) === 2}
                      style="width: {dropPieceSize}px; height: {dropPieceSize}px;"
                    ></div>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        {/each}
      </div>

      <!-- Board stand -->
      <div class="board-stand"></div>
    </div>
  </div>
{:else}
  <!-- Fallback for unknown board types -->
  <div class="unknown-board">
    <p>Unknown board type: {boardType}</p>
  </div>
{/if}

<style>
  /* ============================================================================
   * Grid Board Styles (TicTacToe, Othello)
   * ============================================================================ */
  .grid-board {
    display: grid;
    gap: 4px;
    padding: 8px;
    background: #2a2a4a;
    border-radius: 12px;
  }

  .grid-cell {
    font-weight: bold;
    background: #3a3a5a;
    border: 2px solid transparent;
    border-radius: 8px;
    cursor: default;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .grid-cell.player1 {
    color: #00d9ff;
  }

  .grid-cell.player2 {
    color: #ff6b6b;
  }

  .grid-cell.clickable {
    cursor: pointer;
    border-color: #4a4a6a;
  }

  .grid-cell.clickable:hover {
    background: #4a4a6a;
    border-color: #00d9ff;
  }

  .grid-cell.last-bot-move {
    animation: grid-highlight 0.5s ease-out;
  }

  @keyframes grid-highlight {
    0% {
      background: #ff6b6b44;
    }
    100% {
      background: #3a3a5a;
    }
  }

  .grid-cell:disabled {
    cursor: default;
  }

  /* ============================================================================
   * Drop Column Board Styles (Connect 4)
   * ============================================================================ */
  .drop-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    user-select: none;
  }

  .column-indicators {
    display: flex;
    gap: 4px;
    margin-bottom: 8px;
    height: 50px;
  }

  .column-indicator {
    background: transparent;
    border: none;
    cursor: default;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    padding-bottom: 4px;
  }

  .column-indicator.clickable {
    cursor: pointer;
  }

  .column-indicator.clickable:hover .hover-piece {
    opacity: 1;
    transform: scale(1);
  }

  .hover-piece {
    border-radius: 50%;
    opacity: 0;
    transform: scale(0.8);
    transition: all 0.15s ease;
  }

  .player1-preview {
    background: radial-gradient(circle at 30% 30%, #ff6b6b, #e74c3c);
    box-shadow: 0 2px 8px rgba(231, 76, 60, 0.4);
  }

  .player2-preview {
    background: radial-gradient(circle at 30% 30%, #ffe066, #f1c40f);
    box-shadow: 0 2px 8px rgba(243, 156, 18, 0.4);
  }

  .board-frame {
    position: relative;
    background: linear-gradient(180deg, #1e5799 0%, #2989d8 50%, #1e5799 100%);
    border-radius: 12px;
    padding: 12px;
    box-shadow:
      0 8px 32px rgba(0, 0, 0, 0.3),
      inset 0 2px 4px rgba(255, 255, 255, 0.1),
      inset 0 -2px 4px rgba(0, 0, 0, 0.2);
  }

  .board-grid {
    display: flex;
    flex-direction: column;
    background: linear-gradient(180deg, #2980b9 0%, #3498db 50%, #2980b9 100%);
    padding: 8px;
    border-radius: 8px;
    box-shadow:
      inset 0 2px 8px rgba(0, 0, 0, 0.3),
      inset 0 -1px 2px rgba(255, 255, 255, 0.1);
  }

  .board-row {
    display: flex;
  }

  .drop-cell {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .hole {
    border-radius: 50%;
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    box-shadow:
      inset 0 4px 8px rgba(0, 0, 0, 0.6),
      inset 0 -2px 4px rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }

  .piece {
    border-radius: 50%;
    transition: transform 0.1s ease;
  }

  .piece.player1 {
    background: radial-gradient(circle at 30% 30%, #ff8a8a, #e74c3c 60%, #c0392b);
    box-shadow:
      0 2px 4px rgba(0, 0, 0, 0.3),
      inset 0 2px 4px rgba(255, 255, 255, 0.3),
      inset 0 -2px 4px rgba(0, 0, 0, 0.2);
  }

  .piece.player2 {
    background: radial-gradient(circle at 30% 30%, #ffe066, #f1c40f 60%, #f39c12);
    box-shadow:
      0 2px 4px rgba(0, 0, 0, 0.3),
      inset 0 2px 4px rgba(255, 255, 255, 0.4),
      inset 0 -2px 4px rgba(0, 0, 0, 0.1);
  }

  .drop-cell.just-dropped .piece {
    animation: pop-in 0.3s ease-out;
  }

  @keyframes pop-in {
    0% {
      transform: scale(0.8);
    }
    50% {
      transform: scale(1.1);
    }
    100% {
      transform: scale(1);
    }
  }

  .dropping-piece {
    position: absolute;
    border-radius: 50%;
    z-index: 10;
    animation: drop-piece 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
  }

  .dropping-piece.player1 {
    background: radial-gradient(circle at 30% 30%, #ff8a8a, #e74c3c 60%, #c0392b);
    box-shadow:
      0 4px 8px rgba(0, 0, 0, 0.4),
      inset 0 2px 4px rgba(255, 255, 255, 0.3),
      inset 0 -2px 4px rgba(0, 0, 0, 0.2);
  }

  .dropping-piece.player2 {
    background: radial-gradient(circle at 30% 30%, #ffe066, #f1c40f 60%, #f39c12);
    box-shadow:
      0 4px 8px rgba(0, 0, 0, 0.4),
      inset 0 2px 4px rgba(255, 255, 255, 0.4),
      inset 0 -2px 4px rgba(0, 0, 0, 0.1);
  }

  @keyframes drop-piece {
    0% {
      top: var(--drop-start);
      opacity: 1;
    }
    80% {
      top: var(--drop-end);
    }
    90% {
      top: calc(var(--drop-end) - 4px);
    }
    100% {
      top: var(--drop-end);
      opacity: 1;
    }
  }

  .board-stand {
    width: 100%;
    height: 20px;
    background: linear-gradient(180deg, #1e5799 0%, #0f3460 100%);
    border-radius: 0 0 8px 8px;
    margin-top: -4px;
    box-shadow:
      0 4px 8px rgba(0, 0, 0, 0.3),
      inset 0 1px 2px rgba(255, 255, 255, 0.1);
  }

  /* ============================================================================
   * Unknown Board Type
   * ============================================================================ */
  .unknown-board {
    padding: 2rem;
    background: #4a1a1a;
    border-radius: 12px;
    color: #f66;
  }

  /* ============================================================================
   * Responsive adjustments
   * ============================================================================ */
  @media (max-width: 500px) {
    .column-indicator {
      height: 40px;
    }
  }
</style>
