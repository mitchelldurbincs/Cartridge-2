<script lang="ts">
  interface Props {
    board: number[];
    legalMoves: number[];
    gameOver: boolean;
    lastBotMove: number | null;
    onColumnClick: (column: number) => void;
  }

  let {
    board,
    legalMoves,
    gameOver,
    lastBotMove,
    onColumnClick
  }: Props = $props();

  const COLS = 7;
  const ROWS = 6;
  const CELL_SIZE = 60;
  const HOLE_SIZE = 48;
  const GAP = 4;

  // Track dropping pieces for animation
  let droppingPiece: { column: number; row: number; player: number } | null = $state(null);
  let animatingCells: Set<number> = $state(new Set());

  // Convert board array to 2D grid (row 0 at bottom)
  function getCell(col: number, row: number): number {
    // Board is stored in row-major order with row 0 at bottom
    const index = row * COLS + col;
    return board[index] || 0;
  }

  // Find the row where a piece would land in a column
  function findLandingRow(col: number): number {
    for (let row = 0; row < ROWS; row++) {
      if (getCell(col, row) === 0) {
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
    droppingPiece = { column: col, row: landingRow, player: 1 };

    // Trigger the actual move after a brief delay to show animation
    setTimeout(() => {
      droppingPiece = null;
      onColumnClick(col);
    }, 400); // Match animation duration
  }

  // Check if a cell should show animation (for bot moves)
  $effect(() => {
    if (lastBotMove !== null && lastBotMove >= 0 && lastBotMove < COLS) {
      // Find the row where the bot's piece landed
      const col = lastBotMove;
      for (let row = ROWS - 1; row >= 0; row--) {
        const index = row * COLS + col;
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

  function getCellClass(col: number, row: number): string {
    const value = getCell(col, row);
    let classes = 'cell';
    if (value === 1) classes += ' player1';
    if (value === 2) classes += ' player2';

    const index = row * COLS + col;
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
    return -CELL_SIZE - 20; // Start above the board
  }

  function getDropEndY(row: number): number {
    // Convert row to visual position (row 0 is at bottom, but CSS row 0 is at top)
    const visualRow = ROWS - 1 - row;
    return visualRow * (CELL_SIZE + GAP) + (CELL_SIZE - HOLE_SIZE) / 2;
  }
</script>

<div class="connect4-container">
  <!-- Hover indicators for columns -->
  <div class="column-indicators">
    {#each Array(COLS) as _, col}
      <button
        class="column-indicator"
        class:clickable={isColumnClickable(col)}
        onclick={() => handleColumnClick(col)}
        disabled={!isColumnClickable(col)}
        aria-label={`Drop piece in column ${col + 1}`}
      >
        {#if isColumnClickable(col)}
          <div class="hover-piece player1-preview"></div>
        {/if}
      </button>
    {/each}
  </div>

  <!-- Main board -->
  <div class="board-frame">
    <!-- Dropping piece animation -->
    {#if droppingPiece}
      <div
        class="dropping-piece player1"
        style="
          left: {droppingPiece.column * (CELL_SIZE + GAP) + (CELL_SIZE - HOLE_SIZE) / 2}px;
          --drop-start: {getDropStartY()}px;
          --drop-end: {getDropEndY(droppingPiece.row)}px;
        "
      ></div>
    {/if}

    <!-- Board grid (blue frame with holes) -->
    <div class="board-grid">
      {#each Array(ROWS) as _, visualRow}
        {@const row = ROWS - 1 - visualRow}
        <div class="board-row">
          {#each Array(COLS) as _, col}
            <div class={getCellClass(col, row)}>
              <div class="hole">
                {#if getCell(col, row) !== 0}
                  <div class="piece" class:player1={getCell(col, row) === 1} class:player2={getCell(col, row) === 2}></div>
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

<style>
  .connect4-container {
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
    width: 60px;
    height: 50px;
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
    width: 48px;
    height: 48px;
    border-radius: 50%;
    opacity: 0;
    transform: scale(0.8);
    transition: all 0.15s ease;
  }

  .player1-preview {
    background: radial-gradient(circle at 30% 30%, #ff6b6b, #e74c3c);
    box-shadow: 0 2px 8px rgba(231, 76, 60, 0.4);
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
    gap: 4px;
    background: linear-gradient(180deg, #2980b9 0%, #3498db 50%, #2980b9 100%);
    padding: 8px;
    border-radius: 8px;
    box-shadow:
      inset 0 2px 8px rgba(0, 0, 0, 0.3),
      inset 0 -1px 2px rgba(255, 255, 255, 0.1);
  }

  .board-row {
    display: flex;
    gap: 4px;
  }

  .cell {
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .hole {
    width: 48px;
    height: 48px;
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
    width: 44px;
    height: 44px;
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

  .cell.just-dropped .piece {
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
    width: 44px;
    height: 44px;
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

  /* Responsive adjustments */
  @media (max-width: 500px) {
    .column-indicator {
      width: 45px;
    }
    .cell {
      width: 45px;
      height: 45px;
    }
    .hole {
      width: 36px;
      height: 36px;
    }
    .piece {
      width: 32px;
      height: 32px;
    }
    .hover-piece {
      width: 36px;
      height: 36px;
    }
    .dropping-piece {
      width: 32px;
      height: 32px;
    }
  }
</style>
