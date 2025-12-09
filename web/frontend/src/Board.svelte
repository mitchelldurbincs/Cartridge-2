<script lang="ts">
  interface Props {
    board: number[];
    legalMoves: number[];
    gameOver: boolean;
    lastBotMove: number | null;
    onCellClick: (position: number) => void;
    width?: number;
    height?: number;
    playerSymbols?: string[];
  }

  let {
    board,
    legalMoves,
    gameOver,
    lastBotMove,
    onCellClick,
    width = 3,
    height = 3,
    playerSymbols = ['X', 'O']
  }: Props = $props();

  // Calculate cell size based on board dimensions (max container size 400px)
  const MAX_BOARD_SIZE = 400;
  $: cellSize = Math.floor(Math.min(MAX_BOARD_SIZE / width, MAX_BOARD_SIZE / height));
  $: gridStyle = `grid-template-columns: repeat(${width}, ${cellSize}px)`;
  $: fontSize = Math.max(1, Math.floor(cellSize / 32));

  function getCellSymbol(value: number): string {
    if (value === 0) return '';
    // value 1 = player 1, value 2 = player 2, etc.
    const playerIndex = value - 1;
    return playerSymbols[playerIndex] || String(value);
  }

  function getCellClass(index: number, value: number): string {
    let classes = 'cell';
    if (value === 1) classes += ' player1';
    if (value === 2) classes += ' player2';
    if (value === 0 && legalMoves.includes(index) && !gameOver) classes += ' clickable';
    if (index === lastBotMove) classes += ' last-bot-move';
    return classes;
  }
</script>

<div class="board" style={gridStyle}>
  {#each board as cell, i}
    <button
      class={getCellClass(i, cell)}
      style="width: {cellSize}px; height: {cellSize}px; font-size: {fontSize}rem;"
      onclick={() => onCellClick(i)}
      disabled={cell !== 0 || gameOver || !legalMoves.includes(i)}
    >
      {getCellSymbol(cell)}
    </button>
  {/each}
</div>

<style>
  .board {
    display: grid;
    gap: 4px;
    padding: 8px;
    background: #2a2a4a;
    border-radius: 12px;
  }

  .cell {
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

  .cell.player1 {
    color: #00d9ff;
  }

  .cell.player2 {
    color: #ff6b6b;
  }

  .cell.clickable {
    cursor: pointer;
    border-color: #4a4a6a;
  }

  .cell.clickable:hover {
    background: #4a4a6a;
    border-color: #00d9ff;
  }

  .cell.last-bot-move {
    animation: highlight 0.5s ease-out;
  }

  @keyframes highlight {
    0% {
      background: #ff6b6b44;
    }
    100% {
      background: #3a3a5a;
    }
  }

  .cell:disabled {
    cursor: default;
  }
</style>
