<script lang="ts">
  interface Props {
    board: number[];
    legalMoves: number[];
    gameOver: boolean;
    lastBotMove: number | null;
    onCellClick: (position: number) => void;
  }

  let { board, legalMoves, gameOver, lastBotMove, onCellClick }: Props = $props();

  function getCellSymbol(value: number): string {
    if (value === 1) return 'X';
    if (value === 2) return 'O';
    return '';
  }

  function getCellClass(index: number, value: number): string {
    let classes = 'cell';
    if (value === 1) classes += ' x';
    if (value === 2) classes += ' o';
    if (value === 0 && legalMoves.includes(index) && !gameOver) classes += ' clickable';
    if (index === lastBotMove) classes += ' last-bot-move';
    return classes;
  }
</script>

<div class="board">
  {#each board as cell, i}
    <button
      class={getCellClass(i, cell)}
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
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    padding: 8px;
    background: #2a2a4a;
    border-radius: 12px;
  }

  .cell {
    width: 80px;
    height: 80px;
    font-size: 2.5rem;
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

  .cell.x {
    color: #00d9ff;
  }

  .cell.o {
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
