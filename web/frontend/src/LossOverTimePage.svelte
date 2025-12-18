<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getStats, type HistoryEntry } from './lib/api';

  let history: HistoryEntry[] = $state([]);
  let error: string | null = $state(null);
  let pollInterval: number | undefined;

  // Chart controls
  type RangeOption = 'last100' | 'last500' | 'last1000' | 'all';
  let selectedRange: RangeOption = $state('all');
  let showAvg100: boolean = $state(true);

  const rangeOptions: { value: RangeOption; label: string }[] = [
    { value: 'last100', label: 'Last 100' },
    { value: 'last500', label: 'Last 500' },
    { value: 'last1000', label: 'Last 1000' },
    { value: 'all', label: 'All' },
  ];

  // Full-screen chart dimensions
  const padding = { top: 40, right: 60, bottom: 60, left: 80 };

  const colors = {
    total: '#00d9ff',
    policy: '#ff6b6b',
    value: '#4ecdc4',
    avg100: '#ffd700',  // Gold color for avg100
  };

  async function fetchData() {
    try {
      const stats = await getStats();
      if (stats.history) {
        history = stats.history;
      }
      error = null;
    } catch (e) {
      error = 'Failed to fetch training data';
    }
  }

  onMount(() => {
    fetchData();
    pollInterval = setInterval(fetchData, 5000);
  });

  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
  });

  function filterByRange(data: HistoryEntry[], range: RangeOption): HistoryEntry[] {
    if (range === 'all' || data.length === 0) return data;
    const count = range === 'last100' ? 100 : range === 'last500' ? 500 : 1000;
    return data.slice(-count);
  }

  function computeRollingAverage(data: HistoryEntry[], window: number = 100): { step: number; avg: number }[] {
    if (data.length === 0) return [];
    const sorted = [...data].sort((a, b) => a.step - b.step);
    const result: { step: number; avg: number }[] = [];

    for (let i = 0; i < sorted.length; i++) {
      const start = Math.max(0, i - window + 1);
      const windowData = sorted.slice(start, i + 1);
      const avg = windowData.reduce((sum, d) => sum + d.total_loss, 0) / windowData.length;
      result.push({ step: sorted[i].step, avg });
    }
    return result;
  }

  function getChartData(data: HistoryEntry[], chartWidth: number, chartHeight: number, includeAvg100: boolean) {
    if (data.length === 0) {
      return { xTicks: [], yTicks: [], paths: { total: '', policy: '', value: '', avg100: '' }, points: { total: [], policy: [], value: [] } };
    }

    const sorted = [...data].sort((a, b) => a.step - b.step);

    // Compute rolling average if needed
    const avg100Data = includeAvg100 ? computeRollingAverage(sorted) : [];

    const steps = sorted.map(d => d.step);
    const minStep = Math.min(...steps);
    const maxStep = Math.max(...steps);

    // Include avg100 values in loss range calculation
    const allLosses = sorted.flatMap(d => [d.total_loss, d.policy_loss, d.value_loss]);
    if (includeAvg100 && avg100Data.length > 0) {
      allLosses.push(...avg100Data.map(d => d.avg));
    }
    const maxLoss = Math.max(...allLosses);
    const minLoss = Math.min(...allLosses);

    const yMax = maxLoss === 0 ? 1 : maxLoss * 1.1;
    const yMin = Math.max(0, minLoss * 0.9);
    const yRange = yMax - yMin || 1;

    const xScale = (step: number) => {
      if (maxStep === minStep) return chartWidth / 2;
      return ((step - minStep) / (maxStep - minStep)) * chartWidth;
    };

    const yScale = (loss: number) => {
      return chartHeight - ((loss - yMin) / yRange) * chartHeight;
    };

    const makePath = (key: 'total_loss' | 'policy_loss' | 'value_loss') => {
      return sorted.map((d, i) => {
        const x = xScale(d.step);
        const y = yScale(d[key]);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      }).join(' ');
    };

    const makeAvg100Path = () => {
      if (!includeAvg100 || avg100Data.length === 0) return '';
      return avg100Data.map((d, i) => {
        const x = xScale(d.step);
        const y = yScale(d.avg);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      }).join(' ');
    };

    const makePoints = (key: 'total_loss' | 'policy_loss' | 'value_loss') => {
      return sorted.map(d => ({
        x: xScale(d.step),
        y: yScale(d[key]),
        step: d.step,
        value: d[key],
      }));
    };

    const xTicks = generateTicks(minStep, maxStep, 10).map(v => ({
      value: v,
      x: xScale(v)
    }));

    const yTicks = generateTicks(yMin, yMax, 8).map(v => ({
      value: v,
      y: yScale(v)
    }));

    return {
      xTicks,
      yTicks,
      paths: {
        total: makePath('total_loss'),
        policy: makePath('policy_loss'),
        value: makePath('value_loss'),
        avg100: makeAvg100Path(),
      },
      points: {
        total: makePoints('total_loss'),
        policy: makePoints('policy_loss'),
        value: makePoints('value_loss'),
      },
    };
  }

  function generateTicks(min: number, max: number, count: number): number[] {
    if (min === max) return [min];
    const step = (max - min) / (count - 1);
    return Array.from({ length: count }, (_, i) => min + step * i);
  }

  function formatStep(value: number): string {
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
    return Math.round(value).toString();
  }

  function formatLoss(value: number): string {
    if (value < 0.001) return value.toExponential(2);
    if (value < 0.01) return value.toFixed(4);
    if (value < 1) return value.toFixed(3);
    return value.toFixed(2);
  }

  // Reactive dimensions based on window size
  let innerWidth = $state(typeof window !== 'undefined' ? window.innerWidth : 1200);
  let innerHeight = $state(typeof window !== 'undefined' ? window.innerHeight : 800);

  function handleResize() {
    innerWidth = window.innerWidth;
    innerHeight = window.innerHeight;
  }

  onMount(() => {
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  });

  let width = $derived(Math.min(innerWidth - 40, 1600));
  let height = $derived(Math.min(innerHeight - 160, 800));
  let chartWidth = $derived(width - padding.left - padding.right);
  let chartHeight = $derived(height - padding.top - padding.bottom);
  let filteredHistory = $derived(filterByRange(history, selectedRange));
  let chartData = $derived(getChartData(filteredHistory, chartWidth, chartHeight, showAvg100));
</script>

<div class="page">
  <header>
    <a href="#/" class="back-link">← Back to Game</a>
    <h1>Loss Over Time</h1>
    <div class="controls">
      <label class="control-group">
        <span>Range:</span>
        <select bind:value={selectedRange}>
          {#each rangeOptions as option}
            <option value={option.value}>{option.label}</option>
          {/each}
        </select>
      </label>
      <label class="control-group checkbox">
        <input type="checkbox" bind:checked={showAvg100} />
        <span>Show Avg100</span>
      </label>
      <span class="data-info">
        Showing {filteredHistory.length} of {history.length} points
      </span>
    </div>
  </header>

  {#if error}
    <div class="error">{error}</div>
  {:else if filteredHistory.length > 1}
    <div class="chart-container">
      <svg viewBox="0 0 {width} {height}" width={width} height={height}>
        <g transform="translate({padding.left}, {padding.top})">
          <!-- Grid lines -->
          {#each chartData.yTicks as tick}
            <line
              x1="0"
              y1={tick.y}
              x2={chartWidth}
              y2={tick.y}
              stroke="#3a3a5a"
              stroke-dasharray="4,4"
            />
          {/each}
          {#each chartData.xTicks as tick}
            <line
              x1={tick.x}
              y1="0"
              x2={tick.x}
              y2={chartHeight}
              stroke="#3a3a5a"
              stroke-dasharray="4,4"
            />
          {/each}

          <!-- Loss lines -->
          <path d={chartData.paths.total} fill="none" stroke={colors.total} stroke-width="3" />
          <path d={chartData.paths.policy} fill="none" stroke={colors.policy} stroke-width="2" stroke-opacity="0.9" />
          <path d={chartData.paths.value} fill="none" stroke={colors.value} stroke-width="2" stroke-opacity="0.9" />

          <!-- Avg100 line (dashed, rendered on top) -->
          {#if showAvg100 && chartData.paths.avg100}
            <path d={chartData.paths.avg100} fill="none" stroke={colors.avg100} stroke-width="2.5" stroke-dasharray="8,4" />
          {/if}

          <!-- Data points (show when not too many) -->
          {#if filteredHistory.length <= 50}
            {#each chartData.points.total as point}
              <circle cx={point.x} cy={point.y} r="4" fill={colors.total} />
            {/each}
            {#each chartData.points.policy as point}
              <circle cx={point.x} cy={point.y} r="3" fill={colors.policy} />
            {/each}
            {#each chartData.points.value as point}
              <circle cx={point.x} cy={point.y} r="3" fill={colors.value} />
            {/each}
          {/if}

          <!-- X-axis -->
          <line x1="0" y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#888" stroke-width="2" />
          {#each chartData.xTicks as tick}
            <text
              x={tick.x}
              y={chartHeight + 25}
              text-anchor="middle"
              fill="#aaa"
              font-size="14"
            >
              {formatStep(tick.value)}
            </text>
          {/each}
          <text
            x={chartWidth / 2}
            y={chartHeight + 50}
            text-anchor="middle"
            fill="#888"
            font-size="16"
          >
            Training Step
          </text>

          <!-- Y-axis -->
          <line x1="0" y1="0" x2="0" y2={chartHeight} stroke="#888" stroke-width="2" />
          {#each chartData.yTicks as tick}
            <text
              x="-12"
              y={tick.y + 5}
              text-anchor="end"
              fill="#aaa"
              font-size="14"
            >
              {formatLoss(tick.value)}
            </text>
          {/each}
          <text
            x="-50"
            y={chartHeight / 2}
            text-anchor="middle"
            fill="#888"
            font-size="16"
            transform="rotate(-90, -50, {chartHeight / 2})"
          >
            Loss
          </text>
        </g>
      </svg>

      <!-- Legend -->
      <div class="legend">
        <div class="legend-item">
          <span class="legend-line" style="background: {colors.total}"></span>
          <span>Total Loss</span>
        </div>
        <div class="legend-item">
          <span class="legend-line" style="background: {colors.policy}"></span>
          <span>Policy Loss</span>
        </div>
        <div class="legend-item">
          <span class="legend-line" style="background: {colors.value}"></span>
          <span>Value Loss</span>
        </div>
        {#if showAvg100}
          <div class="legend-item">
            <span class="legend-line dashed" style="background: {colors.avg100}"></span>
            <span>Avg100</span>
          </div>
        {/if}
      </div>

      <div class="stats-summary">
        {#if history.length > 0}
          {@const latest = history[history.length - 1]}
          <div class="stat">
            <span class="stat-label">Latest Step</span>
            <span class="stat-value">{formatStep(latest.step)}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Total Loss</span>
            <span class="stat-value" style="color: {colors.total}">{formatLoss(latest.total_loss)}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Policy Loss</span>
            <span class="stat-value" style="color: {colors.policy}">{formatLoss(latest.policy_loss)}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Value Loss</span>
            <span class="stat-value" style="color: {colors.value}">{formatLoss(latest.value_loss)}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Data Points</span>
            <span class="stat-value">{history.length}</span>
          </div>
        {/if}
      </div>
    </div>
  {:else}
    <div class="no-data">
      <p>Waiting for training data...</p>
      <p class="hint">Start the Python trainer to see loss curves here.</p>
    </div>
  {/if}
</div>

<style>
  .page {
    min-height: 100vh;
    background: #1a1a2e;
    color: #fff;
    padding: 1rem 2rem;
  }

  header {
    display: flex;
    align-items: center;
    gap: 2rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-left: auto;
  }

  .control-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #aaa;
    font-size: 0.9rem;
  }

  .control-group select {
    background: #3a3a5a;
    color: #fff;
    border: 1px solid #4a4a6a;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    font-size: 0.9rem;
    cursor: pointer;
  }

  .control-group select:hover {
    border-color: #00d9ff;
  }

  .control-group.checkbox {
    cursor: pointer;
  }

  .control-group.checkbox input {
    width: 16px;
    height: 16px;
    accent-color: #00d9ff;
    cursor: pointer;
  }

  .data-info {
    color: #666;
    font-size: 0.85rem;
  }

  .back-link {
    color: #00d9ff;
    text-decoration: none;
    font-size: 1rem;
    padding: 0.5rem 1rem;
    background: #2a2a4a;
    border-radius: 8px;
    transition: background 0.2s;
  }

  .back-link:hover {
    background: #3a3a5a;
  }

  h1 {
    color: #00d9ff;
    margin: 0;
    font-size: 1.8rem;
  }

  .chart-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
  }

  svg {
    background: #2a2a4a;
    border-radius: 12px;
  }

  .legend {
    display: flex;
    justify-content: center;
    gap: 3rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1rem;
    color: #ccc;
  }

  .legend-line {
    width: 24px;
    height: 4px;
    border-radius: 2px;
  }

  .legend-line.dashed {
    height: 0;
    border-top: 4px dashed;
  }

  .stats-summary {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    justify-content: center;
    padding: 1rem;
    background: #2a2a4a;
    border-radius: 12px;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
  }

  .stat-label {
    font-size: 0.85rem;
    color: #888;
  }

  .stat-value {
    font-size: 1.2rem;
    font-weight: bold;
  }

  .error {
    color: #f66;
    background: #4a1a1a;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
  }

  .no-data {
    text-align: center;
    padding: 4rem 2rem;
    background: #2a2a4a;
    border-radius: 12px;
    max-width: 500px;
    margin: 2rem auto;
  }

  .no-data p {
    color: #888;
    font-size: 1.2rem;
    margin: 0.5rem 0;
  }

  .no-data .hint {
    font-size: 1rem;
    color: #666;
  }
</style>
