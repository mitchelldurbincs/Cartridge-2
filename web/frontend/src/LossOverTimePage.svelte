<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getStats, type HistoryEntry } from './lib/api';

  let history: HistoryEntry[] = $state([]);
  let error: string | null = $state(null);
  let pollInterval: number | undefined;

  // Full-screen chart dimensions
  const padding = { top: 40, right: 60, bottom: 60, left: 80 };

  const colors = {
    total: '#00d9ff',
    policy: '#ff6b6b',
    value: '#4ecdc4',
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

  function getChartData(data: HistoryEntry[], chartWidth: number, chartHeight: number) {
    if (data.length === 0) {
      return { xTicks: [], yTicks: [], paths: { total: '', policy: '', value: '' }, points: { total: [], policy: [], value: [] } };
    }

    const sorted = [...data].sort((a, b) => a.step - b.step);

    const steps = sorted.map(d => d.step);
    const minStep = Math.min(...steps);
    const maxStep = Math.max(...steps);

    const allLosses = sorted.flatMap(d => [d.total_loss, d.policy_loss, d.value_loss]);
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
  let chartData = $derived(getChartData(history, chartWidth, chartHeight));
</script>

<div class="page">
  <header>
    <a href="#/" class="back-link">← Back to Game</a>
    <h1>Loss Over Time</h1>
  </header>

  {#if error}
    <div class="error">{error}</div>
  {:else if history.length > 1}
    <div class="chart-container">
      <svg viewBox="0 0 {width} {height}" {width} {height}>
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

          <!-- Data points (show when not too many) -->
          {#if history.length <= 50}
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
