<script lang="ts">
  import type { HistoryEntry } from './lib/api';
  import { LARGE_NUMBER_THRESHOLD } from './lib/constants';

  interface Props {
    history: HistoryEntry[];
  }

  let { history }: Props = $props();

  // Chart dimensions
  const width = 400;
  const height = 200;
  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Colors matching the dark theme
  const colors = {
    total: '#00d9ff',   // Cyan - primary brand color
    policy: '#ff6b6b', // Red
    value: '#4ecdc4',  // Teal
  };

  // Compute scales and paths
  function getChartData(data: HistoryEntry[]) {
    if (data.length === 0) {
      return { xTicks: [], yTicks: [], paths: { total: '', policy: '', value: '' }, yMax: 1 };
    }

    const sorted = [...data].sort((a, b) => a.step - b.step);

    // Get data ranges
    const steps = sorted.map(d => d.step);
    const minStep = Math.min(...steps);
    const maxStep = Math.max(...steps);

    const allLosses = sorted.flatMap(d => [d.total_loss, d.policy_loss, d.value_loss]);
    const maxLoss = Math.max(...allLosses);
    const minLoss = Math.min(...allLosses);

    // Add some padding to y-axis
    const yMax = maxLoss === 0 ? 1 : maxLoss * 1.1;
    const yMin = Math.max(0, minLoss * 0.9);
    const yRange = yMax - yMin || 1;

    // Scale functions
    const xScale = (step: number) => {
      if (maxStep === minStep) return chartWidth / 2;
      return ((step - minStep) / (maxStep - minStep)) * chartWidth;
    };

    const yScale = (loss: number) => {
      return chartHeight - ((loss - yMin) / yRange) * chartHeight;
    };

    // Generate paths
    const makePath = (key: 'total_loss' | 'policy_loss' | 'value_loss') => {
      return sorted.map((d, i) => {
        const x = xScale(d.step);
        const y = yScale(d[key]);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      }).join(' ');
    };

    // Generate tick values
    const xTicks = generateTicks(minStep, maxStep, 5).map(v => ({
      value: v,
      x: xScale(v)
    }));

    const yTicks = generateTicks(yMin, yMax, 5).map(v => ({
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
      yMax,
    };
  }

  // Generate nice tick values
  function generateTicks(min: number, max: number, count: number): number[] {
    if (min === max) return [min];
    const step = (max - min) / (count - 1);
    return Array.from({ length: count }, (_, i) => min + step * i);
  }

  // Format tick labels
  function formatStep(value: number): string {
    if (value >= LARGE_NUMBER_THRESHOLD) return `${(value / LARGE_NUMBER_THRESHOLD).toFixed(1)}k`;
    return Math.round(value).toString();
  }

  function formatLoss(value: number): string {
    if (value < 0.01) return value.toExponential(1);
    if (value < 1) return value.toFixed(3);
    return value.toFixed(2);
  }

  // Reactive chart data
  let chartData = $derived(getChartData(history));
</script>

<div class="loss-chart">
  <div class="chart-header">
    <h3>Loss Over Time</h3>
    <a href="#/loss-over-time" class="expand-link" title="View full screen">⛶</a>
  </div>

  {#if history.length > 1}
    <svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">
      <g transform="translate({padding.left}, {padding.top})">
        <!-- Grid lines -->
        {#each chartData.yTicks as tick}
          <line
            x1="0"
            y1={tick.y}
            x2={chartWidth}
            y2={tick.y}
            stroke="#3a3a5a"
            stroke-dasharray="2,2"
          />
        {/each}

        <!-- Loss lines -->
        <path d={chartData.paths.total} fill="none" stroke={colors.total} stroke-width="2" />
        <path d={chartData.paths.policy} fill="none" stroke={colors.policy} stroke-width="1.5" stroke-opacity="0.8" />
        <path d={chartData.paths.value} fill="none" stroke={colors.value} stroke-width="1.5" stroke-opacity="0.8" />

        <!-- X-axis -->
        <line x1="0" y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#666" />
        {#each chartData.xTicks as tick}
          <text
            x={tick.x}
            y={chartHeight + 20}
            text-anchor="middle"
            fill="#888"
            font-size="10"
          >
            {formatStep(tick.value)}
          </text>
        {/each}
        <text
          x={chartWidth / 2}
          y={chartHeight + 35}
          text-anchor="middle"
          fill="#666"
          font-size="11"
        >
          Step
        </text>

        <!-- Y-axis -->
        <line x1="0" y1="0" x2="0" y2={chartHeight} stroke="#666" />
        {#each chartData.yTicks as tick}
          <text
            x="-8"
            y={tick.y + 3}
            text-anchor="end"
            fill="#888"
            font-size="10"
          >
            {formatLoss(tick.value)}
          </text>
        {/each}
        <text
          x="-35"
          y={chartHeight / 2}
          text-anchor="middle"
          fill="#666"
          font-size="11"
          transform="rotate(-90, -35, {chartHeight / 2})"
        >
          Loss
        </text>
      </g>
    </svg>

    <!-- Legend -->
    <div class="legend">
      <div class="legend-item">
        <span class="legend-line" style="background: {colors.total}"></span>
        <span>Total</span>
      </div>
      <div class="legend-item">
        <span class="legend-line" style="background: {colors.policy}"></span>
        <span>Policy</span>
      </div>
      <div class="legend-item">
        <span class="legend-line" style="background: {colors.value}"></span>
        <span>Value</span>
      </div>
    </div>
  {:else}
    <p class="no-data">Waiting for training data...</p>
  {/if}
</div>

<style>
  .loss-chart {
    background: #3a3a5a;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
  }

  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  h3 {
    font-size: 0.9rem;
    color: #888;
    margin: 0;
    font-weight: normal;
  }

  .expand-link {
    color: #888;
    text-decoration: none;
    font-size: 1.1rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .expand-link:hover {
    color: #00d9ff;
    background: #4a4a6a;
  }

  svg {
    width: 100%;
    height: auto;
    display: block;
  }

  .legend {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 0.75rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    color: #aaa;
  }

  .legend-line {
    width: 16px;
    height: 3px;
    border-radius: 1px;
  }

  .no-data {
    color: #666;
    font-size: 0.85rem;
    text-align: center;
    padding: 1rem 0;
    margin: 0;
  }
</style>
