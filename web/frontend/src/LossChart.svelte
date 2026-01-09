<script lang="ts">
  import type { HistoryEntry } from './lib/api';
  import { CHART_COLORS, formatStep, formatLoss, buildChartData } from './lib/chart';

  interface Props {
    history: HistoryEntry[];
  }

  let { history }: Props = $props();

  // Chart dimensions (fixed for small chart)
  const width = 400;
  const height = 200;
  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Reactive chart data using shared builder
  let chartData = $derived(
    buildChartData({
      data: history,
      chartWidth,
      chartHeight,
      xTickCount: 5,
      yTickCount: 5,
    })
  );
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
        <path d={chartData.paths.total} fill="none" stroke={CHART_COLORS.total} stroke-width="2" />
        <path d={chartData.paths.policy} fill="none" stroke={CHART_COLORS.policy} stroke-width="1.5" stroke-opacity="0.8" />
        <path d={chartData.paths.value} fill="none" stroke={CHART_COLORS.value} stroke-width="1.5" stroke-opacity="0.8" />

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
        <span class="legend-line" style="background: {CHART_COLORS.total}"></span>
        <span>Total</span>
      </div>
      <div class="legend-item">
        <span class="legend-line" style="background: {CHART_COLORS.policy}"></span>
        <span>Policy</span>
      </div>
      <div class="legend-item">
        <span class="legend-line" style="background: {CHART_COLORS.value}"></span>
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
