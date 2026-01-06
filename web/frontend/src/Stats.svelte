<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getStats, getModelInfo, type TrainingStats, type ModelInfo } from './lib/api';
  import LossChart from './LossChart.svelte';

  let stats: TrainingStats | null = $state(null);
  let modelInfo: ModelInfo | null = $state(null);
  let error: string | null = $state(null);
  let pollInterval: number | undefined;

  async function fetchData() {
    try {
      const [statsResult, modelResult] = await Promise.all([
        getStats(),
        getModelInfo()
      ]);
      stats = statsResult;
      modelInfo = modelResult;
      error = null;
    } catch (e) {
      error = 'Failed to fetch data';
    }
  }

  onMount(() => {
    fetchData();
    // Poll every 5 seconds
    pollInterval = setInterval(fetchData, 5000);
  });

  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
  });

  function formatNumber(n: number | undefined): string {
    if (n === undefined || n === 0) return '-';
    return n.toFixed(4);
  }

  function formatPercent(n: number | undefined | null): string {
    if (n == null) return '-';
    return `${(n * 100).toFixed(1)}%`;
  }

  function formatTimestamp(ts: number | null | undefined): string {
    if (!ts) return '-';
    const date = new Date(ts * 1000);
    return date.toLocaleTimeString();
  }

  function formatTimeAgo(ts: number | null | undefined): string {
    if (!ts) return '-';
    const now = Date.now() / 1000;
    const diff = now - ts;
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  function getWinRateColor(winRate: number | null | undefined): string {
    if (winRate == null) return '#888';  // Gray - no data
    if (winRate >= 0.7) return '#4f4';  // Green - good
    if (winRate >= 0.5) return '#fa0';  // Orange - okay
    return '#f66';  // Red - poor
  }
</script>

<div class="stats-panel">
  <!-- Model Info Section -->
  <h2>Bot Model</h2>
  {#if modelInfo}
    <div class="model-status" class:loaded={modelInfo.loaded} class:no-model={!modelInfo.loaded}>
      <span class="status-indicator"></span>
      <span class="status-text">{modelInfo.status}</span>
    </div>
    {#if modelInfo.loaded}
      <div class="stat-grid model-grid">
        {#if modelInfo.training_step}
          <div class="stat">
            <span class="label">Training Step</span>
            <span class="value">{modelInfo.training_step.toLocaleString()}</span>
          </div>
        {/if}
        <div class="stat">
          <span class="label">Model Updated</span>
          <span class="value">{formatTimeAgo(modelInfo.file_modified)}</span>
        </div>
        <div class="stat">
          <span class="label">Loaded At</span>
          <span class="value">{formatTimestamp(modelInfo.loaded_at)}</span>
        </div>
      </div>
    {/if}
  {/if}

  <hr class="divider" />

  <!-- Training Stats Section -->
  <h2>Training Stats</h2>

  {#if error}
    <p class="error">{error}</p>
  {:else if stats && stats.step > 0}
    <div class="stat-grid">
      <div class="stat">
        <span class="label">Step</span>
        <span class="value">{stats.step} / {stats.total_steps || '-'}</span>
      </div>
      <div class="stat">
        <span class="label">Total Loss</span>
        <span class="value">{formatNumber(stats.total_loss)}</span>
      </div>
      <div class="stat">
        <span class="label">Policy Loss</span>
        <span class="value">{formatNumber(stats.policy_loss)}</span>
      </div>
      <div class="stat">
        <span class="label">Value Loss</span>
        <span class="value">{formatNumber(stats.value_loss)}</span>
      </div>
      <div class="stat">
        <span class="label">Replay Buffer</span>
        <span class="value">{stats.replay_buffer_size.toLocaleString()}</span>
      </div>
      <div class="stat">
        <span class="label">Learning Rate</span>
        <span class="value">{formatNumber(stats.learning_rate)}</span>
      </div>
      <div class="stat full-width">
        <span class="label">Last Update</span>
        <span class="value">{formatTimestamp(stats.timestamp)}</span>
      </div>
    </div>

    <!-- Loss Chart -->
    {#if stats.history && stats.history.length > 0}
      <LossChart history={stats.history} />
    {/if}

    <!-- Evaluation Section -->
    {#if stats.last_eval}
      <hr class="divider" />
      <h2>Model Evaluation</h2>

      <!-- Main Evaluation Display -->
      <div class="eval-cards">
        <!-- VS Best Model -->
        <div class="eval-card" class:new-best={stats.last_eval.became_new_best}>
          <div class="eval-card-header">
            <span class="opponent-label">vs Best Model</span>
            {#if stats.last_eval.opponent_iteration}
              <span class="opponent-iter">(iter {stats.last_eval.opponent_iteration})</span>
            {/if}
          </div>
          <div class="win-rate-value" style="color: {getWinRateColor(stats.last_eval.win_rate)}">
            {formatPercent(stats.last_eval.win_rate)}
          </div>
          {#if stats.last_eval.became_new_best}
            <div class="new-best-badge">🎉 New Best!</div>
          {/if}
          <div class="eval-details">
            <span>Draw: {formatPercent(stats.last_eval.draw_rate)}</span>
            <span>Loss: {formatPercent(stats.last_eval.loss_rate)}</span>
          </div>
        </div>

        <!-- VS Random (if available) -->
        {#if stats.last_eval.vs_random_win_rate != null}
          <div class="eval-card">
            <div class="eval-card-header">
              <span class="opponent-label">vs Random</span>
            </div>
            <div class="win-rate-value" style="color: {getWinRateColor(stats.last_eval.vs_random_win_rate)}">
              {formatPercent(stats.last_eval.vs_random_win_rate)}
            </div>
            <div class="eval-details">
              <span>Draw: {formatPercent(stats.last_eval.vs_random_draw_rate)}</span>
            </div>
          </div>
        {/if}
      </div>

      <div class="stat-grid">
        <div class="stat">
          <span class="label">Iteration</span>
          <span class="value">{stats.last_eval.current_iteration || '-'}</span>
        </div>
        <div class="stat">
          <span class="label">Games per Eval</span>
          <span class="value">{stats.last_eval.games_played}</span>
        </div>
      </div>

      <!-- Win Rate History Chart (vs Best) -->
      {#if stats.eval_history && stats.eval_history.length > 1}
        <div class="chart-container">
          <h3>Win Rate vs Best Over Time</h3>
          <div class="mini-chart">
            {#each stats.eval_history as evalPoint}
              <div
                class="chart-bar"
                class:new-best-bar={evalPoint.became_new_best}
                style="height: {evalPoint.win_rate * 100}%; background: {evalPoint.became_new_best ? '#ffd700' : getWinRateColor(evalPoint.win_rate)}"
                title="Iter {evalPoint.current_iteration}: {formatPercent(evalPoint.win_rate)}{evalPoint.became_new_best ? ' 🏆' : ''}"
              ></div>
            {/each}
          </div>
          <div class="chart-labels">
            <span>0%</span>
            <span>55% threshold</span>
            <span>100%</span>
          </div>
        </div>
      {/if}
    {:else}
      <hr class="divider" />
      <h2>Model Evaluation</h2>
      <p class="no-data">No evaluation data yet.</p>
      <p class="hint">Evaluation runs automatically during training.</p>
    {/if}
  {:else}
    <p class="no-data">No training data yet.</p>
    <p class="hint">Start the Python trainer to see stats here.</p>
  {/if}
</div>

<style>
  .stats-panel {
    background: #2a2a4a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: left;
  }

  h2 {
    margin: 0 0 1rem 0;
    color: #00d9ff;
    font-size: 1.2rem;
  }

  .divider {
    border: none;
    border-top: 1px solid #3a3a5a;
    margin: 1.5rem 0;
  }

  .model-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    border-radius: 8px;
    background: #3a3a5a;
    margin-bottom: 0.75rem;
  }

  .model-status.loaded {
    background: #1a4a2a;
  }

  .model-status.no-model {
    background: #4a3a1a;
  }

  .status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #888;
  }

  .model-status.loaded .status-indicator {
    background: #4f4;
  }

  .model-status.no-model .status-indicator {
    background: #fa0;
  }

  .status-text {
    font-size: 0.9rem;
    color: #fff;
  }

  .model-grid {
    margin-bottom: 0;
  }

  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }

  .stat {
    background: #3a3a5a;
    padding: 0.75rem;
    border-radius: 8px;
  }

  .stat.full-width {
    grid-column: span 2;
  }

  .label {
    display: block;
    font-size: 0.75rem;
    color: #888;
    margin-bottom: 0.25rem;
  }

  .value {
    font-size: 1.1rem;
    font-weight: bold;
    color: #fff;
  }

  .error {
    color: #f66;
  }

  .no-data {
    color: #888;
    margin-bottom: 0.5rem;
  }

  .hint {
    font-size: 0.85rem;
    color: #666;
  }

  /* Evaluation styles */
  .eval-cards {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .eval-card {
    background: #3a3a5a;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
  }

  .eval-card.new-best {
    background: linear-gradient(135deg, #2a4a2a, #3a5a3a);
    border: 1px solid #4f4;
  }

  .eval-card-header {
    margin-bottom: 0.5rem;
  }

  .opponent-label {
    font-size: 0.85rem;
    color: #00d9ff;
    font-weight: bold;
  }

  .opponent-iter {
    font-size: 0.75rem;
    color: #888;
    margin-left: 0.25rem;
  }

  .win-rate-value {
    font-size: 2rem;
    font-weight: bold;
    line-height: 1;
    margin: 0.5rem 0;
  }

  .new-best-badge {
    font-size: 0.85rem;
    color: #4f4;
    margin-bottom: 0.5rem;
  }

  .eval-details {
    display: flex;
    justify-content: center;
    gap: 1rem;
    font-size: 0.75rem;
    color: #888;
  }

  .new-best-bar {
    border: 1px solid #ffd700;
  }

  /* Mini chart styles */
  .chart-container {
    margin-top: 1rem;
  }

  .chart-container h3 {
    font-size: 0.9rem;
    color: #888;
    margin: 0 0 0.5rem 0;
    font-weight: normal;
  }

  .mini-chart {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 60px;
    padding: 0.5rem;
    background: #3a3a5a;
    border-radius: 8px;
  }

  .chart-bar {
    flex: 1;
    min-width: 4px;
    max-width: 20px;
    border-radius: 2px 2px 0 0;
    transition: height 0.3s ease;
  }

  .chart-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #666;
    margin-top: 0.25rem;
  }
</style>
