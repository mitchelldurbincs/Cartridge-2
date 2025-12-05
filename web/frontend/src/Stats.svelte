<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getStats, getModelInfo, type TrainingStats, type ModelInfo } from './lib/api';

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
  {:else if stats && stats.epoch > 0}
    <div class="stat-grid">
      <div class="stat">
        <span class="label">Epoch</span>
        <span class="value">{stats.epoch}</span>
      </div>
      <div class="stat">
        <span class="label">Total Loss</span>
        <span class="value">{formatNumber(stats.loss)}</span>
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
        <span class="label">Games Played</span>
        <span class="value">{stats.games_played}</span>
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
</style>
