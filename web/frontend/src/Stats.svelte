<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getStats, getModelInfo, getActorStats, type TrainingStats, type ModelInfo, type ActorStats } from './lib/api';
  import { STATS_POLL_INTERVAL_MS, MS_PER_SECOND } from './lib/constants';
  import LossChart from './LossChart.svelte';

  let stats: TrainingStats | null = $state(null);
  let modelInfo: ModelInfo | null = $state(null);
  let actorStats: ActorStats | null = $state(null);
  let error: string | null = $state(null);
  let pollInterval: number | undefined;

  // Training speed tracking
  let prevStats: { step: number; timestamp: number } | null = $state(null);
  let stepsPerSecond: number | null = $state(null);
  let etaSeconds: number | null = $state(null);

  // Smoothed speed (exponential moving average)
  const SPEED_SMOOTHING = 0.3; // Lower = smoother, higher = more responsive

  async function fetchData() {
    try {
      const [statsResult, modelResult, actorResult] = await Promise.all([
        getStats(),
        getModelInfo(),
        getActorStats()
      ]);

      // Calculate training speed from delta
      if (prevStats && statsResult.step > prevStats.step && statsResult.timestamp > prevStats.timestamp) {
        const stepDelta = statsResult.step - prevStats.step;
        const timeDelta = statsResult.timestamp - prevStats.timestamp;
        if (timeDelta > 0) {
          const instantSpeed = stepDelta / timeDelta;
          // Apply exponential smoothing
          if (stepsPerSecond === null) {
            stepsPerSecond = instantSpeed;
          } else {
            stepsPerSecond = SPEED_SMOOTHING * instantSpeed + (1 - SPEED_SMOOTHING) * stepsPerSecond;
          }

          // Calculate ETA
          if (stepsPerSecond > 0 && statsResult.total_steps > 0) {
            const remainingSteps = statsResult.total_steps - statsResult.step;
            etaSeconds = remainingSteps / stepsPerSecond;
          }
        }
      }

      // Store current stats for next comparison
      if (statsResult.step > 0) {
        prevStats = { step: statsResult.step, timestamp: statsResult.timestamp };
      }

      stats = statsResult;
      modelInfo = modelResult;
      actorStats = actorResult;
      error = null;
    } catch (e) {
      error = 'Failed to fetch data';
    }
  }

  onMount(() => {
    fetchData();
    pollInterval = setInterval(fetchData, STATS_POLL_INTERVAL_MS);
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
    const date = new Date(ts * MS_PER_SECOND);
    return date.toLocaleTimeString();
  }

  function formatTimeAgo(ts: number | null | undefined): string {
    if (!ts) return '-';
    const now = Date.now() / MS_PER_SECOND;
    const diff = now - ts;
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  function formatSpeed(speed: number | null): string {
    if (speed === null || speed <= 0) return '-';
    if (speed >= 100) return `${Math.round(speed)} steps/s`;
    if (speed >= 10) return `${speed.toFixed(1)} steps/s`;
    if (speed >= 1) return `${speed.toFixed(2)} steps/s`;
    return `${speed.toFixed(3)} steps/s`;
  }

  function formatEta(seconds: number | null): string {
    if (seconds === null || seconds <= 0) return '-';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) {
      const mins = Math.floor(seconds / 60);
      const secs = Math.round(seconds % 60);
      return `${mins}m ${secs}s`;
    }
    if (seconds < 86400) {
      const hours = Math.floor(seconds / 3600);
      const mins = Math.round((seconds % 3600) / 60);
      return `${hours}h ${mins}m`;
    }
    const days = Math.floor(seconds / 86400);
    const hours = Math.round((seconds % 86400) / 3600);
    return `${days}d ${hours}h`;
  }

  function getProgressPercent(step: number, total: number): number {
    if (total <= 0) return 0;
    return Math.min(100, (step / total) * 100);
  }

  function formatGradNorm(norm: number | null | undefined): string {
    if (norm == null) return '-';
    if (norm >= 100) return norm.toFixed(0);
    if (norm >= 10) return norm.toFixed(1);
    return norm.toFixed(2);
  }

  function getGradNormColor(norm: number | null | undefined): string {
    if (norm == null) return '#888';
    // Green = stable (< 1), Yellow = moderate (1-10), Red = high (> 10)
    if (norm < 1) return '#4f4';
    if (norm < 10) return '#fa0';
    return '#f66';
  }

  // Get the latest gradient norm from history
  let latestGradNorm = $derived.by(() => {
    if (!stats?.history || stats.history.length === 0) return null;
    // Find the most recent entry with grad_norm
    for (let i = stats.history.length - 1; i >= 0; i--) {
      if (stats.history[i].grad_norm != null) {
        return stats.history[i].grad_norm;
      }
    }
    return null;
  });

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
    <!-- Progress Bar -->
    {#if stats.total_steps > 0}
      <div class="progress-container">
        <div class="progress-bar">
          <div
            class="progress-fill"
            style="width: {getProgressPercent(stats.step, stats.total_steps)}%"
          ></div>
        </div>
        <div class="progress-text">
          <span>{stats.step.toLocaleString()} / {stats.total_steps.toLocaleString()} steps</span>
          <span>{getProgressPercent(stats.step, stats.total_steps).toFixed(1)}%</span>
        </div>
      </div>
    {/if}

    <div class="stat-grid">
      <div class="stat">
        <span class="label">Speed</span>
        <span class="value speed-value">{formatSpeed(stepsPerSecond)}</span>
      </div>
      <div class="stat">
        <span class="label">ETA</span>
        <span class="value eta-value">{formatEta(etaSeconds)}</span>
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
        <span class="label">Learning Rate</span>
        <span class="value">{formatNumber(stats.learning_rate)}</span>
      </div>
      <div class="stat">
        <span class="label">Grad Norm</span>
        <span class="value" style="color: {getGradNormColor(latestGradNorm)}">{formatGradNorm(latestGradNorm)}</span>
      </div>
      <div class="stat">
        <span class="label">Replay Buffer</span>
        <span class="value">{stats.replay_buffer_size.toLocaleString()}</span>
      </div>
      <div class="stat">
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

  <!-- Actor Self-Play Stats Section -->
  {#if actorStats && actorStats.episodes_completed > 0}
    <hr class="divider" />
    <h2>Self-Play Stats</h2>
    <div class="stat-grid">
      <div class="stat">
        <span class="label">Episodes</span>
        <span class="value">{actorStats.episodes_completed.toLocaleString()}</span>
      </div>
      <div class="stat">
        <span class="label">Ep/sec</span>
        <span class="value speed-value">{actorStats.episodes_per_second.toFixed(2)}</span>
      </div>
      <div class="stat">
        <span class="label">Avg Length</span>
        <span class="value">{actorStats.avg_episode_length.toFixed(1)}</span>
      </div>
      <div class="stat">
        <span class="label">MCTS Inference</span>
        <span class="value">{(actorStats.mcts_avg_inference_us / 1000).toFixed(1)}ms</span>
      </div>
    </div>

    <!-- Outcome Distribution -->
    {#if actorStats.player1_wins + actorStats.player2_wins + actorStats.draws > 0}
      {@const total = actorStats.player1_wins + actorStats.player2_wins + actorStats.draws}
      {@const p1Pct = (actorStats.player1_wins / total) * 100}
      {@const p2Pct = (actorStats.player2_wins / total) * 100}
      {@const drawPct = (actorStats.draws / total) * 100}
      <div class="outcome-bar">
        <div class="outcome-segment p1-wins" style="width: {p1Pct}%" title="P1 Wins: {actorStats.player1_wins} ({p1Pct.toFixed(1)}%)"></div>
        <div class="outcome-segment draws" style="width: {drawPct}%" title="Draws: {actorStats.draws} ({drawPct.toFixed(1)}%)"></div>
        <div class="outcome-segment p2-wins" style="width: {p2Pct}%" title="P2 Wins: {actorStats.player2_wins} ({p2Pct.toFixed(1)}%)"></div>
      </div>
    {/if}
    <div class="outcome-legend">
      <span class="legend-item"><span class="dot p1"></span>P1: {actorStats.player1_wins}</span>
      <span class="legend-item"><span class="dot draw"></span>Draw: {actorStats.draws}</span>
      <span class="legend-item"><span class="dot p2"></span>P2: {actorStats.player2_wins}</span>
    </div>
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

  /* Progress bar styles */
  .progress-container {
    margin-bottom: 1rem;
  }

  .progress-bar {
    height: 8px;
    background: #3a3a5a;
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00d9ff, #00ff88);
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .progress-text {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.25rem;
  }

  .speed-value {
    color: #00d9ff;
  }

  .eta-value {
    color: #00ff88;
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

  /* Actor stats outcome bar */
  .outcome-bar {
    display: flex;
    height: 12px;
    border-radius: 6px;
    overflow: hidden;
    background: #3a3a5a;
    margin-top: 1rem;
  }

  .outcome-segment {
    transition: width 0.3s ease;
  }

  .outcome-segment.p1-wins {
    background: #4f4;
  }

  .outcome-segment.draws {
    background: #888;
  }

  .outcome-segment.p2-wins {
    background: #f66;
  }

  .outcome-legend {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #aaa;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .dot.p1 {
    background: #4f4;
  }

  .dot.draw {
    background: #888;
  }

  .dot.p2 {
    background: #f66;
  }
</style>
