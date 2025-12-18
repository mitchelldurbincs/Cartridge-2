/**
 * Shared utility functions for the Cartridge2 frontend
 */

// ============================================================================
// Chart Colors
// ============================================================================

export const chartColors = {
  total: '#00d9ff',   // Cyan - primary brand color
  policy: '#ff6b6b', // Red
  value: '#4ecdc4',  // Teal
};

// ============================================================================
// Number Formatting
// ============================================================================

/**
 * Format a training step number for display (e.g., 1500 -> "1.5k", 1500000 -> "1.5M")
 */
export function formatStep(value: number): string {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return Math.round(value).toString();
}

/**
 * Format a loss value for display with appropriate precision
 */
export function formatLoss(value: number): string {
  if (value < 0.001) return value.toExponential(2);
  if (value < 0.01) return value.toFixed(4);
  if (value < 1) return value.toFixed(3);
  return value.toFixed(2);
}

/**
 * Format a number with fixed decimal places, returning '-' for undefined/zero
 */
export function formatNumber(n: number | undefined): string {
  if (n === undefined || n === 0) return '-';
  return n.toFixed(4);
}

/**
 * Format a number as a percentage (e.g., 0.75 -> "75.0%")
 */
export function formatPercent(n: number | undefined): string {
  if (n === undefined) return '-';
  return `${(n * 100).toFixed(1)}%`;
}

// ============================================================================
// Time Formatting
// ============================================================================

/**
 * Format a Unix timestamp as a localized time string
 */
export function formatTimestamp(ts: number | null | undefined): string {
  if (!ts) return '-';
  const date = new Date(ts * 1000);
  return date.toLocaleTimeString();
}

/**
 * Format a Unix timestamp as a relative time string (e.g., "5m ago")
 */
export function formatTimeAgo(ts: number | null | undefined): string {
  if (!ts) return '-';
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

// ============================================================================
// Chart Utilities
// ============================================================================

/**
 * Generate evenly spaced tick values between min and max
 */
export function generateTicks(min: number, max: number, count: number): number[] {
  if (min === max) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + step * i);
}

/**
 * Get a color based on win rate (green for good, orange for okay, red for poor)
 */
export function getWinRateColor(winRate: number): string {
  if (winRate >= 0.7) return '#4f4';  // Green - good
  if (winRate >= 0.5) return '#fa0';  // Orange - okay
  return '#f66';  // Red - poor
}
