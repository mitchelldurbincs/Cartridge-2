# Plan: Add Loss Visualization to Web Frontend

## Summary

Add a loss graph visualization to the web frontend that shows training loss over time, using data from the existing `stats.json` file.

## Current State

The infrastructure already supports this feature:

1. **Backend** (`trainer/src/trainer/trainer.py:269-300`):
   - `TrainerStats.history` already stores loss data points with: `step`, `total_loss`, `value_loss`, `policy_loss`, `learning_rate`
   - History is bounded to last 100 entries (`_max_history`)
   - Written to `stats.json` every `stats_interval` steps

2. **Frontend** (`web/frontend/src/Stats.svelte`):
   - Already polls `/stats` endpoint every 5 seconds
   - Already has a mini bar chart for win rate history
   - Currently shows only the latest loss values, not history

3. **API Types** (`web/frontend/src/lib/api.ts:44-59`):
   - `TrainingStats` interface missing `history` field

## Implementation Plan

### Step 1: Update TypeScript Types

**File:** `web/frontend/src/lib/api.ts`

Add `HistoryEntry` interface and `history` field to `TrainingStats`:

```typescript
export interface HistoryEntry {
  step: number;
  total_loss: number;
  value_loss: number;
  policy_loss: number;
  learning_rate: number;
}

export interface TrainingStats {
  // ... existing fields ...
  history: HistoryEntry[];  // Add this field
}
```

### Step 2: Create LossChart Component

**File:** `web/frontend/src/LossChart.svelte` (new file)

Create a dedicated chart component that:
- Takes `history: HistoryEntry[]` as a prop
- Renders an SVG line chart with:
  - X-axis: training step
  - Y-axis: loss value
  - Three lines: total_loss, policy_loss, value_loss
- Includes:
  - Legend showing which color is which loss type
  - Hover tooltips with exact values
  - Auto-scaling Y-axis based on data range
  - Responsive width

Design approach:
- Use pure SVG (no external charting library) for simplicity and bundle size
- Match existing dark theme colors from Stats.svelte
- Make it responsive with viewBox

### Step 3: Integrate Chart into Stats Panel

**File:** `web/frontend/src/Stats.svelte`

Add the LossChart below the current stats grid:
- Import and use the new LossChart component
- Only render when `history` has more than 1 data point
- Add a section header "Loss Over Time"
- Position it after the current training stats section

### Step 4: Styling

Use consistent styling with existing components:
- Background: `#3a3a5a` (matches existing chart container)
- Text colors: `#888` for labels, `#fff` for values
- Chart line colors:
  - Total loss: `#00d9ff` (cyan - primary brand color)
  - Policy loss: `#ff6b6b` (red)
  - Value loss: `#4ecdc4` (teal)

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `web/frontend/src/lib/api.ts` | Edit | Add `HistoryEntry` interface and `history` field |
| `web/frontend/src/LossChart.svelte` | Create | New SVG line chart component |
| `web/frontend/src/Stats.svelte` | Edit | Import and use LossChart component |

## Alternative Approaches Considered

1. **Separate Tab/Page**: Could add a dedicated "Charts" tab to the main navigation. However, the current UI is simple and doesn't have navigation - adding the chart inline keeps the single-page simplicity.

2. **External Charting Library**: Could use Chart.js, D3, or similar. However:
   - Adds bundle size and dependencies
   - The chart requirements are simple (line chart with 3 series)
   - SVG is sufficient and keeps the bundle light

3. **Canvas instead of SVG**: Canvas could be more performant for very large datasets, but with history capped at 100 points, SVG is perfectly adequate.

## Testing

1. Start trainer to generate some history data
2. Verify chart renders correctly with various data sizes (1 point, few points, 100 points)
3. Verify responsive behavior on different screen sizes
4. Verify polling updates chart as new data arrives

## Notes

- The history is already bounded to 100 entries by the trainer, so no frontend truncation needed
- Chart will gracefully handle edge cases (empty history, single point)
- No backend changes required - data is already being written
