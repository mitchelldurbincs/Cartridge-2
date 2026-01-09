# Chart Abstraction Plan

## Problem Statement

`LossChart.svelte` and `LossOverTimePage.svelte` contain nearly identical implementations of charting logic. This violates DRY and makes maintenance error-prone (e.g., fixing a bug in one place but not the other).

### Duplicated Code

| Function/Constant | LossChart.svelte | LossOverTimePage.svelte |
|-------------------|------------------|-------------------------|
| `colors` object | Lines 19-23 | Lines 30-35 |
| `generateTicks()` | Lines 90-94 | Lines 160-164 |
| `formatStep()` | Lines 97-100 | Lines 166-171 |
| `formatLoss()` | Lines 102-106 | Lines 173-178 |
| `getChartData()` | Lines 26-87 | Lines 72-158 |
| SVG axis rendering | Lines 138-183 | Lines 346-391 |
| Legend markup | Lines 188-201 | Lines 473-492 |

### Differences Between Components

| Feature | LossChart | LossOverTimePage |
|---------|-----------|------------------|
| Size | Fixed 400x200 | Responsive to window |
| Interactivity | None | Hover tooltip, crosshair |
| Data points | Never shown | Shown when ≤50 points |
| Range filter | None | Last 100/500/1000/All |
| Rolling average | None | Optional Avg100 line |
| Expand link | Yes (navigates to full page) | No (is the full page) |
| Polling | None (receives data as prop) | Yes (fetches own data) |

---

## Proposed Architecture

### Option A: Shared Utilities + Two Components (Recommended)

Extract pure functions to `lib/chart.ts`, keep two separate Svelte components that import shared utilities.

```
web/frontend/src/
├── lib/
│   ├── api.ts           (existing)
│   ├── constants.ts     (existing)
│   └── chart.ts         (NEW - shared chart utilities)
├── LossChart.svelte     (simplified, imports from chart.ts)
└── LossOverTimePage.svelte (simplified, imports from chart.ts)
```

**Pros:**
- Minimal refactoring risk
- Each component retains control over its specific features
- Easy to test utility functions in isolation

**Cons:**
- SVG/markup still duplicated (acceptable given different features)

### Option B: Configurable Base Chart Component

Create a single `<BaseChart>` component with props for all variations.

```
web/frontend/src/
├── lib/
│   └── chart.ts         (shared utilities)
├── BaseChart.svelte     (NEW - configurable chart)
├── LossChart.svelte     (thin wrapper around BaseChart)
└── LossOverTimePage.svelte (thin wrapper around BaseChart)
```

**Pros:**
- Maximum code reuse
- Single source of truth for chart rendering

**Cons:**
- Complex prop interface
- Risk of creating a "god component"
- Harder to add component-specific features later

### Recommendation

**Go with Option A.** The two charts have fundamentally different interaction models (passive vs. interactive). Forcing them into a single component would create complexity. Extracting utilities gives 80% of the benefit with 20% of the risk.

---

## Implementation Plan

### Phase 1: Create `lib/chart.ts`

Create a new file with shared types, constants, and pure functions.

```typescript
// web/frontend/src/lib/chart.ts

import { LARGE_NUMBER_THRESHOLD } from './constants';
import type { HistoryEntry } from './api';

// ============================================================================
// Types
// ============================================================================

export interface ChartPoint {
  x: number;
  y: number;
  step: number;
  value: number;
}

export interface ChartTick {
  value: number;
  x?: number;
  y?: number;
}

export interface ChartPaths {
  total: string;
  policy: string;
  value: string;
  avg100?: string;
}

export interface ChartData {
  xTicks: ChartTick[];
  yTicks: ChartTick[];
  paths: ChartPaths;
  points?: {
    total: ChartPoint[];
    policy: ChartPoint[];
    value: ChartPoint[];
  };
}

export interface ChartDimensions {
  width: number;
  height: number;
  padding: { top: number; right: number; bottom: number; left: number };
}

// ============================================================================
// Constants
// ============================================================================

export const CHART_COLORS = {
  total: '#00d9ff',   // Cyan - primary brand color
  policy: '#ff6b6b', // Red
  value: '#4ecdc4',  // Teal
  avg100: '#ffd700', // Gold
} as const;

// ============================================================================
// Formatting Functions
// ============================================================================

export function formatStep(value: number): string {
  const million = LARGE_NUMBER_THRESHOLD * LARGE_NUMBER_THRESHOLD;
  if (value >= million) return `${(value / million).toFixed(1)}M`;
  if (value >= LARGE_NUMBER_THRESHOLD) return `${(value / LARGE_NUMBER_THRESHOLD).toFixed(1)}k`;
  return Math.round(value).toString();
}

export function formatLoss(value: number): string {
  if (value < 0.001) return value.toExponential(2);
  if (value < 0.01) return value.toFixed(4);
  if (value < 1) return value.toFixed(3);
  return value.toFixed(2);
}

// ============================================================================
// Scale & Tick Functions
// ============================================================================

export function generateTicks(min: number, max: number, count: number): number[] {
  if (min === max) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + step * i);
}

export function createScales(
  data: HistoryEntry[],
  chartWidth: number,
  chartHeight: number,
  includeAvg100Data?: { step: number; avg: number }[]
): {
  xScale: (step: number) => number;
  yScale: (loss: number) => number;
  minStep: number;
  maxStep: number;
  yMin: number;
  yMax: number;
} {
  const steps = data.map(d => d.step);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);

  const allLosses = data.flatMap(d => [d.total_loss, d.policy_loss, d.value_loss]);
  if (includeAvg100Data && includeAvg100Data.length > 0) {
    allLosses.push(...includeAvg100Data.map(d => d.avg));
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

  return { xScale, yScale, minStep, maxStep, yMin, yMax };
}

// ============================================================================
// Path Generation
// ============================================================================

export function makePath(
  data: HistoryEntry[],
  key: 'total_loss' | 'policy_loss' | 'value_loss',
  xScale: (step: number) => number,
  yScale: (loss: number) => number
): string {
  const sorted = [...data].sort((a, b) => a.step - b.step);
  return sorted.map((d, i) => {
    const x = xScale(d.step);
    const y = yScale(d[key]);
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');
}

export function makeAvg100Path(
  avg100Data: { step: number; avg: number }[],
  xScale: (step: number) => number,
  yScale: (loss: number) => number
): string {
  if (avg100Data.length === 0) return '';
  return avg100Data.map((d, i) => {
    const x = xScale(d.step);
    const y = yScale(d.avg);
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');
}

// ============================================================================
// Rolling Average
// ============================================================================

export function computeRollingAverage(
  data: HistoryEntry[],
  window: number = 100
): { step: number; avg: number }[] {
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

// ============================================================================
// Main Chart Data Builder
// ============================================================================

export interface BuildChartDataOptions {
  data: HistoryEntry[];
  chartWidth: number;
  chartHeight: number;
  xTickCount?: number;
  yTickCount?: number;
  includePoints?: boolean;
  includeAvg100?: boolean;
}

export function buildChartData(options: BuildChartDataOptions): ChartData {
  const {
    data,
    chartWidth,
    chartHeight,
    xTickCount = 5,
    yTickCount = 5,
    includePoints = false,
    includeAvg100 = false,
  } = options;

  if (data.length === 0) {
    return {
      xTicks: [],
      yTicks: [],
      paths: { total: '', policy: '', value: '' },
      points: includePoints ? { total: [], policy: [], value: [] } : undefined,
    };
  }

  const sorted = [...data].sort((a, b) => a.step - b.step);
  const avg100Data = includeAvg100 ? computeRollingAverage(sorted) : [];

  const { xScale, yScale, minStep, maxStep, yMin, yMax } = createScales(
    sorted,
    chartWidth,
    chartHeight,
    includeAvg100 ? avg100Data : undefined
  );

  const paths: ChartPaths = {
    total: makePath(sorted, 'total_loss', xScale, yScale),
    policy: makePath(sorted, 'policy_loss', xScale, yScale),
    value: makePath(sorted, 'value_loss', xScale, yScale),
  };

  if (includeAvg100) {
    paths.avg100 = makeAvg100Path(avg100Data, xScale, yScale);
  }

  const xTicks = generateTicks(minStep, maxStep, xTickCount).map(v => ({
    value: v,
    x: xScale(v),
  }));

  const yTicks = generateTicks(yMin, yMax, yTickCount).map(v => ({
    value: v,
    y: yScale(v),
  }));

  let points: ChartData['points'];
  if (includePoints) {
    const makePoints = (key: 'total_loss' | 'policy_loss' | 'value_loss') =>
      sorted.map(d => ({
        x: xScale(d.step),
        y: yScale(d[key]),
        step: d.step,
        value: d[key],
      }));

    points = {
      total: makePoints('total_loss'),
      policy: makePoints('policy_loss'),
      value: makePoints('value_loss'),
    };
  }

  return { xTicks, yTicks, paths, points };
}
```

### Phase 2: Refactor `LossChart.svelte`

Simplify to use shared utilities:

```svelte
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

<!-- Template remains similar but uses CHART_COLORS constant -->
```

### Phase 3: Refactor `LossOverTimePage.svelte`

Simplify to use shared utilities while keeping interactive features:

```svelte
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getStats, type HistoryEntry } from './lib/api';
  import { STATS_POLL_INTERVAL_MS } from './lib/constants';
  import {
    CHART_COLORS,
    formatStep,
    formatLoss,
    buildChartData,
    type ChartData,
  } from './lib/chart';

  // ... state variables remain the same ...

  // Use shared builder with additional options
  let chartData = $derived.by(() => {
    return buildChartData({
      data: filteredHistory,
      chartWidth,
      chartHeight,
      xTickCount: 10,
      yTickCount: 8,
      includePoints: true,
      includeAvg100: showAvg100,
    });
  });

  // ... hover logic and interactive features remain component-specific ...
</script>
```

### Phase 4: Update `constants.ts`

Move `LARGE_NUMBER_THRESHOLD` usage note:

```typescript
/** Threshold for formatting large numbers with 'k' suffix (used by chart.ts) */
export const LARGE_NUMBER_THRESHOLD = 1000;
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `src/lib/chart.ts` | **CREATE** - ~180 lines of shared utilities |
| `src/lib/constants.ts` | No change (already has LARGE_NUMBER_THRESHOLD) |
| `src/LossChart.svelte` | **MODIFY** - Remove ~80 lines, import from chart.ts |
| `src/LossOverTimePage.svelte` | **MODIFY** - Remove ~100 lines, import from chart.ts |

### Lines of Code Impact

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| LossChart.svelte | 278 | ~200 | -78 |
| LossOverTimePage.svelte | 701 | ~580 | -121 |
| chart.ts (new) | 0 | ~180 | +180 |
| **Net change** | 979 | ~960 | **-19** |

The net line reduction is small, but the benefit is:
- Single source of truth for chart math
- Easier testing of pure functions
- Consistent formatting across both charts
- Bug fixes apply to both charts automatically

---

## Testing Checklist

After refactoring, verify:

- [ ] `LossChart` renders correctly in Stats panel
- [ ] `LossChart` updates when new training data arrives
- [ ] `LossOverTimePage` renders correctly at `/loss-over-time`
- [ ] Range filter (Last 100/500/1000/All) works
- [ ] Avg100 toggle works
- [ ] Hover tooltip shows correct values
- [ ] Crosshair tracks mouse correctly
- [ ] Data points appear when ≤50 entries
- [ ] Responsive sizing works on window resize
- [ ] Expand link in small chart navigates to full page
- [ ] Back link in full page returns to game
- [ ] No TypeScript errors: `npm run check`
- [ ] Build succeeds: `npm run build`

---

## Migration Notes

### Breaking Changes

None - this is an internal refactor with no API changes.

### Rollback Plan

If issues arise, revert the commit. The refactor should be done in a single atomic commit to make rollback trivial.

---

## Future Enhancements (Out of Scope)

These could build on this foundation later:

1. **Extract SVG axis components** - `<XAxis>`, `<YAxis>` Svelte components
2. **Add chart animation** - Smooth transitions when data updates
3. **Support log scale** - For loss values that span orders of magnitude
4. **Export as image** - Download chart as PNG/SVG
5. **Zoom/pan** - For very long training runs
