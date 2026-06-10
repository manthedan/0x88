/**
 * Dependency-free SVG chart builders for the arena/analysis pages: a small
 * multi-series line chart (per-game eval / move time / NPS series) and a
 * horizontal bar chart (live PUCT-root visit distributions). Pure string
 * builders so they stay Node-testable; callers own layout and live updates.
 * See docs/arena_analysis_roadmap.md stage 2.
 */

export interface ChartPoint { x: number; y: number; }
export interface ChartSeries { label: string; color: string; points: ChartPoint[]; }

export interface LineChartOptions {
  width?: number;
  height?: number;
  yMin?: number;
  yMax?: number;
  /** Tick label formatter for the y axis. */
  formatY?: (value: number) => string;
  /** Optional horizontal reference line (e.g. 0.5 for the eval midline). */
  midline?: number;
}

const FONT = 'font-family="ui-monospace,monospace" font-size="9"';

function escapeXml(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

/**
 * Multi-series line chart. X spans the union of series x ranges; y spans
 * [yMin, yMax] when given, else the data range padded 5%. Returns '' when no
 * series has at least one point.
 */
export function lineChartSvg(series: readonly ChartSeries[], options: LineChartOptions = {}): string {
  const drawn = series.filter((entry) => entry.points.length > 0);
  if (!drawn.length) return '';
  const width = options.width ?? 360;
  const height = options.height ?? 90;
  const pad = { left: 36, right: 8, top: 6, bottom: 14 };
  const xs = drawn.flatMap((entry) => entry.points.map((point) => point.x));
  const ys = drawn.flatMap((entry) => entry.points.map((point) => point.y));
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = options.yMin ?? Math.min(...ys);
  let yMax = options.yMax ?? Math.max(...ys);
  if (yMax === yMin) { yMax += 1; yMin -= 1; }
  if (options.yMin === undefined && options.yMax === undefined) {
    const padY = (yMax - yMin) * 0.05;
    yMin -= padY;
    yMax += padY;
  }
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const px = (x: number) => pad.left + (xMax === xMin ? plotW / 2 : ((x - xMin) / (xMax - xMin)) * plotW);
  const py = (y: number) => pad.top + (1 - (Math.min(yMax, Math.max(yMin, y)) - yMin) / (yMax - yMin)) * plotH;
  const formatY = options.formatY ?? ((value: number) => String(round2(value)));

  const parts: string[] = [];
  parts.push(`<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" role="img">`);
  // y ticks: min / mid / max
  for (const tick of [yMin, (yMin + yMax) / 2, yMax]) {
    const y = py(tick);
    parts.push(`<line x1="${pad.left}" y1="${round2(y)}" x2="${width - pad.right}" y2="${round2(y)}" stroke="#00000018" stroke-width="1"/>`);
    parts.push(`<text x="${pad.left - 3}" y="${round2(y + 3)}" text-anchor="end" fill="#777" ${FONT}>${escapeXml(formatY(tick))}</text>`);
  }
  if (options.midline !== undefined && options.midline > yMin && options.midline < yMax) {
    const y = py(options.midline);
    parts.push(`<line x1="${pad.left}" y1="${round2(y)}" x2="${width - pad.right}" y2="${round2(y)}" stroke="#00000033" stroke-dasharray="3 3" stroke-width="1"/>`);
  }
  // x extents
  parts.push(`<text x="${pad.left}" y="${height - 3}" fill="#777" ${FONT}>${round2(xMin)}</text>`);
  parts.push(`<text x="${width - pad.right}" y="${height - 3}" text-anchor="end" fill="#777" ${FONT}>${round2(xMax)}</text>`);
  for (const entry of drawn) {
    const coords = entry.points.map((point) => `${round2(px(point.x))},${round2(py(point.y))}`);
    if (coords.length === 1) {
      const [coord] = coords;
      const [cx, cy] = coord.split(',');
      parts.push(`<circle cx="${cx}" cy="${cy}" r="2" fill="${entry.color}"/>`);
    } else {
      parts.push(`<polyline points="${coords.join(' ')}" fill="none" stroke="${entry.color}" stroke-width="1.5"/>`);
    }
  }
  parts.push('</svg>');
  return parts.join('');
}

export interface BarItem {
  label: string;
  value: number;
  /** Secondary text rendered after the value (e.g. Q / prior). */
  detail?: string;
  color?: string;
}

export interface BarChartOptions {
  width?: number;
  /** Bar row height including spacing. */
  rowHeight?: number;
  maxValue?: number;
}

/** Horizontal bar chart; bar lengths scale to the max value. '' when empty. */
export function hBarChartSvg(items: readonly BarItem[], options: BarChartOptions = {}): string {
  if (!items.length) return '';
  const width = options.width ?? 360;
  const rowHeight = options.rowHeight ?? 14;
  const labelW = 44;
  const valueW = 120;
  const barW = width - labelW - valueW;
  const maxValue = options.maxValue ?? Math.max(...items.map((item) => item.value), 1);
  const height = items.length * rowHeight + 2;
  const parts: string[] = [];
  parts.push(`<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" role="img">`);
  items.forEach((item, index) => {
    const y = index * rowHeight + 1;
    const length = Math.max(1, (item.value / maxValue) * barW);
    parts.push(`<text x="${labelW - 4}" y="${y + rowHeight - 4}" text-anchor="end" fill="#444" ${FONT}>${escapeXml(item.label)}</text>`);
    parts.push(`<rect x="${labelW}" y="${y + 1.5}" width="${round2(length)}" height="${rowHeight - 5}" rx="2" fill="${item.color ?? '#4a7a2a'}" fill-opacity="0.8"/>`);
    parts.push(`<text x="${labelW + round2(length) + 4}" y="${y + rowHeight - 4}" fill="#555" ${FONT}>${escapeXml(`${item.value}${item.detail ? ` ${item.detail}` : ''}`)}</text>`);
  });
  parts.push('</svg>');
  return parts.join('');
}
