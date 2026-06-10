import assert from 'node:assert/strict';
import { test } from 'node:test';
import { hBarChartSvg, lineChartSvg } from '../src/lc0/charts.ts';

test('lineChartSvg renders polylines per series and y tick labels', () => {
  const svg = lineChartSvg([
    { label: 'a', color: '#123456', points: [{ x: 1, y: 0.2 }, { x: 2, y: 0.8 }] },
    { label: 'b', color: '#654321', points: [{ x: 1, y: 0.6 }, { x: 2, y: 0.4 }] },
  ], { yMin: 0, yMax: 1, formatY: (v) => `${Math.round(v * 100)}%` });
  assert.ok(svg.startsWith('<svg'));
  assert.equal((svg.match(/<polyline/g) ?? []).length, 2);
  assert.ok(svg.includes('#123456') && svg.includes('#654321'));
  assert.ok(svg.includes('100%') && svg.includes('0%'));
});

test('lineChartSvg handles empty, single-point, and flat series', () => {
  assert.equal(lineChartSvg([]), '');
  assert.equal(lineChartSvg([{ label: 'x', color: '#000', points: [] }]), '');
  const single = lineChartSvg([{ label: 'x', color: '#000', points: [{ x: 1, y: 5 }] }]);
  assert.ok(single.includes('<circle'));
  // Flat series must not divide by zero on the y range.
  const flat = lineChartSvg([{ label: 'x', color: '#000', points: [{ x: 1, y: 3 }, { x: 2, y: 3 }] }]);
  assert.ok(flat.includes('<polyline') && !flat.includes('NaN'));
});

test('lineChartSvg clamps values outside the fixed y range', () => {
  const svg = lineChartSvg([{ label: 'x', color: '#000', points: [{ x: 0, y: 99 }, { x: 1, y: -99 }] }], { yMin: 0, yMax: 1 });
  assert.ok(!svg.includes('NaN'));
});

test('hBarChartSvg scales bars to the max value and escapes labels', () => {
  const svg = hBarChartSvg([
    { label: 'e2e4', value: 100, detail: 'Q +0.10' },
    { label: '<b>', value: 50 },
  ]);
  assert.ok(svg.includes('&lt;b&gt;'));
  assert.ok(svg.includes('Q +0.10'));
  const widths = [...svg.matchAll(/<rect[^>]*width="([\d.]+)"/g)].map((m) => Number(m[1]));
  assert.equal(widths.length, 2);
  assert.ok(Math.abs(widths[0] / widths[1] - 2) < 0.05, `bar ratio ${widths[0]}/${widths[1]}`);
  assert.equal(hBarChartSvg([]), '');
});
