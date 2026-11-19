import assert from 'node:assert/strict';
import { test } from 'node:test';
import { barChartRowAt, barChartViewHeight, hBarChartSvg, lineChartFractionAt, lineChartPlotX, lineChartPlotY, lineChartSvg } from '../src/lc0/charts.ts';

test('lineChartSvg renders polylines per series and y tick labels', () => {
  const svg = lineChartSvg(
    [
      {
        label: 'a',
        color: '#123456',
        points: [
          { x: 1, y: 0.2 },
          { x: 2, y: 0.8 },
        ],
      },
      {
        label: 'b',
        color: '#654321',
        points: [
          { x: 1, y: 0.6 },
          { x: 2, y: 0.4 },
        ],
      },
    ],
    { yMin: 0, yMax: 1, formatY: (v) => `${Math.round(v * 100)}%` },
  );
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
  const flat = lineChartSvg([
    {
      label: 'x',
      color: '#000',
      points: [
        { x: 1, y: 3 },
        { x: 2, y: 3 },
      ],
    },
  ]);
  assert.ok(flat.includes('<polyline') && !flat.includes('NaN'));
});

test('lineChartSvg clamps values outside the fixed y range', () => {
  const svg = lineChartSvg(
    [
      {
        label: 'x',
        color: '#000',
        points: [
          { x: 0, y: 99 },
          { x: 1, y: -99 },
        ],
      },
    ],
    { yMin: 0, yMax: 1 },
  );
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

test('line chart hit-test helpers invert the rendered plot coordinates', () => {
  for (const width of [360, 520]) {
    const height = 118;
    const svg = lineChartSvg(
      [
        {
          label: 'x',
          color: '#000',
          points: [
            { x: 0, y: 0 },
            { x: 4, y: 0.25 },
            { x: 10, y: 1 },
          ],
        },
      ],
      { width, height, yMin: 0, yMax: 1 },
    );
    const coords = svg
      .match(/<polyline points="([^"]+)"/)[1]
      .split(' ')
      .map((pair) => pair.split(',').map(Number));
    for (const [[vx, vy], fraction, value] of [
      [coords[0], 0, 0],
      [coords[1], 0.4, 0.25],
      [coords[2], 1, 1],
    ]) {
      assert.ok(Math.abs(lineChartPlotX(fraction, width) - vx) < 0.01, `x at ${fraction}, width ${width}`);
      assert.ok(Math.abs(lineChartPlotY(value, height) - vy) < 0.01, `y at ${value}`);
      assert.ok(Math.abs(lineChartFractionAt(vx, width) - fraction) < 0.001);
    }
  }
});

test('bar chart hit-test helpers find the row under every rendered bar', () => {
  const items = ['a', 'b', 'c', 'd'].map((label, i) => ({ label, value: 10 - i }));
  const svg = hBarChartSvg(items);
  assert.match(svg, new RegExp(`viewBox="0 0 360 ${barChartViewHeight(items.length)}"`));
  const rects = [...svg.matchAll(/<rect[^>]*y="([\d.]+)"[^>]*height="([\d.]+)"/g)].map((m) => [Number(m[1]), Number(m[2])]);
  assert.equal(rects.length, items.length);
  rects.forEach(([top, h], row) => {
    assert.equal(barChartRowAt(top), row);
    assert.equal(barChartRowAt(top + h), row);
  });
});
