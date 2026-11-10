import assert from 'node:assert/strict';
import { test } from 'node:test';
import { arenaChartGroups } from '../src/lc0/arenaChartPresentation.ts';
import { engineBrushes } from '../src/lc0/analysisFormat.ts';
import { lineChartSvg } from '../src/lc0/charts.ts';
const sample = (engineId, engineName, ply) => ({ engineId, engineName, ply });
test('all chart tabs use recorded engine identity colours, including replay', () => {
  const groups = arenaChartGroups([sample('a', 'Lc0 Small', 0), sample('b', 'Stockfish Lite', 1)]);
  assert.deepEqual(groups.map((g) => g.color), ['Lc0 Small', 'Stockfish Lite'].map((n) => engineBrushes(n).swatch));
  assert.deepEqual(groups.map((g) => g.dashed), [false, false]);
});
test('same-hue engine seats retain their colour but get distinct stroke patterns', () => {
  const groups = arenaChartGroups([sample('a', 'Lc0 Small', 0), sample('b', 'Lc0 Medium', 1)]);
  assert.equal(groups[0].color, groups[1].color);
  assert.deepEqual(groups.map((g) => g.dashed), [false, true]);
});
test('exact self-play with a shared worker ID preserves alternating searches', () => {
  const samples = [0, 1, 2, 3].map((ply) => sample('a', 'Lc0 Small', ply));
  const groups = arenaChartGroups(samples);
  assert.deepEqual(groups.map((g) => g.samples.map((s) => s.ply)), [[0, 2], [1, 3]]);
  assert.deepEqual(groups.map((g) => g.dashed), [false, true]);
  assert.equal(arenaChartGroups([]).length, 0);
});
test('SVG renders optional dashed traces without changing chart geometry', () => {
  const html = lineChartSvg([{ label: 'a', color: '#123456', dashed: true, points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] }], { width: 400, height: 64 });
  assert.match(html, /stroke="#123456" stroke-width="1.5" stroke-dasharray="4 3"/);
  assert.match(html, /viewBox="0 0 400 64"/);
});
