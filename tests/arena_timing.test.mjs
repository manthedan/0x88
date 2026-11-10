import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { arenaSideTimeMs } from '../src/lc0/arenaTiming.ts';

test('live totals belong to the current game even while an older chart is under review', () => {
  const reviewed = [{ ply: 0, moveMs: 1200 }, { ply: 1, moveMs: 2400 }];
  let live = reviewed;
  assert.deepEqual(arenaSideTimeMs(live, 'w'), { w: 1200, b: 2400 });
  live = [];
  assert.deepEqual(arenaSideTimeMs(live, 'w'), { w: 0, b: 0 });
  live.push({ ply: 0, moveMs: 300 });
  assert.deepEqual(arenaSideTimeMs(live, 'w'), { w: 300, b: 0 });
  assert.deepEqual(reviewed, [{ ply: 0, moveMs: 1200 }, { ply: 1, moveMs: 2400 }]);
});

test('parallel focus includes searches made in the background, independent of worker identity', () => {
  const first = [{ ply: 0, moveMs: 1000 }];
  const background = [{ ply: 0, moveMs: 200 }, { ply: 1, moveMs: 400 }, { ply: 2, moveMs: 600 }];
  assert.deepEqual(arenaSideTimeMs(first, 'w'), { w: 1000, b: 0 });
  assert.deepEqual(arenaSideTimeMs(background, 'w'), { w: 800, b: 400 });
  background.push({ ply: 3, moveMs: 150 });
  assert.deepEqual(arenaSideTimeMs(background, 'w'), { w: 800, b: 550 });
});

test('the real live strip uses its game trail, not stale counters or the reviewed chart', () => {
  const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
  const render = source.slice(source.indexOf('function renderStrips()'), source.indexOf('function renderSideLabels()'));
  const nodes = new Map();
  const dependencies = {
    stripSides: () => ({ w: {}, b: {} }), board: parseFen(START_FEN), performance: { now: () => 0 },
    liveTrail: { charts: [], openingPlies: 0, entries: [{ fen: START_FEN }] }, arenaSideTimeMs, parseFen,
    running: true, reviewing: false, thinkingSince: new Map(), thinkingEngineIds: new Set(),
    // Old per-page counters and the reviewed samples must not leak into the new game's strip.
    sideTimeMs: { w: 9000, b: 8000 }, gameChartSamples: [{ ply: 0, moveMs: 9000 }],
    el: (id) => { const node = { dataset: {}, querySelector: () => null }; nodes.set(id, node); return node; },
    displayEngineName: String, engineAvatarHtml: () => '', htmlEscape: String, formatSeconds: (ms) => `${ms}ms`,
  };
  new Function(...Object.keys(dependencies), `${stripTypeScriptTypes(render)}\nrenderStrips();`)(...Object.values(dependencies));
  for (const node of nodes.values()) assert.match(node.innerHTML, /strip-clock-value">0ms</);
});

test('search parity starts at the opening side, including black-to-move custom roots', () => {
  assert.deepEqual(arenaSideTimeMs([{ ply: 0, moveMs: 100 }, { ply: 1, moveMs: 200 }, { ply: 2, moveMs: 300 }], 'b'), { w: 200, b: 400 });
});
