import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { arenaSideTimeMs } from '../src/lc0/arenaTiming.ts';

test('live totals belong to the current game even while an older chart is under review', () => {
  const reviewed = [
    { ply: 0, moveMs: 1200 },
    { ply: 1, moveMs: 2400 },
  ];
  let live = reviewed;
  assert.deepEqual(arenaSideTimeMs(live, 'w'), { w: 1200, b: 2400 });
  live = [];
  assert.deepEqual(arenaSideTimeMs(live, 'w'), { w: 0, b: 0 });
  live.push({ ply: 0, moveMs: 300 });
  assert.deepEqual(arenaSideTimeMs(live, 'w'), { w: 300, b: 0 });
  assert.deepEqual(reviewed, [
    { ply: 0, moveMs: 1200 },
    { ply: 1, moveMs: 2400 },
  ]);
});

test('parallel focus includes searches made in the background, independent of worker identity', () => {
  const first = [{ ply: 0, moveMs: 1000 }];
  const background = [
    { ply: 0, moveMs: 200 },
    { ply: 1, moveMs: 400 },
    { ply: 2, moveMs: 600 },
  ];
  assert.deepEqual(arenaSideTimeMs(first, 'w'), { w: 1000, b: 0 });
  assert.deepEqual(arenaSideTimeMs(background, 'w'), { w: 800, b: 400 });
  background.push({ ply: 3, moveMs: 150 });
  assert.deepEqual(arenaSideTimeMs(background, 'w'), { w: 800, b: 550 });
});

test('the real live strip uses its game trail, not stale counters or the reviewed chart', () => {
  const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
  const render = source.slice(source.indexOf('function arenaPreparingGame()'), source.indexOf('function renderSideLabels()'));
  const nodes = new Map();
  const dependencies = {
    boardWhiteId: 'white',
    boardWhiteName: 'Stockfish Lite',
    boardBlackId: 'black',
    boardBlackName: 'Reckless',
    board: parseFen(START_FEN),
    performance: { now: () => 0 },
    liveTrail: { charts: [], openingPlies: 0, entries: [{ fen: START_FEN }] },
    arenaSideTimeMs,
    parseFen,
    running: true,
    reviewing: false,
    thinkingSince: new Map(),
    thinkingEngineIds: new Set(),
    // Old per-page counters and the reviewed samples must not leak into the new game's strip.
    sideTimeMs: { w: 9000, b: 8000 },
    gameChartSamples: [{ ply: 0, moveMs: 9000 }],
    el: (id) => {
      const node = { dataset: {}, querySelector: () => null };
      nodes.set(id, node);
      return node;
    },
    displayEngineName: String,
    engineAvatarHtml: () => '',
    htmlEscape: String,
    formatSeconds: (ms) => `${ms}ms`,
  };
  const draw = new Function(...Object.keys(dependencies), `${stripTypeScriptTypes(render)}\nreturn (trail) => { liveTrail = trail; renderStrips(); };`)(
    ...Object.values(dependencies),
  );
  draw(null);
  for (const node of nodes.values()) {
    assert.match(node.innerHTML, /Preparing engines…/);
    assert.match(node.innerHTML, /arena-avatar placeholder/);
    assert.match(node.innerHTML, /strip-clock-value">—</);
  }
  draw(dependencies.liveTrail);
  for (const node of nodes.values()) {
    assert.doesNotMatch(node.innerHTML, /Preparing engines…/);
    assert.match(node.innerHTML, /strip-clock-value">0ms</);
  }
});

test('stopping Arena before the first game redraws both strips and eval bars in the idle state', async () => {
  const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
  const preparing = source.slice(source.indexOf('function arenaPreparingGame()'), source.indexOf('function stripSides()'));
  const sideLabels = source.slice(source.indexOf('function renderSideLabels()'), source.indexOf('function setBoardSideEngines('));
  const match = source.slice(source.indexOf('  let enginesReady = false;'), source.indexOf('\nasync function startMatch('));
  const cleanup = match.slice(match.lastIndexOf('  } finally {'));
  const redraws = [];
  const stop = new Function(
    'redraws',
    stripTypeScriptTypes(`
      const matchAbort = new AbortController(), mountSignal = new AbortController().signal;
      let abort = matchAbort, running = true, startPending = false, reviewing = false, liveTrail = null, activeEngineIds = [];
      const checkpointWrites = Promise.resolve(), engineOutputs = new Map(), thinkingEngineIds = new Set();
      const el = () => ({toggleAttribute() {}});
      const isMemoryConstrainedDevice = () => false, isStaleMount = () => false;
      const noop = () => {};
      const setArenaStartDisabled = noop, refreshOpeningPreview = noop, refreshStockfishControls = noop, refreshSeatControls = noop;
      const refreshRecklessVariantUi = noop, refreshViridithasVariantUi = noop, refreshBerserkVariantUi = noop;
      const refreshPlentyChessVariantUi = noop, refreshStormphraxVariantUi = noop, renderEngineOutputs = noop;
      ${preparing}
      ${sideLabels}
      const renderStrips = () => redraws.push(['strips', arenaPreparingGame()]);
      const renderEvalBars = () => redraws.push(['eval bars', arenaPreparingGame()]);
      async function stop() {
        renderSideLabels();
        try { matchAbort.abort();
      ${cleanup}
    `) + '\nreturn stop;',
  )(redraws);
  await stop();
  assert.deepEqual(redraws, [
    ['strips', true],
    ['eval bars', true],
    ['strips', false],
    ['eval bars', false],
  ]);
});

test('search parity starts at the opening side, including black-to-move custom roots', () => {
  assert.deepEqual(
    arenaSideTimeMs(
      [
        { ply: 0, moveMs: 100 },
        { ply: 1, moveMs: 200 },
        { ply: 2, moveMs: 300 },
      ],
      'b',
    ),
    { w: 200, b: 400 },
  );
});
