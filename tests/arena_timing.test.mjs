import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { engineBrushes } from '../src/lc0/analysisFormat.ts';
import { engineMonogram } from '../src/lc0/analysisPresentation.ts';
import { arenaSideTimeMs } from '../src/lc0/arenaTiming.ts';
import { arenaComparisonStatus, arenaLaneOutput, compareArenaEvals, evalPercent, searchAgeLabel } from '../src/lc0/evalCompare.ts';

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

for (const [mode, ms, showThinking] of [
  ['movetime', 50, false],
  ['movetime', 100, false],
  ['movetime', 250, false],
  ['movetime', 300, true],
  ['movetime', 1000, true],
  ['fixed', 50, true],
]) {
  test(`Arena thinking indicators with ${mode} and movetime input ${ms}`, () => {
    const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
    const slice = (start, end) => source.slice(source.indexOf(start), source.indexOf(end));
    const thinkingEngineIds = new Set();
    const engineOutputs = new Map();
    const lanes = Array.from({ length: 2 }, () => {
      const classes = new Set();
      const properties = new Map();
      return {
        classes,
        properties,
        classList: { toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)) },
        style: { setProperty: (name, value) => properties.set(name, value) },
        querySelector: () => ({}),
      };
    });
    const nodes = new Map(
      ['arenaEvalBar', 'evalSource', 'whiteSideLabel', 'blackSideLabel'].map((id) => [
        id,
        {
          dataset: {},
          classList: { toggle() {} },
          setAttribute(name, value) {
            this[name] = value;
          },
          querySelector: () => null,
          querySelectorAll: () => lanes,
        },
      ]),
    );
    const deps = {
      running: true,
      reviewing: false,
      board: parseFen(START_FEN),
      lastUci: null,
      boardWhiteId: 'white',
      boardWhiteName: 'Lc0 Small',
      boardBlackId: 'black',
      boardBlackName: 'Stockfish Lite',
      liveTrail: { charts: [], openingPlies: 0, entries: [{ fen: START_FEN }] },
      thinkingEngineIds,
      thinkingSince: new Map([['white', 0]]),
      performance: { now: () => 200 },
      arenaSearchProgress: new Map([['white', { completed: 10, requested: 20 }]]),
      engineOutputs,
      isolatedCpuSeats: new Map(),
      engines: new Map([['white', { name: 'Lc0 Small' }]]),
      selectEl: () => ({ value: mode }),
      inputEl: () => ({ value: String(ms) }),
      el: (id) => nodes.get(id),
      arenaSideTimeMs,
      parseFen,
      boardToFen,
      engineBrushes,
      engineMonogram,
      arenaLaneOutput,
      compareArenaEvals,
      arenaComparisonStatus,
      evalPercent,
      searchAgeLabel,
      displayEngineName: String,
      htmlEscape: String,
      engineAvatarHtml: () => '',
    };
    const draw = new Function(
      ...Object.keys(deps),
      `${stripTypeScriptTypes(`
        ${slice('function hasBoundedArenaSearch(', 'function setArenaSearchProgress(')}
        ${slice('function formatSeconds(', 'function renderBoard(')}
        ${slice('function arenaBudgetMode(', 'function arenaCacheEntries(')}
      `)}\nreturn () => { renderSideLabels(); return engineCardHtml('white'); };`,
    )(...Object.values(deps));
    const waitingCard = draw();
    thinkingEngineIds.add('white');
    const searchingCard = draw();
    const strip = nodes.get('whiteSideLabel');
    assert.equal(strip.dataset.turn, String(showThinking));
    assert.equal(strip.innerHTML.includes('strip-thinking'), showThinking);
    assert.match(strip.innerHTML, /strip-clock-value">0\.20 s</);
    assert.equal(lanes[0].classes.has('thinking'), showThinking);
    assert.equal(lanes[1].classes.has('thinking'), false);
    for (const id of ['arenaEvalBar', 'evalSource']) assert.equal(nodes.get(id)['aria-label'].includes('searching now'), showThinking);
    assert.equal(searchingCard.includes('class="eng-card thinking"'), showThinking);
    assert.equal(searchingCard.includes('eng-thinking-label'), showThinking);
    assert.equal(searchingCard.includes('eng-search-meter'), showThinking);
    if (!showThinking) assert.equal(searchingCard, waitingCard);

    thinkingEngineIds.clear();
    engineOutputs.set('white', { engineId: 'white', engineName: 'Lc0 Small', fen: START_FEN, whiteCp: 60, evalText: '+0.60' });
    const completedCard = draw();
    assert.match(completedCard, /class="eng-eval"[^>]*>\+0\.60</);
    assert.match(nodes.get('evalSource').innerHTML, /<strong>\+0\.60<\/strong>/);
    assert.equal(lanes[0].properties.get('--pct'), `${evalPercent({ scoreCp: 60 })}%`);
    if (!showThinking) {
      thinkingEngineIds.add('white');
      assert.equal(draw(), completedCard);
    }
  });
}

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
