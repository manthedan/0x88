import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { START_FEN } from '../src/chess/board.ts';
import { analysisResultCacheKey } from '../src/lc0/analysisCache.ts';
import { BT4_NET, Bt4WorkerSearcher, T3_NET } from '../src/lc0/bt4Engine.ts';
import { isLc0BigNetVariant } from '../src/lc0/engineCatalog.ts';
import { finishedRunFollowUp } from '../src/lc0/hiddenTabPause.ts';
import { lc0SplitKUrl } from '../src/lc0/lc0SmallModelSelection.ts';

// A big net picks its served model (the shipped export or, on Apple WebGPU, its
// split-K rewrite) when its worker initializes, and that URL is part of every
// Analysis cache and selection key. Keys taken before the first init must not
// be compared with keys taken after it (docs/bignet_smolgen_splitk_2026-11.md §6).

const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
function slice(start, end) {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from, `analysisBrowser.ts slice ${start} .. ${end}`);
  return source.slice(from, to);
}
const keyFunctions = slice('function engineAnalysisCacheKey(', 'function activeEngineRows()');
const bigNetFunctions = slice('/**\n * The big-net session options of an Analysis search', '// Lc0 big nets use hosted ONNX assets');
const analyzeFunction = slice('async function analyzeCurrent(', 'let analysisGameTree = tree;');

const T3_ORIGINAL = T3_NET.modelUrl;
const T3_SPLITK = lc0SplitKUrl(T3_ORIGINAL);

/**
 * The real Bt4WorkerSearcher runs against this scripted worker, so init reuse,
 * dispose-and-reload on other options and servedModelUrl are the shipped code.
 * plan maps a model URL to the init outcomes of its successive workers:
 * { servedModelUrl } or { error }, plus hold (reply only on release()),
 * crashOnSearch (a worker `error` event instead of a result) and searchTicks
 * (delay the result). A search result carries its session's URL.
 */
class FakeWorker {
  static plan = new Map();
  static instances = [];
  static searchLog = [];
  listeners = new Map();
  searches = 0;
  constructor() {
    FakeWorker.instances.push(this);
  }
  addEventListener(type, listener) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }
  emit(type, data) {
    for (const listener of this.listeners.get(type) ?? []) listener(data);
  }
  async reply(data, ticks = 1) {
    for (let i = 0; i < ticks; i++) await new Promise(setImmediate);
    this.emit('message', { data });
  }
  postMessage(message) {
    if (message.type === 'init') {
      this.init = message;
      const outcome = FakeWorker.plan.get(message.modelUrl)?.shift() ?? {};
      this.outcome = outcome;
      this.servedModelUrl = outcome.servedModelUrl;
      const ready = outcome.error
        ? { type: 'error', id: message.id, error: outcome.error }
        : { type: 'ready', id: message.id, backend: 'webgpu', ...(outcome.servedModelUrl ? { servedModelUrl: outcome.servedModelUrl } : {}) };
      if (outcome.hold) this.release = () => this.reply(ready);
      else this.reply(ready);
    } else if (message.type === 'search') {
      this.searches += 1;
      FakeWorker.searchLog.push({ model: this.init.modelUrl, multiPv: message.multiPv });
      if (this.outcome.crashOnSearch) setImmediate(() => this.emit('error', { message: 'injected GPU crash' }));
      else
        this.reply(
          {
            type: 'searchResult',
            id: message.id,
            result: { cancelled: false, move: 'e2e4', visits: message.visits, multiPv: message.multiPv, served: this.servedModelUrl },
          },
          this.outcome.searchTicks ?? 1,
        );
    }
  }
  terminate() {
    this.terminated = true;
  }
}

async function withWorkers(plan, run) {
  const original = globalThis.Worker;
  globalThis.Worker = FakeWorker;
  FakeWorker.plan = new Map(Object.entries(plan));
  FakeWorker.instances = [];
  FakeWorker.searchLog = [];
  try {
    await run();
  } finally {
    globalThis.Worker = original;
  }
}

const searches = () => FakeWorker.instances.map((worker) => worker.searches);

function element() {
  return { textContent: '', toggleAttribute() {}, disabled: false, checked: false };
}

function analysisPage({ rows, searchers, smallLc0Runtime = null }) {
  const code = ts.transpile(
    `
    let analysisAbort = null, analyzing = false, activeAnalysisRunId = 0, analysisRunForced = false;
    let activeAnalysisSelectionKey = '', analysisRunCentipawnKeys = [];
    let analyzeAfterActiveRun = false, analysisReloadWantsSearch = false, reloadLc0AfterActiveRun = false;
    let disposeUnusedEnginesAfterActiveRun = false, hiddenResumeForce = false, hiddenTab = { hidden: false };
    let analysisReloadAbort = null, activeLc0BackendLoad = null, analyzeAfterLc0BackendLoad = false;
    const reviewAbort = null, retry = false, criticalPanel = null, searcher = null;
    // Lc0 Small: the selected runtime vs the one the loaded worker backend runs.
    let currentRuntime = smallLc0Runtime ?? 'onnx', backendRuntime = currentRuntime;
    const workerReady = smallLc0Runtime !== null, smallSearchLog = [];
    const workerLc0Lines = async (_runId, _fen, _visits, _positions, searchedMultiPv, engine) => {
      smallSearchLog.push(backendRuntime);
      return [{ engine, pvUci: ['d2d4'], multipv: 1, scoreText: '+0.10', detail: backendRuntime, pvSan: '', searchedMultiPv }];
    };
    const lc0ServedModelUrl = '', mountAbort = new AbortController();
    const engineLineCache = new Map(), lineCache = new Map(), completeAnalysisKeys = new Set();
    const published = [];
    const tree = { current: { fen: START_FEN }, historyBoards: () => [{ fen: START_FEN }] };
    const elements = new Map();
    const el = (id) => elements.get(id) ?? (elements.set(id, element()), elements.get(id));
    const document = { getElementById: el };
    const BIG_NETS = { bt4: BT4_NET, t3: T3_NET };
    const bigNetFor = (variant) => {
      const key = variant === 't3' ? 't3' : 'bt4';
      return { config: BIG_NETS[key], searcher: searchers.get(key) };
    };
    const peekBigNetSearcher = (key) => searchers.get(key);
    const activeEngineRows = () => rows.map((row) => ({ ...row }));
    let currentMultiPv = 1;
    const multiPv = () => currentMultiPv, selectedLc0Runtime = () => currentRuntime, boardToFen = (board) => board.fen;
    const resourceBroker = { snapshot: () => ({ dial: 'balanced', cpuBudget: 1 }) };
    const engineResourceProfile = () => ({ resourceClass: 'gpu' });
    const parseFen = (fen) => ({ fen }), legalMoves = () => ['e2e4'];
    const analysisGamePosition = () => null, centipawnEvaluatorKeysForRows = () => [];
    const analysisEngineName = (label) => label, rowLabel = (row) => row.variant;
    const criticalAnalysisRefresh = { defer() {} }, deferHiddenTabWork = () => false;
    const rememberDeferredForce = (previous, force) => previous || force;
    const cancelActiveAnalysisRun = () => {
      analysisAbort?.abort();
      analysisAbort = null;
      analyzing = false;
    };
    const retireAnalysisRunCentipawnKeys = () => {}, disposeUnusedEngines = () => {};
    // As reloadLc0ForAnalysis: load the selected runtime's backend, then analyze.
    const reloadLc0ForAnalysis = async (analyzeAfterLoad) => {
      backendRuntime = currentRuntime;
      if (analyzeAfterLoad) void analyzeCurrent({ force: true });
    };
    const clearAnalysisSearchProgress = () => {}, syncBrokerParticipants = () => {}, clearEngineSearchProgress = () => {};
    const showIndeterminateSearchProgress = () => {}, showAnalysisProgress = () => {};
    const showModelProgress = () => {}, hideModelProgress = () => {};
    const renderLines = () => published.push(lineCache.get(tree.current.fen));
    const renderEvalBar = () => {}, setShapes = () => {}, bestShapes = () => [], renderEngineComparison = () => {};
    const cacheEngineLines = (key, lines) => engineLineCache.set(key, lines);
    const lc0AnalysisLines = (result, _fen, engine) => [{ engine, pvUci: [result.move], multipv: 1, scoreText: '+0.20', detail: result.served ?? '', pvSan: '', searchedMultiPv: result.multiPv }];
    const analysisConsensus = () => 'consensus', hiddenAnalysisStopEnabled = () => false;
    const requestHiddenTabWork = (state) => ({ state }), stopHiddenTabResume = (reload, dispose) => ({ reload, dispose });
    ${keyFunctions}
    ${bigNetFunctions}
    ${analyzeFunction}
    // As deferAnalysisSettingsRefresh: a settings change during a run queues a
    // run after it; otherwise it starts one.
    const settingsChanged = () => {
      if (analyzing && analysisAbort && !analysisAbort.signal.aborted) analyzeAfterActiveRun = true;
      else void analyzeCurrent();
    };
    return {
      analyzeCurrent,
      setMultiPv: (value) => {
        currentMultiPv = value;
        settingsChanged();
      },
      setRows: (next) => {
        rows = next;
        settingsChanged();
      },
      // As the lc0RuntimeSelect handler: deferAnalysisSettingsRefresh({ reloadLc0 })
      // during a run, otherwise reloadLc0Backend.
      setRuntime: (runtime) => {
        currentRuntime = runtime;
        if (analyzing && analysisAbort && !analysisAbort.signal.aborted) {
          analyzeAfterActiveRun = true;
          reloadLc0AfterActiveRun = true;
        } else backendRuntime = runtime;
      },
      smallSearchLog,
      idle: () => !analyzing,
      stop: cancelActiveAnalysisRun,
      leave: () => mountAbort.abort(),
      engineAnalysisCacheKey: (row) => engineAnalysisCacheKey(tree.current.fen, row),
      analysisSelectionCacheKey: () => analysisSelectionCacheKey(tree.current.fen),
      engineLineCache, lineCache, completeAnalysisKeys, published, el,
    };
    `,
    { target: ts.ScriptTarget.ES2022 },
  );
  return new Function(
    'START_FEN',
    'BT4_NET',
    'T3_NET',
    'searchers',
    'rows',
    'element',
    'analysisResultCacheKey',
    'isLc0BigNetVariant',
    'finishedRunFollowUp',
    'smallLc0Runtime',
    code,
  )(START_FEN, BT4_NET, T3_NET, searchers, rows, element, analysisResultCacheKey, isLc0BigNetVariant, finishedRunFollowUp, smallLc0Runtime);
}

const t3Row = { family: 'lc0', variant: 't3', strength: 64 };
const bt4Row = { family: 'lc0', variant: 'bt4', strength: 64 };
const urlPattern = (url) => new RegExp(url.replaceAll('.', '\\.'));

/** The served URL in the run's keys is the URL of the session that ran the search. */
function assertPublishedUnder(page, row, servedUrl) {
  const lines = page.lineCache.get(START_FEN);
  assert.equal(lines?.[0]?.pvUci?.[0], 'e2e4', 'completed lines were published');
  assert.equal(lines[0].detail, servedUrl, 'the search ran on the session serving this URL');
  assert.ok(page.published.length > 0);
  const key = page.engineAnalysisCacheKey(row);
  assert.match(key, urlPattern(servedUrl));
  assert.deepEqual([...page.engineLineCache.keys()], [key]);
  assert.deepEqual([...page.completeAnalysisKeys], [page.analysisSelectionCacheKey()]);
}

for (const [device, servedUrl] of [
  ['Apple WebGPU (split-K)', T3_SPLITK],
  ['non-Apple (shipped export)', T3_ORIGINAL],
]) {
  test(`first t3 analysis with an uninitialized searcher publishes and caches under the served URL: ${device}`, () =>
    withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: servedUrl }] }, async () => {
      const t3 = new Bt4WorkerSearcher(T3_NET);
      const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
      assert.equal(t3.servedModelUrl, '');
      await page.analyzeCurrent();
      assert.equal(t3.servedModelUrl, servedUrl);
      assertPublishedUnder(page, t3Row, servedUrl);
      assert.deepEqual(searches(), [1]);
      // The second visit hits the cache: no new search, no new session.
      await page.analyzeCurrent();
      assert.deepEqual(searches(), [1]);
      t3.dispose();
    }));
}

for (const [direction, before, after] of [
  ['split-K -> shipped export (e.g. a WebGPU fallback)', T3_SPLITK, T3_ORIGINAL],
  ['shipped export -> split-K (e.g. split-K became available)', T3_ORIGINAL, T3_SPLITK],
]) {
  test(`a pooled t3 session from Arena (2048 eval-cache entries) is reloaded before the keys: ${direction}`, () =>
    withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: before }, { servedModelUrl: after }] }, async () => {
      const t3 = new Bt4WorkerSearcher(T3_NET);
      await t3.init({ evalCacheEntries: 2048 });
      assert.equal(t3.servedModelUrl, before);
      const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
      await page.analyzeCurrent();
      // Analysis's options differ, so its session is a reload, and the reload's
      // URL (not the pooled session's) is in the keys and ran the search.
      assert.equal(FakeWorker.instances.length, 2);
      assert.equal(FakeWorker.instances[0].terminated, true);
      assert.equal(FakeWorker.instances[1].init.evalCacheEntries, 0);
      assert.deepEqual(searches(), [0, 1]);
      assertPublishedUnder(page, t3Row, after);
      assert.doesNotMatch(page.engineAnalysisCacheKey(t3Row), urlPattern(before));
      t3.dispose();
    }));
}

test('a pooled t3 session already initialized with the Analysis options is reused as is', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    await t3.init({ evalCacheEntries: 0 });
    const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
    await page.analyzeCurrent();
    assert.equal(FakeWorker.instances.length, 1);
    assert.deepEqual(searches(), [1]);
    assertPublishedUnder(page, t3Row, T3_SPLITK);
    t3.dispose();
  }));

for (const how of ['stop', 'leave']) {
  test(`after ${how} during a joined in-flight Arena init, Analysis does not reload the pooled session`, () =>
    withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, hold: true }] }, async () => {
      const t3 = new Bt4WorkerSearcher(T3_NET);
      const arenaInit = t3.init({ evalCacheEntries: 2048 });
      const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
      const run = page.analyzeCurrent();
      await new Promise(setImmediate);
      page[how]();
      FakeWorker.instances[0].release();
      await arenaInit;
      await run;
      await new Promise(setImmediate);
      // Arena's session survives as Arena configured it; nothing was searched or published.
      assert.equal(FakeWorker.instances.length, 1);
      assert.notEqual(FakeWorker.instances[0].terminated, true);
      assert.equal(t3.isInitializedFor({ evalCacheEntries: 2048 }), true);
      assert.equal(t3.servedModelUrl, T3_SPLITK);
      assert.deepEqual(searches(), [0]);
      assert.equal(page.lineCache.get(START_FEN), undefined);
      t3.dispose();
    }));
}

async function settled(page) {
  for (let quiet = 0, i = 0; quiet < 5 && i < 500; i++) {
    await new Promise(setImmediate);
    quiet = page.idle() ? quiet + 1 : 0;
  }
}

test('a MultiPV change during a pending big-net init: the search, the key and the cache all use the new MultiPV', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, hold: true }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
    const run = page.analyzeCurrent();
    await new Promise(setImmediate);
    page.setMultiPv(3);
    FakeWorker.instances[0].release();
    await run;
    await settled(page);
    // No 1-line search was cached under the 3-line key: every search asked for
    // 3 lines, the only entry is the 3-line key, and the queued run reused it.
    assert.deepEqual(FakeWorker.searchLog, [{ model: T3_ORIGINAL, multiPv: 3 }]);
    assert.deepEqual([...page.engineLineCache.keys()], [page.engineAnalysisCacheKey(t3Row)]);
    assert.equal(page.lineCache.get(START_FEN)[0].searchedMultiPv, 3);
    assert.deepEqual([...page.completeAnalysisKeys], [page.analysisSelectionCacheKey()]);
    t3.dispose();
  }));

test('an engine selection change during a pending big-net init restarts the run with the new selection', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, hold: true }], [BT4_NET.modelUrl]: [{ servedModelUrl: BT4_NET.modelUrl }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    const bt4 = new Bt4WorkerSearcher(BT4_NET);
    const page = analysisPage({
      rows: [t3Row],
      searchers: new Map([
        ['t3', t3],
        ['bt4', bt4],
      ]),
    });
    const run = page.analyzeCurrent();
    await new Promise(setImmediate);
    page.setRows([t3Row, bt4Row]);
    FakeWorker.instances[0].release();
    await run;
    await settled(page);
    // One search per engine of the new selection; nothing from the old one.
    assert.deepEqual(
      FakeWorker.searchLog.map((entry) => entry.model),
      [T3_ORIGINAL, BT4_NET.modelUrl],
    );
    const lines = page.lineCache.get(START_FEN);
    assert.deepEqual(
      lines.map((line) => line.detail),
      [T3_SPLITK, BT4_NET.modelUrl],
    );
    assert.deepEqual([...page.engineLineCache.keys()].sort(), [page.engineAnalysisCacheKey(t3Row), page.engineAnalysisCacheKey(bt4Row)].sort());
    assert.deepEqual([...page.completeAnalysisKeys], [page.analysisSelectionCacheKey()]);
    t3.dispose();
    bt4.dispose();
  }));

test('an Lc0 Small runtime change during a pending big-net init: no old-backend lines under the new runtime key', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, hold: true }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    const smallRow = { family: 'lc0', variant: 'small', strength: 64 };
    const page = analysisPage({ rows: [smallRow, t3Row], searchers: new Map([['t3', t3]]), smallLc0Runtime: 'hybrid-a' });
    const oldKey = page.engineAnalysisCacheKey(smallRow);
    const run = page.analyzeCurrent();
    await new Promise(setImmediate);
    page.setRuntime('hybrid-b');
    FakeWorker.instances[0].release();
    await run;
    await settled(page);
    const newKey = page.engineAnalysisCacheKey(smallRow);
    assert.notEqual(newKey, oldKey);
    // The run in flight was abandoned before searching; the queued reload loaded
    // the new runtime and the follow-up run searched on it.
    assert.deepEqual(page.smallSearchLog, ['hybrid-b']);
    assert.equal(page.engineLineCache.get(newKey)[0].detail, 'hybrid-b');
    assert.equal(page.engineLineCache.has(oldKey), false);
    assert.deepEqual(
      page.lineCache.get(START_FEN).map((line) => line.detail),
      ['hybrid-b', T3_SPLITK],
    );
    assert.deepEqual(FakeWorker.searchLog, [{ model: T3_ORIGINAL, multiPv: 1 }]);
    assert.deepEqual([...page.completeAnalysisKeys], [page.analysisSelectionCacheKey()]);
    t3.dispose();
  }));

test('a big-net worker crash mid-search shows its error row and still publishes the other engine', () =>
  withWorkers(
    { [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, searchTicks: 4 }], [BT4_NET.modelUrl]: [{ servedModelUrl: BT4_NET.modelUrl, crashOnSearch: true }] },
    async () => {
      const t3 = new Bt4WorkerSearcher(T3_NET);
      const bt4 = new Bt4WorkerSearcher(BT4_NET);
      const page = analysisPage({
        rows: [t3Row, bt4Row],
        searchers: new Map([
          ['t3', t3],
          ['bt4', bt4],
        ]),
      });
      await page.analyzeCurrent();
      // The crash cleared bt4's live servedModelUrl before t3 finished; the run's
      // keys did not change, so both rows were published.
      assert.equal(bt4.servedModelUrl, '');
      const lines = page.lineCache.get(START_FEN);
      assert.equal(lines.length, 2);
      assert.equal(lines[0].pvUci[0], 'e2e4');
      assert.equal(lines[0].detail, T3_SPLITK);
      assert.match(lines[1].error, /injected GPU crash/);
      assert.deepEqual([...page.engineLineCache.keys()], [page.engineAnalysisCacheKey(t3Row)]);
      // As before this branch: a run with a failed engine finishes, reports it, and is not marked complete.
      assert.equal(page.completeAnalysisKeys.size, 0);
      assert.match(page.el('message').textContent, /engine failures/);
      t3.dispose();
      bt4.dispose();
    },
  ));

test('a big-net init failure stays that engine’s failed row and does not suppress the other engines', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK }], [BT4_NET.modelUrl]: [{ error: 'model download failed' }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    const bt4 = new Bt4WorkerSearcher(BT4_NET);
    const page = analysisPage({
      rows: [t3Row, bt4Row],
      searchers: new Map([
        ['t3', t3],
        ['bt4', bt4],
      ]),
    });
    await page.analyzeCurrent();
    const lines = page.lineCache.get(START_FEN);
    assert.equal(lines.length, 2);
    assert.equal(lines[0].pvUci[0], 'e2e4');
    assert.equal(lines[1].error, 'model download failed');
    // The failed init is reported, not retried by the search.
    assert.equal(FakeWorker.instances.filter((worker) => worker.init.modelUrl === BT4_NET.modelUrl).length, 1);
    assert.deepEqual([...page.engineLineCache.keys()], [page.engineAnalysisCacheKey(t3Row)]);
    assert.equal(page.completeAnalysisKeys.size, 0, 'a run with a failed engine is not marked complete');
    t3.dispose();
    bt4.dispose();
  }));

for (const [label, outcome, expected] of [
  ['Apple WebGPU worker reports the split-K model', { servedModelUrl: T3_SPLITK }, T3_SPLITK],
  ['non-Apple worker reports the shipped export', { servedModelUrl: T3_ORIGINAL }, T3_ORIGINAL],
  ['worker without a served URL means the shipped export', {}, T3_ORIGINAL],
]) {
  test(`Bt4WorkerSearcher.servedModelUrl: ${label}`, () =>
    withWorkers({ [T3_ORIGINAL]: [outcome] }, async () => {
      const searcher = new Bt4WorkerSearcher(T3_NET);
      assert.equal(searcher.servedModelUrl, '');
      assert.equal(searcher.isInitializedFor(), false);
      await searcher.init({ evalCacheEntries: 2048 });
      assert.equal(searcher.servedModelUrl, expected);
      assert.equal(searcher.isInitializedFor({ evalCacheEntries: 2048 }), true);
      assert.equal(searcher.isInitializedFor({ evalCacheEntries: 0 }), false, 'other options would reload the session');
      searcher.dispose();
      assert.equal(searcher.servedModelUrl, '', 'a disposed searcher no longer claims a served model');
      assert.equal(searcher.isInitializedFor({ evalCacheEntries: 2048 }), false);
    }));
}
