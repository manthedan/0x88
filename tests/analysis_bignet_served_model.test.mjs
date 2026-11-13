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
const bigNetFunctions = slice('async function workerBigNetLines(', '// Lc0 big nets use hosted ONNX assets');
const analyzeFunction = slice('async function analyzeCurrent(', 'let analysisGameTree = tree;');

const T3_ORIGINAL = T3_NET.modelUrl;
const T3_SPLITK = lc0SplitKUrl(T3_ORIGINAL);

/** Bt4WorkerSearcher's observable contract: servedModelUrl is empty until init resolves. */
class FakeBigNetSearcher {
  servedModelUrl = '';
  onDownloadProgress = null;
  inits = 0;
  searches = 0;
  constructor(config, servedUrl, failInit = null) {
    this.config = config;
    this.servedUrl = servedUrl;
    this.failInit = failInit;
  }
  async init() {
    if (this.servedModelUrl) return 'webgpu';
    this.inits += 1;
    await new Promise(setImmediate);
    if (this.failInit) throw this.failInit;
    this.servedModelUrl = this.servedUrl;
    return 'webgpu';
  }
  async search(_input, options) {
    await this.init();
    this.searches += 1;
    await new Promise(setImmediate);
    return { cancelled: false, move: 'e2e4', visits: options.visits };
  }
  cancel() {}
}

function element() {
  return { textContent: '', toggleAttribute() {}, disabled: false, checked: false };
}

function analysisPage({ rows, searchers }) {
  const code = ts.transpile(
    `
    let analysisAbort = null, analyzing = false, activeAnalysisRunId = 0, analysisRunForced = false;
    let activeAnalysisSelectionKey = '', analysisRunCentipawnKeys = [];
    let analyzeAfterActiveRun = false, analysisReloadWantsSearch = false, reloadLc0AfterActiveRun = false;
    let disposeUnusedEnginesAfterActiveRun = false, hiddenResumeForce = false, hiddenTab = { hidden: false };
    let analysisReloadAbort = null, activeLc0BackendLoad = null, analyzeAfterLc0BackendLoad = false;
    const reviewAbort = null, retry = false, criticalPanel = null, workerReady = false, searcher = null;
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
    const activeEngineRows = () => rows;
    const multiPv = () => 1, selectedLc0Runtime = () => 'onnx', boardToFen = (board) => board.fen;
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
    const retireAnalysisRunCentipawnKeys = () => {}, disposeUnusedEngines = () => {}, reloadLc0ForAnalysis = async () => {};
    const clearAnalysisSearchProgress = () => {}, syncBrokerParticipants = () => {}, clearEngineSearchProgress = () => {};
    const showIndeterminateSearchProgress = () => {}, showAnalysisProgress = () => {};
    const showModelProgress = () => {}, hideModelProgress = () => {};
    const renderLines = () => published.push(lineCache.get(tree.current.fen));
    const renderEvalBar = () => {}, setShapes = () => {}, bestShapes = () => [], renderEngineComparison = () => {};
    const cacheEngineLines = (key, lines) => engineLineCache.set(key, lines);
    const lc0AnalysisLines = (result, _fen, engine) => [{ engine, pvUci: [result.move], multipv: 1, scoreText: '+0.20', detail: '', pvSan: '' }];
    const analysisConsensus = () => 'consensus', hiddenAnalysisStopEnabled = () => false;
    const requestHiddenTabWork = (state) => ({ state }), stopHiddenTabResume = (reload, dispose) => ({ reload, dispose });
    ${keyFunctions}
    ${bigNetFunctions}
    ${analyzeFunction}
    return {
      analyzeCurrent,
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
    code,
  )(START_FEN, BT4_NET, T3_NET, searchers, rows, element, analysisResultCacheKey, isLc0BigNetVariant, finishedRunFollowUp);
}

const t3Row = { family: 'lc0', variant: 't3', strength: 64 };

for (const [device, servedUrl] of [
  ['Apple WebGPU (split-K)', T3_SPLITK],
  ['non-Apple (shipped export)', T3_ORIGINAL],
]) {
  test(`first t3 analysis with an uninitialized searcher publishes and caches under the served URL: ${device}`, async () => {
    const t3 = new FakeBigNetSearcher(T3_NET, servedUrl);
    const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
    assert.equal(t3.servedModelUrl, '');
    await page.analyzeCurrent();
    assert.equal(t3.servedModelUrl, servedUrl);
    assert.equal(t3.searches, 1);
    // The lines reach the page, are cached under the served model's key, and the
    // completion marker uses the same identity that a later visit computes.
    assert.equal(page.lineCache.get(START_FEN)?.[0]?.pvUci?.[0], 'e2e4');
    assert.ok(page.published.length > 0, 'completed lines were published');
    const key = page.engineAnalysisCacheKey(t3Row);
    assert.match(key, new RegExp(servedUrl.replaceAll('.', '\\.')));
    assert.deepEqual([...page.engineLineCache.keys()], [key]);
    assert.deepEqual([...page.completeAnalysisKeys], [page.analysisSelectionCacheKey()]);
    // The second visit hits the cache: no new search.
    await page.analyzeCurrent();
    assert.equal(t3.searches, 1);
    assert.equal(t3.inits, 1);
  });
}

test('a big-net init failure stays that engine’s failed row and does not suppress the other engines', async () => {
  const t3 = new FakeBigNetSearcher(T3_NET, T3_SPLITK);
  const bt4 = new FakeBigNetSearcher(BT4_NET, BT4_NET.modelUrl, new Error('model download failed'));
  const bt4Row = { family: 'lc0', variant: 'bt4', strength: 64 };
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
  assert.equal(bt4.searches, 0);
  assert.deepEqual([...page.engineLineCache.keys()], [page.engineAnalysisCacheKey(t3Row)]);
  assert.equal(page.completeAnalysisKeys.size, 0, 'a run with a failed engine is not marked complete');
});

class FakeWorker {
  static reply = {};
  listeners = new Map();
  addEventListener(type, listener) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }
  emit(type, data) {
    for (const listener of this.listeners.get(type) ?? []) listener(data);
  }
  postMessage(message) {
    if (message.type === 'init') {
      this.init = message;
      queueMicrotask(() => this.emit('message', { data: { type: 'ready', id: message.id, backend: 'webgpu', ...FakeWorker.reply } }));
    }
  }
  terminate() {}
}

for (const [label, reply, expected] of [
  ['Apple WebGPU worker reports the split-K model', { servedModelUrl: T3_SPLITK }, T3_SPLITK],
  ['non-Apple worker reports the shipped export', { servedModelUrl: T3_ORIGINAL }, T3_ORIGINAL],
  ['worker without a served URL means the shipped export', {}, T3_ORIGINAL],
]) {
  test(`Bt4WorkerSearcher.servedModelUrl: ${label}`, async () => {
    const original = globalThis.Worker;
    globalThis.Worker = FakeWorker;
    FakeWorker.reply = reply;
    const searcher = new Bt4WorkerSearcher(T3_NET);
    try {
      assert.equal(searcher.servedModelUrl, '');
      await searcher.init();
      assert.equal(searcher.servedModelUrl, expected);
      searcher.dispose();
      assert.equal(searcher.servedModelUrl, '', 'a disposed searcher no longer claims a served model');
    } finally {
      globalThis.Worker = original;
    }
  });
}
