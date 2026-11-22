import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { abortSignalAny } from '../src/lc0/abortSignalAny.ts';
import { ANALYSIS_BIG_NET_KEYS, ANALYSIS_BIG_NET_SESSION, createAnalysisEngines } from '../src/lc0/analysis/analysisEngines.ts';
import { analysisRunBroker, analysisRunRowKey, captureAnalysisRun } from '../src/lc0/analysisRunSnapshot.ts';
import {
  analysisCanPrepare,
  analysisIsAnalyzing,
  analysisOperationId,
  appendAnalysisTransitionLog,
  initialAnalysisRunState,
  transitionAnalysis,
} from '../src/lc0/analysisRunState.ts';
import { BT4_NET, Bt4WorkerSearcher, T3_NET } from '../src/lc0/bt4Engine.ts';
import { CriticalAnalysisRefresh } from '../src/lc0/criticalMoments.ts';
import { isLc0BigNetVariant } from '../src/lc0/engineCatalog.ts';
import {
  cancelHiddenTabWork,
  classifyHiddenReload,
  hiddenAnalysisStopEnabled,
  initialHiddenTabPause,
  requestHiddenTabWork,
  transitionHiddenTab,
} from '../src/lc0/hiddenTabPause.ts';
import { lc0SplitKUrl } from '../src/lc0/lc0SmallModelSelection.ts';
import { LC0_MODEL_DOWNLOAD_NOT_APPROVED, untilAborted } from '../src/lc0/modelCache.ts';
import { WorkerDeviceLostError } from '../src/lc0/workerDeviceRecovery.ts';

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
// The controller's own createAnalysisEngines call: port, row source and config.
const engineHost = slice('const engines = createAnalysisEngines({', '\nlet analysisRunState');
const analyzeFunction = slice('async function performAnalysisMaintenance(', 'let analysisGameTree = tree;');
const schedulerFunctions = slice('function hasActiveAnalysisRun(', 'function hiddenTabActive(');
const cancelFunctions = slice('function cancelActiveAnalysisRun(', 'function scheduleAnalyzeCurrent(');
const scheduleFunction = slice('function scheduleAnalyzeCurrent(', 'function assetFileLabel(');
const invalidateFunction = slice('const ANALYSIS_REFRESH_PENDING_MESSAGE', 'function deferAnalysisSettingsRefresh(');
const backendFunction = slice('async function loadLc0Backend(', 'async function loadLc0BackendOwned(');
const reloadFunction = slice('async function reloadLc0Backend(', 'async function init(');
const hiddenFunctions = slice('function hiddenTabActive(', 'function installHiddenTabPause(');
const criticalHandoff = slice('    pauseAnalysis: (maiaOwner) => {', '    updateView: (moment, lines) => {');
const navigationFunction = slice('let analysisGameTree = tree;', 'const MAIA3_PANEL_RATINGS');
const centipawnAvailability = slice('async function refreshCentipawnHybridManifestStatus(', 'function centipawnHistoryFens(');

const T3_ORIGINAL = T3_NET.modelUrl;
const T3_SPLITK = lc0SplitKUrl(T3_ORIGINAL);

// The host formats big-net lines with the real lc0AnalysisLines, whose detail is
// "<visits> visits". The fake worker's result visits encode the session's served
// URL and the searched MultiPV, so a published line still names both.
const servedUrls = [''];
function resultVisits(servedUrl = '', multiPv) {
  if (!servedUrls.includes(servedUrl)) servedUrls.push(servedUrl);
  return servedUrls.indexOf(servedUrl) * 1000 + multiPv;
}
function decodedVisits(line) {
  const match = /^(\d+) visits$/.exec(line.detail);
  return match ? Number(match[1]) : null;
}
/** The served URL a big-net line was searched on; other engines' detail as is. */
const detailOf = (line) => (decodedVisits(line) === null ? line.detail : servedUrls[Math.floor(decodedVisits(line) / 1000)]);
const searchedMultiPvOf = (line) => (decodedVisits(line) === null ? line.searchedMultiPv : decodedVisits(line) % 1000);

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
  removeEventListener(type, listener) {
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter((fn) => fn !== listener),
    );
  }
  emit(type, data) {
    for (const listener of this.listeners.get(type) ?? []) listener(data);
  }
  async reply(data, ticks = 1) {
    for (let i = 0; i < ticks; i++) await new Promise(setImmediate);
    this.emit('message', { data });
  }
  postMessage(message) {
    if (message.type === 'dispose') queueMicrotask(() => this.emit('message', { data: { type: 'disposed', id: message.id } }));
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
      this.searchMessages ??= [];
      this.searchMessages.push(message);
      FakeWorker.searchLog.push({ model: this.init.modelUrl, multiPv: message.multiPv });
      if (this.outcome.crashOnSearch) setImmediate(() => this.emit('error', { message: 'injected GPU crash' }));
      else if (this.outcome.loseDeviceOnSearch) this.reply({ type: 'deviceLost', loss: { reason: 'unknown', message: 'reset' } });
      else
        this.reply(
          {
            type: 'searchResult',
            id: message.id,
            result: { cancelled: false, value: 0, pv: ['e2e4'], children: [], visits: resultVisits(this.servedModelUrl, message.multiPv) },
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
  return {
    textContent: '',
    toggleAttribute(name, force) {
      if (name === 'disabled') this.disabled = force;
    },
    disabled: false,
    checked: false,
  };
}

/**
 * The page's Centipawn evaluator cache, reporting by row variant: the variants
 * whose evaluator was requested (holdCentipawn holds the next load), and the
 * variants each retain kept alive (the run pins plus the visible rows).
 */
function centipawnFixture() {
  const fixture = { centipawnVariants: [], pinRetentions: [], engines: null, held: null };
  const variantOf = (key) => ['bt4-custom', 'bt4-auto', 'bt4-ort'].find((variant) => fixture.engines.centipawnEvaluatorCacheKey(variant) === key) ?? key;
  fixture.variantOf = variantOf;
  fixture.holdCentipawn = () => {
    let release;
    const promise = new Promise((resolve) => {
      release = resolve;
    });
    fixture.held = { promise, release };
    return release;
  };
  fixture.cache = {
    has: () => true,
    async get(key) {
      fixture.centipawnVariants.push(variantOf(key));
      const held = fixture.held;
      fixture.held = null;
      if (held) await held.promise;
      return {};
    },
    retain: (keys) => fixture.pinRetentions.push([...keys].map(variantOf)),
    clear() {},
  };
  return fixture;
}

function analysisPage({ rows, searchers, smallLc0Runtime = null, acquireBigNet = async (key) => searchers.get(key), phone = false, consent = {} }) {
  const admissions = [];
  const centipawn = centipawnFixture();
  // The real host, on the test's searchers and Centipawn cache.
  const hostFactory = (deps) => {
    centipawn.engines = createAnalysisEngines({
      ...deps,
      bigNets: { acquire: (key) => acquireBigNet(key), peek: (key) => searchers.get(key), release() {}, disposeNow() {} },
      centipawnCache: centipawn.cache,
      createStockfish: () => ({ maxThreads: () => 1, setOptions() {}, newGame: async () => {}, analyze: async () => [] }),
    });
    return centipawn.engines;
  };
  const observeTransition = (state, event) => {
    const result = transitionAnalysis(state, event);
    for (const effect of result.effects) if (effect.type === 'search') admissions.push(effect.snapshot);
    return result;
  };
  const code = ts.transpile(
    `
    const { analysisRunBroker, analysisRunRowKey, captureAnalysisRun, parseFen } = snapshotHelpers;
    const { analysisCanPrepare, analysisIsAnalyzing, analysisOperationId, appendAnalysisTransitionLog, initialAnalysisRunState, transitionAnalysis, WorkerDeviceLostError } = stateHelpers;
    let analysisTransitionLog = Object.freeze([]);
    const { cancelHiddenTabWork, classifyHiddenReload, hiddenAnalysisStopEnabled, initialHiddenTabPause, requestHiddenTabWork, transitionHiddenTab } = hiddenHelpers;
    let analysisBackendGeneration = 1;
    let analysisAbort = null, activeAnalysisRunId = 0;
    let activeAnalysisSelectionKey = '';
    let hiddenTab = initialHiddenTabPause(false), analysisReloadAbort = null;
    const maia3PanelTimer = null, maia3PanelEvaluating = false, workspaceAutosave = null, maia3PanelTask = null;
    const analysisAssetPrefetchControllers = new Map();
    const cancelAllAnalysisAssetPrefetches = () => {}, scheduleMaia3Panel = () => {}, prefetchAnalysisAssetsForAddedRow = () => {};
    let analysisRunState = initialAnalysisRunState(), nextAnalysisOperationId = 0;
    const analysisOperations = new Map(), analysisTimers = new Map();
    const activeWorkerSearchId = null, searchWorker = null;
    const reviewAbort = null, retry = false, searcher = null;
    let criticalBusy = false;
    const criticalPanel = { isBusy: () => criticalBusy, invalidate() {} };
    const criticalDeferredMaia = false;
    // Lc0 Small: the selected runtime vs the one the loaded worker backend runs.
    let currentRuntime = smallLc0Runtime ?? 'onnx', backendRuntime = currentRuntime;
    let workerReady = smallLc0Runtime !== null, workerSearchWarm = workerReady;
    const smallSearchLog = [];
    const workerLc0Lines = async (_runId, _fen, _visits, _positions, searchedMultiPv, engine) => {
      smallSearchLog.push(backendRuntime);
      return [{ engine, pvUci: ['d2d4'], multipv: 1, scoreText: '+0.10', detail: backendRuntime, pvSan: '', searchedMultiPv }];
    };
    let lc0ServedModelUrl = '';
    const threadedStockfishAvailable = () => false, stockfishFlavorUrl = () => '/sf.js', jsSidecarWasmUrl = () => '/sf.wasm';
    const withEngineTimeout = (_label, signal, _ms, analyze) => analyze(signal), cpuAnalysisTimeoutMs = () => 1000;
    const stockfishAnalysisLines = (_infos, _fen, engine) => [{ engine, pvUci: ['e2e4'], multipv: 1, scoreText: '+0.10' }];
    let mountAbort = new AbortController();
    const engineLineCache = new Map(), lineCache = new Map(), completeAnalysisKeys = new Set();
    const published = [];
    let history = [parseFen(START_FEN)], cpuHistory = [];
    let tree = { root: { fen: START_FEN }, current: { fen: START_FEN }, historyBoards: () => history };
    const elements = new Map();
    const el = (id) => elements.get(id) ?? (elements.set(id, element()), elements.get(id));
    const selectEl = el, inputEl = el, maybeEl = el;
    const document = { getElementById: el };
    const BIG_NETS = { bt4: BT4_NET, t3: T3_NET };
    const peekBigNetSearcher = (key) => searchers.get(key);
    let engineRows = rows;
    const activeEngineRows = () => engineRows.map((row) => ({ ...row }));
    let currentMultiPv = 1;
    const multiPv = () => currentMultiPv, selectedLc0Runtime = () => currentRuntime;
    const legalMoves = () => ['e2e4'];
    const analysisGamePosition = () => ({ fen: START_FEN, moves: cpuHistory });
    const analysisEngineName = (label) => label, rowLabel = (row) => row.variant;
    const criticalAnalysisRefresh = new stateHelpers.CriticalAnalysisRefresh();
    const pvPreview = null, mainEvaluator = null, saveAnalysisWorkspace = () => {}, clearStalePgnDatabaseSearchResults = () => {}, clearReviewIfMainlineChanged = () => {}, renderAll = () => {};
    const stopGameReview = () => {}, clearReviewState = () => {};
    const disposeUnusedEngines = () => engines.disposeUnusedCentipawnEvaluators();
    const disposeRuntimeResources = () => {};
    let heldBackend = null;
    const backendCompletions = [], backendStarts = [];
    const holdBackend = () => {
      let release;
      const promise = new Promise((resolve) => { release = resolve; });
      heldBackend = { promise, release };
      return release;
    };
    const loadLc0BackendOwned = async (cancelled, signal) => {
      backendStarts.push(signal);
      el('analyze').toggleAttribute('disabled', true);
      selectEl('lc0RuntimeSelect').disabled = true;
      showModelProgress();
      const runtime = currentRuntime;
      const held = heldBackend;
      heldBackend = null;
      if (held) await held.promise;
      if (cancelled()) return false;
      backendRuntime = runtime;
      lc0ServedModelUrl = '/small-served.onnx';
      backendCompletions.push(runtime);
      workerReady = true;
      el('analyze').toggleAttribute('disabled', false);
      selectEl('lc0RuntimeSelect').disabled = false;
      hideModelProgress();
      return true;
    };
    const isStaleMount = (signal) => signal.aborted || signal !== mountAbort.signal;
    const isMemoryConstrainedDevice = () => phone, deviceMemoryClass = () => phone ? 'phone' : 'desktop';
    const unloadMaia3Panel = () => {}, releaseBigNetSearcher = () => {}, preparePhoneBigNet = () => {};
    const { requestLargeNetDownloadConsent, requestLargeNetRedownloadConsent } = consent;
    let maia3PanelBlockedByRun = null;
    const shouldAutoAnalyzePosition = () => true, mainlineNodes = () => [];
    const awaitEngineRetirements = async () => {}, hasEngineRetirements = () => false;
    const reconcileActiveRuntimeFallbackWarning = () => {}, renderRecklessRuntimeInfo = () => {}, renderBigNetWarning = () => {};
    const showAnalysisDeviceWarning = () => {};
    let centipawnHybridManifestStatus = 'unknown';
    // The TVMJS manifest fallback only runs for the legacy research override;
    // the shipped s4fix model is ORT-only and skips the manifest probe.
    const LEGACY_CENTIPAWN_OVERRIDE = true;
    const CENTIPAWN_HYBRID_MANIFEST_URL = 'fake-manifest', resolvePublicAssetUrl = (url) => url;
    const CENTIPAWN_MODEL_URL = 'fake-model', CENTIPAWN_META_URL = 'fake-meta';
    const LEGACY_CENTIPAWN_RESOLVED_MODEL_URL = 'fake-legacy-model', LEGACY_CENTIPAWN_RESOLVED_META_URL = 'fake-legacy-meta';
    const fetch = async () => ({ ok: false }), renderEngineList = () => {};
    const centipawnEngineLabel = (variant) => variant;
    const initialPerformanceDial = () => 'balanced', createBrowserSquareformerRuntimeEvaluator = () => { throw new Error('unused'); };
    const recklessVariantForKey = () => ({}), viridithasVariantForKey = () => ({}), berserkVariantForKey = () => ({});
    const plentyChessVariantForKey = () => ({}), stormphraxVariantForKey = () => ({});
    const preloadAnalysisAssets = async () => {};
    const showLoadingProgressItem = (id) => { if (id === 'model') modelProgressVisible = true; };
    const hideLoadingProgressItem = (id) => { if (id === 'model') modelProgressVisible = false; };
    const chooseMove = async () => ({}), centipawnSearchOptions = {}, centipawnHistoryFens = () => [];
    const centipawnPuctAnalysisLines = (_result, _fen, label) => [{ engine: label, pvUci: ['e2e4'], multipv: 1, scoreText: '+0.10' }];
    const params = new URLSearchParams();
    const clearAnalysisSearchProgress = () => {}, clearEngineSearchProgress = () => {};
    const showIndeterminateSearchProgress = () => {}, showAnalysisProgress = () => {};
    let modelProgressVisible = false;
    const showModelProgress = () => { modelProgressVisible = true; }, hideModelProgress = () => { modelProgressVisible = false; };
    const renderLines = () => published.push(lineCache.get(tree.current.fen));
    const renderEvalBar = () => {}, setShapes = () => {}, bestShapes = () => [], renderEngineComparison = () => {};
    const cacheEngineLines = (key, lines) => engineLineCache.set(key, lines);
    const analysisConsensus = () => 'consensus';
    ${engineHost}
    const { centipawnVariants, holdCentipawn, pinRetentions, variantOf } = centipawn;
    ${keyFunctions}
    ${schedulerFunctions}
    ${hiddenFunctions}
    ${invalidateFunction}
    ${cancelFunctions}
    ${scheduleFunction}
    ${backendFunction}
    ${centipawnAvailability}
    ${reloadFunction}
    ${analyzeFunction}
    ${navigationFunction}
    const criticalCallbacks = { ${criticalHandoff} };
    const settingsChanged = (maintenance) => invalidateAnalysisSettings(maintenance, true);
    return {
      analyzeCurrent, holdBackend, admissions: stateHelpers.admissions,
      recordBigNetConsent: engines.recordBigNetConsent,
      loadBackend: (auto = true) => loadLc0Backend(auto),
      modelProgress: () => modelProgressVisible,
      backendCompletions, backendStarts,
      navigateAutoOff: (replaceTree = false) => { inputEl('autoAnalyze').checked = false; if (replaceTree) tree = { ...tree, current: { ...tree.current } }; afterNavigation(); },
      refreshCentipawn: refreshCentipawnHybridManifestStatus, centipawnVariants, holdCentipawn, pinRetentions,
      snapshotPins: () => engines.analysisRunCentipawnKeys().map(variantOf),
      enterCritical: () => { criticalBusy = true; criticalCallbacks.pauseAnalysis(); },
      settleCritical: () => { criticalBusy = false; criticalCallbacks.settled(); },
      enterMode: (mode) => dispatchAnalysis({ type: 'MODE_ENTER', mode }),
      exitMode: (mode) => dispatchAnalysis({ type: 'MODE_EXIT', mode }),
      schedule: (force) => scheduleAnalyzeCurrent({ force }, 0),
      fireScheduled: () => {
        const ticket = analysisRunState.execution.ticket;
        clearTimeout(analysisTimers.get(ticket)); analysisTimers.delete(ticket);
        return dispatchAnalysis({ type: 'DEBOUNCE_FIRED', ticket });
      },
      dropCriticalRefresh: () => criticalAnalysisRefresh.navigate('other-history', true),
      setMultiPv: (value) => {
        currentMultiPv = value;
        settingsChanged();
      },
      setRows: (next, maintenance) => {
        engineRows = next;
        settingsChanged(maintenance);
      },
      // As the lc0RuntimeSelect handler: deferAnalysisSettingsRefresh({ reloadLc0 })
      // during a run, otherwise reloadLc0Backend.
      setRuntime: (runtime) => {
        currentRuntime = runtime;
        settingsChanged({ reloadLc0: true, disposeUnused: false });
      },
      smallSearchLog,
      idle: () => ['idle', 'disposed'].includes(analysisRunState.execution.tag),
      stop: cancelActiveAnalysisRun,
      leave: () => { mountAbort.abort(); void dispatchAnalysis({ type: 'DISPOSE' }); },
      remount: () => { mountAbort = new AbortController(); analysisRunState = initialAnalysisRunState(); engineLineCache.clear(); completeAnalysisKeys.clear(); },
      hide: () => applyHiddenTab(true),
      show: () => applyHiddenTab(false),
      state: () => analysisRunState,
      setStrength: (strength) => { engineRows = engineRows.map((row) => ({ ...row, strength })); settingsChanged(); },
      setDial: (value) => { engines.broker.setDial(value); settingsChanged(); },
      navigateHistory: (boards, moves) => { history = boards; cpuHistory = moves; return analyzeCurrent(); },
      normalizeRows: (next) => { engineRows = next; invalidateAnalysisSettings(); },
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
    'acquireBigNet',
    'rows',
    'element',
    'snapshotHelpers',
    'isLc0BigNetVariant',
    'stateHelpers',
    'smallLc0Runtime',
    'hiddenHelpers',
    'createAnalysisEngines',
    'centipawn',
    'ANALYSIS_BIG_NET_KEYS',
    'ANALYSIS_BIG_NET_SESSION',
    'phone',
    'consent',
    'untilAborted',
    'abortSignalAny',
    code,
  )(
    START_FEN,
    BT4_NET,
    T3_NET,
    searchers,
    acquireBigNet,
    rows,
    element,
    { analysisRunBroker, analysisRunRowKey, captureAnalysisRun, parseFen },
    isLc0BigNetVariant,
    {
      analysisCanPrepare,
      analysisIsAnalyzing,
      analysisOperationId,
      initialAnalysisRunState,
      appendAnalysisTransitionLog,
      CriticalAnalysisRefresh,
      WorkerDeviceLostError,
      transitionAnalysis: observeTransition,
      admissions,
    },
    smallLc0Runtime,
    { cancelHiddenTabWork, classifyHiddenReload, hiddenAnalysisStopEnabled, initialHiddenTabPause, requestHiddenTabWork, transitionHiddenTab },
    hostFactory,
    centipawn,
    ANALYSIS_BIG_NET_KEYS,
    ANALYSIS_BIG_NET_SESSION,
    phone,
    consent,
    untilAborted,
    abortSignalAny,
  );
}

test('post-capture controls cannot relabel search arguments, cached rows or completed coverage', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    try {
      await t3.init({ evalCacheEntries: 0 });
      const row = { family: 'lc0', variant: 't3', strength: 64 };
      const page = analysisPage({ rows: [row], searchers: new Map([['t3', t3]]) });
      const first = page.analyzeCurrent(); // ready session: capture happens before returning
      const originalKey = page.analysisSelectionCacheKey();
      row.strength = 128;
      page.setMultiPv(3);
      const nextKey = page.analysisSelectionCacheKey();
      assert.notEqual(originalKey, nextKey);
      await first;
      for (let i = 0; i < 20 && !page.idle(); i++) await new Promise(setImmediate);
      assert.equal(page.idle(), true);
      const messages = FakeWorker.instances[0].searchMessages;
      assert.deepEqual(
        messages.map(({ visits, multiPv }) => [visits, multiPv]),
        [
          [64, 1],
          [128, 3],
        ],
      );
      assert.deepEqual(messages[0].input.positions.map(boardToFen), [START_FEN]);
      assert.deepEqual([...page.engineLineCache.keys()], [originalKey, nextKey]);
      assert.deepEqual([...page.completeAnalysisKeys], [originalKey, nextKey]);
      assert.equal(searchedMultiPvOf(page.engineLineCache.get(originalKey)[0]), 1);
      assert.equal(searchedMultiPvOf(page.lineCache.get(START_FEN)[0]), 3, 'only the current selection publishes');
    } finally {
      t3.dispose();
    }
  }));

test('Stop clears the queued MultiPV refresh message even after the old search settles', () =>
  withWorkers({ [T3_ORIGINAL]: [{}] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    try {
      await t3.init({ evalCacheEntries: 0 });
      const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
      const running = page.analyzeCurrent();
      await until(() => FakeWorker.searchLog.length === 1);
      page.setMultiPv(3);
      assert.ok(page.state().wanted, 'the MultiPV change queued a refresh');
      assert.match(page.el('message').textContent, /Finishing previous search/);
      page.stop();
      await running;
      assert.equal(page.idle(), true);
      assert.equal(page.state().wanted, null);
      assert.equal(FakeWorker.searchLog.length, 1, 'Stop discarded the queued search');
      assert.equal(page.el('message').textContent, 'Analysis stopped.');
    } finally {
      t3.dispose();
    }
  }));

const t3Row = { family: 'lc0', variant: 't3', strength: 64 };
const bt4Row = { family: 'lc0', variant: 'bt4', strength: 64 };
const urlPattern = (url) => new RegExp(url.replaceAll('.', '\\.'));

/** The served URL in the run's keys is the URL of the session that ran the search. */
function assertPublishedUnder(page, row, servedUrl) {
  const lines = page.lineCache.get(START_FEN);
  assert.equal(lines?.[0]?.pvUci?.[0], 'e2e4', 'completed lines were published');
  assert.equal(detailOf(lines[0]), servedUrl, 'the search ran on the session serving this URL');
  assert.ok(page.published.length > 0);
  const key = page.engineAnalysisCacheKey(row);
  assert.match(key, urlPattern(servedUrl));
  assert.deepEqual([...page.engineLineCache.keys()], [key]);
  assert.deepEqual([...page.completeAnalysisKeys], [page.analysisSelectionCacheKey()]);
}

test('Analysis retries a lost big-net search under the replacement model key', () =>
  withWorkers(
    {
      [T3_ORIGINAL]: [
        { servedModelUrl: T3_SPLITK, loseDeviceOnSearch: true },
        { servedModelUrl: T3_SPLITK, loseDeviceOnSearch: true },
        { servedModelUrl: T3_ORIGINAL },
      ],
    },
    async () => {
      const t3 = new Bt4WorkerSearcher(T3_NET);
      try {
        const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
        await page.analyzeCurrent();
        await until(() => page.completeAnalysisKeys.size === 1);
        assert.deepEqual(searches(), [1, 1, 1]);
        assert.equal(FakeWorker.instances[2].init.ep, 'wasm');
        assertPublishedUnder(page, t3Row, T3_ORIGINAL);
      } finally {
        await t3.dispose();
      }
    },
  ));

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
    assert.equal(searchedMultiPvOf(page.lineCache.get(START_FEN)[0]), 3);
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
    assert.deepEqual(lines.map(detailOf), [T3_SPLITK, BT4_NET.modelUrl]);
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
    assert.equal(detailOf(page.engineLineCache.get(newKey)[0]), 'hybrid-b');
    assert.equal(page.engineLineCache.has(oldKey), false);
    assert.deepEqual(page.lineCache.get(START_FEN).map(detailOf), ['hybrid-b', T3_SPLITK]);
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
      assert.equal(detailOf(lines[0]), T3_SPLITK);
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

test('an Analysis cache-only load failure asks for forced redownload with a signal that Stop aborts', (t) =>
  withWorkers({ [T3_ORIGINAL]: [{ error: `${LC0_MODEL_DOWNLOAD_NOT_APPROVED}: ${T3_ORIGINAL}` }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    const answer = Promise.withResolvers();
    const requestLargeNetDownloadConsent = t.mock.fn(async () => 'cached');
    const requestLargeNetRedownloadConsent = t.mock.fn(() => answer.promise);
    const page = analysisPage({
      rows: [t3Row],
      searchers: new Map([['t3', t3]]),
      phone: true,
      consent: { requestLargeNetDownloadConsent, requestLargeNetRedownloadConsent },
    });
    page.recordBigNetConsent('t3', 'cached');
    const running = page.analyzeCurrent();
    try {
      await until(() => requestLargeNetRedownloadConsent.mock.callCount() || page.idle());
      assert.equal(requestLargeNetDownloadConsent.mock.callCount(), 0, 'ordinary consent could return cached again');
      assert.equal(requestLargeNetRedownloadConsent.mock.callCount(), 1, 'the cache-only failure must reach the forced-redownload helper');
      const [config, signal] = requestLargeNetRedownloadConsent.mock.calls[0].arguments;
      assert.equal(config, T3_NET);
      assert.equal(signal?.aborted, false, 'the forced-redownload helper receives a live signal');
      page.stop();
      assert.equal(signal.aborted, true, 'Stop aborts the pending consent even while the page is still mounted');
    } finally {
      answer.resolve(false);
      await running;
      await t3.dispose();
    }
    assert.deepEqual(
      FakeWorker.instances.map((worker) => worker.init.noModelDownload),
      [true],
    );
  }));

test('a rejected big-net acquisition reports that row’s error and keeps the other engines running', () =>
  withWorkers({ [BT4_NET.modelUrl]: [{ servedModelUrl: BT4_NET.modelUrl }] }, async () => {
    const bt4 = new Bt4WorkerSearcher(BT4_NET);
    const small = { family: 'lc0', variant: 'small', strength: 64 };
    const acquisitions = [];
    const page = analysisPage({
      rows: [t3Row, small, bt4Row],
      searchers: new Map([['bt4', bt4]]),
      smallLc0Runtime: 'onnx',
      acquireBigNet: async (key) => {
        acquisitions.push(key);
        if (key === 't3') throw new Error('retiring worker cleanup failed');
        return bt4;
      },
    });
    try {
      await page.analyzeCurrent();
      const lines = page.lineCache.get(START_FEN);
      assert.deepEqual(
        lines?.map((line) => line.engine),
        ['Lc0 t3-512 distill', 'Lc0', 'Lc0 BT4-it332'],
      );
      assert.equal(lines[0].error, 'retiring worker cleanup failed');
      assert.deepEqual(lines[1].pvUci, ['d2d4']);
      assert.deepEqual(lines[2].pvUci, ['e2e4']);
      assert.equal(acquisitions.filter((key) => key === 't3').length, 1, 'the failed acquisition is not retried');
      assert.equal(page.completeAnalysisKeys.size, 0);
      assert.match(page.el('message').textContent, /completed with one or more engine failures/);
    } finally {
      await bt4.dispose();
    }
  }));

async function until(predicate) {
  for (let i = 0; i < 100 && !predicate(); i++) await new Promise(setImmediate);
  assert.ok(predicate(), 'deferred preparation reached its expected await');
}

for (const gate of ['own init', 'joined other-options init', 'normalized second init']) {
  for (const change of [
    'MultiPV',
    'runtime',
    'rows',
    'strength',
    'dial',
    'same-FEN history',
    'normalization',
    'Stop',
    'hide',
    'dispose/remount',
    'replacement',
  ]) {
    test(`${change} while held at ${gate}: admission and arguments use one snapshot`, () =>
      withWorkers(
        {
          [T3_ORIGINAL]: [
            { servedModelUrl: T3_SPLITK, hold: true },
            { servedModelUrl: T3_ORIGINAL, hold: gate === 'normalized second init' },
          ],
        },
        async () => {
          const t3 = new Bt4WorkerSearcher(T3_NET);
          try {
            const small = { family: 'lc0', variant: 'small', strength: 64 };
            const joined = gate === 'own init' ? null : t3.init({ evalCacheEntries: 2048 });
            const page = analysisPage({ rows: [small, t3Row], searchers: new Map([['t3', t3]]), smallLc0Runtime: 'hybrid-a' });
            const first = page.analyzeCurrent({ force: true });
            await until(() => FakeWorker.instances[0]?.release);
            if (gate === 'normalized second init') {
              FakeWorker.instances[0].release();
              await joined;
              await until(() => FakeWorker.instances[1]?.release);
            }
            assert.deepEqual(FakeWorker.searchLog, []);
            assert.deepEqual(page.smallSearchLog, []);
            assert.deepEqual(page.admissions, []);
            const oldId = page.state().execution.owner.runId;
            let replacement;
            if (change === 'MultiPV') page.setMultiPv(3);
            if (change === 'runtime') page.setRuntime('hybrid-b');
            if (change === 'rows') page.setRows([t3Row]);
            if (change === 'strength') page.setStrength(128);
            if (change === 'dial') page.setDial('eco');
            if (change === 'same-FEN history') replacement = page.navigateHistory([parseFen(START_FEN), parseFen(START_FEN)], ['g1f3', 'g8f6', 'f3g1', 'f6g8']);
            if (change === 'normalization') page.normalizeRows([small]);
            if (change === 'Stop') page.stop();
            if (change === 'hide') page.hide();
            if (change === 'dispose/remount') {
              page.leave();
              page.remount();
              replacement = page.analyzeCurrent();
            }
            if (change === 'replacement') replacement = page.analyzeCurrent();
            assert.deepEqual(page.admissions, [], 'a settings mutation or replacement cannot search before validation');
            FakeWorker.instances[gate === 'normalized second init' ? 1 : 0].release();
            await Promise.all([first, joined, replacement]);
            await settled(page);
            if (change === 'Stop' || change === 'hide') {
              assert.deepEqual(page.admissions, []);
              assert.deepEqual(FakeWorker.searchLog, []);
              assert.deepEqual(page.smallSearchLog, []);
              if (change === 'Stop') return;
              assert.equal(page.state().wanted.force, true);
              page.show();
              await settled(page);
            }
            assert.equal(page.admissions.length, 1, 'only the validated follow-up searches');
            const captured = page.admissions[0];
            assert.deepEqual([...page.engineLineCache.keys()].sort(), [...captured.rowKeys].sort());
            assert.deepEqual([...page.completeAnalysisKeys], [captured.selectionKey]);
            assert.equal(captured.multiPv, change === 'MultiPV' ? 3 : 1);
            assert.equal(captured.runtime, change === 'runtime' ? 'hybrid-b' : 'hybrid-a');
            assert.equal(captured.broker.dial, change === 'dial' ? 'eco' : 'balanced');
            assert.equal(captured.positions.length, change === 'same-FEN history' ? 2 : 1);
            assert.equal(captured.cpuPosition.moves.length, change === 'same-FEN history' ? 4 : 0);
            assert.ok(captured.rows.every((row) => row.strength === (change === 'strength' ? 128 : 64)));
            if (change === 'rows') assert.equal(captured.rows.length, 1);
            if (change === 'normalization') assert.ok(captured.rows.every((row) => row.variant === 'small'));
            if (change === 'replacement' || change === 'dispose/remount' || change === 'same-FEN history') {
              assert.equal(page.state().execution.tag, 'idle');
              assert.notEqual(oldId, page.state().execution.owner?.runId);
            }
            for (const worker of FakeWorker.instances)
              for (const message of worker.searchMessages ?? []) {
                assert.equal(message.multiPv, captured.multiPv);
                assert.equal(message.visits, captured.rows.find((row) => row.variant === 't3').strength);
                assert.deepEqual(message.input.positions.map(boardToFen), captured.positions.map(boardToFen));
              }
            assert.ok(page.smallSearchLog.every((runtime) => runtime === captured.runtime));
          } finally {
            t3.dispose();
          }
        },
      ));
  }
}

for (const action of ['Stop', 'hide', 'settings', 'replacement', 'dispose/remount']) {
  test(`${action} during Small backend preparation cannot be consumed by an old callback`, async () => {
    const small = { family: 'lc0', variant: 'small', strength: 64 };
    const page = analysisPage({ rows: [small], searchers: new Map() });
    const release = page.holdBackend();
    page.el('autoAnalyze').checked = true;
    const loading = page.loadBackend();
    await page.analyzeCurrent({ force: true });
    assert.deepEqual(page.admissions, []);
    let newer;
    if (action === 'Stop') page.stop();
    if (action === 'hide') page.hide();
    if (action === 'settings') {
      page.setRuntime('hybrid-b');
      page.setMultiPv(3);
    }
    if (action === 'replacement') {
      page.stop();
      newer = page.analyzeCurrent();
    }
    if (action === 'dispose/remount') {
      page.leave();
      page.remount();
      newer = page.analyzeCurrent();
    }
    release();
    await Promise.all([loading, newer]);
    await settled(page);
    if (action === 'Stop' || action === 'hide') {
      assert.deepEqual(page.admissions, []);
      if (action === 'Stop') return;
      page.show();
      await settled(page);
    }
    assert.equal(page.admissions.length, 1);
    assert.deepEqual([...page.completeAnalysisKeys], [page.admissions[0].selectionKey]);
    assert.deepEqual(page.smallSearchLog, [action === 'settings' ? 'hybrid-b' : 'onnx']);
    assert.equal(page.admissions[0].multiPv, action === 'settings' ? 3 : 1);
  });
}

test('Stop followed by hide retires cleanup without leaving Analyze disabled', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, searchTicks: 20 }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    try {
      await t3.init({ evalCacheEntries: 0 });
      const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
      const running = page.analyzeCurrent();
      await until(() => FakeWorker.searchLog.length > 0);
      assert.equal(page.el('analyze').disabled, true);
      page.stop();
      page.hide();
      await running;
      page.show();
      await settled(page);
      assert.equal(page.el('analyze').disabled, false);
      assert.equal(page.el('lc0RuntimeSelect').disabled, false);
      assert.equal(page.admissions.length, 1, 'the cancelled search was not resurrected');
    } finally {
      t3.dispose();
    }
  }));

test('cancelling standalone backend setup restores controls and progress without old-finally ownership', async () => {
  const page = analysisPage({ rows: [{ family: 'lc0', variant: 'small', strength: 64 }], searchers: new Map() });
  const release = page.holdBackend();
  const loading = page.loadBackend();
  assert.equal(page.el('analyze').disabled, true);
  assert.equal(page.modelProgress(), true);
  page.stop();
  page.hide();
  assert.equal(page.el('analyze').disabled, false);
  assert.equal(page.el('lc0RuntimeSelect').disabled, false);
  assert.equal(page.modelProgress(), false);
  release();
  await loading;
  page.show();
  assert.deepEqual(page.admissions, []);
  assert.equal(page.el('analyze').disabled, false);
});

for (const change of ['MultiPV', 'runtime', 'maintenance-only', 'cached navigation']) {
  test(`critical comparison settlement replays ${change} without an interactive owner`, async () => {
    const small = { family: 'lc0', variant: 'small', strength: 64 };
    const page = analysisPage({ rows: [small], searchers: new Map(), smallLc0Runtime: 'onnx' });
    page.enterCritical();
    if (change === 'MultiPV') page.setMultiPv(3);
    else page.setRuntime('hybrid-b');
    if (change === 'cached navigation' || change === 'maintenance-only') page.stop({ preserveDeferredMaintenance: true });
    assert.deepEqual(page.admissions, []);
    page.settleCritical();
    await settled(page);
    if (change === 'maintenance-only' || change === 'cached navigation') assert.deepEqual(page.admissions, []);
    else {
      assert.equal(page.admissions.length, 1);
      assert.equal(page.admissions[0].multiPv, change === 'MultiPV' ? 3 : 1);
      assert.equal(page.admissions[0].runtime, change === 'runtime' ? 'hybrid-b' : 'onnx');
    }
    assert.equal(page.state().execution.tag, 'idle');
    assert.deepEqual(page.state().maintenance, { reloadLc0: false, disposeUnused: false });
  });
}

for (const mode of ['review', 'retry', 'critical']) {
  test(`${mode} handoff retires held big-net preparation and preserves queued reviewer maintenance`, () =>
    withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, hold: true }] }, async () => {
      const t3 = new Bt4WorkerSearcher(T3_NET);
      try {
        const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
        const first = page.analyzeCurrent({ force: true });
        await until(() => FakeWorker.instances[0]?.release);
        page.setRows([t3Row], { reloadLc0: false, disposeUnused: true });
        if (mode === 'critical') page.enterCritical();
        else await page.enterMode(mode);
        await page.analyzeCurrent({ force: true });
        FakeWorker.instances[0].release();
        await first;
        assert.deepEqual(page.admissions, []);
        assert.deepEqual(page.pinRetentions, [], 'old preparation cannot dispose reviewer resources');
        assert.equal(page.state().maintenance.disposeUnused, true);
        assert.deepEqual(page.state().modes, [mode]);
        if (mode === 'critical') {
          page.settleCritical();
          assert.equal(page.state().execution.tag, 'scheduled', 'exact refresh retains its debounce');
          await page.fireScheduled();
        } else {
          await page.exitMode(mode);
          assert.equal(page.state().execution.tag, 'idle', 'review/retry exit does not drain maintenance');
          await page.analyzeCurrent();
        }
        await settled(page);
        assert.equal(page.pinRetentions.length, 1);
        assert.equal(page.admissions.length, 1);
        assert.deepEqual(searches(), [1]);
      } finally {
        t3.dispose();
      }
    }));
}

for (const keepSelection of [false, true]) {
  test(`critical exact-selection force refresh survives only its own selection (${keepSelection})`, async () => {
    const page = analysisPage({ rows: [{ family: 'lc0', variant: 'small', strength: 64 }], searchers: new Map(), smallLc0Runtime: 'onnx' });
    await page.analyzeCurrent();
    page.enterCritical();
    await page.analyzeCurrent({ force: true });
    page.schedule(false);
    assert.equal(page.state().wanted, null, 'the keyed refresh is not a second reducer request');
    if (!keepSelection) page.dropCriticalRefresh();
    page.settleCritical();
    if (keepSelection) await page.fireScheduled();
    await settled(page);
    assert.deepEqual(page.smallSearchLog, keepSelection ? ['onnx', 'onnx'] : ['onnx'], 'force must bypass the cached row only for the matching selection');
    assert.deepEqual(page.state().modes, []);
    page.settleCritical();
    assert.equal(page.state().execution.tag, 'idle', 'settlement never replays twice');
  });
}

for (const kind of ['initial setup', 'runtime reload'])
  for (const replaceTree of [false, true]) {
    test(`auto-off navigation drops search but preserves ${kind} (new tree: ${replaceTree})`, async () => {
      const small = { family: 'lc0', variant: 'small', strength: 64 };
      const page = analysisPage({ rows: [small], searchers: new Map(), smallLc0Runtime: kind === 'runtime reload' ? 'onnx' : null });
      const release = page.holdBackend();
      let loading;
      if (kind === 'initial setup') loading = page.loadBackend();
      else page.setRuntime('hybrid-b');
      const operation = page.state().execution.operationId;
      page.navigateAutoOff(replaceTree);
      assert.equal(page.state().execution.operationId, operation, 'navigation does not cancel position-independent setup');
      assert.equal(page.state().wanted, null);
      release();
      await loading;
      await settled(page);
      assert.deepEqual(page.backendCompletions, [kind === 'runtime reload' ? 'hybrid-b' : 'onnx']);
      assert.deepEqual(page.admissions, []);
      assert.equal(page.el('analyze').disabled, false);
      await page.analyzeCurrent();
      assert.equal(page.admissions.length, 1);
      assert.equal(page.backendCompletions.length, 1, 'the loaded backend is reusable');
    });
  }

for (const showFirst of [false, true]) {
  test(`hidden queued settings survive critical settlement (show first: ${showFirst})`, async () => {
    const page = analysisPage({ rows: [{ family: 'lc0', variant: 'small', strength: 64 }], searchers: new Map(), smallLc0Runtime: 'onnx' });
    page.enterCritical();
    page.setMultiPv(3);
    page.hide();
    assert.deepEqual(page.admissions, []);
    if (showFirst) page.show();
    page.settleCritical();
    if (!showFirst) {
      assert.deepEqual(page.admissions, []);
      page.show();
    }
    await settled(page);
    assert.equal(page.admissions.length, 1);
    assert.equal(page.admissions[0].multiPv, 3);
  });
}

test('Centipawn manifest row fallback invalidates held big-net preparation before snapshot capture', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, hold: true }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    try {
      const page = analysisPage({ rows: [t3Row, { family: 'centipawn', variant: 'bt4-custom', strength: 64 }], searchers: new Map([['t3', t3]]) });
      const first = page.analyzeCurrent();
      await until(() => FakeWorker.instances[0]?.release);
      const revision = page.state().revision;
      await page.refreshCentipawn();
      assert.equal(page.state().revision, revision + 1);
      assert.deepEqual(page.admissions, []);
      FakeWorker.instances[0].release();
      await first;
      await settled(page);
      assert.deepEqual(page.centipawnVariants, ['bt4-auto']);
      assert.equal(page.admissions.length, 1);
      assert.equal(page.admissions[0].rows[1].variant, 'bt4-auto');
      assert.deepEqual([...page.engineLineCache.keys()].sort(), [...page.admissions[0].rowKeys].sort());
      assert.deepEqual([...page.completeAnalysisKeys], [page.admissions[0].selectionKey]);
    } finally {
      t3.dispose();
    }
  }));

test('automatic fallback during backend-only setup cannot recreate a navigation-dropped search', async () => {
  const page = analysisPage({
    rows: [
      { family: 'lc0', variant: 'small', strength: 64 },
      { family: 'centipawn', variant: 'bt4-custom', strength: 64 },
    ],
    searchers: new Map(),
  });
  const release = page.holdBackend();
  const loading = page.loadBackend(false);
  page.navigateAutoOff(true);
  await page.refreshCentipawn();
  assert.equal(page.state().wanted, null);
  release();
  await loading;
  await settled(page);
  assert.deepEqual(page.admissions, []);
  assert.deepEqual(page.centipawnVariants, []);
  await page.analyzeCurrent();
  assert.deepEqual(page.centipawnVariants, ['bt4-auto']);
});

test('Stop and hide release removed Centipawn snapshot pins before the old evaluator finishes loading', async () => {
  const page = analysisPage({ rows: [{ family: 'centipawn', variant: 'bt4-custom', strength: 64 }], searchers: new Map(), smallLc0Runtime: 'onnx' });
  const release = page.holdCentipawn();
  const running = page.analyzeCurrent();
  await until(() => page.centipawnVariants.length === 1);
  assert.deepEqual(page.snapshotPins(), ['bt4-custom']);
  page.setRows([{ family: 'lc0', variant: 'small', strength: 64 }], { reloadLc0: false, disposeUnused: true });
  page.stop({ preserveDeferredMaintenance: true });
  page.hide();
  assert.deepEqual(page.snapshotPins(), []);
  assert.deepEqual(page.pinRetentions.at(-1), []);
  release();
  await running;
  page.show();
  await settled(page);
  assert.deepEqual(page.pinRetentions.at(-1), []);
  assert.equal(page.admissions.length, 1, 'maintenance does not resurrect search');
});

test('an old evaluator callback cannot release a replacement owner’s Centipawn pins', async () => {
  const page = analysisPage({ rows: [{ family: 'centipawn', variant: 'bt4-custom', strength: 64 }], searchers: new Map() });
  const releaseOld = page.holdCentipawn();
  const first = page.analyzeCurrent();
  await until(() => page.centipawnVariants.length === 1);
  page.setRows([{ family: 'centipawn', variant: 'bt4-auto', strength: 64 }]);
  const releaseNew = page.holdCentipawn();
  const replacement = page.analyzeCurrent();
  await until(() => page.centipawnVariants.length === 2);
  assert.deepEqual(page.snapshotPins(), ['bt4-auto']);
  releaseOld();
  await first;
  assert.deepEqual(page.snapshotPins(), ['bt4-auto']);
  releaseNew();
  await replacement;
  assert.deepEqual(page.snapshotPins(), []);
});

test('a settled run keeps the visible Centipawn row’s evaluator alive', async () => {
  const page = analysisPage({ rows: [{ family: 'centipawn', variant: 'bt4-custom', strength: 64 }], searchers: new Map() });
  await page.analyzeCurrent();
  await settled(page);
  assert.deepEqual(page.centipawnVariants, ['bt4-custom']);
  assert.deepEqual(page.snapshotPins(), []);
  assert.deepEqual(page.pinRetentions.at(-1), ['bt4-custom'], 'the run pin is gone; the visible row still keeps its evaluator');
});

test('a stopped big-net search cannot clear the model progress the next owner shows', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, searchTicks: 20 }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    try {
      await t3.init({ evalCacheEntries: 0 });
      const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]) });
      const running = page.analyzeCurrent();
      await until(() => FakeWorker.searchLog.length === 1);
      page.stop();
      const release = page.holdBackend();
      const loading = page.loadBackend();
      assert.equal(page.modelProgress(), true);
      await running;
      for (let i = 0; i < 25; i++) await new Promise(setImmediate);
      assert.equal(page.modelProgress(), true, 'the stopped search cleared progress it no longer owns');
      release();
      await loading;
    } finally {
      t3.dispose();
    }
  }));

test('changing engines after Stop does not wait for a cancelled big-net init to drain', () =>
  withWorkers({ [T3_ORIGINAL]: [{ servedModelUrl: T3_SPLITK, hold: true }] }, async () => {
    const t3 = new Bt4WorkerSearcher(T3_NET);
    try {
      const small = { family: 'lc0', variant: 'small', strength: 64 };
      const page = analysisPage({ rows: [t3Row], searchers: new Map([['t3', t3]]), smallLc0Runtime: 'onnx' });
      const first = page.analyzeCurrent();
      await until(() => FakeWorker.instances[0]?.release);
      page.stop();
      page.setRows([small], { reloadLc0: false, disposeUnused: true });
      await settled(page);
      assert.deepEqual(page.smallSearchLog, ['onnx'], 'replacement searches before the stopped init resolves');
      assert.deepEqual(FakeWorker.searchLog, []);
      assert.equal(page.admissions.length, 1);
      FakeWorker.instances[0].release();
      await first;
      assert.equal(page.admissions.length, 1, 'old init cannot consume or restart the replacement');
      assert.deepEqual([...page.completeAnalysisKeys], [page.admissions[0].selectionKey]);
    } finally {
      t3.dispose();
    }
  }));

test('the default pair gives a cold Lc0 Small the first line before starting Stockfish Lite', async () => {
  const small = { family: 'lc0', variant: 'small', strength: 64 };
  const sf = { family: 'sf', variant: 'lite', strength: 12 };
  const page = analysisPage({ rows: [small, sf], searchers: new Map() });
  const release = page.holdBackend();
  const running = page.analyzeCurrent();
  try {
    await new Promise(setImmediate);
    assert.equal(page.published.length, 0);
    assert.deepEqual(page.smallSearchLog, []);
  } finally {
    release();
    await running;
  }
  assert.deepEqual(
    page.published[0].map((line) => line.engine),
    ['Lc0'],
  );
  assert.deepEqual(
    page.published.at(-1).map((line) => line.engine),
    ['Lc0', 'SF Lite'],
  );
  assert.equal(page.admissions.length, 1);
  assert.deepEqual([...page.completeAnalysisKeys], [page.analysisSelectionCacheKey()]);
  assert.ok(page.engineLineCache.has(page.engineAnalysisCacheKey(small)), 'cache under the URL actually served');
});

test('position changes reuse a mixed Analysis run’s pending Small load until Stop', async () => {
  const rows = [
    { family: 'lc0', variant: 'small', strength: 64 },
    { family: 'sf', variant: 'lite', strength: 12 },
  ];
  const page = analysisPage({ rows, searchers: new Map() });
  const release = page.holdBackend();
  const first = page.analyzeCurrent();
  try {
    await new Promise(setImmediate);
    page.navigateAutoOff(true);
    const next = page.navigateHistory([parseFen(START_FEN), parseFen(START_FEN)], ['g1f3']);
    await new Promise(setImmediate);
    assert.equal(page.backendStarts.length, 1);
    assert.equal(page.backendStarts[0].aborted, false);
    await first;
    page.stop();
    assert.equal(page.backendStarts[0].aborted, true);
    await next;
  } finally {
    release();
  }
  assert.deepEqual(page.smallSearchLog, []);
});

test('hiding mixed Analysis pauses searches without restarting the Small download on show', async () => {
  const page = analysisPage({
    rows: [
      { family: 'lc0', variant: 'small', strength: 64 },
      { family: 'sf', variant: 'lite', strength: 12 },
    ],
    searchers: new Map(),
  });
  const release = page.holdBackend();
  const first = page.analyzeCurrent();
  try {
    await new Promise(setImmediate);
    assert.equal(page.backendStarts.length, 1);
    page.hide();
    await first;
    assert.equal(page.backendStarts[0].aborted, false);
    assert.deepEqual(page.smallSearchLog, []);
    page.show();
    await new Promise(setImmediate);
    assert.equal(page.backendStarts.length, 1);
  } finally {
    release();
  }
  await settled(page);
  assert.deepEqual(page.smallSearchLog, ['onnx']);
});

test('hiding an automatic initial backend load retains its search on show', async () => {
  const small = { family: 'lc0', variant: 'small', strength: 64 };
  const page = analysisPage({ rows: [small], searchers: new Map() });
  page.el('autoAnalyze').checked = true;
  const release = page.holdBackend();
  const loading = page.loadBackend();
  page.hide();
  release();
  await loading;
  assert.deepEqual(page.admissions, []);
  page.show();
  await settled(page);
  assert.equal(page.admissions.length, 1);
});

test('a hidden search queued behind maintenance keeps Stop enabled', async () => {
  const small = { family: 'lc0', variant: 'small', strength: 64 };
  const page = analysisPage({ rows: [small], searchers: new Map(), smallLc0Runtime: 'onnx' });
  const release = page.holdBackend();
  page.setRuntime('hybrid-b');
  assert.equal(page.el('stop').disabled, false);
  page.hide();
  assert.equal(page.el('stop').disabled, false, 'Stop can still cancel the suspended search');
  release();
  await settled(page);
});

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
