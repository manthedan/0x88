import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import * as pool from '../src/lc0/bigNetSessionPool.ts';
import { BIG_NETS } from '../src/lc0/bt4Engine.ts';
import { createCriticalEngines } from '../src/lc0/criticalMomentsEngines.ts';
import { isMemoryConstrainedDevice, phoneReclaimSettings } from '../src/lc0/deviceMemoryClass.ts';
import { disposeOrtWorker } from '../src/lc0/disposeOrtWorker.ts';
import { isLc0BigNetVariant } from '../src/lc0/engineCatalog.ts';
import * as retirement from '../src/lc0/engineRetirement.ts';
import { Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';
import { untilAborted } from '../src/lc0/modelCache.ts';
import { phoneBigNetBlocked, phoneLargeNetPreference } from '../src/lc0/phoneLargeNetPolicy.ts';
import { EngineResourceBroker } from '../src/lc0/resourceBroker.ts';

const { disposeAllBigNetSearchersNow } = pool;
const { awaitEngineRetirements } = retirement;
const tick = () => new Promise(setImmediate);
const row = (variant) => ({ id: variant, label: variant, family: variant === 'maia3' ? 'maia3' : 'lc0', variant });
const nop = () => {};
const node = () => ({ toggleAttribute() {} });
const play = source('playBrowser');
const analysis = source('analysisBrowser');
const arena = source('arenaBrowser');
const keys = ['bt4', 't3'];

function source(name) {
  return readFileSync(new URL(`../src/lc0/${name}.ts`, import.meta.url), 'utf8');
}

// Link the browser owners to the same production pool and retirement module. Only
// native model/worker I/O and unrelated page rendering are replaced.
function slice(text, start, end, from = 0) {
  const first = text.indexOf(start, from);
  const last = text.indexOf(end, first);
  assert.ok(first >= 0 && last > first, `${start} .. ${end}`);
  return text.slice(first, last);
}
function link(code, deps = {}) {
  const bindings = {
    ...pool,
    ...retirement,
    BIG_NETS,
    isMemoryConstrainedDevice,
    phoneReclaimSettings,
    isLc0BigNetVariant,
    untilAborted,
    el: node,
    ...deps,
  };
  return new Function(...Object.keys(bindings), `return ${stripTypeScriptTypes(`(function run(){\n${code}\n})()`)}`)(...Object.values(bindings));
}

function nativeModels(t, { phone = true, gap = 0, retireCpu = true } = {}) {
  const trace = [];
  const models = [];
  const active = new Set();
  function transition(model, phase) {
    if (phase === 'init') active.add(model);
    if (phase === 'terminated') active.delete(model);
    trace.push({ name: model.name, phase, concurrent: active.size });
  }
  class Worker extends EventTarget {
    start(name) {
      this.name = name;
      models.push(this);
      transition(this, 'init');
    }
    reply(data) {
      this.dispatchEvent(new MessageEvent('message', { data }));
    }
    finishInit(error) {
      this.initFinished = true;
      transition(this, error ? 'init-error' : 'ready');
      if (this.initRequest) this.reply({ type: error ? 'error' : 'ready', id: this.initRequest.id, backend: 'webgpu', error });
      else if (error) this.evaluator.reject(error);
      else {
        const dispose = () => (this.disposal ??= disposeOrtWorker(this));
        this.evaluator.resolve({
          modelLoad: {},
          chooseMove: async () => {},
          evaluateConditions: async () => [{ legalPriors: [] }],
          terminate: () => void dispose(),
          dispose,
        });
      }
    }
    finishDispose(error) {
      assert.ok(this.disposeRequest, `${this.name} is retiring`);
      transition(this, error ? 'dispose-error' : 'disposed');
      this.reply({ type: error ? 'error' : 'disposed', id: this.disposeRequest.id, error });
    }
    terminate() {
      transition(this, 'terminated');
    }
    postMessage(message) {
      if (message.type === 'init') {
        this.initRequest = message;
        this.start(Object.values(BIG_NETS).find((config) => config.modelUrl === message.modelUrl).key);
      } else if (message.type === 'dispose') {
        this.disposeRequest = message;
        transition(this, 'dispose');
      } else if (message.type === 'search') {
        queueMicrotask(() => this.reply({ type: 'searchResult', id: message.id, result: { cancelled: false } }));
      } else if (message.type === 'resetSearch') {
        queueMicrotask(() => this.reply({ type: 'resetDone', id: message.id }));
      }
    }
  }
  const previousLocation = globalThis.location;
  const previousWorker = globalThis.Worker;
  const previousStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const stored = new Map();
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value) },
  });
  // The original residency cases isolate terminal disposal from the optional time gap.
  globalThis.location = new URL(`https://test.invalid/?memoryClass=${phone ? 'phone' : 'desktop'}&0x88-reclaim-gap-ms=${gap}&0x88-retire-cpu=${+retireCpu}`);
  globalThis.Worker = Worker;
  phoneLargeNetPreference.setEnabled(true); // Consent has been granted; phone residency must still apply.
  t.after(async () => {
    disposeAllBigNetSearchersNow();
    for (const model of active) if (model.evaluator && !model.initFinished) model.finishInit();
    await tick();
    for (const model of active) if (model.disposeRequest) model.finishDispose();
    await awaitEngineRetirements();
    phoneLargeNetPreference.setEnabled(false);
    globalThis.Worker = previousWorker;
    globalThis.location = previousLocation;
    if (previousStorage) Object.defineProperty(globalThis, 'localStorage', previousStorage);
    else delete globalThis.localStorage;
  });
  assert.equal(isMemoryConstrainedDevice(), phone);
  assert.equal(phoneBigNetBlocked('bt4'), false);
  return {
    trace,
    StockfishEngine: class {
      constructor(options, url) {
        this.options = options;
        this.url = url;
      }
      analyze(depth) {
        if (!this.worker) {
          this.worker = new Worker();
          this.worker.start('sf-lite');
        }
        this.depth = depth;
      }
      dispose() {
        retirement.terminateEngineWorker(this.worker);
      }
    },
    createMaia({ signal }) {
      const model = new Worker();
      model.signal = signal;
      model.evaluator = Promise.withResolvers();
      model.start('maia3');
      // Deliberately ignore cancellation until a tardy native init completes.
      return model.evaluator.promise;
    },
    async get(name) {
      await tick();
      const model = models.find((candidate) => candidate.name === name);
      assert.ok(model, `${name} init: ${JSON.stringify(trace)}`);
      return model;
    },
    initializing() {
      return models.map((model) => model.name);
    },
    assertPeak(expected = 1) {
      assert.equal(Math.max(0, ...trace.map((entry) => entry.concurrent)), expected, JSON.stringify(trace));
    },
  };
}

function playOwner(native) {
  const pgn = '1. e4 *';
  const links = [],
    handoffs = [];
  const owner = link(
    `
    ${slice(play, '  const retireCpuOnUnmount =', '  const releaseDom =', play.indexOf('export function mountPlayBrowser('))}
    const ctx={abort:new AbortController(),consentAbort:new AbortController(),enginePreparationGeneration:0,
      board:{turn:'w'},humanColor:'w',positions:[],moves:[],listeners:[],opponent:{active:{maiaElo:1500}}};
    let option=maia;
    let maia3Promise=null,maia3Status=null,maia3Abort=null,lc0LoadPromise=null;
    const playCentipawn={clear(){}},cpuEnginePromises=new Map();
    const ctxSelectedEngine=()=>option;
    const hook=()=>{},releaseDom=()=>{};
    ${slice(play, 'function retireUnusedPlayNeuralEngines(', 'function ctxEnsureLc0Small(')}
    ${slice(play, 'async function ctxEnsureMaia3(', 'function ctxCpuEngineFor(')}
    ${slice(play, 'async function ctxVerifySelectedEngineReady(', 'function ctxPriorFens(')}
    ${slice(play, "  for (const link of document.querySelectorAll<HTMLAnchorElement>('[data-analyze]'))", "  trackListener(ctx, el('downloadPgn')", play.indexOf('function ctxInit('))}
    function teardown(){${slice(play, '  return () => {\n    try {\n      ctx.autosave?.flush();', '\n  };', play.indexOf('export function mountPlayBrowser('))}\n    };}
    return {prepare:()=>ctxVerifySelectedEngineReady(ctx),ready:()=>ctx.engineReady,
      select:(next)=>{ctxCancelEngineTurn(ctx);option=next;return ctxVerifySelectedEngineReady(ctx);},
      finishMove:()=>ctx.moves.push('e2e4'),unmount:teardown()};`,
    {
      maia: row('maia3'),
      buttonEl: node,
      ctxAsyncUiIsCurrent: () => false,
      ctxSetEngineNote: nop,
      ctxRender: nop,
      ctxHideDownloadProgress: nop,
      ctxDisarmNewGame: nop,
      ctxSetEngineFailure: (_ctx, error) => assert.fail(error),
      ctxLqoSearchOptions: () => ({}),
      Maia3BrowserEvaluator: { create: native.createMaia },
      playRuntimeFallbackAudits: new Map(),
      disposeCachedCpuEngine: nop,
      PLAY_DRAW_SCORE: 0,
      PLAY_SEARCH_CONTEMPT_LIMIT: 0,
      document: { querySelectorAll: () => [{}] },
      trackListener: (_ctx, _target, _type, listener) => links.push(listener),
      ctxExportPgn: () => pgn,
      writeAnalysisHandoff: (value) => handoffs.push(value),
    },
  );
  return {
    ...owner,
    analyzeThisGame() {
      owner.finishMove();
      links[0]({ preventDefault: nop });
      owner.unmount();
      return handoffs.pop();
    },
  };
}

function analysisOwner(native) {
  const message = {};
  return link(
    `
    const mountAbort=new AbortController();
    let explorerMaia=null, activeAnalysisSelectionKey='';
    const analysisRunState={revision:0,execution:{tag:'idle'}},owner={},analysisOperations=new Map();
    const activeAnalysisRunId=1,hiddenTab={hidden:false};
    const automaticAnalysisAllowed=()=>true,analysisIntent=()=>({});
    let analysisAbort=null, criticalDeferredMaia=false,stockfishLite=null,stockfishFull=null;
    ${slice(analysis, 'let analysisReclaimOwner:', 'function deferAnalysisSettingsRefresh(')}
    const analysisReloadAbort=null,activeWorkerSearchId=null,searchWorker=null;
    ${slice(analysis, 'function cancelAnalysisOperation(', 'function scheduleAnalyzeCurrent(')}
    const tree={current:{fen:''}},preparedCpuEngineKeys=new Set(),cpu={disposeAll(){}};
    const recklessEngines=cpu,viridithasEngines=cpu,berserkEngines=cpu,plentyChessEngines=cpu,stormphraxEngines=cpu;
    ${slice(analysis, 'function threadedStockfishAvailable()', 'const ANALYSIS_CPU_ENGINE_OPTIONS')}
    ${slice(analysis, 'function disposeAnalysisCpuEngines()', 'function disposeUnusedEngines()')}
    const criticalCallbacks={${slice(analysis, '    pauseAnalysis: (maiaOwner) => {', '    updateView: (moment, lines) => {')}};
    ${slice(analysis, 'async function bigNetFor(', '\n}', analysis.indexOf('async function bigNetFor('))}}
    ${slice(analysis, 'const ANALYSIS_BIG_NET_SESSION', '/** After initBigNetsForRun')}
    ${slice(analysis, 'const MAIA3_PANEL_RATINGS', 'function scheduleMaia3Panel()')}
    async function prepare(rows, controller = new AbortController()){
      analysisOperations.set(1,controller);
      analysisAbort=controller;
      try {
        ${slice(analysis, '    const bigNetInitErrors =', '    snapshotBigNetServedModels(rows);')}
        for (const row of rows) if (row.family === 'sf') getStockfish('lite').analyze(row.strength);
        return bigNetInitErrors;
      } finally {
        ${slice(analysis, '    if (maia3PanelBlockedByRun === controller)', '\n    }', analysis.indexOf('async function performAnalysis(owner:'))}
        }
        analysisOperations.delete(1);
        analysisAbort=null;
      }
    }
    return {prepare,cancel:()=>cancelAnalysisOperation(1),stop:()=>{stopAnalysisBigNetInit();cancelAnalysisOperation(1);},changeRows:()=>invalidateAnalysisSettings({disposeUnused:true},true),enableMaia3Panel,unloadMaia3Panel,...criticalCallbacks,getStockfish,message,disposeAnalysisCpuEngines};`,
    {
      ANALYSIS_BIG_NET_KEYS: keys,
      message,
      el: (id) => (id === 'message' ? message : node()),
      selectEl: node,
      retireAnalysisRunCentipawnKeys: nop,
      StockfishEngine: native.StockfishEngine,
      stockfishFlavorUrl: (flavor) => flavor,
      cpuEngineHashMbForSurface: () => 128,
      analysisCanPrepare: () => true,
      maybeEl: node,
      dispatchAnalysis: nop,
      clearAnalysisSearchProgress: nop,
      criticalAnalysisRefresh: { take: nop },
      analysisSelectionCacheKey: nop,
      isStaleMount: (signal) => signal.aborted,
      Maia3BrowserEvaluator: { create: native.createMaia },
      showModelProgress: nop,
      hideModelProgress: nop,
      renderExplorerComparison: nop,
      scheduleMaia3Panel: nop,
      renderMaia3Panel: nop,
      LatestOnlyTask: class {
        dispose() {}
      },
    },
  );
}

function arenaOwner(disposeArenaCpuEngines = nop) {
  return link(
    `
    let seats=[],running=false,startPending=false,lc0Cache=null,loadingLc0=false;
    const engines=new Map(), warmupPositions=[];
    ${slice(arena, 'async function bigNetFor(', 'let runtimeIsolation')}
    ${slice(arena, 'function reconcileArenaNeuralSessions()', 'function buildEngines()')}
    ${slice(arena, '  const lc0BigNetWarmup =', '  const centipawnWarmup =')}
    ${slice(arena, 'async function warmUpSelectedEngines(', 'function legalFromUci(')}
    const activeSeatRows=()=>seats;
    return {
      seat:(key)=>{seats=[row(key)];reconcileArenaNeuralSessions();engines.set(key,{bigNet:true,warmup:lc0BigNetWarmup(key)});},
      addCpu:(warmup)=>engines.set('cpu',{warmup}),
      warmup:(key,signal)=>warmUpSelectedEngines([key],signal,0),
      warmupPair:(key,signal)=>warmUpSelectedEngines(['cpu',key],signal,0),
    };`,
    {
      row,
      ARENA_BIG_NET_KEYS: keys,
      arenaRunDownloads: new Set(),
      requestLargeNetDownloadConsent: async () => 'cached',
      disposeArenaCpuEngines,
      arenaCacheEntries: () => 2048,
      showDownloadProgress: nop,
      hideDownloadProgress: nop,
      showEngineWarmupStatus: nop,
      hideEngineWarmupStatus: nop,
      renderCacheInfo: nop,
    },
  );
}

for (const crossRoute of [false, true]) {
  test(`phone keeps one heavy model when ${crossRoute ? 'Play hands off to Analysis BT4' : 'Play switches from Maia3 to BT4'}`, async (t) => {
    const native = nativeModels(t);
    const play = playOwner(native);
    const old = play.prepare();
    const maia = await native.get('maia3');
    let replacement;
    if (crossRoute) {
      assert.equal(play.analyzeThisGame(), '1. e4 *');
      replacement = analysisOwner(native).prepare([row('bt4')]);
    } else replacement = play.select(row('bt4'));
    assert.equal(maia.signal.aborted, true);
    await tick();
    assert.deepEqual(native.initializing(), ['maia3']);
    maia.finishInit(); // Old owner resolves AFTER the picker/handoff has cancelled it.
    await tick();
    assert.deepEqual(native.initializing(), ['maia3']);
    maia.finishDispose();
    (await native.get('bt4')).finishInit();
    await old;
    const result = await replacement;
    if (crossRoute) assert.equal(result.size, 0);
    else assert.equal(play.ready(), true);
    native.assertPeak();
  });
}

test('phone Play waits for a retiring BT4 worker before admitting Queen Odds', async (t) => {
  const native = nativeModels(t);
  const play = playOwner(native);
  const old = play.select(row('bt4'));
  const bt4 = await native.get('bt4');
  bt4.finishInit();
  await old;
  const replacement = play.select(row('lqo'));
  await tick();
  assert.deepEqual(native.initializing(), ['bt4']);
  bt4.finishDispose('native release rejected');
  (await native.get('lqo')).finishInit();
  await replacement;
  assert.equal(play.ready(), true);
  native.assertPeak();
});

test('phone Analysis default plus Maia3 switches to t3 after rejected worker release and termination', async (t) => {
  const native = nativeModels(t);
  const owner = analysisOwner(native);
  const load = owner.enableMaia3Panel();
  const maia = await native.get('maia3');
  maia.finishInit();
  await load;
  const replacement = owner.prepare([row('t3')]);
  await tick();
  assert.deepEqual(native.initializing(), ['maia3']);
  maia.finishDispose('native release rejected');
  (await native.get('t3')).finishInit();
  assert.equal((await replacement).size, 0);
  native.assertPeak();
});

test('phone Arena seat swap waits for terminal disposal of a cancelled BT4 load before t3', async (t) => {
  const native = nativeModels(t);
  const arena = arenaOwner();
  const firstMatch = new AbortController();
  arena.seat('bt4');
  // Stop retires the loading net (bigNetFor); its cache-only init then reports the Stop.
  const loading = assert.rejects(arena.warmup('bt4', firstMatch.signal), { name: 'AbortError' });
  const bt4 = await native.get('bt4');
  firstMatch.abort();
  arena.seat('t3');
  const replacement = arena.warmup('t3', new AbortController().signal);
  await loading;
  bt4.finishInit(); // Stale ready notification; BT4 still counts until terminate().
  await tick();
  assert.deepEqual(native.initializing(), ['bt4']);
  bt4.finishDispose();
  (await native.get('t3')).finishInit();
  await replacement;
  native.assertPeak();
});

test('phone Analysis with BT4 and t3 rows admits only the first large net', async (t) => {
  const native = nativeModels(t);
  const run = analysisOwner(native).prepare([row('bt4'), row('t3')]);
  (await native.get('bt4')).finishInit();
  assert.match((await run).get('t3')?.message, /one large net/);
  assert.deepEqual(native.initializing(), ['bt4']);
  native.assertPeak();
});

test('phone critical-moment Maia3 replaces BT4 and retires before the next net', async (t) => {
  const native = nativeModels(t);
  t.mock.method(Maia3BrowserEvaluator, 'create', native.createMaia);
  const owner = analysisOwner(native);
  const run = owner.prepare([row('bt4')]);
  const bt4 = await native.get('bt4');
  bt4.finishInit();
  assert.equal((await run).size, 0);

  const report = new AbortController();
  const engines = createCriticalEngines(new EngineResourceBroker({ policy: 'exclusive', dial: 'balanced' }), 1500, report.signal);
  owner.pauseAnalysis(report);
  const comparing = engines.lanes.find((lane) => lane.id === 'maia').evaluate({ fen: '' }, report.signal, nop);
  await owner.enableMaia3Panel(); // The interactive Maia3 owner cannot enter while the report owns the slot.
  await tick();
  assert.deepEqual(native.initializing(), ['bt4']);
  bt4.finishDispose();
  const maia = await native.get('maia3');
  maia.finishInit();
  await comparing;
  engines.dispose();
  owner.settled(report);

  const replacement = owner.prepare([row('t3')]);
  await tick();
  assert.deepEqual(native.initializing(), ['bt4', 'maia3']);
  maia.finishDispose();
  (await native.get('t3')).finishInit();
  assert.equal((await replacement).size, 0);
  native.assertPeak();
});

let fakeEpoch = 0;
function fakeTime(t) {
  fakeEpoch += 1_000_000;
  let now = fakeEpoch;
  t.mock.method(performance, 'now', () => now);
  t.mock.timers.enable({ apis: ['setTimeout'] });
  return async (ms) => {
    now += ms;
    t.mock.timers.tick(ms);
    await tick();
  };
}

for (const [name, phone, gap, elapsed, remaining] of [
  ['recent', true, 30000, 6000, 24000],
  ['old', true, 20000, 21000, 0],
  ['disabled', true, 0, 1, 0],
  ['desktop', false, 20000, 1, 0],
]) {
  test(`${name}: Analysis counts reclaim time from the Play worker termination`, async (t) => {
    const native = nativeModels(t, { phone, gap });
    const advance = fakeTime(t);
    const play = playOwner(native);
    const loading = play.prepare();
    const maia = await native.get('maia3');
    maia.finishInit();
    await loading;
    // SvelteKit may already expose the destination URL when Play unmounts.
    globalThis.location = new URL(`https://test.invalid/analysis?memoryClass=${phone ? 'phone' : 'desktop'}`);
    assert.equal(play.analyzeThisGame(), '1. e4 *');
    await tick();
    maia.finishDispose();
    await awaitEngineRetirements();
    await advance(elapsed);
    const analysis = analysisOwner(native);
    const run = analysis.prepare([row('bt4')]);
    if (!phone) assert.equal(analysis.message.textContent, `Loading Lc0 ${BIG_NETS.bt4.name}…`, 'desktop reaches model init without an added yield');
    if (remaining) {
      assert.equal(analysis.message.textContent, 'Freeing memory…');
      await advance(remaining - 1);
      assert.deepEqual(native.initializing(), ['maia3']);
      await advance(1);
    }
    (await native.get('bt4')).finishInit();
    assert.equal((await run).size, 0);
    native.assertPeak();
  });
}

test('cancelling Analysis during reclaim does not start a model or restart a retired CPU row', async (t) => {
  const native = nativeModels(t, { gap: 20000 });
  const advance = fakeTime(t);
  const owner = analysisOwner(native);
  const cpu = { family: 'sf', variant: 'lite', strength: 12 };
  await owner.prepare([cpu]);
  const controller = new AbortController();
  const run = assert.rejects(owner.prepare([cpu, row('bt4')], controller), { name: 'AbortError' });
  assert.equal(owner.message.textContent, 'Freeing memory…');
  owner.cancel();
  assert.equal(owner.message.textContent, 'Analysis stopped.');
  await run;
  await advance(20000);
  assert.deepEqual(
    native.trace.map(({ name, phase }) => `${name}:${phase}`),
    ['sf-lite:init', 'sf-lite:terminated'],
  );
});

test('removing the large row cancels reclaim so CPU-only Analysis can restart without the timer', async (t) => {
  const native = nativeModels(t, { gap: 20000 });
  fakeTime(t);
  const owner = analysisOwner(native);
  t.after(owner.disposeAnalysisCpuEngines);
  const cpu = { family: 'sf', variant: 'lite', strength: 12 };
  await owner.prepare([cpu]);
  const obsolete = assert.rejects(owner.prepare([cpu, row('bt4')]), { name: 'AbortError' });
  assert.equal(owner.message.textContent, 'Freeing memory…');
  owner.changeRows();
  await obsolete;
  await owner.prepare([cpu]);
  assert.deepEqual(native.initializing(), ['sf-lite', 'sf-lite']);
});

for (const retireCpu of [true, false]) {
  test(`Analysis CPU reclaim override=${+retireCpu}: the selected SF Lite row resumes at the same settings`, async (t) => {
    const native = nativeModels(t, { gap: 20000, retireCpu });
    const advance = fakeTime(t);
    const owner = analysisOwner(native);
    t.after(owner.disposeAnalysisCpuEngines);
    const cpu = { family: 'sf', variant: 'lite', strength: 12 };
    await owner.prepare([cpu]);
    const previous = owner.getStockfish('lite');
    const run = owner.prepare([cpu, row('bt4')]);
    if (retireCpu) {
      assert.equal(native.trace.at(-1).phase, 'terminated');
      await advance(19999);
      assert.deepEqual(native.initializing(), ['sf-lite']);
      await advance(1);
    }
    (await native.get('bt4')).finishInit();
    await run;
    const current = owner.getStockfish('lite');
    assert.equal(current === previous, !retireCpu);
    assert.equal(current.depth, cpu.strength);
    assert.equal(current.url, previous.url);
    assert.deepEqual(current.options, previous.options);
    if (retireCpu)
      assert.deepEqual(
        native.trace.slice(-2).map(({ name, phase }) => `${name}:${phase}`),
        ['bt4:ready', 'sf-lite:init'],
      );
  });
}

test('Arena warms the large net before restarting its retired CPU opponent, even with the CPU seat first', async (t) => {
  const native = nativeModels(t, { gap: 20000 });
  const advance = fakeTime(t);
  const cpu = new native.StockfishEngine({}, 'lite');
  cpu.analyze(1);
  const arena = arenaOwner(() => cpu.dispose());
  arena.seat('bt4');
  const restarted = new native.StockfishEngine({}, 'lite');
  arena.addCpu(() => restarted.analyze(1));
  const run = arena.warmupPair('bt4', new AbortController().signal);
  await tick();
  assert.equal(native.trace.at(-1).phase, 'terminated');
  await advance(20000);
  (await native.get('bt4')).finishInit();
  await run;
  assert.deepEqual(
    native.trace.slice(-2).map(({ name, phase }) => `${name}:${phase}`),
    ['bt4:ready', 'sf-lite:init'],
  );
  restarted.dispose();
});

for (const phone of [true, false]) {
  test(`${phone ? 'phone' : 'desktop'} Analysis: enabling Maia3 after a completed BT4 run`, async (t) => {
    const native = nativeModels(t, { phone });
    const owner = analysisOwner(native);
    const first = owner.prepare([row('bt4')]);
    const bt4 = await native.get('bt4');
    bt4.finishInit();
    assert.equal((await first).size, 0);
    const maia = owner.enableMaia3Panel();
    await tick();
    if (phone) {
      assert.deepEqual(native.initializing(), ['bt4']);
      bt4.finishDispose();
    }
    const model = await native.get('maia3');
    model.finishInit();
    await maia;
    owner.unloadMaia3Panel('Done');
    await tick();
    model.finishDispose();
    native.assertPeak(phone ? 1 : 2);
  });
}

test('phone Analysis Stop during a first BT4 download ends the load, then Maia3 waits for its retirement', async (t) => {
  const native = nativeModels(t);
  const owner = analysisOwner(native);
  const run = owner.prepare([row('bt4')]);
  const bt4 = await native.get('bt4');
  owner.stop();
  // The download never finishes: the run must settle without it.
  let timer;
  const stuck = new Promise((resolve) => {
    timer = setTimeout(() => resolve('stuck'), 200);
  });
  try {
    assert.notEqual(await Promise.race([run, stuck]), 'stuck', 'Stop left the run waiting for the big-net download');
  } finally {
    clearTimeout(timer);
  }
  assert.equal(bt4.disposeRequest?.type, 'dispose');
  assert.equal(owner.message.textContent, 'Analysis stopped.', 'Stop replaces the loading status');
  const maia = owner.enableMaia3Panel();
  await tick();
  assert.deepEqual(native.initializing(), ['bt4'], 'Maia3 must wait for the BT4 worker to retire');
  bt4.finishDispose();
  const model = await native.get('maia3');
  model.finishInit();
  await maia;
  native.assertPeak();
});
