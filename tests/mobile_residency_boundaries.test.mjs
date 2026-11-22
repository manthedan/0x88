import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { abortSignalAny } from '../src/lc0/abortSignalAny.ts';
import { analysisCanPrepare } from '../src/lc0/analysisRunState.ts';
import { loadRetirableResource } from '../src/lc0/arenaResourceLoad.ts';
import { awaitEngineRetirements, trackEngineRetirement } from '../src/lc0/engineRetirement.ts';
import { untilAborted } from '../src/lc0/modelCache.ts';
import { Lc0WholeOnnxWebgpuEvaluator } from '../src/lc0/wholeOnnxWebgpuEvaluator.ts';
import { testAnalysisEngines } from './lib/analysis_engines.mjs';

const source = (name) => readFileSync(new URL(`../src/lc0/${name}.ts`, import.meta.url), 'utf8');
function extract(text, first, last) {
  const start = text.indexOf(first);
  return text.slice(start, text.indexOf(last, start));
}
function compile(body, deps) {
  deps = { untilAborted, ...deps };
  return new Function(...Object.keys(deps), `return (${stripTypeScriptTypes(`function run(){${body}\n}`)})();`)(...Object.values(deps));
}

test('canceling a setup Small load restores Start and progress even though its loader no longer owns cleanup', () => {
  const elements = new Map([
    [
      'start',
      {
        disabled: true,
        toggleAttribute(_name, disabled) {
          this.disabled = disabled;
        },
      },
    ],
    ['message', { textContent: 'Loading…' }],
  ]);
  let hidden = false,
    refreshed = false,
    canceled = false;
  const state = compile(
    `
    let running=false,loadingLc0=true,lc0Cache=null;
    const startPending=false;
    const disposeLc0Resources=()=>{loadingLc0=false;canceled();};
    ${extract(source('arenaBrowser'), 'function reconcileArenaNeuralSessions():', 'function buildEngines()')}
    reconcileArenaNeuralSessions();return loadingLc0;
  `,
    {
      canceled: () => {
        canceled = true;
      },
      activeSeatRows: () => [{ family: 'sf', variant: 'lite' }],
      ARENA_BIG_NET_KEYS: [],
      isLc0BigNetVariant: () => false,
      hideDownloadProgress: () => {
        hidden = true;
      },
      refreshSeatControls: () => {
        refreshed = true;
      },
      el: (id) => elements.get(id),
      readyMessage: () => 'Ready.',
      setArenaStartDisabled: (disabled) => {
        elements.get('start').disabled = disabled;
      },
    },
  );
  assert.equal(state, false);
  assert.equal(canceled, true);
  assert.equal(hidden, true);
  assert.equal(refreshed, true);
  assert.equal(elements.get('start').disabled, false);
  assert.equal(elements.get('message').textContent, 'Ready.');
});

test('Analysis runtime teardown publishes main-thread fallback release to the replacement barrier', async () => {
  const release = Promise.withResolvers();
  const dispose = compile(
    `
    let analysisAbort=null,reviewAbort=null,activeWorkerSearchId=null;
    let mainEvaluator={dispose:()=>release},searcher={},lc0ServedModelUrl='small';
    ${extract(source('analysisBrowser'), 'function disposeRuntimeResources():', 'function disposePageResources():')}
    return disposeRuntimeResources;
  `,
    {
      release: release.promise,
      trackEngineRetirement,
      engines: testAnalysisEngines(),
      cancelAllAnalysisAssetPrefetches: () => {},
      resetSearchWorker: () => {},
    },
  );
  dispose();
  let admitted = false;
  const waiting = awaitEngineRetirements().then(() => {
    admitted = true;
  });
  await new Promise(setImmediate);
  assert.equal(admitted, false);
  release.resolve();
  await waiting;
  assert.equal(admitted, true);
  assert.match(
    source('analysisBrowser'),
    /loadRetirableResource\(async \(\) => \(uninstalled = await createFallback\(\)\), fallbackSignal\)/,
    'canceled fallback creation uses the same tracked boundary as Arena',
  );
});

test('Analysis teardown while fallback diagnostics are pending still retires the uninstalled evaluator at once', async (t) => {
  t.mock.method(console, 'warn', () => undefined);
  const release = Promise.withResolvers();
  const diagnosticsGate = Promise.withResolvers();
  let diagnosticsAsked = false;
  const operation = new AbortController();
  const element = () => ({ textContent: '', disabled: false, toggleAttribute() {} });
  const load = compile(
    `
    let mainEvaluator=null,searcher=null,lc0ServedModelUrl='';
    ${extract(source('analysisBrowser'), 'async function loadLc0BackendOwned(', 'async function reloadLc0Backend(')}
    return { load: loadLc0BackendOwned, installed: () => mainEvaluator };
  `,
    {
      loadRetirableResource,
      trackEngineRetirement,
      awaitEngineRetirements,
      selectedLc0Runtime: () => 'onnx',
      lc0RuntimeLabel: () => 'ONNX',
      el: element,
      selectEl: element,
      showModelProgress: () => {},
      hideModelProgress: () => {},
      smallWorkerRecovery: { provider: (preference) => preference },
      retryWorkerDeviceLoss: (work) => work(),
      initWorker: async () => {
        throw new Error('worker init failed');
      },
      mountAbort: new AbortController(),
      LC0_WHOLE_MODEL_WEBGPU_RUNTIME: 'whole-onnx-webgpu',
      MODEL_URL: 'small.onnx',
      LC0_DEVICE_MODEL_SELECTION: true,
      createLc0OnnxEvaluatorFromUrl: async () => ({ servedUrl: 'small.onnx', evaluator: { dispose: () => release.promise } }),
      collectOrtRuntimeDiagnostics: () => {
        diagnosticsAsked = true;
        return diagnosticsGate.promise;
      },
      CachedLc0Evaluator: class {},
      Lc0PuctSearcher: class {},
      publishBrowserRuntimeAudit: () => assert.fail('a cancelled fallback is not published'),
      multiPv: () => 1,
    },
  );
  const loading = load.load(() => operation.signal.aborted, operation.signal);
  for (let i = 0; i < 100 && !diagnosticsAsked; i++) await new Promise(setImmediate);
  assert.equal(diagnosticsAsked, true, `fallback reached diagnostics (load: ${await Promise.race([loading, 'pending'])})`);
  operation.abort(); // route teardown cancels the load during the diagnostics wait
  let admitted = false;
  const barrier = awaitEngineRetirements().then(() => {
    admitted = true;
  });
  await new Promise(setImmediate);
  assert.equal(admitted, false, 'the next route waits for the uninstalled session');
  release.resolve();
  await barrier;
  diagnosticsGate.resolve({ describe: 'wasm' });
  assert.equal(await loading, false);
  assert.equal(load.installed(), null);
});

for (const stale of [false, true])
  test(`Analysis rechecks scheduler revision after retirement (${stale})`, async () => {
    const text = source('analysisBrowser'),
      start = text.indexOf('    if (hasEngineRetirements()) await awaitEngineRetirements();', text.indexOf('async function performAnalysis(owner:'));
    const guard = text.slice(start, text.indexOf('    const uninitializedBigNets', start));
    const owner = { runId: 1, preparationRevision: 1 },
      analysisRunState = {
        execution: { tag: 'preparing', owner },
        revision: 1,
        visible: true,
        modes: [],
        wanted: null,
        maintenance: { reloadLc0: false, disposeUnused: false },
      };
    assert.equal(analysisCanPrepare(analysisRunState, owner), true);
    const gate = Promise.withResolvers(),
      controller = new AbortController();
    let admissions = 0;
    const run = compile(`return async function(){${guard} admit();};`, {
      hasEngineRetirements: () => true,
      awaitEngineRetirements: () => gate.promise,
      controller,
      mountAbort: controller,
      analysisCanPrepare,
      analysisRunState,
      owner,
      admit: () => {
        admissions++;
      },
    });
    const pending = run();
    if (stale) analysisRunState.revision++;
    gate.resolve();
    await pending;
    assert.equal(admissions, stale ? 0 : 1);
  });

test('Arena Stop interrupts a Centipawn warmup consumer without canceling the selected shared owner', async () => {
  const loading = Promise.withResolvers(),
    controller = new AbortController();
  let searches = 0,
    request;
  const text = source('arenaBrowser');
  const warmup = compile(
    `
    ${extract(text, 'function centipawnEvaluator(', 'function disposeUnusedCentipawnEvaluators(')}
    ${extract(text, '  const centipawnWarmup =', '  const stockfishWarmupStatus =')}
    return centipawnWarmup({variant:'ort'});
  `,
    {
      untilAborted,
      centipawnRuntimeForVariant: () => 'ort',
      centipawnRuntimeFallbackForVariant: () => false,
      centipawnModel: () => ({ modelId: 's4', onnx: 'model', meta: 'meta' }),
      centipawnEvaluatorCacheKey: () => 'held',
      centipawnCache: {
        get: (_key, value) => {
          request = value;
          return loading.promise;
        },
      },
      centipawnEngineLabel: () => 'Centipawn',
      CENTIPAWN_HYBRID_MANIFEST_URL: 'manifest',
      params: new URLSearchParams(),
      arenaCacheEntries: () => 128,
      warmupPositions: [{}],
      centipawnSearchOptions: {},
      chooseMove: async () => {
        searches++;
      },
    },
  );
  const pending = warmup(controller.signal),
    settled = pending.then(
      () => 'completed',
      (error) => error.name,
    );
  controller.abort();
  const result = await Promise.race([settled, new Promise((resolve) => setImmediate(() => resolve('still-waiting')))]);
  loading.resolve({});
  await new Promise(setImmediate);
  assert.equal(result, 'AbortError');
  assert.equal(searches, 0);
  assert.equal(request.options.signal, undefined, 'a match consumer does not own the selected seat cache');
});

test('Arena setup runtime edits invalidate resources but do not eagerly allocate Small', async () => {
  const elements = new Map(),
    events = [];
  const reload = compile(
    `let running=false,startPending=false;${extract(source('arenaBrowser'), 'async function reloadLc0Evaluator():', 'function setBenchResult(')}return reloadLc0Evaluator;`,
    {
      abort: { abort: () => events.push('abort') },
      disposeLc0Resources: () => events.push('dispose'),
      reconcileActiveRuntimeFallbackWarning: () => {},
      renderCacheInfo: () => {},
      hideDownloadProgress: () => events.push('hide'),
      refreshSeatControls: () => events.push('controls'),
      renderRuntimeBadge: async () => {},
      readyMessage: () => 'Ready',
      el: (id) => {
        if (!elements.has(id))
          elements.set(id, {
            disabled: true,
            toggleAttribute(_name, disabled) {
              this.disabled = disabled;
            },
          });
        return elements.get(id);
      },
      loadLc0Evaluator: () => assert.fail('only Start/autorun may load a setup runtime'),
    },
  );
  await reload();
  assert.deepEqual(events, ['abort', 'dispose', 'hide', 'controls']);
  assert.equal(elements.get('start').disabled, false);
  assert.equal(elements.get('message').textContent, 'Ready');
});

test('Arena forwards cancellation into whole-model initialization before any fetch', async () => {
  const create = compile(
    `${extract(source('arenaBrowser'), 'async function createSelectedLc0Evaluator(', 'async function loadLc0Evaluator(')}return createSelectedLc0Evaluator;`,
    {
      selectedLc0Runtime: () => 'whole',
      LC0_WHOLE_MODEL_WEBGPU_RUNTIME: 'whole',
      LC0_WHOLE_MODEL_MANIFEST_URL: '/runtimes/lc0-tvmjs-webgpu/t1/f16/v1/manifest.json',
      lc0WholeModelPhysicalBatch: () => 4,
      lc0WholeModelTensorCache: () => true,
      Lc0WholeOnnxWebgpuEvaluator,
    },
  );
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(create(controller.signal), { name: 'AbortError' });
});

for (const reason of ['navigation', 'seat cancellation', 'diagnostics failure'])
  test(`Arena retains native ownership through asynchronous cache handoff: ${reason}`, async () => {
    const diagnostics = Promise.withResolvers(),
      started = Promise.withResolvers(),
      release = Promise.withResolvers(),
      mount = new AbortController();
    let calls = 0;
    const evaluator = {
      dispose: () => {
        calls++;
        return release.promise;
      },
    };
    const api = compile(
      `
    let lc0LoadAbort=null,loadingLc0=false,lc0Cache=null;
    const startPending=false,running=false;
    ${extract(source('arenaBrowser'), 'async function loadLc0Evaluator(', 'async function reloadLc0Evaluator(')}
    return {load:()=>loadLc0Evaluator(),cancel:()=>lc0LoadAbort.abort(),peek:()=>lc0Cache};
  `,
      {
        mountAbort: mount,
        abortSignalAny,
        isStaleMount: (s) => s.aborted,
        selectedLc0Runtime: () => 'onnx',
        selectedSeatsNeedLc0Evaluator: () => true,
        setArenaStartDisabled: () => {},
        el: () => ({ toggleAttribute() {} }),
        refreshSeatControls: () => {},
        readyMessage: () => 'Ready',
        lc0HybridConfigLabel: () => 'onnx',
        lc0RuntimeLabel: () => 'onnx',
        showDownloadProgress: () => {},
        hideDownloadProgress: () => {},
        awaitEngineRetirements,
        trackEngineRetirement,
        loadRetirableResource,
        createSelectedLc0Evaluator: async () => evaluator,
        collectOrtRuntimeDiagnostics: () => {
          started.resolve();
          return diagnostics.promise;
        },
      },
    );
    const loading = api.load();
    await started.promise;
    if (reason === 'navigation') mount.abort();
    else if (reason === 'seat cancellation') api.cancel();
    else {
      diagnostics.reject(new Error('diagnostics failed'));
      await loading;
    }
    let retired = false;
    const waiting = awaitEngineRetirements().then(() => {
      retired = true;
    });
    await new Promise(setImmediate);
    assert.equal(calls, 1);
    assert.equal(retired, false);
    assert.equal(api.peek(), null);
    release.resolve();
    await waiting;
    if (reason !== 'diagnostics failure') diagnostics.resolve({});
    await loading;
    assert.equal(calls, 1, 'native disposal is idempotent across cancellation and the resumed handoff');
    assert.equal(api.peek(), null);
  });

for (const outcome of ['failure', 'cancelled', 'ready']) {
  test(`Arena Small preparation preserves old exportable games until ready (${outcome})`, async () => {
    const text = source('arenaBrowser');
    const setup = extract(text, '  if (selectedSeatsNeedLc0Evaluator() && !lc0Cache) {', '  const runCheckpoint =');
    assert.match(text, /const runCheckpoint = [^;]+;[\s\S]*await saveArenaCheckpoint\(runCheckpoint,[\s\S]*games\.length = 0;/);
    const load = Promise.withResolvers();
    const matchAbort = new AbortController();
    const mountSignal = new AbortController().signal;
    const cancel = { hidden: true };
    const message = { textContent: '' };
    const run = compile(
      `
      return async function(){
        const abort=matchAbort;
        const startupCancelled=()=>abort.signal.aborted;
        let lc0Cache=null;
        const loadLc0Evaluator=async (_mount, signal)=>{
          if(signal!==matchAbort.signal) throw new Error('setup load must be cancellable');
          lc0Cache=await load;
        };
        ${setup}
        replaceResults();
      };`,
      {
        load: load.promise,
        matchAbort,
        mountSignal,
        selectedSeatsNeedLc0Evaluator: () => true,
        // Desktop: the large-net consent step that follows Small loading is a no-op.
        isMemoryConstrainedDevice: () => false,
        arenaRunDownloads: new Set(),
        isStaleMount: (signal) => signal.aborted,
        el: (id) => (id === 'cancelSetupLoad' ? cancel : message),
        replaceResults: () => {
          results = [];
        },
      },
    );
    let results = ['old game PGN'];
    const pending = run();
    assert.equal(cancel.hidden, false);
    assert.deepEqual(results, ['old game PGN']);
    if (outcome === 'cancelled') matchAbort.abort();
    load.resolve(outcome === 'failure' ? null : {});
    await pending;
    assert.equal(cancel.hidden, true);
    assert.deepEqual(results, outcome === 'ready' ? [] : ['old game PGN']);
  });
}

test("Stop during retirement reaches Arena's stopped-state handler rather than leaving a live match", async () => {
  const text = source('arenaBrowser'),
    start = text.indexOf('    await untilAborted(awaitEngineRetirements(), matchAbort.signal);', text.indexOf('  let enginesReady = false;'));
  const guard = text.slice(start, text.indexOf('    if (isolatedCpuPool())', start));
  const catchStart = text.indexOf('  } catch (error) {', start),
    caught = text.slice(catchStart, text.indexOf('  } finally {', catchStart));
  const gate = Promise.withResolvers(),
    matchAbort = new AbortController(),
    nodes = new Map([
      ['message', { textContent: 'Stopping…' }],
      ['pairing', {}],
    ]);
  trackEngineRetirement(gate.promise);
  const run = compile(
    `return async function(){let arenaMatchState='live',enginesReady=false;const mode='match',score={games:0},schedule=[];try{${guard}throw new Error('must not start');${caught}}return arenaMatchState;};`,
    {
      awaitEngineRetirements,
      isStaleMount: () => false,
      mountSignal: matchAbort.signal,
      matchAbort,
      isAbortError: (e) => e.name === 'AbortError',
      el: (id) => nodes.get(id),
    },
  );
  const pending = run();
  matchAbort.abort();
  const stopped = await Promise.race([pending, new Promise((resolve) => setImmediate(() => resolve('still-waiting')))]);
  let admitted = false;
  const next = awaitEngineRetirements().then(() => {
    admitted = true;
  });
  await new Promise(setImmediate);
  const blocked = !admitted;
  gate.resolve();
  await next;
  await pending;
  assert.equal(stopped, 'stopped', 'Stop must settle before the old native release');
  assert.equal(blocked, true, 'stopping a consumer must not drop the old native retirement');
  assert.equal(nodes.get('message').textContent, 'Match stopped.');
});

for (const canceled of [false, true])
  test(`private Arena CPU startup awaits neural retirement and then checks cancellation (${canceled})`, async () => {
    const text = source('arenaBrowser');
    const start = text.indexOf('  let enginesReady = false;\n  try {');
    const entryStart = text.indexOf('    await untilAborted(awaitEngineRetirements(), matchAbort.signal);', start);
    const entry = text.slice(entryStart, text.indexOf('    } else {', entryStart));
    assert.ok(start >= 0 && entryStart > start && entry.includes('runIsolatedCpuTournament'));
    const gate = Promise.withResolvers(),
      controller = new AbortController();
    let starts = 0;
    const run = compile(`return async function(){let enginesReady=false;${entry}}};`, {
      awaitEngineRetirements: () => gate.promise,
      isStaleMount: () => false,
      mountSignal: controller.signal,
      matchAbort: controller,
      isolatedCpuPool: () => true,
      el: () => ({}),
      runIsolatedCpuTournament: async () => {
        starts++;
      },
      schedule: [],
      pendingIndices: [],
      byPid: new Map(),
      completeGame: () => {},
    });
    const pending = run();
    assert.equal(starts, 0);
    if (canceled) controller.abort();
    gate.resolve();
    if (canceled) await assert.rejects(pending, { name: 'AbortError' });
    else await pending;
    assert.equal(starts, canceled ? 0 : 1);
  });

// A first warmup evaluation waiting on the hybrid runtime's pack download.
const pendingInit = Promise.withResolvers();
const inner = { abortPendingInit: () => pendingInit.reject(new DOMException('Init cancelled', 'AbortError')) };

test('Arena Stop during a cold hybrid Small warmup cancels the pack init instead of waiting for it', async () => {
  const text = source('arenaBrowser');
  const warmup = compile(
    `const warmupPositions = [];
    ${extract(text, '  const lc0SearchWarmup = (engineId: string) => async (signal: AbortSignal) => {', '  const lc0BigNetWarmup =')}
    return lc0SearchWarmup('lc0');`,
    {
      lc0Cache: { inner },
      lc0SearcherFor: () => ({ search: () => pendingInit.promise, resetTree() {} }),
      renderCacheInfo: () => {},
    },
  );
  const controller = new AbortController();
  const running = warmup(controller.signal);
  controller.abort();
  const settled = await Promise.race([
    running.then(
      () => 'done',
      (error) => error.name,
    ),
    new Promise((resolve) => setTimeout(() => resolve('stalled'), 100)),
  ]);
  assert.equal(settled, 'AbortError', 'Arena would stay "Stopping…" until the pack request settled');
});
