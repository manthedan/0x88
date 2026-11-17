import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { analysisCanPrepare } from '../src/lc0/analysisRunState.ts';
import { loadRetirableResource } from '../src/lc0/arenaResourceLoad.ts';
import { awaitEngineRetirements, trackEngineRetirement } from '../src/lc0/engineRetirement.ts';

const source = (name) => readFileSync(new URL(`../src/lc0/${name}.ts`, import.meta.url), 'utf8');
function extract(text, first, last) {
  const start = text.indexOf(first);
  return text.slice(start, text.indexOf(last, start));
}
function compile(body, deps) {
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
    let analysisAbort=null,analysisRunCentipawnKeys=new Set(),reviewAbort=null,activeWorkerSearchId=null;
    let mainEvaluator={dispose:()=>release},searcher={},lc0ServedModelUrl='small';
    const preparedCpuEngineKeys=new Set(),cpu={disposeAll(){}};
    const recklessEngines=cpu,viridithasEngines=cpu,berserkEngines=cpu,plentyChessEngines=cpu,stormphraxEngines=cpu;
    ${extract(source('analysisBrowser'), 'function disposeRuntimeResources():', 'function disposePageResources():')}
    return disposeRuntimeResources;
  `,
    {
      release: release.promise,
      trackEngineRetirement,
      ANALYSIS_BIG_NET_KEYS: [],
      cancelAllAnalysisAssetPrefetches: () => {},
      resetSearchWorker: () => {},
      disposeStockfish: () => {},
      disposeCentipawnEvaluators: () => {},
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
    /loadRetirableResource\(createFallback, fallbackSignal\)/,
    'canceled fallback creation uses the same tracked boundary as Arena',
  );
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

for (const pendingNative of [false, true])
  test(`Play retires Centipawn's pending/loaded owner (${pendingNative})`, async () => {
    const native = Promise.withResolvers(),
      release = Promise.withResolvers();
    let disposals = 0,
      signal;
    const evaluator = {
      dispose: () => {
        disposals++;
        return release.promise;
      },
    };
    class Cache {
      constructor(inner) {
        this.inner = inner;
      }
      clear() {}
    }
    const text = source('playBrowser');
    const api = compile(
      `
    let centipawnEvaluator=null,centipawnLoadPromise=null,centipawnEvaluatorGeneration=0,centipawnLoadAbort=null;
    let maia3Promise=null,lc0LoadPromise=null;
    ${extract(text, 'async function disposePlayCentipawn(', 'function ctxEnsureLc0Small(')}
    ${extract(text, 'function ctxEnsureCentipawn(', 'async function ctxEnsureMaia3(')}
    return {load:()=>ctxEnsureCentipawn({}),retire:()=>retireUnusedPlayNeuralEngines({family:'sf'}),peek:()=>centipawnEvaluator};
  `,
      {
        CachedEvaluator: Cache,
        loadRetirableResource,
        trackEngineRetirement,
        awaitEngineRetirements,
        releaseUnusedBigNetSearchers: () => {},
        ctxAsyncUiIsCurrent: () => false,
        ctxIsDisposed: () => false,
        CENTIPAWN_MODEL_URL: '/models/s4fix_v_ema.onnx',
        CENTIPAWN_META_URL: '/models/s4fix_v_ema.meta.json',
        createBrowserSquareformerRuntimeEvaluator: async (_spec, options) => {
          signal = options.signal;
          return native.promise;
        },
      },
    );
    const loading = api.load();
    const canceled = pendingNative ? assert.rejects(loading, { name: 'AbortError' }) : null;
    await new Promise(setImmediate); // The native creation has begun before its owner is canceled.
    if (!pendingNative) {
      native.resolve({ evaluator, resolvedRuntime: 'ort' });
      await loading;
    }
    let retired = false;
    const retirement = api.retire().then(() => {
      retired = true;
    });
    assert.equal(api.peek(), null);
    assert.equal(signal.aborted, true);
    if (pendingNative) {
      await canceled;
      native.resolve({ evaluator, resolvedRuntime: 'ort' });
    }
    await new Promise(setImmediate);
    assert.equal(disposals, 1);
    assert.equal(retired, false);
    release.resolve();
    await retirement;
    assert.equal(retired, true);
  });

test("Stop during retirement reaches Arena's stopped-state handler rather than leaving a live match", async () => {
  const text = source('arenaBrowser'),
    start = text.indexOf('    await awaitEngineRetirements();', text.indexOf('  let enginesReady = false;'));
  const guard = text.slice(start, text.indexOf('    if (isolatedCpuPool())', start));
  const catchStart = text.indexOf('  } catch (error) {', start),
    caught = text.slice(catchStart, text.indexOf('  } finally {', catchStart));
  const gate = Promise.withResolvers(),
    matchAbort = new AbortController(),
    nodes = new Map([
      ['message', { textContent: 'Stopping…' }],
      ['pairing', {}],
    ]);
  const run = compile(
    `return async function(){let arenaMatchState='live',enginesReady=false;const mode='match',score={games:0},schedule=[];try{${guard}throw new Error('must not start');${caught}}return arenaMatchState;};`,
    {
      awaitEngineRetirements: () => gate.promise,
      isStaleMount: () => false,
      mountSignal: matchAbort.signal,
      matchAbort,
      isAbortError: (e) => e.name === 'AbortError',
      el: (id) => nodes.get(id),
    },
  );
  const pending = run();
  matchAbort.abort();
  gate.resolve();
  assert.equal(await pending, 'stopped');
  assert.equal(nodes.get('message').textContent, 'Match stopped.');
});

for (const canceled of [false, true])
  test(`private Arena CPU startup awaits neural retirement and then checks cancellation (${canceled})`, async () => {
    const text = source('arenaBrowser');
    const start = text.indexOf('  let enginesReady = false;\n  try {');
    const entryStart = text.indexOf('    await awaitEngineRetirements();', start);
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
