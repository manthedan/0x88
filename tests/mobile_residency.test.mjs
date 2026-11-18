import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { deviceMemoryClass, shouldAutoAnalyzePosition } from '../src/lc0/deviceMemoryClass.ts';
import { awaitEngineRetirements, trackEngineRetirement } from '../src/lc0/engineRetirement.ts';
import { untilAborted } from '../src/lc0/modelCache.ts';

const source = (name) => readFileSync(new URL(`../src/lc0/${name}.ts`, import.meta.url), 'utf8');
function extract(text, start, end) {
  return text.slice(text.indexOf(start), text.indexOf(end, text.indexOf(start)));
}
function compile(code, deps = {}) {
  deps = { untilAborted, ...deps };
  return new Function(...Object.keys(deps), `return (${stripTypeScriptTypes(`function run(){${code}\n}`)})();`)(...Object.values(deps));
}
const play = source('playBrowser');
function playOwner(options = {}) {
  const loads = [],
    releases = [],
    gates = [];
  const ctx = { enginePreparationGeneration: 0, opponent: { active: { maiaElo: 1500 } } };
  const api = compile(
    `
    let lc0Searcher=null,lc0LoadPromise=null,lc0SmallOwner={evaluator:null,abort:new AbortController()};
    let maia3Promise=null,maia3Status=null,maia3Generation=0,maia3Owner=null;
    let centipawnLoadPromise=null,centipawnEvaluator=null,centipawnEvaluatorGeneration=0,centipawnLoadAbort=null;
    ${extract(play, 'function retireUnusedPlayNeuralEngines(', 'function ctxEnsureCentipawn(')}
    ${extract(play, 'async function ctxEnsureMaia3(', 'function ctxCpuEngineFor(')}
    return {retire:retireUnusedPlayNeuralEngines,small:()=>ctxEnsureLc0Small(ctx),maia:()=>ctxEnsureMaia3(ctx),peek:()=>lc0Searcher};
  `,
    {
      ctx,
      trackEngineRetirement,
      awaitEngineRetirements,
      releaseUnusedBigNetSearchers: () => {},
      ctxAsyncUiIsCurrent: () => false,
      MODEL_URL: '/small.onnx',
      LC0_DEVICE_MODEL_SELECTION: true,
      createLc0OnnxEvaluatorFromUrl: async (_url, loadOptions) => {
        loads.push('small');
        if (options.createSmall) return options.createSmall(loadOptions);
        return {
          evaluator: {
            dispose: () => {
              releases.push('small');
              return gates.shift()?.promise;
            },
          },
        };
      },
      CachedLc0Evaluator: class {
        constructor(inner) {
          this.inner = inner;
        }
        dispose() {
          return this.inner.dispose();
        }
      },
      Lc0PuctSearcher: class {
        resetTree() {}
      },
      Maia3BrowserEvaluator: {
        create: async (createOptions) => {
          loads.push('maia');
          if (options.createMaia) return options.createMaia(createOptions);
          return {
            modelLoad: { bytes: 1 },
            dispose: () => {
              releases.push('maia');
              return gates.shift()?.promise ?? Promise.resolve();
            },
          };
        },
      },
    },
  );
  return { ...api, loads, releases, gates };
}

for (const old of ['maia', 'small']) {
  test(`Play ${old} retires before a replacement, reloads lazily, and keeps the selected owner`, async () => {
    const api = playOwner();
    const first = await api[old]();
    await api.retire(old === 'maia' ? { family: 'maia3' } : { family: 'lc0', variant: 'small' });
    assert.equal(await api[old](), first);
    assert.equal(api.releases.length, 0);
    const gate = Promise.withResolvers();
    api.gates.push(gate);
    let settled = false;
    const retiring = api.retire({ family: 'lc0', variant: 'bt4' }).then(() => {
      settled = true;
    });
    await new Promise(setImmediate);
    assert.deepEqual(api.releases, [old]);
    assert.equal(settled, false, 'wait for graceful/native release before loading BT4');
    gate.resolve();
    await retiring;
    assert.notEqual(await api[old](), first);
    assert.deepEqual(api.loads, [old, old]);
    await api.retire();
    assert.deepEqual(api.releases, [old, old]);
  });
}

test('switching away from a stalled Small download aborts it rather than blocking the replacement', async () => {
  let aborted = false;
  const api = playOwner({
    createSmall: ({ load }) =>
      new Promise((_resolve, reject) => {
        load.signal.addEventListener(
          'abort',
          () => {
            aborted = true;
            reject(load.signal.reason);
          },
          { once: true },
        );
      }),
  });
  const loading = assert.rejects(api.small(), { name: 'AbortError' });
  await api.retire({ family: 'sf' });
  await loading;
  assert.equal(aborted, true);
});

test('switching away from a stalled Maia initialization aborts promptly but still awaits its published worker cleanup', async () => {
  const cleanup = Promise.withResolvers();
  let aborted = false;
  const api = playOwner({
    createMaia: ({ signal, onInitCleanup }) =>
      new Promise((_resolve, reject) => {
        signal.addEventListener(
          'abort',
          () => {
            aborted = true;
            onInitCleanup(cleanup.promise);
            reject(signal.reason);
          },
          { once: true },
        );
      }),
  });
  const loading = assert.rejects(api.maia(), { name: 'AbortError' });
  let settled = false;
  const retiring = api.retire({ family: 'sf' }).then(() => {
    settled = true;
  });
  await loading;
  await new Promise(setImmediate);
  assert.equal(aborted, true);
  assert.equal(settled, false, 'the aborted init still owns a retiring worker');
  cleanup.resolve();
  await retiring;
});

test('Play unmount calls the same awaitable retirement for both neural owners', async () => {
  const api = playOwner();
  await api.small();
  await api.maia();
  const cleanup = extract(play, '    ctx.autosave?.flush();', '    if (ctx.resignArmTimer)');
  compile(cleanup, {
    ctx: { disposed: false, abort: new AbortController(), consentAbort: new AbortController() },
    ctxCancelEngineTurn: () => {},
    retireUnusedPlayNeuralEngines: api.retire,
    centipawnEvaluatorGeneration: 0,
    centipawnEvaluator: null,
    centipawnLoadPromise: null,
    playRuntimeFallbackAudits: new Map(),
    disposeCachedCpuEngine: () => {},
  });
  await awaitEngineRetirements();
  assert.deepEqual(api.releases.sort(), ['maia', 'small']);
});

test('a Small initialization finishing after retirement cannot repopulate the page owner', async () => {
  const gate = Promise.withResolvers(),
    released = [];
  const api = compile(
    `
    let lc0Searcher=null,lc0LoadPromise=null,lc0SmallOwner={evaluator:null,abort:new AbortController()};
    let maia3Promise=null,maia3Status=null,maia3Generation=0,maia3Owner=null;
    let centipawnLoadPromise=null,centipawnEvaluator=null,centipawnEvaluatorGeneration=0,centipawnLoadAbort=null;
    ${extract(play, 'function retireUnusedPlayNeuralEngines(', 'function ctxEnsureCentipawn(')}
    return {load:()=>ctxEnsureLc0Small(ctx),retire:retireUnusedPlayNeuralEngines,peek:()=>lc0Searcher};
  `,
    {
      ctx: {},
      trackEngineRetirement,
      awaitEngineRetirements,
      releaseUnusedBigNetSearchers: () => {},
      ctxAsyncUiIsCurrent: () => false,
      MODEL_URL: 'small',
      LC0_DEVICE_MODEL_SELECTION: true,
      createLc0OnnxEvaluatorFromUrl: () => gate.promise,
      CachedLc0Evaluator: class {
        constructor(inner) {
          this.inner = inner;
        }
        dispose() {
          released.push(1);
          return this.inner.dispose();
        }
      },
      Lc0PuctSearcher: class {
        resetTree() {}
      },
    },
  );
  const loading = api.load(),
    retiring = api.retire();
  gate.resolve({ evaluator: { dispose: async () => {} } });
  await loading;
  await retiring;
  assert.equal(api.peek(), null);
  assert.deepEqual(released, [1]);
});

test('Analysis retires Small only after its last row is removed (eval bar is cached-row-only; reviewers are CPU-only)', async () => {
  const text = source('analysisBrowser'),
    events = [],
    active = [];
  const run = () =>
    compile(
      `
    let mainEvaluator={dispose:async()=>events.push('main')},searcher={};
    ${extract(text, 'function disposeUnusedEngines(', '// Default: one best-move arrow')}
    disposeUnusedEngines();
  `,
      {
        events,
        trackEngineRetirement,
        ANALYSIS_BIG_NET_KEYS: [],
        disposeUnusedCentipawnEvaluators: () => {},
        activeEngineRows: () => active,
        isLc0BigNetVariant: (v) => ['bt4', 't3', 'lqo'].includes(v),
        resetSearchWorker: () => events.push('worker'),
        reviewAbort: null,
        disposeStockfish: () => {},
        recklessEngines: { retain() {} },
        viridithasEngines: { retain() {} },
        berserkEngines: { retain() {} },
        plentyChessEngines: { retain() {} },
        stormphraxEngines: { retain() {} },
        renderRecklessRuntimeInfo: () => {},
      },
    );
  active.push({ family: 'lc0', variant: 'small' });
  run();
  assert.deepEqual(events, []);
  active.pop();
  run();
  await awaitEngineRetirements();
  assert.deepEqual(events, ['worker', 'main']);
});

test('Analysis unloading Maia clears the panel and awaits even an in-flight initialization', async () => {
  const pending = Promise.withResolvers(),
    release = Promise.withResolvers(),
    elements = new Map();
  let disposals = 0,
    progressHidden = false;
  const api = compile(
    `
    let maia3PanelLoadGeneration=1,maia3PanelTask={dispose(){}},maia3PanelTimer=null;
    let maia3PanelPending=pending,maia3PanelEvaluator=null,maia3PanelLoading=true,explorerMaia={},maia3PanelAbort=null,maia3PanelBlockedByRun=null;
    ${extract(source('analysisBrowser'), 'function unloadMaia3Panel(', 'async function enableMaia3Panel(')}
    unloadMaia3Panel('Unloaded');
    return ()=>({maia3PanelEvaluator,maia3PanelLoading,maia3PanelLoadGeneration,explorerMaia});
  `,
    {
      pending: pending.promise,
      trackEngineRetirement,
      renderExplorerComparison: () => {},
      hideModelProgress: () => {
        progressHidden = true;
      },
      maia3PanelStatus: () => {},
      el: (id) => {
        if (!elements.has(id)) elements.set(id, {});
        return elements.get(id);
      },
    },
  );
  let done = false;
  const waiting = awaitEngineRetirements().then(() => {
    done = true;
  });
  pending.resolve({
    dispose: () => {
      disposals++;
      return release.promise;
    },
  });
  await new Promise(setImmediate);
  assert.equal(done, false);
  assert.equal(disposals, 1);
  release.resolve();
  await waiting;
  assert.equal(progressHidden, true, 'canceled Maia download progress is removed by its retiring owner');
  assert.equal(api().maia3PanelLoading, false);
  assert.equal(api().explorerMaia, null);
  assert.equal(elements.get('maia3Enable').hidden, false);
  assert.equal(elements.get('maia3Enable').disabled, false);
  assert.equal(elements.get('maia3Grid').hidden, true);
  const perform = extract(source('analysisBrowser'), 'async function performAnalysis(owner:', 'let analysisGameTree = tree;');
  assert.match(perform, /isMemoryConstrainedDevice\(\).*rows.some/s);
  assert.ok(perform.indexOf('unloadMaia3Panel(') < perform.indexOf('initBigNetsForRun(uninitializedBigNets'));
});

test('Analysis cancels a pending Maia panel load, awaits terminal cleanup, and re-enable reloads it', async () => {
  const text = source('analysisBrowser'),
    cleanup = Promise.withResolvers(),
    elements = new Map();
  let creates = 0;
  const api = compile(
    `
    let maia3PanelLoadGeneration=0,maia3PanelTask=null,maia3PanelTimer=null,maia3PanelPending=null;
    let maia3PanelEvaluator=null,maia3PanelLoading=false,maia3PanelAbort=null,explorerMaia=null,maia3PanelBlockedByRun=null;
    ${extract(text, 'function unloadMaia3Panel(', 'function scheduleMaia3Panel(')}
    return {enable:enableMaia3Panel,unload:unloadMaia3Panel};
  `,
    {
      trackEngineRetirement,
      awaitEngineRetirements,
      mountAbort: new AbortController(),
      isStaleMount: () => false,
      isMemoryConstrainedDevice: () => false,
      el: (id) => {
        if (!elements.has(id)) elements.set(id, {});
        return elements.get(id);
      },
      maia3PanelStatus: () => {},
      showModelProgress: () => {},
      hideModelProgress: () => {},
      renderExplorerComparison: () => {},
      scheduleMaia3Panel: () => {},
      LatestOnlyTask: class {
        dispose() {}
      },
      Maia3BrowserEvaluator: {
        create: ({ signal, onInitCleanup }) => {
          creates++;
          if (creates > 1) return Promise.resolve({ dispose: async () => {} });
          return new Promise((_resolve, reject) =>
            signal.addEventListener(
              'abort',
              () => {
                onInitCleanup(cleanup.promise);
                reject(signal.reason);
              },
              { once: true },
            ),
          );
        },
      },
    },
  );
  const enabling = api.enable();
  await new Promise(setImmediate);
  api.unload('Unloaded');
  await enabling;
  let retired = false;
  const waiting = awaitEngineRetirements().then(() => {
    retired = true;
  });
  await new Promise(setImmediate);
  assert.equal(retired, false);
  cleanup.resolve();
  await waiting;
  await api.enable();
  assert.equal(creates, 2);
  assert.equal(elements.get('maia3Enable').hidden, true);
  api.unload('Done');
  await awaitEngineRetirements();
});

test('Arena immediately detaches big nets and waits for ack before the next warmup', async () => {
  const gate = Promise.withResolvers(),
    events = [];
  const pool = new BigNetSessionPool(() => ({
    dispose: () => {
      events.push('dispose');
      return gate.promise;
    },
  }));
  pool.acquire('t3');
  pool.release('t3', 0);
  assert.equal(pool.has('t3'), false);
  const text = source('arenaBrowser');
  assert.equal([...text.matchAll(/releaseBigNetSearcher\(key, 0\)/g)].length, 2);
  const warm = extract(text, 'async function warmUpSelectedEngines(', '\nfunction ');
  // Extract the exact function, excluding subsequent helpers of either async/sync kind.
  const fn = warm.slice(0, warm.indexOf('\n}\n') + 3);
  const run = compile(`${fn};return warmUpSelectedEngines;`, {
    awaitEngineRetirements,
    engines: new Map([['next', { warmup: async () => events.push('warmup') }]]),
    el: () => ({}),
    showEngineWarmupStatus: () => {},
    renderCacheInfo: () => {},
    hideEngineWarmupStatus: () => {},
    sleep: async () => {},
  });
  const running = run(['next'], new AbortController().signal, 0);
  await new Promise(setImmediate);
  assert.deepEqual(events, ['dispose']);
  gate.resolve();
  await running;
  assert.deepEqual(events, ['dispose', 'warmup']);
});

test('Arena mount never eagerly loads Small; Start and fixed-suite autoruns still own lazy loads and badge remains', () => {
  const text = source('arenaBrowser'),
    init = extract(text, 'async function init(mountSignal:', 'function resetArenaPageState(');
  assert.doesNotMatch(init, /await loadLc0Evaluator/);
  assert.match(init, /renderRuntimeBadge\(\)/);
  const lazyLoad = extract(text, 'async function loadLc0Evaluator(', 'async function reloadLc0Evaluator(');
  assert.match(lazyLoad, /lc0Cache = new CachedLc0Evaluator[\s\S]*renderRuntimeBadge\(\)/, 'badge refresh follows lazy native session creation');
  assert.match(init, /runFixedSuiteBenchAutorun\(\)/);
  assert.match(init, /runArenaBenchAutorun\(\)/);
  assert.match(extract(text, 'async function startMatchUnchecked(', 'async function startMatch('), /await loadLc0Evaluator\(mountSignal, matchAbort.signal\)/);
  assert.match(extract(text, 'async function runFixedSuiteBenchAutorun(', 'async function runArenaBenchAutorun('), /if \(!lc0Cache\) await loadLc0Evaluator/);
});

test('Arena lazy mount enables Start and bench autoruns observe ready controls without loading a model', () => {
  const text = source('arenaBrowser');
  const tail = extract(text, '  // startMatch / the fixed-suite autorun', '\nlet lastArenaMountHref');
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
    ['message', {}],
    ['arenaMemoryNote', {}],
  ]);
  let autoruns = 0;
  const autorun = () => {
    assert.equal(elements.get('start').disabled, false);
    autoruns++;
  };
  compile(
    `${extract(text, 'let startDisabled = true;', 'function runtimeAuditIdentityForRow(')}\n${tail.slice(0, tail.lastIndexOf('\n}'))}\nrefreshArenaMemoryGuard();`,
    {
      checkpoint: null,
      checkpointUnreadable: false,
      el: (id) => elements.get(id),
      arenaTournamentMode: () => 'match',
      arenaBigNetPairingBlocked: () => false,
      arenaBigNetSeatBlocked: () => false,
      seatRows: [],
      ARENA_BIG_NET_MEMORY_REASON: 'Blocked',
      readyMessage: () => 'Ready.',
      refreshSeatControls: () => {},
      renderRuntimeBadge: () => {},
      runFixedSuiteBenchAutorun: autorun,
      runArenaBenchAutorun: autorun,
    },
  );
  assert.equal(autoruns, 2);
  assert.equal(elements.get('start').disabled, false, 'availability refresh must not undo readiness');
  assert.equal(elements.get('message').textContent, 'Ready.');
});

test('phone auto-analysis gate leaves Analyze enabled and an explicit tap admits a real scheduler request', async () => {
  const text = source('analysisBrowser');
  const elements = new Map();
  const el = (id) => {
    if (!elements.has(id))
      elements.set(id, {
        disabled: true,
        toggleAttribute(_name, disabled) {
          this.disabled = disabled;
        },
      });
    return elements.get(id);
  };
  const gate = extract(text, '  if (!automaticAnalysisAllowed()) {', '\nlet lastAnalysisMountHref');
  await compile(`return async function(){${gate.slice(0, gate.lastIndexOf('\n}'))}\n};`, {
    automaticAnalysisAllowed: () => false,
    el,
    selectEl: el,
  })();
  assert.equal(el('analyze').disabled, false);
  assert.equal(el('lc0RuntimeSelect').disabled, false);
  const requests = [];
  const analyze = compile(`${extract(text, 'let analysisHasExplicitInput = false;', 'async function performAnalysis(owner:')} return analyzeCurrent;`, {
    shouldAutoAnalyzePosition,
    deviceMemoryClass: () => 'phone',
    tree: { root: { fen: 'start' } },
    START_FEN: 'start',
    mainlineNodes: () => [{}],
    mountAbort: new AbortController(),
    analysisRunState: { modes: [] },
    deferHiddenTabWork: () => {},
    dispatchAnalysis: async (event) => requests.push(event),
    analysisIntent: (force) => ({ force }),
  });
  await analyze();
  assert.equal(requests.length, 0);
  await analyze({ force: true });
  assert.equal(requests.length, 1);
  assert.equal(requests[0].intent.force, true);
});

test('phone auto-analysis gate: default untouched only; explicit input, moves, nondefault FEN and desktops start', () => {
  assert.equal(deviceMemoryClass('?memoryClass=phone', { userAgent: 'desktop' }), 'phone');
  assert.equal(deviceMemoryClass('?memoryClass=desktop', { userAgent: 'iPhone' }), 'desktop');
  assert.equal(deviceMemoryClass('', { userAgent: 'Mozilla iPhone' }), 'phone');
  assert.equal(shouldAutoAnalyzePosition('phone', false, 'start', 'start', 0), false);
  for (const args of [
    ['phone', true, 'start', 'start', 0],
    ['phone', false, 'other', 'start', 0],
    ['phone', false, 'start', 'start', 1],
    ['desktop', false, 'start', 'start', 0],
  ])
    assert.equal(shouldAutoAnalyzePosition(...args), true);
  const text = source('analysisBrowser');
  assert.match(text, /if \(options.force\) analysisHasExplicitInput = true/);
  assert.match(text, /if \(!automaticAnalysisAllowed\(\)\) \{/);
});
