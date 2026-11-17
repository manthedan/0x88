import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
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
    await pending;
    assert.equal(starts, canceled ? 0 : 1);
  });
