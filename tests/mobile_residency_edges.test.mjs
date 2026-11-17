import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { awaitEngineRetirements, trackEngineRetirement } from '../src/lc0/engineRetirement.ts';

const source = (name) => readFileSync(new URL(`../src/lc0/${name}.ts`, import.meta.url), 'utf8');
function extract(text, first, last) {
  const from = text.indexOf(first);
  return text.slice(from, text.indexOf(last, from));
}
function compile(body, deps) {
  return new Function(...Object.keys(deps), `return (${stripTypeScriptTypes(`function run(){${body}\n}`)})();`)(...Object.values(deps));
}

for (const scenario of ['untouched phone', 'last Small row removed', 'active Small'])
  test(`Analysis runtime maintenance: ${scenario}`, async () => {
    let loads = 0,
      invalidations = 0;
    const reload = compile(`${extract(source('analysisBrowser'), 'async function reloadLc0Backend(', 'async function init(')} return reloadLc0Backend;`, {
      lineCache: new Map(),
      completeAnalysisKeys: new Set(),
      criticalPanel: null,
      disposeRuntimeResources: () => {
        invalidations++;
      },
      automaticAnalysisAllowed: () => scenario !== 'untouched phone',
      activeEngineRows: () => [{ family: 'lc0', variant: scenario === 'last Small row removed' ? 'bt4' : 'small' }],
      isLc0BigNetVariant: (variant) => ['bt4', 't3', 'lqo'].includes(variant),
      reconcileActiveRuntimeFallbackWarning: () => {},
      renderRecklessRuntimeInfo: () => {},
      mountAbort: new AbortController(),
      loadLc0Backend: async () => {
        loads++;
        return true;
      },
    });
    assert.equal(await reload('none'), true);
    assert.equal(invalidations, 1);
    assert.equal(loads, scenario === 'active Small' ? 1 : 0);
  });

test('the Maia reload control cannot race a constrained big-net run, but works once that owner settles', async () => {
  const text = source('analysisBrowser'),
    elements = new Map();
  let creates = 0;
  const api = compile(
    `
    let maia3PanelLoadGeneration=0,maia3PanelTask=null,maia3PanelTimer=null,maia3PanelPending=null;
    let maia3PanelEvaluator=null,maia3PanelLoading=false,maia3PanelAbort=null,explorerMaia=null;
    const controller=new AbortController();let maia3PanelBlockedByRun=controller;
    ${extract(text, 'function unloadMaia3Panel(', 'function scheduleMaia3Panel(')}
    unloadMaia3Panel('Unloaded');
    return {enable:enableMaia3Panel,settle(){
      ${extract(text, '    if (maia3PanelBlockedByRun === controller) {', '    const owns = analysisOperationId')}
    },unload:unloadMaia3Panel};
  `,
    {
      trackEngineRetirement,
      awaitEngineRetirements,
      mountAbort: new AbortController(),
      isStaleMount: () => false,
      el: (id) => {
        if (!elements.has(id))
          elements.set(id, {
            toggleAttribute(_key, value) {
              this.disabled = value;
            },
          });
        return elements.get(id);
      },
      maybeEl: (id) => elements.get(id),
      maia3PanelStatus: () => {},
      showModelProgress: () => {},
      hideModelProgress: () => {},
      renderExplorerComparison: () => {},
      scheduleMaia3Panel: () => {},
      LatestOnlyTask: class {
        dispose() {}
      },
      Maia3BrowserEvaluator: {
        create: async () => {
          creates++;
          return { dispose: async () => {} };
        },
      },
    },
  );
  assert.equal(elements.get('maia3Enable').disabled, true);
  await api.enable();
  assert.equal(creates, 0, 'programmatic calls must obey the gate too');
  api.settle();
  assert.equal(elements.get('maia3Enable').disabled, false);
  await api.enable();
  assert.equal(creates, 1);
  api.unload('Done');
  await awaitEngineRetirements();
});

test('Arena reconciles Small and big-net residency from the active seats before rebuilding engines', async () => {
  const text = source('arenaBrowser'),
    events = [],
    release = Promise.withResolvers();
  const pool = new BigNetSessionPool((config) => ({
    cancel() {
      events.push(`cancel:${config.key}`);
    },
    dispose() {
      events.push(`dispose:${config.key}`);
    },
  }));
  pool.acquire('t3');
  pool.acquire('bt4');
  let rows = [
    { family: 'lc0', variant: 'small' },
    { family: 'lc0', variant: 't3' },
  ];
  const api = compile(
    `
    let running=false,lc0Cache={dispose:()=>{events.push('small');return release;}},loadingLc0=false,lc0LoadAbort=null;
    let arenaLc0ServedModelUrl='',activeEngineIds=[];
    const lc0Searchers=new Map(),lastLc0SearchResults=new Map(),pendingLc0ReplyProbes=new Map(),lc0TreeTelemetry=new Map();
    const uciTelemetry=new Map(),bt4Telemetry=new Map(),engineOutputs=new Map(),engineOutputHistory=[],thinkingEngineIds=new Set();
    let engineOutputTotalCount=0;
    ${extract(text, 'function disposeLc0Resources():', 'function disposeRuntimeResources():')}
    ${extract(text, 'function reconcileArenaNeuralSessions():', 'function buildEngines()')}
    return reconcileArenaNeuralSessions;
  `,
    {
      events,
      release: release.promise,
      trackEngineRetirement,
      activeSeatRows: () => rows,
      ARENA_BIG_NET_KEYS: ['t3', 'bt4'],
      isLc0BigNetVariant: (variant) => ['bt4', 't3', 'lqo'].includes(variant),
      peekBigNetSearcher: (key) => pool.peek(key),
      releaseBigNetSearcher: (key, ttl) => pool.release(key, ttl),
    },
  );
  api();
  assert.deepEqual(events, ['cancel:bt4', 'dispose:bt4']);
  rows = [{ family: 'lc0', variant: 't3' }];
  api();
  assert.equal(events.at(-1), 'small');
  let admitted = false;
  const waiting = awaitEngineRetirements().then(() => {
    admitted = true;
  });
  await new Promise(setImmediate);
  assert.equal(admitted, false);
  release.resolve();
  await waiting;
  rows = [{ family: 'sf', variant: 'lite' }];
  api();
  assert.equal(pool.has('t3'), false);
  assert.ok(events.includes('dispose:t3'));
  assert.match(
    extract(text, 'function buildEngines()', 'async function warmUpSelectedEngines('),
    /function buildEngines\(\) \{\s*reconcileArenaNeuralSessions\(\)/,
  );
  for (const first of ["el('arenaSeatList').addEventListener('click'", "el('tournamentModeSelect').addEventListener('change'"]) {
    const handler = extract(text, first, '\n  });');
    assert.match(handler, /buildEngines\(\)/);
  }
});
