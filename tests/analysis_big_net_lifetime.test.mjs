import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { createAnalysisEngines } from '../src/lc0/analysis/analysisEngines.ts';
import { BIG_NET_SESSION_IDLE_TTL_MS, BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';

// Execute the actual page disposal functions on the real engine host, not a duplicate policy helper.
const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
const unused = source.slice(source.indexOf('function disposeUnusedEngines()'), source.indexOf('function bestShapes()'));
const runtime = source.slice(source.indexOf('function disposeRuntimeResources()'), source.indexOf('function disposePageResources()'));
function controller(activeKeys = []) {
  const events = [],
    timers = [];
  const pool = new BigNetSessionPool(
    (config) => {
      const session = {
        key: config.key,
        cancels: 0,
        disposals: 0,
        cancel() {
          this.cancels++;
          events.push(`cancel:${this.key}`);
        },
        dispose() {
          this.disposals++;
          events.push(`dispose:${this.key}`);
        },
      };
      events.push(`create:${config.key}`);
      return session;
    },
    {
      setTimeout(fn, delay) {
        const timer = { fn, delay, cleared: false };
        timers.push(timer);
        return timer;
      },
      clearTimeout(timer) {
        timer.cleared = true;
      },
    },
  );
  const active = new Set(activeKeys);
  const noop = () => {};
  const engines = createAnalysisEngines({
    port: { live: () => true, loading: noop, runtimeInfoChanged: noop },
    dial: 'balanced',
    centipawn: { legacyOverride: false, modelUrl: '', metaUrl: '', legacyModelUrl: '', legacyMetaUrl: '', manifestUrl: '' },
    loadCentipawn: () => assert.fail('unused'),
    centipawnCache: { retain: noop, clear: () => events.push('clear:centipawn'), has: () => false, get: () => assert.fail('unused') },
    variants: {},
    activeRows: () => [],
    preload: async () => {},
    createStockfish: () => ({ dispose: () => events.push('dispose:sf') }),
    bigNets: { peek: (key) => pool.peek(key), release: (key, ttl) => pool.release(key, ttl), acquire: () => assert.fail('unused'), disposeNow: noop },
  });
  const engineRows = [...active].map((variant) => ({ family: 'lc0', variant }));
  const code = ts.transpile(
    `
    const noop=()=>{};
    const activeEngineRows=()=>[],reviewEngineRows=()=>[],renderRecklessRuntimeInfo=noop,isLc0BigNetVariant=()=>true;
    const trackEngineRetirement=(promise)=>promise;
    let analysisAbort={abort:noop},reviewAbort=null,activeWorkerSearchId=null,searchWorker=null;
    const cancelAllAnalysisAssetPrefetches=noop,resetSearchWorker=noop;
    let mainEvaluator={dispose:()=>events.push('dispose:main')},searcher={},lc0ServedModelUrl='test';
    ${unused}
    ${runtime}
    return {disposeUnusedEngines,disposeRuntimeResources};
  `,
    { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext },
  );
  return { ...new Function('engines', 'engineRows', 'events', code)(engines, engineRows, events), engines, pool, events, timers, active };
}

test('Analysis t3 -> BT4 switch destroys the previous worker before replacement acquisition, without an idle timer', () => {
  const api = controller(['bt4']);
  const previous = api.pool.acquire('t3');
  api.disposeUnusedEngines();
  assert.equal(previous.cancels, 1);
  assert.equal(previous.disposals, 1);
  assert.equal(api.pool.has('t3'), false);
  assert.equal(api.timers.length, 0, 'no three-minute retained worker');
  const next = api.pool.acquire('bt4');
  assert.equal(next.disposals, 0);
  assert.ok(api.events.indexOf('dispose:t3') < api.events.indexOf('create:bt4'));
  assert.notEqual(api.pool.acquire('t3'), previous, 'switching back creates a fresh worker; model-byte caching is independent');
  api.pool.disposeAllNow();
});

test('Analysis retains every explicitly active big net, including multiple active rows', () => {
  const api = controller(['t3', 'bt4']);
  const t3 = api.pool.acquire('t3'),
    bt4 = api.pool.acquire('bt4');
  api.disposeUnusedEngines();
  assert.equal(api.pool.peek('t3'), t3);
  assert.equal(api.pool.peek('bt4'), bt4);
  assert.equal(t3.disposals, 0);
  assert.equal(bt4.disposals, 0);
  assert.equal(api.timers.length, 0);
  api.pool.disposeAllNow();
});

test('Analysis releases each of its big nets without a row, and leaves other pooled nets to their owners', () => {
  const api = controller();
  const t3 = api.pool.acquire('t3'),
    bt4 = api.pool.acquire('bt4'),
    other = api.pool.acquire('lqo');
  api.disposeUnusedEngines();
  assert.equal(t3.disposals, 1);
  assert.equal(bt4.disposals, 1);
  assert.equal(other.cancels + other.disposals, 0);
  assert.equal(api.timers.length, 0);
  api.pool.disposeAllNow();
});

test('Analysis runtime/page teardown cancels and destroys all big nets synchronously', () => {
  const api = controller(['t3', 'bt4']);
  const nets = ['t3', 'bt4'].map((key) => api.pool.acquire(key));
  api.engines.getStockfish('lite');
  api.engines.pinAnalysisRunCentipawnKeys([{ family: 'centipawn', variant: 'bt4-auto' }]);
  api.events.length = 0;
  api.disposeRuntimeResources();
  assert.ok(nets.every((net) => net.cancels === 1 && net.disposals === 1));
  assert.ok(['t3', 'bt4'].every((key) => !api.pool.has(key)));
  assert.equal(api.timers.length, 0);
  // Big nets, then Centipawn, then CPU engines, then the Small evaluator.
  assert.deepEqual(api.events, ['cancel:bt4', 'dispose:bt4', 'cancel:t3', 'dispose:t3', 'clear:centipawn', 'dispose:sf', 'dispose:main']);
  assert.deepEqual(api.engines.analysisRunCentipawnKeys(), [], 'teardown drops the run pins');
});

test('the shared pool default remains three minutes for callers that explicitly choose retention', () => {
  const api = controller();
  const net = api.pool.acquire('lqo');
  api.pool.release('lqo');
  assert.equal(BIG_NET_SESSION_IDLE_TTL_MS, 180000);
  assert.equal(api.timers[0].delay, 180000);
  assert.equal(net.disposals, 0);
  api.pool.disposeAllNow();
});
