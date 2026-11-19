import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { BIG_NET_SESSION_IDLE_TTL_MS, BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';

// Execute the actual page disposal functions, not a duplicate policy helper.
const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
const unused = source.slice(source.indexOf('function disposeAnalysisCpuEngines()'), source.indexOf('function bestShapes()'));
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
  const code = ts.transpile(
    `
    const ANALYSIS_BIG_NET_KEYS=['lqo','t3','bt4'];
    const usesBigNetRow=key=>active.has(key),peekBigNetSearcher=key=>pool.peek(key),releaseBigNetSearcher=(key,ttl)=>pool.release(key,ttl);
    const noop=()=>{},cpu={retain:noop,disposeAll:noop};
    const recklessEngines=cpu,viridithasEngines=cpu,berserkEngines=cpu,plentyChessEngines=cpu,stormphraxEngines=cpu;
    const activeEngineRows=()=>[],reviewEngineRows=()=>[],disposeUnusedCentipawnEvaluators=noop,disposeStockfish=noop,renderRecklessRuntimeInfo=noop;
    const trackEngineRetirement=noop;
    let analysisAbort={abort:noop},analysisRunCentipawnKeys=new Set(),reviewAbort=null,activeWorkerSearchId=null,searchWorker=null;
    const cancelAllAnalysisAssetPrefetches=noop,preparedCpuEngineKeys=new Set(),resetSearchWorker=noop,disposeCentipawnEvaluators=noop;
    let mainEvaluator={dispose:noop},searcher={},lc0ServedModelUrl='test';
    ${unused}
    ${runtime}
    return {disposeUnusedEngines,disposeRuntimeResources};
  `,
    { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext },
  );
  return { ...new Function('pool', 'active', code)(pool, active), pool, events, timers, active };
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
    bt4 = api.pool.acquire('bt4'),
    unused = api.pool.acquire('lqo');
  api.disposeUnusedEngines();
  assert.equal(api.pool.peek('t3'), t3);
  assert.equal(api.pool.peek('bt4'), bt4);
  assert.equal(t3.disposals, 0);
  assert.equal(bt4.disposals, 0);
  assert.equal(unused.disposals, 1);
  assert.equal(api.timers.length, 0);
  api.pool.disposeAllNow();
});

test('Analysis runtime/page teardown cancels and destroys all big nets synchronously', () => {
  const api = controller(['t3', 'bt4']);
  const nets = ['lqo', 't3', 'bt4'].map((key) => api.pool.acquire(key));
  api.disposeRuntimeResources();
  assert.ok(nets.every((net) => net.cancels === 1 && net.disposals === 1));
  assert.ok(['lqo', 't3', 'bt4'].every((key) => !api.pool.has(key)));
  assert.equal(api.timers.length, 0);
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
