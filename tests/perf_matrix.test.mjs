import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { digestReport, phaseOf, soakUrl, summarize, WORKER_READY_SCRIPT } from '../scripts/perf_matrix/common.mjs';

const runner = readFileSync(new URL('../scripts/perf_matrix/run_matrix.mjs', import.meta.url), 'utf8');
const validity = new Function(`${runner.slice(runner.indexOf('function validity(r)'), runner.indexOf('function aggregate(dir)'))}; return validity;`)();

test('matrix keeps load/search and teardown boundaries distinct', () => {
  const progress = (phases) => ({ state: 'running', steps: [{ phases }] });
  assert.equal(phaseOf(null, false), 'baseline');
  assert.equal(phaseOf(progress([]), true), 'load');
  assert.equal(phaseOf(progress(['resident-after-load']), true), 'search');
  assert.equal(phaseOf(progress(['after-search']), true), 'dispose');
  assert.equal(phaseOf({ state: 'done' }, true), 'settle');
});

test('matrix rejects reused/dirty processes, non-warm wire traffic and failed loads', () => {
  const valid = {
    browser: 'safari',
    cache: 'warm-fresh',
    report: { status: 'ok' },
    pidReused: false,
    previousPageProcessExited: true,
    baselineMb: 60,
    lifeMaxAtBaselineMb: 75,
    cacheVerified: true,
  };
  assert.deepEqual(validity(valid), []);
  for (const change of [
    { pidReused: true },
    { previousPageProcessExited: false },
    { lifeMaxAtBaselineMb: 200 },
    { cacheVerified: false },
    { crashed: true },
    { timedOut: true },
    { pageReadyMs: 61000 },
    { report: { status: 'failed' } },
  ])
    assert.ok(validity({ ...valid, ...change }).length);
});

test('matrix stats retain the whole spread and ignore unavailable values', () => {
  assert.deepEqual(summarize([10, 20, 30, NaN, undefined]), { n: 3, median: 20, min: 10, max: 30, range: 20, mad: 10 });
  assert.equal(summarize([]).n, 0);
  assert.equal(new URL(soakUrl('http://127.0.0.1:5000/', 'maia3', { ortDequantFold: '1' })).searchParams.get('ortDequantFold'), '1');
});

test('worker observation records effective IO/fold settings without changing the init payload', () => {
  const window = {};
  class NativeWorker {
    addEventListener(_type, fn) {
      this.receive = fn;
    }
    postMessage(message) {
      this.posted = message;
    }
  }
  new Function('window', 'Worker', WORKER_READY_SCRIPT)(window, NativeWorker);
  const worker = new window.Worker();
  const init = { type: 'init', dequantFold: false };
  worker.postMessage(init);
  worker.receive({ data: { type: 'ready', ortIoMode: 'gpu-capture', ortRuntime: { dequantFoldEnabled: false }, model: new ArrayBuffer(100) } });
  assert.equal(worker.posted, init);
  assert.equal(window.__perfWorkerReady[0].ortIoMode, 'gpu-capture');
  assert.equal(window.__perfWorkerReady[0].ortRuntime.dequantFoldEnabled, false);
  assert.equal(Object.hasOwn(window.__perfWorkerReady[0], 'model'), false);
});

test('external owned-Chrome disconnect rejects pending CDP calls instead of stranding the benchmark lock', async () => {
  const text = readFileSync(new URL('../scripts/perf_matrix/chrome_session.mjs', import.meta.url), 'utf8');
  const Cdp = new Function('WebSocket', `${text.slice(text.indexOf('class Cdp'), text.indexOf('const proxy ='))}; return Cdp;`)(
    class {
      send() {}
    },
  );
  const cdp = new Cdp('ws://owned-test');
  const done = cdp.send('completed');
  cdp.ws.onmessage({ data: JSON.stringify({ id: 1, result: { ok: true } }) });
  assert.deepEqual(await done, { ok: true });
  const pending = cdp.send('pending');
  cdp.ws.onclose();
  await assert.rejects(pending, /owned Chrome connection closed/);
  assert.equal(cdp.pending.size, 0);
});

test('report digest preserves effective disabled folding without presenting smoke speed as a benchmark', () => {
  const digest = digestReport({ steps: [{ status: 'ok', runtime: { maia3: { ortRuntime: { dequantFoldEnabled: false } } } }] });
  assert.equal(digest.dequantFold, 'false');
  assert.equal(digest.status, 'ok');
});
