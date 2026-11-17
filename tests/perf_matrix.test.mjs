import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import test from 'node:test';
import { digestReport, phaseOf, soakUrl, summarize, WORKER_READY_SCRIPT } from '../scripts/perf_matrix/common.mjs';
import { startCountingProxy } from '../scripts/perf_matrix/proxy.mjs';

const runner = readFileSync(new URL('../scripts/perf_matrix/run_matrix.mjs', import.meta.url), 'utf8');
const validity = new Function(`${runner.slice(runner.indexOf('function validity(r)'), runner.indexOf('function aggregate(dir)'))}; return validity;`)();

test('proxy counts extensionless v2 identity/Brotli model bytes but not runtime artifacts', async (t) => {
  const modelHash = 'a'.repeat(64),
    representationHash = 'b'.repeat(64),
    runtimeHash = 'c'.repeat(64);
  const upstream = createServer((req, res) => {
    if (req.url.endsWith('/manifest.json')) {
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ models: [{ sha256: modelHash }] }));
    } else res.end(Buffer.alloc(req.url.includes('/br/') ? 60 : 100));
  });
  await new Promise((resolve) => upstream.listen(0, '127.0.0.1', resolve));
  const proxy = await startCountingProxy(`http://127.0.0.1:${upstream.address().port}`);
  t.after(async () => {
    await proxy.close();
    await new Promise((resolve) => {
      upstream.close(resolve);
      upstream.closeAllConnections();
    });
  });
  const paths = [
    `/artifacts/sha256/${modelHash}/identity`,
    `/artifacts/sha256/${modelHash}/br/${representationHash}`,
    `/artifacts/sha256/${runtimeHash}/identity`,
    '/models/lc0/net.onnx',
    '/models/lc0/manifest.json',
  ];
  for (const path of paths) await (await fetch(proxy.url + path)).arrayBuffer();
  const model = proxy.modelRequests(0);
  assert.deepEqual(
    model.map((r) => r.path),
    [paths[0], paths[1], paths[3]],
  );
  assert.equal(
    model.reduce((sum, r) => sum + r.bytes, 0),
    260,
  );
});

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
    { context: { secure: false, crossOriginIsolated: true } },
    { context: { secure: true, crossOriginIsolated: false } },
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

test('mini mutex releases only its own owner and never deletes a stale foreign directory', () => {
  const body = runner.slice(runner.indexOf('function acquireLock(dir)'), runner.indexOf('/** Validity rules'));
  let owner = '',
    exists = false,
    removals = 0;
  const make = new Function(
    'mkdirSync',
    'writeFileSync',
    'readFileSync',
    'rmSync',
    'join',
    'hostname',
    'process',
    'Atomics',
    `${body}; return {acquireLock,releaseLock};`,
  );
  const api = make(
    () => {
      if (exists) throw Object.assign(Error('exists'), { code: 'EEXIST' });
      exists = true;
    },
    (_path, text) => {
      owner = text;
    },
    () => owner,
    () => {
      removals++;
      exists = false;
    },
    (...parts) => parts.join('/'),
    () => 'mini',
    {
      pid: 123,
      kill() {
        throw Object.assign(Error('gone'), { code: 'ESRCH' });
      },
    },
    {
      wait() {
        throw Error('waiting safely');
      },
    },
  );
  assert.equal(api.acquireLock('/owned'), true);
  api.releaseLock('/owned');
  assert.equal(removals, 1);
  exists = true;
  owner = 'mini 999 old';
  api.releaseLock('/owned');
  assert.equal(removals, 1);
  assert.throws(() => api.acquireLock('/owned'), /waiting safely/);
  assert.equal(removals, 1, 'stale-lock takers must not race to erase a new live owner');
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

test('Chrome preserves completed cold evidence and records a later warm harness failure', async () => {
  const source = readFileSync(new URL('../scripts/perf_matrix/chrome_session.mjs', import.meta.url), 'utf8');
  const body = source.slice(source.indexOf('const runs = [];'), source.indexOf('async function measure('));
  const cold = { pid: 123, targetId: 'owned', report: { status: 'ok' } };
  let written,
    closed = 0;
  const execute = new Function(
    'spec',
    'measure',
    'cdp',
    'processes',
    'sleep',
    'ab',
    'proxy',
    'writeFileSync',
    'outPath',
    'log',
    `return (async () => {${body}})();`,
  );
  await execute(
    { net: 'maia3', plan: ['cold', 'warm-fresh'], cdpUrl: 'ws://owned' },
    async (kind) => {
      if (kind === 'cold') return cold;
      throw Error('injected warm navigation failure');
    },
    {
      send: async () => {},
      ws: {
        close() {
          closed++;
        },
      },
    },
    async () => [],
    async () => {},
    () => {},
    { close: async () => {} },
    (_path, text) => {
      written = JSON.parse(text);
    },
    'owned.json',
    () => {},
  );
  assert.equal(written.length, 2);
  assert.deepEqual(written[0], cold);
  assert.equal(written[1].cache, 'warm-fresh');
  assert.equal(written[1].report.status, 'harness-error');
  assert.match(written[1].report.error, /injected warm/);
  assert.equal(closed, 1);
});

test('Chrome done footprint reads the actual renderer mark, not the Safari mark schema', () => {
  const source = readFileSync(new URL('../scripts/perf_matrix/chrome_session.mjs', import.meta.url), 'utf8');
  const assignment = source.match(/record\.doneMb = [^;]+;/)[0];
  const read = new Function('record', 'marks', `${assignment} return record.doneMb;`);
  assert.equal(read({}, { settle: { rendererTopMb: 321 } }), 321);
  assert.equal(read({}, {}), null);
});

test('zero-wait Safari creation returns a pairing conflict after one attempt so the outer runner can yield benchlock', async () => {
  const source = readFileSync(new URL('../scripts/perf_matrix/safari_session.mjs', import.meta.url), 'utf8');
  const body = source.slice(source.indexOf('async function createSession()'), source.indexOf('const proxy ='));
  let calls = 0;
  const run = new Function('spec', 'wd', 'sleep', 'log', `${body}; return createSession();`);
  await assert.rejects(
    run(
      { sessionWaitMs: 0 },
      async () => {
        calls++;
        throw Error('already paired with another WebDriver session');
      },
      () => {
        throw Error('must not sleep inside the lock');
      },
      () => {},
    ),
    /already paired/,
  );
  assert.equal(calls, 1);
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
