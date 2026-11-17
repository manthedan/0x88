import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import { startCountingProxy } from '../scripts/perf_matrix/proxy.mjs';
import { compareRuns, JOURNEYS, parseArgs, plan, stats, verdict } from '../scripts/perf_routes/gate.mjs';

const args = (...extra) => parseArgs(['--baseline', 'main', '--candidate', 'HEAD', '--dry-run', ...extra]);

test('route gate strict parsing, mandatory arms, mutex, URLs, repeats, subsets', () => {
  assert.equal(args().repeats, 5);
  assert.deepEqual(args().journeys, JOURNEYS);
  assert.equal(parseArgs(['--help']).help, true);
  for (const argv of [
    [],
    ['--baseline'],
    ['--baseline', 'a', '--candidate', 'b'],
    ['--oops'],
    ['--repeats', '4'],
    ['--repeats', '5.5'],
    ['--repeats', 'NaN'],
    ['--browser', 'firefox'],
    ['--journey', 'oops'],
    ['--benchlock', 'a', '--lock-dir', 'b'],
    ['--baseline-url', 'file:///tmp'],
    ['--baseline-url', 'http://localhost'],
    ['--driver', 'http://user:pass@localhost'],
  ]) {
    assert.throws(() => (argv.length && argv[0] !== '--baseline' && argv[0] !== '--oops' ? args(...argv) : parseArgs(argv)), undefined, JSON.stringify(argv));
  }
  assert.deepEqual(args('--journey', 'navigation', '--journey', 'navigation').journeys, ['navigation']);
  assert.throws(() => args('--out', 'a', '--out', 'b'), /Duplicate/);
});

test('route plan interleaves >=5 repeats and alternates arm order, no grouped A then B drift', () => {
  const cells = plan(args());
  assert.equal(cells.length, 140);
  assert.deepEqual(
    cells.slice(0, 2).map((c) => c.arm),
    ['baseline', 'candidate'],
  );
  assert.deepEqual(
    cells
      .filter((c) => c.repeat === 1)
      .slice(0, 2)
      .map((c) => c.arm),
    ['candidate', 'baseline'],
  );
  for (const j of JOURNEYS)
    for (const b of ['chrome', 'safari'])
      for (const a of ['baseline', 'candidate']) assert.equal(cells.filter((c) => c.journey === j && c.browser === b && c.arm === a).length, 5);
});
const same = (n) => Array(5).fill(n);
test('verdict direction and strict memory/speed/long-task floors', () => {
  assert.equal(verdict('memory.peakMb', same(100), same(105)).status, 'within-noise');
  assert.equal(verdict('memory.peakMb', same(100), same(106)).status, 'regressed');
  assert.equal(verdict('speed.lc0-visits', same(100), same(94)).status, 'regressed');
  assert.equal(verdict('speed.lc0-visits', same(100), same(106)).status, 'improved');
  assert.equal(verdict('longTaskTotalMs', same(100), same(109)).status, 'within-noise');
  assert.equal(verdict('longTaskTotalMs', same(100), same(111)).status, 'regressed');
});
test('MAD uses both arms, noisy takes precedence, missing/short cohorts and zero baselines never pass silently', () => {
  assert.equal(verdict('memory.peakMb', [98, 99, 100, 101, 102], same(106)).threshold, 0.05);
  assert.ok(Math.abs(verdict('memory.peakMb', [90, 95, 100, 105, 110], same(110)).threshold - 0.15) < 1e-12);
  assert.equal(verdict('memory.peakMb', [80, 90, 100, 110, 120], same(140)).status, 'noisy');
  assert.equal(verdict('memory.peakMb', same(100), [80, 90, 100, 110, 120]).status, 'noisy');
  assert.equal(verdict('jsBytes', same(100), [110]).status, 'insufficient');
  assert.equal(verdict('jsBytes', same(100), []).status, 'unavailable');
  assert.equal(verdict('workers.residual', same(0), same(1)).status, 'regressed');
  assert.equal(verdict('workers.residual', same(0), same(0)).status, 'within-noise');
  assert.equal(stats([0.01, 0.02, 0.03]).mad, 0.009999999999999998);
  assert.throws(() => stats([1, NaN]));
});
test('self-comparison balanced deterministic samples have no false regressions; invalid attempts cannot disappear', () => {
  const options = args('--browser', 'chrome', '--journey', 'analysis-default');
  const rows = plan(options).map((c) => ({
    ...c,
    status: 'ok',
    metrics: {
      'memory.peakMb': 100,
      'memory.settledMb': 80,
      'memory.residualMb': 40,
      'workers.peak': 2,
      'workers.residual': 0,
      boardInteractiveMs: 40,
      firstLineMs: 200,
      jsBytes: 1000,
      'speed.lc0-visits': 100,
      'speed.stockfish-nps': 100,
      inputLatencyMs: 20,
      longTaskCount: 0,
      longTaskTotalMs: 0,
      longTaskP95Ms: 0,
      'memory.gpuPeakMb': 10,
    },
  }));
  assert.equal(compareRuns(rows, options)[0].status, 'clean');
  rows[0].status = 'invalid';
  rows[0].error = 'crash';
  assert.equal(compareRuns(rows, options)[0].status, 'incomplete');
  assert.equal(compareRuns(rows, options)[0].invalid.length, 1);
  assert.equal(compareRuns([], options)[0].status, 'incomplete');
});
test('proxy prepends early route probes only to identity HTML, preserving script byte accounting', async (t) => {
  const upstream = createServer((req, res) => {
    res.setHeader('content-type', req.url === '/page' ? 'text/html' : 'text/javascript');
    if (req.url === '/page') {
      assert.equal(req.headers['accept-encoding'], 'identity');
      res.end('<html><head><script>app()</script></head></html>');
    } else res.end('hello');
  });
  await new Promise((r) => upstream.listen(0, '127.0.0.1', r));
  const proxy = await startCountingProxy(`http://127.0.0.1:${upstream.address().port}`, { htmlScript: 'probe()' });
  t.after(async () => {
    await proxy.close();
    await new Promise((r) => {
      upstream.close(r);
      upstream.closeAllConnections();
    });
  });
  assert.match(await (await fetch(proxy.url + '/page')).text(), /<head><script>probe\(\)<\/script><script>app/);
  assert.equal(await (await fetch(proxy.url + '/app.js')).text(), 'hello');
  assert.equal(proxy.requests.find((r) => r.path === '/app.js').bytes, 5);
});

test('route probes retain original worker payloads, count teardown once and parse both UCI transports', () => {
  class NativeWorker {
    constructor() {
      this.listeners = {};
    }
    addEventListener(type, callback) {
      this.listeners[type] = callback;
    }
    postMessage(message) {
      this.original = message;
    }
    terminate() {
      this.terminated = true;
    }
    receive(data) {
      this.listeners.message({ data });
    }
  }
  const window = { Worker: NativeWorker };
  runInNewContext(readFileSync(new URL('../scripts/perf_routes/probe.js', import.meta.url), 'utf8'), {
    window,
    performance: { now: () => 100, setResourceTimingBufferSize() {} },
    PerformanceObserver: { supportedEntryTypes: [] },
    document: {},
    location: { pathname: '/app/analysis/' },
    setTimeout() {},
    setInterval() {},
  });
  const sf = new window.Worker('stockfish.js'),
    cpu = new window.Worker('blob:cpu');
  const init = { type: 'init', id: 1, jsUrl: '/berserk/berserk.js', model: new ArrayBuffer(2) };
  cpu.postMessage(init);
  assert.equal(cpu.original, init);
  sf.receive('id name Stockfish 19');
  sf.receive('info depth 3 nps 1200');
  cpu.receive({ type: 'line', line: 'id name Berserk' });
  cpu.receive({ type: 'line', line: 'info depth 3 nps 900' });
  assert.equal(window.__routePerf.rates['stockfish-nps'][0], 1200);
  assert.equal(window.__routePerf.rates['berserk-nps'][0], 900);
  assert.equal(window.__routePerf.peakWorkers, 2);
  cpu.receive({ type: 'disposed' });
  cpu.terminate();
  sf.terminate();
  assert.equal(window.__routePerf.workers, 0);
  assert.equal(cpu.terminated, true);
});

test('missing throughput for one selected engine cannot yield a clean comparison', () => {
  const options = args('--browser', 'safari', '--journey', 'arena-cpu');
  const rows = plan(options).map((c) => ({
    ...c,
    status: 'ok',
    metrics: {
      'memory.peakMb': 100,
      'memory.settledMb': 80,
      'memory.residualMb': 40,
      'workers.peak': 2,
      'workers.residual': 0,
      boardInteractiveMs: 40,
      firstLineMs: 200,
      jsBytes: 1000,
      'speed.stockfish-nps': 100,
    },
  }));
  assert.equal(compareRuns(rows, options)[0].status, 'incomplete');
  assert.equal(compareRuns(rows, options)[0].metrics.find((m) => m.name === 'speed.berserk-nps').status, 'unavailable');
});
