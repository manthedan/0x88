import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer, get } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import { startCountingProxy } from '../scripts/perf_matrix/proxy.mjs';
import {
  chromeFootprint,
  compareRuns,
  footprintSum,
  JOURNEYS,
  jsBytesByRoute,
  naturalVisibilityValid,
  parseArgs,
  peakFootprint,
  plan,
  previewEnvironment,
  requestFailures,
  safariWebContentPid,
  stats,
  verdict,
} from '../scripts/perf_routes/gate.mjs';
import { chooseHumanMove, playHumanTurn } from '../scripts/perf_routes/human.mjs';
import { arenaGameFailed, DOCS_READY, observeMovePvs, requireAnalysisForMove } from '../scripts/perf_routes/journeys.mjs';
import { acquireLock, claimOutput, cleanupMarker, releaseAfterSession, safariCreationRefused, waitForCleanup } from '../scripts/perf_routes/lifecycle.mjs';

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

test('served build URLs must be origins; path prefixes cannot be measured', async () => {
  const withUrls = (baseline, candidate) => args('--baseline-url', baseline, '--candidate-url', candidate);
  const origin = 'http://127.0.0.1:1234';
  assert.equal(withUrls(origin, `${origin}/`)['baseline-url'], origin);
  for (const suffix of ['/branch/', '/index.html', '/?ref=main', '/#candidate']) {
    assert.throws(() => withUrls(origin + suffix, origin), /origin-only/);
    assert.throws(() => withUrls(origin, origin + suffix), /origin-only/);
  }
  await assert.rejects(startCountingProxy(`${origin}/branch/`), /origin-only/);
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
      res.write('<html><head>'); // Deliberately chunked upstream HTML.
      res.end('<script>app()</script></head></html>');
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
  const htmlResponse = await fetch(proxy.url + '/page');
  assert.equal(htmlResponse.headers.has('transfer-encoding'), false);
  assert.match(await htmlResponse.text(), /<head><script>probe\(\)<\/script><script>app/);
  assert.equal(await (await fetch(proxy.url + '/app.js')).text(), 'hello');
  assert.equal(proxy.requests.find((r) => r.path === '/app.js').bytes, 5);
  assert.ok(proxy.requests.every((r) => r.complete && !r.failure));
});

test('proxy records upstream failure before headers, truncated bodies and downstream aborts', async (t) => {
  const upstream = createServer((req, res) => {
    if (req.url === '/no-headers.js') res.destroy();
    else if (req.url === '/truncated.js' || req.url === '/disconnect.js') {
      res.writeHead(200, { 'content-length': 999 });
      res.write('partial', () => {
        if (req.url === '/truncated.js') res.destroy();
      });
    } else res.writeHead(404).end();
  });
  await new Promise((r) => upstream.listen(0, '127.0.0.1', r));
  const proxy = await startCountingProxy(`http://127.0.0.1:${upstream.address().port}`);
  t.after(async () => {
    await proxy.close();
    await new Promise((r) => {
      upstream.close(r);
      upstream.closeAllConnections();
    });
  });
  assert.equal((await fetch(`${proxy.url}/no-headers.js`)).status, 502);
  await assert.rejects(fetch(`${proxy.url}/truncated.js`).then((r) => r.text()));
  await new Promise((r) => {
    get(`${proxy.url}/disconnect.js`, (res) => {
      res.once('data', () => res.destroy());
      res.once('close', r);
    });
  });
  await fetch(`${proxy.url}/reckless/optional.js`, { method: 'HEAD' });
  const failures = requestFailures(proxy.requests);
  for (const path of ['/no-headers.js', '/truncated.js', '/disconnect.js']) {
    const entry = failures.unexpected.find((r) => r.path === path);
    assert.ok(entry?.failure, `${path} must have a transport failure`);
    assert.equal(entry.complete, false);
  }
  assert.equal(failures.unexpected.find((r) => r.path === '/truncated.js').status, 200);
  assert.equal(failures.unexpected.find((r) => r.path === '/truncated.js').clientAborted, false);
  assert.equal(failures.unexpected.find((r) => r.path === '/disconnect.js').clientAborted, true);
  assert.equal(failures.optional[0].path, '/reckless/optional.js');
  const late = { at: 10, path: '/docs/chunk.js', status: 200, complete: false, failure: null };
  const bigNetProbe = { path: '/models/lc0/t3-big.onnx', method: 'HEAD', status: 404, complete: true, failure: null };
  const brokenProbe = { ...bigNetProbe, complete: false, failure: 'disconnected' };
  const missingModel = { ...bigNetProbe, method: 'GET' };
  const queenOdds = { ...bigNetProbe, path: '/models/lc0/lqo_v2.f16.qdq8.onnx' };
  const logos = ['/engine-logos/stockfish.png', '/engine-logos/berserk.jpg'].map((path) => ({ ...bigNetProbe, path }));
  const recklessCancel = { ...late, method: 'GET', path: '/reckless/engine.wasm', clientAborted: true };
  const recklessBroken = { ...recklessCancel, clientAborted: false, failure: 'upstream truncated' };
  assert.deepEqual(requestFailures([bigNetProbe, queenOdds, ...logos, recklessCancel, brokenProbe, missingModel, late, recklessBroken]), {
    optional: [bigNetProbe, queenOdds, ...logos, recklessCancel],
    unexpected: [brokenProbe, missingModel, late, recklessBroken],
  });
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
    location: { pathname: '/app/analysis/', href: 'http://probe/app/analysis/', origin: 'http://probe' },
    URL,
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
  cpu.postMessage({ type: 'init', id: 2, modelUrl: 'https://assets.example/net.onnx' });
  assert.equal(window.__routePerf.externalAssets[0], 'https://assets.example/net.onnx');
  cpu.receive({ type: 'searchProgress', id: 3, progress: { fen: 'current', visits: 500, completedVisits: 5, elapsedMs: 1000 } });
  cpu.receive({ type: 'searchResult', id: 3, result: { fen: 'current', visits: 502, stats: { completedVisits: 7 }, elapsedMs: 1000 } });
  cpu.receive({ type: 'searchResult', id: 4, result: { fen: 'cached', visits: 500, stats: { completedVisits: 0 }, elapsedMs: 1 } });
  assert.deepEqual(Array.from(window.__routePerf.rates['lc0-visits']), [5, 7]);
  assert.equal(window.__routePerf.analysisProgress.at(-1).fen, 'cached', 'a zero-new-work response proves the FEN request without inflating speed');
  assert.equal(window.__routePerf.analysisProgress[0].fen, 'current');
  assert.equal(window.__routePerf.peakWorkers, 2);
  cpu.receive({ type: 'disposed' });
  cpu.terminate();
  sf.terminate();
  assert.equal(window.__routePerf.workers, 0);
  assert.equal(cpu.terminated, true);
  const initialErrors = [...window.__routePerf.errors];
  cpu.receive({ type: 'error', error: 'dispose failed' });
  assert.equal(initialErrors.length, 0);
  assert.equal(window.__routePerf.errors.at(-1), 'dispose failed');
  const source = readFileSync(new URL('../scripts/perf_routes/session.mjs', import.meta.url), 'utf8');
  assert.ok(source.indexOf('result.finalWorkerErrors =') > source.indexOf("result.metrics['workers.residual'] ="));
  assert.match(source, /if \(result\.finalWorkerErrors\.length\) throw/);
});

test('probe disconnects and drops Play observer/host on navigation and detects external fetches', async () => {
  let poll,
    disconnected = 0,
    observed = 0;
  const location = { pathname: '/app/play/', href: 'http://probe/app/play/', origin: 'http://probe' };
  const host = { querySelector: () => ({ textContent: 'Lc0 search: 10 v/s' }) };
  const document = { querySelector: () => host, querySelectorAll: () => [] };
  const window = { Worker: class {}, fetch: async () => ({ url: 'https://assets.example/glue.js' }) };
  runInNewContext(readFileSync(new URL('../scripts/perf_routes/probe.js', import.meta.url), 'utf8'), {
    window,
    location,
    document,
    URL,
    performance: { now: () => 100, setResourceTimingBufferSize() {} },
    PerformanceObserver: { supportedEntryTypes: [] },
    MutationObserver: class {
      observe() {
        observed++;
      }
      disconnect() {
        disconnected++;
      }
    },
    setTimeout() {},
    setInterval(callback) {
      poll = callback;
    },
  });
  poll();
  assert.equal(observed, 1);
  location.pathname = '/docs/';
  poll();
  assert.equal(disconnected, 1);
  location.pathname = '/app/play/';
  poll();
  assert.equal(observed, 2, 'returning Play gets a fresh observer, not the detached host');
  await window.fetch('https://assets.example/glue.js');
  assert.equal(window.__routePerf.externalAssets[0], 'https://assets.example/glue.js');
});

test('Arena main-thread speed subtracts reused visits instead of trusting the displayed root-total rate', () => {
  let poll,
    fen = 'first',
    visits = '100',
    diagnostic = 'Small: searches 1 · last 100ms · reused visits 80';
  const card = {
    dataset: { engineId: 'lc0:small' },
    querySelector: () => ({ dataset: { fen } }),
    querySelectorAll: () => [{ querySelector: (s) => ({ textContent: s === 'dt' ? 'Visits' : visits }) }],
  };
  const document = { querySelector: () => ({ textContent: diagnostic }), querySelectorAll: () => [card] };
  const window = { Worker: class {} };
  runInNewContext(readFileSync(new URL('../scripts/perf_routes/probe.js', import.meta.url), 'utf8'), {
    window,
    document,
    URL,
    location: { pathname: '/app/arena/', href: 'http://probe/app/arena/', origin: 'http://probe' },
    performance: { now: () => 100, setResourceTimingBufferSize() {} },
    PerformanceObserver: { supportedEntryTypes: [] },
    setTimeout() {},
    setInterval(callback) {
      poll = callback;
    },
  });
  poll();
  fen = 'second';
  visits = '50';
  diagnostic = 'Small: searches 2 · last 100ms · reused visits 90';
  poll();
  assert.deepEqual(Array.from(window.__routePerf.rates['lc0-visits']), [200, 400]);
  fen = 'missed';
  diagnostic = 'Small: searches 4 · last 100ms · reused visits 100';
  poll();
  assert.equal(window.__routePerf.rates['lc0-visits'].length, 2, 'do not attribute multiple skipped searches to one card');
});

test('host-side human moves are deterministic/legal and reject terminal or wrong-turn games', () => {
  assert.equal(playHumanTurn('You are in check'), true);
  assert.equal(playHumanTurn('Your move'), true);
  assert.equal(playHumanTurn('Lc0 Small wins — checkmate'), false);
  assert.equal(playHumanTurn('Lc0 Small to move'), false);
  const opening = chooseHumanMove([]);
  assert.equal(chooseHumanMove([]), opening);
  assert.match(opening, /^[a-h][12][a-h][34]$/);
  assert.match(chooseHumanMove(['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6', 'd3', 'Bc5']), /^[a-h][1-8][a-h][1-8]$/);
  assert.throws(() => chooseHumanMove(['e4']), /White turn/);
  assert.throws(() => chooseHumanMove(['not-san']), /Cannot replay/);
  assert.throws(() => chooseHumanMove(['f3', 'e5', 'g4', 'Qh4#']), /ended before/);
});

test('hidden intervals invalidate natural measurements even if visibility is restored', () => {
  assert.equal(naturalVisibilityValid([{ state: 'visible' }, { state: 'hidden' }, { state: 'visible' }], 'visible'), false);
  assert.equal(naturalVisibilityValid([{ state: 'visible' }], 'hidden'), false);
  assert.equal(naturalVisibilityValid([{ state: 'visible' }], 'visible'), true);
  assert.equal(naturalVisibilityValid([], 'visible'), false);
});

test('residual Docs readiness rejects an error page with a matching pathname', () => {
  const ready = new Function('document', 'location', `return Boolean(${DOCS_READY});`);
  const location = { pathname: '/docs/' };
  assert.equal(ready({ querySelector: () => null }, location), false);
  assert.equal(ready({ querySelector: (selector) => (selector === '#doc-h1' ? { textContent: 'Using 0x88' } : {}) }, location), true);
});

test('visited routes with cached JS are explicit zero baselines, not missing samples', () => {
  const marks = [
    { route: 'analysis', at: 0 },
    { route: 'play', at: 10 },
    { route: 'arena', at: 20 },
  ];
  const baseline = jsBytesByRoute([{ path: '/app.js', at: 1, bytes: 100 }], marks);
  assert.deepEqual(baseline, { analysis: 100, play: 0, arena: 0 });
  const candidate = jsBytesByRoute(
    [
      { path: '/app.js', at: 1, bytes: 100 },
      { path: '/lazy.js', at: 11, bytes: 50 },
    ],
    marks,
  );
  assert.equal(verdict('jsBytes.play', same(baseline.play), same(candidate.play)).status, 'regressed');
  assert.deepEqual(previewEnvironment(12345, { PORT: '9999', OTHER: 'kept' }), { PORT: '12345', OTHER: 'kept' });
});

test('Safari peaks use kernel intervals; Chrome totals never sum independent process maxima', () => {
  const samples = [
    { rendererMb: 10, gpuMb: 20, rendererIntervalMb: 80 },
    { rendererMb: 20, gpuMb: 5, rendererIntervalMb: 25 },
  ];
  assert.equal(peakFootprint('safari', samples, 70), 80);
  assert.equal(peakFootprint('safari', samples, 90), 90);
  assert.equal(peakFootprint('chrome', samples), 30);
  assert.throws(() => peakFootprint('chrome', []), /No footprint/);
});

test('Chrome GPU and renderer sums reject missing or invalid reads instead of reducing the total', () => {
  const readings = new Map([[100, { cur: 5, life: 10, interval: 8 }]]);
  assert.throws(() => footprintSum([100, 101], readings, 'Chrome GPU'), /Chrome GPU footprint for PID 101/);
  assert.throws(() => footprintSum([], readings, 'Chrome GPU'), /No Chrome GPU PID/);
  readings.set(101, { cur: 3, life: 7, interval: NaN });
  assert.throws(() => footprintSum([100, 101], readings, 'Chrome GPU'), /Chrome GPU footprint for PID 101/);
  assert.throws(() => footprintSum([999], readings, 'renderer'), /renderer footprint for PID 999/);
  readings.get(101).interval = 4;
  assert.deepEqual(footprintSum([100, 101], readings, 'Chrome GPU'), { cur: 8, life: 17, interval: 12 });
});

test('Chrome retries only a verified renderer exit, without losing interval peaks or accepting a missing GPU', async () => {
  const old = { rendererPids: [1, 2], gpuPids: [3] };
  const fresh = { rendererPids: [2, 4], gpuPids: [3] };
  const first = new Map([
    [2, { cur: 4, life: 12, interval: 12 }],
    [3, { cur: 2, life: 10, interval: 10 }],
  ]);
  const second = new Map([
    [2, { cur: 3, life: 12, interval: 5 }],
    [3, { cur: 3, life: 10, interval: 4 }],
    [4, { cur: 2, life: 3, interval: 3 }],
  ]);
  const discover = (states) => async () => states.shift();
  const read = (reads) => () => reads.shift();
  const sample = await chromeFootprint(discover([old, fresh]), read([first, second]));
  assert.deepEqual(sample.render, { cur: 5, life: 15, interval: 15 });
  assert.deepEqual(sample.gpu, { cur: 3, life: 10, interval: 10 });
  await assert.rejects(chromeFootprint(discover([old, old]), read([first])), /renderer footprint for PID 1/);
  await assert.rejects(chromeFootprint(discover([old]), read([new Map()])), /Chrome GPU footprint for PID 3/);
});

test('Safari requires exactly one new WebContent PID without ranking by footprint', () => {
  const before = new Set([10, 12]);
  assert.throws(() => safariWebContentPid(before, [10, 12]), /found 0/);
  assert.throws(() => safariWebContentPid(before, [10, 12, 14, 16]), /found 2/);
  assert.equal(safariWebContentPid(before, [10, 12, 14]), 14);
});

test('Analysis records when every engine has a PV and accepts only in-window evidence, even when polled late', async () => {
  const move = { fen: 'new', started: 10 };
  const window = { __routePerf: { analysisProgress: [{ fen: 'old', at: 10 }] }, __routeMove: { start: 10 }, __analysisSchedulingState: () => ({ fen: 'new' }) };
  let engines = ['Lc0'],
    now = 10,
    mutation,
    disconnected = false;
  const document = {
    querySelector: () => ({ querySelectorAll: () => engines.map((engine) => ({ dataset: { pvFen: 'new', engine }, querySelector: () => true })) }),
    querySelectorAll: () => [1, 2],
  };
  const evaluate = async (expression) =>
    runInNewContext(expression, {
      window,
      document,
      performance: { now: () => now },
      setTimeout: () => 1,
      clearTimeout() {},
      MutationObserver: class {
        constructor(fn) {
          mutation = fn;
        }
        observe() {}
        disconnect() {
          disconnected = true;
        }
      },
    });
  await evaluate(`(${observeMovePvs.toString()})(window.__routeMove)`);
  assert.equal(window.__routeMove.pvAt, undefined);
  engines = ['Lc0', 'SF'];
  now = 11;
  mutation();
  assert.equal(window.__routeMove.pvAt, 11);
  assert.equal(disconnected, true);
  now = 12;
  const observe = () => requireAnalysisForMove(evaluate, move, 1);
  await assert.rejects(observe(), /within 1ms of user move/);
  window.__routePerf.analysisProgress = [
    { fen: 'new', at: 9 },
    { fen: 'new', at: 12 },
  ];
  await assert.rejects(observe(), /within 1ms of user move/);
  window.__routePerf.analysisProgress.push({ fen: 'new', at: 11 });
  assert.equal((await observe()).workerAt, 11);
  window.__routeMove.pvAt = 12;
  await assert.rejects(observe(), /within 1ms of user move/);
});

test('Arena rejects illegal/missing moves even when a finished-game chip exists', () => {
  for (const reason of ['illegal e2e5', 'resigned', 'cancelled', 'forfeit', 'failed', 'timeout']) assert.equal(arenaGameFailed(`1-0 (${reason})`), true);
  for (const reason of ['checkmate', 'stalemate', 'threefold repetition', 'ply limit']) assert.equal(arenaGameFailed(`1/2-1/2 (${reason})`), false);
});

test('UA collection runs only after natural footprint and worker observations are frozen', () => {
  const source = readFileSync(new URL('../scripts/perf_routes/session.mjs', import.meta.url), 'utf8');
  assert.equal((source.match(/await uaMemory\(/g) ?? []).length, 1);
  assert.ok(source.indexOf('await uaMemory(') > source.indexOf("result.metrics['workers.residual'] ="));
  assert.ok(source.indexOf('await uaMemory(') > source.indexOf("result.metrics['memory.residualMb'] ="));
});

function scratch(t) {
  const dir = mkdtempSync(join(tmpdir(), 'route-gate-test-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test('output claim rejects any existing directory, even before the winner writes its plan', (t) => {
  const out = join(scratch(t), 'nested', 'cohort');
  claimOutput(out);
  assert.throws(() => claimOutput(out), /already exists/);
  assert.equal(existsSync(join(out, 'plan.json')), false);
});

test('interrupted mutex wait and release cannot remove another owner', async (t) => {
  const lock = join(scratch(t), 'mutex');
  const controller = new AbortController();
  const release = await acquireLock(lock, controller.signal);
  const owner = join(lock, 'owner');
  writeFileSync(owner, 'replacement owner\n');
  release();
  const waiting = acquireLock(lock, controller.signal);
  controller.abort(Error('test interrupt'));
  await assert.rejects(waiting, /test interrupt/);
  assert.equal(readFileSync(owner, 'utf8'), 'replacement owner\n');
});

test('Safari releases after a refused connect, but not when the result of POST /session could have been lost', () => {
  const wrapped = (cause) => new TypeError('fetch failed', { cause });
  const refused = Object.assign(Error('port closed'), { code: 'ECONNREFUSED', syscall: 'connect' });
  const reset = Object.assign(Error('peer reset'), { code: 'ECONNRESET', syscall: 'read' });
  assert.equal(safariCreationRefused(wrapped(refused)), true);
  assert.equal(safariCreationRefused(wrapped(new AggregateError([refused, refused]))), true);
  assert.equal(safariCreationRefused(wrapped(new AggregateError([refused, reset]))), false);
  assert.equal(safariCreationRefused(wrapped(reset)), false);
  assert.equal(safariCreationRefused(Object.assign(Error('already paired'), { webDriverError: 'session not created' })), true);
});

test('unconfirmed owned browser cleanup retains the mutex', async (t) => {
  const dir = scratch(t);
  const lock = join(dir, 'mutex');
  const output = join(dir, 'attempt.json');
  const release = await acquireLock(lock, new AbortController().signal);
  writeFileSync(cleanupMarker(output), 'Owned Safari session test-session\n');
  assert.equal(releaseAfterSession(output, release), false);
  assert.equal(existsSync(join(lock, 'owner')), true);
  rmSync(cleanupMarker(output));
  assert.equal(releaseAfterSession(output, release), true);
  assert.equal(existsSync(lock), false);
});

test('wrapped session retries owned Safari teardown without returning while cleanup is pending', async (t) => {
  const output = join(scratch(t), 'attempt.json');
  const marker = cleanupMarker(output);
  writeFileSync(marker, 'test session');
  let attempts = 0;
  await waitForCleanup(
    output,
    async () => {
      assert.equal(existsSync(marker), true);
      if (++attempts === 1) throw Error('temporary driver timeout');
    },
    async () => {},
  );
  assert.equal(attempts, 2);
  assert.equal(existsSync(marker), false);
});

for (const [signal, code] of [
  ['SIGINT', 130],
  ['SIGTERM', 143],
]) {
  test(`${signal} waits for owned session cleanup before releasing mutex`, { timeout: 5000 }, async (t) => {
    const lock = join(scratch(t), 'mutex');
    const module = new URL('../scripts/perf_routes/lifecycle.mjs', import.meta.url).href;
    const session = `
      process.on('SIGTERM', () => setTimeout(() => {
        if (require('fs').existsSync(${JSON.stringify(join(lock, 'owner'))})) console.log('cleaned up under mutex');
        process.exit(2);
      }, 25));
      console.log('ready');
      setInterval(() => {}, 1000);`;
    const coordinator = spawn(process.execPath, [
      '--input-type=module',
      '-e',
      `
      import { acquireLock, runOwnedSession, watchShutdown } from ${JSON.stringify(module)};
      const { signal, dispose } = watchShutdown();
      const release = await acquireLock(${JSON.stringify(lock)}, signal);
      try {
        await runOwnedSession(process.execPath, ['-e', ${JSON.stringify(session)}], signal, false);
        process.exitCode = signal.reason?.exitCode;
      } finally { release(); dispose(); }`,
    ]);
    const closed = once(coordinator, 'close');
    let output = '';
    coordinator.stdout.on('data', (data) => {
      output += data;
      if (String(data).includes('ready')) coordinator.kill(signal);
    });
    assert.deepEqual(await closed, [code, null]);
    assert.match(output, /cleaned up under mutex/);
    assert.equal(existsSync(lock), false);
  });
}

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
