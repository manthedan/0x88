import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import { startCountingProxy } from '../scripts/perf_matrix/proxy.mjs';
import {
  compareRuns,
  JOURNEYS,
  jsBytesByRoute,
  naturalVisibilityValid,
  parseArgs,
  peakFootprint,
  plan,
  previewEnvironment,
  stats,
  verdict,
} from '../scripts/perf_routes/gate.mjs';
import { chooseHumanMove, playHumanTurn } from '../scripts/perf_routes/human.mjs';
import { arenaGameFailed, DOCS_READY, requireAnalysisForMove } from '../scripts/perf_routes/journeys.mjs';

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
  const source = readFileSync(new URL('../scripts/perf_routes/session.mjs', import.meta.url), 'utf8');
  assert.match(source, /result\.requests,\s*\.\.\.result\.postNavigationRequests/);
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

test('Analysis requires post-move FEN-associated worker progress plus every rendered engine PV', async () => {
  const window = { __routePerf: { analysisProgress: [{ fen: 'old', at: 20 }] } };
  let rowCount = 2;
  const document = {
    querySelectorAll: (selector) =>
      selector.includes('engineSummaries')
        ? ['Lc0', 'SF'].slice(0, rowCount).map((engine) => ({ dataset: { pvFen: 'new', engine }, querySelector: () => true }))
        : [1, 2],
  };
  const evaluate = async (expression) => new Function('window', 'document', `return (${expression});`)(window, document);
  await assert.rejects(requireAnalysisForMove(evaluate, { fen: 'new', started: 10 }, 1), /Timed out/);
  window.__routePerf.analysisProgress = [{ fen: 'new', at: 9 }];
  await assert.rejects(requireAnalysisForMove(evaluate, { fen: 'new', started: 10 }, 1), /Timed out/);
  window.__routePerf.analysisProgress = [{ fen: 'new', at: 20 }];
  assert.equal(await requireAnalysisForMove(evaluate, { fen: 'new', started: 10 }, 1), true);
  rowCount = 1;
  await assert.rejects(requireAnalysisForMove(evaluate, { fen: 'new', started: 10 }, 1), /Timed out/);
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
