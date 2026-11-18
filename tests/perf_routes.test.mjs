import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, utimesSync, writeFileSync } from 'node:fs';
import { createServer, get } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import { startCountingProxy } from '../scripts/perf_matrix/proxy.mjs';
import { linkAssets, localAssets } from '../scripts/perf_routes/assets.mjs';
import {
  chromeFootprint,
  compareRuns,
  footprintSum,
  incompleteJsTransfers,
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
import {
  acquireLock,
  checkSessionCancellation,
  claimOutput,
  cleanupMarker,
  liveOwner,
  lockRun,
  ownerRecord,
  releaseAfterSession,
  runOwnedSession,
  safariCreationRefused,
  waitForCleanup,
} from '../scripts/perf_routes/lifecycle.mjs';
import {
  buildFingerprint,
  manifestMismatches,
  measuringFiles,
  readJson,
  recordAttempt,
  reusableBuild,
  runManifest,
  selectSessions,
  sessionId,
  sharesBaselineBuild,
  writeJsonAtomic,
} from '../scripts/perf_routes/resume.mjs';

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
  assert.throws(() => args('--repeats', '1'), /same-build smoke/);
});

test('a one-repeat same-build smoke measures sessions without passing the gate', () => {
  const smokeArgs = ['--baseline', 'main', '--candidate', 'main', '--dry-run', '--browser', 'chrome', '--journey', 'analysis-four', '--repeats', '1'];
  const options = parseArgs(smokeArgs);
  assert.equal(plan(options).length, 2);
  const [summary] = compareRuns(
    plan(options).map((cell) => ({ ...cell, status: 'ok', metrics: { jsBytes: 10 } })),
    options,
  );
  assert.equal(summary.invalid.length, 0);
  assert.equal(summary.metrics.find((m) => m.name === 'jsBytes').status, 'insufficient');
  assert.equal(summary.status, 'incomplete');
  assert.throws(() => parseArgs([...smokeArgs, '--baseline-url', 'http://localhost:80', '--candidate-url', 'http://localhost:81']));
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
  const optionalProbes = ['/engine-logos/stockfish.png', '/engine-logos/berserk.jpg', '/stormphrax/stormphrax-emscripten.wasm'].map((path) => ({
    ...bigNetProbe,
    path,
  }));
  const failedDownload = { ...optionalProbes.at(-1), method: 'GET' };
  const recklessCancel = { ...late, method: 'GET', path: '/reckless/engine.wasm', clientAborted: true };
  const recklessBroken = { ...recklessCancel, clientAborted: false, failure: 'upstream truncated' };
  assert.deepEqual(
    requestFailures([bigNetProbe, queenOdds, ...optionalProbes, recklessCancel, brokenProbe, missingModel, late, recklessBroken, failedDownload]),
    {
      optional: [bigNetProbe, queenOdds, ...optionalProbes, recklessCancel],
      teardownAborts: [],
      unexpected: [brokenProbe, missingModel, late, recklessBroken, failedDownload],
    },
  );
});

test('a request the proxy had to close is not a browser teardown abort', async (t) => {
  const upstream = createServer((req, res) => {
    if (req.url === '/pending.data') res.writeHead(200).write('partial');
    else res.writeHead(404).end();
  });
  await new Promise((r) => upstream.listen(0, '127.0.0.1', r));
  t.after(
    () =>
      new Promise((r) => {
        upstream.close(r);
        upstream.closeAllConnections();
      }),
  );
  const proxy = await startCountingProxy(`http://127.0.0.1:${upstream.address().port}`);
  const response = await fetch(`${proxy.url}/pending.data`);
  assert.equal(response.status, 200);
  const cutoff = { at: Date.now(), requestCount: proxy.requests.length };
  await proxy.close();
  const [request] = proxy.requests;
  assert.ok(request.endedAt >= cutoff.at);
  assert.equal(request.clientAborted, false);
  assert.deepEqual(requestFailures(proxy.requests, 'analysis-four', { cutoff }).unexpected, [request]);
});

test('availability probe cancellation needs a confirmed missing sibling, not merely a client disconnect', () => {
  const absent = { method: 'HEAD', path: '/berserk/berserk-9b84c340af7e.nn', at: 1, endedAt: 4, complete: true, status: 404 };
  const sibling = { ...absent, path: '/berserk/berserk.wasm', at: 2, endedAt: 5, complete: false, status: null, clientAborted: true };
  const notConcurrent = { ...sibling, at: 6, endedAt: 7 };
  const otherFamily = { ...sibling, path: '/stormphrax/stormphrax-emscripten.wasm' };
  const httpError = { ...sibling, status: 503 };
  assert.deepEqual(requestFailures([absent, sibling, notConcurrent, otherFamily, httpError]), {
    optional: [absent, sibling],
    teardownAborts: [],
    unexpected: [notConcurrent, otherFamily, httpError],
  });
});

test('Analysis Four may cancel its unused Viridithas prefetch, but upstream errors remain fatal', () => {
  const cancelled = { method: 'GET', path: '/viridithas/viridithas.wasm', clientAborted: true, complete: false, status: 200 };
  const truncated = { ...cancelled, clientAborted: false, failure: 'upstream truncated' };
  const httpError = { ...cancelled, status: 404 };
  assert.deepEqual(requestFailures([cancelled, truncated, httpError], 'analysis-four'), {
    optional: [cancelled],
    teardownAborts: [],
    unexpected: [truncated, httpError],
  });
  assert.deepEqual(requestFailures([cancelled], 'analysis-default').unexpected, [cancelled]);
});

// residency-vs-main sessions 017-chrome-analysis-four-baseline-r1 and 007-chrome-play-lc0-candidate-r0.
const cancelledRequests = JSON.parse(readFileSync(new URL('./fixtures/perf_routes_cancelled_requests.json', import.meta.url), 'utf8'));

test('recorded mid-route Berserk abort is redundant only with a previously completed GET of its sidecar', () => {
  const { journey, routeMarks, requests, firstDocsRequestAt } = cancelledRequests.analysisFour;
  const [prior, aborted, retry] = requests;
  assert.ok(aborted.endedAt < firstDocsRequestAt);
  assert.deepEqual(requestFailures(requests, journey, { routeMarks, cutoff: { at: firstDocsRequestAt, requestCount: requests.length } }).optional, [aborted]);
  for (const earlier of [
    { ...prior, status: 206 },
    { ...prior, method: 'HEAD' },
  ]) {
    assert.deepEqual(requestFailures([earlier, aborted, retry], journey).unexpected, [aborted]);
  }
  const otherAsset = { ...aborted, path: '/berserk/other.data' };
  assert.deepEqual(requestFailures([prior, otherAsset], journey).unexpected, [otherAsset]);
});

test('recorded Maia3 prefetch abort is optional in Play Lc0 but not if Maia3 is selected', () => {
  const { journey, requests } = cancelledRequests.playLc0;
  assert.deepEqual(requestFailures(requests, journey).optional, requests);
  assert.deepEqual(requestFailures(requests, 'play-maia3').unexpected, requests);
});

test('only client aborts from the outgoing route during navigation, or at/after the final cutoff, are teardown', () => {
  const {
    journey,
    routeMarks,
    requests: [, recorded],
    firstDocsRequestAt,
  } = cancelledRequests.analysisFour;
  const at = firstDocsRequestAt;
  const readyAt = at + 50;
  const measuredUntil = readyAt + 20000;
  const transitionStart = { ...recorded, endedAt: at };
  const transitionEnd = { ...recorded, endedAt: readyAt };
  const sessionEnd = { ...recorded, at: readyAt, endedAt: measuredUntil };
  const close = { ...sessionEnd, status: null, endedAt: measuredUntil + 125 };
  const fatal = [
    recorded,
    { ...recorded, endedAt: at - 1 },
    { ...recorded, endedAt: readyAt + 1 },
    { ...recorded, at: at + 1, endedAt: readyAt },
    { ...recorded, at: routeMarks.at(-1).at - 1, endedAt: readyAt },
    { ...sessionEnd, endedAt: measuredUntil - 1 },
    { ...sessionEnd, status: 503 },
    { ...transitionStart, status: 404 },
    { ...sessionEnd, clientAborted: false, failure: 'Truncated upstream response' },
    { ...sessionEnd, endedAt: undefined, clientAborted: false, failure: null },
  ];
  const exempted = [transitionStart, transitionEnd, sessionEnd, close];
  const cutoff = { at: measuredUntil, requestCount: fatal.length + exempted.length };
  const postMeasurementError = { ...sessionEnd, at: measuredUntil, endedAt: measuredUntil + 10, status: 503 };
  assert.deepEqual(
    requestFailures([...fatal, ...exempted, postMeasurementError], journey, { routeMarks: [...routeMarks, { route: 'docs', at, readyAt }], cutoff }),
    { optional: [], teardownAborts: exempted, unexpected: fatal },
  );
});

test('a request that survives an intermediate route is not excused by the next transition', () => {
  const marks = [
    { route: 'analysis', at: 0, readyAt: 100 },
    { route: 'play', at: 1000, readyAt: 1100 },
    { route: 'arena', at: 2000, readyAt: 2100 },
  ];
  const base = {
    method: 'GET',
    path: '/models/x.onnx',
    status: 200,
    complete: false,
    clientAborted: true,
    failure: 'Browser disconnected before response finished',
  };
  const leftover = { ...base, at: 500, endedAt: 2050 };
  const fromPlay = { ...base, at: 1500, endedAt: 2050 };
  assert.deepEqual(requestFailures([leftover, fromPlay], 'navigation', { routeMarks: marks, cutoff: { at: 9000, requestCount: 2 } }), {
    optional: [],
    teardownAborts: [fromPlay],
    unexpected: [leftover],
  });
});

test('the JS deadline ignores bodyless probes and optional prefetches, not late or aborted required bodies', () => {
  const get = { at: 10, method: 'GET', path: '/required.mjs?v=1', status: 200, bytes: 1, complete: false };
  const head = { ...get, method: 'HEAD', path: '/berserk/probe.js', bytes: 0 };
  const optional = { ...get, path: '/reckless/prefetch.js' };
  const done = { ...get, path: '/ready.js', complete: true };
  const abandoned = { ...get, path: '/other.js', endedAt: 20, clientAborted: true };
  assert.deepEqual(incompleteJsTransfers([head, optional, done, get, abandoned]), [get, abandoned]);
  assert.deepEqual(requestFailures([head], 'analysis-four').unexpected, [head]);
  assert.deepEqual(requestFailures([{ ...head, clientAborted: true, endedAt: 20 }], 'analysis-four', { cutoff: { at: 20, requestCount: 1 } }).unexpected, []);
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
  const old = { rendererPids: [1, 2, 5], gpuPids: [3] };
  const fresh = { rendererPids: [2, 4], gpuPids: [3] };
  const first = new Map([
    [2, { cur: 4, life: 12, interval: 12 }],
    [3, { cur: 2, life: 10, interval: 10 }],
    [5, { cur: 15, life: 20, interval: 20 }],
  ]);
  const second = new Map([
    [2, { cur: 3, life: 12, interval: 5 }],
    [3, { cur: 3, life: 10, interval: 4 }],
    [4, { cur: 2, life: 3, interval: 3 }],
  ]);
  const discover = (states) => async () => states.shift();
  const read = (reads) => () => reads.shift();
  const sample = await chromeFootprint(discover([old, fresh]), read([first, second]));
  assert.deepEqual(sample.render, { cur: 5, life: 15, interval: 32 });
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

test('interrupt marks a benchlock-wrapped session cancelled before browser startup without killing its wrapper', async (t) => {
  const output = join(scratch(t), 'attempt.json');
  const module = new URL('../scripts/perf_routes/lifecycle.mjs', import.meta.url).href;
  const controller = new AbortController();
  const done = runOwnedSession(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `
    import { checkSessionCancellation } from ${JSON.stringify(module)};
    try { checkSessionCancellation(${JSON.stringify(output)}, new AbortController().signal); }
    catch (error) { process.exitCode = error.message === 'Interrupted before browser startup' ? 2 : 3; }`,
    ],
    controller.signal,
    output,
  );
  controller.abort(Error('test interrupt'));
  assert.equal(await done, 2);
  assert.throws(() => checkSessionCancellation(output, new AbortController().signal), /Interrupted before browser startup/);
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

test('--resume continues an explicit --out directory only', () => {
  assert.equal(args('--out', 'x', '--resume').resume, true);
  assert.throws(() => args('--resume'), /--resume needs the --out/);
  assert.throws(() => args('--out', 'x', '--resume', '--resume'), /Duplicate/);
});

test('resume manifest names every cohort-defining difference and ignores paths/locks', () => {
  const arms = {
    baseline: { kind: 'archived-source', input: 'main', commit: 'a'.repeat(40) },
    candidate: { kind: 'archived-source', input: 'HEAD', commit: 'b'.repeat(40) },
  };
  const extras = { host: 'mac', harness: 'h1' };
  const saved = runManifest(args('--out', 'one'), arms, extras);
  assert.deepEqual(
    manifestMismatches(saved, runManifest(args('--out', 'two', '--resume', '--benchlock', '/l', '--driver', 'http://127.0.0.1:1'), arms, extras)),
    [],
  );
  // JSON round trip (as stored on disk) still matches.
  assert.deepEqual(manifestMismatches(JSON.parse(JSON.stringify(saved)), runManifest(args(), arms, extras)), []);
  // After `main` moves, resuming with the recorded SHAs is the same cohort; the ref is provenance only.
  const bySha = { baseline: { ...arms.baseline, input: arms.baseline.commit }, candidate: { ...arms.candidate, input: arms.candidate.commit } };
  assert.deepEqual(manifestMismatches(saved, runManifest(args(), bySha, extras)), []);
  assert.equal(saved.inputs.baseline, 'main');
  // A served arm's only identity is the operator's build identifier: same origin, new SHA is a new cohort.
  const served = (candidate) => ({
    baseline: { kind: 'externally-served', input: 'a1', url: 'http://127.0.0.1:1' },
    candidate: { kind: 'externally-served', input: candidate, url: 'http://127.0.0.1:2' },
  });
  assert.deepEqual(manifestMismatches(runManifest(args(), served('b1'), extras), runManifest(args(), served('b1'), extras)), []);
  assert.deepEqual(manifestMismatches(runManifest(args(), served('b1'), extras), runManifest(args(), served('b2'), extras)), ['arms.candidate.input']);
  const moved = { ...arms, candidate: { ...arms.candidate, commit: 'c'.repeat(40) } };
  assert.deepEqual(manifestMismatches(saved, runManifest(args(), moved, extras)), ['arms.candidate.commit']);
  assert.deepEqual(manifestMismatches(saved, runManifest(args('--repeats', '6', '--browser', 'chrome'), arms, extras)), ['options.browser', 'options.repeats']);
  assert.deepEqual(manifestMismatches(saved, runManifest(args('--journey', 'navigation'), arms, extras)), ['options.journeys']);
  assert.deepEqual(manifestMismatches(saved, runManifest(args(), arms, { host: 'mini', harness: 'h2' })), ['harness', 'host']);
  const prebuilt = { ...arms, baseline: { kind: 'prebuilt', input: 'dist', dist: '/d', version: '{"version":"1"}' } };
  assert.deepEqual(manifestMismatches(saved, runManifest(args(), prebuilt, extras)), [
    'arms.baseline.commit',
    'arms.baseline.dist',
    'arms.baseline.kind',
    'arms.baseline.version',
  ]);
});

test('resume reuses every recorded outcome and reruns only missing or operator-stopped sessions', () => {
  const options = args('--browser', 'chrome', '--journey', 'navigation');
  const cells = plan(options);
  const ids = cells.map(sessionId);
  assert.equal(ids[0], '000-chrome-navigation-baseline-r0');
  assert.equal(ids[3], '003-chrome-navigation-baseline-r1');
  const result = (i, extra) => ({ ...cells[i], base: 'http://127.0.0.1:9', status: 'ok', metrics: { jsBytes: i }, ...extra });
  const files = new Map([
    [ids[0], { result: result(0) }],
    [ids[1], { result: result(1, { status: 'invalid', error: 'Error: board never interactive' }) }],
    [ids[2], { result: result(2, { status: 'invalid', error: 'Interrupted by SIGINT' }) }],
    [ids[3], { result: result(3, { status: 'invalid', error: 'Error: Interrupted before browser startup\n    at x' }) }],
    [ids[4], { result: result(4, { status: 'invalid', error: 'Error: session not created: already paired' }) }],
    // A benchlock cancel can land while the browser already runs; its finished result is an outcome.
    [ids[5], { result: result(5), cancelled: true }],
    [ids[6], { result: null }], // unreadable / truncated
    [ids[7], { result: result(7, { status: 'invalid', error: 'Error: Interrupted before browser startup' }), cancelled: true }],
  ]);
  const selected = selectSessions(cells, (id) => files.get(id) ?? { result: null });
  assert.deepEqual(
    selected.map((s) => s.id),
    ids,
  );
  assert.deepEqual(
    selected.map((s) => s.cell),
    cells,
  );
  // A genuine failure stays in the denominator, exactly as in a fresh run.
  assert.deepEqual(
    selected.map((s, i) => (s.reuse ? i : null)).filter((i) => i !== null),
    [0, 1, 5],
  );
  assert.equal(selected[1].reuse.error, 'Error: board never interactive');
  // A fresh run selects every session.
  assert.ok(selectSessions(cells, () => ({ result: null })).every((s) => s.reuse === null));
  // Reused rows plus new rows give the same summary shape as a fresh cohort.
  const rows = selected.map((s, i) => s.reuse ?? result(i));
  assert.deepEqual(
    compareRuns(rows, options).map((g) => [g.browser, g.journey, g.invalid.length]),
    [['chrome', 'navigation', 1]],
  );

  assert.throws(
    () => selectSessions(cells, (id) => (id === ids[2] ? { result: result(2), cleanupPending: true } : { result: null })),
    /cleanup unconfirmed for 002/,
  );
  assert.throws(() => selectSessions(cells, (id) => (id === ids[0] ? { result: result(1) } : { result: null })), /different arm/);
  // An orphaned wrapper from a SIGKILLed coordinator may still start this cell's browser.
  assert.throws(() => selectSessions(cells, (id) => (id === ids[6] ? { result: null, owner: 4242 } : { result: null })), /006.*live process 4242/);
});

test('arm builds link ignored local assets instead of copying them, never replacing tracked bytes', () => {
  const tmp = mkdtempSync(join(tmpdir(), 'perf-routes-assets-'));
  try {
    const write = (path, data) => {
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, data);
    };
    const root = join(tmp, 'repo/wt'),
      arm = join(tmp, 'repo/wt/out/builds/baseline'),
      dist = join(arm, 'dist-client');
    write(join(tmp, 'repo/models/store/net.onnx'), 'model-bytes');
    write(join(tmp, 'repo/models/store/old.onnx'), 'old-model');
    write(join(root, 'public/runtimes/r/big.wasm'), 'runtime');
    write(join(root, 'public/tracked.txt'), 'worktree version');
    write(join(root, 'public/engine.wasm'), 'engine');
    mkdirSync(join(root, 'public/models/lc0'), { recursive: true });
    symlinkSync('../../../../models/store/net.onnx', join(root, 'public/models/lc0/net.onnx'));
    write(join(arm, 'public/tracked.txt'), 'historical version');
    mkdirSync(join(arm, 'public/models/maia3'), { recursive: true });
    symlinkSync('missing/target.wasm', join(arm, 'public/engine.wasm')); // dangling archived link
    write(
      join(arm, 'public/models/lc0/manifest.json'),
      JSON.stringify({
        models: [
          { file: 'old.onnx', source: '../models/store/old.onnx' },
          { file: 'net.onnx', source: 'x' },
        ],
      }),
    );
    write(join(arm, 'public/models/maia3/manifest.json'), JSON.stringify({ models: [{ file: 'gone.onnx', source: 'nowhere/gone.onnx' }] }));

    const links = localAssets(root, arm);
    assert.deepEqual([...links.keys()].sort(), ['engine.wasm', 'models/lc0/net.onnx', 'models/lc0/old.onnx', 'runtimes/r/big.wasm']);
    assert.equal(readFileSync(join(arm, 'public/tracked.txt'), 'utf8'), 'historical version');
    assert.equal(existsSync(join(arm, 'public/engine.wasm')), false, 'dangling archived link removed before the build stats public/');
    assert.equal(existsSync(join(arm, 'public/runtimes')), false, 'nothing large is staged for Vite to copy');

    write(join(dist, 'runtimes/r/big.wasm'), 'build output');
    linkAssets(links, dist);
    assert.equal(readFileSync(join(dist, 'runtimes/r/big.wasm'), 'utf8'), 'build output');
    for (const [path, bytes] of [
      ['models/lc0/net.onnx', 'model-bytes'],
      ['models/lc0/old.onnx', 'old-model'],
      ['engine.wasm', 'engine'],
    ]) {
      assert.ok(lstatSync(join(dist, path)).isSymbolicLink(), path);
      assert.equal(readFileSync(join(dist, path), 'utf8'), bytes);
    }
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

test('the isolated static server serves a linked asset like a copied one, under its own origin', async () => {
  const tmp = mkdtempSync(join(tmpdir(), 'perf-routes-serve-'));
  const free = createServer();
  await new Promise((r) => free.listen(0, '127.0.0.1', r));
  const { port } = free.address();
  await new Promise((r) => free.close(r));
  const dist = join(tmp, 'dist');
  mkdirSync(join(dist, 'models/lc0'), { recursive: true });
  writeFileSync(join(tmp, 'net.onnx'), Buffer.alloc(70000, 7));
  writeFileSync(join(dist, 'models/lc0/copy.onnx'), Buffer.alloc(70000, 7));
  symlinkSync(join(tmp, 'net.onnx'), join(dist, 'models/lc0/link.onnx'));
  const server = spawn(process.execPath, [new URL('../scripts/serve_isolated_static.mjs', import.meta.url).pathname, dist, String(port)], {
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  try {
    await once(server.stdout, 'data');
    const fetchHead = async (path) => {
      const r = await fetch(`http://127.0.0.1:${port}${path}`);
      const body = Buffer.from(await r.arrayBuffer());
      const headers = Object.fromEntries(['content-length', 'content-type', 'cache-control', 'cross-origin-resource-policy'].map((h) => [h, r.headers.get(h)]));
      return { status: r.status, headers, body };
    };
    const [copied, linked] = [await fetchHead('/models/lc0/copy.onnx'), await fetchHead('/models/lc0/link.onnx')];
    assert.equal(linked.status, 200);
    assert.deepEqual(linked.headers, copied.headers);
    assert.ok(linked.body.equals(copied.body));
  } finally {
    server.kill();
    await once(server, 'close');
    rmSync(tmp, { recursive: true, force: true });
  }
});

test('one coordinator owns an output directory until it releases, and never removes another owner', () => {
  const dir = mkdtempSync(join(tmpdir(), 'perf-routes-lock-'));
  try {
    const release = lockRun(dir);
    assert.throws(() => lockRun(dir), /Another coordinator .* owns/);
    release();
    const again = lockRun(dir);
    writeFileSync(join(dir, 'coordinator.lock'), 'other-host 1\n');
    again();
    assert.equal(readFileSync(join(dir, 'coordinator.lock'), 'utf8'), 'other-host 1\n');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('coordinator-only failures are persisted as session outcomes; interruptions stay resumable', () => {
  const dir = mkdtempSync(join(tmpdir(), 'perf-routes-attempt-'));
  try {
    const cells = plan(args('--browser', 'chrome', '--journey', 'navigation')).slice(0, 4);
    const file = (i) => join(dir, `${sessionId(cells[i], i)}.json`);
    // 0: the wrapper/session exited without writing a result.
    assert.equal(recordAttempt(cells[0], file(0), { status: 2, cleanupConfirmed: true }).error, 'session process exited 2');
    // 1: the operator stopped the run before the session wrote anything.
    recordAttempt(cells[1], file(1), { status: null, cleanupConfirmed: true, interruption: 'Interrupted by SIGINT' });
    // 2: a measured session whose browser cleanup the coordinator could not confirm.
    writeFileSync(file(2), JSON.stringify({ ...cells[2], status: 'ok', metrics: { jsBytes: 1 } }));
    assert.match(recordAttempt(cells[2], file(2), { status: 0, cleanupConfirmed: false, lockDir: '/m' }).error, /cleanup unconfirmed.*mutex retained at \/m/);
    // 3: a normal result is returned untouched.
    writeFileSync(file(3), JSON.stringify({ ...cells[3], status: 'ok', metrics: { jsBytes: 2 } }));
    assert.equal(recordAttempt(cells[3], file(3), { status: 0, cleanupConfirmed: true }).status, 'ok');

    const selected = selectSessions(cells, (id) => ({ result: readJson(join(dir, `${id}.json`)) }));
    assert.deepEqual(
      selected.map((s) => s.reuse?.status ?? 'rerun'),
      ['invalid', 'rerun', 'invalid', 'ok'],
    );
    assert.equal(selected[0].reuse.error, 'session process exited 2');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('build records are atomic and any unreadable or foreign record is a rebuild, not a crash', () => {
  const dir = mkdtempSync(join(tmpdir(), 'perf-routes-build-'));
  try {
    const dist = join(dir, 'dist-client');
    mkdirSync(join(dist, 'app/analysis'), { recursive: true });
    writeFileSync(join(dist, 'app/analysis/index.html'), '');
    const record = join(dir, 'baseline.json');
    writeJsonAtomic(record, { dist, commit: 'a' });
    assert.equal(existsSync(`${record}.tmp`), false);
    assert.equal(reusableBuild(record, 'a').dist, dist);
    assert.equal(reusableBuild(record, 'b'), null);
    writeFileSync(record, '{"dist": "/tru');
    assert.equal(reusableBuild(record, 'a'), null);
    writeJsonAtomic(record, { dist: join(dir, 'gone'), commit: 'a' });
    assert.equal(reusableBuild(record, 'a'), null);
    assert.equal(reusableBuild(join(dir, 'missing.json'), 'a'), null);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('harness identity covers every measuring dependency, including move choice and memory sampling', () => {
  const root = new URL('..', import.meta.url).pathname;
  const files = measuringFiles(root, 'scripts/perf_routes/session.mjs', [
    'scripts/perf_matrix/pidfp.c',
    'scripts/serve_isolated_static.mjs',
    'scripts/nope.mjs',
  ]);
  for (const file of [
    'scripts/perf_routes/session.mjs',
    'scripts/perf_routes/journeys.mjs',
    'scripts/perf_routes/human.mjs',
    'scripts/perf_routes/gate.mjs',
    'scripts/perf_routes/lifecycle.mjs',
    'scripts/perf_routes/probe.js',
    'scripts/perf_matrix/common.mjs',
    'scripts/perf_matrix/proxy.mjs',
    'scripts/perf_matrix/pidfp.c',
    'scripts/lib/perf_canary_stats.mjs',
    'scripts/serve_isolated_static.mjs',
    'src/chess/board.ts',
    'src/chess/moveCodec.ts',
    'src/chess/movegen.ts',
    'src/chess/san.ts',
  ])
    assert.ok(files.includes(file), file);
  for (const file of ['scripts/perf_routes/run.mjs', 'scripts/perf_routes/resume.mjs', 'scripts/nope.mjs']) assert.equal(files.includes(file), false, file);
});

test('a session wrapper records its PID before waiting and clears it on exit; only live owners count', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'perf-routes-owner-'));
  try {
    const resultFile = join(dir, '000.json');
    const signal = new AbortController().signal;
    const running = runOwnedSession(process.execPath, ['-e', 'setTimeout(() => {}, 300)'], signal, null, resultFile);
    const pid = Number(readFileSync(ownerRecord(resultFile), 'utf8'));
    assert.equal(liveOwner(resultFile), pid);
    assert.equal(await running, 0);
    assert.equal(existsSync(ownerRecord(resultFile)), false);
    assert.equal(liveOwner(resultFile), null);
    // A record left by a SIGKILLed coordinator whose wrapper has since exited is not an owner.
    const exited = spawn(process.execPath, ['-e', '']);
    await once(exited, 'close');
    writeFileSync(ownerRecord(resultFile), `${exited.pid}\n`);
    assert.equal(liveOwner(resultFile), null);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('a failed owner record stops the just-spawned session before the caller can release the mutex', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'perf-routes-owner-fail-'));
  const pidFile = join(dir, 'child.pid');
  try {
    const child = `require('node:fs').writeFileSync(${JSON.stringify(pidFile)}, String(process.pid)); setTimeout(() => {}, 30000)`;
    // The sessions directory is missing, so the owner record cannot be written.
    await assert.rejects(runOwnedSession(process.execPath, ['-e', child], new AbortController().signal, null, join(dir, 'missing/000.json')), /ENOENT/);
    // A child left running would have written its PID by now.
    await new Promise((r) => setTimeout(r, 500));
    if (existsSync(pidFile)) {
      const pid = Number(readFileSync(pidFile, 'utf8'));
      assert.throws(() => process.kill(pid, 0), { code: 'ESRCH' }, 'the unrecorded child must be stopped');
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('the candidate shares the baseline build by resolved commit, and a resume keeps the recorded mapping', () => {
  const sha = (commit) => ({ kind: 'archived-source', commit });
  assert.equal(sharesBaselineBuild({ baseline: sha('a'), candidate: sha('a') }), true, 'main HEAD resolving to one SHA');
  assert.equal(sharesBaselineBuild({ baseline: sha('a'), candidate: sha('b') }), false);
  const prebuilt = { kind: 'prebuilt', dist: '/x' };
  assert.equal(sharesBaselineBuild({ baseline: prebuilt, candidate: prebuilt }), false, 'prebuilt arms are served as given');
  assert.equal(sharesBaselineBuild({ baseline: { kind: 'externally-served' }, candidate: { kind: 'externally-served' } }, null), false);
  // A run that built both arms separately keeps doing so when resumed by SHA, and vice versa.
  const separate = { baseline: { dist: '/o/builds/baseline/dist-client' }, candidate: { dist: '/o/builds/candidate/dist-client' } };
  assert.equal(sharesBaselineBuild({ baseline: sha('a'), candidate: sha('a') }, separate), false);
  const shared = { baseline: { dist: '/o/builds/baseline/dist-client' }, candidate: { dist: '/o/builds/baseline/dist-client' } };
  assert.equal(sharesBaselineBuild({ baseline: sha('a'), candidate: sha('a') }, shared), true);
  // builds.json from a run interrupted before both arms were recorded: fall back to the commit rule.
  assert.equal(sharesBaselineBuild({ baseline: sha('a'), candidate: sha('a') }, { baseline: separate.baseline }), true);
});

test('a prebuilt dist is identified by its paths, sizes, mtimes and link targets, not its path alone', () => {
  const dist = mkdtempSync(join(tmpdir(), 'perf-routes-dist-'));
  try {
    mkdirSync(join(dist, '_app/immutable'), { recursive: true });
    writeFileSync(join(dist, '_app/immutable/entry.abc.js'), 'x');
    symlinkSync('/models/a.onnx', join(dist, 'model.onnx'));
    const first = buildFingerprint(dist);
    assert.equal(buildFingerprint(dist), first);
    utimesSync(join(dist, '_app/immutable/entry.abc.js'), new Date(1000), new Date(1000));
    const touched = buildFingerprint(dist);
    assert.notEqual(touched, first, 'a rebuild in place rewrites files');
    writeFileSync(join(dist, '_app/immutable/entry.abc.js'), 'xy');
    utimesSync(join(dist, '_app/immutable/entry.abc.js'), new Date(1000), new Date(1000));
    const resized = buildFingerprint(dist);
    assert.notEqual(resized, touched, 'size');
    rmSync(join(dist, 'model.onnx'));
    symlinkSync('/models/b.onnx', join(dist, 'model.onnx'));
    const relinked = buildFingerprint(dist);
    assert.notEqual(relinked, resized, 'link target');
    writeFileSync(join(dist, 'new.js'), '');
    assert.notEqual(buildFingerprint(dist), relinked, 'added file');
  } finally {
    rmSync(dist, { recursive: true, force: true });
  }
});

test('builds stage per attempt and rename into place; session attempts clear stale results before each try', () => {
  const source = readFileSync(new URL('../scripts/perf_routes/run.mjs', import.meta.url), 'utf8');
  const build = source.slice(source.indexOf('function build('), source.indexOf('async function serve('));
  assert.match(build, /const stage = `\$\{dir\}\.staging-\$\{process\.pid\}-\$\{Date\.now\(\)\}`/);
  assert.ok(!/command\([^)]*\bdir\)/.test(build) && !/'-C', dir\]/.test(build), 'tar/npm never write into the final arm directory');
  assert.ok(build.indexOf('renameSync(stage, dir)') > build.indexOf("npm', 'run', 'build:client'"), 'renamed only after the build');
  assert.ok(build.indexOf('writeJsonAtomic(record') > build.indexOf('renameSync(stage, dir)'), 'recorded only once in place');
  const loop = source.slice(source.indexOf('      do {'), source.indexOf('} while (cleanupConfirmed && status === 75'));
  assert.match(loop, /rmSync\(cancellationMarker\(resultFile\), \{ force: true \}\);\s*rmSync\(resultFile, \{ force: true \}\);[\s\S]*acquireLock/);
  assert.match(source, /arm === 'candidate' && shared \? \{ \.\.\.builds\.baseline \}/);
});
