// Pure route-gate planning/statistics. Never round before making a verdict.
import { mad, median } from '../lib/perf_canary_stats.mjs';
import { buildOrigin } from '../perf_matrix/proxy.mjs';

export const JOURNEYS = ['analysis-default', 'analysis-four', 'play-maia3', 'play-lc0', 'arena-cpu', 'arena-lc0', 'navigation'];
export const HELP = `npm run perf:routes -- --baseline <built-dir|commit> --candidate <built-dir|commit>
  [--browser chrome|safari|all] [--repeats 5] [--out DIR]
  --benchlock PATH              machine-wide laptop timing mutex
  --lock-dir PATH               shared mini mutex (instead of --benchlock)
  --driver URL                  existing Safari driver; never started/stopped
  --baseline-url URL --candidate-url URL  use already served build origins (mini transport)
  --journey NAME                repeatable subset (not a full merge gate)
  --repeats 1                   same-build smoke only; cannot pass the merge gate
  --report DIR                  regenerate summary from runs.jsonl
  --build-only                  prepare archived product builds, no measurements
  --dry-run                     validate/print plan without building or writing
Full gate: seven journeys, >=5 interleaved repeats, 60s analysis, 20s settle.
Exit: 0 clean, 1 regressed, 2 incomplete/noisy/invalid. No push or deployment.`;

export function parseArgs(argv) {
  const options = { browser: 'all', repeats: 5, journeys: [], driver: 'http://127.0.0.1:4445' };
  const values = new Set([
    'baseline',
    'candidate',
    'browser',
    'repeats',
    'out',
    'benchlock',
    'lock-dir',
    'driver',
    'baseline-url',
    'candidate-url',
    'journey',
    'report',
  ]);
  const flags = new Set(['dry-run', 'build-only', 'help']);
  const seen = new Set();
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i].replace(/^--/, '');
    if (!argv[i].startsWith('--') || (!values.has(key) && !flags.has(key))) throw Error(`Unknown argument: ${argv[i]}`);
    if (seen.has(key) && key !== 'journey') throw Error(`Duplicate --${key}`);
    seen.add(key);
    if (flags.has(key)) {
      options[key] = true;
      continue;
    }
    const value = argv[++i];
    if (!value || value.startsWith('--')) throw Error(`Missing value for --${key}`);
    if (key === 'journey') options.journeys.push(value);
    else {
      options[key] = value;
    }
  }
  options.repeats = Number(options.repeats);
  const selfComparison = options.baseline === options.candidate && options['baseline-url'] === options['candidate-url'];
  if (!Number.isInteger(options.repeats) || (options.repeats < 5 && !(options.repeats === 1 && selfComparison)))
    throw Error('--repeats must be an integer >=5 (or 1 for a same-build smoke)');
  if (!['chrome', 'safari', 'all'].includes(options.browser)) throw Error('Invalid --browser');
  if (options.benchlock && options['lock-dir']) throw Error('Use --benchlock OR --lock-dir');
  if (options.journeys.some((j) => !JOURNEYS.includes(j))) throw Error('Unknown --journey');
  options.journeys = options.journeys.length ? [...new Set(options.journeys)] : [...JOURNEYS];
  const driver = new URL(options.driver);
  if (!['http:', 'https:'].includes(driver.protocol) || driver.username || driver.password) throw Error('Invalid --driver');
  for (const key of ['baseline-url', 'candidate-url']) if (options[key]) buildOrigin(options[key]);
  if (!!options['baseline-url'] !== !!options['candidate-url']) throw Error('Both served build URLs are required');
  if (!options.help && !options.report && (!options.baseline || !options.candidate)) throw Error('--baseline and --candidate are required');
  if (!options.help && !options.report && !options['dry-run'] && !options['build-only'] && !options.benchlock && !options['lock-dir'])
    throw Error('A machine-wide --benchlock or --lock-dir is required');
  return options;
}

export function plan(options) {
  const browsers = options.browser === 'all' ? ['safari', 'chrome'] : [options.browser];
  return Array.from({ length: options.repeats }, (_, repeat) =>
    browsers.flatMap((browser) =>
      options.journeys.flatMap((journey) =>
        (repeat % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate']).map((arm) => ({ browser, journey, arm, repeat })),
      ),
    ),
  ).flat();
}

export function stats(values) {
  if (!values.length) return { n: 0, median: null, mad: null, madRatio: null, min: null, max: null };
  const center = median(values),
    spread = mad(values);
  return {
    n: values.length,
    median: center,
    mad: spread,
    madRatio: center === 0 ? (spread === 0 ? 0 : Infinity) : spread / Math.abs(center),
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

export function metricPolicy(name) {
  return { floor: /longTask/.test(name) ? 0.1 : 0.05, higherIsBetter: name.startsWith('speed.') };
}

export function verdict(name, baseline, candidate, expected = 5) {
  const a = stats(baseline),
    b = stats(candidate),
    { floor, higherIsBetter } = metricPolicy(name);
  const threshold = Math.max(floor, 3 * (a.madRatio ?? 0), 3 * (b.madRatio ?? 0));
  // With zero baseline, an increase is a real absolute change, not a division by zero or an automatic pass.
  const deltaRatio = a.median === 0 ? (b.median === 0 ? 0 : null) : (b.median - a.median) / Math.abs(a.median);
  let status = 'within-noise';
  if (!a.n || !b.n) status = 'unavailable';
  else if (a.n < expected || b.n < expected) status = 'insufficient';
  else if (a.madRatio > floor || b.madRatio > floor) status = 'noisy';
  else if (a.median === 0 && b.median !== 0) status = higherIsBetter ? 'improved' : 'regressed';
  else if (Math.abs(deltaRatio) > threshold + 1e-12) status = (higherIsBetter ? deltaRatio > 0 : deltaRatio < 0) ? 'improved' : 'regressed';
  return { name, baseline: a, candidate: b, deltaRatio, threshold, status };
}

export const naturalVisibilityValid = (events, finalState) =>
  finalState === 'visible' && events.length > 0 && events.every((event) => event.state === 'visible');

export const previewEnvironment = (port, env = process.env) => ({ ...env, PORT: String(port) });

export const isJsTransfer = (r) => r.method !== 'HEAD' && /\.m?js(?:\?|$)/.test(r.path);

export function jsBytesByRoute(requests, marks) {
  const bytes = Object.fromEntries(marks.map((mark) => [mark.route, 0]));
  for (const r of requests.filter(isJsTransfer)) {
    const route = marks.findLast((mark) => mark.at <= r.at)?.route ?? marks[0].route;
    bytes[route] += r.bytes;
  }
  return bytes;
}

export function safariWebContentPid(before, after) {
  const fresh = after.filter((pid) => !before.has(pid));
  if (fresh.length !== 1) throw Error(`Require exactly one fresh Safari WebContent PID; found ${fresh.length}`);
  return fresh[0];
}

const validFootprint = (row) => row && [row.cur, row.life, row.interval].every((n) => Number.isFinite(n) && n > 0);

export function footprintSum(pids, readings, label) {
  if (!pids.length) throw Error(`No ${label} PID`);
  return pids.reduce(
    (sum, pid) => {
      const row = readings.get(pid);
      if (!validFootprint(row)) throw Error(`Missing/invalid ${label} footprint for PID ${pid}`);
      return { cur: sum.cur + row.cur, life: sum.life + row.life, interval: sum.interval + row.interval };
    },
    { cur: 0, life: 0, interval: 0 },
  );
}

export async function chromeFootprint(discover, read) {
  const first = await discover();
  const initial = read([...first.rendererPids, ...first.gpuPids], { reset: true });
  const gpu = footprintSum(first.gpuPids, initial, 'Chrome GPU');
  const unreadable = first.rendererPids.filter((pid) => !validFootprint(initial.get(pid)));
  if (!unreadable.length) return { render: footprintSum(first.rendererPids, initial, 'renderer'), gpu };

  const current = await discover();
  if (unreadable.some((pid) => current.rendererPids.includes(pid))) footprintSum(first.rendererPids, initial, 'renderer');
  const refreshed = read([...current.rendererPids, ...current.gpuPids], { reset: true });
  // The first read reset live counters. Carry those high-water marks into the retried sample.
  for (const [pid, row] of refreshed) {
    const previous = initial.get(pid);
    if (validFootprint(row) && validFootprint(previous)) refreshed.set(pid, { ...row, interval: Math.max(row.interval, previous.interval) });
  }
  const retryRender = footprintSum(current.rendererPids, refreshed, 'renderer');
  const retryGpu = footprintSum(current.gpuPids, refreshed, 'Chrome GPU');
  const observedRendererInterval = first.rendererPids.reduce((sum, pid) => {
    const row = initial.get(pid);
    return validFootprint(row) ? sum + row.interval : sum;
  }, 0);
  retryRender.interval = Math.max(retryRender.interval, observedRendererInterval);
  retryGpu.interval = Math.max(retryGpu.interval, gpu.interval);
  return { render: retryRender, gpu: retryGpu };
}

const cpuProbeFamily = (r) =>
  r.method === 'HEAD' ? /^\/(stockfish|berserk|plentychess|stormphrax|viridithas)\/[^/?]+\.(?:m?js|wasm|data|nnue|nn)(?:\?|$)/.exec(r.path)?.[1] : undefined;
const absent = (r) => r.complete && !r.failure && [404, 410].includes(r.status);
const reckless = (r) => ['GET', 'HEAD'].includes(r.method) && r.path.startsWith('/reckless/');

export const incompleteJsTransfers = (requests) => requests.filter((r) => isJsTransfer(r) && !r.complete && !reckless(r));

export function requestFailures(requestLog, journey, { routeMarks = [], cutoff } = {}) {
  const requests = requestLog.slice(0, cutoff?.requestCount);
  const optional = [],
    teardownAborts = [],
    unexpected = [];
  const missingRuntimeProbes = requests.filter((r) => absent(r) && cpuProbeFamily(r));
  for (const r of requests) {
    if (r.complete && !r.failure && r.status < 400) continue;
    const family = cpuProbeFamily(r);
    const probe =
      family ||
      (r.method === 'HEAD' &&
        /^\/(?:models\/lc0\/(?:(?:BT4-|t3-)[^/?]+|lqo_v2\.f16\.qdq8)\.onnx|engine-logos\/[^/?]+\.(?:svg|png|jpe?g))(?:\?|$)/i.test(r.path));
    if ((probe || reckless(r)) && absent(r)) {
      optional.push(r);
      continue;
    }
    if (r.clientAborted && (r.status === null || r.status < 400)) {
      if (r.endedAt >= cutoff?.at || routeMarks.some((m) => r.at < m.at && r.endedAt >= m.at && r.endedAt <= m.readyAt)) {
        teardownAborts.push(r);
        continue;
      }
      const siblingCancelled =
        family && missingRuntimeProbes.some((p) => cpuProbeFamily(p) === family && p.at <= r.endedAt && r.at <= p.endedAt && p.endedAt <= r.endedAt);
      const temporaryViridithas = journey === 'analysis-four' && r.method === 'GET' && /^\/viridithas\/[^/?]+\.wasm(?:\?|$)/.test(r.path);
      const unselectedMaia3 = journey === 'play-lc0' && r.method === 'GET' && r.path === '/models/maia3/maia3_simplified.qdq8.graph.onnx';
      // Analysis Four can cancel redundant preparation for this sidecar after a full copy was fetched.
      const redundantBerserk =
        journey === 'analysis-four' &&
        r.method === 'GET' &&
        r.path === '/berserk/berserk-emscripten.data' &&
        requests.some((p) => p.method === 'GET' && p.path === r.path && p.complete && !p.failure && p.status === 200 && p.endedAt <= r.at);
      if (siblingCancelled || reckless(r) || temporaryViridithas || unselectedMaia3 || redundantBerserk) {
        optional.push(r);
        continue;
      }
    }
    unexpected.push(r);
  }
  return { optional, teardownAborts, unexpected };
}

export function peakFootprint(browser, samples, navigationPeak = 0) {
  if (!samples.length) throw Error('No footprint samples');
  return browser === 'safari'
    ? Math.max(navigationPeak, ...samples.map((s) => s.rendererIntervalMb))
    : Math.max(...samples.map((s) => s.rendererMb + (s.gpuMb ?? 0)));
}

export const REQUIRED = [
  'memory.peakMb',
  'memory.settledMb',
  'memory.residualMb',
  'workers.peak',
  'workers.residual',
  'boardInteractiveMs',
  'firstLineMs',
  'jsBytes',
];
export function compareRuns(runs, options) {
  const groups = [...new Set(plan(options).map((r) => `${r.browser}/${r.journey}`))];
  return groups.map((key) => {
    const [browser, journey] = key.split('/');
    const rows = runs.filter((r) => r.browser === browser && r.journey === journey);
    const valid = rows.filter((r) => r.status === 'ok');
    const engines =
      journey === 'analysis-four'
        ? ['lc0-visits', 'stockfish-nps', 'berserk-nps', 'plentychess-nps']
        : journey === 'arena-cpu'
          ? ['stockfish-nps', 'berserk-nps']
          : journey === 'analysis-default' || journey === 'arena-lc0'
            ? ['lc0-visits', 'stockfish-nps']
            : journey === 'play-maia3'
              ? ['maia3-evals']
              : journey === 'play-lc0'
                ? ['lc0-visits']
                : [];
    const required = [
      ...REQUIRED,
      ...engines.map((e) => `speed.${e}`),
      ...(journey.startsWith('analysis') || journey.startsWith('play') ? ['inputLatencyMs'] : []),
      ...(browser === 'chrome' && journey.startsWith('analysis') ? ['longTaskCount', 'longTaskTotalMs', 'longTaskP95Ms'] : []),
      ...(browser === 'chrome' ? ['memory.gpuPeakMb'] : []),
    ];
    const names = [...new Set([...required, ...rows.flatMap((r) => Object.keys(r.metrics ?? {}))])];
    const metrics = names.map((name) =>
      verdict(
        name,
        ...['baseline', 'candidate'].map((arm) =>
          valid
            .filter((r) => r.arm === arm)
            .map((r) => r.metrics[name])
            .filter(Number.isFinite),
        ),
        Math.max(5, options.repeats),
      ),
    );
    const missing = metrics.filter((m) => required.includes(m.name) && m.status === 'unavailable');
    const incomplete = rows.length !== options.repeats * 2 || rows.some((r) => r.status !== 'ok') || missing.length > 0;
    return {
      browser,
      journey,
      invalid: rows.filter((r) => r.status !== 'ok').map((r) => ({ arm: r.arm, repeat: r.repeat, error: r.error })),
      metrics,
      status: metrics.some((m) => m.status === 'regressed')
        ? 'regressed'
        : incomplete || metrics.some((m) => ['noisy', 'insufficient'].includes(m.status))
          ? 'incomplete'
          : 'clean',
    };
  });
}
