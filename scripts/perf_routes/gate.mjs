// Pure route-gate planning/statistics. Never round before making a verdict.
import { mad, median } from '../lib/perf_canary_stats.mjs';

export const JOURNEYS = ['analysis-default', 'analysis-four', 'play-maia3', 'play-lc0', 'arena-cpu', 'arena-lc0', 'navigation'];
export const HELP = `npm run perf:routes -- --baseline <built-dir|commit> --candidate <built-dir|commit>
  [--browser chrome|safari|all] [--repeats 5] [--out DIR]
  --benchlock PATH              machine-wide laptop timing mutex
  --lock-dir PATH               shared mini mutex (instead of --benchlock)
  --driver URL                  existing Safari driver; never started/stopped
  --baseline-url URL --candidate-url URL  use already served builds (mini transport)
  --journey NAME                repeatable subset (not a full merge gate)
  --report DIR                  regenerate summary from raw sessions
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
  if (!Number.isInteger(options.repeats) || options.repeats < 5) throw Error('--repeats must be an integer >=5');
  if (!['chrome', 'safari', 'all'].includes(options.browser)) throw Error('Invalid --browser');
  if (options.benchlock && options['lock-dir']) throw Error('Use --benchlock OR --lock-dir');
  if (options.journeys.some((j) => !JOURNEYS.includes(j))) throw Error('Unknown --journey');
  options.journeys = options.journeys.length ? [...new Set(options.journeys)] : [...JOURNEYS];
  for (const key of ['driver', 'baseline-url', 'candidate-url'])
    if (options[key]) {
      const u = new URL(options[key]);
      if (!['http:', 'https:'].includes(u.protocol) || u.username || u.password) throw Error(`Invalid --${key}`);
    }
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
        options.repeats,
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
