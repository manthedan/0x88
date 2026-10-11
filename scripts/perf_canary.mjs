#!/usr/bin/env node
// Local performance canary. Not a CI gate: hardware noise is too wide to fail
// a pull request. One CPU fixed suite (Node WASM via emscripten_uci_bench) and
// an optional browser model-latency fixture. Repeats are reduced to a median
// and MAD and compared with the previous JSON in the results directory.
import { spawnSync } from 'node:child_process';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseScriptArgs } from './lib/cli.mjs';
import { compareMetric, mad, median } from './lib/perf_canary_stats.mjs';

const DEFAULT_OUT = '.local-dev-artifacts/perf-canary';
const DEFAULT_JS = 'public/berserk/berserk-emscripten.js';
const DEFAULT_DEPTH = 8;
const DEFAULT_POSITIONS = 20;
const DEFAULT_REPEATS = 3;
const DEFAULT_HASH_MB = 16;
const BENCH_TIMEOUT_MS = 10 * 60 * 1000;

const USAGE = `Usage: node scripts/perf_canary.mjs [options]

Run the CPU fixed-suite canary (and, with --browser, one browser model-latency fixture).
Write JSON under .local-dev-artifacts/perf-canary and diff it against the previous run.
This is a canary, not a CI gate. Time it under the repo benchlock.

Options:
  --js PATH            Emscripten UCI artifact (default ${DEFAULT_JS})
  --depth N            Fixed depth (default ${DEFAULT_DEPTH})
  --positions N        Rotated-FEN prefix length, max 20 (default ${DEFAULT_POSITIONS})
  --hash N             Hash MB (default ${DEFAULT_HASH_MB})
  --repeats N          Within-run repeats for median and MAD (default ${DEFAULT_REPEATS})
  --out DIR            Results directory (default ${DEFAULT_OUT})
  --browser            Also run the hybrid input-path browser fixture
  --browser-port N     Port for that fixture's dev server (default 5188)
  --strict             Exit 1 when a comparison status is regressed
  --dry-run            Print the plan as JSON and do not measure
  -h, --help           Show this help
`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      js: { type: 'string', default: DEFAULT_JS },
      depth: { type: 'string', default: String(DEFAULT_DEPTH) },
      positions: { type: 'string', default: String(DEFAULT_POSITIONS) },
      hash: { type: 'string', default: String(DEFAULT_HASH_MB) },
      repeats: { type: 'string', default: String(DEFAULT_REPEATS) },
      out: { type: 'string', default: DEFAULT_OUT },
      'browser-port': { type: 'string', default: '5188' },
      browser: { type: 'boolean', default: false },
      strict: { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.depth = Number(args.depth);
  args.positions = Number(args.positions);
  args.hash = Number(args.hash);
  args.repeats = Number(args.repeats);
  args.browserPort = Number(args.browserPort);
  for (const [name, value] of [
    ['depth', args.depth],
    ['positions', args.positions],
    ['hash', args.hash],
    ['repeats', args.repeats],
    ['browser-port', args.browserPort],
  ]) {
    if (!Number.isInteger(value) || value < 1) throw new Error(`--${name} must be a positive integer`);
  }
  if (args.positions > 20) throw new Error('--positions cannot exceed the 20-position rotated FEN suite');
  return args;
}

function gitText(args) {
  const result = spawnSync('git', args, { encoding: 'utf8' });
  if (result.status !== 0) return null;
  return result.stdout.trim();
}

function cpuCommand(args) {
  return [
    'scripts/emscripten_uci_bench.mjs',
    '--js',
    args.js,
    '--depths',
    String(args.depth),
    '--positions',
    String(args.positions),
    '--hash',
    String(args.hash),
  ];
}

function browserCommand(args) {
  return [
    '--experimental-strip-types',
    'scripts/lc0_browser_hybrid_input_bench.mjs',
    '--backends',
    'wasm',
    '--iters',
    '1',
    '--warmup',
    '1',
    '--head-backend',
    'ort',
    '--port',
    String(args.browserPort),
    '--host',
    '127.0.0.1',
  ];
}

function readSummary(report, jsPath, depth) {
  const summary = report?.summary?.[jsPath]?.[`depth${depth}`];
  if (!summary) throw new Error(`Bench report has no summary for ${jsPath} depth ${depth}`);
  for (const key of ['npsMedian', 'engineTimeMsMedian', 'nodesTotal']) {
    if (!Number.isFinite(summary[key])) throw new Error(`Bench summary ${key} is not finite`);
  }
  return summary;
}

function runCpuRepeat(args) {
  const result = spawnSync(process.execPath, cpuCommand(args), { encoding: 'utf8', timeout: BENCH_TIMEOUT_MS });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(result.stderr?.trim() || result.stdout?.trim() || `CPU bench exited ${result.status}`);
  }
  let report;
  try {
    report = JSON.parse(result.stdout);
  } catch {
    throw new Error(`CPU bench did not print JSON: ${result.stdout?.slice(0, 400)}`);
  }
  const summary = readSummary(report, args.js, args.depth);
  return {
    npsMedian: summary.npsMedian,
    engineTimeMsMedian: summary.engineTimeMsMedian,
    nodesTotal: summary.nodesTotal,
    positions: report.positions ?? args.positions,
  };
}

function runBrowser(args) {
  const result = spawnSync(process.execPath, browserCommand(args), { encoding: 'utf8', timeout: BENCH_TIMEOUT_MS });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    return { status: 'failed', error: (result.stderr || result.stdout || `browser fixture exited ${result.status}`).slice(-2000) };
  }
  return { status: 'ran', stdoutTail: result.stdout.slice(-2000) };
}

export const CANARY_FILE = /^canary-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z\.json$/;

export function previousCanary(outDir) {
  let names;
  try {
    names = readdirSync(outDir);
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
  // Filenames are UTC stamps from stamp() (`canary-2026-10-11T08-59-33-383Z.json`).
  // Lexicographic order is chronological. A looser prefix would let
  // `canary-2026-notes.json` sort after a real stamp and become the baseline.
  const candidates = names.filter((name) => CANARY_FILE.test(name)).sort();
  if (!candidates.length) return null;
  const name = candidates.at(-1);
  return { name, report: JSON.parse(readFileSync(path.join(outDir, name), 'utf8')) };
}

export function baselineKey(report) {
  return {
    arch: report?.environment?.host?.arch ?? null,
    cpuModel: report?.environment?.host?.cpuModel ?? null,
    artifact: report?.runtimeConfiguration?.modelArtifact?.path ?? null,
    hashMb: report?.runtimeConfiguration?.hashMb ?? null,
    depth: report?.benchmarkProtocol?.depth ?? null,
    positionCount: report?.benchmarkProtocol?.positionCount ?? null,
    suiteId: report?.benchmarkProtocol?.suiteId ?? null,
  };
}

export function sameBaseline(previousReport, current) {
  const prior = baselineKey(previousReport);
  return (
    prior.arch === current.arch &&
    prior.cpuModel === current.cpuModel &&
    prior.artifact === current.artifact &&
    prior.hashMb === current.hashMb &&
    prior.depth === current.depth &&
    prior.positionCount === current.positionCount &&
    prior.suiteId === current.suiteId &&
    prior.suiteId !== null
  );
}

function metricSamples(repeats, key) {
  return repeats.map((repeat) => repeat[key]);
}

function buildComparison(currentRepeats, previous) {
  if (!previous?.report?.cpu?.aggregate) return null;
  const prior = previous.report.cpu.aggregate;
  const metrics = [
    compareMetric({ name: 'npsMedian', previous: prior.npsMedian.median, samples: metricSamples(currentRepeats, 'npsMedian'), higherIsBetter: true }),
    compareMetric({
      name: 'engineTimeMsMedian',
      previous: prior.engineTimeMsMedian.median,
      samples: metricSamples(currentRepeats, 'engineTimeMsMedian'),
      higherIsBetter: false,
    }),
  ];
  return { previousFile: previous.name, previousCreatedAt: previous.report.createdAt ?? null, metrics };
}

function aggregate(repeats, key) {
  const samples = metricSamples(repeats, key);
  return { median: median(samples), mad: mad(samples), samples };
}

function stamp() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const plan = {
    cpu: [process.execPath, ...cpuCommand(args)],
    repeats: args.repeats,
    browser: args.browser ? [process.execPath, ...browserCommand(args)] : null,
    out: args.out,
  };
  if (args.dryRun) {
    console.log(JSON.stringify(plan, null, 2));
    return;
  }
  const repeats = [];
  for (let index = 0; index < args.repeats; index += 1) repeats.push(runCpuRepeat(args));
  const nodes = new Set(repeats.map((repeat) => repeat.nodesTotal));
  const browser = args.browser
    ? runBrowser(args)
    : { status: 'skipped', reason: 'Browser model-latency fixture not requested. Pass --browser to run scripts/lc0_browser_hybrid_input_bench.mjs.' };
  const found = previousCanary(args.out);
  const currentKey = {
    arch: os.arch(),
    cpuModel: os.cpus()[0]?.model ?? null,
    artifact: args.js,
    hashMb: args.hash,
    depth: args.depth,
    positionCount: args.positions,
    suiteId: 'rotated-fen-20',
  };
  const compatible = found && sameBaseline(found.report, currentKey);
  const previous = compatible ? found : null;
  const skippedBaseline = found && !compatible ? found.name : null;
  const comparison = buildComparison(repeats, previous);
  const cpus = os.cpus();
  const report = {
    schema: 'browser-engine.benchmark-artifact.v1',
    artifactKind: 'perf-canary',
    createdAt: new Date().toISOString(),
    git: { repo: '0x88', sha: gitText(['rev-parse', 'HEAD']), dirty: Boolean(gitText(['status', '--porcelain'])) },
    runtimeConfiguration: {
      schema: 'browser-engine.runtime-configuration.v1',
      engineFamily: 'berserk',
      modelId: 'berserk-emscripten-scalar',
      modelArtifact: { kind: 'wasm-engine', path: args.js },
      ruleset: { id: 'standard-chess' },
      runtimeBackend: 'wasm-engine',
      runtimeConfigId: 'perf-canary-berserk-scalar',
      threads: 1,
      hashMb: args.hash,
    },
    benchmarkProtocol: {
      name: 'fixed-suite-search-throughput',
      suiteId: 'rotated-fen-20',
      positionCount: args.positions,
      depth: args.depth,
      repeats: args.repeats,
      harness: 'scripts/emscripten_uci_bench.mjs',
      comparison: 'previous canary on the same host, artifact, depth, positions, and hash; threshold max(10%, 3*MAD/median)',
      ciGate: false,
    },
    environment: {
      host: {
        os: `${os.platform()} ${os.release()}`,
        arch: os.arch(),
        cpuModel: cpus[0]?.model ?? null,
        cpuCount: cpus.length,
        node: process.version,
      },
    },
    cpu: {
      repeats,
      deterministicNodes: nodes.size === 1,
      aggregate: {
        npsMedian: aggregate(repeats, 'npsMedian'),
        engineTimeMsMedian: aggregate(repeats, 'engineTimeMsMedian'),
      },
    },
    browser,
    comparison,
    skippedBaseline,
    promotionAssessment: {
      role: 'canary',
      ciGate: false,
      note: 'Not a promotion gate. A noisy-run or a same-machine spread above about 10% means the host is too loud for this instrument.',
    },
  };
  mkdirSync(args.out, { recursive: true });
  const fileName = `canary-${stamp()}.json`;
  const filePath = path.join(args.out, fileName);
  writeFileSync(filePath, `${JSON.stringify(report, null, 2)}\n`);
  writeFileSync(path.join(args.out, 'latest.json'), `${JSON.stringify(report, null, 2)}\n`);
  printSummary(report, filePath);
  if (args.browser && browser.status === 'failed') process.exitCode = 1;
  if (args.strict && comparison?.metrics.some((metric) => metric.status === 'regressed')) process.exitCode = 1;
}

function formatRatio(value) {
  if (value === null || value === undefined) return 'n/a';
  return `${(value * 100).toFixed(1)}%`;
}

function printSummary(report, filePath) {
  const nps = report.cpu.aggregate.npsMedian;
  const time = report.cpu.aggregate.engineTimeMsMedian;
  console.log(
    `CPU canary ${report.runtimeConfiguration.modelArtifact.path} depth ${report.benchmarkProtocol.depth} x ${report.benchmarkProtocol.positionCount}, repeats ${report.benchmarkProtocol.repeats}`,
  );
  console.log(`nps median ${Math.round(nps.median)} MAD ${Math.round(nps.mad)} (${formatRatio(nps.mad / Math.max(nps.median, 1e-9))})`);
  console.log(`engine time ms median ${time.median.toFixed(1)} MAD ${time.mad.toFixed(1)} (${formatRatio(time.mad / Math.max(time.median, 1e-9))})`);
  console.log(report.cpu.deterministicNodes ? 'nodes identical across repeats' : 'nodes differed across repeats');
  if (report.skippedBaseline) {
    console.log(`previous canary ${report.skippedBaseline} is a different host or suite; this run is a new baseline`);
  } else if (!report.comparison) console.log('no previous canary; this run is the baseline');
  else {
    for (const metric of report.comparison.metrics) {
      console.log(
        `${metric.name} ${formatRatio(metric.deltaRatio)} vs ${report.comparison.previousFile}: ${metric.status} (threshold ${formatRatio(metric.threshold)})`,
      );
    }
  }
  if (report.browser.status !== 'ran') console.log(`browser fixture: ${report.browser.status}${report.browser.reason ? ` — ${report.browser.reason}` : ''}`);
  console.log(`wrote ${filePath}`);
}

const invokedDirectly = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  try {
    main();
  } catch (error) {
    console.error(error.stack ?? error.message);
    process.exit(1);
  }
}
