#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { parseScriptArgs } from './lib/cli.mjs';
import { spawnCapture } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;

const USAGE = `Usage: node scripts/lc0_browser_hybrid_wgsl_batch_bench.mjs [options]

Compares experimental WGSL-head evaluateBatch modes through the browser hybrid search benchmark.

Options:
  --out PATH            Artifact path (default /tmp/lc0_wgsl_batch_bench.json)
  --host HOST           Vite host (default ${DEFAULT_HOST})
  --port N              Vite port (default ${DEFAULT_PORT})
  --base-url URL        Use an existing server instead of starting Vite
  --visits N            Fixed PUCT visits per search cell (default 32)
  --batches LIST        Comma-separated batch sizes (default 1,2,4,8)
  --modes LIST          Comma-separated WGSL batch modes: serial,physical (default serial,physical)
  --layers N            Encoder layers (default 10)
  --eval-iters N        Single-position warm eval timed iterations per cell (default 1)
  --eval-warmup N       Single-position warm eval warmup iterations per cell (default 0)
  --batch-eval-iters N  Timed evaluateBatch iterations per cell (default 1)
  --batch-eval-warmup N evaluateBatch warmup iterations per cell (default 0)
  --search-iters N      Search timed iterations per cell (default 1)
  --search-warmup N     Search warmup iterations per cell (default 0)
  --timeout MS          Per-cell browser timeout (default 180000)
  --agent-browser BIN   Browser automation binary
  --dry-run             Print planned cells and exit
  -h, --help            Show this help
`;

function parseList(raw, parse, name) {
  const values = String(raw ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(parse);
  if (!values.length || values.some((value) => value === undefined || Number.isNaN(value))) throw new Error(`Invalid --${name}: ${raw}`);
  return values;
}

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      out: { type: 'string', default: '/tmp/lc0_wgsl_batch_bench.json' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      visits: { type: 'string', default: '32' },
      batches: { type: 'string', default: '1,2,4,8' },
      modes: { type: 'string', default: 'serial,physical' },
      layers: { type: 'string', default: '10' },
      'eval-iters': { type: 'string', default: '1' },
      'eval-warmup': { type: 'string', default: '0' },
      'batch-eval-iters': { type: 'string', default: '1' },
      'batch-eval-warmup': { type: 'string', default: '0' },
      'search-iters': { type: 'string', default: '1' },
      'search-warmup': { type: 'string', default: '0' },
      timeout: { type: 'string', default: '180000' },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      'base-url': { type: 'string' },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.visits = Number(args.visits);
  args.layers = Number(args.layers);
  args.evalIters = Number(args.evalIters);
  args.evalWarmup = Number(args.evalWarmup);
  args.batchEvalIters = Number(args.batchEvalIters);
  args.batchEvalWarmup = Number(args.batchEvalWarmup);
  args.searchIters = Number(args.searchIters);
  args.searchWarmup = Number(args.searchWarmup);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.batches = parseList(args.batches, Number, 'batches');
  args.modes = parseList(args.modes, (value) => value, 'modes');
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const mode of args.modes) if (!['serial', 'physical'].includes(mode)) throw new Error(`Invalid mode: ${mode}`);
  for (const [name, value] of [
    ['port', args.port],
    ['visits', args.visits],
    ['layers', args.layers],
    ['timeout', args.timeoutMs],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  for (const [name, values] of [['batches', args.batches]]) {
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error(`Invalid --${name}: ${values.join(',')}`);
  }
  for (const [name, value] of [
    ['eval-iters', args.evalIters],
    ['eval-warmup', args.evalWarmup],
    ['batch-eval-iters', args.batchEvalIters],
    ['batch-eval-warmup', args.batchEvalWarmup],
    ['search-iters', args.searchIters],
    ['search-warmup', args.searchWarmup],
  ]) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function compactCell(result, combo) {
  const batchTiming = result.batchEval?.lastBackendTiming ?? {};
  const searchStats = result.search?.stats ?? {};
  return {
    ...combo,
    backend: result.backend,
    wgslBatchMode: result.wgslBatchMode,
    bestMove: result.search?.bestMove,
    batchEvalMeanMs: result.batchEval?.timingStats?.meanMs,
    batchEvalReadbackMs: batchTiming.readbackSyncedMs,
    batchEvalReadbackBytes: batchTiming.readbackBytes,
    batchEvalReadbackMapCount: batchTiming.readbackMapCount,
    physicalBatchSize: batchTiming.physicalBatchSize,
    searchMeanMs: result.search?.timingStats?.meanMs,
    visitsPerSecond: result.search?.visitsPerSecond,
    completedVisitsPerSecond: result.search?.completedVisitsPerSecond,
    completedVisits: searchStats.completedVisits,
    maxEvalBatch: searchStats.maxEvalBatch,
    evalBatchSizeHistogram: searchStats.evalBatchSizeHistogram,
    stopReason: searchStats.stopReason,
    correctness: {
      bestMove: result.search?.bestMove,
      batchEvalAllBestMovesMatch: result.batchEval?.allBestMovesMatch,
      stopReason: searchStats.stopReason,
      completedRequestedVisits: searchStats.completedVisits === result.visits,
    },
  };
}

async function runCell(args, combo, index, total) {
  const session = `lc0-wgsl-batch-bench-${process.pid}-${index}`;
  const commandArgs = [
    'run',
    'lc0:browser-hybrid-search-bench',
    '--',
    '--base-url',
    args.baseUrl,
    '--agent-browser',
    args.agentBrowser,
    '--session',
    session,
    '--head-backend',
    'wgsl',
    '--wgsl-batch-mode',
    combo.mode,
    '--visits',
    String(args.visits),
    '--batch',
    String(combo.batch),
    '--layers',
    String(args.layers),
    '--eval-iters',
    String(args.evalIters),
    '--eval-warmup',
    String(args.evalWarmup),
    '--batch-eval-iters',
    String(args.batchEvalIters),
    '--batch-eval-warmup',
    String(args.batchEvalWarmup),
    '--search-iters',
    String(args.searchIters),
    '--search-warmup',
    String(args.searchWarmup),
    '--timeout',
    String(args.timeoutMs),
  ];
  process.stderr.write(`[wgsl-batch] ${index}/${total} mode=${combo.mode} batch=${combo.batch}\n`);
  const started = Date.now();
  // Sub-cells are unbounded (timeoutMs: 0); stderr streams live during long cells.
  const stdout = await spawnCapture('npm', commandArgs, { timeoutMs: 0, echoStderr: true });
  const result = JSON.parse(stdout.slice(stdout.indexOf('{')));
  if (result.backend !== 'lc0web-wgsl-encoder-wgsl-heads') throw new Error(`unexpected backend: ${result.backend}`);
  return { combo, elapsedMs: Date.now() - started, result, summary: compactCell(result, combo) };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const combos = [];
  for (const mode of args.modes) for (const batch of args.batches) combos.push({ mode, batch });
  if (args.dryRun) {
    console.log(JSON.stringify({ baseUrl: args.baseUrl, combos }, null, 2));
    return;
  }
  const server = startViteServer(args);
  const startedAt = new Date().toISOString();
  try {
    if (server) await server.ready;
    await waitForHttp(args.baseUrl);
    const cells = [];
    for (let i = 0; i < combos.length; i++) cells.push(await runCell(args, combos[i], i + 1, combos.length));
    const artifact = {
      status: 'LC0_WGSL_BATCH_BENCH_DONE',
      startedAt,
      finishedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      visits: args.visits,
      layers: args.layers,
      eval: { warmup: args.evalWarmup, iterations: args.evalIters },
      batchEval: { warmup: args.batchEvalWarmup, iterations: args.batchEvalIters },
      search: { warmup: args.searchWarmup, iterations: args.searchIters },
      cells,
      summary: cells.map((cell) => cell.summary),
    };
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, JSON.stringify(artifact, null, 2));
    console.log(JSON.stringify({ status: artifact.status, out: args.out, cells: cells.length, summary: artifact.summary }, null, 2));
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
