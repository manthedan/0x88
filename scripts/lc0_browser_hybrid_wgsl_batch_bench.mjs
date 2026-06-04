#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;

function usage() {
  console.log(`Usage: node scripts/lc0_browser_hybrid_wgsl_batch_bench.mjs [options]

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
`);
}

function parseList(raw, parse, name) {
  const values = String(raw ?? '').split(',').map((s) => s.trim()).filter(Boolean).map(parse);
  if (!values.length || values.some((value) => value === undefined || Number.isNaN(value))) throw new Error(`Invalid --${name}: ${raw}`);
  return values;
}

function parseArgs(argv) {
  const args = {
    out: '/tmp/lc0_wgsl_batch_bench.json',
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    visits: 32,
    batches: [1, 2, 4, 8],
    modes: ['serial', 'physical'],
    layers: 10,
    evalIters: 1,
    evalWarmup: 0,
    batchEvalIters: 1,
    batchEvalWarmup: 0,
    searchIters: 1,
    searchWarmup: 0,
    timeoutMs: 180_000,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    dryRun: false,
    explicitBaseUrl: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--out') args.out = next();
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--base-url') { args.baseUrl = next(); args.explicitBaseUrl = true; }
    else if (arg === '--visits') args.visits = Number(next());
    else if (arg === '--batches') args.batches = parseList(next(), Number, 'batches');
    else if (arg === '--modes') args.modes = parseList(next(), (value) => value, 'modes');
    else if (arg === '--layers') args.layers = Number(next());
    else if (arg === '--eval-iters') args.evalIters = Number(next());
    else if (arg === '--eval-warmup') args.evalWarmup = Number(next());
    else if (arg === '--batch-eval-iters') args.batchEvalIters = Number(next());
    else if (arg === '--batch-eval-warmup') args.batchEvalWarmup = Number(next());
    else if (arg === '--search-iters') args.searchIters = Number(next());
    else if (arg === '--search-warmup') args.searchWarmup = Number(next());
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const mode of args.modes) if (!['serial', 'physical'].includes(mode)) throw new Error(`Invalid mode: ${mode}`);
  for (const [name, value] of [['port', args.port], ['visits', args.visits], ['layers', args.layers], ['timeout', args.timeoutMs]]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  for (const [name, values] of [['batches', args.batches]]) {
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error(`Invalid --${name}: ${values.join(',')}`);
  }
  for (const [name, value] of [['eval-iters', args.evalIters], ['eval-warmup', args.evalWarmup], ['batch-eval-iters', args.batchEvalIters], ['batch-eval-warmup', args.batchEvalWarmup], ['search-iters', args.searchIters], ['search-warmup', args.searchWarmup]]) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function spawnCapture(command, commandArgs, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, commandArgs, { stdio: ['ignore', 'pipe', 'pipe'], ...options });
    const chunks = { stdout: [], stderr: [] };
    child.stdout.on('data', (chunk) => chunks.stdout.push(chunk));
    child.stderr.on('data', (chunk) => {
      chunks.stderr.push(chunk);
      if (options.echoStderr) process.stderr.write(chunk);
    });
    child.on('error', reject);
    child.on('close', (status) => {
      const stdout = Buffer.concat(chunks.stdout).toString('utf8');
      const stderr = Buffer.concat(chunks.stderr).toString('utf8');
      if (status !== 0) return reject(new Error(`${command} ${commandArgs.join(' ')} failed with ${status}: ${stderr || stdout}`));
      resolve({ stdout, stderr });
    });
  });
}

function startServer(args) {
  if (args.explicitBaseUrl) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  server.stdout.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  server.stderr.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  return server;
}

async function waitForServer(baseUrl, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(new URL('/lc0-policy-only.html', baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await delay(250);
  }
  throw new Error(`Vite dev server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
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
    'run', 'lc0:browser-hybrid-search-bench', '--',
    '--base-url', args.baseUrl,
    '--agent-browser', args.agentBrowser,
    '--session', session,
    '--head-backend', 'wgsl',
    '--wgsl-batch-mode', combo.mode,
    '--visits', String(args.visits),
    '--batch', String(combo.batch),
    '--layers', String(args.layers),
    '--eval-iters', String(args.evalIters),
    '--eval-warmup', String(args.evalWarmup),
    '--batch-eval-iters', String(args.batchEvalIters),
    '--batch-eval-warmup', String(args.batchEvalWarmup),
    '--search-iters', String(args.searchIters),
    '--search-warmup', String(args.searchWarmup),
    '--timeout', String(args.timeoutMs),
  ];
  process.stderr.write(`[wgsl-batch] ${index}/${total} mode=${combo.mode} batch=${combo.batch}\n`);
  const started = Date.now();
  const { stdout } = await spawnCapture('npm', commandArgs, { echoStderr: true });
  const result = JSON.parse(stdout.slice(stdout.indexOf('{')));
  if (result.backend !== 'lc0web-wgsl-encoder-wgsl-heads') throw new Error(`unexpected backend: ${result.backend}`);
  return { combo, elapsedMs: Date.now() - started, result, summary: compactCell(result, combo) };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const combos = [];
  for (const mode of args.modes) for (const batch of args.batches) combos.push({ mode, batch });
  if (args.dryRun) {
    console.log(JSON.stringify({ baseUrl: args.baseUrl, combos }, null, 2));
    return;
  }
  const server = startServer(args);
  const startedAt = new Date().toISOString();
  try {
    await waitForServer(args.baseUrl);
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
