#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { applyLc0RuntimePreset, lc0RuntimeConfiguration, LC0_WEBGPU_RESEARCH_B4_PRESET } from './lc0_runtime_presets.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;

function usage() {
  console.log(`Usage: node scripts/lc0_browser_hybrid_search_reuse_cache_matrix.mjs [options]\n\nRuns repeated-position hybrid search benchmarks over tree-reuse and eval-cache settings.\n\nOptions:\n  --out PATH            Matrix artifact path (default /tmp/lc0_hybrid_search_reuse_cache_matrix.json)\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port (default ${DEFAULT_PORT})\n  --base-url URL        Use an existing server instead of starting Vite\n  --visits LIST         Comma-separated visits list (default 32,128)\n  --preset NAME         Runtime/search preset, e.g. ${LC0_WEBGPU_RESEARCH_B4_PRESET} (only fills unset runtime knobs)\n  --batches LIST        Comma-separated batch sizes (default 1,4)\n  --batch-pipeline-depth N\n                        LC0 batch pipeline depth (default 1; >1 is speculative search semantics)\n  --head-backends LIST  Comma-separated head backends: ort,wgsl (default ort,wgsl)\n  --input-backend NAME  Hybrid input backend: js, wgsl, or wasm (default js)\n  --encoder-kernel NAME Hybrid encoder kernel (default hand)\n  --legal-priors-backend NAME\n                        Hybrid legal-priors backend: js, wasm, or gpu (default js; gpu requires WGSL heads)\n  --reuse-tree LIST     Comma-separated booleans (default 0,1)\n  --eval-cache LIST     Comma-separated cache entry counts (default 0,2048)\n  --layers N            Encoder layers (default 10)\n  --eval-iters N        Warm eval timed iterations per cell (default 0)\n  --eval-warmup N       Warm eval warmup iterations per cell (default 0)\n  --search-iters N      Repeated timed searches per cell (default 5)\n  --search-warmup N     Repeated warmup searches per cell (default 0)\n  --timeout MS          Per-cell browser timeout (default 180000)\n  --agent-browser BIN   Browser automation binary\n  --dry-run             Print planned cells and exit\n  -h, --help            Show this help\n`);
}

function parseBool(raw) {
  const normalized = String(raw).trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  throw new Error(`Invalid boolean: ${raw}`);
}

function parseList(raw, parse, name) {
  const values = String(raw ?? '').split(',').map((s) => s.trim()).filter(Boolean).map(parse);
  if (!values.length || values.some((value) => value === undefined || (typeof value === 'number' && Number.isNaN(value)))) throw new Error(`Invalid --${name}: ${raw}`);
  return values;
}

function parseArgs(argv) {
  const args = {
    out: '/tmp/lc0_hybrid_search_reuse_cache_matrix.json',
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    visits: [32, 128],
    preset: '',
    batches: [1, 4],
    batchPipelineDepth: 1,
    headBackends: ['ort', 'wgsl'],
    inputBackend: 'js',
    encoderKernel: 'hand',
    legalPriorsBackend: 'js',
    reuseTree: [false, true],
    evalCacheEntries: [0, 2048],
    layers: 10,
    evalIters: 0,
    evalWarmup: 0,
    searchIters: 5,
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
    else if (arg === '--base-url') {
      args.baseUrl = next();
      args.explicitBaseUrl = true;
    } else if (arg === '--visits') args.visits = parseList(next(), Number, 'visits');
    else if (arg === '--preset') args.preset = next();
    else if (arg === '--batches') args.batches = parseList(next(), Number, 'batches');
    else if (arg === '--batch-pipeline-depth' || arg === '--pipeline-depth') args.batchPipelineDepth = Number(next());
    else if (arg === '--head-backends') args.headBackends = parseList(next(), (value) => value, 'head-backends');
    else if (arg === '--input-backend') args.inputBackend = next();
    else if (arg === '--encoder-kernel' || arg === '--encoder-kernel-variant') args.encoderKernel = next();
    else if (arg === '--legal-priors-backend' || arg === '--hybrid-legal-priors') args.legalPriorsBackend = next();
    else if (arg === '--reuse-tree') args.reuseTree = parseList(next(), parseBool, 'reuse-tree');
    else if (arg === '--eval-cache') args.evalCacheEntries = parseList(next(), Number, 'eval-cache');
    else if (arg === '--layers') args.layers = Number(next());
    else if (arg === '--eval-iters') args.evalIters = Number(next());
    else if (arg === '--eval-warmup') args.evalWarmup = Number(next());
    else if (arg === '--search-iters') args.searchIters = Number(next());
    else if (arg === '--search-warmup') args.searchWarmup = Number(next());
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  applyLc0RuntimePreset(args, argv);
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const backend of args.headBackends) if (!['ort', 'wgsl'].includes(backend)) throw new Error(`Invalid backend: ${backend}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(args.encoderKernel)) throw new Error(`Invalid --encoder-kernel: ${args.encoderKernel}`);
  if (!['js', 'wasm', 'gpu'].includes(args.legalPriorsBackend)) throw new Error(`Invalid --legal-priors-backend: ${args.legalPriorsBackend}`);
  if (args.legalPriorsBackend === 'gpu' && args.headBackends.some((backend) => backend !== 'wgsl')) throw new Error('--legal-priors-backend gpu requires --head-backends wgsl');
  if (args.batchPipelineDepth > 1) process.stderr.write('[reuse-cache] warning: batchPipelineDepth > 1 is speculative parallel search; depth=1 is the parity-preserving baseline.\n');
  for (const [name, value] of [['port', args.port], ['batch-pipeline-depth', args.batchPipelineDepth], ['layers', args.layers], ['eval-iters', args.evalIters], ['eval-warmup', args.evalWarmup], ['search-iters', args.searchIters], ['search-warmup', args.searchWarmup], ['timeout', args.timeoutMs]]) {
    if (!Number.isFinite(value) || value < 0 || (!['eval-iters', 'eval-warmup', 'search-warmup'].includes(name) && value <= 0)) throw new Error(`Invalid --${name}: ${value}`);
  }
  for (const [name, values] of [['visits', args.visits], ['batches', args.batches]]) {
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error(`Invalid --${name}: ${values.join(',')}`);
  }
  if (args.evalCacheEntries.some((value) => !Number.isFinite(value) || value < 0)) throw new Error(`Invalid --eval-cache: ${args.evalCacheEntries.join(',')}`);
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
      const response = await fetch(new URL('/single-engine', baseUrl), { cache: 'no-store' });
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
  const stats = result.search?.aggregateStats ?? {};
  const lastStats = result.search?.stats ?? {};
  const cacheFootprint = result.cacheFootprint;
  const executionFootprint = result.executionFootprint;
  return {
    ...combo,
    backend: result.backend,
    bestMove: result.search?.bestMove,
    searchMeanMs: result.search?.timingStats?.meanMs,
    requestedVisitsPerSecond: result.search?.visitsPerSecond,
    completedVisitsPerSecond: result.search?.completedVisitsPerSecond,
    completedVisits: stats.completedVisits,
    evalCalls: stats.evalCalls,
    batchEvalCalls: stats.batchEvalCalls,
    maxEvalBatch: stats.maxEvalBatch,
    evalBatchSizeHistogram: stats.evalBatchSizeHistogram,
    averageEvalBatchSize: stats.averageEvalBatchSize,
    cacheHits: stats.cacheHits,
    neuralEvalMisses: stats.neuralEvalMisses,
    cacheHitRate: stats.cacheHitRate,
    rootReusedCount: stats.rootReusedCount,
    samples: stats.samples,
    stopReasons: stats.stopReasons,
    lastRootReused: lastStats.rootReused,
    lastCompletedVisits: lastStats.completedVisits,
    cacheFootprintBytes: cacheFootprint?.approxBytes,
    cacheFootprintEntries: cacheFootprint?.entries,
    cacheFootprintMaxEntries: cacheFootprint?.maxEntries,
    executionFootprintBytes: executionFootprint?.gpuBufferBytes,
  };
}

async function runCell(args, combo, index, total) {
  const session = `lc0-hybrid-reuse-cache-${process.pid}-${index}`;
  const commandArgs = [
    'run', 'lc0:browser-hybrid-search-bench', '--',
    '--base-url', args.baseUrl,
    '--agent-browser', args.agentBrowser,
    '--session', session,
    '--head-backend', combo.headBackend,
    '--visits', String(combo.visits),
    '--batch', String(combo.batch),
    '--batch-pipeline-depth', String(args.batchPipelineDepth),
    '--layers', String(args.layers),
    '--input-backend', args.inputBackend,
    '--encoder-kernel', args.encoderKernel,
    '--legal-priors-backend', args.legalPriorsBackend,
    '--eval-iters', String(args.evalIters),
    '--eval-warmup', String(args.evalWarmup),
    '--search-iters', String(args.searchIters),
    '--search-warmup', String(args.searchWarmup),
    combo.reuseTree ? '--reuse-tree' : '--no-reuse-tree',
    combo.reuseTree ? '--no-reset-between-searches' : '--reset-between-searches',
    '--eval-cache-entries', String(combo.evalCacheEntries),
    '--timeout', String(args.timeoutMs),
  ];
  if (args.preset) commandArgs.push('--preset', args.preset);
  process.stderr.write(`[reuse-cache] ${index}/${total} backend=${combo.headBackend} visits=${combo.visits} batch=${combo.batch} depth=${args.batchPipelineDepth} input=${args.inputBackend} encoder=${args.encoderKernel} legal=${args.legalPriorsBackend} reuse=${combo.reuseTree ? 1 : 0} cache=${combo.evalCacheEntries}\n`);
  const started = Date.now();
  const { stdout } = await spawnCapture('npm', commandArgs, { echoStderr: true });
  const result = JSON.parse(stdout.slice(stdout.indexOf('{')));
  return { combo, elapsedMs: Date.now() - started, result, summary: compactCell(result, combo) };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const combos = [];
  for (const headBackend of args.headBackends) {
    for (const visits of args.visits) {
      for (const batch of args.batches) {
        for (const reuseTree of args.reuseTree) {
          for (const evalCacheEntries of args.evalCacheEntries) combos.push({ headBackend, visits, batch, reuseTree, evalCacheEntries });
        }
      }
    }
  }
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
      status: 'LC0_HYBRID_SEARCH_REUSE_CACHE_MATRIX_DONE',
      startedAt,
      finishedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      layers: args.layers,
      runtimeConfiguration: lc0RuntimeConfiguration({
        preset: args.preset,
        headBackends: args.headBackends,
        inputBackend: args.inputBackend,
        encoderKernel: args.encoderKernel,
        legalPriorsBackend: args.legalPriorsBackend,
        batches: args.batches,
        batchPipelineDepth: args.batchPipelineDepth,
      }),
      eval: { warmup: args.evalWarmup, iterations: args.evalIters },
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
