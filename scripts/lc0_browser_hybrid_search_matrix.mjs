#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;

function usage() {
  console.log(`Usage: node scripts/lc0_browser_hybrid_search_matrix.mjs [options]\n\nRuns the browser hybrid search benchmark over encoder-kernel/head-backend/visits/batch combinations and writes a JSON matrix artifact.\n\nOptions:\n  --out PATH            Matrix artifact path (default /tmp/lc0_hybrid_search_matrix.json)\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port (default ${DEFAULT_PORT})\n  --base-url URL        Use an existing server instead of starting Vite\n  --visits LIST         Comma-separated visits list (default 1,32,128)\n  --batches LIST        Comma-separated batch sizes (default 1,2,4,8)\n  --batch-pipeline-depths LIST\n                       Comma-separated experimental search pipeline depths (default 1)\n  --head-backends LIST  Comma-separated head backends: ort,wgsl (default ort; include wgsl to opt into experimental WGSL heads)\n  --legal-priors-backends LIST\n                       Comma-separated legal-prior backends: js,wasm,gpu (default js; gpu requires WGSL heads; opt-in)\n  --encoder-kernels LIST\n                       Comma-separated encoder kernels: hand,tvm-packed-f16,mixed-tvm-ffn,mixed-tvm-ffn-outproj,mixed-tvm-ffn-smolgen-project (default hand)\n  --repeats N           Repeat each cell, alternating variants in repeat order (default 1)\n  --layers N            Encoder layers (default 10)\n  --eval-iters N        Warm eval timed iterations per cell (default 3)\n  --eval-warmup N       Warm eval warmup iterations per cell (default 1)\n  --search-iters N      Search timed iterations per cell (default 3)\n  --search-warmup N     Search warmup iterations per cell (default 1)\n  --timeout MS          Per-cell browser timeout (default 180000)\n  --agent-browser BIN   Browser automation binary\n  --dry-run             Print planned cells and exit\n  -h, --help            Show this help\n`);
}

function parseList(raw, parse, name) {
  const values = String(raw ?? '').split(',').map((s) => s.trim()).filter(Boolean).map(parse);
  if (!values.length || values.some((value) => value === undefined || Number.isNaN(value))) throw new Error(`Invalid --${name}: ${raw}`);
  return values;
}

function parseArgs(argv) {
  const args = {
    out: '/tmp/lc0_hybrid_search_matrix.json',
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    visits: [1, 32, 128],
    batches: [1, 2, 4, 8],
    batchPipelineDepths: [1],
    headBackends: ['ort'],
    legalPriorsBackends: ['js'],
    encoderKernels: ['hand'],
    repeats: 1,
    layers: 10,
    evalIters: 3,
    evalWarmup: 1,
    searchIters: 3,
    searchWarmup: 1,
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
    }
    else if (arg === '--visits') args.visits = parseList(next(), Number, 'visits');
    else if (arg === '--batches') args.batches = parseList(next(), Number, 'batches');
    else if (arg === '--batch-pipeline-depths' || arg === '--pipeline-depths') args.batchPipelineDepths = parseList(next(), Number, 'batch-pipeline-depths');
    else if (arg === '--head-backends') args.headBackends = parseList(next(), (value) => value, 'head-backends');
    else if (arg === '--legal-priors-backends' || arg === '--legal-priors-backend') args.legalPriorsBackends = parseList(next(), (value) => value, 'legal-priors-backends');
    else if (arg === '--encoder-kernels') args.encoderKernels = parseList(next(), (value) => value, 'encoder-kernels');
    else if (arg === '--repeats') args.repeats = Number(next());
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
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const backend of args.headBackends) if (!['ort', 'wgsl'].includes(backend)) throw new Error(`Invalid backend: ${backend}`);
  for (const backend of args.legalPriorsBackends) if (!['js', 'wasm', 'gpu'].includes(backend)) throw new Error(`Invalid legal-priors backend: ${backend}`);
  for (const kernel of args.encoderKernels) if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(kernel)) throw new Error(`Invalid encoder kernel: ${kernel}`);
  for (const [name, value] of [['port', args.port], ['layers', args.layers], ['repeats', args.repeats], ['search-iters', args.searchIters], ['timeout', args.timeoutMs]]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (!Number.isFinite(args.evalIters) || args.evalIters < 0) throw new Error(`Invalid --eval-iters: ${args.evalIters}`);
  for (const [name, values] of [['visits', args.visits], ['batches', args.batches], ['batch-pipeline-depths', args.batchPipelineDepths]]) {
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error(`Invalid --${name}: ${values.join(',')}`);
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

function histogramAverage(histogram = {}) {
  const entries = Object.entries(histogram);
  const calls = entries.reduce((sum, [, count]) => sum + Number(count), 0);
  const items = entries.reduce((sum, [size, count]) => sum + Number(size) * Number(count), 0);
  return calls > 0 ? Number((items / calls).toFixed(4)) : undefined;
}

function compactCell(result, combo) {
  const timing = result.eval?.lastBackendTiming ?? {};
  const stats = result.search?.stats ?? {};
  const aggregateStats = result.search?.aggregateStats ?? {};
  const searchTiming = aggregateStats.evalBackendTimingMeans ?? stats.evalBackendTimingMeans ?? {};
  const searchTimingPerPosition = aggregateStats.evalBackendTimingPerPositionMeans ?? stats.evalBackendTimingPerPositionMeans ?? {};
  return {
    ...combo,
    backend: result.backend,
    encoderKernelVariant: result.encoderKernelVariant ?? combo.encoderKernel,
    legalPriorsBackend: result.legalPriorsBackend ?? combo.legalPriorsBackend ?? 'js',
    evalBestMove: result.eval?.bestMove,
    searchBestMove: result.search?.bestMove,
    evalMeanMs: result.eval?.timingStats?.meanMs,
    searchMeanMs: result.search?.timingStats?.meanMs,
    visitsPerSecond: result.search?.visitsPerSecond,
    completedVisits: stats.completedVisits,
    evalCalls: stats.evalCalls,
    batchEvalCalls: stats.batchEvalCalls,
    maxEvalBatch: stats.maxEvalBatch,
    evalBatchSizeHistogram: stats.evalBatchSizeHistogram,
    averageEvalBatchSize: histogramAverage(stats.evalBatchSizeHistogram),
    cacheHits: stats.cacheHits,
    neuralEvalMisses: stats.neuralEvalMisses,
    rootReused: stats.rootReused,
    stopReason: stats.stopReason,
    totalEvalMs: timing.totalEvalMs,
    inputBuildMs: timing.inputBuildMs,
    readbackSyncedMs: timing.readbackSyncedMs,
    headRunMs: timing.headRunMs,
    legalPriorsMs: timing.legalPriorsMs,
    legalPriorsBridgeCopyMs: timing.legalPriorsBridgeCopyMs,
    legalPriorsWasmRunMs: timing.legalPriorsWasmRunMs,
    legalPriorsWasmTotalMs: timing.legalPriorsWasmTotalMs,
    readbackBytes: timing.readbackBytes,
    readbackMapCount: timing.readbackMapCount,
    searchEvalBackendTimingSamples: aggregateStats.evalBackendTimingSamples ?? stats.evalBackendTimingSamples,
    searchEvalBackendTimingPositions: aggregateStats.evalBackendTimingPositions ?? stats.evalBackendTimingPositions,
    searchTotalEvalMs: searchTiming.totalEvalMs,
    searchInputBuildMs: searchTiming.inputBuildMs,
    searchCommandEncodeMs: searchTiming.commandEncodeMs,
    searchQueueSubmitMs: searchTiming.queueSubmitMs,
    searchReadbackSyncedMs: searchTiming.readbackSyncedMs,
    searchHeadRunMs: searchTiming.headRunMs,
    searchLegalPriorsMs: searchTiming.legalPriorsMs,
    searchLegalPriorsBridgeCopyMs: searchTiming.legalPriorsBridgeCopyMs,
    searchLegalPriorsWasmRunMs: searchTiming.legalPriorsWasmRunMs,
    searchLegalPriorsWasmTotalMs: searchTiming.legalPriorsWasmTotalMs,
    searchReadbackBytes: searchTiming.readbackBytes,
    searchReadbackMapCount: searchTiming.readbackMapCount,
    searchReadbackSyncedMsPerPosition: searchTimingPerPosition.readbackSyncedMs,
    searchReadbackBytesPerPosition: searchTimingPerPosition.readbackBytes,
    batchPipelineDepth: stats.batchPipelineDepth,
    batchPipelineFlushes: stats.batchPipelineFlushes,
    maxBatchPipelineBatches: stats.maxBatchPipelineBatches,
  };
}

async function runCell(args, combo, index, total) {
  const session = `lc0-hybrid-matrix-${process.pid}-${index}`;
  const commandArgs = [
    'run', 'lc0:browser-hybrid-search-bench', '--',
    '--base-url', args.baseUrl,
    '--agent-browser', args.agentBrowser,
    '--session', session,
    '--head-backend', combo.headBackend,
    '--legal-priors-backend', combo.legalPriorsBackend,
    '--encoder-kernel', combo.encoderKernel,
    '--visits', String(combo.visits),
    '--batch', String(combo.batch),
    '--batch-pipeline-depth', String(combo.batchPipelineDepth),
    '--layers', String(args.layers),
    '--eval-iters', String(args.evalIters),
    '--eval-warmup', String(args.evalWarmup),
    '--search-iters', String(args.searchIters),
    '--search-warmup', String(args.searchWarmup),
    '--timeout', String(args.timeoutMs),
  ];
  process.stderr.write(`[matrix] ${index}/${total} repeat=${combo.repeat} kernel=${combo.encoderKernel} backend=${combo.headBackend} legal=${combo.legalPriorsBackend} visits=${combo.visits} batch=${combo.batch} pipe=${combo.batchPipelineDepth}\n`);
  const started = Date.now();
  const { stdout } = await spawnCapture('npm', commandArgs, { echoStderr: true });
  const result = JSON.parse(stdout.slice(stdout.indexOf('{')));
  return { combo, elapsedMs: Date.now() - started, result, summary: compactCell(result, combo) };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const combos = [];
  for (let repeat = 1; repeat <= args.repeats; repeat++) {
    for (const encoderKernel of args.encoderKernels) {
      for (const headBackend of args.headBackends) {
        for (const legalPriorsBackend of args.legalPriorsBackends) {
          if (legalPriorsBackend === 'gpu' && headBackend !== 'wgsl') continue;
          for (const visits of args.visits) {
            for (const batch of args.batches) {
              for (const batchPipelineDepth of args.batchPipelineDepths) combos.push({ repeat, encoderKernel, headBackend, legalPriorsBackend, visits, batch, batchPipelineDepth });
            }
          }
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
      status: 'LC0_HYBRID_SEARCH_MATRIX_DONE',
      startedAt,
      finishedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      layers: args.layers,
      encoderKernels: args.encoderKernels,
      legalPriorsBackends: args.legalPriorsBackends,
      repeats: args.repeats,
      batchPipelineDepths: args.batchPipelineDepths,
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
