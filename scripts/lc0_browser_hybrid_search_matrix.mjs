#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { applyLc0RuntimePreset, LC0_WEBGPU_RESEARCH_B4_PRESET, lc0RuntimeConfiguration } from './lc0_runtime_presets.mjs';
import { parseScriptArgs } from './lib/cli.mjs';
import { spawnCapture } from './lib/process.mjs';
import { waitForOutput } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;

const USAGE = `Usage: node scripts/lc0_browser_hybrid_search_matrix.mjs [options]\n\nRuns the browser hybrid search benchmark over encoder-kernel/head-backend/visits/batch combinations and writes a JSON matrix artifact.\n\nOptions:\n  --out PATH            Matrix artifact path (default /tmp/lc0_hybrid_search_matrix.json)\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port (default ${DEFAULT_PORT})\n  --base-url URL        Use an existing server instead of starting Vite\n  --visits LIST         Comma-separated visits list (default 1,32,128)\n  --preset NAME         Runtime/search preset, e.g. ${LC0_WEBGPU_RESEARCH_B4_PRESET} (only fills unset runtime knobs)\n  --batches LIST        Comma-separated batch sizes (default 1,2,4,8)\n  --batch-pipeline-depths LIST\n                       Comma-separated experimental search pipeline depths (default 1)\n  --head-backends LIST  Comma-separated head backends: ort,wgsl (default ort; include wgsl to opt into experimental WGSL heads)\n  --input-backend MODE  Hybrid input backend for all cells: js, wgsl, or wasm (default js)\n  --legal-priors-backends LIST\n                       Comma-separated legal-prior backends: js,wasm,gpu (default js; gpu requires WGSL heads; opt-in)\n  --encoder-kernels LIST\n                       Comma-separated encoder kernels: hand,tvm-packed-f16,mixed-tvm-ffn,mixed-tvm-ffn-outproj,mixed-tvm-ffn-smolgen-project (default hand)\n  --repeats N           Repeat each cell, alternating variants in repeat order (default 1)\n  --layers N            Encoder layers (default 10)\n  --eval-iters N        Warm eval timed iterations per cell (default 3)\n  --eval-warmup N       Warm eval warmup iterations per cell (default 1)\n  --search-iters N      Search timed iterations per cell (default 3)\n  --search-warmup N     Search warmup iterations per cell (default 1)\n  --timeout MS          Per-cell browser timeout (default 180000)\n  --agent-browser BIN   Browser automation binary\n  --dry-run             Print planned cells and exit\n  -h, --help            Show this help\n`;

function parseList(raw, parse, name) {
  const values = String(raw ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(parse);
  if (!values.length || values.some((value) => value === undefined || Number.isNaN(value))) throw new Error(`Invalid --${name}: ${raw}`);
  return values;
}

const FLAG_ALIASES = { '--pipeline-depths': '--batch-pipeline-depths', '--legal-priors-backend': '--legal-priors-backends' };

function parseArgs(argv) {
  argv = argv.map((arg) => FLAG_ALIASES[arg] ?? arg);
  const args = parseScriptArgs(argv, {
    options: {
      out: { type: 'string', default: '/tmp/lc0_hybrid_search_matrix.json' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      visits: { type: 'string', default: '1,32,128' },
      preset: { type: 'string', default: '' },
      batches: { type: 'string', default: '1,2,4,8' },
      'batch-pipeline-depths': { type: 'string', default: '1' },
      'head-backends': { type: 'string', default: 'ort' },
      'input-backend': { type: 'string', default: 'js' },
      'legal-priors-backends': { type: 'string', default: 'js' },
      'encoder-kernels': { type: 'string', default: 'hand' },
      repeats: { type: 'string', default: '1' },
      layers: { type: 'string', default: '10' },
      'eval-iters': { type: 'string', default: '3' },
      'eval-warmup': { type: 'string', default: '1' },
      'search-iters': { type: 'string', default: '3' },
      'search-warmup': { type: 'string', default: '1' },
      timeout: { type: 'string', default: '180000' },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      'base-url': { type: 'string' },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.repeats = Number(args.repeats);
  args.layers = Number(args.layers);
  args.evalIters = Number(args.evalIters);
  args.evalWarmup = Number(args.evalWarmup);
  args.searchIters = Number(args.searchIters);
  args.searchWarmup = Number(args.searchWarmup);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.visits = parseList(args.visits, Number, 'visits');
  args.batches = parseList(args.batches, Number, 'batches');
  args.batchPipelineDepths = parseList(args.batchPipelineDepths, Number, 'batch-pipeline-depths');
  args.headBackends = parseList(args.headBackends, (value) => value, 'head-backends');
  args.legalPriorsBackends = parseList(args.legalPriorsBackends, (value) => value, 'legal-priors-backends');
  args.encoderKernels = parseList(args.encoderKernels, (value) => value, 'encoder-kernels');
  args.explicitBaseUrl = args.baseUrl !== undefined;
  applyLc0RuntimePreset(args, argv);
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const backend of args.headBackends) if (!['ort', 'wgsl'].includes(backend)) throw new Error(`Invalid backend: ${backend}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  for (const backend of args.legalPriorsBackends) if (!['js', 'wasm', 'gpu'].includes(backend)) throw new Error(`Invalid legal-priors backend: ${backend}`);
  for (const kernel of args.encoderKernels)
    if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(kernel))
      throw new Error(`Invalid encoder kernel: ${kernel}`);
  for (const [name, value] of [
    ['port', args.port],
    ['layers', args.layers],
    ['repeats', args.repeats],
    ['search-iters', args.searchIters],
    ['timeout', args.timeoutMs],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (!Number.isFinite(args.evalIters) || args.evalIters < 0) throw new Error(`Invalid --eval-iters: ${args.evalIters}`);
  for (const [name, values] of [
    ['visits', args.visits],
    ['batches', args.batches],
    ['batch-pipeline-depths', args.batchPipelineDepths],
  ]) {
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error(`Invalid --${name}: ${values.join(',')}`);
  }
  return args;
}

function startServer(args) {
  if (args.explicitBaseUrl) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  const echoOutput = (chunk) => process.stderr.write(`[vite] ${chunk}`);
  server.stdout.on('data', echoOutput);
  server.stderr.on('data', echoOutput);
  server.ready = waitForOutput(server, {
    match: (text) => /ready in \d+\s*ms/.test(text) || text.includes(`:${args.port}/`),
    timeoutMs: 30_000,
    label: `Vite dev server (port ${args.port})`,
  });
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
    combo.headBackend,
    '--input-backend',
    args.inputBackend,
    '--legal-priors-backend',
    combo.legalPriorsBackend,
    '--encoder-kernel',
    combo.encoderKernel,
    '--visits',
    String(combo.visits),
    '--batch',
    String(combo.batch),
    '--batch-pipeline-depth',
    String(combo.batchPipelineDepth),
    '--layers',
    String(args.layers),
    '--eval-iters',
    String(args.evalIters),
    '--eval-warmup',
    String(args.evalWarmup),
    '--search-iters',
    String(args.searchIters),
    '--search-warmup',
    String(args.searchWarmup),
    '--timeout',
    String(args.timeoutMs),
  ];
  process.stderr.write(
    `[matrix] ${index}/${total} repeat=${combo.repeat} kernel=${combo.encoderKernel} backend=${combo.headBackend} legal=${combo.legalPriorsBackend} visits=${combo.visits} batch=${combo.batch} pipe=${combo.batchPipelineDepth}\n`,
  );
  const started = Date.now();
  // Sub-cells are unbounded (timeoutMs: 0); stderr streams live during long cells.
  const stdout = await spawnCapture('npm', commandArgs, { timeoutMs: 0, echoStderr: true });
  const result = JSON.parse(stdout.slice(stdout.indexOf('{')));
  return { combo, elapsedMs: Date.now() - started, result, summary: compactCell(result, combo) };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const combos = [];
  for (let repeat = 1; repeat <= args.repeats; repeat++) {
    for (const encoderKernel of args.encoderKernels) {
      for (const headBackend of args.headBackends) {
        for (const legalPriorsBackend of args.legalPriorsBackends) {
          if (legalPriorsBackend === 'gpu' && headBackend !== 'wgsl') continue;
          for (const visits of args.visits) {
            for (const batch of args.batches) {
              for (const batchPipelineDepth of args.batchPipelineDepths)
                combos.push({ repeat, encoderKernel, headBackend, legalPriorsBackend, visits, batch, batchPipelineDepth });
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
    if (server) await server.ready;
    await waitForServer(args.baseUrl);
    const cells = [];
    for (let i = 0; i < combos.length; i++) cells.push(await runCell(args, combos[i], i + 1, combos.length));
    const artifact = {
      status: 'LC0_HYBRID_SEARCH_MATRIX_DONE',
      startedAt,
      finishedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      layers: args.layers,
      runtimeConfiguration: lc0RuntimeConfiguration(args),
      encoderKernels: args.encoderKernels,
      inputBackend: args.inputBackend,
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
