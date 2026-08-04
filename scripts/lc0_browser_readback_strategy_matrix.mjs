#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { arch, cpus, platform, release, totalmem } from 'node:os';
import { dirname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseScriptArgs } from './lib/cli.mjs';
import { spawnCapture } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_FENS = 'eval/opening_suite_uho_lite_v1.fen';

const USAGE = `Usage: node scripts/lc0_browser_readback_strategy_matrix.mjs [options]\n\nRuns a fixed-FEN browser matrix comparing ORT WebGPU output-download modes with custom WGSL-head search/readback modes.\n\nOptions:\n  --out PATH            Matrix artifact path (default /tmp/lc0_readback_strategy_matrix.json)\n  --base-url URL        Use an existing dev server\n  --host HOST           Vite host when auto-starting (default ${DEFAULT_HOST})\n  --port N              Vite port when auto-starting (default ${DEFAULT_PORT})\n  --fens PATH           FEN file (default ${DEFAULT_FENS})\n  --max-positions N     Max FENs to use (default 4)\n  --repeats N           Repeat each strategy/FEN cell (default 1)\n  --strategies LIST     Comma-separated: ort-cpu,ort-gpu,wgsl-pipe1,wgsl-gpu-legal,wgsl-pipe2,wgsl-gpu-legal-pipe2 (default all)\n  --ort-iters N         ORT timed eval iterations per FEN (default 3)\n  --ort-warmup N        ORT warmup eval iterations per FEN (default 1)\n  --wgsl-eval-iters N   WGSL warm eval iterations per FEN (default 2)\n  --wgsl-search-iters N WGSL fixed-visit searches per FEN (default 2)\n  --wgsl-search-warmup N\n                       WGSL search warmup searches per FEN (default 1)\n  --visits N            WGSL fixed PUCT visits (default 32)\n  --batch N             WGSL search leaf batch size (default 4)
  --pipe2-batch N       Effective batch cap for wgsl-pipe2; lower this to bound overlap experiments (default 4)\n  --input-backend NAME  WGSL strategy input backend: js, wgsl, or wasm (default js)\n  --encoder-kernel NAME WGSL strategy encoder kernel variant (default hand)\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --timeout MS          Per-cell timeout (default ${DEFAULT_TIMEOUT_MS})\n  --no-server           Do not auto-start Vite\n  --dry-run             Print planned cells and exit\n  -h, --help            Show this help\n`;

function intArg(value, label, min, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Math.floor(Number(value));
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) throw new Error(`Invalid ${label}: ${value}`);
  return parsed;
}

function parseList(raw) {
  return String(raw)
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

const FLAG_ALIASES = { '--encoder-kernel-variant': '--encoder-kernel' };

function parseArgs(argv) {
  argv = argv.map((arg) => FLAG_ALIASES[arg] ?? arg);
  const args = parseScriptArgs(argv, {
    options: {
      out: { type: 'string', default: '/tmp/lc0_readback_strategy_matrix.json' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      fens: { type: 'string', default: DEFAULT_FENS },
      'max-positions': { type: 'string', default: '4' },
      repeats: { type: 'string', default: '1' },
      strategies: { type: 'string', default: 'ort-cpu,ort-gpu,wgsl-pipe1,wgsl-gpu-legal,wgsl-pipe2,wgsl-gpu-legal-pipe2' },
      'ort-iters': { type: 'string', default: '3' },
      'ort-warmup': { type: 'string', default: '1' },
      'wgsl-eval-iters': { type: 'string', default: '2' },
      'wgsl-search-iters': { type: 'string', default: '2' },
      'wgsl-search-warmup': { type: 'string', default: '1' },
      visits: { type: 'string', default: '32' },
      batch: { type: 'string', default: '4' },
      'pipe2-batch': { type: 'string', default: '4' },
      'input-backend': { type: 'string', default: 'js' },
      'encoder-kernel': { type: 'string', default: 'hand' },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      'base-url': { type: 'string' },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = intArg(args.port, '--port', 1, 65535);
  args.maxPositions = intArg(args.maxPositions, '--max-positions', 1, 10_000);
  args.repeats = intArg(args.repeats, '--repeats', 1, 100);
  args.ortIters = intArg(args.ortIters, '--ort-iters', 1, 1000);
  args.ortWarmup = intArg(args.ortWarmup, '--ort-warmup', 0, 100);
  args.wgslEvalIters = intArg(args.wgslEvalIters, '--wgsl-eval-iters', 0, 100);
  args.wgslSearchIters = intArg(args.wgslSearchIters, '--wgsl-search-iters', 1, 100);
  args.wgslSearchWarmup = intArg(args.wgslSearchWarmup, '--wgsl-search-warmup', 0, 100);
  args.visits = intArg(args.visits, '--visits', 1, 1_000_000);
  args.batch = intArg(args.batch, '--batch', 1, 512);
  args.pipe2Batch = intArg(args.pipe2Batch, '--pipe2-batch', 1, 512);
  args.timeoutMs = intArg(args.timeout, '--timeout', 1, 600_000);
  delete args.timeout;
  args.strategies = parseList(args.strategies);
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  const valid = new Set(['ort-cpu', 'ort-gpu', 'wgsl-pipe1', 'wgsl-gpu-legal', 'wgsl-pipe2', 'wgsl-gpu-legal-pipe2']);
  for (const strategy of args.strategies) if (!valid.has(strategy)) throw new Error(`Invalid strategy: ${strategy}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid inputBackend: ${args.inputBackend}`);
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(args.encoderKernel))
    throw new Error(`Invalid encoderKernel: ${args.encoderKernel}`);
  return args;
}

async function loadFens(path, maxPositions) {
  const text = await readFile(path, 'utf8');
  const fens = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))
    .map((line) => line.split(/\s+;|\s+#/)[0].trim())
    .filter(Boolean)
    .slice(0, maxPositions);
  if (!fens.length) throw new Error(`No FENs loaded from ${path}`);
  return fens;
}

function parseJsonStdout(stdout) {
  const start = stdout.indexOf('{');
  if (start < 0) throw new Error(`No JSON object in command output:\n${stdout}`);
  return JSON.parse(stdout.slice(start));
}

function commandForCell(args, cell) {
  if (cell.strategy === 'ort-cpu' || cell.strategy === 'ort-gpu') {
    const commandArgs = [
      'scripts/lc0_browser_ort_readback_profile.mjs',
      '--base-url',
      args.baseUrl,
      '--agent-browser',
      args.agentBrowser,
      '--session',
      `lc0-readback-matrix-${process.pid}-${cell.index}`,
      '--fen',
      cell.fen,
      '--iters',
      String(args.ortIters),
      '--warmup',
      String(args.ortWarmup),
      '--timeout',
      String(args.timeoutMs),
    ];
    if (cell.strategy === 'ort-cpu') commandArgs.push('--no-gpu-outputs');
    return { command: 'node', commandArgs };
  }
  const pipelineDepth = cell.strategy === 'wgsl-pipe2' || cell.strategy === 'wgsl-gpu-legal-pipe2' ? 2 : 1;
  const effectiveBatch = pipelineDepth > 1 ? Math.min(args.batch, args.pipe2Batch) : args.batch;
  const legalPriorsBackend = cell.strategy === 'wgsl-gpu-legal' || cell.strategy === 'wgsl-gpu-legal-pipe2' ? 'gpu' : 'js';
  return {
    command: 'node',
    commandArgs: [
      'scripts/lc0_browser_hybrid_search_bench.mjs',
      '--base-url',
      args.baseUrl,
      '--agent-browser',
      args.agentBrowser,
      '--session',
      `lc0-readback-matrix-${process.pid}-${cell.index}`,
      '--fen',
      cell.fen,
      '--head-backend',
      'wgsl',
      '--wgsl-batch-mode',
      'physical',
      '--input-backend',
      args.inputBackend,
      '--encoder-kernel',
      args.encoderKernel,
      '--legal-priors-backend',
      legalPriorsBackend,
      '--visits',
      String(args.visits),
      '--batch',
      String(effectiveBatch),
      '--batch-pipeline-depth',
      String(pipelineDepth),
      '--eval-iters',
      String(args.wgslEvalIters),
      '--eval-warmup',
      '1',
      '--search-iters',
      String(args.wgslSearchIters),
      '--search-warmup',
      String(args.wgslSearchWarmup),
      '--timeout',
      String(args.timeoutMs),
    ],
  };
}

function _pick(obj, path) {
  let cur = obj;
  for (const part of path.split('.')) cur = cur?.[part];
  return cur;
}

function compactResult(strategy, result) {
  if (strategy.startsWith('ort-')) {
    const last = result.lastBackendTiming ?? {};
    const stats = result.phaseTimingStats ?? {};
    const mean = (key) => stats[key]?.meanMs;
    return {
      backend: result.backend,
      avgMs: result.avgMs,
      evalsPerSecond: result.evalsPerSecond,
      bestMove: result.bestMove,
      q: result.q,
      ortRunMsMean: mean('ortRunMs'),
      ortAllGetDataMsMean: mean('ortAllGetDataMs'),
      webgpuSubmitCountMean: mean('webgpuSubmitCount'),
      webgpuMapAsyncCountMean: mean('webgpuMapAsyncCount'),
      webgpuMapAsyncMsMean: mean('webgpuMapAsyncMs'),
      webgpuCopyBufferToBufferCountMean: mean('webgpuCopyBufferToBufferCount'),
      webgpuCopyBufferToBufferBytesMean: mean('webgpuCopyBufferToBufferBytes'),
      webgpuMapReadBufferBytesMean: mean('webgpuMapReadBufferBytes'),
      webgpuCreateBufferCountMean: mean('webgpuCreateBufferCount'),
      ortKernelCountLast: last.ortKernelCount,
      readbackBytesLast: last.readbackBytes,
    };
  }
  const searchStats = result.search?.aggregateStats ?? result.search?.stats ?? {};
  const timing = result.eval?.lastBackendTiming ?? {};
  const searchTiming = searchStats.evalBackendTimingMeans ?? {};
  return {
    backend: result.backend,
    inputBackend: result.inputBackend,
    encoderKernelVariant: result.encoderKernelVariant,
    legalPriorsBackend: result.legalPriorsBackend ?? (strategy === 'wgsl-gpu-legal' || strategy === 'wgsl-gpu-legal-pipe2' ? 'gpu' : 'js'),
    batchSize: result.batchSize,
    evalMeanMs: result.eval?.timingStats?.meanMs,
    searchMeanMs: result.search?.timingStats?.meanMs,
    visitsPerSecond: result.search?.visitsPerSecond,
    bestMove: result.search?.bestMove ?? result.eval?.bestMove,
    pv: result.search?.pv,
    evalTotalEvalMs: timing.totalEvalMs,
    evalReadbackSyncedMs: timing.readbackSyncedMs,
    evalReadbackMapCount: timing.readbackMapCount,
    evalReadbackBytes: timing.readbackBytes,
    searchTotalEvalMs: searchTiming.totalEvalMs,
    searchCommandEncodeMs: searchTiming.commandEncodeMs,
    searchQueueSubmitMs: searchTiming.queueSubmitMs,
    searchReadbackSyncedMs: searchTiming.readbackSyncedMs,
    searchReadbackMapCount: searchTiming.readbackMapCount,
    searchReadbackBytes: searchTiming.readbackBytes,
    completedVisits: searchStats.completedVisits,
    evalCalls: searchStats.evalCalls,
    batchEvalCalls: searchStats.batchEvalCalls,
    maxEvalBatch: searchStats.maxEvalBatch,
    batchPipelineDepth: searchStats.batchPipelineDepth,
    batchPipelineFlushes: searchStats.batchPipelineFlushes,
    maxBatchPipelineBatches: searchStats.maxBatchPipelineBatches,
  };
}

async function runCell(args, cell, total) {
  const { command, commandArgs } = commandForCell(args, cell);
  process.stderr.write(`[readback-matrix] ${cell.index}/${total} repeat=${cell.repeat} fen=${cell.fenIndex + 1} strategy=${cell.strategy}\n`);
  const started = Date.now();
  // Sub-cells are unbounded (timeoutMs: 0); stderr streams live during long cells.
  const stdout = await spawnCapture(command, commandArgs, { timeoutMs: 0, echoStderr: true });
  const result = parseJsonStdout(stdout);
  return { ...cell, elapsedMs: Date.now() - started, result, summary: compactResult(cell.strategy, result) };
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return undefined;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function numericStats(values) {
  const samples = values.map(Number).filter(Number.isFinite);
  if (!samples.length) return undefined;
  let sum = 0;
  let min = samples[0];
  let max = samples[0];
  for (const value of samples) {
    sum += value;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  const mean = sum / samples.length;
  const variance = samples.length > 1 ? samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (samples.length - 1) : 0;
  return {
    samples: samples.length,
    mean,
    median: median(samples),
    min,
    max,
    variance,
    standardDeviation: Math.sqrt(variance),
    coefficientOfVariation: mean === 0 ? 0 : Math.sqrt(variance) / Math.abs(mean),
  };
}

function summarize(cells) {
  const groups = new Map();
  for (const cell of cells) {
    if (!cell.summary) continue;
    const items = groups.get(cell.strategy) ?? [];
    items.push(cell.summary);
    groups.set(cell.strategy, items);
  }
  return Object.fromEntries(
    Array.from(groups.entries()).map(([strategy, items]) => {
      const get = (key) => median(items.map((item) => Number(item[key])));
      const stats = (key) => numericStats(items.map((item) => item[key]));
      return [
        strategy,
        {
          samples: items.length,
          avgMsMedian: get('avgMs'),
          evalMeanMsMedian: get('evalMeanMs'),
          searchMeanMsMedian: get('searchMeanMs'),
          visitsPerSecondMedian: get('visitsPerSecond'),
          batchSizeMedian: get('batchSize'),
          batchPipelineDepthMedian: get('batchPipelineDepth'),
          ortRunMsMedian: get('ortRunMsMean'),
          ortAllGetDataMsMedian: get('ortAllGetDataMsMean'),
          webgpuMapAsyncMsMedian: get('webgpuMapAsyncMsMean'),
          readbackSyncedMsMedian: get('evalReadbackSyncedMs'),
          searchReadbackSyncedMsMedian: get('searchReadbackSyncedMs'),
          readbackBytesMedian: get('evalReadbackBytes'),
          searchReadbackBytesMedian: get('searchReadbackBytes'),
          mapReadBufferBytesMedian: get('webgpuMapReadBufferBytesMean'),
          evalMeanMsStats: stats('evalMeanMs'),
          searchMeanMsStats: stats('searchMeanMs'),
          visitsPerSecondStats: stats('visitsPerSecond'),
          evalReadbackBytesStats: stats('evalReadbackBytes'),
          searchReadbackBytesStats: stats('searchReadbackBytes'),
        },
      ];
    }),
  );
}

function searchParity(cells, strategies) {
  const control = strategies.find((strategy) => !strategy.startsWith('ort-'));
  if (!control) return undefined;
  const groups = new Map();
  for (const cell of cells) {
    if (!cell.summary || cell.strategy.startsWith('ort-')) continue;
    const key = `${cell.repeat}:${cell.fenIndex}`;
    const group = groups.get(key) ?? {};
    group[cell.strategy] = cell.summary;
    groups.set(key, group);
  }
  return Object.fromEntries(
    strategies
      .filter((strategy) => strategy !== control && !strategy.startsWith('ort-'))
      .map((strategy) => {
        const pairs = Array.from(groups.values()).filter((group) => group[control] && group[strategy]);
        return [
          strategy,
          {
            control,
            comparableCells: pairs.length,
            bestMoveMatches: pairs.filter((group) => group[control].bestMove === group[strategy].bestMove).length,
            pvMatches: pairs.filter((group) => JSON.stringify(group[control].pv) === JSON.stringify(group[strategy].pv)).length,
            completedVisitsMatches: pairs.filter((group) => group[control].completedVisits === group[strategy].completedVisits).length,
          },
        ];
      }),
  );
}

function environmentReport(cells) {
  const cpuList = cpus();
  const browserInfoByStrategy = {};
  for (const cell of cells) {
    if (cell.result?.browserInfo && !browserInfoByStrategy[cell.strategy]) browserInfoByStrategy[cell.strategy] = cell.result.browserInfo;
  }
  return {
    node: process.version,
    platform: platform(),
    architecture: arch(),
    osRelease: release(),
    cpuModel: cpuList[0]?.model,
    logicalCpuCount: cpuList.length,
    totalMemoryBytes: totalmem(),
    browserInfoByStrategy,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const fens = await loadFens(args.fens, args.maxPositions);
  const plan = [];
  for (let repeat = 1; repeat <= args.repeats; repeat++) {
    for (let fenIndex = 0; fenIndex < fens.length; fenIndex++) {
      for (const strategy of args.strategies) plan.push({ index: plan.length + 1, repeat, fenIndex, fen: fens[fenIndex], strategy });
    }
  }
  if (args.dryRun) {
    console.log(
      JSON.stringify(
        {
          status: 'LC0_READBACK_STRATEGY_MATRIX_DRY_RUN',
          baseUrl: args.baseUrl,
          fens,
          plan: plan.map((cell) => ({ ...cell, command: commandForCell(args, cell) })),
        },
        null,
        2,
      ),
    );
    return;
  }
  const server = startViteServer(args);
  const startedAt = new Date().toISOString();
  try {
    if (server) await server.ready;
    await waitForHttp(args.baseUrl);
    const cells = [];
    for (const cell of plan) {
      try {
        cells.push(await runCell(args, cell, plan.length));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        process.stderr.write(`[readback-matrix] ${cell.index}/${plan.length} strategy=${cell.strategy} failed: ${message}\n`);
        cells.push({ ...cell, elapsedMs: undefined, error: message });
      }
    }
    const artifact = {
      status: 'LC0_READBACK_STRATEGY_MATRIX_DONE',
      startedAt,
      finishedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      args: { ...args, agentBrowser: undefined },
      environment: environmentReport(cells),
      fens,
      cells,
      summary: summarize(cells),
      searchParity: searchParity(cells, args.strategies),
      note: 'Short browser matrix for attribution only. Use larger repeats/cross-host runs before promotion.',
    };
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, JSON.stringify(artifact, null, 2));
    console.log(JSON.stringify({ status: artifact.status, out: args.out, cells: cells.length, summary: artifact.summary }, null, 2));
  } finally {
    server?.kill('SIGTERM');
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error.stack ?? error.message);
    process.exit(1);
  });
}
