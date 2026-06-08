#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_FENS = 'eval/opening_suite_uho_lite_v1.fen';

function usage() {
  console.log(`Usage: node scripts/lc0_browser_readback_strategy_matrix.mjs [options]\n\nRuns a fixed-FEN browser matrix comparing ORT WebGPU output-download modes with custom WGSL-head search/readback modes.\n\nOptions:\n  --out PATH            Matrix artifact path (default /tmp/lc0_readback_strategy_matrix.json)\n  --base-url URL        Use an existing dev server\n  --host HOST           Vite host when auto-starting (default ${DEFAULT_HOST})\n  --port N              Vite port when auto-starting (default ${DEFAULT_PORT})\n  --fens PATH           FEN file (default ${DEFAULT_FENS})\n  --max-positions N     Max FENs to use (default 4)\n  --repeats N           Repeat each strategy/FEN cell (default 1)\n  --strategies LIST     Comma-separated: ort-cpu,ort-gpu,wgsl-pipe1,wgsl-gpu-legal,wgsl-pipe2,wgsl-gpu-legal-pipe2 (default all)\n  --ort-iters N         ORT timed eval iterations per FEN (default 3)\n  --ort-warmup N        ORT warmup eval iterations per FEN (default 1)\n  --wgsl-eval-iters N   WGSL warm eval iterations per FEN (default 2)\n  --wgsl-search-iters N WGSL fixed-visit searches per FEN (default 2)\n  --wgsl-search-warmup N\n                       WGSL search warmup searches per FEN (default 1)\n  --visits N            WGSL fixed PUCT visits (default 32)\n  --batch N             WGSL search leaf batch size (default 4)
  --pipe2-batch N       Effective batch cap for wgsl-pipe2; lower this to bound overlap experiments (default 4)\n  --input-backend NAME  WGSL strategy input backend: js, wgsl, or wasm (default js)\n  --encoder-kernel NAME WGSL strategy encoder kernel variant (default hand)\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --timeout MS          Per-cell timeout (default ${DEFAULT_TIMEOUT_MS})\n  --no-server           Do not auto-start Vite\n  --dry-run             Print planned cells and exit\n  -h, --help            Show this help\n`);
}

function intArg(value, label, min, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Math.floor(Number(value));
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) throw new Error(`Invalid ${label}: ${value}`);
  return parsed;
}

function parseList(raw) {
  return String(raw).split(',').map((s) => s.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    out: '/tmp/lc0_readback_strategy_matrix.json',
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    fens: DEFAULT_FENS,
    maxPositions: 4,
    repeats: 1,
    strategies: ['ort-cpu', 'ort-gpu', 'wgsl-pipe1', 'wgsl-gpu-legal', 'wgsl-pipe2', 'wgsl-gpu-legal-pipe2'],
    ortIters: 3,
    ortWarmup: 1,
    wgslEvalIters: 2,
    wgslSearchIters: 2,
    wgslSearchWarmup: 1,
    visits: 32,
    batch: 4,
    pipe2Batch: 4,
    inputBackend: 'js',
    encoderKernel: 'hand',
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    noServer: false,
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
    else if (arg === '--base-url') { args.baseUrl = next(); args.explicitBaseUrl = true; }
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = intArg(next(), '--port', 1, 65535);
    else if (arg === '--fens') args.fens = next();
    else if (arg === '--max-positions') args.maxPositions = intArg(next(), '--max-positions', 1, 10_000);
    else if (arg === '--repeats') args.repeats = intArg(next(), '--repeats', 1, 100);
    else if (arg === '--strategies') args.strategies = parseList(next());
    else if (arg === '--ort-iters') args.ortIters = intArg(next(), '--ort-iters', 1, 1000);
    else if (arg === '--ort-warmup') args.ortWarmup = intArg(next(), '--ort-warmup', 0, 100);
    else if (arg === '--wgsl-eval-iters') args.wgslEvalIters = intArg(next(), '--wgsl-eval-iters', 0, 100);
    else if (arg === '--wgsl-search-iters') args.wgslSearchIters = intArg(next(), '--wgsl-search-iters', 1, 100);
    else if (arg === '--wgsl-search-warmup') args.wgslSearchWarmup = intArg(next(), '--wgsl-search-warmup', 0, 100);
    else if (arg === '--visits') args.visits = intArg(next(), '--visits', 1, 1_000_000);
    else if (arg === '--batch') args.batch = intArg(next(), '--batch', 1, 512);
    else if (arg === '--pipe2-batch') args.pipe2Batch = intArg(next(), '--pipe2-batch', 1, 512);
    else if (arg === '--input-backend') args.inputBackend = next();
    else if (arg === '--encoder-kernel' || arg === '--encoder-kernel-variant') args.encoderKernel = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--timeout') args.timeoutMs = intArg(next(), '--timeout', 1, 600_000);
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  const valid = new Set(['ort-cpu', 'ort-gpu', 'wgsl-pipe1', 'wgsl-gpu-legal', 'wgsl-pipe2', 'wgsl-gpu-legal-pipe2']);
  for (const strategy of args.strategies) if (!valid.has(strategy)) throw new Error(`Invalid strategy: ${strategy}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid inputBackend: ${args.inputBackend}`);
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(args.encoderKernel)) throw new Error(`Invalid encoderKernel: ${args.encoderKernel}`);
  return args;
}

async function loadFens(path, maxPositions) {
  const text = await readFile(path, 'utf8');
  const fens = text.split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))
    .map((line) => line.split(/\s+;|\s+#/)[0].trim())
    .filter(Boolean)
    .slice(0, maxPositions);
  if (!fens.length) throw new Error(`No FENs loaded from ${path}`);
  return fens;
}

function startServer(args) {
  if (args.noServer) return null;
  const viteBin = process.platform === 'win32' ? 'node_modules/.bin/vite.cmd' : 'node_modules/.bin/vite';
  const server = spawn(viteBin, ['--host', args.host, '--port', String(args.port), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, FORCE_COLOR: '0' } });
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
    } catch (error) { lastError = error; }
    await delay(250);
  }
  throw new Error(`Vite dev server did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
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

function parseJsonStdout(stdout) {
  const start = stdout.indexOf('{');
  if (start < 0) throw new Error(`No JSON object in command output:\n${stdout}`);
  return JSON.parse(stdout.slice(start));
}

function commandForCell(args, cell) {
  if (cell.strategy === 'ort-cpu' || cell.strategy === 'ort-gpu') {
    const commandArgs = [
      'scripts/lc0_browser_ort_readback_profile.mjs',
      '--base-url', args.baseUrl,
      '--agent-browser', args.agentBrowser,
      '--session', `lc0-readback-matrix-${process.pid}-${cell.index}`,
      '--fen', cell.fen,
      '--iters', String(args.ortIters),
      '--warmup', String(args.ortWarmup),
      '--timeout', String(args.timeoutMs),
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
      '--base-url', args.baseUrl,
      '--agent-browser', args.agentBrowser,
      '--session', `lc0-readback-matrix-${process.pid}-${cell.index}`,
      '--fen', cell.fen,
      '--head-backend', 'wgsl',
      '--wgsl-batch-mode', 'physical',
      '--input-backend', args.inputBackend,
      '--encoder-kernel', args.encoderKernel,
      '--legal-priors-backend', legalPriorsBackend,
      '--visits', String(args.visits),
      '--batch', String(effectiveBatch),
      '--batch-pipeline-depth', String(pipelineDepth),
      '--eval-iters', String(args.wgslEvalIters),
      '--eval-warmup', '1',
      '--search-iters', String(args.wgslSearchIters),
      '--search-warmup', String(args.wgslSearchWarmup),
      '--timeout', String(args.timeoutMs),
    ],
  };
}

function pick(obj, path) {
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
    batchPipelineDepth: result.batchPipelineDepth,
    evalMeanMs: result.eval?.timingStats?.meanMs,
    searchMeanMs: result.search?.timingStats?.meanMs,
    visitsPerSecond: result.search?.visitsPerSecond,
    bestMove: result.search?.bestMove ?? result.eval?.bestMove,
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
  const { stdout } = await spawnCapture(command, commandArgs, { echoStderr: true });
  const result = parseJsonStdout(stdout);
  return { ...cell, elapsedMs: Date.now() - started, result, summary: compactResult(cell.strategy, result) };
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return undefined;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function summarize(cells) {
  const groups = new Map();
  for (const cell of cells) {
    if (!cell.summary) continue;
    const items = groups.get(cell.strategy) ?? [];
    items.push(cell.summary);
    groups.set(cell.strategy, items);
  }
  return Object.fromEntries(Array.from(groups.entries()).map(([strategy, items]) => {
    const get = (key) => median(items.map((item) => Number(item[key])));
    return [strategy, {
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
    }];
  }));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const fens = await loadFens(args.fens, args.maxPositions);
  const plan = [];
  for (let repeat = 1; repeat <= args.repeats; repeat++) {
    for (let fenIndex = 0; fenIndex < fens.length; fenIndex++) {
      for (const strategy of args.strategies) plan.push({ index: plan.length + 1, repeat, fenIndex, fen: fens[fenIndex], strategy });
    }
  }
  if (args.dryRun) {
    console.log(JSON.stringify({ status: 'LC0_READBACK_STRATEGY_MATRIX_DRY_RUN', baseUrl: args.baseUrl, fens, plan: plan.map((cell) => ({ ...cell, command: commandForCell(args, cell) })) }, null, 2));
    return;
  }
  const server = startServer(args);
  const startedAt = new Date().toISOString();
  try {
    await waitForServer(args.baseUrl);
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
      fens,
      cells,
      summary: summarize(cells),
      note: 'Short browser matrix for attribution only. Use larger repeats/cross-host runs before promotion.',
    };
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, JSON.stringify(artifact, null, 2));
    console.log(JSON.stringify({ status: artifact.status, out: args.out, cells: cells.length, summary: artifact.summary }, null, 2));
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => { console.error(error.stack ?? error.message); process.exit(1); });
