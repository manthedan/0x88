#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { applyLc0RuntimePreset, lc0RuntimeConfiguration, LC0_WEBGPU_RESEARCH_B4_PRESET } from './lc0_runtime_presets.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5181;

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_hybrid_move_sequence_cache_matrix.mjs [options]\n\nRuns LC0 hybrid move-sequence search benchmarks over tree-reuse and eval-cache settings.\n\nOptions:\n  --out PATH            Matrix artifact path (default /tmp/lc0_hybrid_move_sequence_cache_matrix.json)\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port (default ${DEFAULT_PORT})\n  --base-url URL        Use an existing server instead of starting Vite\n  --fen FEN             Starting FEN (default browser start position)\n  --plies N             LC0-driven sequence plies per cell (default 8)\n  --visits LIST         Comma-separated visits list (default 32)\n  --preset NAME         Runtime/search preset, e.g. ${LC0_WEBGPU_RESEARCH_B4_PRESET} (only fills unset runtime knobs)\n  --batches LIST        Comma-separated batch sizes (default 1,4)\n  --batch-pipeline-depth N\n                        LC0 batch pipeline depth (default 1; >1 is speculative search semantics)\n  --head-backends LIST  Comma-separated head backends: ort,wgsl (default wgsl)\n  --input-backend NAME  Hybrid input backend: js, wgsl, or wasm (default js)\n  --encoder-kernel NAME Hybrid encoder kernel (default hand)\n  --legal-priors-backend NAME\n                        Hybrid legal-priors backend: js, wasm, or gpu (default js; gpu requires WGSL heads)\n  --reuse-tree LIST     Comma-separated booleans (default 0,1)\n  --eval-cache LIST     Comma-separated cache entry counts (default 0,2048)\n  --layers N            Encoder layers (default 10)\n  --timeout MS          Per-cell browser timeout (default 240000)\n  --agent-browser BIN   Browser automation binary\n  --dry-run             Print planned cells and exit\n  -h, --help            Show this help\n`);
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
    out: '/tmp/lc0_hybrid_move_sequence_cache_matrix.json',
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    plies: 8,
    visits: [32],
    preset: '',
    batches: [1, 4],
    batchPipelineDepth: 1,
    headBackends: ['wgsl'],
    inputBackend: 'js',
    encoderKernel: 'hand',
    legalPriorsBackend: 'js',
    reuseTree: [false, true],
    evalCacheEntries: [0, 2048],
    layers: 10,
    timeoutMs: 240_000,
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
    } else if (arg === '--fen') args.fen = next();
    else if (arg === '--plies' || arg === '--move-sequence-plies') args.plies = Number(next());
    else if (arg === '--visits') args.visits = parseList(next(), Number, 'visits');
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
  if (args.batchPipelineDepth > 1) process.stderr.write('[move-sequence] warning: batchPipelineDepth > 1 is speculative parallel search; depth=1 is the parity-preserving baseline.\n');
  for (const [name, value] of [['port', args.port], ['plies', args.plies], ['batch-pipeline-depth', args.batchPipelineDepth], ['layers', args.layers], ['timeout', args.timeoutMs]]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  for (const [name, values] of [['visits', args.visits], ['batches', args.batches]]) {
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error(`Invalid --${name}: ${values.join(',')}`);
  }
  if (args.evalCacheEntries.some((value) => !Number.isFinite(value) || value < 0)) throw new Error(`Invalid --eval-cache: ${args.evalCacheEntries.join(',')}`);
  return args;
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

function runAgent(args, session, commandArgs, timeoutMs = 30_000) {
  const fullArgs = ['--json', '--session', session, ...commandArgs];
  return new Promise((resolve, reject) => {
    const child = spawn(args.agentBrowser, fullArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
    const chunks = { stdout: [], stderr: [] };
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn(value);
    };
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stdout.on('data', (chunk) => chunks.stdout.push(chunk));
    child.stderr.on('data', (chunk) => chunks.stderr.push(chunk));
    child.on('error', (error) => finish(reject, error));
    child.on('close', (status) => {
      const stdout = Buffer.concat(chunks.stdout).toString('utf8');
      const stderr = Buffer.concat(chunks.stderr).toString('utf8');
      if (status !== 0) return finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} failed: ${stderr || stdout}`));
      try {
        const parsed = stdout ? JSON.parse(stdout.trim()) : null;
        if (parsed && typeof parsed === 'object' && 'success' in parsed) {
          if (parsed.success === false) return finish(reject, new Error(`${args.agentBrowser} ${fullArgs.slice(1).join(' ')} failed: ${parsed.error ?? stdout}`));
          return finish(resolve, parsed.data ?? parsed);
        }
        return finish(resolve, parsed);
      } catch (error) {
        return finish(reject, error);
      }
    });
  });
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

function runtimeConfiguration(args) {
  return lc0RuntimeConfiguration({
    preset: args.preset,
    runtimes: args.headBackends.map((backend) => backend === 'wgsl' ? 'hybrid-wgsl-heads' : 'hybrid'),
    headBackend: args.headBackends.length === 1 ? args.headBackends[0] : undefined,
    inputBackend: args.inputBackend,
    encoderKernel: args.encoderKernel,
    legalPriorsBackend: args.legalPriorsBackend,
    batches: args.batches,
    batchPipelineDepth: args.batchPipelineDepth,
  });
}

function moveSequenceUrl(args, combo) {
  const url = new URL('/lc0-policy-only.html', args.baseUrl);
  url.searchParams.set('moveSequenceBench', '1');
  url.searchParams.set('runtime', 'hybrid');
  if (args.fen) url.searchParams.set('fen', args.fen);
  if (combo.headBackend !== 'ort') url.searchParams.set('headBackend', combo.headBackend);
  if (combo.headBackend === 'wgsl') url.searchParams.set('wgslBatchMode', 'physical');
  if (args.inputBackend !== 'js') url.searchParams.set('inputBackend', args.inputBackend);
  if (args.legalPriorsBackend !== 'js') url.searchParams.set('legalPriorsBackend', args.legalPriorsBackend);
  if (args.encoderKernel !== 'hand') url.searchParams.set('encoderKernel', args.encoderKernel);
  url.searchParams.set('encoderLayers', String(args.layers));
  url.searchParams.set('visits', String(combo.visits));
  url.searchParams.set('batch', String(combo.batch));
  if (args.batchPipelineDepth !== 1) url.searchParams.set('batchPipelineDepth', String(args.batchPipelineDepth));
  url.searchParams.set('plies', String(args.plies));
  url.searchParams.set('reuseTree', combo.reuseTree ? '1' : '0');
  url.searchParams.set('resetBetweenPlies', combo.reuseTree ? '0' : '1');
  if (combo.evalCacheEntries > 0) url.searchParams.set('evalCacheEntries', String(combo.evalCacheEntries));
  url.searchParams.set('ep', 'wasm');
  url.searchParams.set('packVerify', '0');
  return String(url);
}

function compactCell(result, combo) {
  const stats = result.search?.aggregateStats ?? {};
  const cacheFootprint = result.cacheFootprint;
  const executionFootprint = result.executionFootprint;
  return {
    ...combo,
    backend: result.backend,
    completedPlies: result.completedPlies,
    finalFen: result.finalFen,
    searchMeanMs: result.search?.timingStats?.meanMs,
    requestedVisitsPerSecond: result.search?.requestedVisitsPerSecond,
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
    cacheFootprintBytes: cacheFootprint?.approxBytes,
    cacheFootprintEntries: cacheFootprint?.entries,
    cacheFootprintMaxEntries: cacheFootprint?.maxEntries,
    executionFootprintBytes: executionFootprint?.gpuBufferBytes,
  };
}

async function closeAgentSession(args, session) {
  try {
    await runAgent(args, session, ['close'], 5_000);
  } catch (error) {
    process.stderr.write(`[move-sequence] warning: failed to close agent-browser session ${session}: ${error.message ?? error}\n`);
  }
}

async function runCell(args, combo, index, total) {
  const session = `lc0-hybrid-move-sequence-${process.pid}-${index}`;
  const url = moveSequenceUrl(args, combo);
  process.stderr.write(`[move-sequence] ${index}/${total} backend=${combo.headBackend} visits=${combo.visits} batch=${combo.batch} depth=${args.batchPipelineDepth} input=${args.inputBackend} encoder=${args.encoderKernel} legal=${args.legalPriorsBackend} plies=${args.plies} reuse=${combo.reuseTree ? 1 : 0} cache=${combo.evalCacheEntries}\n`);
  const started = Date.now();
  try {
    await runAgent(args, session, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, session, ['wait', '--text', 'HYBRID_MOVE_SEQUENCE_BENCH_', '--timeout', String(chunk)], chunk + 5_000);
        const text = textFromGetResult(await runAgent(args, session, ['get', 'text', '#benchResult'], 30_000));
        if (text.startsWith('HYBRID_MOVE_SEQUENCE_BENCH_FAILED')) throw new Error(text);
        const result = JSON.parse(text);
        if (result.status !== 'HYBRID_MOVE_SEQUENCE_BENCH_DONE') throw new Error(`unexpected benchmark status: ${result.status}`);
        const expectedBackend = combo.headBackend === 'wgsl' ? 'lc0web-wgsl-encoder-wgsl-heads' : 'lc0web-wgsl-encoder-ort-heads';
        if (result.backend !== expectedBackend) throw new Error(`unexpected hybrid backend: ${result.backend}`);
        if ((result.encoderKernelVariant ?? 'hand') !== args.encoderKernel) throw new Error(`unexpected encoder kernel variant: ${result.encoderKernelVariant ?? 'hand'}`);
        if ((result.legalPriorsBackend ?? 'js') !== args.legalPriorsBackend) throw new Error(`unexpected legal-priors backend: ${result.legalPriorsBackend ?? 'js'}`);
        return { combo, elapsedMs: Date.now() - started, result: { ...result, scriptPreset: args.preset || null }, summary: compactCell(result, combo) };
      } catch (error) {
        if (Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for HYBRID_MOVE_SEQUENCE_BENCH_DONE after ${args.timeoutMs}ms`);
  } finally {
    await closeAgentSession(args, session);
  }
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
    console.log(JSON.stringify({ baseUrl: args.baseUrl, combos, runtimeConfiguration: runtimeConfiguration(args) }, null, 2));
    return;
  }
  const server = startServer(args);
  const startedAt = new Date().toISOString();
  try {
    await waitForServer(args.baseUrl);
    const cells = [];
    for (let i = 0; i < combos.length; i++) cells.push(await runCell(args, combos[i], i + 1, combos.length));
    const artifact = {
      status: 'LC0_HYBRID_MOVE_SEQUENCE_CACHE_MATRIX_DONE',
      startedAt,
      finishedAt: new Date().toISOString(),
      baseUrl: args.baseUrl,
      benchmarkProtocol: {
        name: 'lc0-hybrid-move-sequence-cache-reuse-matrix',
        plies: args.plies,
        visits: args.visits,
        batches: args.batches,
        cacheEntries: args.evalCacheEntries,
        reuseTree: args.reuseTree,
        batchPipelineDepth: args.batchPipelineDepth,
        startFen: args.fen ?? 'startpos',
      },
      runtimeConfiguration: runtimeConfiguration(args),
      environment: {
        baseUrl: args.baseUrl,
        agentBrowser: args.agentBrowser,
      },
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
