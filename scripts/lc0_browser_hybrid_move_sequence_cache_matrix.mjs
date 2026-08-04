#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { applyLc0RuntimePreset, LC0_WEBGPU_RESEARCH_B4_PRESET, lc0RuntimeConfiguration } from './lc0_runtime_presets.mjs';
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5181;

const USAGE = `Usage: node --experimental-strip-types scripts/lc0_browser_hybrid_move_sequence_cache_matrix.mjs [options]\n\nRuns LC0 hybrid move-sequence search benchmarks over tree-reuse and eval-cache settings.\n\nOptions:\n  --out PATH            Matrix artifact path (default /tmp/lc0_hybrid_move_sequence_cache_matrix.json)\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port (default ${DEFAULT_PORT})\n  --base-url URL        Use an existing server instead of starting Vite\n  --fen FEN             Starting FEN (default browser start position)\n  --plies N             LC0-driven sequence plies per cell (default 8)\n  --visits LIST         Comma-separated visits list (default 32)\n  --preset NAME         Runtime/search preset, e.g. ${LC0_WEBGPU_RESEARCH_B4_PRESET} (only fills unset runtime knobs)\n  --batches LIST        Comma-separated batch sizes (default 1,4)\n  --batch-pipeline-depth N\n                        LC0 batch pipeline depth (default 1; >1 is speculative search semantics)\n  --head-backends LIST  Comma-separated head backends: ort,wgsl (default wgsl)\n  --input-backend NAME  Hybrid input backend: js, wgsl, or wasm (default js)\n  --encoder-kernel NAME Hybrid encoder kernel (default hand)\n  --legal-priors-backend NAME\n                        Hybrid legal-priors backend: js, wasm, or gpu (default js; gpu requires WGSL heads)\n  --reuse-tree LIST     Comma-separated booleans (default 0,1)\n  --eval-cache LIST     Comma-separated cache entry counts (default 0,2048)\n  --layers N            Encoder layers (default 10)\n  --timeout MS          Per-cell browser timeout (default 240000)\n  --agent-browser BIN   Browser automation binary\n  --dry-run             Print planned cells and exit\n  -h, --help            Show this help\n`;

function parseBool(raw) {
  const normalized = String(raw).trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  throw new Error(`Invalid boolean: ${raw}`);
}

function parseList(raw, parse, name) {
  const values = String(raw ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(parse);
  if (!values.length || values.some((value) => value === undefined || (typeof value === 'number' && Number.isNaN(value))))
    throw new Error(`Invalid --${name}: ${raw}`);
  return values;
}

const FLAG_ALIASES = {
  '--move-sequence-plies': '--plies',
  '--pipeline-depth': '--batch-pipeline-depth',
  '--encoder-kernel-variant': '--encoder-kernel',
  '--hybrid-legal-priors': '--legal-priors-backend',
};

function parseArgs(argv) {
  argv = argv.map((arg) => FLAG_ALIASES[arg] ?? arg);
  const args = parseScriptArgs(argv, {
    options: {
      out: { type: 'string', default: '/tmp/lc0_hybrid_move_sequence_cache_matrix.json' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      plies: { type: 'string', default: '8' },
      visits: { type: 'string', default: '32' },
      preset: { type: 'string', default: '' },
      batches: { type: 'string', default: '1,4' },
      'batch-pipeline-depth': { type: 'string', default: '1' },
      'head-backends': { type: 'string', default: 'wgsl' },
      'input-backend': { type: 'string', default: 'js' },
      'encoder-kernel': { type: 'string', default: 'hand' },
      'legal-priors-backend': { type: 'string', default: 'js' },
      'reuse-tree': { type: 'string', default: 'false,true' },
      'eval-cache': { type: 'string', default: '0,2048' },
      layers: { type: 'string', default: '10' },
      timeout: { type: 'string', default: '240000' },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      'base-url': { type: 'string' },
      fen: { type: 'string' },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.plies = Number(args.plies);
  args.batchPipelineDepth = Number(args.batchPipelineDepth);
  args.layers = Number(args.layers);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.visits = parseList(args.visits, Number, 'visits');
  args.batches = parseList(args.batches, Number, 'batches');
  args.headBackends = parseList(args.headBackends, (value) => value, 'head-backends');
  args.reuseTree = parseList(args.reuseTree, parseBool, 'reuse-tree');
  args.evalCacheEntries = parseList(args.evalCache, Number, 'eval-cache');
  delete args.evalCache;
  args.explicitBaseUrl = args.baseUrl !== undefined;
  applyLc0RuntimePreset(args, argv);
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  for (const backend of args.headBackends) if (!['ort', 'wgsl'].includes(backend)) throw new Error(`Invalid backend: ${backend}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(args.encoderKernel))
    throw new Error(`Invalid --encoder-kernel: ${args.encoderKernel}`);
  if (!['js', 'wasm', 'gpu'].includes(args.legalPriorsBackend)) throw new Error(`Invalid --legal-priors-backend: ${args.legalPriorsBackend}`);
  if (args.legalPriorsBackend === 'gpu' && args.headBackends.some((backend) => backend !== 'wgsl'))
    throw new Error('--legal-priors-backend gpu requires --head-backends wgsl');
  if (args.batchPipelineDepth > 1)
    process.stderr.write('[move-sequence] warning: batchPipelineDepth > 1 is speculative parallel search; depth=1 is the parity-preserving baseline.\n');
  for (const [name, value] of [
    ['port', args.port],
    ['plies', args.plies],
    ['batch-pipeline-depth', args.batchPipelineDepth],
    ['layers', args.layers],
    ['timeout', args.timeoutMs],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  for (const [name, values] of [
    ['visits', args.visits],
    ['batches', args.batches],
  ]) {
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error(`Invalid --${name}: ${values.join(',')}`);
  }
  if (args.evalCacheEntries.some((value) => !Number.isFinite(value) || value < 0)) throw new Error(`Invalid --eval-cache: ${args.evalCacheEntries.join(',')}`);
  return args;
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

function runtimeConfiguration(args) {
  return lc0RuntimeConfiguration({
    preset: args.preset,
    runtimes: args.headBackends.map((backend) => (backend === 'wgsl' ? 'hybrid-wgsl-heads' : 'hybrid')),
    headBackend: args.headBackends.length === 1 ? args.headBackends[0] : undefined,
    inputBackend: args.inputBackend,
    encoderKernel: args.encoderKernel,
    legalPriorsBackend: args.legalPriorsBackend,
    batches: args.batches,
    batchPipelineDepth: args.batchPipelineDepth,
  });
}

function moveSequenceUrl(args, combo) {
  const url = new URL('/single-engine', args.baseUrl);
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
  process.stderr.write(
    `[move-sequence] ${index}/${total} backend=${combo.headBackend} visits=${combo.visits} batch=${combo.batch} depth=${args.batchPipelineDepth} input=${args.inputBackend} encoder=${args.encoderKernel} legal=${args.legalPriorsBackend} plies=${args.plies} reuse=${combo.reuseTree ? 1 : 0} cache=${combo.evalCacheEntries}\n`,
  );
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
        if ((result.encoderKernelVariant ?? 'hand') !== args.encoderKernel)
          throw new Error(`unexpected encoder kernel variant: ${result.encoderKernelVariant ?? 'hand'}`);
        if ((result.legalPriorsBackend ?? 'js') !== args.legalPriorsBackend)
          throw new Error(`unexpected legal-priors backend: ${result.legalPriorsBackend ?? 'js'}`);
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
  const server = startViteServer(args);
  const startedAt = new Date().toISOString();
  try {
    if (server) await server.ready;
    await waitForHttp(args.baseUrl);
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
