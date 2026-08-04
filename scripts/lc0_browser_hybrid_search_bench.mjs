#!/usr/bin/env node
import { applyLc0RuntimePreset, LC0_WEBGPU_RESEARCH_B4_PRESET, lc0RuntimeConfiguration } from './lc0_runtime_presets.mjs';
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 180_000;

const USAGE = `Usage: node --experimental-strip-types scripts/lc0_browser_hybrid_search_bench.mjs [options]\n\nRuns a bounded browser benchmark for the hybrid WGSL encoder + ORT heads evaluator, including warm eval latency and fixed-visit PUCT search latency.\n\nOptions:\n  --base-url URL        Use an existing dev server (default http://${DEFAULT_HOST}:${DEFAULT_PORT})\n  --port N             Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST          Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       agent-browser session name\n  --timeout MS         Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --fen FEN            Position to benchmark (default page start position)\n  --visits N           Fixed PUCT visits per timed search (default 32)\n  --preset NAME        Runtime/search preset, e.g. ${LC0_WEBGPU_RESEARCH_B4_PRESET} (only fills unset runtime knobs)\n  --batch N            Search leaf batch size (default 1)\n  --batch-pipeline-depth N\n                       Experimental leaf-batch pipeline depth for deferred readback/search scheduling (default 1)\n  --layers N           Encoder layers for hybrid path (default 10)\n  --head-backend MODE  Hybrid head backend: ort or wgsl (default ort)\n  --wgsl-batch-mode MODE\n                       WGSL-head evaluateBatch mode: physical or serial (default physical)\n  --input-backend MODE Hybrid input backend: js, wgsl, or wasm (default js)\n  --legal-priors-backend MODE\n                       Legal-prior backend: js, wasm, or gpu (default js; gpu requires WGSL heads; opt-in)\n  --encoder-kernel MODE\n                       Hybrid encoder kernels: hand, tvm-packed-f16, mixed-tvm-ffn, or mixed-tvm-ffn-outproj, mixed-tvm-ffn-smolgen-project (default hand)\n  --eval-iters N       Timed warm eval iterations (default 3, max 100; 0 for search-only)\n  --eval-warmup N      Warm eval warmup iterations (default 1, max 20)\n  --batch-eval-iters N Timed evaluateBatch iterations at --batch size (default 0)\n  --batch-eval-warmup N\n                       evaluateBatch warmup iterations (default 0)\n  --search-iters N     Timed fixed-visit searches (default 3, max 50)\n  --search-warmup N    Search warmup iterations (default 1, max 10)\n  --reuse-tree         Reuse the worker search tree across repeated searches\n  --reset-between-searches\n                       Reset the tree before every search even when reuse is enabled\n  --no-reset-between-searches\n                       Keep the tree between repeated searches\n  --eval-cache-entries N\n                       Enable worker-side LC0 eval cache with this many entries\n  --pack-verify        Enable shard sha256 verification (default skipped for benchmarking)\n  --no-server          Do not auto-start Vite\n  --dry-run            Print URL and exit\n  -h, --help           Show this help\n`;

const FLAG_ALIASES = { '--hybrid-legal-priors': '--legal-priors-backend', '--pipeline-depth': '--batch-pipeline-depth' };

function parseArgs(argv) {
  argv = argv.map((arg) => FLAG_ALIASES[arg] ?? arg);
  const args = parseScriptArgs(argv, {
    options: {
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      session: { type: 'string', default: process.env.AGENT_BROWSER_SESSION ?? `lc0-hybrid-search-bench-${process.pid}` },
      visits: { type: 'string', default: '32' },
      preset: { type: 'string', default: '' },
      batch: { type: 'string', default: '1' },
      'batch-pipeline-depth': { type: 'string', default: '1' },
      layers: { type: 'string', default: '10' },
      'head-backend': { type: 'string', default: 'ort' },
      'wgsl-batch-mode': { type: 'string', default: 'physical' },
      'input-backend': { type: 'string', default: 'js' },
      'legal-priors-backend': { type: 'string', default: 'js' },
      'encoder-kernel': { type: 'string', default: 'hand' },
      'eval-iters': { type: 'string', default: '3' },
      'eval-warmup': { type: 'string', default: '1' },
      'batch-eval-iters': { type: 'string', default: '0' },
      'batch-eval-warmup': { type: 'string', default: '0' },
      'search-iters': { type: 'string', default: '3' },
      'search-warmup': { type: 'string', default: '1' },
      'reuse-tree': { type: 'boolean', default: false },
      'no-reuse-tree': { type: 'boolean', default: false },
      'reset-between-searches': { type: 'boolean' },
      'no-reset-between-searches': { type: 'boolean' },
      'eval-cache-entries': { type: 'string', default: '0' },
      'pack-verify': { type: 'boolean', default: false },
      'base-url': { type: 'string' },
      fen: { type: 'string' },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.visits = Number(args.visits);
  args.batch = Number(args.batch);
  args.batchPipelineDepth = Number(args.batchPipelineDepth);
  args.layers = Number(args.layers);
  args.evalIters = Number(args.evalIters);
  args.evalWarmup = Number(args.evalWarmup);
  args.batchEvalIters = Number(args.batchEvalIters);
  args.batchEvalWarmup = Number(args.batchEvalWarmup);
  args.searchIters = Number(args.searchIters);
  args.searchWarmup = Number(args.searchWarmup);
  args.evalCacheEntries = Number(args.evalCacheEntries);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  const reuseTreeIndex = Math.max(argv.lastIndexOf('--reuse-tree'), argv.lastIndexOf('--no-reuse-tree'));
  args.reuseTree = reuseTreeIndex >= 0 && argv[reuseTreeIndex] === '--reuse-tree';
  delete args.noReuseTree;
  const resetIndex = Math.max(argv.lastIndexOf('--reset-between-searches'), argv.lastIndexOf('--no-reset-between-searches'));
  args.resetBetweenSearches = resetIndex < 0 ? undefined : argv[resetIndex] === '--reset-between-searches';
  delete args.noResetBetweenSearches;
  args.explicitBaseUrl = args.baseUrl !== undefined;
  applyLc0RuntimePreset(args, argv);
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!['ort', 'wgsl'].includes(args.headBackend)) throw new Error(`Invalid --head-backend: ${args.headBackend}`);
  if (!['physical', 'serial'].includes(args.wgslBatchMode)) throw new Error(`Invalid --wgsl-batch-mode: ${args.wgslBatchMode}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  if (!['js', 'wasm', 'gpu'].includes(args.legalPriorsBackend)) throw new Error(`Invalid --legal-priors-backend: ${args.legalPriorsBackend}`);
  if (args.legalPriorsBackend === 'gpu' && args.headBackend !== 'wgsl') throw new Error('--legal-priors-backend gpu requires --head-backend wgsl');
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj', 'mixed-tvm-ffn-smolgen-project'].includes(args.encoderKernel))
    throw new Error(`Invalid --encoder-kernel: ${args.encoderKernel}`);
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
    ['visits', args.visits],
    ['batch', args.batch],
    ['batch-pipeline-depth', args.batchPipelineDepth],
    ['layers', args.layers],
    ['eval-iters', args.evalIters],
    ['eval-warmup', args.evalWarmup],
    ['batch-eval-iters', args.batchEvalIters],
    ['batch-eval-warmup', args.batchEvalWarmup],
    ['search-iters', args.searchIters],
    ['search-warmup', args.searchWarmup],
    ['eval-cache-entries', args.evalCacheEntries],
  ]) {
    if (
      !Number.isFinite(value) ||
      value < 0 ||
      (!['eval-iters', 'eval-warmup', 'batch-eval-iters', 'batch-eval-warmup', 'search-warmup', 'eval-cache-entries'].includes(name) && value <= 0)
    )
      throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function benchmarkUrl(args) {
  const url = new URL('/single-engine', args.baseUrl);
  url.searchParams.set('hybridSearchBench', '1');
  url.searchParams.set('runtime', 'hybrid');
  if (args.fen) url.searchParams.set('fen', args.fen);
  if (args.headBackend !== 'ort') url.searchParams.set('headBackend', args.headBackend);
  if (args.headBackend === 'wgsl') url.searchParams.set('wgslBatchMode', args.wgslBatchMode);
  if (args.inputBackend !== 'js') url.searchParams.set('inputBackend', args.inputBackend);
  if (args.legalPriorsBackend !== 'js') url.searchParams.set('legalPriorsBackend', args.legalPriorsBackend);
  if (args.encoderKernel !== 'hand') url.searchParams.set('encoderKernel', args.encoderKernel);
  url.searchParams.set('encoderLayers', String(args.layers));
  url.searchParams.set('visits', String(args.visits));
  url.searchParams.set('batch', String(args.batch));
  if (args.batchPipelineDepth !== 1) url.searchParams.set('batchPipelineDepth', String(args.batchPipelineDepth));
  url.searchParams.set('hybridEvalBenchIters', String(args.evalIters));
  url.searchParams.set('hybridEvalBenchWarmup', String(args.evalWarmup));
  url.searchParams.set('hybridBatchEvalIters', String(args.batchEvalIters));
  url.searchParams.set('hybridBatchEvalWarmup', String(args.batchEvalWarmup));
  url.searchParams.set('hybridSearchIters', String(args.searchIters));
  url.searchParams.set('hybridSearchWarmup', String(args.searchWarmup));
  url.searchParams.set('reuseTree', args.reuseTree ? '1' : '0');
  if (args.resetBetweenSearches !== undefined) url.searchParams.set('resetBetweenSearches', args.resetBetweenSearches ? '1' : '0');
  if (args.evalCacheEntries > 0) url.searchParams.set('evalCacheEntries', String(args.evalCacheEntries));
  url.searchParams.set('ep', 'wasm');
  if (!args.packVerify) url.searchParams.set('packVerify', '0');
  return String(url);
}

async function closeAgentSession(args) {
  try {
    await runAgent(args, ['close'], 5_000);
  } catch (error) {
    process.stderr.write(`[lc0-hybrid-search-bench] warning: failed to close agent-browser session ${args.session}: ${error.message ?? error}\n`);
  }
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

function browserInfoFromEvalResult(result) {
  const value = result?.value ?? result?.result ?? result;
  if (
    !value ||
    typeof value !== 'object' ||
    typeof value.userAgent !== 'string' ||
    typeof value.platform !== 'string' ||
    !Number.isFinite(value.hardwareConcurrency)
  ) {
    throw new Error(`agent-browser eval returned unexpected browser info: ${JSON.stringify(result)}`);
  }
  return {
    userAgent: value.userAgent,
    platform: value.platform,
    hardwareConcurrency: value.hardwareConcurrency,
  };
}

async function readBrowserInfo(args) {
  const result = await runAgent(
    args,
    [
      'eval',
      `(() => ({
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    hardwareConcurrency: navigator.hardwareConcurrency,
  }))()`,
    ],
    30_000,
  );
  return browserInfoFromEvalResult(result);
}

async function runBrowserBenchmark(args) {
  const url = benchmarkUrl(args);
  process.stderr.write(`[lc0-hybrid-search-bench] ${url}\n`);
  try {
    await runAgent(args, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, ['wait', '--text', 'HYBRID_SEARCH_BENCH_', '--timeout', String(chunk)], chunk + 5_000);
        const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 30_000));
        if (text.startsWith('HYBRID_SEARCH_BENCH_FAILED')) throw new Error(text);
        const result = JSON.parse(text);
        if (result.status !== 'HYBRID_SEARCH_BENCH_DONE') throw new Error(`unexpected benchmark status: ${result.status}`);
        const expectedBackend = args.headBackend === 'wgsl' ? 'lc0web-wgsl-encoder-wgsl-heads' : 'lc0web-wgsl-encoder-ort-heads';
        if (result.backend !== expectedBackend) throw new Error(`unexpected hybrid backend: ${result.backend}`);
        if ((result.encoderKernelVariant ?? 'hand') !== args.encoderKernel)
          throw new Error(`unexpected encoder kernel variant: ${result.encoderKernelVariant ?? 'hand'}`);
        if ((result.legalPriorsBackend ?? 'js') !== args.legalPriorsBackend)
          throw new Error(`unexpected legal-priors backend: ${result.legalPriorsBackend ?? 'js'}`);
        const browserInfo = await readBrowserInfo(args);
        return { ...result, browserInfo, scriptPreset: args.preset || null, runtimeConfiguration: lc0RuntimeConfiguration(args) };
      } catch (error) {
        if (Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for HYBRID_SEARCH_BENCH_DONE after ${args.timeoutMs}ms`);
  } finally {
    await closeAgentSession(args);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.dryRun) {
    console.log(benchmarkUrl(args));
    return;
  }
  const server = startViteServer(args);
  try {
    if (server) await server.ready;
    await waitForHttp(args.baseUrl, { timeoutMs: 30_000 });
    const result = await runBrowserBenchmark(args);
    console.log(JSON.stringify(result, null, 2));
  } finally {
    server?.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
