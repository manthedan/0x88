#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 180_000;

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_hybrid_search_bench.mjs [options]\n\nRuns a bounded browser benchmark for the hybrid WGSL encoder + ORT heads evaluator, including warm eval latency and fixed-visit PUCT search latency.\n\nOptions:\n  --base-url URL        Use an existing dev server (default http://${DEFAULT_HOST}:${DEFAULT_PORT})\n  --port N             Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST          Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       agent-browser session name\n  --timeout MS         Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --visits N           Fixed PUCT visits per timed search (default 32)\n  --batch N            Search leaf batch size (default 1)\n  --batch-pipeline-depth N\n                       Experimental leaf-batch pipeline depth for deferred readback/search scheduling (default 1)\n  --layers N           Encoder layers for hybrid path (default 10)\n  --head-backend MODE  Hybrid head backend: ort or wgsl (default ort)\n  --wgsl-batch-mode MODE\n                       WGSL-head evaluateBatch mode: physical or serial (default physical)\n  --input-backend MODE Hybrid input backend: js, wgsl, or wasm (default js)\n  --legal-priors-backend MODE\n                       Legal-prior backend: js, wasm, or gpu (default js; gpu requires WGSL heads; opt-in)\n  --encoder-kernel MODE\n                       Hybrid encoder kernels: hand, tvm-packed-f16, mixed-tvm-ffn, or mixed-tvm-ffn-outproj (default hand)\n  --eval-iters N       Timed warm eval iterations (default 3, max 100; 0 for search-only)\n  --eval-warmup N      Warm eval warmup iterations (default 1, max 20)\n  --batch-eval-iters N Timed evaluateBatch iterations at --batch size (default 0)\n  --batch-eval-warmup N\n                       evaluateBatch warmup iterations (default 0)\n  --search-iters N     Timed fixed-visit searches (default 3, max 50)\n  --search-warmup N    Search warmup iterations (default 1, max 10)\n  --reuse-tree         Reuse the worker search tree across repeated searches\n  --reset-between-searches\n                       Reset the tree before every search even when reuse is enabled\n  --no-reset-between-searches\n                       Keep the tree between repeated searches\n  --eval-cache-entries N\n                       Enable worker-side LC0 eval cache with this many entries\n  --pack-verify        Enable shard sha256 verification (default skipped for benchmarking)\n  --no-server          Do not auto-start Vite\n  --dry-run            Print URL and exit\n  -h, --help           Show this help\n`);
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-hybrid-search-bench-${process.pid}`,
    visits: 32,
    batch: 1,
    batchPipelineDepth: 1,
    layers: 10,
    headBackend: 'ort',
    wgslBatchMode: 'physical',
    inputBackend: 'js',
    legalPriorsBackend: 'js',
    encoderKernel: 'hand',
    evalIters: 3,
    evalWarmup: 1,
    batchEvalIters: 0,
    batchEvalWarmup: 0,
    searchIters: 3,
    searchWarmup: 1,
    reuseTree: false,
    resetBetweenSearches: undefined,
    evalCacheEntries: 0,
    packVerify: false,
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
    if (arg === '--base-url') {
      args.baseUrl = next();
      args.explicitBaseUrl = true;
    }
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--host') args.host = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--visits') args.visits = Number(next());
    else if (arg === '--batch') args.batch = Number(next());
    else if (arg === '--batch-pipeline-depth' || arg === '--pipeline-depth') args.batchPipelineDepth = Number(next());
    else if (arg === '--layers') args.layers = Number(next());
    else if (arg === '--head-backend') args.headBackend = next();
    else if (arg === '--wgsl-batch-mode') args.wgslBatchMode = next();
    else if (arg === '--input-backend') args.inputBackend = next();
    else if (arg === '--legal-priors-backend' || arg === '--hybrid-legal-priors') args.legalPriorsBackend = next();
    else if (arg === '--encoder-kernel') args.encoderKernel = next();
    else if (arg === '--eval-iters') args.evalIters = Number(next());
    else if (arg === '--eval-warmup') args.evalWarmup = Number(next());
    else if (arg === '--batch-eval-iters') args.batchEvalIters = Number(next());
    else if (arg === '--batch-eval-warmup') args.batchEvalWarmup = Number(next());
    else if (arg === '--search-iters') args.searchIters = Number(next());
    else if (arg === '--search-warmup') args.searchWarmup = Number(next());
    else if (arg === '--reuse-tree') args.reuseTree = true;
    else if (arg === '--no-reuse-tree') args.reuseTree = false;
    else if (arg === '--reset-between-searches') args.resetBetweenSearches = true;
    else if (arg === '--no-reset-between-searches') args.resetBetweenSearches = false;
    else if (arg === '--eval-cache-entries') args.evalCacheEntries = Number(next());
    else if (arg === '--pack-verify') args.packVerify = true;
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!['ort', 'wgsl'].includes(args.headBackend)) throw new Error(`Invalid --head-backend: ${args.headBackend}`);
  if (!['physical', 'serial'].includes(args.wgslBatchMode)) throw new Error(`Invalid --wgsl-batch-mode: ${args.wgslBatchMode}`);
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  if (!['js', 'wasm', 'gpu'].includes(args.legalPriorsBackend)) throw new Error(`Invalid --legal-priors-backend: ${args.legalPriorsBackend}`);
  if (args.legalPriorsBackend === 'gpu' && args.headBackend !== 'wgsl') throw new Error('--legal-priors-backend gpu requires --head-backend wgsl');
  if (!['hand', 'tvm-packed-f16', 'mixed-tvm-ffn', 'mixed-tvm-ffn-outproj'].includes(args.encoderKernel)) throw new Error(`Invalid --encoder-kernel: ${args.encoderKernel}`);
  for (const [name, value] of [
    ['port', args.port], ['timeout', args.timeoutMs], ['visits', args.visits], ['batch', args.batch], ['batch-pipeline-depth', args.batchPipelineDepth], ['layers', args.layers],
    ['eval-iters', args.evalIters], ['eval-warmup', args.evalWarmup], ['batch-eval-iters', args.batchEvalIters], ['batch-eval-warmup', args.batchEvalWarmup], ['search-iters', args.searchIters], ['search-warmup', args.searchWarmup], ['eval-cache-entries', args.evalCacheEntries],
  ]) {
    if (!Number.isFinite(value) || value < 0 || (!['eval-iters', 'eval-warmup', 'batch-eval-iters', 'batch-eval-warmup', 'search-warmup', 'eval-cache-entries'].includes(name) && value <= 0)) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function benchmarkUrl(args) {
  const url = new URL('/lc0-policy-only.html', args.baseUrl);
  url.searchParams.set('hybridSearchBench', '1');
  url.searchParams.set('runtime', 'hybrid');
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

function runAgent(args, commandArgs, timeoutMs = 30_000) {
  const fullArgs = ['--json', '--session', args.session, ...commandArgs];
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

async function closeAgentSession(args) {
  try {
    await runAgent(args, ['close'], 5_000);
  } catch (error) {
    process.stderr.write(`[lc0-hybrid-search-bench] warning: failed to close agent-browser session ${args.session}: ${error.message ?? error}\n`);
  }
}

async function waitForServer(baseUrl, timeoutMs) {
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

function startServer(args) {
  if (args.noServer) return null;
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  server.stdout.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  server.stderr.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));
  return server;
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
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
        await runAgent(args, ['wait', '--text', 'HYBRID_SEARCH_BENCH_DONE', '--timeout', String(chunk)], chunk + 5_000);
        const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 30_000));
        const result = JSON.parse(text);
        if (result.status !== 'HYBRID_SEARCH_BENCH_DONE') throw new Error(`unexpected benchmark status: ${result.status}`);
        const expectedBackend = args.headBackend === 'wgsl' ? 'lc0web-wgsl-encoder-wgsl-heads' : 'lc0web-wgsl-encoder-ort-heads';
        if (result.backend !== expectedBackend) throw new Error(`unexpected hybrid backend: ${result.backend}`);
        if ((result.encoderKernelVariant ?? 'hand') !== args.encoderKernel) throw new Error(`unexpected encoder kernel variant: ${result.encoderKernelVariant ?? 'hand'}`);
        if ((result.legalPriorsBackend ?? 'js') !== args.legalPriorsBackend) throw new Error(`unexpected legal-priors backend: ${result.legalPriorsBackend ?? 'js'}`);
        return result;
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
  if (args.help) return usage();
  if (args.dryRun) {
    console.log(benchmarkUrl(args));
    return;
  }
  const server = startServer(args);
  try {
    await waitForServer(args.baseUrl, 30_000);
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
