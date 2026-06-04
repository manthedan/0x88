#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 180_000;

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_hybrid_search_bench.mjs [options]\n\nRuns a bounded browser benchmark for the hybrid WGSL encoder + ORT heads evaluator, including warm eval latency and fixed-visit PUCT search latency.\n\nOptions:\n  --base-url URL        Use an existing dev server (default http://${DEFAULT_HOST}:${DEFAULT_PORT})\n  --port N             Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST          Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       agent-browser session name\n  --timeout MS         Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --visits N           Fixed PUCT visits per timed search (default 32)\n  --batch N            Search leaf batch size (default 1)\n  --layers N           Encoder layers for hybrid path (default 10)\n  --eval-iters N       Timed warm eval iterations (default 3, max 100)\n  --eval-warmup N      Warm eval warmup iterations (default 1, max 20)\n  --search-iters N     Timed fixed-visit searches (default 3, max 50)\n  --search-warmup N    Search warmup iterations (default 1, max 10)\n  --pack-verify        Enable shard sha256 verification (default skipped for benchmarking)\n  --no-server          Do not auto-start Vite\n  --dry-run            Print URL and exit\n  -h, --help           Show this help\n`);
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
    layers: 10,
    evalIters: 3,
    evalWarmup: 1,
    searchIters: 3,
    searchWarmup: 1,
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
    else if (arg === '--layers') args.layers = Number(next());
    else if (arg === '--eval-iters') args.evalIters = Number(next());
    else if (arg === '--eval-warmup') args.evalWarmup = Number(next());
    else if (arg === '--search-iters') args.searchIters = Number(next());
    else if (arg === '--search-warmup') args.searchWarmup = Number(next());
    else if (arg === '--pack-verify') args.packVerify = true;
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  for (const [name, value] of [
    ['port', args.port], ['timeout', args.timeoutMs], ['visits', args.visits], ['batch', args.batch], ['layers', args.layers],
    ['eval-iters', args.evalIters], ['eval-warmup', args.evalWarmup], ['search-iters', args.searchIters], ['search-warmup', args.searchWarmup],
  ]) {
    if (!Number.isFinite(value) || value < 0 || (name !== 'eval-warmup' && name !== 'search-warmup' && value <= 0)) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function benchmarkUrl(args) {
  const url = new URL('/lc0-policy-only.html', args.baseUrl);
  url.searchParams.set('hybridSearchBench', '1');
  url.searchParams.set('runtime', 'hybrid');
  url.searchParams.set('encoderLayers', String(args.layers));
  url.searchParams.set('visits', String(args.visits));
  url.searchParams.set('batch', String(args.batch));
  url.searchParams.set('hybridEvalBenchIters', String(args.evalIters));
  url.searchParams.set('hybridEvalBenchWarmup', String(args.evalWarmup));
  url.searchParams.set('hybridSearchIters', String(args.searchIters));
  url.searchParams.set('hybridSearchWarmup', String(args.searchWarmup));
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
        if (result.backend !== 'lc0web-wgsl-encoder-ort-heads') throw new Error(`unexpected hybrid backend: ${result.backend}`);
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
