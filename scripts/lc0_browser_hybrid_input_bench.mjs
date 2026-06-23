#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 240_000;

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/lc0_browser_hybrid_input_bench.mjs [options]\n\nRuns the browser LC0 hybrid input-path benchmark over all 16 representative fixtures.\n\nOptions:\n  --base-url URL        Use an existing dev server (default http://${DEFAULT_HOST}:${DEFAULT_PORT})\n  --port N             Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST          Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       agent-browser session name\n  --timeout MS         Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --layers N           Encoder layers for hybrid path (default 10)\n  --head-backend MODE  Hybrid head backend: ort or wgsl (default ort)\n  --backends LIST      Input backends to compare (default js,wasm; choices js,wgsl,wasm)\n  --legal-priors-backend MODE\n                       Legal-prior backend used for all input-backend cells: js, wasm, or gpu (default js; gpu requires WGSL heads)\n  --iters N            Timed iterations per fixture/backend (default 1)\n  --warmup N           Warmup evals per backend (default 1)\n  --pack-verify        Enable shard sha256 verification (default skipped for benchmarking)\n  --no-server          Do not auto-start Vite\n  --dry-run            Print URL and exit\n  -h, --help           Show this help\n`);
}

function parseArgs(argv) {
  const args = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
    session: process.env.AGENT_BROWSER_SESSION ?? `lc0-hybrid-input-bench-${process.pid}`,
    layers: 10,
    headBackend: 'ort',
    backends: 'js,wasm',
    legalPriorsBackend: 'js',
    iters: 1,
    warmup: 1,
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
    } else if (arg === '--port') args.port = Number(next());
    else if (arg === '--host') args.host = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--session') args.session = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--layers') args.layers = Number(next());
    else if (arg === '--head-backend') args.headBackend = next();
    else if (arg === '--backends') args.backends = next();
    else if (arg === '--legal-priors-backend' || arg === '--hybrid-legal-priors') args.legalPriorsBackend = next();
    else if (arg === '--iters') args.iters = Number(next());
    else if (arg === '--warmup') args.warmup = Number(next());
    else if (arg === '--pack-verify') args.packVerify = true;
    else if (arg === '--no-server') args.noServer = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!['ort', 'wgsl'].includes(args.headBackend)) throw new Error(`Invalid --head-backend: ${args.headBackend}`);
  for (const backend of args.backends.split(',').map((entry) => entry.trim()).filter(Boolean)) {
    if (!['js', 'wgsl', 'wasm'].includes(backend)) throw new Error(`Invalid input backend in --backends: ${backend}`);
  }
  if (!['js', 'wasm', 'gpu'].includes(args.legalPriorsBackend)) throw new Error(`Invalid --legal-priors-backend: ${args.legalPriorsBackend}`);
  if (args.legalPriorsBackend === 'gpu' && args.headBackend !== 'wgsl') throw new Error('--legal-priors-backend gpu requires --head-backend wgsl');
  for (const [name, value] of [['port', args.port], ['timeout', args.timeoutMs], ['layers', args.layers], ['iters', args.iters], ['warmup', args.warmup]]) {
    if (!Number.isFinite(value) || value < 0 || (name !== 'warmup' && value <= 0)) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function benchmarkUrl(args) {
  const url = new URL('/single-engine', args.baseUrl);
  url.searchParams.set('hybridInputBench', '1');
  url.searchParams.set('runtime', 'hybrid');
  if (args.headBackend !== 'ort') url.searchParams.set('headBackend', args.headBackend);
  if (args.legalPriorsBackend !== 'js') url.searchParams.set('legalPriorsBackend', args.legalPriorsBackend);
  url.searchParams.set('inputBenchBackends', args.backends);
  url.searchParams.set('encoderLayers', String(args.layers));
  url.searchParams.set('hybridInputBenchIters', String(args.iters));
  url.searchParams.set('hybridInputBenchWarmup', String(args.warmup));
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
    process.stderr.write(`[lc0-hybrid-input-bench] warning: failed to close agent-browser session ${args.session}: ${error.message ?? error}\n`);
  }
}

async function waitForServer(baseUrl, timeoutMs) {
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
  process.stderr.write(`[lc0-hybrid-input-bench] ${url}\n`);
  try {
    await runAgent(args, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, ['wait', '--text', 'HYBRID_INPUT_BENCH_DONE', '--timeout', String(chunk)], chunk + 5_000);
        const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 30_000));
        const result = JSON.parse(text);
        if (result.status !== 'HYBRID_INPUT_BENCH_DONE') throw new Error(`unexpected benchmark status: ${result.status}`);
        if (result.fixtureCount !== 16) throw new Error(`expected 16 representative fixtures, got ${result.fixtureCount}`);
        return result;
      } catch (error) {
        if (Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for HYBRID_INPUT_BENCH_DONE after ${args.timeoutMs}ms`);
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
