#!/usr/bin/env node
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 240_000;

const USAGE = `Usage: node --experimental-strip-types scripts/lc0_browser_hybrid_input_bench.mjs [options]\n\nRuns the browser LC0 hybrid input-path benchmark over all 16 representative fixtures.\n\nOptions:\n  --base-url URL        Use an existing dev server (default http://${DEFAULT_HOST}:${DEFAULT_PORT})\n  --port N             Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST          Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN  Browser automation binary (default: AGENT_BROWSER_BIN or agent-browser)\n  --session NAME       agent-browser session name\n  --timeout MS         Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --layers N           Encoder layers for hybrid path (default 10)\n  --head-backend MODE  Hybrid head backend: ort or wgsl (default ort)\n  --backends LIST      Input backends to compare (default js,wasm; choices js,wgsl,wasm)\n  --legal-priors-backend MODE\n                       Legal-prior backend used for all input-backend cells: js, wasm, or gpu (default js; gpu requires WGSL heads)\n  --iters N            Timed iterations per fixture/backend (default 1)\n  --warmup N           Warmup evals per backend (default 1)\n  --pack-verify        Enable shard sha256 verification (default skipped for benchmarking)\n  --no-server          Do not auto-start Vite\n  --dry-run            Print URL and exit\n  -h, --help           Show this help\n`;

const FLAG_ALIASES = { '--hybrid-legal-priors': '--legal-priors-backend' };

function parseArgs(argv) {
  argv = argv.map((arg) => FLAG_ALIASES[arg] ?? arg);
  const args = parseScriptArgs(argv, {
    options: {
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      session: { type: 'string', default: process.env.AGENT_BROWSER_SESSION ?? `lc0-hybrid-input-bench-${process.pid}` },
      layers: { type: 'string', default: '10' },
      'head-backend': { type: 'string', default: 'ort' },
      backends: { type: 'string', default: 'js,wasm' },
      'legal-priors-backend': { type: 'string', default: 'js' },
      iters: { type: 'string', default: '1' },
      warmup: { type: 'string', default: '1' },
      'pack-verify': { type: 'boolean', default: false },
      'base-url': { type: 'string' },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.layers = Number(args.layers);
  args.iters = Number(args.iters);
  args.warmup = Number(args.warmup);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!['ort', 'wgsl'].includes(args.headBackend)) throw new Error(`Invalid --head-backend: ${args.headBackend}`);
  for (const backend of args.backends
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)) {
    if (!['js', 'wgsl', 'wasm'].includes(backend)) throw new Error(`Invalid input backend in --backends: ${backend}`);
  }
  if (!['js', 'wasm', 'gpu'].includes(args.legalPriorsBackend)) throw new Error(`Invalid --legal-priors-backend: ${args.legalPriorsBackend}`);
  if (args.legalPriorsBackend === 'gpu' && args.headBackend !== 'wgsl') throw new Error('--legal-priors-backend gpu requires --head-backend wgsl');
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
    ['layers', args.layers],
    ['iters', args.iters],
    ['warmup', args.warmup],
  ]) {
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

async function closeAgentSession(args) {
  try {
    await runAgent(args, ['close'], 5_000);
  } catch (error) {
    process.stderr.write(`[lc0-hybrid-input-bench] warning: failed to close agent-browser session ${args.session}: ${error.message ?? error}\n`);
  }
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
