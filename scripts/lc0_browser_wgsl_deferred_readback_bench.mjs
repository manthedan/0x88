#!/usr/bin/env node
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 240_000;

const USAGE = `Usage: node --experimental-strip-types scripts/lc0_browser_wgsl_deferred_readback_bench.mjs [--batch 4] [--iters 4] [--warmup 1] [--fixture-limit 4]\n`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      'base-url': { type: 'string' },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      host: { type: 'string', default: DEFAULT_HOST },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      session: { type: 'string', default: process.env.AGENT_BROWSER_SESSION ?? `lc0-wgsl-deferred-readback-${process.pid}` },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      layers: { type: 'string', default: '10' },
      'input-backend': { type: 'string', default: 'js' },
      batch: { type: 'string', default: '4' },
      iters: { type: 'string', default: '4' },
      warmup: { type: 'string', default: '1' },
      'fixture-limit': { type: 'string', default: '4' },
      'pack-verify': { type: 'boolean', default: false },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.layers = Number(args.layers);
  args.batch = Number(args.batch);
  args.iters = Number(args.iters);
  args.warmup = Number(args.warmup);
  args.fixtureLimit = Number(args.fixtureLimit);
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
    ['layers', args.layers],
    ['batch', args.batch],
    ['iters', args.iters],
    ['warmup', args.warmup],
    ['fixture-limit', args.fixtureLimit],
  ]) {
    if (!Number.isFinite(value) || value < 0 || (!['warmup'].includes(name) && value <= 0)) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function benchmarkUrl(args) {
  const url = new URL('/single-engine', args.baseUrl);
  url.searchParams.set('wgslDeferredReadbackBench', '1');
  url.searchParams.set('runtime', 'hybrid');
  url.searchParams.set('headBackend', 'wgsl');
  url.searchParams.set('inputBackend', args.inputBackend);
  url.searchParams.set('encoderLayers', String(args.layers));
  url.searchParams.set('deferredReadbackBatch', String(args.batch));
  url.searchParams.set('deferredReadbackIters', String(args.iters));
  url.searchParams.set('deferredReadbackWarmup', String(args.warmup));
  url.searchParams.set('fixtureLimit', String(args.fixtureLimit));
  url.searchParams.set('ep', 'wasm');
  if (!args.packVerify) url.searchParams.set('packVerify', '0');
  return String(url);
}

async function closeAgentSession(args) {
  try {
    await runAgent(args, ['close'], 5_000);
  } catch (error) {
    process.stderr.write(`[lc0-wgsl-deferred-readback] warning: failed to close session: ${error.message ?? error}\n`);
  }
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

async function runBrowserBenchmark(args) {
  const url = benchmarkUrl(args);
  process.stderr.write(`[lc0-wgsl-deferred-readback] ${url}\n`);
  try {
    await runAgent(args, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      try {
        await runAgent(args, ['wait', '--text', 'WGSL_DEFERRED_READBACK_BENCH_DONE', '--timeout', String(chunk)], chunk + 5_000);
        const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 30_000));
        const result = JSON.parse(text);
        if (result.status !== 'WGSL_DEFERRED_READBACK_BENCH_DONE') throw new Error(`unexpected status: ${result.status}`);
        return result;
      } catch (error) {
        if (Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for WGSL_DEFERRED_READBACK_BENCH_DONE after ${args.timeoutMs}ms`);
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
