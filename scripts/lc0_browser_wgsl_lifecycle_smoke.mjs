#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent, spawnCapture } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5179;
const DEFAULT_TIMEOUT_MS = 480_000;

const USAGE = `Usage: node --experimental-strip-types scripts/lc0_browser_wgsl_lifecycle_smoke.mjs [options]\n\nRuns repeated browser/WebGPU WGSL-head deferred-readback lifecycle cycles. Each cycle asks the worker to create a fresh hybrid runtime, exercise physical WGSL batch buffers plus deferred double-buffer readback, destroy the runtime, and report browser memory samples when available.\n\nOptions:\n  --out PATH            Write full JSON artifact\n  --base-url URL        Use an existing dev server\n  --port N              Vite port when auto-starting (default ${DEFAULT_PORT})\n  --host HOST           Vite host when auto-starting (default ${DEFAULT_HOST})\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --session NAME        agent-browser session name\n  --timeout MS          Total browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --cycles N            Runtime create/exercise/destroy cycles (default 3)\n  --layers N            Encoder layers for hybrid path (default 10)\n  --input-backend MODE  js, wgsl, or wasm input path (default js)\n  --legal-priors-backend MODE\n                       Legal-prior backend: js, wasm, or gpu (default js; gpu is opt-in)\n  --batch N             Physical WGSL batch size (default 4)\n  --iters N             Timed batches per immediate/deferred mode and cycle (default 4)\n  --warmup N            Warmup batches per mode and cycle (default 1)\n  --fixture-limit N     Representative fixtures per cycle (default 4)\n  --pause-ms N          Pause between cycles (default 0)\n  --pack-verify         Enable shard sha256 verification (default skipped for smoke speed)\n  --allow-mismatches    Exit 0 even if any immediate/deferred best moves differ\n  --skip-leak-check     Skip final browser/process leak check\n  --no-server           Do not auto-start Vite\n  --dry-run             Print URL and exit\n  -h, --help            Show this help\n`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      out: { type: 'string' },
      'base-url': { type: 'string' },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      host: { type: 'string', default: DEFAULT_HOST },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      session: { type: 'string', default: process.env.AGENT_BROWSER_SESSION ?? `lc0-wgsl-lifecycle-${process.pid}` },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      cycles: { type: 'string', default: '3' },
      layers: { type: 'string', default: '10' },
      'input-backend': { type: 'string', default: 'js' },
      'legal-priors-backend': { type: 'string', default: 'js' },
      'hybrid-legal-priors': { type: 'string' },
      batch: { type: 'string', default: '4' },
      iters: { type: 'string', default: '4' },
      warmup: { type: 'string', default: '1' },
      'fixture-limit': { type: 'string', default: '4' },
      'pause-ms': { type: 'string', default: '0' },
      'pack-verify': { type: 'boolean', default: false },
      'allow-mismatches': { type: 'boolean', default: false },
      'skip-leak-check': { type: 'boolean', default: false },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  if (args.hybridLegalPriors !== undefined) args.legalPriorsBackend = args.hybridLegalPriors;
  delete args.hybridLegalPriors;
  args.port = Number(args.port);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.cycles = Number(args.cycles);
  args.layers = Number(args.layers);
  args.batch = Number(args.batch);
  args.iters = Number(args.iters);
  args.warmup = Number(args.warmup);
  args.fixtureLimit = Number(args.fixtureLimit);
  args.pauseMs = Number(args.pauseMs);
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!['js', 'wgsl', 'wasm'].includes(args.inputBackend)) throw new Error(`Invalid --input-backend: ${args.inputBackend}`);
  if (!['js', 'wasm', 'gpu'].includes(args.legalPriorsBackend)) throw new Error(`Invalid --legal-priors-backend: ${args.legalPriorsBackend}`);
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
    ['cycles', args.cycles],
    ['layers', args.layers],
    ['batch', args.batch],
    ['iters', args.iters],
    ['warmup', args.warmup],
    ['fixture-limit', args.fixtureLimit],
    ['pause-ms', args.pauseMs],
  ]) {
    if (!Number.isFinite(value) || value < 0 || (!['warmup', 'pause-ms'].includes(name) && value <= 0)) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function lifecycleUrl(args) {
  const url = new URL('/single-engine', args.baseUrl);
  url.searchParams.set('wgslDeferredReadbackLifecycle', '1');
  url.searchParams.set('runtime', 'hybrid');
  url.searchParams.set('headBackend', 'wgsl');
  url.searchParams.set('wgslBatchMode', 'physical');
  url.searchParams.set('inputBackend', args.inputBackend);
  if (args.legalPriorsBackend !== 'js') url.searchParams.set('legalPriorsBackend', args.legalPriorsBackend);
  url.searchParams.set('encoderLayers', String(args.layers));
  url.searchParams.set('lifecycleCycles', String(args.cycles));
  url.searchParams.set('deferredReadbackBatch', String(args.batch));
  url.searchParams.set('deferredReadbackIters', String(args.iters));
  url.searchParams.set('deferredReadbackWarmup', String(args.warmup));
  url.searchParams.set('fixtureLimit', String(args.fixtureLimit));
  url.searchParams.set('lifecyclePauseMs', String(args.pauseMs));
  url.searchParams.set('ep', 'wasm');
  if (!args.packVerify) url.searchParams.set('packVerify', '0');
  return String(url);
}

async function closeAgentSession(args) {
  try {
    await runAgent(args, ['close'], 5_000);
  } catch (error) {
    process.stderr.write(`[lc0-wgsl-lifecycle] warning: failed to close session: ${error.message ?? error}\n`);
  }
}

function escapeRegex(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function leakCheck(args, options = {}) {
  await runAgent(args, ['close'], 10_000).catch((error) => process.stderr.write(`[lc0-wgsl-lifecycle] warning: close failed: ${error.message ?? error}\n`));
  await delay(1000);
  const stdout = await spawnCapture('ps', ['-axo', 'pid,rss,command'], { timeoutMs: 10_000 });
  const scopedPatterns = [escapeRegex(args.session)];
  if (options.checkVite !== false) scopedPatterns.push(`vite .*${escapeRegex(args.port)}`);
  const pattern = new RegExp(scopedPatterns.join('|'));
  const leaks = stdout
    .split('\n')
    .filter((line) => pattern.test(line) && !/lc0_browser_wgsl_lifecycle_smoke|npm run lc0:browser-wgsl-lifecycle-smoke/.test(line));
  if (leaks.length) throw new Error(`WGSL lifecycle browser/process leak check failed:\n${leaks.join('\n')}`);
  return { status: 'LC0_WGSL_LIFECYCLE_LEAK_CHECK_CLEAN' };
}

function textFromGetResult(result) {
  if (typeof result?.text === 'string') return result.text;
  if (typeof result === 'string') return result;
  throw new Error(`agent-browser get text returned unexpected payload: ${JSON.stringify(result)}`);
}

async function runBrowserLifecycle(args) {
  const url = lifecycleUrl(args);
  process.stderr.write(`[lc0-wgsl-lifecycle] ${url}\n`);
  try {
    await runAgent(args, ['open', url], 30_000);
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      const chunk = Math.min(25_000, Math.max(1000, deadline - Date.now()));
      let doneSeen = false;
      try {
        await runAgent(args, ['wait', '--text', 'WGSL_DEFERRED_READBACK_LIFECYCLE_DONE', '--timeout', String(chunk)], chunk + 5_000);
        doneSeen = true;
        const text = textFromGetResult(await runAgent(args, ['get', 'text', '#benchResult'], 30_000));
        const result = JSON.parse(text);
        if (result.status !== 'WGSL_DEFERRED_READBACK_LIFECYCLE_DONE') throw new Error(`unexpected status: ${result.status}`);
        return result;
      } catch (error) {
        if (doneSeen || Date.now() >= deadline) throw error;
      }
    }
    throw new Error(`Timed out waiting for WGSL_DEFERRED_READBACK_LIFECYCLE_DONE after ${args.timeoutMs}ms`);
  } finally {
    await closeAgentSession(args);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.dryRun) {
    console.log(lifecycleUrl(args));
    return;
  }
  const server = startViteServer(args);
  let result;
  let runError;
  try {
    if (server) await server.ready;
    await waitForHttp(args.baseUrl);
    result = await runBrowserLifecycle(args);
  } catch (error) {
    runError = error;
  } finally {
    server?.kill('SIGTERM');
    if (server) await delay(1000);
  }
  const leak = args.skipLeakCheck ? { status: 'LC0_WGSL_LIFECYCLE_LEAK_CHECK_SKIPPED' } : await leakCheck(args, { checkVite: !args.noServer });
  if (runError) throw runError;
  const artifact = { ...result, leak };
  if (args.out) {
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, JSON.stringify(artifact, null, 2));
  }
  const summary = {
    status: artifact.status,
    out: args.out,
    cycles: artifact.cycles,
    inputCount: artifact.inputCount,
    allCyclesBestMovesMatch: artifact.allCyclesBestMovesMatch,
    failedCycles: artifact.failedCycles,
    lastImmediateEvalsPerSecond: artifact.cycleResults?.at(-1)?.immediate?.evalsPerSecond,
    lastDeferredEvalsPerSecond: artifact.cycleResults?.at(-1)?.deferred?.evalsPerSecond,
    leak: artifact.leak?.status,
  };
  console.log(JSON.stringify(summary, null, 2));
  if (!args.allowMismatches && !artifact.allCyclesBestMovesMatch) {
    throw new Error(
      `WGSL deferred-readback lifecycle best-move mismatch in cycles: ${artifact.failedCycles?.join(',') || 'unknown'}; pass --allow-mismatches for artifact capture`,
    );
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
